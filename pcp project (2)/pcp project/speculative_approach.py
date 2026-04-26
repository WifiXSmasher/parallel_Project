"""
Speculative Reference Frame Processing — eliminates ref-frame blocking.

Non-ref frames speculatively use the PREVIOUS ref frame's detections
for carry-forward instead of waiting. When the current ref frame
completes, we validate via IoU-based similarity:
  >= 95% → COMMIT (speculation was correct)
  <  95% → ROLLBACK (reapply carry-forward from correct ref)
"""
import time
import cv2
import numpy as np
import torch
from utils import Detection, compute_iou

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Import core logic from proposed approach
from proposed_approach import (
    extract_rois_from_mvs, assign_complexities, schedule_rois,
    carry_forward_detections, calibrate_complexity_edges,
    load_models_sequentially, GOP_SIZE, FLOW_THRESHOLD,
    MIN_ROI_AREA, MORPH_KERNEL
)

SPECULATION_THRESHOLD = 0.95


def compute_ref_similarity(dets_a, dets_b, iou_thresh=0.5):
    """IoU-based detection similarity between two ref frames."""
    if not dets_a and not dets_b: return 1.0
    if not dets_a or not dets_b: return 0.0
    matched, used = 0, set()
    for da in dets_a:
        best_iou, best_j = 0, -1
        for j, db in enumerate(dets_b):
            if j in used or da.class_id != db.class_id: continue
            iou = compute_iou(da.bbox_xyxy, db.bbox_xyxy)
            if iou > best_iou: best_iou, best_j = iou, j
        if best_iou >= iou_thresh and best_j >= 0:
            matched += 1; used.add(best_j)
    return matched / max(len(dets_a), len(dets_b))


def run_speculative_approach(video_path, model_configs, complexity_edges,
                              max_frames=300):
    """
    Simulated-parallel speculative pipeline.

    Returns TWO result dicts:
      1. Without rollback cost (ideal speculation)
      2. With rollback cost added to affected frames
    """
    import av
    from ultralytics import YOLO

    print("  [Speculative] Loading models...")
    models = load_models_sequentially(model_configs)
    num_models = len(models)
    largest_idx = num_models - 1

    # Warmup + benchmark
    dummy = np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8)
    for m in models:
        for _ in range(2): m.predict(dummy, verbose=False, device=DEVICE)

    processing_rates = {}
    for i, m in enumerate(models):
        t0 = time.perf_counter()
        for _ in range(5): m.predict(dummy, verbose=False, device=DEVICE)
        processing_rates[i] = 5.0 / (time.perf_counter() - t0)

    container = av.open(video_path)
    vs = container.streams.video[0]
    vs.codec_context.options = {'flags2': '+export_mvs'}

    queue_state = {i: 0.0 for i in range(num_models)}
    ref_detections = {}     # ref_frame_idx -> detections
    prev_ref_dets = None    # detections from the PREVIOUS ref frame
    prev_ref_idx = None

    # Results without rollback cost
    res_no_rb = {'method': 'Speculative (ideal)',
                 'frame_latencies': [], 'all_detections': {},
                 'detection_counts': []}
    # Results with rollback cost
    res_rb = {'method': 'Speculative (+rollback)',
              'frame_latencies': [], 'all_detections': {},
              'detection_counts': []}

    stats = {'commits': 0, 'rollbacks': 0, 'direct': 0, 'rollback_frames': set()}

    for frame_idx, frame_av in enumerate(container.decode(vs)):
        if frame_idx >= max_frames: break
        try:
            frame = frame_av.to_ndarray(format='bgr24')
        except Exception:
            continue

        is_ref = (frame_idx % GOP_SIZE == 0)

        if is_ref:
            # --- Reference frame: full inference ---
            t0 = time.perf_counter()
            r = models[largest_idx].predict(frame, verbose=False,
                                            conf=0.3, device=DEVICE, imgsz=640)
            compute_time = time.perf_counter() - t0

            dets = []
            if r and len(r[0].boxes) > 0:
                for box in r[0].boxes:
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
                    cls = int(box.cls[0]); conf = float(box.conf[0])
                    dets.append(Detection(
                        bbox_xyxy=(x1, y1, x2, y2), class_id=cls,
                        class_name=models[largest_idx].names.get(cls, str(cls)),
                        confidence=conf, model_id=largest_idx))

            # Validate speculation for the GOP that just ended
            if prev_ref_dets is not None and prev_ref_idx is not None:
                sim = compute_ref_similarity(prev_ref_dets, dets)
                gop_start = prev_ref_idx + 1
                gop_end = frame_idx
                spec_frames = [f for f in range(gop_start, gop_end)
                               if f in res_no_rb['all_detections']]

                if sim >= SPECULATION_THRESHOLD:
                    stats['commits'] += len(spec_frames)
                    if spec_frames:
                        print(f"  [Spec] ✓ COMMIT {len(spec_frames)} frames "
                              f"(sim={sim:.3f})")
                else:
                    # ROLLBACK: reapply carry-forward from correct ref
                    stats['rollbacks'] += len(spec_frames)
                    for sf in spec_frames:
                        stats['rollback_frames'].add(sf)
                        # Recompute: remove old carried, add new
                        old_dets = res_no_rb['all_detections'][sf]
                        # Keep only ROI-detected dets (not carried)
                        roi_dets = [d for d in old_dets if d.model_id != -99]
                        # Carry from correct ref
                        # We don't have masks stored, so carry all
                        new_carried = []
                        for cd in dets:
                            if not any(cd.class_id == rd.class_id and
                                       compute_iou(cd.bbox_xyxy, rd.bbox_xyxy) > 0.3
                                       for rd in roi_dets):
                                new_carried.append(cd)
                        corrected = roi_dets + new_carried
                        res_no_rb['all_detections'][sf] = corrected
                        res_rb['all_detections'][sf] = corrected
                        # Update counts
                        idx_in_list = list(res_no_rb['all_detections'].keys()).index(sf)
                        if idx_in_list < len(res_no_rb['detection_counts']):
                            res_no_rb['detection_counts'][idx_in_list] = len(corrected)
                            res_rb['detection_counts'][idx_in_list] = len(corrected)

                    if spec_frames:
                        print(f"  [Spec] ✗ ROLLBACK {len(spec_frames)} frames "
                              f"(sim={sim:.3f})")

            ref_detections[frame_idx] = dets
            prev_ref_dets = dets
            prev_ref_idx = frame_idx

            for r_dict in [res_no_rb, res_rb]:
                r_dict['frame_latencies'].append(compute_time)
                r_dict['all_detections'][frame_idx] = dets
                r_dict['detection_counts'].append(len(dets))

            print(f"  [Spec] Frame {frame_idx:3d} [ REF ] | "
                  f"{compute_time*1000:6.1f}ms | Dets: {len(dets):3d}")

        else:
            # --- Non-ref frame: MV + SA + simulated parallel ---
            t_oh = time.perf_counter()
            rois, mask = extract_rois_from_mvs(frame_av, frame)

            if rois:
                assign_complexities(rois)
                assignments = schedule_rois(rois, queue_state,
                                            processing_rates, complexity_edges)
            else:
                assignments = {}

            overhead = time.perf_counter() - t_oh

            # Inference per model (simulated parallel)
            per_model_times = {}
            all_dets = []
            for mid in range(num_models):
                roi_list = assignments.get(mid, [])
                if not roi_list:
                    per_model_times[mid] = 0.0; continue
                crops, vinfo = [], []
                for roi in roi_list:
                    ch, cw = roi.crop.shape[:2]
                    sc = max(32/cw, 32/ch) if cw < 32 or ch < 32 else 1.0
                    resized = cv2.resize(roi.crop, None, fx=sc, fy=sc) \
                              if sc != 1.0 else roi.crop
                    crops.append(resized)
                    x, y, bw, bh = roi.bbox
                    vinfo.append((x, y, bw, bh, resized.shape[1], resized.shape[0]))
                t_m = time.perf_counter()
                res_all = models[mid].predict(crops, verbose=False, conf=0.3,
                                              device=DEVICE, stream=False, imgsz=160)
                per_model_times[mid] = time.perf_counter() - t_m
                for (x, y, bw, bh, rw, rh), r_obj in zip(vinfo, res_all):
                    if r_obj and len(r_obj.boxes) > 0:
                        sx, sy = bw / rw, bh / rh
                        for box in r_obj.boxes:
                            bx1, by1, bx2, by2 = box.xyxy[0].cpu().numpy()
                            cls = int(box.cls[0]); conf = float(box.conf[0])
                            all_dets.append(Detection(
                                bbox_xyxy=(int(x+bx1*sx), int(y+by1*sy),
                                           int(x+bx2*sx), int(y+by2*sy)),
                                class_id=cls,
                                class_name=models[mid].names.get(cls, str(cls)),
                                confidence=conf, model_id=mid))

            # SPECULATIVE CARRY-FORWARD:
            # Use prev_ref_dets (already available) instead of waiting
            # for current GOP's ref frame
            last_ref = frame_idx - (frame_idx % GOP_SIZE)
            if last_ref in ref_detections:
                # Ref already done — direct carry
                carry_src = ref_detections[last_ref]
                stats['direct'] += 1
            elif prev_ref_dets is not None:
                # Ref NOT done — speculate with previous ref
                carry_src = prev_ref_dets
            else:
                carry_src = []

            for cd in carry_src:
                if not any(cd.class_id == rd.class_id and
                           compute_iou(cd.bbox_xyxy, rd.bbox_xyxy) > 0.3
                           for rd in all_dets):
                    all_dets.append(cd)

            compute_time = max(per_model_times.values()) if per_model_times else 0.0
            frame_lat = overhead + compute_time

            # Rollback cost: ~1ms for re-applying carry-forward
            rollback_cost = 0.001 if frame_idx in stats.get('rollback_frames', set()) else 0.0

            for r_dict, lat in [(res_no_rb, frame_lat),
                                (res_rb, frame_lat + rollback_cost)]:
                r_dict['frame_latencies'].append(lat)
                r_dict['all_detections'][frame_idx] = all_dets
                r_dict['detection_counts'].append(len(all_dets))

            roi_ct = sum(len(assignments.get(m, [])) for m in range(num_models))
            print(f"  [Spec] Frame {frame_idx:3d} [P-frm] | "
                  f"OH: {overhead*1000:5.1f}ms | "
                  f"Compute: {compute_time*1000:5.1f}ms | "
                  f"Total: {frame_lat*1000:6.1f}ms | "
                  f"ROIs: {roi_ct} | Dets: {len(all_dets):3d}")

    container.close()

    total_spec = stats['commits'] + stats['rollbacks']
    if total_spec > 0:
        print(f"\n  [Spec Stats] Commits: {stats['commits']} | "
              f"Rollbacks: {stats['rollbacks']} | Direct: {stats['direct']} | "
              f"Commit Rate: {stats['commits']/total_spec*100:.1f}%")

    return res_no_rb, res_rb
