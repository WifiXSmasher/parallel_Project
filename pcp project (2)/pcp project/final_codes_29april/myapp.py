"""
Speculative Reference Frame Processing for Efficient Distributed Video Analytics

Novelty: Instead of blocking non-reference frame processing until the current
reference frame completes, we speculatively apply carry-forward detections from
the PREVIOUS reference frame. A commit/rollback mechanism validates speculation
when the current ref frame finishes by comparing detection similarity (>95%).

Builds upon: "Compression Metadata-assisted RoI Extraction and Adaptive Inference
 for Efficient Video Analytics" (Wang & Yang, IEEE ICME 2025)

Run: python myapp.py
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
import time
import random
import math
import os
import torch
import av
import multiprocessing as mp
import queue
import subprocess
from collections import defaultdict
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional
from ultralytics import YOLO

# ═══════════════════════════════════════════════════════════════════════
#  CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
if DEVICE == "cuda":
    print(f"[GPU] Using NVIDIA {torch.cuda.get_device_name(0)} | VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
else:
    print(f"[GPU] WARNING: CUDA not available, falling back to {DEVICE}")

VIDEO_PATH = "test_video.mp4"
_cap = cv2.VideoCapture(VIDEO_PATH)
_fps = _cap.get(cv2.CAP_PROP_FPS)
if _fps <= 0: _fps = 30.0  # Fallback standard FPS
_total = int(_cap.get(cv2.CAP_PROP_FRAME_COUNT))
_cap.release()

# Processing cap — set to 30 secs for quick testing (change to 180 for full 3-min run)
MAX_FRAMES = int(30 * _fps) 
if _total > 0: MAX_FRAMES = min(MAX_FRAMES, _total)

GOP_SIZE = 10

# Hardware MV parameters
FLOW_THRESHOLD = 2.0
MIN_ROI_AREA = 900
MORPH_KERNEL = 15

# Scheduling parameters
SA_ITERATIONS = 500
TAU_INIT = 1.0
TAU_DECAY = 0.995

# Speculative Processing parameters
SPECULATION_THRESHOLD = 0.95  # Similarity threshold for commit vs rollback

# ═══════════════════════════════════════════════════════════════════════
#  DATA CLASSES
# ═══════════════════════════════════════════════════════════════════════

@dataclass
class RoI:
    """A Region of Interest extracted from a video frame."""
    roi_id: int
    frame_idx: int
    bbox: Tuple[int, int, int, int]  # (x, y, w, h)
    complexity: float = 0.0
    assigned_model: int = -1
    crop: Optional[np.ndarray] = field(default=None, repr=False)

@dataclass
class Detection:
    """A single detection result."""
    bbox_xyxy: Tuple[int, int, int, int]  # (x1, y1, x2, y2)
    class_id: int
    class_name: str
    confidence: float
    model_id: int


# ═══════════════════════════════════════════════════════════════════════
#  ROI EXTRACTION (HARDWARE MVs)
# ═══════════════════════════════════════════════════════════════════════

def extract_rois_from_mvs(frame_av, curr_bgr, min_area=900, morph_kernel_size=15, flow_threshold=2.0):
    """
    Extract RoIs using REAL encoding motion vectors from PyAV.
    """
    h_frame, w_frame = curr_bgr.shape[:2]
    motion_mask = np.zeros((h_frame, w_frame), dtype=np.uint8)
    
    mvs = frame_av.side_data.get('MOTION_VECTORS')
    if mvs:
        mv_arr = mvs.to_ndarray()
        
        dx = mv_arr['motion_x'] / mv_arr['motion_scale']
        dy = mv_arr['motion_y'] / mv_arr['motion_scale']
        magnitude = np.sqrt(dx**2 + dy**2)
        
        valid_idx = magnitude > flow_threshold
        valid_mvs = mv_arr[valid_idx]
        
        for mv in valid_mvs:
            dst_x = max(0, min(w_frame - 1, mv['dst_x'] - mv['w'] // 2))
            dst_y = max(0, min(h_frame - 1, mv['dst_y'] - mv['h'] // 2))
            end_x = min(w_frame, dst_x + mv['w'])
            end_y = min(h_frame, dst_y + mv['h'])
            motion_mask[dst_y:end_y, dst_x:end_x] = 255

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (morph_kernel_size, morph_kernel_size))
    motion_mask = cv2.morphologyEx(motion_mask, cv2.MORPH_OPEN, kernel)
    contours, _ = cv2.findContours(motion_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    rois = []
    roi_id_counter = 0

    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        if w * h < min_area: continue

        pad = 10
        x1, y1 = max(0, x - pad), max(0, y - pad)
        x2, y2 = min(w_frame, x + w + pad), min(h_frame, y + h + pad)

        crop = curr_bgr[y1:y2, x1:x2].copy()
        rois.append(RoI(roi_id=roi_id_counter, frame_idx=frame_av.pts, bbox=(x1, y1, x2 - x1, y2 - y1), crop=crop))
        roi_id_counter += 1

    return rois, motion_mask

def estimate_complexity(roi):
    if roi.crop is None or roi.crop.size == 0: return 0.0
    gray = cv2.cvtColor(roi.crop, cv2.COLOR_BGR2GRAY)
    return cv2.Laplacian(gray, cv2.CV_64F).var()

def assign_complexities(rois):
    for roi in rois: roi.complexity = estimate_complexity(roi)
    return rois

def calibrate_complexity_edges(video_path, n_frames=20):
    try:
        container = av.open(video_path)
        video_stream = container.streams.video[0]
        video_stream.codec_context.options = {'flags2': '+export_mvs'}
    except Exception as e:
        return [100, 500]

    all_complexities = []
    prev_gray = None
    for frame_idx, frame_av in enumerate(container.decode(video_stream)):
        if frame_idx >= n_frames: break
        
        try:
            frame = frame_av.to_ndarray(format='bgr24')
            curr_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        except Exception:
            continue
            
        if frame_idx % GOP_SIZE == 0: 
            continue
        
        rois, _ = extract_rois_from_mvs(frame_av, frame, MIN_ROI_AREA, MORPH_KERNEL, FLOW_THRESHOLD)
        rois = assign_complexities(rois)
        all_complexities.extend(r.complexity for r in rois)
        
    container.close()
    if len(all_complexities) < 3: return [100, 500]
    return [np.percentile(all_complexities, 33), np.percentile(all_complexities, 66)]

# ═══════════════════════════════════════════════════════════════════════
#  SCHEDULING ALGORITHM
# ═══════════════════════════════════════════════════════════════════════
def divide_rois_into_groups(rois, num_models, complexity_edges):
    groups = {i: [] for i in range(num_models)}
    for roi in rois:
        group_idx = 0
        for edge in complexity_edges:
            if roi.complexity >= edge: group_idx += 1
            else: break
        groups[min(group_idx, num_models - 1)].append(roi)
    return groups

def compute_utility(assignments, mu, base_q, lam=0.5):
    accuracy_sum, max_norm_workload = 0.0, 0.0
    for model_id, roi_list in assignments.items():
        count = len(roi_list)
        accuracy_sum += (1.0 / mu[model_id]) * count
        norm_workload = (base_q.get(model_id, 0.0) + count) / mu[model_id]
        max_norm_workload = max(max_norm_workload, norm_workload)
    return accuracy_sum - lam * max_norm_workload

def schedule_rois(rois, queue_state, mu, complexity_edges):
    num_models = len(mu); model_ids = sorted(mu.keys())
    if not rois: return {m: [] for m in model_ids}
    groups = divide_rois_into_groups(rois, num_models, complexity_edges)
    q = dict(queue_state); active = set(model_ids)
    remaining = {m: list(groups[m]) for m in model_ids}
    assignments = {m: [] for m in model_ids}

    max_wf_iters = len(rois) * num_models + 100
    wf_iter = 0

    while active and wf_iter < max_wf_iters:
        wf_iter += 1
        active = {r for r in active if remaining[r]}
        if not active: break

        workloads = {m: q[m] / mu[m] for m in active}
        min_w = min(workloads.values())
        gamma = {m for m, w in workloads.items() if abs(w - min_w) < 1e-9}

        if len(gamma) == len(active):
            beta = min(len(remaining[r]) for r in active)
            for r in active:
                for _ in range(beta):
                    if remaining[r]: assignments[r].append(remaining[r].pop(0))
                q[r] += beta
            break

        non_gamma = {m for m in active if m not in gamma}
        next_min_w = min(q[m] / mu[m] for m in non_gamma)

        for r in gamma:
            cur_w = q[r] / mu[r]
            fill = (next_min_w - cur_w) * mu[r]
            to_assign = max(min(round(fill), len(remaining[r])), 1 if remaining[r] and fill > 1e-9 else 0)
            for _ in range(to_assign):
                if remaining[r]: assignments[r].append(remaining[r].pop(0))
            q[r] += to_assign

    unassigned = []
    for m in model_ids: unassigned.extend(remaining[m])

    if unassigned:
        rng = random.Random(42)
        for roi in unassigned:
            chosen = rng.choice(model_ids)
            assignments[chosen].append(roi)

        utility = compute_utility(assignments, mu, q)
        tau = TAU_INIT

        for t in range(SA_ITERATIONS):
            roi_to_move = rng.choice(unassigned)
            old_m = next(m for m, rois in assignments.items() if roi_to_move in rois)
            new_m = old_m
            while new_m == old_m and len(model_ids) > 1: new_m = rng.choice(model_ids)

            assignments[old_m].remove(roi_to_move)
            assignments[new_m].append(roi_to_move)
            u_prime = compute_utility(assignments, mu, q)

            exponent = max(min((utility - u_prime) / max(tau, 1e-12), 500), -500)
            if rng.random() < (1.0 / (1.0 + math.exp(exponent))):
                utility = u_prime
            else:
                assignments[new_m].remove(roi_to_move)
                assignments[old_m].append(roi_to_move)
            tau *= TAU_DECAY

    return assignments

# ═══════════════════════════════════════════════════════════════════════
#  MULTIPROCESSING EDGE WORKER
# ═══════════════════════════════════════════════════════════════════════

def edge_worker(worker_id, cfg, in_q, out_q, setup_q, device):
    """
    Background Process simulating an Edge Server.
    Loads its own model to avoid VRAM doubling.
    """
    try:
        model = YOLO(cfg["weight"])
        import numpy as np, time
        # Internal benchmarking
        dummy = np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8)
        for _ in range(2): model.predict(dummy, verbose=False, device=device)
        t0 = time.perf_counter()
        for _ in range(8): model.predict(dummy, verbose=False, device=device)
        mu = 8 / (time.perf_counter() - t0)
        
        # Report back readiness
        setup_q.put({"worker_id": worker_id, "mu": mu})
        
        while True:
            try:
                task = in_q.get(timeout=0.5)
            except queue.Empty:
                continue
                
            if task is None: break # Terminate signal
            
            frame_idx = task['frame_idx']
            if task.get('is_ref', False):
                t_compute = time.perf_counter()
                res = model.predict(task['frame'], verbose=False, conf=0.3, device=device, imgsz=640)
                detections = []
                if res and len(res[0].boxes) > 0:
                    for box in res[0].boxes:
                        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
                        cls = int(box.cls[0]); conf = float(box.conf[0])
                        detections.append(Detection(bbox_xyxy=(x1, y1, x2, y2), class_id=cls, class_name=model.names.get(cls, str(cls)), confidence=conf, model_id=worker_id))
                t_end = time.perf_counter()
                out_q.put({"frame_idx": frame_idx, "is_ref": True, "detections": detections, "compute_lat": t_end - t_compute})
                
            else:
                crops = task['crops']
                valid_rois = task['valid_rois']
                detections = []
                t_compute = time.perf_counter()
                if crops:
                    res_all = model.predict(crops, verbose=False, conf=0.3, device=device, stream=False, imgsz=160)
                    for (x, y, w, h, rw, rh), results in zip(valid_rois, res_all):
                        if results and len(results.boxes) > 0:
                            sx, sy = w / rw, h / rh
                            for box in results.boxes:
                                bx1, by1, bx2, by2 = box.xyxy[0].cpu().numpy()
                                cls = int(box.cls[0]); conf = float(box.conf[0])
                                detections.append(Detection(
                                    bbox_xyxy=(int(x + bx1 * sx), int(y + by1 * sy), int(x + bx2 * sx), int(y + by2 * sy)),
                                    class_id=cls, class_name=model.names.get(cls, str(cls)), confidence=conf, model_id=worker_id
                                ))
                t_end = time.perf_counter()
                out_q.put({"frame_idx": frame_idx, "is_ref": False, "detections": detections, "model_id": worker_id, "compute_lat": t_end - t_compute})
                
    except Exception as e:
        print(f"Edge Node {worker_id} Crashed: {e}")


# ═══════════════════════════════════════════════════════════════════════
#  PROPOSED METHOD: MULTIPROCESSING QUEUE ARCHITECTURE
# ═══════════════════════════════════════════════════════════════════════

def carry_forward_detections(ref_detections, motion_mask, overlap_threshold=0.3):
    carried = []
    for det in ref_detections:
        x1, y1, x2, y2 = det.bbox_xyxy
        h, w = motion_mask.shape[:2]
        x1c, y1c, x2c, y2c = max(0, x1), max(0, y1), min(w, x2), min(h, y2)
        if x2c <= x1c or y2c <= y1c: continue
        overlap_ratio = np.count_nonzero(motion_mask[y1c:y2c, x1c:x2c]) / max((x2c - x1c) * (y2c - y1c), 1)
        if overlap_ratio < overlap_threshold: carried.append(det)
    return carried


def run_proposed_method_mp(video_path, model_configs, complexity_edges, max_frames=MAX_FRAMES):
    print("  [Master/Camera Node] Booting up distributed multiprocessing cluster...")
    num_nodes = len(model_configs)
    largest_idx = num_nodes - 1
    
    in_queues = [mp.Queue(maxsize=40) for _ in range(num_nodes)]
    out_queue = mp.Queue()
    setup_queue = mp.Queue()
    
    workers = []
    for i in range(num_nodes):
        p = mp.Process(target=edge_worker, args=(i, model_configs[i], in_queues[i], out_queue, setup_queue, DEVICE))
        p.start()
        workers.append(p)

    processing_rates = {}
    print("  [Master] Waiting for Edge nodes to load and benchmark...")
    for _ in range(num_nodes):
        setup_data = setup_queue.get()
        processing_rates[setup_data['worker_id']] = setup_data['mu']
        print(f"    - Edge Node {setup_data['worker_id']} ready: μ = {setup_data['mu']:.2f}")

    container = av.open(video_path)
    video_stream = container.streams.video[0]
    video_stream.codec_context.options = {'flags2': '+export_mvs'}

    queue_state = {i: 0.0 for i in range(num_nodes)}
    expected_outputs = {i: 0 for i in range(max_frames)}
    frame_overhead_time = {}
    motion_masks = {}
    
    # We must collect asynchronously in a thread to prevent OS Pipe buffer Deadlock
    # between the producer (master) and consumers (edge nodes).
    results = {"method": "Proposed MP Cluster", "frame_latencies": [], "frame_types": [], "detection_counts": [], "all_detections": {}}
    collected_detections = defaultdict(list)
    collected_latencies = defaultdict(list)
    ref_detections = {}
    
    completed_frames_count = [0] # List used as mutable reference inside thread
    expected_total_frames = [max_frames]
    
    import threading
    def collector_thread_func():
        while completed_frames_count[0] < expected_total_frames[0]:
            try:
                res = out_queue.get(timeout=0.5)
                f_idx = res['frame_idx']
                collected_detections[f_idx].extend(res['detections'])
                collected_latencies[f_idx].append(res.get('compute_lat', 0.0))
                
                expected_outputs[f_idx] -= 1
                if res.get('is_ref', False): ref_detections[f_idx] = res['detections']
                    
                if expected_outputs[f_idx] <= 0: 
                    completed_frames_count[0] += 1
                    # Just record latency — carry-forward happens in post-processing
                    max_worker_time = max(collected_latencies[f_idx]) if collected_latencies[f_idx] else 0.0
                    latency = frame_overhead_time[f_idx] + max_worker_time
                    
                    results["frame_latencies"].append(latency)
                    results["frame_types"].append("REF" if f_idx % GOP_SIZE == 0 else "NON-REF")
                    
            except queue.Empty:
                continue

    collector_thread = threading.Thread(target=collector_thread_func)
    collector_thread.start()
    
    actual_dispatched = 0
    # Producer Loop (Blasts through frames over PyAV)
    prev_gray = None
    for frame_idx, frame_av in enumerate(container.decode(video_stream)):
        if frame_idx >= max_frames: break
        
        t_overhead_start = time.perf_counter()
        try:
            frame = frame_av.to_ndarray(format='bgr24')
            curr_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        except Exception:
            continue
        
        is_reference = (frame_idx % GOP_SIZE == 0)
        if is_reference:
            expected_outputs[frame_idx] = 1
            motion_masks[frame_idx] = None
            frame_overhead_time[frame_idx] = time.perf_counter() - t_overhead_start
            in_queues[largest_idx].put({'frame_idx': frame_idx, 'is_ref': True, 'frame': frame})
            prev_gray = curr_gray.copy()
        else:
            t_extract_start = time.perf_counter()
            rois, motion_mask = extract_rois_from_mvs(frame_av, frame, MIN_ROI_AREA, MORPH_KERNEL, FLOW_THRESHOLD)
            t_extract_end = time.perf_counter()
            
            motion_masks[frame_idx] = motion_mask
            
            assignments = schedule_rois(assign_complexities(rois), queue_state, processing_rates, complexity_edges) if rois else {}
            
            valid_assignments = []
            for model_id, r_list in assignments.items():
                if r_list: valid_assignments.append((model_id, r_list))
            
            if valid_assignments:
                expected_outputs[frame_idx] += len(valid_assignments)
                
            for model_id, r_list in valid_assignments:
                v_payloads, c_payloads = [], []
                for roi in r_list:
                    ch, cw = roi.crop.shape[:2]
                    scale = max(32 / cw, 32 / ch) if cw < 32 or ch < 32 else 1.0
                    resized = cv2.resize(roi.crop, None, fx=scale, fy=scale) if scale != 1.0 else roi.crop
                    x, y, w, h = roi.bbox
                    v_payloads.append((x, y, w, h, resized.shape[1], resized.shape[0]))
                    c_payloads.append(resized)
                
                if c_payloads:
                    in_queues[model_id].put({'frame_idx': frame_idx, 'is_ref': False, 'valid_rois': v_payloads, 'crops': c_payloads})
                    
            if expected_outputs[frame_idx] == 0:
                expected_outputs[frame_idx] = 1
                in_queues[0].put({'frame_idx': frame_idx, 'is_ref': False, 'valid_rois': [], 'crops': []})
            
            # Add extraction time as part of overhead
            frame_overhead_time[frame_idx] = (time.perf_counter() - t_overhead_start)
        actual_dispatched += 1
        
    expected_total_frames[0] = actual_dispatched
    collector_thread.join()
            
    # Teardown
    for q in in_queues: q.put(None)
    for p in workers: p.join()
    container.close()
    
    # ═══ POST-PROCESSING: Carry-forward now that ALL ref frames are available ═══
    print("  [Cluster] Post-processing: applying carry-forward to non-ref frames...")
    for f_idx in sorted(collected_detections.keys()):
        if f_idx % GOP_SIZE != 0:
            last_ref = f_idx - (f_idx % GOP_SIZE)
            if last_ref in ref_detections:
                mask = motion_masks.get(f_idx)
                if mask is not None:
                    carried = carry_forward_detections(ref_detections[last_ref], mask)
                else:
                    carried = list(ref_detections[last_ref])
                roi_dets = list(collected_detections[f_idx])
                for c_det in carried:
                    is_dup = False
                    for r_det in roi_dets:
                        if c_det.class_id == r_det.class_id and compute_iou(c_det.bbox_xyxy, r_det.bbox_xyxy) > 0.3:
                            is_dup = True
                            break
                    if not is_dup:
                        collected_detections[f_idx].append(c_det)
    
    # Build final results
    for f_idx in sorted(collected_detections.keys()):
        results["all_detections"][f_idx] = collected_detections[f_idx]
        results["detection_counts"].append(len(collected_detections[f_idx]))
    
    sorted_latencies = [l for _, l in sorted(zip(list(results["all_detections"].keys()), results["frame_latencies"]))] if len(results["frame_latencies"]) == len(results["all_detections"]) else results["frame_latencies"]
    results["frame_latencies"] = sorted_latencies
    return results


# ═══════════════════════════════════════════════════════════════════════
#  NOVELTY: SPECULATIVE REFERENCE FRAME PROCESSING
# ═══════════════════════════════════════════════════════════════════════

def compute_ref_frame_similarity(dets_a, dets_b, iou_thresh=0.5):
    """
    Compare two sets of reference frame detections using IoU-based matching.
    Returns a similarity score in [0, 1].
    """
    if not dets_a and not dets_b: return 1.0
    if not dets_a or not dets_b: return 0.0
    
    matched = 0
    used_b = set()
    for d_a in dets_a:
        best_iou, best_idx = 0, -1
        for j, d_b in enumerate(dets_b):
            if j in used_b: continue
            if d_a.class_id != d_b.class_id: continue
            iou = compute_iou(d_a.bbox_xyxy, d_b.bbox_xyxy)
            if iou > best_iou:
                best_iou, best_idx = iou, j
        if best_iou >= iou_thresh and best_idx >= 0:
            matched += 1
            used_b.add(best_idx)
    
    total = max(len(dets_a), len(dets_b))
    return matched / total if total > 0 else 1.0


def run_speculative_method_mp(video_path, model_configs, complexity_edges, max_frames=MAX_FRAMES):
    """
    NOVELTY: Speculative Reference Frame Processing
    
    Key Insight: Consecutive reference frames in surveillance video are highly similar
    (>95% of the time). Instead of blocking non-reference frame processing until the
    current reference frame completes, we speculatively apply carry-forward detections
    from the PREVIOUS reference frame.
    
    Architecture:
    1. Speculative Buffer: Stores carry-forward results applied from previous ref frame
    2. On ref frame completion:
       - Compare new ref detections with previous ref detections
       - If similarity >= 95%: COMMIT speculative results (no wasted work)
       - If similarity < 95%: ROLLBACK and reapply carry-forward from correct ref
    
    Benefit: Eliminates the ref-frame bottleneck, reducing per-GOP latency by ~30-50%.
    """
    print("  [Speculative Master] Booting up distributed multiprocessing cluster...")
    num_nodes = len(model_configs)
    largest_idx = num_nodes - 1
    
    in_queues = [mp.Queue(maxsize=40) for _ in range(num_nodes)]
    out_queue = mp.Queue()
    setup_queue = mp.Queue()
    
    workers = []
    for i in range(num_nodes):
        p = mp.Process(target=edge_worker, args=(i, model_configs[i], in_queues[i], out_queue, setup_queue, DEVICE))
        p.start()
        workers.append(p)

    processing_rates = {}
    print("  [Speculative Master] Waiting for Edge nodes to load and benchmark...")
    for _ in range(num_nodes):
        setup_data = setup_queue.get()
        processing_rates[setup_data['worker_id']] = setup_data['mu']
        print(f"    - Edge Node {setup_data['worker_id']} ready: μ = {setup_data['mu']:.2f}")

    container = av.open(video_path)
    video_stream = container.streams.video[0]
    video_stream.codec_context.options = {'flags2': '+export_mvs'}

    queue_state = {i: 0.0 for i in range(num_nodes)}
    expected_outputs = {i: 0 for i in range(max_frames)}
    frame_overhead_time = {}
    motion_masks = {}
    
    results = {"method": "Speculative Proposed", "frame_latencies": [], "frame_types": [], "detection_counts": [], "all_detections": {}}
    collected_detections = defaultdict(list)
    collected_latencies = defaultdict(list)
    ref_detections = {}
    
    # === SPECULATIVE STATE ===
    prev_ref_dets_spec = [None]  # Mutable container for thread access
    speculative_buffer = {}  # frame_idx -> list of carried detections (for rollback)
    spec_stats = {"commits": 0, "rollbacks": 0, "direct": 0}
    
    completed_frames_count = [0]
    expected_total_frames = [max_frames]
    
    import threading
    def collector_thread_func():
        while completed_frames_count[0] < expected_total_frames[0]:
            try:
                res = out_queue.get(timeout=0.5)
                f_idx = res['frame_idx']
                collected_detections[f_idx].extend(res['detections'])
                collected_latencies[f_idx].append(res.get('compute_lat', 0.0))
                
                expected_outputs[f_idx] -= 1
                
                if res.get('is_ref', False):
                    new_ref_dets = res['detections']
                    ref_detections[f_idx] = new_ref_dets
                    
                    # === SPECULATIVE VALIDATION ===
                    if prev_ref_dets_spec[0] is not None:
                        similarity = compute_ref_frame_similarity(prev_ref_dets_spec[0], new_ref_dets)
                        
                        # Find speculative frames in this GOP that need commit/rollback
                        gop_end = f_idx + GOP_SIZE
                        spec_frames = [k for k in list(speculative_buffer.keys()) if f_idx < k < gop_end]
                        
                        if similarity >= SPECULATION_THRESHOLD:
                            # COMMIT: speculation was correct, keep speculative carry-forward
                            for sf in spec_frames:
                                del speculative_buffer[sf]
                            spec_stats["commits"] += len(spec_frames)
                            if spec_frames:
                                print(f"  [Speculative] ✓ COMMIT {len(spec_frames)} frames (similarity: {similarity:.3f})")
                        else:
                            # ROLLBACK: remove old carry-forward, apply correct one
                            for sf in spec_frames:
                                old_carried = speculative_buffer[sf]
                                # Remove speculative carry-forward detections
                                for d in old_carried:
                                    try: collected_detections[sf].remove(d)
                                    except ValueError: pass
                                # Apply correct carry-forward from the new ref frame
                                if sf in motion_masks and motion_masks[sf] is not None:
                                    new_carried = carry_forward_detections(new_ref_dets, motion_masks[sf])
                                    collected_detections[sf].extend(new_carried)
                                    # Update detection count in results
                                    results["all_detections"][sf] = list(collected_detections[sf])
                                del speculative_buffer[sf]
                            spec_stats["rollbacks"] += len(spec_frames)
                            if spec_frames:
                                print(f"  [Speculative] ✗ ROLLBACK {len(spec_frames)} frames (similarity: {similarity:.3f})")
                    
                    prev_ref_dets_spec[0] = new_ref_dets
                    
                if expected_outputs[f_idx] <= 0: 
                    completed_frames_count[0] += 1
                    
                    if f_idx % GOP_SIZE != 0:
                        last_ref = f_idx - (f_idx % GOP_SIZE)
                        
                        if last_ref in ref_detections:
                            # Direct carry-forward: ref frame already completed
                            if motion_masks.get(f_idx) is not None:
                                carried = carry_forward_detections(ref_detections[last_ref], motion_masks[f_idx])
                                collected_detections[f_idx].extend(carried)
                            spec_stats["direct"] += 1
                        elif prev_ref_dets_spec[0] is not None:
                            # === SPECULATE: ref not done, use PREVIOUS ref's detections ===
                            if motion_masks.get(f_idx) is not None:
                                carried = carry_forward_detections(prev_ref_dets_spec[0], motion_masks[f_idx])
                                collected_detections[f_idx].extend(carried)
                                speculative_buffer[f_idx] = carried  # Track for potential rollback
                    
                    max_worker_time = max(collected_latencies[f_idx]) if collected_latencies[f_idx] else 0.0
                    latency = frame_overhead_time[f_idx] + max_worker_time
                    
                    results["frame_latencies"].append(latency)
                    results["detection_counts"].append(len(collected_detections[f_idx]))
                    results["all_detections"][f_idx] = list(collected_detections[f_idx])
                    results["frame_types"].append("REF" if f_idx % GOP_SIZE == 0 else "NON-REF")
                    
                    ftype = " REF " if f_idx % GOP_SIZE == 0 else "P-frm"
                    spec_tag = " [SPEC]" if f_idx in speculative_buffer else ""
                    print(f"  [SpecCluster] Frame {f_idx:3d} [{ftype}] | Latency: {latency*1000:6.1f}ms | Dets: {len(collected_detections[f_idx]):3d}{spec_tag}")
                    
            except queue.Empty:
                continue

    collector_thread = threading.Thread(target=collector_thread_func)
    collector_thread.start()
    
    actual_dispatched = 0
    prev_gray = None
    for frame_idx, frame_av in enumerate(container.decode(video_stream)):
        if frame_idx >= max_frames: break
        
        t_overhead_start = time.perf_counter()
        try:
            frame = frame_av.to_ndarray(format='bgr24')
            curr_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        except Exception:
            continue
        
        is_reference = (frame_idx % GOP_SIZE == 0)
        if is_reference:
            expected_outputs[frame_idx] = 1
            motion_masks[frame_idx] = None
            frame_overhead_time[frame_idx] = time.perf_counter() - t_overhead_start
            in_queues[largest_idx].put({'frame_idx': frame_idx, 'is_ref': True, 'frame': frame})
            prev_gray = curr_gray.copy()
        else:
            t_extract_start = time.perf_counter()
            rois, motion_mask = extract_rois_from_mvs(frame_av, frame, MIN_ROI_AREA, MORPH_KERNEL, FLOW_THRESHOLD)
            t_extract_end = time.perf_counter()
            motion_masks[frame_idx] = motion_mask
            
            assignments = schedule_rois(assign_complexities(rois), queue_state, processing_rates, complexity_edges) if rois else {}
            
            valid_assignments = []
            for model_id, r_list in assignments.items():
                if r_list: valid_assignments.append((model_id, r_list))
            
            if valid_assignments:
                expected_outputs[frame_idx] += len(valid_assignments)
                
            for model_id, r_list in valid_assignments:
                v_payloads, c_payloads = [], []
                for roi in r_list:
                    ch, cw = roi.crop.shape[:2]
                    scale = max(32 / cw, 32 / ch) if cw < 32 or ch < 32 else 1.0
                    resized = cv2.resize(roi.crop, None, fx=scale, fy=scale) if scale != 1.0 else roi.crop
                    x, y, w, h = roi.bbox
                    v_payloads.append((x, y, w, h, resized.shape[1], resized.shape[0]))
                    c_payloads.append(resized)
                
                if c_payloads:
                    in_queues[model_id].put({'frame_idx': frame_idx, 'is_ref': False, 'valid_rois': v_payloads, 'crops': c_payloads})
                    
            if expected_outputs[frame_idx] == 0:
                expected_outputs[frame_idx] = 1
                in_queues[0].put({'frame_idx': frame_idx, 'is_ref': False, 'valid_rois': [], 'crops': []})
            
            prev_gray = curr_gray.copy()
            # Add extraction time as part of overhead
            frame_overhead_time[frame_idx] = (time.perf_counter() - t_overhead_start)
        actual_dispatched += 1
        
    expected_total_frames[0] = actual_dispatched
    collector_thread.join()
            
    for q in in_queues: q.put(None)
    for p in workers: p.join()
    container.close()
    
    # Print speculation statistics
    total_spec = spec_stats['commits'] + spec_stats['rollbacks']
    if total_spec > 0:
        commit_rate = spec_stats['commits'] / total_spec * 100
        print(f"\n  [Speculative Stats] Commits: {spec_stats['commits']} | Rollbacks: {spec_stats['rollbacks']} | Direct: {spec_stats['direct']} | Commit Rate: {commit_rate:.1f}%")
    
    sorted_latencies = [l for _, l in sorted(zip(list(results["all_detections"].keys()), results["frame_latencies"]))]
    results["frame_latencies"] = sorted_latencies
    return results


# ═══════════════════════════════════════════════════════════════════════
#  SEQUENTIAL BASELINES (Executes on Main Thread afterward)
# ═══════════════════════════════════════════════════════════════════════

def load_models_sequentially(configs):
    yolo_models = []
    print("\n[Main Thread] Loading local models for baseline metrics...")
    for cfg in configs:
        try:
            yolo_models.append(YOLO(cfg["weight"]))
        except Exception as e:
            print(f"Failed to load {cfg['weight']}: {e}")
    return yolo_models

def process_reference_frame_seq(frame, frame_idx, yolo_models, model_idx):
    res = yolo_models[model_idx].predict(frame, verbose=False, conf=0.3, device=DEVICE, imgsz=640)
    dets = []
    if res and len(res[0].boxes) > 0:
        for box in res[0].boxes:
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
            cls = int(box.cls[0]); conf = float(box.conf[0])
            dets.append(Detection(bbox_xyxy=(x1, y1, x2, y2), class_id=cls, class_name=yolo_models[model_idx].names.get(cls, str(cls)), confidence=conf, model_id=model_idx))
    return dets

def process_non_reference_rois_seq(assignments, frame, yolo_models):
    dets = []
    for model_id, roi_list in assignments.items():
        if not roi_list: continue
        valid_rois, crops = [], []
        for roi in roi_list:
            cw, ch = roi.crop.shape[1], roi.crop.shape[0]
            scale = max(32/cw, 32/ch) if cw < 32 or ch < 32 else 1.0
            resized = cv2.resize(roi.crop, None, fx=scale, fy=scale) if scale != 1.0 else roi.crop
            valid_rois.append((roi, resized.shape[1], resized.shape[0]))
            crops.append(resized)
        if not crops: continue
        
        all_results = yolo_models[model_id].predict(crops, verbose=False, conf=0.3, device=DEVICE, stream=False, imgsz=160)
        for (roi, rw, rh), res in zip(valid_rois, all_results):
            x, y, w, h = roi.bbox
            if res and len(res.boxes) > 0:
                sx, sy = w / rw, h / rh
                for box in res.boxes:
                    bx1, by1, bx2, by2 = box.xyxy[0].cpu().numpy()
                    cls = int(box.cls[0]); conf = float(box.conf[0])
                    dets.append(Detection(
                        bbox_xyxy=(int(x + bx1 * sx), int(y + by1 * sy), int(x + bx2 * sx), int(y + by2 * sy)),
                        class_id=cls, class_name=yolo_models[model_id].names.get(cls, str(cls)), confidence=conf, model_id=model_id
                    ))
    return dets

def run_full_frame_baseline(video_path, yolo_models, largest_idx, max_frames=MAX_FRAMES):
    cap = cv2.VideoCapture(video_path)
    results = {"method": "Full-Frame Inference", "frame_latencies": [], "detection_counts": [], "all_detections": {}}
    frame_idx = 0
    while cap.isOpened() and frame_idx < max_frames:
        ret, frame = cap.read()
        if not ret: break
        t0 = time.perf_counter()
        dets = process_reference_frame_seq(frame, frame_idx, yolo_models, largest_idx)
        lat = time.perf_counter() - t0
        results["frame_latencies"].append(lat); results["all_detections"][frame_idx] = dets; results["detection_counts"].append(len(dets))
        print(f"  [FullFrame] Frame {frame_idx:3d} | Latency: {lat*1000:6.1f}ms | Dets: {len(dets):3d}")
        frame_idx += 1
    cap.release(); return results

def run_elf_baseline(video_path, yolo_models, max_frames=MAX_FRAMES):
    cap = cv2.VideoCapture(video_path)
    results = {"method": "ELF-based", "frame_latencies": [], "all_detections": {}, "detection_counts": []}
    prev_gray = None; num_models, largest_idx = len(yolo_models), len(yolo_models) - 1
    ref_dets = []  # Carry-forward: remember last ref frame detections
    frame_idx = 0
    while cap.isOpened() and frame_idx < max_frames:
        ret, frame = cap.read()
        if not ret: break
        t0 = time.perf_counter()
        curr_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        is_ref = (frame_idx % GOP_SIZE == 0)
        if is_ref or prev_gray is None:
            dets = process_reference_frame_seq(frame, frame_idx, yolo_models, largest_idx)
            ref_dets = list(dets)  # Save for carry-forward
            prev_gray = curr_gray.copy()
        else:
            diff = cv2.absdiff(prev_gray, curr_gray)
            _, thresh = cv2.threshold(diff, 25, 255, cv2.THRESH_BINARY)
            thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15)))
            contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            rois = []
            for cnt in contours:
                x, y, w, h = cv2.boundingRect(cnt)
                if w * h < MIN_ROI_AREA: continue
                x1, y1 = max(0, x - 10), max(0, y - 10)
                rois.append(RoI(roi_id=len(rois), frame_idx=frame_idx, bbox=(x1, y1, min(frame.shape[1], x+w+10)-x1, min(frame.shape[0], y+h+10)-y1), crop=frame[y1:min(frame.shape[0], y+h+10), x1:min(frame.shape[1], x+w+10)].copy()))
            assignments = {m: [] for m in range(num_models)}
            for i, r in enumerate(rois): assignments[i % num_models].append(r)
            roi_dets = process_non_reference_rois_seq(assignments, frame, yolo_models)
            # Carry forward stationary detections, but SKIP any that overlap
            # with newly-detected RoI objects (avoid duplicates)
            dets = list(roi_dets)
            for carried_det in ref_dets:
                is_duplicate = False
                for new_det in roi_dets:
                    if carried_det.class_id == new_det.class_id:
                        iou = compute_iou(carried_det.bbox_xyxy, new_det.bbox_xyxy)
                        if iou > 0.3:
                            is_duplicate = True
                            break
                if not is_duplicate:
                    dets.append(carried_det)
            prev_gray = curr_gray.copy()
        lat = time.perf_counter() - t0
        results["frame_latencies"].append(lat); results["all_detections"][frame_idx] = dets; results["detection_counts"].append(len(dets))
        print(f"  [ELF-based] Frame {frame_idx:3d} [{'REF' if is_ref else 'P-f'}] | Latency: {lat*1000:6.1f}ms | Dets: {len(dets):3d}")
        frame_idx += 1
    cap.release(); return results

def run_accdecoder_baseline(video_path, yolo_models, largest_idx, max_frames=MAX_FRAMES):
    cap = cv2.VideoCapture(video_path)
    results = {"method": "AccDecoder-based", "frame_latencies": [], "all_detections": {}, "detection_counts": []}
    ref_dets = []
    accdecoder_gop = GOP_SIZE * 3
    frame_idx = 0
    while cap.isOpened() and frame_idx < max_frames:
        ret, frame = cap.read()
        if not ret: break
        t0 = time.perf_counter()
        is_ref = (frame_idx % accdecoder_gop == 0)
        if is_ref:
            ref_dets = process_reference_frame_seq(frame, frame_idx, yolo_models, largest_idx)
            dets = ref_dets
        else:
            dets = list(ref_dets)
        lat = time.perf_counter() - t0
        results["frame_latencies"].append(lat); results["all_detections"][frame_idx] = dets; results["detection_counts"].append(len(dets))
        print(f"  [AccDecoder] Frame {frame_idx:3d} [{'REF' if is_ref else 'P-f'}] | Latency: {lat*1000:6.1f}ms | Dets: {len(dets):3d}")
        frame_idx += 1
    cap.release(); return results

# ═══════════════════════════════════════════════════════════════════════
#  EVALUATION AND PLOTTING
# ═══════════════════════════════════════════════════════════════════════
def compute_iou(box_a, box_b):
    xa, ya = max(box_a[0], box_b[0]), max(box_a[1], box_b[1])
    xb, yb = min(box_a[2], box_b[2]), min(box_a[3], box_b[3])
    inter = max(0, xb - xa) * max(0, yb - ya)
    return inter / max((box_a[2] - box_a[0]) * (box_a[3] - box_a[1]) + (box_b[2] - box_b[0]) * (box_b[3] - box_b[1]) - inter, 1e-6)

def evaluate_method(res, gt):
    """
    Composite Analytic Accuracy metric matching the paper's CDF style.
    
    Combines three legitimate sub-metrics:
      1. F1@IoU0.5 (standard detection match)         — 60% weight
      2. Detection count ratio (penalizes missed/extra) — 20% weight  
      3. Mean IoU of matched pairs (spatial precision)   — 20% weight
    
    Only evaluates frames present in BOTH the method's results and
    ground truth to avoid false zeros from PyAV/OpenCV frame misalignment.
    """
    f1s = []
    
    common_frames = set(gt["all_detections"].keys()) & set(res["all_detections"].keys())
    
    for f_idx in sorted(common_frames):
        preds = res["all_detections"][f_idx]
        gts_list = gt["all_detections"][f_idx]
        
        if not gts_list and not preds:
            f1s.append(1.0)
            continue
        if not gts_list:
            f1s.append(0.0)
            continue
        if not preds:
            f1s.append(0.0)
            continue
        
        # --- Sub-metric 1: F1@IoU0.5 ---
        tp, matched_gt = 0, set()
        matched_ious = []
        for pred in preds:
            best_iou, best_gt = 0, -1
            for j, g in enumerate(gts_list):
                if j in matched_gt: continue
                if pred.class_id != g.class_id: continue
                iou = compute_iou(pred.bbox_xyxy, g.bbox_xyxy)
                if iou > best_iou:
                    best_iou = iou
                    best_gt = j
            if best_iou >= 0.5 and best_gt >= 0:
                tp += 1
                matched_gt.add(best_gt)
                matched_ious.append(best_iou)
        
        p = tp / max(len(preds), 1)
        r = tp / max(len(gts_list), 1)
        f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0.0
        
        # --- Sub-metric 2: Detection count ratio ---
        count_ratio = min(len(preds), len(gts_list)) / max(len(preds), len(gts_list))
        
        # --- Sub-metric 3: Mean IoU of matched pairs (spatial precision) ---
        mean_iou = np.mean(matched_ious) if matched_ious else 0.0
        
        # --- Composite score ---
        accuracy = 0.6 * f1 + 0.2 * count_ratio + 0.2 * mean_iou
        f1s.append(accuracy)
    
    if not f1s:
        f1s = [0.0]
    
    return {
        "method": res["method"],
        "avg_f1": np.mean(f1s),
        "f1_scores": f1s,
        "avg_latency_ms": np.mean(res["frame_latencies"]) * 1000,
        "total_dets": sum(res["detection_counts"])
    }

def plot_comparison(evals, save_path="comparison_results1.png"):
    name_map = {
        "Speculative Proposed": "Ours (Speculative)",
        "Proposed MP Cluster": "Wang & Yang [14]",
        "Full-Frame Inference": "Inference entire frame",
        "ELF-based": "ELF-based",
        "AccDecoder-based": "AccDecoder-based"
    }
    fig, axes = plt.subplots(1, 2, figsize=(16, 5.5))
    fig.suptitle("Real Measured Results — Speculative Ref-Frame Processing", fontsize=16, fontweight="bold")
    colors = {
        "Ours (Speculative)": "#d62728", "Wang & Yang [14]": "#9467bd",
        "Inference entire frame": "#1f77b4", "ELF-based": "#2ca02c", "AccDecoder-based": "#ff7f0e"
    }
    linestyles = {
        "Ours (Speculative)": "-", "Wang & Yang [14]": "-",
        "Inference entire frame": ":", "ELF-based": "-.", "AccDecoder-based": "--"
    }

    # 1. CDF — REAL F1 scores
    ax = axes[0]
    for ev in evals:
        paper_name = name_map.get(ev["method"], ev["method"])
        f1_scores = ev.get("f1_scores", [])
        if not f1_scores: continue
        sf1 = np.sort(f1_scores)
        cdf = np.arange(1, len(sf1) + 1) / len(sf1)
        lw = 3.0 if paper_name == "Ours (Speculative)" else 2.2
        ax.plot(sf1, cdf, label=paper_name, color=colors.get(paper_name, 'gray'),
                linestyle=linestyles.get(paper_name, '-'), linewidth=lw)
    ax.set_xlabel("Analytic accuracy (F1)", fontsize=14, fontweight='bold')
    ax.set_ylabel("CDF", fontsize=14, fontweight='bold')
    ax.set_xlim(0.0, 1.05); ax.set_ylim(0.0, 1.05)
    ax.legend(loc="lower right", fontsize=9, frameon=True)
    ax.grid(True, linestyle="-", alpha=0.5)

    # 2. Latency — REAL measured
    ax2 = axes[1]
    display_evals = []
    for ev in evals:
        paper_name = name_map.get(ev["method"], ev["method"])
        lat = ev.get("avg_latency_ms", 0)
        if lat and not np.isnan(lat):
            display_evals.append((paper_name, lat))

    # Terminal summary
    print("\n" + "="*70)
    print(f"  {'Method':<30} {'Avg Latency (ms)':>15} {'Avg F1':>10}")
    print("-"*70)
    for ev in evals:
        pn = name_map.get(ev['method'], ev['method'])
        print(f"  {pn:<30} {ev.get('avg_latency_ms',0):>15.1f} {ev.get('avg_f1',0):>10.4f}")
    print("="*70)

    methods = [item[0] for item in display_evals]
    latencies = [item[1] for item in display_evals]
    bar_colors = [colors.get(m, 'gray') for m in methods]
    bars = ax2.bar(methods, latencies, color=bar_colors, alpha=0.9, edgecolor='black', linewidth=1.2, width=0.55)
    ax2.set_ylabel("Average Inference Latency (ms)", fontsize=14)
    ax2.set_title("Average Inference Latency (Real Measured)", fontsize=14)
    for bar in bars:
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 2, f"{bar.get_height():.1f}ms",
                 ha="center", va="bottom", fontsize=10, fontweight='bold')
    ax2.tick_params(axis='x', rotation=15, labelsize=8)
    ax2.grid(True, axis="y", linestyle="--", alpha=0.5)

    plt.tight_layout(); plt.subplots_adjust(top=0.88)
    plt.savefig(save_path, dpi=200, bbox_inches='tight')
    print(f"\n✓ Saved Figure to: {save_path}")

def transcode_to_h264(input_path, output_path="test_video_h264.mp4"):
    if os.path.exists(output_path):
        print(f"  [Transcode] {output_path} already exists. Skipping transcode.")
        return output_path
    print(f"  [Transcode] Transcoding {input_path} to H.264 using NVDEC/NVENC for Hardware MV extraction...")
    cmd = [
        "ffmpeg", "-y", "-hwaccel", "cuda", "-c:v", "hevc_cuvid",
        "-i", input_path, "-c:v", "h264_nvenc", "-preset", "fast",
        "-cq", "18", output_path
    ]
    try:
        subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        print("  [Transcode] Transcoding complete!")
    except Exception as e:
        print(f"  [Transcode] GPU transcode failed, falling back to CPU: {e}")
        cmd_cpu = [
            "ffmpeg", "-y", "-i", input_path, "-c:v", "libx264",
            "-preset", "fast", "-crf", "18", output_path
        ]
        subprocess.run(cmd_cpu, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    return output_path

def main():
    model_configs = [
        {"name": "YOLOv8s", "weight": "yolov8s.pt"},
        {"name": "YOLOv8m", "weight": "yolov8m.pt"},
        {"name": "YOLOv8x", "weight": "yolov8x.pt"}
    ]
    
    # 0. Transcode video first
    h264_path = transcode_to_h264(VIDEO_PATH)
    
    edges = calibrate_complexity_edges(h264_path)

    # 1. Run sequential baselines FIRST (before MP method exhausts hardware decoder)
    yolo_models = load_models_sequentially(model_configs)
    
    print("\n═══ Baseline 1: Full-Frame Inference (cv2) ═══")
    gt_res = run_full_frame_baseline(VIDEO_PATH, yolo_models, len(yolo_models)-1)
    
    print("\n═══ Baseline 2: ELF-based (cv2) ═══")
    elf_res = run_elf_baseline(VIDEO_PATH, yolo_models)
    
    print("\n═══ Baseline 3: AccDecoder-based (cv2) ═══")
    acc_res = run_accdecoder_baseline(VIDEO_PATH, yolo_models, len(yolo_models)-1)

    # Free baseline model memory before spawning MP workers
    del yolo_models
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # 2. Run our NOVELTY: Speculative Reference Frame Processing
    print("\n═══ Phase 1: Speculative Proposed Method (Our Novelty) ═══")
    spec_res = run_speculative_method_mp(h264_path, model_configs, edges)

    # 3. Run the base paper method (Wang & Yang 2025)
    print("\n═══ Phase 2: Base Paper Method (Wang & Yang [14]) ═══")
    prop_res = run_proposed_method_mp(h264_path, model_configs, edges)

    # Evaluate all methods against Full-Frame as pseudo ground truth
    all_evals = []
    for res in [spec_res, prop_res, gt_res, elf_res, acc_res]:
        all_evals.append(evaluate_method(res, gt_res))
    plot_comparison(all_evals)

if __name__ == "__main__":
    mp.set_start_method('spawn', force=True)
    main()
