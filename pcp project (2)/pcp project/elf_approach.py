import torch
import time
import cv2
import numpy as np
from utils import RoI, Detection, compute_iou

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MIN_ROI_AREA = 900
GOP_SIZE = 15

def run_elf_approach(video_path: str, models: list, max_frames: int) -> dict:
    """
    ELF (MobiCom '21) reimplementation.
    Uses frame differencing + greedy multi-model queue balancing.
    """
    cap = cv2.VideoCapture(video_path)
    results = {'method': 'ELF-based', 'frame_latencies': [],
               'all_detections': {}, 'detection_counts': []}
    num_models   = len(models)
    largest_idx  = num_models - 1
    ref_dets     = []
    prev_gray    = None
    queue_loads  = {i: 0.0 for i in range(num_models)}

    fi = 0
    while cap.isOpened() and fi < max_frames:
        ret, frame = cap.read()
        if not ret: break
        t0        = time.perf_counter()
        curr_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        is_ref    = (fi % GOP_SIZE == 0)

        if is_ref:
            res = models[largest_idx].predict(frame, verbose=False, conf=0.3, device=DEVICE, imgsz=640)
            dets = []
            if res and len(res[0].boxes) > 0:
                for box in res[0].boxes:
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
                    cls  = int(box.cls[0]); conf = float(box.conf[0])
                    dets.append(Detection(
                        bbox_xyxy=(x1, y1, x2, y2), class_id=cls,
                        class_name=models[largest_idx].names.get(cls, str(cls)),
                        confidence=conf, model_id=largest_idx))
                        
            ref_dets = list(dets)
            lat      = time.perf_counter() - t0
            print(f"  [ELF] Frame {fi:3d} [ REF ] | {lat*1000:6.1f}ms | Dets: {len(dets):3d}")
        else:
            if prev_gray is not None:
                diff = cv2.absdiff(curr_gray, prev_gray)
            else:
                diff = np.zeros_like(curr_gray)

            # NOTE: ELF threshold kept high to avoid excessive false positives
            _, thresh = cv2.threshold(diff, 15, 255, cv2.THRESH_BINARY)
            kern   = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
            thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kern)
            contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            h_f, w_f = frame.shape[:2]
            rois = []
            for cnt in contours:
                x, y, wc, hc = cv2.boundingRect(cnt)
                if wc * hc < MIN_ROI_AREA:
                    continue
                x1 = max(0, x-10);      y1 = max(0, y-10)
                x2 = min(w_f, x+wc+10); y2 = min(h_f, y+hc+10)
                rois.append(RoI(roi_id=len(rois), frame_idx=fi,
                                bbox=(x1, y1, x2-x1, y2-y1),
                                crop=frame[y1:y2, x1:x2].copy()))

            # Workload-balanced scheduling (no complexity routing)
            assignments = {i: [] for i in range(num_models)}
            for roi in rois:
                best_m = min(range(num_models), key=lambda m: queue_loads[m])
                assignments[best_m].append(roi)
                queue_loads[best_m] += 1.0

            t_infer = time.perf_counter()
            roi_dets = []
            for model_id, roi_list in assignments.items():
                if not roi_list: continue
                crops, valid = [], []
                for roi in roi_list:
                    cw, ch = roi.crop.shape[1], roi.crop.shape[0]
                    sc     = max(32/cw, 32/ch) if cw < 32 or ch < 32 else 1.0
                    resized = cv2.resize(roi.crop, None, fx=sc, fy=sc) if sc != 1.0 else roi.crop
                    crops.append(resized)
                    valid.append((roi, resized.shape[1], resized.shape[0]))
                
                all_res = models[model_id].predict(crops, verbose=False, conf=0.3, device=DEVICE, stream=False, imgsz=320)
                for (roi, rw, rh), r_obj in zip(valid, all_res):
                    x, y, bw, bh = roi.bbox
                    if r_obj and len(r_obj.boxes) > 0:
                        sx, sy = bw / rw, bh / rh
                        for box in r_obj.boxes:
                            bx1, by1, bx2, by2 = box.xyxy[0].cpu().numpy()
                            cls  = int(box.cls[0]); conf = float(box.conf[0])
                            roi_dets.append(Detection(
                                bbox_xyxy=(int(x+bx1*sx), int(y+by1*sy),
                                           int(x+bx2*sx), int(y+by2*sy)),
                                class_id=cls,
                                class_name=models[model_id].names.get(cls, str(cls)),
                                confidence=conf, model_id=model_id))

            t_infer_ms = (time.perf_counter() - t_infer) * 1000

            dets = list(roi_dets)
            for cd in ref_dets:
                if not any(cd.class_id == rd.class_id and compute_iou(cd.bbox_xyxy, rd.bbox_xyxy) > 0.3 for rd in roi_dets):
                    dets.append(cd)

            lat = time.perf_counter() - t0
            print(f"  [ELF] Frame {fi:3d} [P-frm] | Total: {lat*1000:6.1f}ms | ROIs: {len(rois):2d} | Infer: {t_infer_ms:5.1f}ms | Dets: {len(dets):3d}")

        prev_gray = curr_gray
        results['frame_latencies'].append(lat)
        results['all_detections'][fi]  = dets
        results['detection_counts'].append(len(dets))
        fi += 1

    cap.release()
    return results
