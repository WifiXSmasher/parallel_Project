import torch
import time
import cv2
from ultralytics import YOLO
from utils import Detection

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def run_full_frame_baseline(video_path: str, models: list, largest_idx: int, max_frames: int) -> dict:
    """YOLOv8x inference on every frame. Used as accuracy ground truth."""
    cap = cv2.VideoCapture(video_path)
    results = {'method': 'Full-Frame Inference', 'frame_latencies': [],
               'detection_counts': [], 'all_detections': {}}
    fi = 0
    while cap.isOpened() and fi < max_frames:
        ret, frame = cap.read()
        if not ret: break
        t0 = time.perf_counter()
        
        # Inference
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
                    
        lat = time.perf_counter() - t0
        results['frame_latencies'].append(lat)
        results['all_detections'][fi]  = dets
        results['detection_counts'].append(len(dets))
        print(f"  [FullFrame] Frame {fi:3d} | {lat*1000:6.1f}ms | Dets: {len(dets):3d}")
        fi += 1
    cap.release()
    return results
