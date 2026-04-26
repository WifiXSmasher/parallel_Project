import torch
import time
import numpy as np
import cv2
import av
from utils import Detection

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def warp_detection_with_mvs(det: Detection, frame_av, h: int, w: int) -> Detection:
    mvs = frame_av.side_data.get('MOTION_VECTORS')
    if mvs is None: return det
    mv_arr = mvs.to_ndarray()
    if len(mv_arr) == 0: return det

    scale = mv_arr['motion_scale'].astype(float)
    scale[scale == 0] = 1.0
    dx = mv_arr['motion_x'] / scale
    dy = mv_arr['motion_y'] / scale

    src_x = mv_arr['dst_x'] - dx
    src_y = mv_arr['dst_y'] - dy

    x1, y1, x2, y2 = det.bbox_xyxy
    inside = ((src_x >= x1) & (src_x <= x2) & (src_y >= y1) & (src_y <= y2))
    if not np.any(inside): return det

    mean_dx = float(np.mean(dx[inside]))
    mean_dy = float(np.mean(dy[inside]))

    return Detection(
        bbox_xyxy=(
            int(np.clip(x1 + mean_dx, 0, w - 1)),
            int(np.clip(y1 + mean_dy, 0, h - 1)),
            int(np.clip(x2 + mean_dx, 0, w - 1)),
            int(np.clip(y2 + mean_dy, 0, h - 1)),
        ),
        class_id=det.class_id,
        class_name=det.class_name,
        confidence=det.confidence,
        model_id=det.model_id,
    )

def get_frame_motion_intensity(frame_av) -> float:
    """Proxy for AccDecoder's DRL state variable (motion level)."""
    mvs = frame_av.side_data.get('MOTION_VECTORS')
    if mvs is None: return 0.0
    mv_arr = mvs.to_ndarray()
    if len(mv_arr) == 0: return 0.0
    scale = mv_arr['motion_scale'].astype(float)
    scale[scale == 0] = 1.0
    dx = mv_arr['motion_x'] / scale
    dy = mv_arr['motion_y'] / scale
    return float(np.mean(np.hypot(dx, dy)))

def run_accdecoder_approach(video_path: str, models: list, largest_idx: int, max_frames: int) -> dict:
    """
    AccDecoder (MobiCom '22) reimplementation.
    Uses MV-warped bounding boxes to reuse detections.
    UPGRADE: Implements adaptive keyframe selection (mimicking DRL).
    """
    try:
        container = av.open(video_path)
        vs = container.streams.video[0]
        vs.codec_context.options = {'flags2': '+export_mvs'}
    except Exception as e:
        print(f"  [AccDecoder] Cannot open with PyAV: {e}")
        return {'method': 'AccDecoder-based', 'frame_latencies': [], 'all_detections': {}, 'detection_counts': []}

    results = {'method': 'AccDecoder-based', 'frame_latencies': [], 'all_detections': {}, 'detection_counts': []}
    warped_dets = []
    
    # Adaptive Keyframe logic
    frames_since_ref = 0
    max_gop = 30
    motion_spike_threshold = 5.0  # If average MV magnitude > 5.0, trigger reference frame

    for fi, fav in enumerate(container.decode(vs)):
        if fi >= max_frames: break
        t0 = time.perf_counter()
        
        try:
            frame = fav.to_ndarray(format='bgr24')
        except Exception:
            continue
            
        h, w = frame.shape[:2]
        motion_intensity = get_frame_motion_intensity(fav)
        
        # DRL-mimicking logic: Force reference if motion spikes or GOP exceeded
        is_ref = (frames_since_ref >= max_gop) or (frames_since_ref > 0 and motion_intensity > motion_spike_threshold)
        if fi == 0: is_ref = True
        
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
            warped_dets = list(dets)
            frames_since_ref = 0
            lat = time.perf_counter() - t0
            print(f"  [AccDecoder] Frame {fi:3d} [ REF ] | {lat*1000:6.1f}ms | Dets: {len(dets):3d} (Motion: {motion_intensity:.1f})")
        else:
            warped_dets = [warp_detection_with_mvs(d, fav, h, w) for d in warped_dets]
            dets = list(warped_dets)
            frames_since_ref += 1
            lat = time.perf_counter() - t0
            print(f"  [AccDecoder] Frame {fi:3d} [P-frm] | {lat*1000:6.1f}ms | Dets: {len(dets):3d} (Motion: {motion_intensity:.1f})")

        results['frame_latencies'].append(lat)
        results['all_detections'][fi]  = dets
        results['detection_counts'].append(len(dets))

    container.close()
    return results
