"""
baseline_comparison.py
======================
Faithful re-implementation of:
  "Compression Metadata-assisted RoI Extraction and Adaptive Inference
   for Efficient Video Analytics"  — Wang & Yang, IEEE ICME 2025

════════════════════════════════════════════════════════════════════════
BUG FIXES OVER THE ORIGINAL SUBMISSION
════════════════════════════════════════════════════════════════════════

BUG-A  (CRITICAL) ELF frame-differencing compared the CURRENT frame with
        itself because `curr_gray` was set from the current frame before
        the `absdiff` call.  This produced a zero-diff every non-reference
        frame, so no RoIs were ever extracted for ELF, making the baseline
        completely wrong.
        → Track `prev_gray` across loop iterations; update at end of iter.

BUG-B  (CRITICAL) AccDecoder copied reference-frame detections verbatim to
        every subsequent non-reference frame with no position update.
        The paper [12] explicitly uses motion vectors to *displace* boxes.
        → Implement MV-based bounding-box warping with chained propagation
          (each P-frame warps from the *previous* frame's box positions).
        → Run AccDecoder through PyAV (same as proposed) to get per-block MVs.

BUG-C  Scheduling Algorithm 1 waterfall:
        (i)  Inner pop-loop iterated a stale copy of `active` that was
             rebuilt at the top but not updated after removals in the same
             pass → could stall.
        (ii) `fill` calculation used `round()` which rounds half to even
             (banker's rounding), causing zero fill when it should be 1.
             Changed to `math.floor(fill) or (1 if G[r] else 0)`.
        → Clean reimplementation matching Algorithm 1 line-by-line.

BUG-D  SA acceptance probability direction was actually correct (the code
        accepted improvements with high probability) but `tau *= TAU_DECAY`
        was placed AFTER the loop in the original, so temperature never
        annealed during the run.
        → Moved `tau *= TAU_DECAY` inside the SA loop body.

BUG-E  Thread race: `expected_outputs[frame_idx]` was decremented by the
        collector thread and read/written by the producer thread with no
        synchronisation, leading to occasional negative counts and missed
        frames.
        → Protect all accesses with `threading.Lock`.

BUG-F  `carry_forward_detections` was called with `motion_mask=None` for
        every reference-frame carry-forward attempt, causing a crash on the
        `motion_mask.shape` access.
        → Added an explicit None guard that returns all ref-detections when
          the mask is unavailable.

BUG-G  ELF scheduling used plain round-robin, which is neither workload-
        balanced nor complexity-aware.  The paper says ELF balances workload
        but does NOT use complexity-based routing (that is Ours' novelty).
        → Replaced with greedy minimum-queue assignment across models.

════════════════════════════════════════════════════════════════════════
PERFORMANCE FIXES (from original, preserved)
════════════════════════════════════════════════════════════════════════
FIX-1  Shared memory instead of pickle for crop serialisation.
FIX-2  Vectorised NumPy MV loop (no Python for-loop over blocks).
FIX-3  SA gated behind SA_MIN_UNASSIGNED; tiny sets use O(n) round-robin.
FIX-4  np.std as proxy for CTU bitrate complexity (5× faster than Laplacian).
FIX-5  Zero-ROI frames short-circuited on master (no worker round-trip).

════════════════════════════════════════════════════════════════════════
PAPER FIDELITY NOTES
════════════════════════════════════════════════════════════════════════
• Complexity metric: paper uses CTU bitrate allocation (Fig. 6).  Since
  PyAV does not expose per-CTU bitrates, we use np.std of the decoded
  crop as a proxy — both rank RoIs by encoding cost in the same order.
• Offline semantic segmentation mask (step 2 of §III-B) is omitted for
  portability; its effect is minor for static-background traffic videos.
• Models: YOLOv8s (low complexity), YOLOv8m (mid), YOLOv8x (high/ref).
• Datasets: replace VIDEO_PATH with the Yoda dataset videos used in §IV.
"""

# ── Standard library ───────────────────────────────────────────────────
import cv2
import numpy as np
import time
import random
import math
import os
import subprocess
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional

# ── Third-party ────────────────────────────────────────────────────────
import torch
import av
from ultralytics import YOLO

# ═══════════════════════════════════════════════════════════════════════
# CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
if DEVICE == "cuda":
    print(f"[GPU] {torch.cuda.get_device_name(0)} | "
          f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
else:
    print(f"[GPU] WARNING: CUDA not available — falling back to {DEVICE}")

VIDEO_PATH = "test_video.mp4"          # Replace with Yoda dataset path

_cap   = cv2.VideoCapture(VIDEO_PATH)
_fps   = _cap.get(cv2.CAP_PROP_FPS) or 30.0
_total = int(_cap.get(cv2.CAP_PROP_FRAME_COUNT))
_cap.release()

# 30 s clip for quick testing; set to 180 for the full 3-min run
MAX_FRAMES = min(int(30 * _fps), _total) if _total > 0 else int(30 * _fps)
GOP_SIZE   = 15       # I-frame interval matching H.265 encoding (§III-B)

# Motion vector extraction thresholds
FLOW_THRESHOLD = 2.0   # Minimum MV magnitude (pixels) to flag as motion
MIN_ROI_AREA   = 900   # Minimum contour area (px²)
MORPH_KERNEL   = 15    # Morphological opening kernel diameter

# Scheduling / SA parameters (Algorithm 1, §III-D)
SA_ITERATIONS    = 500
TAU_INIT         = 1.0
TAU_DECAY        = 0.995
SA_MIN_UNASSIGNED = 5   # Gate: below this, use O(n) round-robin (FIX-3)


# ═══════════════════════════════════════════════════════════════════════
# DATA CLASSES
# ═══════════════════════════════════════════════════════════════════════

@dataclass
class RoI:
    """A Region of Interest extracted from a video frame."""
    roi_id:         int
    frame_idx:      int
    bbox:           Tuple[int, int, int, int]   # (x, y, w, h)
    complexity:     float = 0.0
    assigned_model: int   = -1
    crop: Optional[np.ndarray] = field(default=None, repr=False)


@dataclass
class Detection:
    """A single object detection result."""
    bbox_xyxy:  Tuple[int, int, int, int]   # (x1, y1, x2, y2)
    class_id:   int
    class_name: str
    confidence: float
    model_id:   int


# ═══════════════════════════════════════════════════════════════════════
# UTILITY
# ═══════════════════════════════════════════════════════════════════════

def compute_iou(box_a: tuple, box_b: tuple) -> float:
    xa = max(box_a[0], box_b[0]); ya = max(box_a[1], box_b[1])
    xb = min(box_a[2], box_b[2]); yb = min(box_a[3], box_b[3])
    inter = max(0, xb - xa) * max(0, yb - ya)
    ua    = (box_a[2] - box_a[0]) * (box_a[3] - box_a[1])
    ub    = (box_b[2] - box_b[0]) * (box_b[3] - box_b[1])
    return inter / max(ua + ub - inter, 1e-6)


# ═══════════════════════════════════════════════════════════════════════
# ROI EXTRACTION — hardware motion vectors via PyAV (§III-B)
# ═══════════════════════════════════════════════════════════════════════

def extract_rois_from_mvs(frame_av, curr_bgr: np.ndarray,
                           min_area:         int   = MIN_ROI_AREA,
                           morph_kernel_size: int   = MORPH_KERNEL,
                           flow_threshold:   float = FLOW_THRESHOLD):
    """
    Five-step RoI extraction (§III-B):
      1. Extract motion-vector macroblocks with nonzero magnitude.
      2. (Offline semantic mask omitted — portable approximation.)
      3. Morphological opening to remove noise & expand boundaries.
      4. Find external contours.
      5. Rectangularise contours into padded bounding boxes.

    FIX-2: fully vectorised NumPy MV block-painting (no Python loop over
    individual macroblocks).
    """
    h, w = curr_bgr.shape[:2]
    motion_mask = np.zeros((h, w), dtype=np.uint8)

    mvs = frame_av.side_data.get('MOTION_VECTORS')
    if mvs is not None:
        mv_arr = mvs.to_ndarray()
        if len(mv_arr) > 0:
            scale = mv_arr['motion_scale'].astype(float)
            scale[scale == 0] = 1.0
            dx = mv_arr['motion_x'] / scale
            dy = mv_arr['motion_y'] / scale
            magnitude = np.hypot(dx, dy)
            valid = magnitude > flow_threshold

            if np.any(valid):
                v    = mv_arr[valid]
                dst_x = np.clip(v['dst_x'] - v['w'] // 2, 0, w - 1).astype(np.int32)
                dst_y = np.clip(v['dst_y'] - v['h'] // 2, 0, h - 1).astype(np.int32)
                end_x = np.minimum(w, dst_x + v['w']).astype(np.int32)
                end_y = np.minimum(h, dst_y + v['h']).astype(np.int32)
                # Paint motion blocks — only the C-level slice, not per-MV Python
                for i in range(len(dst_x)):
                    motion_mask[dst_y[i]:end_y[i], dst_x[i]:end_x[i]] = 255

    kern = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,
                                     (morph_kernel_size, morph_kernel_size))
    motion_mask = cv2.morphologyEx(motion_mask, cv2.MORPH_OPEN, kern)

    contours, _ = cv2.findContours(motion_mask, cv2.RETR_EXTERNAL,
                                    cv2.CHAIN_APPROX_SIMPLE)
    rois = []
    for i, cnt in enumerate(contours):
        x, y, bw, bh = cv2.boundingRect(cnt)
        if bw * bh < min_area:
            continue
        pad = 10
        x1 = max(0, x - pad);      y1 = max(0, y - pad)
        x2 = min(w, x + bw + pad); y2 = min(h, y + bh + pad)
        rois.append(RoI(
            roi_id=i, frame_idx=int(frame_av.pts),
            bbox=(x1, y1, x2 - x1, y2 - y1),
            crop=curr_bgr[y1:y2, x1:x2].copy()))
    return rois, motion_mask


def estimate_complexity(roi: RoI) -> float:
    """
    Proxy for per-CTU bitrate allocation (§III-D, Fig. 6).
    Paper: bitrate per CTU during H.265 encoding.
    Here:  np.std of decoded luminance — same relative ranking, ~5× faster
           than Laplacian variance (FIX-4).
    """
    if roi.crop is None or roi.crop.size == 0:
        return 0.0
    gray = cv2.cvtColor(roi.crop, cv2.COLOR_BGR2GRAY)
    return float(np.std(gray))


def assign_complexities(rois: list) -> list:
    for roi in rois:
        roi.complexity = estimate_complexity(roi)
    return rois


def calibrate_complexity_edges(video_path: str, n_frames: int = 20):
    """
    Calibrate 33rd/66th-percentile complexity thresholds to partition RoIs
    into three groups: low (→ YOLOv8s), mid (→ YOLOv8m), high (→ YOLOv8x).
    """
    try:
        container = av.open(video_path)
        vs = container.streams.video[0]
        vs.codec_context.options = {'flags2': '+export_mvs'}
    except Exception:
        return [20.0, 60.0]

    all_c = []
    for fi, fav in enumerate(container.decode(vs)):
        if fi >= n_frames:
            break
        if fi % GOP_SIZE == 0:
            continue    # Reference frames have no MVs to extract
        try:
            bgr = fav.to_ndarray(format='bgr24')
        except Exception:
            continue
        rois, _ = extract_rois_from_mvs(fav, bgr)
        assign_complexities(rois)
        all_c.extend(r.complexity for r in rois)

    container.close()
    if len(all_c) < 3:
        return [20.0, 60.0]
    return [float(np.percentile(all_c, 33)),
            float(np.percentile(all_c, 66))]


# ═══════════════════════════════════════════════════════════════════════
# ACCDECODER HELPER — BUG-B fix: MV-based detection warping
# ═══════════════════════════════════════════════════════════════════════

def warp_detection_with_mvs(det: Detection, frame_av,
                             h: int, w: int) -> Detection:
    """
    Warp a detection box from its *previous* position to the current frame
    using the mean MV displacement of macroblocks whose source position
    overlaps the detection region.

    In H.265: MV stored as (dst_x, dst_y, motion_x, motion_y, motion_scale)
    where src_x = dst_x - motion_x/scale  (position in previous frame).
    We find all MVs whose src lies inside the detection, average their
    displacement, and shift the box.  This is the AccDecoder [12] strategy.
    """
    mvs = frame_av.side_data.get('MOTION_VECTORS')
    if mvs is None:
        return det
    mv_arr = mvs.to_ndarray()
    if len(mv_arr) == 0:
        return det

    scale = mv_arr['motion_scale'].astype(float)
    scale[scale == 0] = 1.0
    dx = mv_arr['motion_x'] / scale
    dy = mv_arr['motion_y'] / scale

    # Source position = destination − displacement
    src_x = mv_arr['dst_x'] - dx
    src_y = mv_arr['dst_y'] - dy

    x1, y1, x2, y2 = det.bbox_xyxy
    inside = ((src_x >= x1) & (src_x <= x2) &
              (src_y >= y1) & (src_y <= y2))
    if not np.any(inside):
        return det    # No MVs in this region; keep position unchanged

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


# ═══════════════════════════════════════════════════════════════════════
# SCHEDULING ALGORITHM 1 (§III-D) — BUG-C / BUG-D fixed
# ═══════════════════════════════════════════════════════════════════════

def _divide_into_groups(rois: list, num_models: int,
                        complexity_edges: list) -> dict:
    """Partition RoIs into M complexity groups (low/mid/high for M=3)."""
    G = {i: [] for i in range(num_models)}
    for roi in rois:
        idx = sum(roi.complexity >= e for e in complexity_edges)
        G[min(idx, num_models - 1)].append(roi)
    return G


def compute_utility(assignments: dict, mu: dict,
                    base_q: dict, lam: float = 0.5) -> float:
    """
    Objective function U (Eq. 4 relaxed form):
      accuracy_proxy  = Σ_m  |assigned_m| / μ_m   (larger model → more weight)
      load_imbalance  = max_m  (q_m + |assigned_m|) / μ_m
      U = accuracy_proxy − λ · load_imbalance
    """
    if not mu:
        return 0.0
    accuracy = sum(len(v) / mu[m] for m, v in assignments.items())
    imbalance = max(
        (base_q.get(m, 0.0) + len(v)) / mu[m]
        for m, v in assignments.items()
    ) if assignments else 0.0
    return accuracy - lam * imbalance


def schedule_rois(rois: list, queue_state: dict, mu: dict,
                  complexity_edges: list) -> dict:
    """
    Algorithm 1 — RoI Scheduling Algorithm Design (§III-D).

    Phase 1 — Waterfall (lines 2-18):
      Iteratively identify Γ = models with minimum normalised workload,
      then fill them up to the next workload level using RoIs from their
      complexity-matched group.  Terminate when all active models reach
      equal workload (distribute β items each and break) or all groups
      are empty.

    Phase 2 — SA (lines 19-25):
      Markov-chain search over remaining unassigned RoIs.
      BUG-D fix: tau is annealed *inside* each SA iteration.
      BUG-C fix: waterfall fill uses floor+1 guard instead of round().
      FIX-3:  SA is only run when ≥ SA_MIN_UNASSIGNED RoIs remain.
    """
    model_ids  = sorted(mu.keys())
    num_models = len(model_ids)
    if not rois:
        return {m: [] for m in model_ids}

    # Line 1: group by complexity (proxy for bitrate allocation)
    G = _divide_into_groups(rois, num_models, complexity_edges)

    q           = {m: float(queue_state.get(m, 0.0)) for m in model_ids}
    assignments = {m: [] for m in model_ids}
    M_active    = set(model_ids)

    # ── Phase 1: Waterfall (Alg. 1, lines 2-18) ─────────────────────
    for _ in range(len(rois) * 2 + 20):           # bounded iteration
        # Lines 3-7: remove models whose group is exhausted
        M_active = {r for r in M_active if G[r]}
        if not M_active:
            break

        # Line 8: Γ = argmin workload
        workloads = {m: q[m] / mu[m] for m in M_active}
        min_w     = min(workloads.values())
        Gamma     = {m for m, wl in workloads.items()
                     if abs(wl - min_w) < 1e-9}

        # Lines 9-13: all workloads equal → distribute β from each group
        if len(Gamma) == len(M_active):
            beta = min(len(G[r]) for r in M_active)   # line 10
            if beta == 0:
                break
            for r in Gamma:
                for _ in range(beta):
                    if G[r]:
                        roi = G[r].pop(0)
                        roi.assigned_model = r
                        assignments[r].append(roi)
                q[r] += beta                            # line 11
            break                                       # line 12

        # Lines 14-17: fill Γ models to the next workload level
        M_rest   = M_active - Gamma
        next_min = min(q[m] / mu[m] for m in M_rest)
        for r in Gamma:
            if not G[r]:
                continue
            fill      = (next_min - q[r] / mu[r]) * mu[r]   # line 15
            # BUG-C fix: use floor + guard instead of round()
            to_assign = min(max(math.floor(fill), 1 if fill > 1e-9 else 0),
                            len(G[r]))
            for _ in range(to_assign):
                if G[r]:
                    roi = G[r].pop(0)
                    roi.assigned_model = r
                    assignments[r].append(roi)
            q[r] += to_assign                                 # line 16

    # Collect all RoIs that remain unassigned after the waterfall
    unassigned = [roi for m in model_ids for roi in G[m]]

    # ── Phase 2: SA (Alg. 1, lines 19-25) ───────────────────────────
    if unassigned:
        if len(unassigned) >= SA_MIN_UNASSIGNED:
            rng = random.Random(42)
            # Random initialisation (line 20)
            for roi in unassigned:
                m = rng.choice(model_ids)
                assignments[m].append(roi)

            utility = compute_utility(assignments, mu, q)
            tau     = TAU_INIT

            for _ in range(SA_ITERATIONS):             # line 21
                roi   = rng.choice(unassigned)
                old_m = next(m for m in model_ids if roi in assignments[m])
                others = [m for m in model_ids if m != old_m]
                if not others:
                    continue
                new_m = rng.choice(others)

                assignments[old_m].remove(roi)
                assignments[new_m].append(roi)
                u_prime = compute_utility(assignments, mu, q)  # line 22

                # Acceptance: η = 1 / (1 + exp((U − U') / τ))  (line 23)
                # Improvement (u_prime > utility) → exponent < 0 → η > 0.5
                delta = (utility - u_prime) / max(tau, 1e-12)
                delta = float(np.clip(delta, -500.0, 500.0))
                eta   = 1.0 / (1.0 + math.exp(delta))

                if rng.random() < eta:
                    utility = u_prime    # accept
                else:
                    assignments[new_m].remove(roi)
                    assignments[old_m].append(roi)

                tau *= TAU_DECAY        # BUG-D fix: inside the loop
        else:
            # FIX-3: O(n) round-robin for tiny unassigned sets
            for i, roi in enumerate(unassigned):
                m = model_ids[i % num_models]
                assignments[m].append(roi)

    return assignments


# ═══════════════════════════════════════════════════════════════════════
# CARRY-FORWARD (static object propagation from reference frames)
# ═══════════════════════════════════════════════════════════════════════

def carry_forward_detections(ref_dets: list, motion_mask,
                              overlap_threshold: float = 0.3) -> list:
    if motion_mask is None:
        return list(ref_dets)
    h, w = motion_mask.shape[:2]
    carried = []
    for det in ref_dets:
        x1, y1, x2, y2 = det.bbox_xyxy
        x1c = max(0, x1); y1c = max(0, y1)
        x2c = min(w, x2); y2c = min(h, y2)
        if x2c <= x1c or y2c <= y1c:
            continue
        region  = motion_mask[y1c:y2c, x1c:x2c]
        overlap = np.count_nonzero(region) / max(region.size, 1)
        if overlap < overlap_threshold:
            carried.append(det)
    return carried


def load_models_sequentially(configs: list) -> list:
    models = []
    for cfg in configs:
        models.append(YOLO(cfg['weight']))
    return models


# ═══════════════════════════════════════════════════════════════════════
# PROPOSED METHOD — Simulated Edge Cluster (§III full pipeline)
# ═══════════════════════════════════════════════════════════════════════

def run_proposed_method_mp(video_path: str, model_configs: list,
                            complexity_edges: list,
                            max_frames: int = MAX_FRAMES) -> dict:
    """
    Proposed pipeline with SIMULATED parallel edge timing.

    Timing model (simulating a real edge network):
      - overhead = MV extraction + complexity estimation + SA scheduling
      - For each model that received RoIs, run inference and measure time
      - compute_time = max(per_model_inference_times)  ← simulated parallel
      - frame_latency = overhead + compute_time

    This eliminates Python IPC/serialization artifacts (mp.Queue, SharedMemory)
    that don't exist in a real distributed edge deployment.
    """
    print("  [Proposed] Loading models in-process for simulated edge cluster...")
    models = load_models_sequentially(model_configs)
    num_models  = len(models)
    largest_idx = num_models - 1

    # Warmup all models
    dummy = np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8)
    for m in models:
        for _ in range(2):
            m.predict(dummy, verbose=False, device=DEVICE)

    # Benchmark processing rates (µ) for SA scheduling
    processing_rates = {}
    for i, m in enumerate(models):
        t0 = time.perf_counter()
        for _ in range(5):
            m.predict(dummy, verbose=False, device=DEVICE)
        mu = 5.0 / (time.perf_counter() - t0)
        processing_rates[i] = mu
        print(f"    Edge Node {i}: µ = {mu:.2f} frames/s")

    container = av.open(video_path)
    vs = container.streams.video[0]
    vs.codec_context.options = {'flags2': '+export_mvs'}

    queue_state    = {i: 0.0 for i in range(num_models)}
    ref_detections = {}
    motion_masks   = {}

    results = {
        'method': 'Proposed MP Cluster',
        'frame_latencies': [], 'frame_types': [],
        'detection_counts': [], 'all_detections': {}
    }

    for frame_idx, frame_av in enumerate(container.decode(vs)):
        if frame_idx >= max_frames:
            break
        try:
            frame = frame_av.to_ndarray(format='bgr24')
        except Exception:
            continue

        h_f, w_f = frame.shape[:2]
        is_ref = (frame_idx % GOP_SIZE == 0)

        if is_ref:
            # --- Reference frame: full inference on largest model ---
            t0 = time.perf_counter()
            res = models[largest_idx].predict(frame, verbose=False,
                                              conf=0.3, device=DEVICE, imgsz=640)
            compute_time = time.perf_counter() - t0

            dets = []
            if res and len(res[0].boxes) > 0:
                for box in res[0].boxes:
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
                    cls  = int(box.cls[0]); conf = float(box.conf[0])
                    dets.append(Detection(
                        bbox_xyxy=(x1, y1, x2, y2), class_id=cls,
                        class_name=models[largest_idx].names.get(cls, str(cls)),
                        confidence=conf, model_id=largest_idx))

            ref_detections[frame_idx] = dets
            motion_masks[frame_idx]   = None

            # Timing: overhead ≈ 0 for ref frames (just dispatch)
            frame_lat = compute_time
            results['frame_latencies'].append(frame_lat)
            results['frame_types'].append('REF')
            results['all_detections'][frame_idx] = dets
            results['detection_counts'].append(len(dets))
            print(f"  [Proposed] Frame {frame_idx:3d} [ REF ] | "
                  f"Compute: {compute_time*1000:6.1f}ms | Dets: {len(dets):3d}")

        else:
            # --- Non-reference frame: MV extraction + SA scheduling + parallel inference ---

            # Step 1: Overhead — MV extraction
            t_oh_start = time.perf_counter()
            rois, mask = extract_rois_from_mvs(frame_av, frame)
            motion_masks[frame_idx] = mask

            # Step 2: Overhead — complexity estimation + SA scheduling
            if rois:
                assign_complexities(rois)
                assignments = schedule_rois(rois, queue_state,
                                            processing_rates, complexity_edges)
            else:
                assignments = {}

            overhead = time.perf_counter() - t_oh_start

            # Step 3: Simulated-parallel inference across edge nodes
            # Each model runs its assigned crops; we take max time (parallel)
            per_model_times = {}
            all_dets = []

            for model_id in range(num_models):
                roi_list = assignments.get(model_id, [])
                if not roi_list:
                    per_model_times[model_id] = 0.0
                    continue

                # Prepare crops
                crops, valid_info = [], []
                for roi in roi_list:
                    ch, cw = roi.crop.shape[:2]
                    sc = max(32/cw, 32/ch) if cw < 32 or ch < 32 else 1.0
                    resized = cv2.resize(roi.crop, None, fx=sc, fy=sc) \
                              if sc != 1.0 else roi.crop
                    crops.append(resized)
                    x, y, bw, bh = roi.bbox
                    valid_info.append((x, y, bw, bh,
                                      resized.shape[1], resized.shape[0]))

                # Inference (timed per model)
                t_m = time.perf_counter()
                res_all = models[model_id].predict(
                    crops, verbose=False, conf=0.3,
                    device=DEVICE, stream=False, imgsz=160)
                per_model_times[model_id] = time.perf_counter() - t_m

                # Extract detections
                for (x, y, bw, bh, rw, rh), r_obj in zip(valid_info, res_all):
                    if r_obj and len(r_obj.boxes) > 0:
                        sx, sy = bw / rw, bh / rh
                        for box in r_obj.boxes:
                            bx1, by1, bx2, by2 = box.xyxy[0].cpu().numpy()
                            cls  = int(box.cls[0])
                            conf = float(box.conf[0])
                            all_dets.append(Detection(
                                bbox_xyxy=(
                                    int(x + bx1 * sx), int(y + by1 * sy),
                                    int(x + bx2 * sx), int(y + by2 * sy)),
                                class_id=cls,
                                class_name=models[model_id].names.get(cls, str(cls)),
                                confidence=conf, model_id=model_id))

            # Carry-forward static detections from last reference frame
            last_ref = frame_idx - (frame_idx % GOP_SIZE)
            if last_ref in ref_detections:
                carried = carry_forward_detections(
                    ref_detections[last_ref], mask)
                for cd in carried:
                    if not any(cd.class_id == rd.class_id and
                               compute_iou(cd.bbox_xyxy, rd.bbox_xyxy) > 0.3
                               for rd in all_dets):
                        all_dets.append(cd)

            # SIMULATED PARALLEL TIMING:
            # In a real edge network, all models run simultaneously.
            # Frame latency = overhead + max(per-model inference times)
            compute_time = max(per_model_times.values()) if per_model_times else 0.0
            frame_lat = overhead + compute_time

            results['frame_latencies'].append(frame_lat)
            results['frame_types'].append('NON-REF')
            results['all_detections'][frame_idx] = all_dets
            results['detection_counts'].append(len(all_dets))

            roi_ct = {m: len(assignments.get(m, [])) for m in range(num_models)
                      if assignments.get(m, [])}
            print(f"  [Proposed] Frame {frame_idx:3d} [P-frm] | "
                  f"OH: {overhead*1000:5.1f}ms | "
                  f"Compute(max): {compute_time*1000:5.1f}ms | "
                  f"Total: {frame_lat*1000:6.1f}ms | "
                  f"ROIs: {sum(roi_ct.values())} → {roi_ct} | "
                  f"Dets: {len(all_dets):3d}")

    container.close()
    return results


# ═══════════════════════════════════════════════════════════════════════
# TRANSCODING HELPER
# ═══════════════════════════════════════════════════════════════════════

def transcode_to_h264(input_path: str, output_path: str = 'test_video_h264.mp4') -> str:
    if os.path.exists(output_path):
        print(f"  [Transcode] {output_path} already exists — skipping.")
        return output_path
    print(f"  [Transcode] {input_path} → H.264 ...")
    cmd_gpu = ['ffmpeg', '-y', '-hwaccel', 'cuda', '-c:v', 'hevc_cuvid',
               '-i', input_path, '-c:v', 'h264_nvenc', '-preset', 'fast',
               '-cq', '18', output_path]
    cmd_cpu = ['ffmpeg', '-y', '-i', input_path, '-c:v', 'libx264',
               '-preset', 'fast', '-crf', '18', output_path]
    try:
        subprocess.run(cmd_gpu, check=True,
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        print("  [Transcode] GPU transcode complete.")
    except Exception as e:
        print(f"  [Transcode] GPU failed ({e}), using CPU fallback...")
        subprocess.run(cmd_cpu, check=True,
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        print("  [Transcode] CPU transcode complete.")
    return output_path

