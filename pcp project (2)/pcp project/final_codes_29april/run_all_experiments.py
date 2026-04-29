import os
import torch
from ultralytics import YOLO

from utils import evaluate_method, plot_comparison
from proposed_approach import transcode_to_h264, calibrate_complexity_edges, run_proposed_method_mp
from full_frame_ground_truth import run_full_frame_baseline
from elf_approach import run_elf_approach
from accdecoder_approach import run_accdecoder_approach
from speculative_approach import run_speculative_approach

def load_models_sequentially(configs: list) -> list:
    models = []
    print("\n[Main] Loading models sequentially...")
    for cfg in configs:
        try:
            models.append(YOLO(cfg['weight']))
            print(f"  ✓ {cfg['weight']}")
        except Exception as e:
            print(f"  ✗ {cfg['weight']}: {e}")
    return models

def main():
    VIDEO_PATH = "test_video.mp4"
    max_frames = 300

    model_configs = [
        {'name': 'YOLOv8s', 'weight': '../../yolov8s.pt'},
        {'name': 'YOLOv8m', 'weight': '../../yolov8m.pt'},
        {'name': 'YOLOv8x', 'weight': '../../yolov8x.pt'},
    ]

    if not os.path.exists('../../yolov8s.pt'):
        model_configs = [
            {'name': 'YOLOv8s', 'weight': 'yolov8s.pt'},
            {'name': 'YOLOv8m', 'weight': 'yolov8m.pt'},
            {'name': 'YOLOv8x', 'weight': 'yolov8x.pt'}
        ]

    # 1. Prep
    h264_path = transcode_to_h264(VIDEO_PATH)
    edges = calibrate_complexity_edges(h264_path)
    print(f"\n  Complexity thresholds: {edges[0]:.2f} / {edges[1]:.2f}")

    # 2. Load models for sequential baselines
    yolo_models = load_models_sequentially(model_configs)
    largest_idx = len(yolo_models) - 1

    # 3. Ground Truth
    print('\n═══ Ground Truth: Full-Frame Inference ═══')
    gt_res = run_full_frame_baseline(VIDEO_PATH, yolo_models, largest_idx, max_frames=max_frames)

    # 4. ELF
    print('\n═══ Baseline: ELF-based ═══')
    elf_res = run_elf_approach(VIDEO_PATH, yolo_models, max_frames=max_frames)

    # 5. AccDecoder
    print('\n═══ Baseline: AccDecoder-based ═══')
    acc_res = run_accdecoder_approach(h264_path, yolo_models, largest_idx, max_frames=max_frames)

    del yolo_models
    if torch.cuda.is_available(): torch.cuda.empty_cache()

    # 6. Proposed (paper's approach)
    print('\n═══ Proposed: Simulated Edge Cluster ═══')
    prop_res = run_proposed_method_mp(h264_path, model_configs, edges, max_frames=max_frames)

    if torch.cuda.is_available(): torch.cuda.empty_cache()

    # 7. Speculative (your novelty) — returns two result dicts
    print('\n═══ Speculative: Ref-Frame Speculation ═══')
    spec_ideal, spec_rb = run_speculative_approach(h264_path, model_configs, edges, max_frames=max_frames)

    # 8. Evaluate and Plot (6 bars)
    all_evals = []
    for res in [gt_res, prop_res, elf_res, acc_res, spec_ideal, spec_rb]:
        all_evals.append(evaluate_method(res, gt_res))

    plot_comparison(all_evals, save_path="comparison_results_final.png")

if __name__ == '__main__':
    main()
