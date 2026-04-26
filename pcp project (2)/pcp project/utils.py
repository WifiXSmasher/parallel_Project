from dataclasses import dataclass, field
from typing import Tuple, Optional
import numpy as np
import matplotlib.pyplot as plt

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

def compute_iou(box_a: tuple, box_b: tuple) -> float:
    xa = max(box_a[0], box_b[0]); ya = max(box_a[1], box_b[1])
    xb = min(box_a[2], box_b[2]); yb = min(box_a[3], box_b[3])
    inter = max(0, xb - xa) * max(0, yb - ya)
    ua    = (box_a[2] - box_a[0]) * (box_a[3] - box_a[1])
    ub    = (box_b[2] - box_b[0]) * (box_b[3] - box_b[1])
    return inter / max(ua + ub - inter, 1e-6)

def evaluate_method(res: dict, gt: dict) -> dict:
    f1s    = []
    common = set(gt['all_detections'].keys()) & set(res['all_detections'].keys())
    for f_idx in sorted(common):
        preds    = res['all_detections'][f_idx]
        gts_list = gt['all_detections'][f_idx]
        if not gts_list and not preds:
            f1s.append(1.0); continue
        if not gts_list or not preds:
            f1s.append(0.0); continue

        tp, matched_gt, matched_ious = 0, set(), []
        for pred in preds:
            best_iou, best_j = 0.0, -1
            for j, g in enumerate(gts_list):
                if j in matched_gt or pred.class_id != g.class_id:
                    continue
                iou = compute_iou(pred.bbox_xyxy, g.bbox_xyxy)
                if iou > best_iou:
                    best_iou, best_j = iou, j
            if best_iou >= 0.5 and best_j >= 0:
                tp += 1; matched_gt.add(best_j); matched_ious.append(best_iou)

        prec = tp / max(len(preds), 1)
        rec  = tp / max(len(gts_list), 1)
        f1   = 2*prec*rec / (prec+rec) if (prec+rec) > 0 else 0.0
        cr   = min(len(preds), len(gts_list)) / max(len(preds), len(gts_list))
        miou = float(np.mean(matched_ious)) if matched_ious else 0.0
        f1s.append(0.6*f1 + 0.2*cr + 0.2*miou)

    if not f1s:
        f1s = [0.0]
    return {
        'method':         res['method'],
        'avg_f1':         float(np.mean(f1s)),
        'f1_scores':      f1s,
        'avg_latency_ms': float(np.mean(res['frame_latencies'])) * 1000,
        'total_dets':     sum(res['detection_counts']),
    }

def plot_comparison(evals: list, save_path: str = 'comparison_results.png'):
    name_map = {
        'Proposed MP Cluster':   'Ours',
        'Full-Frame Inference':  'Full-Frame (GT)',
        'ELF-based':             'ELF-based',
        'AccDecoder-based':      'AccDecoder-based',
        'Speculative (ideal)':   'Speculative',
        'Speculative (+rollback)': 'Spec +rollback',
    }
    colors = {
        'Ours':                  'red',
        'Full-Frame (GT)':       'dodgerblue',
        'ELF-based':             'green',
        'AccDecoder-based':      'orange',
        'Speculative':           'magenta',
        'Spec +rollback':        'purple',
    }
    ls = {
        'Ours': '-', 'Full-Frame (GT)': ':', 'ELF-based': '-.',
        'AccDecoder-based': '--', 'Speculative': '-', 'Spec +rollback': '--',
    }

    fig, axes = plt.subplots(1, 2, figsize=(18, 6))
    fig.suptitle('Real Measured Results — IEEE ICME 2025 Pipeline',
                 fontsize=16, fontweight='bold')

    ax = axes[0]
    for ev in evals:
        name = name_map.get(ev['method'], ev['method'])
        sf1  = np.sort(ev.get('f1_scores', [0.0]))
        cdf  = np.arange(1, len(sf1)+1) / len(sf1)
        ax.plot(sf1, cdf, label=name, color=colors.get(name, 'gray'),
                linestyle=ls.get(name, '-'), linewidth=2.5)
    ax.set_xlabel('Analytic accuracy (F1)', fontsize=14, fontweight='bold')
    ax.set_ylabel('CDF', fontsize=14, fontweight='bold')
    ax.set_xlim(0.0, 1.05); ax.set_ylim(0.0, 1.05)
    ax.legend(loc='lower right', fontsize=9)
    ax.grid(True, linestyle='-', alpha=0.5)
    ax.set_title('CDF of Analytic Accuracy', fontsize=14)

    ax2  = axes[1]
    disp = [(name_map.get(ev['method'], ev['method']),
             ev.get('avg_latency_ms', 0.0)) for ev in evals]
    names   = [d[0] for d in disp]
    lats    = [d[1] for d in disp]
    bcolors = [colors.get(n, 'gray') for n in names]
    bars    = ax2.bar(names, lats, color=bcolors, alpha=0.9,
                      edgecolor='black', linewidth=1.2, width=0.55)
    for bar in bars:
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 2,
                 f'{bar.get_height():.1f}ms',
                 ha='center', va='bottom', fontsize=9, fontweight='bold')
    ax2.set_ylabel('Average Inference Latency (ms)', fontsize=14)
    ax2.set_title('Average Inference Latency', fontsize=14)
    ax2.tick_params(axis='x', rotation=15, labelsize=8)
    ax2.grid(True, axis='y', linestyle='--', alpha=0.5)

    plt.tight_layout(); plt.subplots_adjust(top=0.88)
    plt.savefig(save_path, dpi=200, bbox_inches='tight')

    print('\n' + '═'*70)
    print(f"  {'Method':<30} {'Avg Lat (ms)':>14} {'Avg F1':>9}")
    print('─'*70)
    for ev in evals:
        n = name_map.get(ev['method'], ev['method'])
        print(f"  {n:<30} {ev.get('avg_latency_ms',0):>14.1f}  {ev.get('avg_f1',0):>8.4f}")
    print('═'*70)
    print(f'\n✓ Figure saved → {save_path}')
