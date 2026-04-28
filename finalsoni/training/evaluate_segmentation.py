"""
Mask-based evaluation and tuning of the segmentation pipeline.

Two modes:

  1. evaluate(seg, splits, split="test")
       Run a fixed segmenter on a split, report per-image and aggregate
       Dice / IoU / bbox-IoU.

  2. tune_classical(splits, grid, max_train=N)
       Grid-search the ClassicalConfig parameters on the train+val splits
       (lesion-bearing samples only - no masks on `normal` class), then
       return the best config and held-out test metrics.

We deliberately tune on train+val and report on test. The grid is small
(~50-200 combinations) because the search space is well-understood.
"""

from __future__ import annotations

import itertools
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import cv2
import numpy as np

from pipeline.preprocessing import PreprocessConfig, preprocess
from pipeline.segmentation import (ClassicalConfig, ClassicalSegmenter,
                                   SegmentationOutput,
                                   bbox_from_mask, bbox_iou, dice, iou)
from .dataset import BUSISample, Splits, load_pair


# ---------------------------------------------------------------------------
# Per-image evaluation
# ---------------------------------------------------------------------------

def _resize_mask(mask: np.ndarray, target_size: int) -> np.ndarray:
    return cv2.resize(mask, (target_size, target_size),
                      interpolation=cv2.INTER_NEAREST)


def evaluate(segmenter,
             samples: List[BUSISample],
             pre_cfg: PreprocessConfig,
             include_normal: bool = False,
             progress: Optional[Callable[[int, int], None]] = None,
             ) -> Dict:
    """
    Returns aggregate + per-image metrics. By default skips `normal` class
    (no lesion mask makes Dice trivially 1 if pred is empty, 0 otherwise).
    """
    per_img: List[Dict] = []
    dices: List[float] = []
    ious: List[float] = []
    biou: List[float] = []
    detected = 0          # predicted any pixels
    correctly_empty = 0    # normal sample, pred empty
    n_lesion = 0           # has lesion ground truth
    n_normal = 0

    n = len(samples)
    for i, s in enumerate(samples):
        img, gt_mask = load_pair(s)
        gray = preprocess(img, pre_cfg)
        gt_resized = _resize_mask(gt_mask, pre_cfg.target_size)

        out: SegmentationOutput = segmenter.predict(gray)
        pred = out.mask

        has_gt = bool((gt_resized > 0).sum() > 0)
        has_pred = bool((pred > 0).sum() > 0)

        if not has_gt:
            n_normal += 1
            if not has_pred:
                correctly_empty += 1
            per_img.append({
                "stem": s.stem, "klass": s.klass,
                "has_gt": False, "has_pred": has_pred,
                "dice": None, "iou": None, "bbox_iou": None,
            })
            if include_normal and has_pred:
                # treat as 0-IoU; lets us include false-positives in averages
                dices.append(0.0); ious.append(0.0); biou.append(0.0)
            if progress: progress(i + 1, n)
            continue

        n_lesion += 1
        if has_pred:
            detected += 1

        d = dice(pred, gt_resized)
        u = iou(pred, gt_resized)
        gt_bbox = bbox_from_mask(gt_resized)
        pred_bbox = bbox_from_mask(pred)
        bi = bbox_iou(gt_bbox, pred_bbox) if (gt_bbox and pred_bbox) else 0.0

        dices.append(d); ious.append(u); biou.append(bi)
        per_img.append({
            "stem": s.stem, "klass": s.klass,
            "has_gt": True, "has_pred": has_pred,
            "dice": float(d), "iou": float(u), "bbox_iou": float(bi),
        })
        if progress: progress(i + 1, n)

    agg = {
        "n_samples": n,
        "n_lesion": n_lesion,
        "n_normal": n_normal,
        "detection_rate_lesion": float(detected / max(n_lesion, 1)),
        "specificity_normal":     float(correctly_empty / max(n_normal, 1)) if n_normal else None,
        "mean_dice":     float(np.mean(dices)) if dices else 0.0,
        "median_dice":   float(np.median(dices)) if dices else 0.0,
        "mean_iou":      float(np.mean(ious))  if ious  else 0.0,
        "median_iou":    float(np.median(ious)) if ious else 0.0,
        "mean_bbox_iou": float(np.mean(biou))  if biou  else 0.0,
    }
    return {"aggregate": agg, "per_image": per_img}


# ---------------------------------------------------------------------------
# Grid search for ClassicalConfig
# ---------------------------------------------------------------------------

DEFAULT_GRID: Dict[str, List] = {
    "sal_percentile":      [88, 90, 92, 94, 96],
    "w_dark":              [0.55, 0.65, 0.75],
    "w_log":               [0.20, 0.30],
    "w_edge":              [0.05, 0.15],
    "open_ksize":          [3, 5],
    "close_ksize":         [5, 7],
    "min_area_frac":       [0.003, 0.005, 0.01],
    "max_area_frac":       [0.55],
    "min_solidity":        [0.30, 0.40],
    "max_aspect_ratio":    [4.0, 5.0],
    "min_contrast_out_in": [0.03, 0.06],
}


def _iter_grid(grid: Dict[str, List]) -> List[Dict]:
    keys = list(grid.keys())
    out: List[Dict] = []
    for vals in itertools.product(*[grid[k] for k in keys]):
        out.append(dict(zip(keys, vals)))
    return out


def tune_classical(splits: Splits,
                   pre_cfg: PreprocessConfig,
                   grid: Optional[Dict[str, List]] = None,
                   max_train: Optional[int] = None,
                   max_combos: Optional[int] = None,
                   verbose: bool = True) -> Tuple[ClassicalConfig, Dict]:
    """
    Sweep over `grid` on (train + val), pick the config with the highest
    mean Dice on lesion-bearing samples. Returns (best_cfg, report).
    """
    grid = grid or DEFAULT_GRID
    combos = _iter_grid(grid)
    if max_combos is not None and len(combos) > max_combos:
        # subsample (deterministic) to keep runtime bounded
        rng = np.random.default_rng(42)
        idx = rng.choice(len(combos), size=max_combos, replace=False)
        combos = [combos[int(i)] for i in idx]

    # Tune on lesion-bearing samples in train+val.
    tune_samples = [s for s in (splits.train + splits.val)
                    if s.klass in ("benign", "malignant")]
    if max_train is not None:
        tune_samples = tune_samples[:max_train]

    if verbose:
        print(f"[tune] {len(combos)} configs against {len(tune_samples)} lesion samples")

    best_score = -1.0
    best_cfg: Optional[ClassicalConfig] = None
    leaderboard: List[Dict] = []

    for i, params in enumerate(combos):
        cfg = ClassicalConfig(**params)
        seg = ClassicalSegmenter(cfg)
        rep = evaluate(seg, tune_samples, pre_cfg, include_normal=False)
        score = rep["aggregate"]["mean_dice"]
        leaderboard.append({"params": params,
                            "mean_dice": score,
                            "median_dice": rep["aggregate"]["median_dice"],
                            "mean_iou": rep["aggregate"]["mean_iou"]})
        if verbose and (i % 10 == 0 or i == len(combos) - 1):
            print(f"  [{i+1:>4}/{len(combos)}] mean_dice={score:.3f} "
                  f"best={best_score:.3f}")
        if score > best_score:
            best_score = score
            best_cfg = cfg

    leaderboard.sort(key=lambda x: -x["mean_dice"])
    report = {
        "best_mean_dice": best_score,
        "best_params": asdict(best_cfg) if best_cfg else None,
        "n_combos_tried": len(combos),
        "top10": leaderboard[:10],
    }
    return best_cfg, report
