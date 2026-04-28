"""
Auto-evaluate the full SoniSight pipeline over every BUSI image.

For each image:
  - run the same pipeline the API uses (preprocess + segment + features + classify)
  - record the true label (from the BUSI folder name)
  - record whether the system was correct
  - record every feature value, segmentation stat, and classification probability

The output JSON is the input to a follow-up analysis that finds the actual
feature thresholds separating system errors from system successes.

Usage:
    python -m scripts.eval_dataset \
        --busi /path/to/Dataset_BUSI_with_GT \
        --out  artifacts/eval_dataset.json

Optional:
    --limit_per_class N    # cap to N samples per class for a quick run
    --binary               # treat benign+malignant as one "suspicious" class (default)
    --three_class          # keep normal/benign/malignant separate
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Dict, List

import cv2
import numpy as np

from pipeline.pipeline import PipelineConfig, SoniSightPipeline
from pipeline.preprocessing import PreprocessConfig
from training.dataset import discover_busi


def _build_pipeline(artifacts_dir: Path, seg_backend: str = "classical") -> SoniSightPipeline:
    """Build the same pipeline the API serves, using the saved artifacts."""
    seg_cfg = artifacts_dir / "seg_config.json"
    classifier = artifacts_dir / "classifier.joblib"
    unet_ckpt = artifacts_dir / "unet.pt"
    cfg = PipelineConfig(
        preprocess=PreprocessConfig(target_size=256),
        seg_backend=seg_backend,
        seg_config_path=str(seg_cfg) if seg_cfg.exists() else None,
        unet_ckpt_path=str(unet_ckpt) if unet_ckpt.exists() else None,
        classifier_path=str(classifier) if classifier.exists() else None,
    )
    return SoniSightPipeline(cfg)


def _decide_correctness(true_class: str, predicted_label: str | None,
                        binary: bool) -> Dict:
    """
    Compare ground truth to prediction and label which bucket this image
    falls into:
      TP - true positive (lesion present, system flagged suspicious)
      TN - true negative (no lesion, system said normal)
      FP - false positive (no lesion, system flagged suspicious)
      FN - false negative (lesion present, system said normal)
    """
    if binary:
        true_is_lesion = true_class in ("benign", "malignant")
        pred_is_lesion = (predicted_label == "suspicious")
    else:
        true_is_lesion = true_class != "normal"
        pred_is_lesion = (predicted_label in ("benign", "malignant", "suspicious"))

    if true_is_lesion and pred_is_lesion:
        bucket = "TP"
    elif (not true_is_lesion) and (not pred_is_lesion):
        bucket = "TN"
    elif (not true_is_lesion) and pred_is_lesion:
        bucket = "FP"
    else:
        bucket = "FN"
    return {
        "true_class": true_class,
        "true_is_lesion": bool(true_is_lesion),
        "predicted_label": predicted_label,
        "pred_is_lesion": bool(pred_is_lesion),
        "bucket": bucket,
        "correct": bucket in ("TP", "TN"),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--busi", required=True, help="path to Dataset_BUSI_with_GT")
    ap.add_argument("--out", default="artifacts/eval_dataset.json")
    ap.add_argument("--artifacts", default="artifacts",
                    help="dir containing seg_config.json + classifier.joblib")
    ap.add_argument("--seg_backend", default="classical",
                    choices=["classical", "unet"],
                    help="which segmentation backend to evaluate")
    ap.add_argument("--limit_per_class", type=int, default=None,
                    help="cap to this many samples per class (for a quick run)")
    ap.add_argument("--binary", action="store_true", default=True,
                    help="(default) collapse benign+malignant -> suspicious")
    ap.add_argument("--three_class", action="store_true",
                    help="keep normal/benign/malignant separate")
    args = ap.parse_args()

    binary = not args.three_class

    # -- discover all BUSI images ------------------------------------------
    samples = discover_busi(args.busi)
    if args.limit_per_class:
        kept: List = []
        per_class: Dict[str, int] = {}
        for s in samples:
            n = per_class.get(s.klass, 0)
            if n < args.limit_per_class:
                kept.append(s)
                per_class[s.klass] = n + 1
        samples = kept

    print(f"[eval] running pipeline over {len(samples)} images...")
    counts = {}
    for s in samples:
        counts[s.klass] = counts.get(s.klass, 0) + 1
    print(f"[eval] class distribution: {counts}")

    # -- build pipeline (same one the API uses) ----------------------------
    pipeline = _build_pipeline(Path(args.artifacts), seg_backend=args.seg_backend)
    if pipeline.classifier is None:
        print("[eval] WARNING: no classifier loaded - probabilities will be missing")
    print(f"[eval] segmentation backend: {args.seg_backend}")

    # -- run over every sample --------------------------------------------
    rows: List[Dict] = []
    bucket_totals = {"TP": 0, "TN": 0, "FP": 0, "FN": 0}
    t0 = time.time()

    for i, s in enumerate(samples):
        try:
            bgr = cv2.imread(s.image_path, cv2.IMREAD_COLOR)
            if bgr is None:
                print(f"  [{i+1}/{len(samples)}] FAILED to read {s.image_path}")
                continue

            result = pipeline.analyze(bgr, run_classifier=True)

            # Pull just the parts we want for analysis. Skip the overlay PNG -
            # that's huge (megabytes) and not useful for distribution analysis.
            cls = result.get("classification") or {}
            verdict = _decide_correctness(
                true_class=s.klass,
                predicted_label=cls.get("predicted_label"),
                binary=binary,
            )
            bucket_totals[verdict["bucket"]] += 1

            row = {
                "image_path": s.image_path,
                "image_name": Path(s.image_path).name,
                **verdict,
                "p_suspicious": (cls.get("probabilities", {}) or {}).get("suspicious"),
                "p_normal":     (cls.get("probabilities", {}) or {}).get("normal"),
                "segmentation": result.get("segmentation"),
                "features": result.get("features"),
            }
            rows.append(row)

        except Exception as e:
            print(f"  [{i+1}/{len(samples)}] ERROR on {s.image_path}: {e}")
            continue

        if (i + 1) % 25 == 0 or (i + 1) == len(samples):
            elapsed = time.time() - t0
            rate = (i + 1) / elapsed
            eta = (len(samples) - (i + 1)) / max(rate, 1e-6)
            print(f"  [{i+1}/{len(samples)}] {rate:.1f} img/s, ETA {eta:.0f}s "
                  f"  TP={bucket_totals['TP']} TN={bucket_totals['TN']} "
                  f"FP={bucket_totals['FP']} FN={bucket_totals['FN']}")

    # -- compute aggregate metrics ----------------------------------------
    n = max(1, len(rows))
    tp, tn, fp, fn = (bucket_totals[k] for k in ("TP", "TN", "FP", "FN"))
    accuracy = (tp + tn) / n
    sensitivity = tp / max(tp + fn, 1)   # recall on lesions
    specificity = tn / max(tn + fp, 1)   # true-normal rate
    precision   = tp / max(tp + fp, 1)
    f1          = 2 * tp / max(2 * tp + fp + fn, 1)

    summary = {
        "n_images": n,
        "buckets": bucket_totals,
        "accuracy": round(accuracy, 4),
        "sensitivity_recall": round(sensitivity, 4),
        "specificity": round(specificity, 4),
        "precision": round(precision, 4),
        "f1": round(f1, 4),
        "binary_mode": binary,
    }

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps({
        "summary": summary,
        "rows": rows,
    }, indent=2, default=str))

    # -- print final table -----------------------------------------------
    print()
    print("=" * 60)
    print("FINAL RESULTS")
    print("=" * 60)
    print(f"  Total images:    {n}")
    print(f"  TP (lesion correctly flagged):    {tp}")
    print(f"  TN (normal correctly cleared):    {tn}")
    print(f"  FP (normal wrongly flagged):      {fp}")
    print(f"  FN (lesion missed):               {fn}")
    print()
    print(f"  Accuracy:                  {accuracy:.3f}")
    print(f"  Sensitivity (recall):      {sensitivity:.3f}   <- catches real lesions")
    print(f"  Specificity:               {specificity:.3f}   <- doesn't fire on normals")
    print(f"  Precision:                 {precision:.3f}")
    print(f"  F1:                        {f1:.3f}")
    print()
    print(f"  Saved -> {out_path}")
    print(f"  Send this file to Claude for analysis.")


if __name__ == "__main__":
    main()
