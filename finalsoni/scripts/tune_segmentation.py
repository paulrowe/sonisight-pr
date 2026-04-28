"""
Step 2 of the training pipeline:
    Tune the classical segmenter's parameters on (train + val) and report
    test-set Dice/IoU. Saves the best config to artifacts/seg_config.json.

Usage:
    python -m scripts.tune_segmentation \
        --splits artifacts/splits.json \
        --out artifacts/seg_config.json \
        --report artifacts/seg_report.json

Optional flags reduce the grid for fast iteration:
    --max_combos 30   # cap configurations evaluated
    --max_train  120  # cap lesion samples used during tuning
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from pipeline.preprocessing import PreprocessConfig
from pipeline.segmentation import ClassicalConfig, ClassicalSegmenter
from training.dataset import Splits
from training.evaluate_segmentation import evaluate, tune_classical


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--splits", required=True)
    ap.add_argument("--out", default="artifacts/seg_config.json")
    ap.add_argument("--report", default="artifacts/seg_report.json")
    ap.add_argument("--max_combos", type=int, default=None)
    ap.add_argument("--max_train", type=int, default=None)
    ap.add_argument("--target_size", type=int, default=256)
    args = ap.parse_args()

    splits = Splits.load(args.splits)
    pre_cfg = PreprocessConfig(target_size=args.target_size)

    print("=== tuning classical segmenter ===")
    best_cfg, tune_report = tune_classical(splits, pre_cfg,
                                           max_combos=args.max_combos,
                                           max_train=args.max_train)

    seg = ClassicalSegmenter(best_cfg)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    seg.save(args.out)
    print(f"saved best config -> {args.out}")

    print("=== final test-set evaluation (best config) ===")
    test_lesion = [s for s in splits.test if s.klass in ("benign", "malignant")]
    test_normal = [s for s in splits.test if s.klass == "normal"]

    eval_lesion = evaluate(seg, test_lesion, pre_cfg, include_normal=False)
    eval_normal = evaluate(seg, test_normal, pre_cfg, include_normal=False)
    eval_all    = evaluate(seg, splits.test, pre_cfg, include_normal=True)

    final = {
        "tuning": tune_report,
        "test_lesion_only": eval_lesion["aggregate"],
        "test_normal_only": eval_normal["aggregate"],
        "test_all_with_normal_penalty": eval_all["aggregate"],
        "config": json.loads(Path(args.out).read_text()),
    }
    Path(args.report).parent.mkdir(parents=True, exist_ok=True)
    Path(args.report).write_text(json.dumps(final, indent=2))
    print(f"saved report -> {args.report}")
    print()
    print(f"  TEST (lesion only): mean_dice={eval_lesion['aggregate']['mean_dice']:.3f} "
          f"mean_iou={eval_lesion['aggregate']['mean_iou']:.3f}")
    print(f"  TEST (normal only): specificity={eval_normal['aggregate']['specificity_normal']}")


if __name__ == "__main__":
    main()
