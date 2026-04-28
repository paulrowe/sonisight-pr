"""
Step 3 of the training pipeline:
    Train the Random Forest classifier on handcrafted features.

Usage:
    python -m scripts.train_classifier \
        --busi /path/Dataset_BUSI_with_GT \
        --splits artifacts/splits.json \
        --seg_config artifacts/seg_config.json \
        --out artifacts
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from training.train_classifier import train_classifier


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--busi", required=True)
    ap.add_argument("--splits", required=True)
    ap.add_argument("--seg_config", default="artifacts/seg_config.json")
    ap.add_argument("--out", default="artifacts")
    ap.add_argument("--three_class", action="store_true")
    args = ap.parse_args()

    report = train_classifier(args.busi, args.splits, args.out,
                              seg_config_path=args.seg_config,
                              binary=not args.three_class)

    print()
    print("=== summary ===")
    print(f"  cv mean f1_macro:  {report['cv']['cv_mean_f1']}")
    print(f"  test (gt mask)  :  acc={report['test_with_ground_truth_mask']['accuracy']:.3f} "
          f"f1={report['test_with_ground_truth_mask']['f1']:.3f}")
    print(f"  test (pred mask):  acc={report['test_with_predicted_mask']['accuracy']:.3f} "
          f"f1={report['test_with_predicted_mask']['f1']:.3f}")
    print(f"  saved model     :  {report['model_path']}")


if __name__ == "__main__":
    main()
