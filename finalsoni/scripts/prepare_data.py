"""
Step 1 of the training pipeline:
    Scan the BUSI dataset, split it 70/15/15 by class, and save the split
    manifest so every downstream step uses the same partitions.

Usage:
    python -m scripts.prepare_data \
        --busi /path/to/Dataset_BUSI_with_GT \
        --out artifacts/splits.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from training.dataset import (class_distribution, discover_busi,
                              make_splits)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--busi", required=True, help="path to Dataset_BUSI_with_GT")
    ap.add_argument("--out", default="artifacts/splits.json")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--val_frac", type=float, default=0.15)
    ap.add_argument("--test_frac", type=float, default=0.15)
    args = ap.parse_args()

    samples = discover_busi(args.busi)
    print(f"discovered {len(samples)} samples")
    print(f"class distribution: {class_distribution(samples)}")

    splits = make_splits(samples, val_frac=args.val_frac,
                         test_frac=args.test_frac, seed=args.seed)
    print(f"  train={len(splits.train)} ({class_distribution(splits.train)})")
    print(f"  val  ={len(splits.val)}   ({class_distribution(splits.val)})")
    print(f"  test ={len(splits.test)}  ({class_distribution(splits.test)})")

    splits.save(args.out)
    print(f"saved -> {args.out}")


if __name__ == "__main__":
    main()
