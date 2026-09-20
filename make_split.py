#!/usr/bin/env python3
"""
make_split.py - deterministic dev/test split of the BUSI dataset.

Writes busi_split.json mapping each image filename -> "dev" or "test".
Stratified by class, fixed seed, so the split is reproducible and reportable
in the paper's methods.

We develop and tune ONLY on dev. The test split stays frozen and is scored
once, at the very end. That discipline is what makes an accuracy improvement
defensible instead of suspicious.

Usage:  python make_split.py
"""

import json
import random
from pathlib import Path

DATASET_DIR = Path("Dataset_BUSI_with_GT")
CLASSES = ["normal", "benign", "malignant"]
TEST_FRAC = 0.30
SEED = 42
OUT = Path("busi_split.json")


def is_ultrasound_image(name: str) -> bool:
    low = name.lower()
    return low.endswith(".png") and "_mask" not in low


def main():
    if not DATASET_DIR.exists():
        raise SystemExit(f"Dataset folder not found: {DATASET_DIR.resolve()}")

    rng = random.Random(SEED)
    split = {}
    print(f"Split: {int((1-TEST_FRAC)*100)}% dev / {int(TEST_FRAC*100)}% test, seed={SEED}\n")

    for cls in CLASSES:
        cdir = DATASET_DIR / cls
        imgs = sorted(p.name for p in cdir.iterdir() if is_ultrasound_image(p.name))
        rng.shuffle(imgs)
        n_test = round(len(imgs) * TEST_FRAC)
        test = set(imgs[:n_test])
        for name in imgs:
            split[name] = "test" if name in test else "dev"
        print(f"  {cls:<10} total {len(imgs):4d}  ->  dev {len(imgs)-n_test:4d}   test {n_test:4d}")

    OUT.write_text(json.dumps(split, indent=0))
    print(f"\nWrote {OUT.resolve()}  ({len(split)} images)")


if __name__ == "__main__":
    main()
