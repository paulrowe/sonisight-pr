"""
Convenience: run all three training steps in order.

Usage:
    python -m scripts.train_all --busi /path/Dataset_BUSI_with_GT
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


def run(args: list[str]) -> None:
    print("\n$ " + " ".join(args))
    subprocess.check_call(args)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--busi", required=True)
    ap.add_argument("--out", default="artifacts")
    ap.add_argument("--max_combos", type=int, default=60,
                    help="cap segmentation grid combos (None = full)")
    args = ap.parse_args()

    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    splits = out / "splits.json"
    seg_cfg = out / "seg_config.json"

    py = sys.executable

    run([py, "-m", "scripts.prepare_data",
         "--busi", args.busi, "--out", str(splits)])
    run([py, "-m", "scripts.tune_segmentation",
         "--splits", str(splits), "--out", str(seg_cfg),
         "--report", str(out / "seg_report.json"),
         "--max_combos", str(args.max_combos)])
    run([py, "-m", "scripts.train_classifier",
         "--busi", args.busi, "--splits", str(splits),
         "--seg_config", str(seg_cfg), "--out", str(out)])

    print("\n=== done ===")
    print(f"  splits         -> {splits}")
    print(f"  seg config     -> {seg_cfg}")
    print(f"  classifier     -> {out / 'classifier.joblib'}")
    print("Next: start the API with `uvicorn main:app --reload`")


if __name__ == "__main__":
    main()
