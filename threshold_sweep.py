#!/usr/bin/env python3
"""
threshold_sweep.py - see the sensitivity/specificity tradeoff on the dev split.

Re-reads eval_results_dev.csv (no rerun needed) and reports sensitivity,
specificity, and per-class flagged rates across a range of thresholds, so you
can choose a triage operating point ON DEV. You then report test once at the
threshold you pick here.

Usage:  python threshold_sweep.py
"""

import csv
from pathlib import Path

CSV_PATH = Path("eval_results_dev.csv")
THRESHOLDS = [0.50, 0.45, 0.40, 0.35, 0.30, 0.25, 0.20]


def load():
    rows = []
    with open(CSV_PATH, newline="") as f:
        for r in csv.DictReader(f):
            if r.get("error"):
                continue
            try:
                r["p"] = float(r["prob_suspicious"])
            except (ValueError, KeyError):
                continue
            rows.append(r)
    return rows


def main():
    if not CSV_PATH.exists():
        raise SystemExit(f"{CSV_PATH} not found.")
    rows = load()
    pos = [r for r in rows if r["true_binary"] == "suspicious"]
    neg = [r for r in rows if r["true_binary"] == "normal"]
    ben = [r for r in rows if r["true_class"] == "benign"]
    mal = [r for r in rows if r["true_class"] == "malignant"]

    print(f"dev images: {len(rows)}   suspicious: {len(pos)}   normal: {len(neg)}\n")
    print(f"{'thresh':>7} {'sens':>7} {'spec':>7} {'benign%':>8} {'malig%':>8}")
    for t in THRESHOLDS:
        sens = sum(1 for r in pos if r["p"] >= t) / len(pos) if pos else 0
        spec = sum(1 for r in neg if r["p"] < t) / len(neg) if neg else 0
        bflag = sum(1 for r in ben if r["p"] >= t) / len(ben) if ben else 0
        mflag = sum(1 for r in mal if r["p"] >= t) / len(mal) if mal else 0
        print(f"{t:>7.2f} {sens:>7.3f} {spec:>7.3f} {100*bflag:>7.1f}% {100*mflag:>7.1f}%")

    print("\nmalig% = fraction of malignant flagged (your most important number).")


if __name__ == "__main__":
    main()
