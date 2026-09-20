#!/usr/bin/env python3
"""
compute_metrics.py - metrics for one BUSI split.

Reads a results CSV (default eval_results_dev.csv) and prints the confusion
matrix, accuracy / sensitivity / specificity, a per-class breakdown, and the
detection_method distribution (how often Gemini fired vs fell back).

Usage:
    python compute_metrics.py                      # reads eval_results_dev.csv
    python compute_metrics.py eval_results_test.csv

Standard library only.
"""

import csv
import sys
from collections import Counter, defaultdict
from pathlib import Path

CSV_PATH = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("eval_results_dev.csv")
THRESHOLD = 0.5  # predict "suspicious" if prob_suspicious >= THRESHOLD


def load_rows():
    rows = []
    with open(CSV_PATH, newline="") as f:
        for r in csv.DictReader(f):
            if r.get("error"):
                continue
            try:
                r["prob_suspicious"] = float(r["prob_suspicious"])
            except (ValueError, KeyError):
                continue
            rows.append(r)
    return rows


def main():
    if not CSV_PATH.exists():
        raise SystemExit(f"{CSV_PATH} not found. Run eval_busi.py first.")

    rows = load_rows()
    n = len(rows)
    if n == 0:
        raise SystemExit("No scored rows found.")

    for r in rows:
        r["pred_binary"] = "suspicious" if r["prob_suspicious"] >= THRESHOLD else "normal"

    tp = sum(1 for r in rows if r["true_binary"] == "suspicious" and r["pred_binary"] == "suspicious")
    fn = sum(1 for r in rows if r["true_binary"] == "suspicious" and r["pred_binary"] == "normal")
    tn = sum(1 for r in rows if r["true_binary"] == "normal" and r["pred_binary"] == "normal")
    fp = sum(1 for r in rows if r["true_binary"] == "normal" and r["pred_binary"] == "suspicious")

    accuracy = (tp + tn) / n
    sensitivity = tp / (tp + fn) if (tp + fn) else float("nan")
    specificity = tn / (tn + fp) if (tn + fp) else float("nan")

    print(f"File: {CSV_PATH.name}   images: {n}   threshold: {THRESHOLD}\n")

    print("Confusion matrix (positive = suspicious):")
    print("                   pred suspicious   pred normal")
    print(f"  true suspicious       {tp:5d}           {fn:5d}")
    print(f"  true normal           {fp:5d}           {tn:5d}\n")

    print(f"Accuracy     : {accuracy:.3f}")
    print(f"Sensitivity  : {sensitivity:.3f}   (of all suspicious, fraction flagged)")
    print(f"Specificity  : {specificity:.3f}   (of all normal, fraction cleared)\n")

    print("Per-class breakdown (% scored correctly):")
    for cls in ["normal", "benign", "malignant"]:
        sub = [r for r in rows if r["true_class"] == cls]
        if not sub:
            continue
        target = "normal" if cls == "normal" else "suspicious"
        correct = sum(1 for r in sub if r["pred_binary"] == target)
        print(f"  {cls:<10} {correct:4d}/{len(sub):<4d}  ({100*correct/len(sub):5.1f}%)   [target = {target}]")

    # detection_method distribution - now that it's logged correctly
    print("\nROI source (detection_method) by class:")
    by_cls = defaultdict(Counter)
    for r in rows:
        by_cls[r["true_class"]][r.get("detection_method") or "(blank)"] += 1
    for cls in ["normal", "benign", "malignant"]:
        if cls in by_cls:
            dist = ", ".join(f"{k}: {v}" for k, v in sorted(by_cls[cls].items()))
            print(f"  {cls:<10} {dist}")


if __name__ == "__main__":
    main()
