#!/usr/bin/env python3
"""
fit_fusion.py - train the fusion head on DEV and test the hybrid claim.

This script is the falsifiability check for the whole project. It fits three
logistic-regression heads on the dev split using cross-validation:

    Gemini features only  (g_*)
    CV features only      (cv_*)
    Fused                 (both)

If Fused does not beat BOTH single-source heads, the pipeline is not meaningfully
hybrid and the paper should say so. That is a real result either way - and it is
the exact question v1 failed (v1's CV stage added 0.000 AUROC).

Writes fusion_model.json (coefficients) for pipeline_v2 to load at serve time.

Usage:
    python fit_fusion.py eval_hybrid_v2_dev.csv

Requires: pip install scikit-learn
"""

import csv, json, sys
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

CSV_PATH = sys.argv[1] if len(sys.argv) > 1 else "eval_hybrid_v2_dev.csv"
OUT = "fusion_model.json"


def load(path):
    rows = [r for r in csv.DictReader(open(path, newline="")) if not r.get("error")]
    if not rows: raise SystemExit("no usable rows")
    keys = [k for k in rows[0] if k.startswith(("g_", "cv_"))]
    X, y, p_sys = [], [], []
    for r in rows:
        try:
            X.append([float(r[k] or 0.0) for k in keys])
        except ValueError:
            continue
        y.append(1 if r["true_binary"] == "suspicious" else 0)
        try: p_sys.append(float(r["prob_suspicious"]))
        except (ValueError, KeyError): p_sys.append(np.nan)
    return np.array(X), np.array(y), np.array(p_sys), keys


def evaluate(X, y, keys, subset_prefix=None, label=""):
    idx = [i for i, k in enumerate(keys)
           if subset_prefix is None or k.startswith(subset_prefix)]
    if not idx: return None, None
    Xs = X[:, idx]
    mdl = make_pipeline(StandardScaler(),
                        LogisticRegression(max_iter=3000, class_weight="balanced"))
    cv = StratifiedKFold(5, shuffle=True, random_state=42)
    p = cross_val_predict(mdl, Xs, y, cv=cv, method="predict_proba")[:, 1]
    auc = roc_auc_score(y, p)
    print(f"  {label:<26} AUROC = {auc:.3f}   ({len(idx)} features)")
    return auc, (mdl, idx, p)


def op_point(y, p, target_sens=0.90):
    """Report the operating point that hits target sensitivity, chosen on dev."""
    ths = np.unique(np.round(p, 3))
    best = None
    for t in ths:
        pred = (p >= t).astype(int)
        tp = int(((pred == 1) & (y == 1)).sum()); fn = int(((pred == 0) & (y == 1)).sum())
        tn = int(((pred == 0) & (y == 0)).sum()); fp = int(((pred == 1) & (y == 0)).sum())
        sens = tp / (tp + fn) if tp + fn else 0
        spec = tn / (tn + fp) if tn + fp else 0
        if sens >= target_sens and (best is None or spec > best[2]):
            best = (t, sens, spec, (tp, fn, tn, fp))
    return best


def main():
    X, y, p_sys, keys = load(CSV_PATH)
    print(f"{CSV_PATH}: {len(y)} rows, {int(y.sum())} suspicious, {len(keys)} features\n")

    if not np.isnan(p_sys).all():
        m = ~np.isnan(p_sys)
        print(f"  {'shipped pipeline score':<26} AUROC = {roc_auc_score(y[m], p_sys[m]):.3f}\n")

    print("Feature-group ablation (5-fold CV on dev):")
    auc_g, _ = evaluate(X, y, keys, "g_", "Gemini features only")
    auc_c, _ = evaluate(X, y, keys, "cv_", "CV features only")
    auc_f, fused = evaluate(X, y, keys, None, "FUSED (both)")

    print()
    if auc_f and auc_g and auc_c:
        gain = auc_f - max(auc_g, auc_c)
        verdict = ("HYBRID JUSTIFIED: fusion beats both single-source heads."
                   if gain > 0.01 else
                   "NOT MEANINGFULLY HYBRID: fusion adds <=0.01 AUROC over the best single source.")
        print(f"  fused - best_single = {gain:+.3f}  ->  {verdict}")

    if fused:
        mdl, idx, p = fused
        best = op_point(y, p, target_sens=0.90)
        if best:
            t, sens, spec, (tp, fn, tn, fp) = best
            print(f"\n  dev operating point at >=0.90 sensitivity: threshold {t:.3f}")
            print(f"    sensitivity {sens:.3f}   specificity {spec:.3f}   (TP{tp} FN{fn} TN{tn} FP{fp})")

        # refit on all of dev and export coefficients for serving
        mdl.fit(X[:, idx], y)
        sc, lr = mdl.named_steps["standardscaler"], mdl.named_steps["logisticregression"]
        # fold standardization into the linear terms so serving needs no scaler
        coefs = (lr.coef_[0] / sc.scale_).tolist()
        intercept = float(lr.intercept_[0] - np.sum(lr.coef_[0] * sc.mean_ / sc.scale_))
        json.dump({"keys": [keys[i] for i in idx], "coefs": coefs,
                   "intercept": intercept}, open(OUT, "w"), indent=1)
        print(f"\n  wrote {OUT} (load automatically by pipeline_v2)")
        top = sorted(zip([keys[i] for i in idx], lr.coef_[0]), key=lambda t: -abs(t[1]))[:10]
        print("\n  strongest standardized coefficients:")
        for k, c in top: print(f"    {k:<28} {c:+.3f}")


if __name__ == "__main__":
    main()
