#!/usr/bin/env python3
"""
eval_busi_v2.py - run the hybrid v2 pipeline over a BUSI split, dumping ALL features.

Same split discipline as before (dev by default; test only once, at the end),
but now writes every Gemini semantic feature and every CV quantitative feature
as its own column. fit_fusion.py consumes that to train the fusion head and to
run the feature-group ablation.

Usage:
    1. uvicorn main:app --port 8000       (with .env present)
    2. python eval_busi_v2.py

Env:
    SPLIT is set below. MODE is read by the SERVER (SONI_MODE), not here - set it
    before launching uvicorn, e.g.:  SONI_MODE=gemini_only uvicorn main:app --port 8000
"""

import csv, json, os, time
from pathlib import Path
import requests

SPLIT = os.getenv("SPLIT", "dev")
MODE_TAG = os.getenv("MODE_TAG", "hybrid_v2")   # only used to name the output file
API_URL = "http://localhost:8000/predict"
DATASET_DIR = Path("Dataset_BUSI_with_GT")
SPLIT_FILE = Path("busi_split.json")
CLASSES = ["normal", "benign", "malignant"]
OUTPUT_CSV = Path(f"eval_{MODE_TAG}_{SPLIT}.csv")
MAX_RETRIES, TIMEOUT = 4, 180

BINARY = {"normal": "normal", "benign": "suspicious", "malignant": "suspicious"}
BASE_FIELDS = ["filename", "split", "true_class", "true_binary",
               "prob_normal", "prob_suspicious", "detection_method",
               "mass_present", "scorer", "votes", "rationale", "error"]


def is_img(n): return n.lower().endswith(".png") and "_mask" not in n.lower()


def post(path):
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            with open(path, "rb") as fh:
                r = requests.post(API_URL, params={"source": "live"},
                                  files={"file": (path.name, fh, "image/png")}, timeout=TIMEOUT)
            r.raise_for_status()
            return r.json()
        except Exception as e:
            if attempt == MAX_RETRIES: raise
            print(f"    retry {attempt}: {e}")
            time.sleep(2 ** attempt)


def main():
    if not SPLIT_FILE.exists(): raise SystemExit("run make_split.py first")
    split_map = json.loads(SPLIT_FILE.read_text())

    done = set()
    if OUTPUT_CSV.exists():
        with open(OUTPUT_CSV, newline="") as f:
            done = {r["filename"] for r in csv.DictReader(f)}

    work = []
    for cls in CLASSES:
        d = DATASET_DIR / cls
        if not d.exists(): continue
        for img in sorted(d.iterdir()):
            if is_img(img.name) and split_map.get(img.name) == SPLIT and img.name not in done:
                work.append((cls, img))
    print(f"MODE_TAG={MODE_TAG} SPLIT={SPLIT}: {len(done)} done, {len(work)} to do\n")

    # discover feature columns from the first successful response
    feat_keys, writer, f = None, None, None
    try:
        for i, (cls, img) in enumerate(work, 1):
            print(f"[{i}/{len(work)}] {cls}/{img.name}")
            row = {"filename": img.name, "split": SPLIT, "true_class": cls,
                   "true_binary": BINARY[cls]}
            feats = {}
            try:
                res = post(img)
                pr, de = res.get("probabilities", {}), res.get("descriptors", {})
                row["prob_normal"] = pr.get("normal", "")
                row["prob_suspicious"] = pr.get("suspicious", "")
                row["detection_method"] = de.get("detection_method", "")
                row["mass_present"] = de.get("mass_present", "")
                row["scorer"] = de.get("scorer", "")
                row["votes"] = de.get("votes", "")
                row["rationale"] = res.get("rationale", "")
                feats = res.get("features", {}) or {}
            except Exception as e:
                row["error"] = str(e)
                print(f"    FAILED: {e}")

            if writer is None:
                feat_keys = sorted(feats.keys())
                fields = BASE_FIELDS + feat_keys
                new = not OUTPUT_CSV.exists()
                f = open(OUTPUT_CSV, "a", newline="")
                writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
                if new: writer.writeheader()

            full = {k: "" for k in BASE_FIELDS + (feat_keys or [])}
            full.update(row)
            for k in (feat_keys or []): full[k] = feats.get(k, "")
            writer.writerow(full); f.flush()
    finally:
        if f: f.close()
    print(f"\nDone -> {OUTPUT_CSV.resolve()}")


if __name__ == "__main__":
    main()
