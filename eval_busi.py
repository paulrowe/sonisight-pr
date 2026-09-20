#!/usr/bin/env python3
"""
eval_busi.py - run the SoniSight pipeline over a BUSI split.

Reads busi_split.json (from make_split.py) and scores ONLY the split named in
SPLIT below. Default is "dev" so you never touch the test set by accident.
When you're done improving, set SPLIT = "test" and run it ONCE.

Skips *_mask*.png files. Logs detection_method (gemini / saliency-fallback /
none) so you can audit which ROI path ran. Resumes if interrupted.

Usage:
    1. uvicorn main:app --port 8000      (one terminal, with .env present)
    2. python eval_busi.py               (another terminal)

Requires:  pip install requests
"""

import csv
import json
import time
from pathlib import Path

import requests

# ---- config ----
SPLIT = "dev"                                  # "dev" while improving; "test" only at the very end
API_URL = "http://localhost:8000/predict"
DATASET_DIR = Path("Dataset_BUSI_with_GT")
SPLIT_FILE = Path("busi_split.json")
CLASSES = ["normal", "benign", "malignant"]
OUTPUT_CSV = Path(f"eval_results_{SPLIT}.csv")
MAX_RETRIES = 4
TIMEOUT = 120

BINARY = {"normal": "normal", "benign": "suspicious", "malignant": "suspicious"}

FIELDS = [
    "filename", "split", "true_class", "true_binary",
    "prob_normal", "prob_suspicious",
    "detection_method",
    "mass_present", "shape", "margins", "texture",
    "cyst_like", "circularity", "contrast_out_in",
    "rationale", "error",
]


def is_ultrasound_image(name: str) -> bool:
    low = name.lower()
    return low.endswith(".png") and "_mask" not in low


def already_done() -> set:
    done = set()
    if OUTPUT_CSV.exists():
        with open(OUTPUT_CSV, newline="") as f:
            for row in csv.DictReader(f):
                done.add(row["filename"])
    return done


def post_image(path: Path) -> dict:
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            with open(path, "rb") as fh:
                files = {"file": (path.name, fh, "image/png")}
                r = requests.post(API_URL, params={"source": "live"},
                                  files=files, timeout=TIMEOUT)
            r.raise_for_status()
            return r.json()
        except Exception as e:
            if attempt == MAX_RETRIES:
                raise
            wait = 2 ** attempt
            print(f"    retry {attempt}/{MAX_RETRIES} after error: {e} (waiting {wait}s)")
            time.sleep(wait)


def main():
    if not DATASET_DIR.exists():
        raise SystemExit(f"Dataset folder not found: {DATASET_DIR.resolve()}")
    if not SPLIT_FILE.exists():
        raise SystemExit("busi_split.json not found. Run:  python make_split.py")

    split_map = json.loads(SPLIT_FILE.read_text())
    done = already_done()
    new_file = not OUTPUT_CSV.exists()

    work = []
    for cls in CLASSES:
        cdir = DATASET_DIR / cls
        if not cdir.exists():
            print(f"WARNING: missing class folder {cdir}")
            continue
        for img in sorted(cdir.iterdir()):
            if not is_ultrasound_image(img.name):
                continue
            if split_map.get(img.name) != SPLIT:      # only this split
                continue
            if img.name in done:
                continue
            work.append((cls, img))

    total = len(work)
    print(f"SPLIT = {SPLIT}.  {len(done)} already scored, {total} to do.\n")

    with open(OUTPUT_CSV, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        if new_file:
            writer.writeheader()

        for i, (cls, img) in enumerate(work, 1):
            print(f"[{i}/{total}] {cls}/{img.name}")
            row = {k: "" for k in FIELDS}
            row["filename"] = img.name
            row["split"] = SPLIT
            row["true_class"] = cls
            row["true_binary"] = BINARY[cls]
            try:
                res = post_image(img)
                probs = res.get("probabilities", {})
                desc = res.get("descriptors", {})
                row["prob_normal"] = probs.get("normal", "")
                row["prob_suspicious"] = probs.get("suspicious", "")
                row["detection_method"] = desc.get("detection_method", "")
                row["mass_present"] = desc.get("mass_present", "")
                row["shape"] = desc.get("shape", "")
                row["margins"] = desc.get("margins", "")
                row["texture"] = desc.get("texture", "")
                row["cyst_like"] = desc.get("cyst_like", "")
                row["circularity"] = desc.get("circularity", "")
                row["contrast_out_in"] = desc.get("contrast_out_in", "")
                row["rationale"] = res.get("rationale", "")
            except Exception as e:
                row["error"] = str(e)
                print(f"    FAILED: {e}")
            writer.writerow(row)
            f.flush()

    print(f"\nDone. Results in {OUTPUT_CSV.resolve()}")


if __name__ == "__main__":
    main()
