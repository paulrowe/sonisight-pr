#!/usr/bin/env python3
"""
spotcheck.py - is the plane run contaminated?

Re-scores a few specific images that the plane run handled oddly, and prints
what the pipeline says NOW (with confirmed internet). Crucially, it prints
detection_method, so you can see whether Gemini ran or the saliency fallback did.

The tell: the plane run called normal (16) a spiculated mass at 0.95 suspicious.
If live Gemini now returns "no mass / low prob" on that same image, the plane
run had fallen back to saliency -> the CSV is contaminated -> we rerun clean.

Usage:
    1. uvicorn main:app --port 8000   (in one terminal, with .env present)
    2. python spotcheck.py            (in another)
"""

import requests

API_URL = "http://localhost:8000/predict"

# (folder, filename) -> what the PLANE run reported, for comparison
CASES = [
    ("normal",  "normal (16).png",  "plane: spiculated, 0.95 suspicious"),
    ("normal",  "normal (2).png",   "plane: spiculated, 0.95 suspicious"),
    ("normal",  "normal (9).png",   "plane: spiculated, 0.93 suspicious"),
    ("normal",  "normal (101).png", "plane: no mass, 0.05 (control - should stay clean)"),
    ("benign",  "benign (1).png",   "plane: no mass found, 0.30 (benign miss)"),
]


def main():
    for folder, name, plane_note in CASES:
        path = f"Dataset_BUSI_with_GT/{folder}/{name}"
        try:
            with open(path, "rb") as fh:
                files = {"file": (name, fh, "image/png")}
                r = requests.post(API_URL, params={"source": "live"},
                                  files=files, timeout=120)
            r.raise_for_status()
            res = r.json()
            desc = res.get("descriptors", {})
            probs = res.get("probabilities", {})
            print(f"\n{folder}/{name}")
            print(f"  {plane_note}")
            print(f"  NOW -> detection_method = {desc.get('detection_method')}")
            print(f"         mass_present={desc.get('mass_present')}  "
                  f"shape={desc.get('shape')}  margins={desc.get('margins')}")
            print(f"         prob_suspicious = {probs.get('suspicious')}")
        except Exception as e:
            print(f"\n{folder}/{name}\n  ERROR: {e}")


if __name__ == "__main__":
    main()
