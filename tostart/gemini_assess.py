#!/usr/bin/env python3
"""
gemini_assess.py - structured Gemini assessment for SoniSight's hybrid pipeline.

The old pipeline asked Gemini for a bounding box and used ONE bit of its output
(box or no box). On dev, that single bit explained 100% of the system's
discriminative power AND 100% of its errors: all 71 false negatives were images
where Gemini returned no box, and all 15 false positives were images where it
boxed normal tissue.

This module addresses that in three ways:

  1. assess_image()      - asks for structured JSON (lesion presence, box, and
                           semantic BI-RADS-style descriptors), so every field
                           becomes a usable feature instead of one bit.
  2. self-consistency    - n independent calls, majority vote on presence,
                           median box. Borderline cases are where single-sample
                           LLM calls are noisiest, and borderline cases are
                           exactly where all the errors live.
  3. verify_candidate()  - a yes/no verdict on a CV-proposed crop. This powers
                           the recall recovery loop: OpenCV proposes regions
                           (high recall, low precision), Gemini adjudicates
                           (high precision). Neither half can do this alone -
                           that is what makes the pipeline genuinely hybrid.

All functions take a `generate` callable so they can be unit-tested without a
live API key:  generate(parts: list) -> str
"""

import json
import re
import statistics
from typing import Callable, Dict, List, Optional, Tuple

# ---------------------------------------------------------------------------
# prompts
# ---------------------------------------------------------------------------

ASSESS_PROMPT = """You are assisting with breast ultrasound triage research.

Many of these images are NORMAL breast tissue with no discrete lesion. Decide
first whether a discrete focal lesion is genuinely present, then describe it.

Do not treat normal fibroglandular tissue, diffuse shadowing, speckle, ribs, or
imaging artifacts as lesions.

Respond with ONLY a JSON object, no markdown fences, no commentary:

{
  "lesion_present": true or false,
  "confidence": 0.0 to 1.0,
  "bbox": [x, y, w, h] or null,
  "shape": "round" | "oval" | "irregular" | null,
  "margins": "circumscribed" | "microlobulated" | "indistinct" | "spiculated" | null,
  "echogenicity": "anechoic" | "hypoechoic" | "isoechoic" | "hyperechoic" | "complex" | null,
  "orientation": "parallel" | "not_parallel" | null,
  "posterior": "none" | "shadowing" | "enhancement" | null,
  "suspicion": 0.0 to 1.0
}

Definitions that matter: "orientation" is "not_parallel" when the lesion's long
axis is perpendicular to the skin (taller-than-wide). "suspicion" is your
overall probability that this lesion warrants further workup (0 = clearly
benign, 1 = highly suspicious for malignancy).

bbox coordinates are pixels in the image provided. If lesion_present is false,
set bbox and all descriptive fields to null and suspicion to 0.0.
"""

SENSITIVE_PROMPT = """You are re-examining a breast ultrasound that a first pass
judged to contain NO lesion. Your job is to catch subtle lesions that a
conservative first read may have missed.

Look specifically for: small hypoechoic areas, subtle architectural distortion,
focal areas that interrupt the normal tissue plane, lesions near the image edge,
and low-contrast masses.

Be more sensitive than usual: if there is a plausible focal lesion, report it.
Still do not report diffuse shadowing, speckle, ribs, or artifacts.

Respond with ONLY a JSON object, no markdown fences:

{
  "lesion_present": true or false,
  "confidence": 0.0 to 1.0,
  "bbox": [x, y, w, h] or null,
  "suspicion": 0.0 to 1.0
}
"""

VERIFY_PROMPT = """This is a cropped region from a breast ultrasound, proposed
by an automated detector. The detector has low precision, so this crop may well
contain only normal tissue.

Decide whether this crop contains a genuine discrete focal lesion.

Respond with ONLY a JSON object, no markdown fences:

{
  "is_lesion": true or false,
  "confidence": 0.0 to 1.0,
  "shape": "round" | "oval" | "irregular" | null,
  "margins": "circumscribed" | "microlobulated" | "indistinct" | "spiculated" | null,
  "echogenicity": "anechoic" | "hypoechoic" | "isoechoic" | "hyperechoic" | "complex" | null,
  "suspicion": 0.0 to 1.0
}
"""


# ---------------------------------------------------------------------------
# parsing
# ---------------------------------------------------------------------------

def parse_json_response(text: str) -> Optional[dict]:
    """
    Robustly pull a JSON object out of a model response. Handles markdown
    fences, leading prose, and trailing commentary. Returns None on failure.
    """
    if not text:
        return None
    t = text.strip()
    t = re.sub(r"^```(?:json)?", "", t).strip()
    t = re.sub(r"```$", "", t).strip()
    try:
        return json.loads(t)
    except Exception:
        pass
    # fall back to the outermost {...} span
    start, end = t.find("{"), t.rfind("}")
    if start >= 0 and end > start:
        try:
            return json.loads(t[start:end + 1])
        except Exception:
            return None
    return None


def _clean_bbox(bbox, img_w: int, img_h: int) -> Optional[List[int]]:
    """Validate/clamp a bbox to the image. Returns None if unusable."""
    if not bbox or not isinstance(bbox, (list, tuple)) or len(bbox) != 4:
        return None
    try:
        x, y, w, h = [int(round(float(v))) for v in bbox]
    except Exception:
        return None
    x, y = max(0, min(x, img_w - 1)), max(0, min(y, img_h - 1))
    w, h = max(1, min(w, img_w - x)), max(1, min(h, img_h - y))
    if w < 8 or h < 8:
        return None
    return [x, y, w, h]


EMPTY_ASSESSMENT = {
    "lesion_present": False,
    "confidence": 0.0,
    "bbox": None,
    "shape": None,
    "margins": None,
    "echogenicity": None,
    "orientation": None,
    "posterior": None,
    "suspicion": 0.0,
    "votes_present": 0,
    "votes_total": 0,
    "source": "none",
}


# ---------------------------------------------------------------------------
# assessment with self-consistency voting
# ---------------------------------------------------------------------------

def assess_image(png_bytes: bytes, img_w: int, img_h: int,
                 generate: Callable[[List], str],
                 n_votes: int = 3) -> Dict:
    """
    Run the structured assessment n_votes times and combine by majority vote.

    Combination rules:
      - lesion_present: majority of successful votes
      - bbox: median of each coordinate across votes that found a lesion
              (median is robust to one outlier box)
      - categorical fields: mode among lesion-positive votes
      - confidence / suspicion: mean among lesion-positive votes
    """
    votes = []
    for _ in range(max(1, n_votes)):
        try:
            raw = generate([ASSESS_PROMPT, {"mime_type": "image/png", "data": png_bytes}])
            parsed = parse_json_response(raw)
            if parsed is not None:
                votes.append(parsed)
        except Exception:
            continue

    out = dict(EMPTY_ASSESSMENT)
    out["votes_total"] = len(votes)
    if not votes:
        return out

    positives = [v for v in votes if bool(v.get("lesion_present"))]
    out["votes_present"] = len(positives)

    if len(positives) * 2 <= len(votes):     # not a strict majority -> negative
        return out

    boxes = [_clean_bbox(v.get("bbox"), img_w, img_h) for v in positives]
    boxes = [b for b in boxes if b]
    if not boxes:
        # majority said lesion but no usable box; treat as no detection
        return out

    out["lesion_present"] = True
    out["bbox"] = [int(statistics.median([b[i] for b in boxes])) for i in range(4)]
    out["source"] = "gemini"

    def mode_of(field):
        vals = [v.get(field) for v in positives if v.get(field)]
        return statistics.mode(vals) if vals else None

    for field in ("shape", "margins", "echogenicity", "orientation", "posterior"):
        out[field] = mode_of(field)

    def mean_of(field, default=0.0):
        vals = []
        for v in positives:
            try:
                vals.append(float(v.get(field)))
            except (TypeError, ValueError):
                continue
        return sum(vals) / len(vals) if vals else default

    out["confidence"] = mean_of("confidence")
    out["suspicion"] = mean_of("suspicion")
    return out


def sensitive_second_pass(png_bytes: bytes, img_w: int, img_h: int,
                          generate: Callable[[List], str]) -> Optional[Dict]:
    """
    Re-examine an image the first pass called negative, with a recall-oriented
    prompt. Returns an assessment dict if it now finds a lesion, else None.
    """
    try:
        raw = generate([SENSITIVE_PROMPT, {"mime_type": "image/png", "data": png_bytes}])
    except Exception:
        return None
    parsed = parse_json_response(raw)
    if not parsed or not bool(parsed.get("lesion_present")):
        return None
    box = _clean_bbox(parsed.get("bbox"), img_w, img_h)
    if not box:
        return None
    out = dict(EMPTY_ASSESSMENT)
    out.update({
        "lesion_present": True,
        "bbox": box,
        "source": "gemini-sensitive",
        "votes_total": 1,
        "votes_present": 1,
    })
    for f in ("confidence", "suspicion"):
        try:
            out[f] = float(parsed.get(f, 0.0))
        except (TypeError, ValueError):
            out[f] = 0.0
    return out


def verify_candidate(crop_png: bytes, generate: Callable[[List], str]) -> Optional[Dict]:
    """
    Ask Gemini whether a CV-proposed crop is a genuine lesion.
    Returns the parsed verdict dict if it says yes, else None.
    """
    try:
        raw = generate([VERIFY_PROMPT, {"mime_type": "image/png", "data": crop_png}])
    except Exception:
        return None
    parsed = parse_json_response(raw)
    if not parsed or not bool(parsed.get("is_lesion")):
        return None
    return parsed
