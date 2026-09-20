#!/usr/bin/env python3
"""
pipeline_v2.py - SoniSight hybrid v2 orchestration.

Design goal: a pipeline where BOTH components measurably contribute, unlike v1
where a single Gemini bit (box / no box) explained 100% of discriminative power
and 100% of errors.

Detection (attacks the 71 dev false negatives):
    1. Gemini structured assessment, 3-vote self-consistency.
    2. If negative -> recall-oriented sensitive second pass.
    3. If still negative -> OpenCV proposes up to K candidate regions
       (high recall, low precision) and Gemini verifies each crop.
       This step is the hybrid mechanism: neither half can do it alone.

Characterization (fixes the dead features):
    GrabCut segmentation inside the box + continuous, clinically-grounded
    features (taller-than-wide, posterior shadowing, margin sharpness,
    angular margin variance, GLCM texture, relative echogenicity).

Scoring:
    - MODE "rules"  : transparent rule score (works with no trained head)
    - MODE "fusion" : learned logistic-regression head over Gemini semantic
                      features + CV quantitative features (fit on DEV only,
                      see fit_fusion.py). Falls back to rules if absent.

MODES (for the ablation grid):
    "hybrid_v2"    full pipeline above
    "gemini_only"  Gemini assessment only; ignores all CV features
    "opencv_only"  CV proposer + CV features only; never calls Gemini
"""

import io
import json
import os
from typing import Callable, Dict, List, Optional

import cv2
import numpy as np
from PIL import Image

from hybrid_features import (FEATURE_KEYS, default_features, features_from_roi,
                             propose_candidates)
from gemini_assess import (EMPTY_ASSESSMENT, assess_image, sensitive_second_pass,
                           verify_candidate)

MODE = os.getenv("SONI_MODE", "hybrid_v2")   # hybrid_v2 | gemini_only | opencv_only
N_VOTES = int(os.getenv("SONI_VOTES", "3"))
MAX_CANDIDATES = int(os.getenv("SONI_CANDIDATES", "3"))
FUSION_PATH = os.getenv("SONI_FUSION", "fusion_model.json")

# categorical Gemini fields -> one-hot feature names (must match fit_fusion.py)
CAT_FIELDS = {
    "shape": ["round", "oval", "irregular"],
    "margins": ["circumscribed", "microlobulated", "indistinct", "spiculated"],
    "echogenicity": ["anechoic", "hypoechoic", "isoechoic", "hyperechoic", "complex"],
    "orientation": ["parallel", "not_parallel"],
    "posterior": ["none", "shadowing", "enhancement"],
}


def _pil_to_png_bytes(pil_img: Image.Image) -> bytes:
    buf = io.BytesIO()
    pil_img.save(buf, format="PNG")
    return buf.getvalue()


# ---------------------------------------------------------------------------
# detection
# ---------------------------------------------------------------------------

def detect(bgr: np.ndarray, pil: Image.Image,
           generate: Optional[Callable] = None,
           mode: str = "hybrid_v2") -> Dict:
    """
    Locate a lesion. Returns an assessment dict (see gemini_assess.EMPTY_ASSESSMENT)
    with 'source' recording which stage found it - that field is the audit trail
    for how much each component contributes.
    """
    H, W = bgr.shape[:2]

    if mode == "opencv_only":
        boxes = propose_candidates(bgr, k=1)
        out = dict(EMPTY_ASSESSMENT)
        if boxes:
            out.update({"lesion_present": True, "bbox": list(boxes[0]),
                        "source": "opencv-proposer", "confidence": 0.5})
        return out

    if generate is None:
        return dict(EMPTY_ASSESSMENT)

    png = _pil_to_png_bytes(pil)

    # stage 1: structured assessment with self-consistency voting
    a = assess_image(png, W, H, generate, n_votes=N_VOTES)
    if a["lesion_present"]:
        return a

    if mode == "gemini_only":
        return a

    # stage 2: recall-oriented second look
    s = sensitive_second_pass(png, W, H, generate)
    if s is not None:
        return s

    # stage 3: CV proposes, Gemini verifies (the hybrid recall recovery)
    for (x, y, w, h) in propose_candidates(bgr, k=MAX_CANDIDATES):
        crop = bgr[y:y + h, x:x + w]
        if crop.size == 0:
            continue
        crop_png = _pil_to_png_bytes(
            Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)))
        v = verify_candidate(crop_png, generate)
        if v is None:
            continue
        out = dict(EMPTY_ASSESSMENT)
        out.update({
            "lesion_present": True,
            "bbox": [x, y, w, h],
            "source": "cv-proposed-gemini-verified",
            "votes_total": 1,
            "votes_present": 1,
        })
        for f in ("shape", "margins", "echogenicity"):
            out[f] = v.get(f)
        for f in ("confidence", "suspicion"):
            try:
                out[f] = float(v.get(f, 0.0))
            except (TypeError, ValueError):
                out[f] = 0.0
        return out

    return a   # genuinely nothing found


# ---------------------------------------------------------------------------
# feature assembly
# ---------------------------------------------------------------------------

def build_feature_row(assessment: Dict, cv_feats: Dict) -> Dict[str, float]:
    """
    Flat numeric feature row combining Gemini semantics and CV measurements.
    Used both for the fusion head and for the ablation (which subset wins).
    """
    row: Dict[str, float] = {}
    row["g_present"] = 1.0 if assessment.get("lesion_present") else 0.0
    row["g_confidence"] = float(assessment.get("confidence") or 0.0)
    row["g_suspicion"] = float(assessment.get("suspicion") or 0.0)
    vt = float(assessment.get("votes_total") or 0.0)
    row["g_vote_frac"] = (float(assessment.get("votes_present") or 0.0) / vt) if vt else 0.0
    for field, vals in CAT_FIELDS.items():
        cur = assessment.get(field)
        for v in vals:
            row[f"g_{field}_{v}"] = 1.0 if cur == v else 0.0
    for k in FEATURE_KEYS:
        row[f"cv_{k}"] = float(cv_feats.get(k, 0.0))
    return row


GEMINI_FEATURE_PREFIX = "g_"
CV_FEATURE_PREFIX = "cv_"


def feature_keys() -> List[str]:
    return sorted(build_feature_row(dict(EMPTY_ASSESSMENT), default_features()).keys())


# ---------------------------------------------------------------------------
# scoring
# ---------------------------------------------------------------------------

def rule_score(assessment: Dict, cv: Dict) -> float:
    """
    Transparent fallback score used when no fusion head is trained. Unlike v1's
    rules, this consumes BOTH Gemini semantics and CV measurements, so it isn't
    a restatement of a single bit.
    """
    if not assessment.get("lesion_present"):
        return 0.05

    # start from Gemini's own suspicion if it gave one, else a neutral prior
    s = float(assessment.get("suspicion") or 0.0)
    if s <= 0.0:
        s = 0.45

    # Gemini semantics
    if assessment.get("margins") in ("spiculated", "indistinct", "microlobulated"):
        s += 0.12
    if assessment.get("shape") == "irregular":
        s += 0.10
    if assessment.get("orientation") == "not_parallel":
        s += 0.10
    if assessment.get("posterior") == "shadowing":
        s += 0.08
    if assessment.get("posterior") == "enhancement":
        s -= 0.08
    if assessment.get("echogenicity") == "anechoic":
        s -= 0.10        # simple cyst

    # CV measurements (things the LLM cannot measure by eye)
    if cv.get("lesion_found"):
        if cv.get("depth_width_ratio", 0) > 1.0:      # taller-than-wide
            s += 0.10
        if cv.get("angular_margin_var", 0) > 0.22:    # irregular boundary
            s += 0.06
        if cv.get("posterior_shadow", 0) > 0.10:
            s += 0.06
        elif cv.get("posterior_shadow", 0) < -0.10:
            s -= 0.06
        if cv.get("circularity", 0) > 0.80 and cv.get("angular_margin_var", 1) < 0.12:
            s -= 0.10                                  # round + smooth

    return float(min(0.97, max(0.03, s)))


class FusionHead:
    """Logistic regression head loaded from fusion_model.json (fit on dev only)."""

    def __init__(self, keys: List[str], coefs: List[float], intercept: float):
        self.keys, self.coefs, self.intercept = keys, coefs, intercept

    @classmethod
    def load(cls, path: str = FUSION_PATH) -> Optional["FusionHead"]:
        try:
            with open(path) as f:
                d = json.load(f)
            return cls(d["keys"], d["coefs"], d["intercept"])
        except Exception:
            return None

    def predict_proba(self, row: Dict[str, float]) -> float:
        z = self.intercept + sum(c * float(row.get(k, 0.0))
                                 for k, c in zip(self.keys, self.coefs))
        return float(1.0 / (1.0 + np.exp(-z)))


_FUSION = FusionHead.load()


# ---------------------------------------------------------------------------
# top-level entry point
# ---------------------------------------------------------------------------

def analyze(bgr: np.ndarray, pil: Image.Image,
            generate: Optional[Callable] = None,
            mode: str = None) -> Dict:
    """
    Full v2 analysis. Returns a dict with:
      probabilities, descriptors (legacy-compatible), features (flat row),
      assessment (raw), detection_source, rationale
    """
    mode = mode or MODE
    assessment = detect(bgr, pil, generate=generate, mode=mode)

    cv_feats = default_features()
    if assessment.get("lesion_present") and assessment.get("bbox") and mode != "gemini_only":
        x, y, w, h = assessment["bbox"]
        # pad the box slightly: margins matter and a tight box clips them
        pad = int(0.10 * max(w, h))
        H, W = bgr.shape[:2]
        x0, y0 = max(0, x - pad), max(0, y - pad)
        x1, y1 = min(W, x + w + pad), min(H, y + h + pad)
        roi = bgr[y0:y1, x0:x1]
        if roi.size:
            cv_feats = features_from_roi(roi)

    row = build_feature_row(assessment, cv_feats)

    if _FUSION is not None and mode == "hybrid_v2":
        p_susp = _FUSION.predict_proba(row)
        scorer = "fusion"
    else:
        p_susp = rule_score(assessment, cv_feats)
        scorer = "rules"

    descriptors = {
        "mass_present": bool(assessment.get("lesion_present")),
        "detection_method": assessment.get("source", "none"),
        "shape": assessment.get("shape") or "none",
        "margins": assessment.get("margins") or "none",
        "echogenicity": assessment.get("echogenicity") or "none",
        "orientation": assessment.get("orientation") or "none",
        "posterior": assessment.get("posterior") or "none",
        "depth_width_ratio": round(cv_feats.get("depth_width_ratio", 0.0), 4),
        "angular_margin_var": round(cv_feats.get("angular_margin_var", 0.0), 4),
        "posterior_shadow": round(cv_feats.get("posterior_shadow", 0.0), 4),
        "margin_sharpness": round(cv_feats.get("margin_sharpness", 0.0), 4),
        "circularity": round(cv_feats.get("circularity", 0.0), 4),
        "rel_echogenicity": round(cv_feats.get("rel_echogenicity", 0.0), 4),
        "cv_lesion_found": bool(cv_feats.get("lesion_found")),
        "scorer": scorer,
        "votes": f"{assessment.get('votes_present',0)}/{assessment.get('votes_total',0)}",
    }

    if not assessment.get("lesion_present"):
        rationale = "No discrete focal lesion identified; tissue appears within normal limits."
    else:
        bits = []
        if assessment.get("shape"):
            bits.append(f"{assessment['shape']} shape")
        if assessment.get("margins"):
            bits.append(f"{assessment['margins']} margins")
        if cv_feats.get("depth_width_ratio", 0) > 1.0:
            bits.append("taller-than-wide orientation")
        if cv_feats.get("posterior_shadow", 0) > 0.10:
            bits.append("posterior shadowing")
        elif cv_feats.get("posterior_shadow", 0) < -0.10:
            bits.append("posterior enhancement")
        rationale = ("Detected lesion shows " + ", ".join(bits) + "."
                     if bits else "Lesion detected; features indeterminate.")

    return {
        "probabilities": {"normal": round(1.0 - p_susp, 3), "suspicious": round(p_susp, 3)},
        "descriptors": descriptors,
        "features": row,
        "assessment": assessment,
        "detection_source": assessment.get("source", "none"),
        "rationale": rationale,
        "roi_box": assessment.get("bbox"),
        "mode": mode,
    }
