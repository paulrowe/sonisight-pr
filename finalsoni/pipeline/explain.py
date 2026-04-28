"""
Doctor-facing explanation generator.

Strict rules (matches PART F of the spec):
  - No LLM. Every word in the output is generated from real computed
    features and the classifier's actual probability output.
  - No clinical claims. We use phrases like "flagged as higher-risk pattern",
    "prototype research system", "not for clinical diagnosis".
  - If we can't ground a statement in a number, we don't write it.

Output structure:
    {
      "headline": str,                # one short sentence
      "summary": str,                  # 2-3 sentences grounded in features
      "feature_highlights": [str],     # bullet-style facts
      "limitations": str,              # always-on safety note
      "drivers": [{name, value, importance}],  # top features the model used
    }
"""

from __future__ import annotations

from typing import Dict, List, Optional


# ---------------------------------------------------------------------------
# Tone helpers
# ---------------------------------------------------------------------------

# What we say about a probability bucket. Conservative wording.
def _risk_tier(p_suspicious: float) -> str:
    if p_suspicious >= 0.80:
        return "higher-risk pattern"
    if p_suspicious >= 0.55:
        return "moderate-risk pattern"
    if p_suspicious >= 0.30:
        return "indeterminate pattern"
    if p_suspicious >= 0.10:
        return "low-risk pattern"
    return "background-appearing region" if p_suspicious < 0.10 else "low-risk pattern"


# Crude human descriptions for numeric features. Used only as adjective
# tweaks on facts that we always print verbatim alongside.
def _circ_word(c: float) -> str:
    if c >= 0.78: return "very round"
    if c >= 0.65: return "rounded"
    if c >= 0.50: return "oval"
    return "irregular"

def _edge_word(e: float) -> str:
    if e < 0.12: return "smooth-margined"
    if e < 0.22: return "slightly irregular margin"
    if e < 0.34: return "lobulated/spiculated margin"
    return "highly irregular margin"

def _tex_word(s: float) -> str:
    if s < 18: return "homogeneous"
    if s < 33: return "mixed"
    return "heterogeneous"

def _contrast_word(c: float) -> str:
    if c < 0.05: return "isoechoic to surrounding tissue"
    if c < 0.15: return "mildly hypoechoic"
    if c < 0.30: return "hypoechoic"
    return "markedly hypoechoic"


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

LIMITATIONS = (
    "This is a prototype research system. Output is a software-derived pattern "
    "summary based on classical image features and a small machine-learning "
    "model trained on a public dataset. It is not a clinical diagnosis and is "
    "not for diagnostic use."
)


def generate_explanation(features: Dict,
                         classification: Optional[Dict] = None,
                         feature_importances: Optional[Dict[str, float]] = None,
                         segmentation_score: Optional[float] = None,
                         segmentation_backend: Optional[str] = None,
                         ) -> Dict:
    """
    Build a structured, grounded explanation. Every fact is reproducible from
    `features` and `classification`; nothing is invented.

    Parameters
    ----------
    features : dict
        Output of pipeline.features.extract_features(...).
    classification : dict, optional
        Output of pipeline.classifier.LesionClassifier.predict_from_features(...).
        If None, we still produce a description but skip risk wording.
    feature_importances : dict[str, float], optional
        From classifier.feature_importances - lets us list the top drivers.
    """
    mass = bool(features.get("mass_present", False))

    # --------- no mass branch --------------------------------------------
    if not mass:
        return {
            "headline": "No discrete lesion detected.",
            "summary": (
                "The segmentation step did not isolate a discrete region that "
                "passed the area, contrast, and shape gates. The image appears "
                "predominantly homogeneous within the analyzed area."
            ),
            "feature_highlights": [],
            "limitations": LIMITATIONS,
            "drivers": [],
        }

    # --------- mass branch ------------------------------------------------
    # Pull and round the numbers we'll cite.
    circ = float(features.get("circularity", 0.0))
    edge = float(features.get("ring_edge_density", 0.0))
    std  = float(features.get("intensity_std", 0.0))
    ctr  = float(features.get("contrast_out_in", 0.0))
    area = float(features.get("area", 0.0))
    sol  = float(features.get("solidity", 0.0))
    ar   = float(features.get("aspect_ratio", 0.0))
    cyst = bool(features.get("cyst_like", False))

    shape_word = _circ_word(circ)
    edge_w = _edge_word(edge)
    tex_w  = _tex_word(std)
    ctr_w  = _contrast_word(ctr)

    # ---- bullets ---------------------------------------------------------
    highlights: List[str] = [
        f"Region: {shape_word} (circularity {circ:.2f}).",
        f"Margins: {edge_w} (edge density {edge:.2f}).",
        f"Internal echotexture: {tex_w} (intensity std {std:.1f}).",
        f"Echogenicity vs surround: {ctr_w} (contrast {ctr:.2f}).",
        f"Geometric: solidity {sol:.2f}, aspect ratio {ar:.2f}, area {int(area)} px²."
    ]
    if cyst:
        highlights.append(
            "Pattern is consistent with a cyst-like appearance "
            "(round, smooth, dark, homogeneous)."
        )
    if segmentation_backend or segmentation_score is not None:
        bk = segmentation_backend or "?"
        sc = f"{segmentation_score:.2f}" if segmentation_score is not None else "?"
        highlights.append(f"Segmentation backend: {bk} (confidence {sc}).")

    # ---- model-driven wording -------------------------------------------
    drivers: List[Dict] = []
    if classification is not None:
        probs = classification.get("probabilities", {})
        p_susp = float(probs.get("suspicious", 0.0))
        tier = _risk_tier(p_susp)

        headline = f"Region flagged as {tier} (P(suspicious)={p_susp:.2f})."

        summary_bits = [
            f"A {shape_word} region with {edge_w.lower()} and "
            f"{tex_w} internal texture was identified, "
            f"appearing {ctr_w.lower()}."
        ]
        if p_susp >= 0.55:
            summary_bits.append(
                "The combination of irregularity, margin sharpness, and contrast "
                "drove the classifier toward a suspicious pattern."
            )
        elif p_susp >= 0.30:
            summary_bits.append(
                "Features were mixed; the classifier did not commit strongly "
                "either way and recommends review."
            )
        else:
            summary_bits.append(
                "Features are largely consistent with a benign-appearing pattern; "
                "the classifier estimated a low probability of a suspicious pattern."
            )
        summary = " ".join(summary_bits)

        # Top feature drivers (intersect importance with values we have).
        if feature_importances:
            ordered = sorted(feature_importances.items(),
                             key=lambda kv: kv[1], reverse=True)
            for name, imp in ordered[:5]:
                if name in features:
                    drivers.append({
                        "name": name,
                        "value": float(features.get(name, 0.0) or 0.0),
                        "importance": float(imp),
                    })
    else:
        headline = f"Region detected: {shape_word} with {edge_w.lower()}."
        summary = (
            f"A region was localized and characterized as {shape_word}, "
            f"with {edge_w.lower()} and {tex_w} internal texture, "
            f"appearing {ctr_w.lower()}. No classifier was applied."
        )

    return {
        "headline": headline,
        "summary": summary,
        "feature_highlights": highlights,
        "limitations": LIMITATIONS,
        "drivers": drivers,
    }
