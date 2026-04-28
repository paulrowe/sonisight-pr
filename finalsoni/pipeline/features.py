"""
Feature extraction from a segmented lesion region.

Input:  preprocessed grayscale image (HxW uint8) + binary mask (HxW uint8/bool)
Output: dict[str, float | str | bool] with stable, JSON-safe keys.

Feature families:
  Geometric (from mask)
      area, perimeter, equivalent_diameter, bbox_w, bbox_h,
      circularity, solidity, extent, eccentricity, aspect_ratio,
      major_axis, minor_axis, orientation_deg
  Photometric (from grayscale within mask)
      mean_intensity, intensity_std, intensity_min, intensity_max,
      intensity_skew, intensity_p10, intensity_p90
  Margin / surround
      ring_edge_density, contrast_out_in, surround_mean
  Texture (GLCM, scikit-image)
      glcm_contrast, glcm_dissimilarity, glcm_homogeneity,
      glcm_energy, glcm_correlation, glcm_asm
  Categorical (rule-based, derived from above for human-readable summaries)
      shape, margins, texture, cyst_like

The categorical fields are rule-based so they're explainable; they're not used
as model inputs (the classifier consumes the numerical columns).

We import scikit-image lazily because it's a non-trivial dep. If it's missing
we still return all the geometric, photometric, margin, and categorical fields
- only the GLCM texture columns become NaN. The pipeline degrades gracefully.
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import cv2
import numpy as np


# Stable, ordered list of *numeric* feature names. The classifier uses
# exactly these columns in this order so a saved model is portable.
NUMERIC_FEATURES: Tuple[str, ...] = (
    # geometric
    "area", "perimeter", "equivalent_diameter",
    "bbox_w", "bbox_h", "aspect_ratio",
    "circularity", "solidity", "extent", "eccentricity",
    "major_axis", "minor_axis", "orientation_deg",
    # photometric
    "mean_intensity", "intensity_std", "intensity_min", "intensity_max",
    "intensity_skew", "intensity_p10", "intensity_p90",
    # margin / surround
    "ring_edge_density", "contrast_out_in", "surround_mean",
    # texture (GLCM)
    "glcm_contrast", "glcm_dissimilarity", "glcm_homogeneity",
    "glcm_energy", "glcm_correlation", "glcm_asm",
)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _largest_contour(mask: np.ndarray) -> Optional[np.ndarray]:
    cnts, _ = cv2.findContours((mask > 0).astype(np.uint8),
                               cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not cnts:
        return None
    return max(cnts, key=cv2.contourArea)


def _safe_skew(x: np.ndarray) -> float:
    """Pearson's third standardized moment, NaN-safe."""
    if x.size < 2:
        return 0.0
    m = float(x.mean())
    s = float(x.std())
    if s < 1e-6:
        return 0.0
    return float(((x - m) ** 3).mean() / (s ** 3))


def _glcm_features(gray_in_lesion: np.ndarray) -> Dict[str, float]:
    """
    Gray-Level Co-occurrence Matrix features. Returns NaNs if scikit-image
    isn't available so callers can still operate.
    """
    keys = ["glcm_contrast", "glcm_dissimilarity", "glcm_homogeneity",
            "glcm_energy", "glcm_correlation", "glcm_asm"]
    if gray_in_lesion.size < 16:
        return {k: float("nan") for k in keys}

    try:
        # Modern skimage (>=0.19) uses graycomatrix/graycoprops; older versions
        # used greycomatrix/greycoprops. Support both.
        try:
            from skimage.feature import graycomatrix, graycoprops
        except ImportError:
            from skimage.feature import greycomatrix as graycomatrix  # type: ignore
            from skimage.feature import greycoprops as graycoprops    # type: ignore
    except Exception:
        return {k: float("nan") for k in keys}

    # Quantize to 32 levels - GLCM is O(L^2) memory.
    q = (gray_in_lesion.astype(np.float32) / 255.0 * 31.0).astype(np.uint8)
    glcm = graycomatrix(q,
                        distances=[1, 2],
                        angles=[0, np.pi / 4, np.pi / 2, 3 * np.pi / 4],
                        levels=32, symmetric=True, normed=True)

    def _mean_prop(name: str) -> float:
        vals = graycoprops(glcm, name)
        return float(np.mean(vals))

    return {
        "glcm_contrast":      _mean_prop("contrast"),
        "glcm_dissimilarity": _mean_prop("dissimilarity"),
        "glcm_homogeneity":   _mean_prop("homogeneity"),
        "glcm_energy":        _mean_prop("energy"),
        "glcm_correlation":   _mean_prop("correlation"),
        "glcm_asm":           _mean_prop("ASM"),
    }


def _regionprops(mask: np.ndarray) -> Dict[str, float]:
    """
    Geometric properties via skimage.measure.regionprops if available, with
    OpenCV fallback. We always return the same keys.
    """
    binary = (mask > 0).astype(np.uint8)
    if binary.sum() == 0:
        return {k: 0.0 for k in (
            "equivalent_diameter", "extent", "eccentricity",
            "major_axis", "minor_axis", "orientation_deg",
        )}

    try:
        from skimage.measure import regionprops, label as sk_label
    except Exception:
        # Fallback: limited but stable.
        c = _largest_contour(mask)
        if c is None:
            return {"equivalent_diameter": 0.0, "extent": 0.0, "eccentricity": 0.0,
                    "major_axis": 0.0, "minor_axis": 0.0, "orientation_deg": 0.0}
        x, y, w, h = cv2.boundingRect(c)
        area = float(cv2.contourArea(c))
        eq_d = float(np.sqrt(4 * area / np.pi)) if area > 0 else 0.0
        extent = area / max(w * h, 1)
        # Approximate axes via fitEllipse if enough points
        major = minor = orient = 0.0
        if len(c) >= 5:
            (_, _), (MA, ma), angle = cv2.fitEllipse(c)
            major, minor, orient = float(max(MA, ma)), float(min(MA, ma)), float(angle)
        ecc = float(np.sqrt(max(0.0, 1 - (minor / max(major, 1e-6)) ** 2))) if major > 0 else 0.0
        return {
            "equivalent_diameter": eq_d, "extent": float(extent),
            "eccentricity": ecc, "major_axis": major, "minor_axis": minor,
            "orientation_deg": orient,
        }

    # Use the largest connected region.
    lab = sk_label(binary)
    props = regionprops(lab)
    if not props:
        return {"equivalent_diameter": 0.0, "extent": 0.0, "eccentricity": 0.0,
                "major_axis": 0.0, "minor_axis": 0.0, "orientation_deg": 0.0}
    p = max(props, key=lambda r: r.area)
    # scikit-image 0.26 renamed several regionprops attributes; support both.
    eq_d = (p.equivalent_diameter_area if hasattr(p, "equivalent_diameter_area")
            else p.equivalent_diameter)
    major = (p.axis_major_length if hasattr(p, "axis_major_length")
             else p.major_axis_length)
    minor = (p.axis_minor_length if hasattr(p, "axis_minor_length")
             else p.minor_axis_length)
    return {
        "equivalent_diameter": float(eq_d),
        "extent": float(p.extent),
        "eccentricity": float(p.eccentricity),
        "major_axis": float(major),
        "minor_axis": float(minor),
        "orientation_deg": float(np.degrees(p.orientation)),
    }


def _empty_features() -> Dict[str, float]:
    """Default values for when no mass is present."""
    return {k: 0.0 for k in NUMERIC_FEATURES}


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def extract_features(gray: np.ndarray, mask: np.ndarray) -> Dict:
    """
    Compute all features for a single (preprocessed gray, binary mask) pair.

    Returns
    -------
    dict
        Numeric columns from `NUMERIC_FEATURES` plus categorical fields:
        `mass_present` (bool), `shape`, `margins`, `texture`, `cyst_like`.

        When mask is empty the numeric columns are 0 and `mass_present=False`.
    """
    if gray.ndim != 2:
        raise ValueError("extract_features expects a grayscale image")
    if mask.shape != gray.shape:
        raise ValueError(f"mask shape {mask.shape} != gray shape {gray.shape}")

    binary = (mask > 0).astype(np.uint8)
    H, W = gray.shape

    if binary.sum() == 0:
        out = _empty_features()
        out.update({
            "mass_present": False, "shape": "none",
            "margins": "none", "texture": "homogeneous", "cyst_like": False,
        })
        return out

    cnt = _largest_contour(binary)
    if cnt is None:
        out = _empty_features()
        out.update({
            "mass_present": False, "shape": "none",
            "margins": "none", "texture": "homogeneous", "cyst_like": False,
        })
        return out

    # ---- geometric ---------------------------------------------------------
    area = float(cv2.contourArea(cnt))
    perim = float(cv2.arcLength(cnt, True))
    x, y, w, h = cv2.boundingRect(cnt)
    bw, bh = float(w), float(h)
    ar = float(max(bw, bh) / max(min(bw, bh), 1e-6))
    circularity = (4.0 * np.pi * area) / (perim * perim + 1e-6)

    hull = cv2.convexHull(cnt)
    hull_area = cv2.contourArea(hull) + 1e-6
    solidity = area / hull_area

    rp = _regionprops(binary)

    # ---- photometric -------------------------------------------------------
    inside_vals = gray[binary > 0]
    mean_in = float(inside_vals.mean())
    std_in = float(inside_vals.std())

    # ---- margin / surround -------------------------------------------------
    ring = cv2.dilate(binary, np.ones((3, 3), np.uint8))
    ring = cv2.subtract(ring, binary)
    edges = cv2.Canny(gray, 30, 100)
    ring_edges = float((edges[ring > 0] > 0).mean()) if ring.sum() > 0 else 0.0

    surround = cv2.dilate(ring, np.ones((9, 9), np.uint8))
    surround = cv2.subtract(surround, binary)
    surround_vals = gray[surround > 0]
    mean_out = float(surround_vals.mean()) if surround_vals.size else mean_in
    contrast_out_in = max(0.0, (mean_out - mean_in) / (mean_out + 1e-6))

    # ---- texture (GLCM on the bounding-box crop) ---------------------------
    crop = gray[y:y + h, x:x + w]
    glcm = _glcm_features(crop)

    # ---- assemble numeric dict --------------------------------------------
    feats: Dict[str, float] = {
        "area": area,
        "perimeter": perim,
        "equivalent_diameter": rp["equivalent_diameter"],
        "bbox_w": bw,
        "bbox_h": bh,
        "aspect_ratio": ar,
        "circularity": float(circularity),
        "solidity": float(solidity),
        "extent": rp["extent"],
        "eccentricity": rp["eccentricity"],
        "major_axis": rp["major_axis"],
        "minor_axis": rp["minor_axis"],
        "orientation_deg": rp["orientation_deg"],
        "mean_intensity": mean_in,
        "intensity_std": std_in,
        "intensity_min": float(inside_vals.min()),
        "intensity_max": float(inside_vals.max()),
        "intensity_skew": _safe_skew(inside_vals.astype(np.float32)),
        "intensity_p10": float(np.percentile(inside_vals, 10)),
        "intensity_p90": float(np.percentile(inside_vals, 90)),
        "ring_edge_density": ring_edges,
        "contrast_out_in": float(contrast_out_in),
        "surround_mean": mean_out,
        **glcm,
    }

    # ---- categorical (rule-based, for explanation only) -------------------
    if circularity >= 0.73:
        shape = "round"
    elif circularity >= 0.60:
        shape = "oval"
    else:
        shape = "irregular"

    if ring_edges < 0.16:
        margins = "smooth"
    elif ring_edges < 0.26:
        margins = "lobulated"
    else:
        margins = "spiculated"

    if std_in < 18:
        texture = "homogeneous"
    elif std_in < 33:
        texture = "mixed"
    else:
        texture = "heterogeneous"

    cyst_like = bool(circularity >= 0.75 and ring_edges < 0.10
                     and contrast_out_in >= 0.25 and std_in < 18.0)

    # tiny + low-contrast -> downgrade categorical labels but keep numeric
    area_frac = area / (H * W)
    looks_real = not (area_frac < 0.003 and contrast_out_in < 0.04)

    feats.update({
        "mass_present": bool(looks_real),
        "shape": shape if looks_real else "none",
        "margins": margins if looks_real else "none",
        "texture": texture,
        "cyst_like": cyst_like,
    })
    return feats


def features_vector(features: Dict) -> np.ndarray:
    """
    Pack a feature dict into a fixed-order numpy vector for the classifier.
    Missing keys (e.g. GLCM when skimage missing) become 0.
    """
    return np.asarray([float(features.get(k, 0.0) or 0.0)
                       for k in NUMERIC_FEATURES], dtype=np.float32)
