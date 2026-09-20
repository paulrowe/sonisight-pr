#!/usr/bin/env python3
"""
hybrid_features.py - quantitative CV features for SoniSight's hybrid pipeline.

Why this module exists: the old pipeline segmented a tight ROI with Otsu and
derived three coarse categorical descriptors (shape/margins/texture). On dev,
those turned out to carry ZERO discriminative signal - 348/372 boxed cases were
labeled "spiculated" and mean circularity was 0.398 for suspicious vs 0.395 for
normal. A near-constant feature is not a feature.

This module replaces that with:
  1. GrabCut segmentation seeded from the ROI box (a real lesion boundary,
     instead of a global threshold on a crop where the lesion fills the frame).
  2. Continuous, clinically-grounded ultrasound features that a vision-language
     model cannot measure by eye - which is the whole point of a hybrid:

     - depth_width_ratio   "taller-than-wide" is a classic malignancy sign
     - margin_sharpness    intensity gradient magnitude across the boundary
     - angular_margin_var  variance of radial distance = real spiculation proxy
     - posterior_shadow    acoustic shadowing behind the lesion (malignant) vs
                           enhancement (benign cyst)
     - glcm_*              Haralick-style texture (contrast, homogeneity, entropy)
     - rel_echogenicity    lesion brightness relative to surrounding tissue

All features are numpy-only (no scikit-image) to keep the deployment light.

Every function is defensive: if segmentation fails, features_from_roi returns a
dict with lesion_found=False and NaN-free defaults, so callers never crash.
"""

from typing import Dict, Optional, Tuple

import cv2
import numpy as np


# ----------------------------------------------------------------------------
# segmentation
# ----------------------------------------------------------------------------

def segment_lesion_grabcut(roi_bgr: np.ndarray, iters: int = 5) -> Optional[np.ndarray]:
    """
    Segment the lesion inside an ROI crop using GrabCut.

    The ROI is assumed to be a tight-ish box around a candidate lesion, so we
    seed GrabCut with a rectangle inset from the crop edges: the inner region is
    "probably foreground", the outer frame is "probably background". This is a
    much better fit than Otsu, which has no notion of where the lesion is.

    Returns a uint8 mask (255 = lesion) or None if segmentation fails.
    """
    h, w = roi_bgr.shape[:2]
    if h < 20 or w < 20:
        return None

    # inset rectangle: the lesion generally occupies the central majority of a
    # tight ROI, while the crop edges include surrounding tissue.
    mx, my = max(1, int(0.12 * w)), max(1, int(0.12 * h))
    rect = (mx, my, max(1, w - 2 * mx), max(1, h - 2 * my))

    mask = np.zeros((h, w), np.uint8)
    bgd, fgd = np.zeros((1, 65), np.float64), np.zeros((1, 65), np.float64)
    try:
        cv2.grabCut(roi_bgr, mask, rect, bgd, fgd, iters, cv2.GC_INIT_WITH_RECT)
    except Exception:
        return None

    out = np.where((mask == cv2.GC_FGD) | (mask == cv2.GC_PR_FGD), 255, 0).astype(np.uint8)

    # keep only the largest connected component, and clean it up
    k = np.ones((3, 3), np.uint8)
    out = cv2.morphologyEx(out, cv2.MORPH_OPEN, k, iterations=1)
    out = cv2.morphologyEx(out, cv2.MORPH_CLOSE, k, iterations=2)
    cnts, _ = cv2.findContours(out, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return None
    biggest = max(cnts, key=cv2.contourArea)
    area_frac = cv2.contourArea(biggest) / float(h * w)
    if area_frac < 0.02 or area_frac > 0.98:
        return None
    clean = np.zeros_like(out)
    cv2.drawContours(clean, [biggest], -1, 255, -1)
    return clean


# ----------------------------------------------------------------------------
# texture: GLCM (numpy only)
# ----------------------------------------------------------------------------

def glcm_features(gray: np.ndarray, mask: np.ndarray, levels: int = 16) -> Dict[str, float]:
    """
    Haralick-style texture features from a gray-level co-occurrence matrix,
    computed only over masked (lesion) pixels, averaged over 4 directions.
    """
    out = {"glcm_contrast": 0.0, "glcm_homogeneity": 0.0, "glcm_entropy": 0.0}
    ys, xs = np.where(mask > 0)
    if ys.size < 30:
        return out

    y0, y1, x0, x1 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
    patch = gray[y0:y1, x0:x1].astype(np.float32)
    sub = mask[y0:y1, x0:x1] > 0
    if patch.size < 30:
        return out

    # quantize to `levels`, marking non-lesion pixels as -1 so they're excluded
    lo, hi = float(patch[sub].min()), float(patch[sub].max())
    if hi <= lo:
        return out
    q = np.floor((patch - lo) / (hi - lo + 1e-9) * (levels - 1e-6)).astype(np.int16)
    q[~sub] = -1

    offsets = [(0, 1), (1, 0), (1, 1), (1, -1)]
    contrast = homogeneity = entropy = 0.0
    used = 0
    Hq, Wq = q.shape
    for dy, dx in offsets:
        # window of "reference" pixels, and the same window shifted by (dy, dx)
        y0, y1 = max(0, -dy), Hq - max(0, dy)
        x0, x1 = max(0, -dx), Wq - max(0, dx)
        if y1 <= y0 or x1 <= x0:
            continue
        a = q[y0:y1, x0:x1]
        b = q[y0 + dy:y1 + dy, x0 + dx:x1 + dx]
        if a.shape != b.shape or a.size == 0:
            continue
        ok = (a >= 0) & (b >= 0)
        if ok.sum() < 20:
            continue
        av, bv = a[ok].astype(np.int32), b[ok].astype(np.int32)
        P = np.zeros((levels, levels), np.float64)
        np.add.at(P, (av, bv), 1.0)
        P += P.T                      # symmetric
        s = P.sum()
        if s <= 0:
            continue
        P /= s
        i_idx, j_idx = np.indices(P.shape)
        diff = (i_idx - j_idx).astype(np.float64)
        contrast += float((P * diff ** 2).sum())
        homogeneity += float((P / (1.0 + np.abs(diff))).sum())
        nz = P[P > 0]
        entropy += float(-(nz * np.log2(nz)).sum())
        used += 1

    if used:
        out["glcm_contrast"] = contrast / used
        out["glcm_homogeneity"] = homogeneity / used
        out["glcm_entropy"] = entropy / used
    return out


# ----------------------------------------------------------------------------
# main feature extraction
# ----------------------------------------------------------------------------

def _angular_margin_variance(cnt: np.ndarray) -> float:
    """
    Spiculation proxy: normalized variance of the radial distance from the
    centroid to the boundary. Smooth/round lesions -> low variance;
    spiculated/irregular -> high. This is what "spiculated" should have meant.
    """
    pts = cnt.reshape(-1, 2).astype(np.float64)
    if pts.shape[0] < 8:
        return 0.0
    cx, cy = pts[:, 0].mean(), pts[:, 1].mean()
    r = np.hypot(pts[:, 0] - cx, pts[:, 1] - cy)
    if r.mean() <= 1e-6:
        return 0.0
    return float(r.std() / r.mean())


def _margin_sharpness(gray: np.ndarray, mask: np.ndarray) -> float:
    """
    Mean Sobel gradient magnitude on the lesion boundary band, normalized.
    Benign lesions tend to have sharp, well-defined margins; malignant ones
    are often ill-defined. Continuous, unlike the old smooth/lobulated bins.
    """
    band = cv2.subtract(cv2.dilate(mask, np.ones((5, 5), np.uint8)),
                        cv2.erode(mask, np.ones((5, 5), np.uint8)))
    if band.sum() == 0:
        return 0.0
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    magn = np.sqrt(gx ** 2 + gy ** 2)
    return float(magn[band > 0].mean() / 255.0)


def _posterior_shadow(gray: np.ndarray, mask: np.ndarray) -> float:
    """
    Acoustic shadowing: ultrasound beams travel top->bottom, so we compare mean
    intensity in a band BELOW the lesion against a lateral reference band at the
    same depth.

    Returns (lateral - below) / lateral:
      > 0  darker below  = shadowing        (suspicious / malignant)
      < 0  brighter below = enhancement     (benign cyst)
    """
    ys, xs = np.where(mask > 0)
    if ys.size == 0:
        return 0.0
    H, W = gray.shape
    y_bot, x0, x1 = int(ys.max()), int(xs.min()), int(xs.max())
    lesion_h = max(4, int(ys.max() - ys.min()))
    band_h = max(4, lesion_h // 2)

    ya, yb = y_bot + 2, min(H, y_bot + 2 + band_h)
    if ya >= yb or x1 <= x0:
        return 0.0
    below = gray[ya:yb, x0:x1 + 1]
    if below.size < 10:
        return 0.0

    pad = max(6, (x1 - x0) // 2)
    left = gray[ya:yb, max(0, x0 - pad):x0] if x0 > 0 else np.empty((0,))
    right = gray[ya:yb, x1 + 1:min(W, x1 + 1 + pad)] if x1 + 1 < W else np.empty((0,))
    ref = np.concatenate([left.ravel(), right.ravel()])
    if ref.size < 10:
        return 0.0

    m_below, m_ref = float(below.mean()), float(ref.mean())
    if m_ref <= 1e-6:
        return 0.0
    return float((m_ref - m_below) / m_ref)


def default_features() -> Dict[str, float]:
    """Feature dict for "no lesion segmented" - same keys, neutral values."""
    return {
        "lesion_found": 0.0,
        "area_frac": 0.0,
        "circularity": 0.0,
        "solidity": 0.0,
        "depth_width_ratio": 0.0,
        "angular_margin_var": 0.0,
        "margin_sharpness": 0.0,
        "rel_echogenicity": 0.0,
        "intensity_std": 0.0,
        "posterior_shadow": 0.0,
        "glcm_contrast": 0.0,
        "glcm_homogeneity": 0.0,
        "glcm_entropy": 0.0,
    }


def features_from_roi(roi_bgr: np.ndarray) -> Dict[str, float]:
    """
    Extract the full quantitative feature vector from an ROI crop.

    Note: NO 5% edge crop here (unlike the old preprocess()), because on a tight
    ROI that crop removes the lesion margin - exactly where spiculation lives.
    """
    f = default_features()
    if roi_bgr is None or roi_bgr.size == 0:
        return f

    if roi_bgr.ndim == 2:
        roi_bgr = cv2.cvtColor(roi_bgr, cv2.COLOR_GRAY2BGR)

    gray = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.medianBlur(gray, 3)   # speckle only; no CLAHE (it distorts echogenicity)

    mask = segment_lesion_grabcut(roi_bgr)
    if mask is None:
        return f

    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return f
    cnt = max(cnts, key=cv2.contourArea)
    area = float(cv2.contourArea(cnt))
    if area <= 0:
        return f

    H, W = gray.shape
    x, y, w, h = cv2.boundingRect(cnt)
    perim = float(cv2.arcLength(cnt, True))
    hull_area = float(cv2.contourArea(cv2.convexHull(cnt))) + 1e-6

    inside = gray[mask > 0]
    outside_band = cv2.subtract(cv2.dilate(mask, np.ones((15, 15), np.uint8)), mask)
    outside = gray[outside_band > 0] if outside_band.sum() > 0 else inside
    mean_in = float(inside.mean()) if inside.size else 0.0
    mean_out = float(outside.mean()) if outside.size else mean_in

    f["lesion_found"] = 1.0
    f["area_frac"] = area / float(H * W)
    f["circularity"] = (4.0 * np.pi * area) / (perim * perim + 1e-6)
    f["solidity"] = area / hull_area
    # taller-than-wide: > 1 is the classic malignancy sign
    f["depth_width_ratio"] = float(h) / float(w + 1e-6)
    f["angular_margin_var"] = _angular_margin_variance(cnt)
    f["margin_sharpness"] = _margin_sharpness(gray, mask)
    # negative = hypoechoic (darker than surroundings), typical of real lesions
    f["rel_echogenicity"] = float((mean_in - mean_out) / (mean_out + 1e-6))
    f["intensity_std"] = float(inside.std()) if inside.size else 0.0
    f["posterior_shadow"] = _posterior_shadow(gray, mask)
    f.update(glcm_features(gray, mask))
    return f


FEATURE_KEYS = sorted(default_features().keys())


# ----------------------------------------------------------------------------
# candidate proposer: high-recall regions for Gemini to verify
# ----------------------------------------------------------------------------

def propose_candidates(bgr: np.ndarray, k: int = 4) -> list:
    """
    Propose up to k candidate lesion boxes from a FULL image, tuned for RECALL,
    not precision. Most will be wrong; that's fine - Gemini verifies them.

    This is the hybrid mechanism for the 71 false negatives: when Gemini's
    first pass says "no lesion", CV suggests where to look again.

    Returns a list of (x, y, w, h) in original image coordinates, best-first.
    """
    if bgr is None or bgr.size == 0:
        return []
    if bgr.ndim == 2:
        bgr = cv2.cvtColor(bgr, cv2.COLOR_GRAY2BGR)

    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.medianBlur(gray, 5)
    gray = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(gray)
    H, W = gray.shape
    A = float(H * W)

    cands = []
    # multi-scale dark-blob search: lesions are hypoechoic, so threshold at
    # several low percentiles and collect plausible connected components.
    for pct in (8, 15, 25, 35):
        thr = float(np.percentile(gray, pct))
        bw = (gray <= thr).astype(np.uint8) * 255
        bw = cv2.morphologyEx(bw, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8), iterations=1)
        bw = cv2.morphologyEx(bw, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8), iterations=2)
        cnts, _ = cv2.findContours(bw, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for c in cnts:
            a = cv2.contourArea(c)
            if a <= 0:
                continue
            frac = a / A
            if frac < 0.01 or frac > 0.60:      # generous bounds on purpose
                continue
            x, y, w, h = cv2.boundingRect(c)
            ar = max(w / (h + 1e-6), h / (w + 1e-6))
            if ar > 4.0:
                continue
            # darker than surroundings + reasonably compact scores higher
            m = np.zeros((H, W), np.uint8)
            cv2.drawContours(m, [c], -1, 255, -1)
            ring = cv2.subtract(cv2.dilate(m, np.ones((15, 15), np.uint8)), m)
            mi = float(gray[m > 0].mean())
            mo = float(gray[ring > 0].mean()) if ring.sum() > 0 else mi
            darkness = max(0.0, (mo - mi) / (mo + 1e-6))
            perim = cv2.arcLength(c, True)
            circ = (4.0 * np.pi * a) / (perim * perim + 1e-6)
            cands.append((darkness + 0.5 * circ, (int(x), int(y), int(w), int(h))))

    # non-maximum suppression by IoU so we return diverse regions
    cands.sort(key=lambda t: -t[0])
    kept = []

    def iou(b1, b2):
        ax, ay, aw, ah = b1
        bx, by, bw_, bh = b2
        ix = max(0, min(ax + aw, bx + bw_) - max(ax, bx))
        iy = max(0, min(ay + ah, by + bh) - max(ay, by))
        inter = ix * iy
        union = aw * ah + bw_ * bh - inter
        return inter / (union + 1e-6)

    for _, box in cands:
        if all(iou(box, kb) < 0.35 for kb in kept):
            kept.append(box)
        if len(kept) >= k:
            break
    return kept
