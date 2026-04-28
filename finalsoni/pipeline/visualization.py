"""
Doctor-facing overlays for the frontend.

We produce three layers and a composite:
  - bbox overlay        (just the predicted ROI rectangle)
  - segmentation overlay (filled translucent mask + contour)
  - composite           (everything together with a small risk badge)

The composite is what the frontend usually wants. Each function returns a
base64 PNG string so the FastAPI response can ship them directly.
"""

from __future__ import annotations

import base64
from typing import Optional, Tuple

import cv2
import numpy as np


# ---------------------------------------------------------------------------
# Color & encoding helpers
# ---------------------------------------------------------------------------

# BGR (OpenCV) - chosen for adequate contrast on grayscale ultrasound.
COLOR_BBOX        = (0, 255, 0)      # green
COLOR_MASK        = (50, 200, 255)   # warm cyan (good visibility on US)
COLOR_CONTOUR     = (0, 220, 255)    # bright yellow-cyan
COLOR_RISK_LOW    = (80, 200, 80)
COLOR_RISK_MID    = (60, 200, 220)
COLOR_RISK_HIGH   = (60, 80, 230)


def _to_bgr(img: np.ndarray) -> np.ndarray:
    if img.ndim == 2:
        return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    if img.shape[2] == 4:
        return cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
    return img.copy()


def encode_png_b64(bgr: np.ndarray) -> Optional[str]:
    ok, png = cv2.imencode(".png", bgr)
    if not ok:
        return None
    return base64.b64encode(png.tobytes()).decode("ascii")


# ---------------------------------------------------------------------------
# Layers
# ---------------------------------------------------------------------------

def draw_bbox(image: np.ndarray,
              bbox: Optional[Tuple[int, int, int, int]],
              color: Tuple[int, int, int] = COLOR_BBOX,
              thickness: int = 2) -> np.ndarray:
    out = _to_bgr(image)
    if bbox is None:
        return out
    x, y, w, h = bbox
    cv2.rectangle(out, (x, y), (x + w, y + h), color, thickness)
    return out


def draw_mask(image: np.ndarray,
              mask: Optional[np.ndarray],
              alpha: float = 0.35,
              color: Tuple[int, int, int] = COLOR_MASK) -> np.ndarray:
    out = _to_bgr(image)
    if mask is None or mask.sum() == 0:
        return out
    color_layer = np.zeros_like(out)
    color_layer[mask > 0] = color
    blended = cv2.addWeighted(out, 1.0, color_layer, alpha, 0)
    # contour for crisp boundary
    cnts, _ = cv2.findContours((mask > 0).astype(np.uint8),
                               cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(blended, cnts, -1, COLOR_CONTOUR, 2)
    return blended


def _risk_color(p_suspicious: float) -> Tuple[int, int, int]:
    if p_suspicious >= 0.55:
        return COLOR_RISK_HIGH
    if p_suspicious >= 0.30:
        return COLOR_RISK_MID
    return COLOR_RISK_LOW


def draw_badge(image: np.ndarray,
               p_suspicious: Optional[float] = None,
               label: Optional[str] = None) -> np.ndarray:
    """Small filled rectangle in the top-left with the risk score."""
    out = _to_bgr(image)
    if p_suspicious is None and label is None:
        return out
    text = label or f"P(suspicious)={p_suspicious:.2f}"
    color = _risk_color(p_suspicious if p_suspicious is not None else 0.0)

    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.55
    th = 1
    (tw, theight), _ = cv2.getTextSize(text, font, scale, th)
    pad = 6
    x0, y0 = 8, 8
    cv2.rectangle(out, (x0, y0), (x0 + tw + 2 * pad, y0 + theight + 2 * pad),
                  color, -1)
    cv2.putText(out, text, (x0 + pad, y0 + theight + pad - 2),
                font, scale, (255, 255, 255), th, cv2.LINE_AA)
    return out


def composite_overlay(image: np.ndarray,
                      mask: Optional[np.ndarray] = None,
                      bbox: Optional[Tuple[int, int, int, int]] = None,
                      p_suspicious: Optional[float] = None,
                      mass_present: bool = True) -> np.ndarray:
    """
    Build the everything-in-one overlay used by the frontend.
    If `mass_present` is False we deliberately skip the bbox/mask so the user
    isn't shown a false-positive marker on a normal-looking image.
    """
    out = _to_bgr(image)
    if mass_present:
        out = draw_mask(out, mask)
        out = draw_bbox(out, bbox)
    out = draw_badge(out, p_suspicious=p_suspicious)
    return out


def composite_b64(image: np.ndarray,
                  mask: Optional[np.ndarray] = None,
                  bbox: Optional[Tuple[int, int, int, int]] = None,
                  p_suspicious: Optional[float] = None,
                  mass_present: bool = True) -> Optional[str]:
    return encode_png_b64(
        composite_overlay(image, mask=mask, bbox=bbox,
                          p_suspicious=p_suspicious, mass_present=mass_present)
    )
