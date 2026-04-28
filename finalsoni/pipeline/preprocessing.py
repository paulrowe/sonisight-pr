"""
Ultrasound image preprocessing.

Pipeline:
  BGR/RGB/grayscale  ->  grayscale
  border crop (removes obvious black/letterbox margins)
  resize to fixed size (default 256x256, configurable)
  speckle denoise (Non-Local Means or median, configurable)
  CLAHE contrast normalization
  optional intensity normalization to [0, 1] float

The output is always a uint8 grayscale image at the requested size unless
`return_float=True`, in which case it's float32 in [0, 1].

Design notes:
- Ultrasound speckle is multiplicative noise. Median filtering is fast and
  preserves edges reasonably; fastNlMeansDenoising is slower but noticeably
  cleaner for downstream segmentation. We default to median for the live
  endpoint and offer NLM for offline/batch eval.
- CLAHE (clipLimit=2.0, tileGridSize=8x8) is the standard breast-US contrast
  recipe used in several BUSI papers and matches the original SoniSight code.
- Border cleanup removes 5% margins by default. Real US machines often add a
  black border, ruler, or text label that confuses thresholding. We also do an
  intensity-based crop that trims any nearly-pure-black rim.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import cv2
import numpy as np


@dataclass
class PreprocessConfig:
    """Tunable preprocessing parameters. Persist this with the model."""
    target_size: int = 256                   # square resize target
    border_crop_frac: float = 0.03           # fraction trimmed from each side
    intensity_border_thresh: int = 8         # black-rim trim threshold (0..255)
    denoise: str = "median"                  # "median" | "nlm" | "none"
    median_ksize: int = 3
    nlm_h: int = 7                           # filter strength for fastNlMeans
    clahe_clip: float = 2.0
    clahe_tile: int = 8
    normalize: bool = False                  # if True returns float32 [0, 1]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _normalize01(x: np.ndarray) -> np.ndarray:
    """Min-max normalize to [0, 1] as float32. Preserved from original code."""
    x = x.astype(np.float32)
    return (x - x.min()) / (x.max() - x.min() + 1e-6)


def _to_grayscale(img: np.ndarray) -> np.ndarray:
    """Accept any of: gray (H,W), BGR (H,W,3), BGRA (H,W,4). Return uint8 (H,W)."""
    if img.ndim == 2:
        gray = img
    elif img.ndim == 3 and img.shape[2] == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    elif img.ndim == 3 and img.shape[2] == 4:
        gray = cv2.cvtColor(img, cv2.COLOR_BGRA2GRAY)
    else:
        raise ValueError(f"Unsupported image shape: {img.shape}")
    if gray.dtype != np.uint8:
        # rescale to uint8 if we got 16-bit or float
        gray = cv2.normalize(gray, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    return gray


def _crop_borders(gray: np.ndarray, cfg: PreprocessConfig) -> Tuple[np.ndarray, Tuple[int, int, int, int]]:
    """
    Two-stage border cleanup:
      1) Trim a fixed fraction from every side (gets rid of tiny scanner-UI rims).
      2) Trim any remaining nearly-black rim by row/column intensity.

    Returns (cropped, (x0, y0, x1, y1)) so we can later map predictions back
    to the original image coordinate system.
    """
    h, w = gray.shape
    m = int(round(cfg.border_crop_frac * min(h, w)))
    if h > 2 * m and w > 2 * m:
        x0, y0, x1, y1 = m, m, w - m, h - m
        gray = gray[y0:y1, x0:x1]
    else:
        x0, y0, x1, y1 = 0, 0, w, h

    # Intensity-based trim: shave rows/cols whose mean intensity is below
    # `intensity_border_thresh`. This handles the dark letterbox added by many
    # ultrasound machines.
    t = cfg.intensity_border_thresh
    row_mean = gray.mean(axis=1)
    col_mean = gray.mean(axis=0)
    keep_rows = np.where(row_mean > t)[0]
    keep_cols = np.where(col_mean > t)[0]
    if keep_rows.size > 0 and keep_cols.size > 0:
        r0, r1 = int(keep_rows[0]), int(keep_rows[-1]) + 1
        c0, c1 = int(keep_cols[0]), int(keep_cols[-1]) + 1
        # Only apply if it actually trims something meaningful (>=2% of the dim).
        if (r1 - r0) >= 0.5 * gray.shape[0] and (c1 - c0) >= 0.5 * gray.shape[1]:
            gray = gray[r0:r1, c0:c1]
            x0 += c0
            y0 += r0
            x1 = x0 + (c1 - c0)
            y1 = y0 + (r1 - r0)

    return gray, (x0, y0, x1, y1)


def _denoise(gray: np.ndarray, cfg: PreprocessConfig) -> np.ndarray:
    if cfg.denoise == "median":
        return cv2.medianBlur(gray, cfg.median_ksize)
    if cfg.denoise == "nlm":
        # h ~7-10 is a good range for ultrasound speckle without over-smoothing.
        return cv2.fastNlMeansDenoising(gray, None, h=cfg.nlm_h,
                                        templateWindowSize=7, searchWindowSize=21)
    return gray


def _clahe(gray: np.ndarray, cfg: PreprocessConfig) -> np.ndarray:
    clahe = cv2.createCLAHE(clipLimit=cfg.clahe_clip,
                            tileGridSize=(cfg.clahe_tile, cfg.clahe_tile))
    return clahe.apply(gray)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def preprocess(image: np.ndarray,
               cfg: Optional[PreprocessConfig] = None,
               return_float: bool = False) -> np.ndarray:
    """
    Run the standard preprocessing chain and return a single grayscale array
    at `cfg.target_size x cfg.target_size`. Same name as the original
    `preprocess` function so callers don't break.
    """
    cfg = cfg or PreprocessConfig()
    gray = _to_grayscale(image)
    gray, _ = _crop_borders(gray, cfg)
    gray = cv2.resize(gray, (cfg.target_size, cfg.target_size),
                      interpolation=cv2.INTER_AREA)
    gray = _denoise(gray, cfg)
    gray = _clahe(gray, cfg)
    if return_float or cfg.normalize:
        return _normalize01(gray)
    return gray


def preprocess_with_meta(image: np.ndarray,
                         cfg: Optional[PreprocessConfig] = None
                         ) -> Tuple[np.ndarray, dict]:
    """
    Like `preprocess` but also returns metadata describing the geometric
    transform applied. Inference code uses this to map predicted masks/boxes
    back to original-image coordinates.
    """
    cfg = cfg or PreprocessConfig()
    h0, w0 = image.shape[:2]
    gray = _to_grayscale(image)
    cropped, (x0, y0, x1, y1) = _crop_borders(gray, cfg)
    cropped_h, cropped_w = cropped.shape

    resized = cv2.resize(cropped, (cfg.target_size, cfg.target_size),
                         interpolation=cv2.INTER_AREA)
    out = _denoise(resized, cfg)
    out = _clahe(out, cfg)
    if cfg.normalize:
        out = _normalize01(out)

    meta = {
        "orig_shape": (h0, w0),
        "crop_xyxy": (x0, y0, x1, y1),
        "cropped_shape": (cropped_h, cropped_w),
        "resized_shape": (cfg.target_size, cfg.target_size),
        "scale_x": cropped_w / cfg.target_size,
        "scale_y": cropped_h / cfg.target_size,
    }
    return out, meta


def map_mask_to_original(mask_resized: np.ndarray, meta: dict) -> np.ndarray:
    """
    Convert a binary mask predicted on the preprocessed (resized, cropped)
    image back into the original image coordinate system. Used for overlays.
    """
    h0, w0 = meta["orig_shape"]
    x0, y0, x1, y1 = meta["crop_xyxy"]
    crop_h, crop_w = meta["cropped_shape"]

    if mask_resized.dtype != np.uint8:
        mask_resized = (mask_resized > 0).astype(np.uint8) * 255

    cropped = cv2.resize(mask_resized, (crop_w, crop_h),
                         interpolation=cv2.INTER_NEAREST)
    full = np.zeros((h0, w0), dtype=np.uint8)
    full[y0:y1, x0:x1] = cropped
    return full
