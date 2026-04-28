"""
Lesion localization / segmentation.

Two backends, both evaluable against ground-truth masks:

    1. ClassicalSegmenter  - intensity + saliency + morphology + contour scoring.
                             All parameters are exposed in `ClassicalConfig`
                             so they can be tuned on a training set against
                             real masks (see training/evaluate_segmentation.py).

    2. UNetSegmenter       - thin wrapper around training.unet.UNet. Loaded
                             lazily so the rest of the system runs without
                             PyTorch installed.

Both backends conform to the same interface:

    seg = SegmenterBackend.load(...)
    out = seg.predict(gray)                        # gray: HxW uint8
    out = SegmentationOutput(
        mask:   np.ndarray,    # HxW uint8, 0 or 255
        bbox:   tuple|None,    # (x, y, w, h) in `gray`'s coords, or None
        contour: np.ndarray|None,
        score:  float,         # backend-specific confidence in [0, 1]
        meta:   dict,
    )

Why this split?
- Classical CV is fast, transparent, and runs anywhere. With mask-tuned
  parameters it's a respectable baseline (~0.45-0.55 Dice on BUSI in our runs)
  and crucially, it lets reviewers see exactly what knobs were turned.
- U-Net trained on BUSI typically gets 0.70-0.80 Dice. We provide it as an
  upgrade path but not as the only option, because doctors and reviewers can
  always inspect the classical pipeline's reasoning.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np

from .preprocessing import _normalize01


# ---------------------------------------------------------------------------
# Common output type
# ---------------------------------------------------------------------------

@dataclass
class SegmentationOutput:
    mask: np.ndarray
    bbox: Optional[Tuple[int, int, int, int]]
    contour: Optional[np.ndarray]
    score: float
    meta: dict = field(default_factory=dict)


# ===========================================================================
# 1) Classical, mask-tunable segmenter
# ===========================================================================

@dataclass
class ClassicalConfig:
    """
    Knobs for the classical pipeline. Defaults are reasonable starting points;
    the trainer in evaluate_segmentation.py can override them by sweeping on
    the train split and saving the best Dice config to artifacts/seg_config.json
    """
    # Saliency mixture weights (sum doesn't have to be 1; we normalize after).
    w_dark: float = 0.65         # how much to weight "darkness" channel
    w_log:  float = 0.25         # Laplacian-of-Gaussian (blob-ness)
    w_edge: float = 0.10         # Canny edges (smoothed)

    # Adaptive thresholding percentile for saliency map.
    sal_percentile: int = 92     # higher = stricter

    # Morphological cleanup kernel sizes (pixel).
    open_ksize: int = 5
    close_ksize: int = 7
    open_iters: int = 1
    close_iters: int = 1

    # Candidate filtering.
    min_area_frac: float = 0.005  # ignore blobs < 0.5% of image area
    max_area_frac: float = 0.55   # and > 55% (probably background)
    min_solidity: float = 0.30
    max_aspect_ratio: float = 4.5
    border_margin: int = 2        # px from border to count as "touching"

    # Lesion-specific contrast gate. Lesions are usually hypoechoic (darker
    # than surrounding tissue). require at least this much contrast_out_in.
    min_contrast_out_in: float = 0.04

    # Scoring weights (used to pick the BEST candidate when several pass).
    s_circularity: float = 1.0
    s_contrast: float = 1.4
    s_solidity: float = 0.5
    s_edge_penalty: float = 0.6


class ClassicalSegmenter:
    """
    Multi-stage segmentation:
      1. Build a saliency map: w_dark*dark + w_log*log + w_edge*edges.
      2. Threshold at `sal_percentile`-th percentile.
      3. Morphological open then close.
      4. Find external contours.
      5. Filter on area, solidity, aspect ratio, border, contrast.
      6. Score remaining candidates and pick the best.
    """

    def __init__(self, cfg: Optional[ClassicalConfig] = None):
        self.cfg = cfg or ClassicalConfig()

    # ---- params io ---------------------------------------------------------

    @classmethod
    def load(cls, config_path: Optional[str | Path] = None) -> "ClassicalSegmenter":
        if config_path is None:
            return cls()
        config_path = Path(config_path)
        if not config_path.exists():
            return cls()
        with config_path.open() as f:
            data = json.load(f)
        return cls(ClassicalConfig(**data))

    def save(self, config_path: str | Path) -> None:
        Path(config_path).write_text(json.dumps(asdict(self.cfg), indent=2))

    # ---- core --------------------------------------------------------------

    def _build_saliency(self, gray: np.ndarray) -> np.ndarray:
        cfg = self.cfg
        dark = 1.0 - _normalize01(gray.astype(np.float32))

        g_blur = cv2.GaussianBlur(gray, (0, 0), 1.2)
        log = _normalize01(np.abs(cv2.Laplacian(g_blur, cv2.CV_32F, ksize=3)))

        edges = cv2.Canny(gray, 30, 100).astype(np.float32)
        edges = _normalize01(cv2.GaussianBlur(edges, (5, 5), 0))

        S = cfg.w_dark * dark + cfg.w_log * log + cfg.w_edge * edges
        return _normalize01(S)

    def _propose_mask(self, gray: np.ndarray) -> np.ndarray:
        cfg = self.cfg
        S = self._build_saliency(gray)
        T = float(np.percentile(S, cfg.sal_percentile))
        m = (S >= T).astype(np.uint8) * 255
        if cfg.open_ksize > 1 and cfg.open_iters > 0:
            m = cv2.morphologyEx(m, cv2.MORPH_OPEN,
                                 np.ones((cfg.open_ksize, cfg.open_ksize), np.uint8),
                                 iterations=cfg.open_iters)
        if cfg.close_ksize > 1 and cfg.close_iters > 0:
            m = cv2.morphologyEx(m, cv2.MORPH_CLOSE,
                                 np.ones((cfg.close_ksize, cfg.close_ksize), np.uint8),
                                 iterations=cfg.close_iters)
        return m

    def _score_contour(self, c: np.ndarray, gray: np.ndarray
                       ) -> Tuple[float, Optional[dict]]:
        cfg = self.cfg
        H, W = gray.shape
        A = H * W

        x, y, w, h = cv2.boundingRect(c)
        if (x <= cfg.border_margin or y <= cfg.border_margin or
                (x + w) >= W - cfg.border_margin or (y + h) >= H - cfg.border_margin):
            return -1e9, None

        area = float(cv2.contourArea(c))
        if area <= 0:
            return -1e9, None
        af = area / A
        if af < cfg.min_area_frac or af > cfg.max_area_frac:
            return -1e9, None

        hull = cv2.convexHull(c)
        ha = cv2.contourArea(hull) + 1e-6
        solidity = area / ha
        if solidity < cfg.min_solidity:
            return -1e9, None

        ar = max(w / (h + 1e-6), h / (w + 1e-6))
        if ar > cfg.max_aspect_ratio:
            return -1e9, None

        # contrast inside-vs-surrounding
        m = np.zeros((H, W), np.uint8)
        cv2.drawContours(m, [c], -1, 255, -1)
        ring = cv2.dilate(m, np.ones((7, 7), np.uint8))
        ring = cv2.subtract(ring, m)
        inside = gray[m > 0]
        outside = gray[ring > 0]
        if inside.size == 0 or outside.size == 0:
            return -1e9, None
        mean_in = float(inside.mean())
        mean_out = float(outside.mean())
        contrast_out_in = max(0.0, (mean_out - mean_in) / (mean_out + 1e-6))
        if contrast_out_in < cfg.min_contrast_out_in:
            return -1e9, None

        # margin/edge density
        edges = cv2.Canny(gray, 30, 100)
        ring_edges = float((edges[ring > 0] > 0).mean()) if ring.sum() > 0 else 0.0

        # circularity
        perim = cv2.arcLength(c, True)
        circ = (4.0 * np.pi * area) / (perim * perim + 1e-6)

        score = (cfg.s_circularity * circ
                 + cfg.s_contrast * contrast_out_in
                 + cfg.s_solidity * solidity
                 - cfg.s_edge_penalty * max(0.0, ring_edges - 0.25))

        return score, {
            "area": area, "area_frac": af, "solidity": solidity,
            "aspect_ratio": ar, "contrast_out_in": contrast_out_in,
            "ring_edges": ring_edges, "circularity": circ,
        }

    def predict(self, gray: np.ndarray) -> SegmentationOutput:
        if gray.ndim != 2:
            raise ValueError(f"ClassicalSegmenter expects grayscale, got {gray.shape}")

        proposal = self._propose_mask(gray)
        cnts, _ = cv2.findContours(proposal, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        best_score = -1e9
        best_c: Optional[np.ndarray] = None
        best_meta: Optional[dict] = None
        for c in cnts:
            s, m = self._score_contour(c, gray)
            if s > best_score:
                best_score, best_c, best_meta = s, c, m

        H, W = gray.shape
        out_mask = np.zeros((H, W), dtype=np.uint8)
        if best_c is None or best_score <= -1e8:
            return SegmentationOutput(
                mask=out_mask, bbox=None, contour=None, score=0.0,
                meta={"backend": "classical", "found": False},
            )

        cv2.drawContours(out_mask, [best_c], -1, 255, -1)
        x, y, w, h = cv2.boundingRect(best_c)

        # Map raw score into a soft [0, 1] confidence. Squash with a sigmoid-ish
        # transform; this is *not* a probability, just a relative ranking signal.
        conf = float(1.0 / (1.0 + np.exp(-(best_score - 1.0))))

        meta = {"backend": "classical", "found": True, **(best_meta or {})}
        return SegmentationOutput(mask=out_mask, bbox=(int(x), int(y), int(w), int(h)),
                                  contour=best_c, score=conf, meta=meta)


# ===========================================================================
# 2) U-Net wrapper (optional, lazy import)
# ===========================================================================

class UNetSegmenter:
    """
    Wraps a trained U-Net checkpoint. Imports torch lazily so the package can
    be used without PyTorch installed (classical path always works).
    """

    def __init__(self, model, device: str, threshold: float = 0.5):
        self.model = model
        self.device = device
        self.threshold = threshold

    @classmethod
    def load(cls, ckpt_path: str | Path,
             device: Optional[str] = None,
             threshold: float = 0.5) -> "UNetSegmenter":
        try:
            import torch
            from training.unet import UNet  # local import - heavy dep
        except Exception as e:
            raise RuntimeError(
                "PyTorch is required to use the U-Net backend. "
                "Install with `pip install torch` or use the classical backend."
            ) from e

        device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        model = UNet(in_channels=1, out_channels=1)
        state = torch.load(str(ckpt_path), map_location=device)
        if isinstance(state, dict) and "state_dict" in state:
            state = state["state_dict"]
        model.load_state_dict(state)
        model.to(device).eval()
        return cls(model=model, device=device, threshold=threshold)

    def predict(self, gray: np.ndarray) -> SegmentationOutput:
        import torch  # safe here - load() already verified torch is available

        if gray.ndim != 2:
            raise ValueError("UNetSegmenter expects grayscale input")
        H, W = gray.shape

        x = gray.astype(np.float32) / 255.0
        x = torch.from_numpy(x)[None, None, :, :].to(self.device)

        with torch.no_grad():
            logits = self.model(x)
            prob = torch.sigmoid(logits)[0, 0].cpu().numpy()

        m = (prob >= self.threshold).astype(np.uint8) * 255
        if m.sum() == 0:
            return SegmentationOutput(
                mask=np.zeros((H, W), np.uint8), bbox=None, contour=None,
                score=float(prob.max()),
                meta={"backend": "unet", "found": False, "max_prob": float(prob.max())},
            )

        # Keep only the largest connected component to avoid scattered specks.
        n, lab, stats, _ = cv2.connectedComponentsWithStats(m, connectivity=8)
        if n <= 1:
            return SegmentationOutput(
                mask=np.zeros((H, W), np.uint8), bbox=None, contour=None,
                score=float(prob.max()),
                meta={"backend": "unet", "found": False},
            )
        biggest = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
        m = (lab == biggest).astype(np.uint8) * 255

        cnts, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contour = max(cnts, key=cv2.contourArea) if cnts else None
        bbox = cv2.boundingRect(contour) if contour is not None else None

        # Confidence = mean probability within predicted mask.
        score = float(prob[m > 0].mean()) if m.sum() else 0.0

        return SegmentationOutput(
            mask=m,
            bbox=tuple(int(v) for v in bbox) if bbox else None,
            contour=contour,
            score=score,
            meta={"backend": "unet", "found": True,
                  "mean_prob": score, "max_prob": float(prob.max())},
        )


# ===========================================================================
# Mask metrics (used by training/evaluate_segmentation.py)
# ===========================================================================

def iou(pred: np.ndarray, gt: np.ndarray) -> float:
    p = (pred > 0).astype(np.uint8)
    g = (gt > 0).astype(np.uint8)
    inter = int(np.logical_and(p, g).sum())
    union = int(np.logical_or(p, g).sum())
    if union == 0:
        return 1.0 if inter == 0 else 0.0
    return inter / union


def dice(pred: np.ndarray, gt: np.ndarray) -> float:
    p = (pred > 0).astype(np.uint8)
    g = (gt > 0).astype(np.uint8)
    s = int(p.sum() + g.sum())
    if s == 0:
        return 1.0
    inter = int(np.logical_and(p, g).sum())
    return 2.0 * inter / s


def bbox_from_mask(mask: np.ndarray) -> Optional[Tuple[int, int, int, int]]:
    """Tight bounding box around non-zero pixels of a binary mask."""
    if mask.sum() == 0:
        return None
    ys, xs = np.where(mask > 0)
    x0, x1 = int(xs.min()), int(xs.max())
    y0, y1 = int(ys.min()), int(ys.max())
    return (x0, y0, x1 - x0 + 1, y1 - y0 + 1)


def bbox_iou(b1: Tuple[int, int, int, int],
             b2: Tuple[int, int, int, int]) -> float:
    x1, y1, w1, h1 = b1
    x2, y2, w2, h2 = b2
    xa, ya = max(x1, x2), max(y1, y2)
    xb, yb = min(x1 + w1, x2 + w2), min(y1 + h1, y2 + h2)
    iw, ih = max(0, xb - xa), max(0, yb - ya)
    inter = iw * ih
    union = w1 * h1 + w2 * h2 - inter
    return inter / union if union > 0 else 0.0
