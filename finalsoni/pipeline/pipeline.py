"""
End-to-end inference pipeline.

This is the single function the FastAPI endpoint calls. Splitting the
orchestration here (instead of in main.py) lets us unit-test it directly and
keeps the web layer thin.

Stages:
    raw image
      -> preprocess (resized + denoised + CLAHE grayscale, plus geometric meta)
      -> segment (classical or u-net)  -> binary mask + bbox in *resized* coords
      -> map mask to original-image coords for overlays
      -> extract_features(resized_gray, resized_mask)
      -> classifier.predict_from_features(features)
      -> generate_explanation(...)
      -> visualization.composite_b64(...)

Notes:
    - All numeric features are computed at the resized scale (256x256 by
      default). Consistency with training is what matters; geometric features
      are scale-dependent and the classifier was trained at the same scale.
    - The bbox we return to the frontend is in ORIGINAL image coordinates.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

import cv2
import numpy as np

from .preprocessing import (PreprocessConfig, map_mask_to_original,
                            preprocess_with_meta)
from .segmentation import (ClassicalSegmenter, SegmentationOutput,
                           UNetSegmenter, bbox_from_mask)
from .features import extract_features
from .classifier import LesionClassifier
from .explain import generate_explanation
from .visualization import composite_b64


# ---------------------------------------------------------------------------
# Resource holder
# ---------------------------------------------------------------------------

@dataclass
class PipelineConfig:
    preprocess: PreprocessConfig
    seg_backend: str = "classical"           # "classical" | "unet"
    seg_config_path: Optional[str] = None    # tuned classical params
    unet_ckpt_path: Optional[str] = None
    unet_threshold: float = 0.5
    classifier_path: Optional[str] = None    # joblib .pkl


class SoniSightPipeline:
    """
    Holds loaded models so we don't reload on every request.
    Construct once at FastAPI startup, call `analyze()` per request.
    """

    def __init__(self, cfg: PipelineConfig):
        self.cfg = cfg
        self.segmenter = self._build_segmenter()
        self.classifier = self._build_classifier()

    def _build_segmenter(self):
        if self.cfg.seg_backend == "unet":
            if not self.cfg.unet_ckpt_path:
                raise ValueError("seg_backend=unet but unet_ckpt_path not set")
            return UNetSegmenter.load(
                self.cfg.unet_ckpt_path, threshold=self.cfg.unet_threshold)
        return ClassicalSegmenter.load(self.cfg.seg_config_path)

    def _build_classifier(self) -> Optional[LesionClassifier]:
        if not self.cfg.classifier_path:
            return None
        path = Path(self.cfg.classifier_path)
        if not path.exists():
            return None
        return LesionClassifier.load(path)

    # ---- main entry point ------------------------------------------------

    def analyze(self, image: np.ndarray, run_classifier: bool = True
                ) -> Dict[str, Any]:
        """
        Args
        ----
        image : np.ndarray
            BGR or grayscale array as decoded from the upload.
        run_classifier : bool
            If False or no classifier loaded, skip Part E and only return
            segmentation + features + a feature-only explanation.

        Returns
        -------
        dict with keys:
            preprocess_meta, segmentation, features,
            classification (or None), explanation,
            overlay_png_base64
        """
        # 1) preprocess
        gray, meta = preprocess_with_meta(image, self.cfg.preprocess)

        # 2) segment (operates on resized gray)
        seg: SegmentationOutput = self.segmenter.predict(gray)

        # 3) features (computed at resized scale, matching training)
        features = extract_features(gray, seg.mask)

        # 4) optional classifier
        classification = None
        importances = None
        if run_classifier and self.classifier is not None:
            classification = self.classifier.predict_from_features(features)
            importances = self.classifier.feature_importances

        # 5) explanation
        p_susp = None
        if classification is not None:
            p_susp = float(classification.get("probabilities", {})
                                          .get("suspicious", 0.0))
        explanation = generate_explanation(
            features=features,
            classification=classification,
            feature_importances=importances,
            segmentation_score=seg.score,
            segmentation_backend=seg.meta.get("backend"),
        )

        # 6) build overlay on original image (so the user sees their image)
        original_bgr = image if image.ndim == 3 else cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        if image.ndim == 3 and image.shape[2] == 3:
            # could be RGB from PIL; assume BGR (caller's responsibility)
            pass
        full_mask = map_mask_to_original(seg.mask, meta)
        full_bbox = bbox_from_mask(full_mask)
        overlay_b64 = composite_b64(
            original_bgr, mask=full_mask, bbox=full_bbox,
            p_suspicious=p_susp,
            mass_present=bool(features.get("mass_present", False)),
        )

        # 7) prepare JSON response
        response = {
            "preprocess": {
                "target_size": self.cfg.preprocess.target_size,
                "denoise": self.cfg.preprocess.denoise,
                "crop_xyxy": meta["crop_xyxy"],
                "orig_shape": meta["orig_shape"],
            },
            "segmentation": {
                "backend": seg.meta.get("backend"),
                "found_mass": bool(features.get("mass_present", False)),
                "bbox_orig": list(full_bbox) if full_bbox else None,
                "bbox_resized": list(seg.bbox) if seg.bbox else None,
                "confidence": float(seg.score),
                "meta": {k: (float(v) if isinstance(v, (int, float)) else v)
                         for k, v in seg.meta.items()
                         if k != "backend"},
            },
            "features": features,
            "classification": classification,
            "explanation": explanation,
            "overlay_png_base64": overlay_b64,
        }
        # Scrub NaN/Inf out of the ENTIRE response, recursively. NaN is not
        # valid JSON and starlette's encoder will raise on it; we replace
        # NaN/Inf with None so the response always serializes.
        return _json_safe(response)


def _json_safe(obj):
    """Recursively make a structure JSON-safe.

    - NaN / +Inf / -Inf  -> None
    - numpy scalar types -> python int/float
    - floats are rounded to 6 decimals for readability
    - dicts and lists are walked
    - everything else is passed through unchanged
    """
    # Numpy scalars first - they're instances of np.floating, not float
    if isinstance(obj, np.floating):
        f = float(obj)
        return None if (np.isnan(f) or np.isinf(f)) else round(f, 6)
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.ndarray):
        return _json_safe(obj.tolist())

    # Plain python types
    if isinstance(obj, float):
        return None if (np.isnan(obj) or np.isinf(obj)) else round(obj, 6)
    if isinstance(obj, dict):
        return {k: _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_json_safe(v) for v in obj]
    return obj
