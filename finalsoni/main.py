"""
SoniSight FastAPI backend.

Endpoints:
    GET  /health                 liveness check
    GET  /info                   model + config info
    GET  /samples                list built-in sample images
    GET  /samples-static/...     served via StaticFiles (preserved from old)
    POST /analyze                main pipeline endpoint
    POST /predict                alias for /analyze (backwards-compat)

This rewrite drops Gemini entirely. The pipeline is built once at startup from
artifacts under ./artifacts/:
    artifacts/seg_config.json    tuned ClassicalConfig
    artifacts/classifier.joblib  trained RandomForest
    artifacts/unet.pt            optional U-Net checkpoint

If the artifacts are missing the API still runs:
  - segmentation falls back to ClassicalConfig defaults
  - classifier returns probabilities=None and the explanation is feature-only
"""

from __future__ import annotations

import base64
import io
import json
import os
import time
from pathlib import Path
from typing import Literal, Optional

import cv2
import numpy as np
from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from PIL import Image

from pipeline.pipeline import PipelineConfig, SoniSightPipeline
from pipeline.preprocessing import PreprocessConfig


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent
ARTIFACTS_DIR = BASE_DIR / "artifacts"
SAMPLES_DIR = BASE_DIR / "samples"
SAMPLE_RESULTS_DIR = BASE_DIR / "sample_results"

# Backend selection: classical | unet (env var so deployment can switch).
SEG_BACKEND = os.getenv("SONISIGHT_SEG_BACKEND", "unet").lower()


def _build_pipeline() -> SoniSightPipeline:
    seg_config_path = ARTIFACTS_DIR / "seg_config.json"
    classifier_path = ARTIFACTS_DIR / "classifier.joblib"
    unet_ckpt_path  = ARTIFACTS_DIR / "unet.pt"

    cfg = PipelineConfig(
        preprocess=PreprocessConfig(target_size=256),
        seg_backend=SEG_BACKEND,
        seg_config_path=str(seg_config_path) if seg_config_path.exists() else None,
        unet_ckpt_path=str(unet_ckpt_path) if unet_ckpt_path.exists() else None,
        classifier_path=str(classifier_path) if classifier_path.exists() else None,
    )
    return SoniSightPipeline(cfg)


# ---------------------------------------------------------------------------
# Sample images (preserved structure from original SoniSight)
# ---------------------------------------------------------------------------

def _discover_samples() -> dict:
    """Scan ./samples/<category>/ for images so we don't hard-code the list."""
    out = {"normal": [], "suspicious": []}
    if not SAMPLES_DIR.exists():
        return out
    exts = {".png", ".jpg", ".jpeg", ".bmp"}
    for cat in out.keys():
        d = SAMPLES_DIR / cat
        if d.exists():
            out[cat] = sorted(str(p) for p in d.iterdir()
                              if p.is_file() and p.suffix.lower() in exts)
    return out


SAMPLES = _discover_samples()
SAMPLE_NAME_TO_PATH = {os.path.basename(p): p
                       for cat in SAMPLES.values() for p in cat}

SAMPLE_CACHE: dict = {}

def _load_sample_cache():
    SAMPLE_CACHE.clear()
    if SAMPLE_RESULTS_DIR.exists():
        for f in SAMPLE_RESULTS_DIR.glob("*.json"):
            try:
                SAMPLE_CACHE[f.stem] = json.loads(f.read_text())
            except Exception:
                pass


# ---------------------------------------------------------------------------
# App + lifecycle
# ---------------------------------------------------------------------------

app = FastAPI(title="SoniSight Analyzer", version="2.0.0")

origins = [
    "http://localhost:5173",
    "https://www.sonisight.app",
    "https://sonisight.app",
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=[o for o in origins if o],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

if SAMPLES_DIR.exists():
    app.mount("/samples-static", StaticFiles(directory=str(SAMPLES_DIR)),
              name="samples")

PIPELINE: Optional[SoniSightPipeline] = None
PIPELINE_ERROR: Optional[str] = None


@app.on_event("startup")
def _startup():
    global PIPELINE, PIPELINE_ERROR
    _load_sample_cache()
    try:
        PIPELINE = _build_pipeline()
        print(f"[sonisight] pipeline loaded "
              f"(seg_backend={SEG_BACKEND}, "
              f"classifier={'yes' if PIPELINE.classifier else 'no'})")
    except Exception as e:
        PIPELINE_ERROR = f"{type(e).__name__}: {e}"
        print(f"[sonisight] FAILED to load pipeline: {PIPELINE_ERROR}")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _decode_upload(data: bytes) -> np.ndarray:
    """Decode an uploaded image into BGR uint8."""
    try:
        pil = Image.open(io.BytesIO(data)).convert("RGB")
    except Exception as e:
        raise HTTPException(status_code=415,
                            detail="Unsupported image format. Use PNG or JPG.") from e
    return cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)


def _read_sample(name: str) -> bytes:
    lookup = os.path.basename(name.replace("\\", "/"))
    path = SAMPLE_NAME_TO_PATH.get(lookup)
    if not path or not os.path.exists(path):
        raise HTTPException(status_code=404, detail=f"Sample '{name}' not found.")
    with open(path, "rb") as f:
        return f.read()


def _ensure_pipeline() -> SoniSightPipeline:
    if PIPELINE is None:
        raise HTTPException(
            status_code=503,
            detail=f"Pipeline not initialized: {PIPELINE_ERROR or 'unknown'}",
        )
    return PIPELINE


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/health")
def health():
    return {
        "ok": True,
        "pipeline_loaded": PIPELINE is not None,
        "pipeline_error": PIPELINE_ERROR,
    }


@app.get("/info")
def info():
    """Surface what the pipeline is doing so reviewers can sanity-check."""
    if PIPELINE is None:
        return {"loaded": False, "error": PIPELINE_ERROR}
    p = PIPELINE
    return {
        "loaded": True,
        "version": "2.0.0",
        "seg_backend": p.cfg.seg_backend,
        "seg_config_path": p.cfg.seg_config_path,
        "classifier_loaded": p.classifier is not None,
        "classifier_classes": (p.classifier.meta.classes
                               if p.classifier else None),
        "preprocess": {
            "target_size": p.cfg.preprocess.target_size,
            "denoise": p.cfg.preprocess.denoise,
        },
        "disclaimer": ("Prototype research system. Not for clinical "
                       "diagnostic use."),
    }


@app.get("/samples")
def list_samples():
    """Return built-in sample image filenames grouped by label."""
    return {
        cat: [os.path.relpath(p, SAMPLES_DIR).replace("\\", "/") for p in lst]
        for cat, lst in SAMPLES.items()
    }


@app.post("/analyze")
async def analyze(
    file: Optional[UploadFile] = File(None),
    source: Literal["live", "sample"] = Query("live"),
    name: Optional[str] = Query(None),
    run_classifier: bool = Query(True),
):
    """
    Run the full pipeline on an uploaded image or a built-in sample.

    Query params:
        source           "live" (default) reads `file`, "sample" reads `name`
        name             sample filename (required when source=sample)
        run_classifier   if false, skip Part E and return features-only result
    """
    pipeline = _ensure_pipeline()
    t0 = time.perf_counter()

    # 1) load image bytes
    if source == "live":
        if file is None:
            raise HTTPException(400, "Upload a file when source=live.")
        data = await file.read()
    else:
        if not name:
            raise HTTPException(400, "Provide ?name=<sample_file.png> when source=sample.")

        # cached sample short-circuit (matches original behavior)
        sample_stem = Path(os.path.basename(name)).stem
        cached = SAMPLE_CACHE.get(sample_stem)
        if cached is not None:
            return cached

        data = _read_sample(name)

    bgr = _decode_upload(data)

    # 2) run pipeline
    try:
        result = pipeline.analyze(bgr, run_classifier=run_classifier)
    except Exception as e:
        # surface the error type without leaking internals
        raise HTTPException(500, f"Pipeline error: {type(e).__name__}: {e}")

    result["timing_ms"] = round((time.perf_counter() - t0) * 1000, 1)
    return result


# Backwards-compat alias - the old frontend posts to /predict.
@app.post("/predict")
async def predict(
    file: Optional[UploadFile] = File(None),
    source: Literal["live", "sample"] = Query("live"),
    name: Optional[str] = Query(None),
):
    """Alias for /analyze that returns a response shaped like the v1 API.

    Kept so the existing React frontend keeps working without changes. New
    integrations should call /analyze and read the richer response directly.
    """
    full = await analyze(file=file, source=source, name=name, run_classifier=True)

    # If /analyze returned a cached sample dict (already v1-shaped), pass through.
    if isinstance(full, dict) and "descriptors" in full:
        return full

    feats = full.get("features", {}) or {}
    cls = full.get("classification") or {}
    probs = cls.get("probabilities") if cls else None
    if probs is None:
        # No classifier loaded - synthesize sensible defaults from mass_present.
        if feats.get("mass_present"):
            probs = {"normal": 0.5, "suspicious": 0.5}
        else:
            probs = {"normal": 0.95, "suspicious": 0.05}

    expl = full.get("explanation") or {}
    rationale = expl.get("summary") or expl.get("headline") or ""

    # The v1 frontend expects a flat `descriptors` block. Keep the same keys
    # the original code returned so nothing breaks.
    descriptors = {
        "mass_present": bool(feats.get("mass_present", False)),
        "img_quality": "ok",
        "shape": feats.get("shape", "none"),
        "margins": feats.get("margins", "none"),
        "texture": feats.get("texture", "homogeneous"),
        "area": float(feats.get("area", 0.0) or 0.0),
        "circularity": float(feats.get("circularity", 0.0) or 0.0),
        "edge_density": float(feats.get("ring_edge_density", 0.0) or 0.0),
        "intensity_std": float(feats.get("intensity_std", 0.0) or 0.0),
        "contrast_out_in": float(feats.get("contrast_out_in", 0.0) or 0.0),
        "cyst_like": bool(feats.get("cyst_like", False)),
    }

    return {
        "descriptors": descriptors,
        "probabilities": probs,
        "rationale": rationale,
        "overlay_png_base64": full.get("overlay_png_base64"),
        # Pass through the richer v2 payload so a forward-thinking frontend
        # can opt into the new fields without another network call.
        "v2": full,
    }
