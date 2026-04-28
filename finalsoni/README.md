# SoniSight v2

Breast ultrasound analysis — modular CV/ML pipeline with a FastAPI backend.

This is a rebuild of the original SoniSight prototype. The first version
relied on Gemini for both lesion localization and prediction, which made it
hard to evaluate, hard to reproduce, and indefensible in front of clinicians.
v2 replaces all LLM-driven decisions with a real, mask-evaluable computer
vision pipeline.

> **Disclaimer.** This is a research prototype. Output is a software-derived
> pattern summary and is not a clinical diagnosis. Not for diagnostic use.

---

## What changed from v1

| Stage | v1 | v2 |
| --- | --- | --- |
| Lesion ROI | Gemini bounding-box prompt | Mask-tuned classical CV (default) or U-Net (optional) |
| Classification | Gemini prompt over descriptors | Random Forest on handcrafted features |
| Explanation | Gemini-written rationale | Deterministic, feature-grounded text |
| Evaluation | None | Dice / IoU / bbox-IoU on held-out test split |
| Reproducibility | Depends on the model behind the API | Self-contained; deterministic seeds |

---

## Repository layout

```
sonisight/
├── main.py                      # FastAPI app
├── pipeline/
│   ├── preprocessing.py         # Part B
│   ├── segmentation.py          # Part C — classical + optional UNet wrapper
│   ├── features.py              # Part D — handcrafted features
│   ├── classifier.py            # Part E — RandomForest wrapper
│   ├── explain.py               # Part F — grounded explanation
│   ├── visualization.py         # overlay rendering
│   └── pipeline.py              # end-to-end orchestrator
├── training/
│   ├── dataset.py               # BUSI loader + splits
│   ├── evaluate_segmentation.py # Dice/IoU eval + grid search
│   ├── unet.py                  # optional U-Net + train loop
│   └── train_classifier.py      # feature extraction + RF training
├── scripts/
│   ├── prepare_data.py
│   ├── tune_segmentation.py
│   ├── train_classifier.py
│   └── train_all.py             # runs all three in order
├── artifacts/                   # saved configs / models / metrics
└── samples/                     # built-in demo images for the frontend
```

---

## Install

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
# Optional, for the U-Net backend:
pip install torch
```

---

## End-to-end training (BUSI)

Place the dataset at any path (the loader uses your exact filename pattern,
e.g. `malignant (1).png` and `malignant (1)_mask.png`):

```
Dataset_BUSI_with_GT/
    normal/
    benign/        (optional — loader handles its absence)
    malignant/
```

One-shot:

```bash
python -m scripts.train_all --busi /path/to/Dataset_BUSI_with_GT
```

That runs three steps. You can also run them individually:

```bash
# 1) Stratified, seeded 70/15/15 split saved to artifacts/splits.json.
python -m scripts.prepare_data \
    --busi /path/to/Dataset_BUSI_with_GT \
    --out artifacts/splits.json

# 2) Mask-tuned classical segmentation. Sweeps a small parameter grid on
#    train+val, picks best mean Dice, evaluates on the test split.
python -m scripts.tune_segmentation \
    --splits artifacts/splits.json \
    --out artifacts/seg_config.json \
    --report artifacts/seg_report.json

# 3) Train Random Forest on handcrafted features extracted from
#    ground-truth masks. Reports test-set metrics with BOTH the GT mask and
#    the predicted mask so the train/test mismatch is visible.
python -m scripts.train_classifier \
    --busi /path/to/Dataset_BUSI_with_GT \
    --splits artifacts/splits.json \
    --seg_config artifacts/seg_config.json \
    --out artifacts
```

Optional U-Net backend (recommended once classical baseline is reported):

```bash
python -m training.unet \
    --busi /path/to/Dataset_BUSI_with_GT \
    --splits artifacts/splits.json \
    --out artifacts/unet.pt \
    --epochs 50 --batch 8
```

Then start the API with the U-Net backend:

```bash
SONISIGHT_SEG_BACKEND=unet uvicorn main:app --reload
```

---

## Running the API

```bash
uvicorn main:app --reload --port 8000
```

Endpoints:

| Method | Path | Purpose |
| --- | --- | --- |
| GET  | `/health` | Liveness + pipeline status |
| GET  | `/info`   | Pipeline configuration summary |
| GET  | `/samples` | Built-in sample filenames |
| POST | `/analyze` | Main pipeline endpoint |
| POST | `/predict` | v1-shaped alias for the existing React frontend |

The pipeline still runs if any of the artifacts (`seg_config.json`,
`classifier.joblib`, `unet.pt`) are missing — segmentation falls back to
ClassicalConfig defaults and the classifier section returns null.

### `/analyze` request

`multipart/form-data` with one of:

```
POST /analyze?source=live                  # body: file=<image upload>
POST /analyze?source=sample&name=<file>    # use built-in sample
```

### `/analyze` response (abridged)

```json
{
  "preprocess": { "target_size": 256, "denoise": "median",
                  "crop_xyxy": [12, 8, 488, 412], "orig_shape": [420, 500] },
  "segmentation": {
    "backend": "classical",
    "found_mass": true,
    "bbox_orig": [128, 142, 96, 84],
    "confidence": 0.71
  },
  "features": {
    "area": 6312.0, "perimeter": 312.4,
    "circularity": 0.61, "solidity": 0.88,
    "ring_edge_density": 0.21, "contrast_out_in": 0.27,
    "intensity_std": 26.4, "glcm_contrast": 7.92, ...
    "mass_present": true, "shape": "oval",
    "margins": "lobulated", "texture": "mixed", "cyst_like": false
  },
  "classification": {
    "probabilities": { "normal": 0.18, "suspicious": 0.82 },
    "predicted_label": "suspicious", "model_loaded": true
  },
  "explanation": {
    "headline": "Region flagged as higher-risk pattern (P(suspicious)=0.82).",
    "summary": "An oval region with lobulated/spiculated margins ...",
    "feature_highlights": ["Region: oval (circularity 0.61).", "..."],
    "limitations": "This is a prototype research system ...",
    "drivers": [
      {"name": "ring_edge_density", "value": 0.21, "importance": 0.14},
      ...
    ]
  },
  "overlay_png_base64": "iVBORw0KGgo..."
}
```

The `overlay_png_base64` is a PNG of the original image with the predicted
segmentation mask, bounding box, and a risk badge overlaid.

---

## Design choices (the parts worth defending)

**Mask-tuned classical CV as default.**
Every parameter of the classical segmenter is exposed in `ClassicalConfig`.
`scripts/tune_segmentation.py` sweeps a small grid of these parameters on the
train+val split, picks the highest mean Dice configuration, then reports
test-set metrics. This makes "we tuned classical CV against real masks"
reproducible and inspectable, which is far more defensible than hand-tuned
thresholds. The chosen config is persisted to `artifacts/seg_config.json`.

**U-Net as an opt-in upgrade, not a replacement.**
A small U-Net (4 down/up blocks, BCE+Dice loss) is included and easy to train
on BUSI, but the classical path is always available. Doctors and reviewers
can interrogate the classical pipeline; they cannot meaningfully interrogate a
black-box network. Both backends conform to the same `SegmentationOutput`
interface, so swapping is a config change.

**Random Forest, not Logistic Regression or XGBoost.**
With ~780 BUSI samples and ~30 handcrafted features, RF wins on:
- non-linear interactions (circularity matters most when contrast is high)
- no scaling needed, robust to outliers
- exposes `feature_importances_` for explanation
LR underfits these interactions; XGBoost works but adds a heavier dep for no
measurable gain at this scale.

**Honest train/test mismatch reporting.**
The classifier is trained on features computed from **ground-truth masks**,
but at inference time it sees features from **predicted masks**. We report
both numbers in `artifacts/classifier_report.json` so you can see exactly how
much accuracy the segmenter is costing you.

**No-mass short-circuit.**
If the segmenter doesn't find a region, the classifier returns
`{normal: 0.95, suspicious: 0.05}` rather than running on a zero feature
vector. This avoids fabricating a probability from no signal.

**Deterministic explanation.**
`pipeline/explain.py` is a pure function over `(features, classification,
feature_importances)`. Every numeric value cited in the explanation is
present in those inputs. There is no LLM in the inference path. This was the
single biggest fragility in v1.

---

## Known limitations (be ready for these in review)

- BUSI is small (~780 images) and from a single institution. Cross-institution
  generalization is not demonstrated.
- The classical segmenter's mean Dice on lesion-bearing test samples is
  bounded by the saliency+morphology approach. Expect 0.45–0.60 with a tuned
  config; U-Net typically reaches 0.70–0.80.
- The classifier sees predicted-mask features at inference, GT-mask features
  at training. The reported test number with predicted masks is the honest
  one to quote.
- Normal-class images have empty masks. The segmenter's specificity on these
  is reported separately (fraction of normal images on which the segmenter
  correctly produced no prediction).
- "Normal vs suspicious" is binary by design. A 3-class run (normal / benign
  / malignant) is supported via `--three_class` on the classifier trainer.

---

## Frontend integration

The existing React frontend should keep working unchanged because `/predict`
is preserved as a v1-shaped alias. New features (segmentation overlay,
feature highlights, drivers) are accessible either by:

- reading the `v2` field on the `/predict` response, or
- migrating the call to `/analyze`.

`overlay_png_base64` is the same field name as v1; just paint it where you
already do.
