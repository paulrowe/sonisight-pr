# SoniSight — working context for Claude Code

Read this before touching anything. It carries the state, the findings, and the
rules that make this project's results defensible.

## What this is

SoniSight is a breast ultrasound triage prototype: it takes an ultrasound image
and outputs a binary `normal` vs `suspicious` call with a plain-language
rationale and an ROI overlay. FastAPI backend, React/Vite frontend, Gemini 2.5
Flash for vision, OpenCV for quantitative features. Live at sonisight.app.

It is a **research prototype, not a diagnostic device**. Keep that framing in
docs, README, and UI copy. Do not describe it as "diagnostic."

The current goal is a publishable system paper built around an ablation: does a
hybrid LLM + classical-CV pipeline beat either component alone?

## Dataset

BUSI (Breast Ultrasound Images), at `Dataset_BUSI_with_GT/{normal,benign,malignant}/`.
- ~780 images: 133 normal, 437 benign, 210 malignant
- Each class folder **mixes real images and segmentation masks**. Files named
  `*_mask.png` are masks — always filter them out. Every script does this via
  `"_mask" not in name.lower()`.
- Labels collapse to binary for triage scoring:
  `normal -> normal`, `benign -> suspicious`, `malignant -> suspicious`.
- Gitignored (large). So are all `eval_*.csv` outputs and `fusion_model.json`.

## RESEARCH INTEGRITY RULES — do not violate these

These are the project's most valuable asset. A defensible 84% beats an
indefensible 95%.

1. **`busi_split.json` is the frozen dev/test split** (70/30, stratified, seed 42,
   produced by `make_split.py`). It is committed. Never regenerate it.
2. **All development, debugging, threshold selection, and model fitting happens
   on the `dev` split only.**
3. **The `test` split is scored ONCE, at the very end, after the system is
   frozen.** Never iterate against test numbers. Never "just peek."
4. **Never tune parameters to make a metric go up.** Fix bugs by understanding
   why they are bugs, validated on specific hand-checked images. If a change
   can only be justified by "the aggregate improved," it is overfitting.
5. Threshold/operating point is chosen on dev, then applied unchanged to test.
6. If asked to inflate a number, refuse and explain. This has come up before and
   the answer stands.

## Critical finding from v1 (this drove the whole v2 redesign)

v1 was: Gemini returns a bounding box -> OpenCV segments inside it -> extracts
shape/margins/texture -> hand-written rules produce a probability.

Measured on the dev split (546 images):

| system | AUROC |
|---|---|
| full v1 rule-based pipeline | 0.830 |
| the single bit `mass_present`, alone | 0.830 |
| trained model over all v1 descriptors (5-fold CV) | 0.834 |
| trained model over descriptors *without* `mass_present` | 0.834 |

**The OpenCV + rules stage contributed zero discriminative signal.** At threshold
0.30, the prediction matched "did Gemini return a box?" on 536/546 images (98.2%).

Error decomposition was absolute:
- **All 71 false negatives**: Gemini returned no box on an image that had a lesion
  (45 benign, 26 malignant)
- **All 15 false positives**: Gemini boxed normal tissue

Zero errors came from the classifier. Cause: after relaxing the segmentation
gates to fix a sensitivity bug, the descriptor extractor became degenerate — it
labeled 348/372 boxed cases "spiculated" and 299/372 "irregular," and mean
circularity was 0.398 (suspicious) vs 0.395 (normal). A near-constant feature
carries no information.

**Implication: 100% of the error budget is detection, not classification.**

## v2 architecture (built, not yet validated on real data)

Detection — attacks the 71 false negatives, in cascade:
1. `assess_image()` — structured JSON from Gemini (lesion presence, bbox, shape,
   margins, echogenicity, orientation, posterior features, suspicion), run 3×
   with **self-consistency voting**: majority vote on presence, median bbox,
   mode on categoricals.
2. `sensitive_second_pass()` — if negative, re-ask with a recall-oriented prompt.
3. **CV proposes / Gemini verifies** — if still negative, `propose_candidates()`
   generates high-recall/low-precision regions and Gemini adjudicates each crop.
   This is the genuine hybrid mechanism; neither component can do it alone.

Characterization — fixes the dead features:
- GrabCut segmentation seeded from the bbox (not Otsu on a tight crop)
- Continuous, clinically-grounded features an LLM cannot measure by eye:
  `depth_width_ratio` (taller-than-wide = classic malignancy sign),
  `posterior_shadow` (shadowing vs enhancement), `margin_sharpness`,
  `angular_margin_var` (real spiculation proxy), GLCM texture, `rel_echogenicity`
- ROI is padded 10% before extraction (a tight box clips the diagnostic margin)
- No 5% edge crop inside ROIs — that was removing the lesion margin

Scoring: `rule_score()` (transparent, consumes both sources) or `FusionHead`
(logistic regression over Gemini + CV features, fit on dev, loaded from
`fusion_model.json` if present).

## Files

| file | role | committed |
|---|---|---|
| `main.py` | FastAPI app. `/predict` routes to v2 when `SONI_MODE` is set | yes |
| `pipeline_v2.py` | v2 orchestration, feature assembly, scoring | yes |
| `hybrid_features.py` | GrabCut segmentation + quantitative CV features + candidate proposer | yes |
| `gemini_assess.py` | structured prompts, JSON parsing, voting, verification | yes |
| `eval_busi_v2.py` | runs a split, dumps every feature as a CSV column | yes |
| `fit_fusion.py` | trains fusion head on dev, runs feature-group ablation | yes |
| `make_split.py` / `busi_split.json` | the frozen split | yes |
| `eval_busi.py`, `compute_metrics.py`, `threshold_sweep.py` | v1 eval tooling | yes |
| `spotcheck.py` | throwaway 5-image diagnostic | optional |
| `Dataset_BUSI_with_GT/`, `eval_*.csv`, `fusion_model.json` | data/outputs | **gitignored** |

`SONI_MODE` env var selects the pipeline: `legacy` (default, = v1),
`hybrid_v2`, `gemini_only`, `opencv_only`. The last two are ablation arms.

## Commands

```bash
# serve (v2)
SONI_MODE=hybrid_v2 uvicorn main:app --port 8000 --reload

# quick 5-image sanity check
python spotcheck.py

# dev evaluation (slow: 3 votes + possible verification calls per image)
SPLIT=dev MODE_TAG=hybrid_v2 python eval_busi_v2.py

# train fusion head + the hybrid-justification ablation
python fit_fusion.py eval_hybrid_v2_dev.csv

# v1 metrics tooling (still works on v1 CSVs)
python compute_metrics.py eval_results_dev.csv
python threshold_sweep.py
```

Gotchas:
- `eval_*.py` resumes by skipping filenames already in the CSV. **Delete the CSV
  before re-running after a code change**, or it will score nothing.
- uvicorn without `--reload` keeps stale code in memory. This has burned a full
  eval run before.
- Requires `.env` with `GEMINI_API_KEY`. Network must be stable for the whole
  run — an offline stretch silently changes behavior and corrupts results.
- `pip install requests scikit-learn`

## Where things stand / next steps

1. **v2 is written and unit-tested but has never run on real BUSI images.** The
   CV features were validated on synthetic lesions, where they separate cleanly
   (circularity 0.851 benign vs 0.277 malignant; depth_width_ratio 0.688 vs
   1.126; posterior_shadow -0.111 vs +0.191). Real ultrasound speckle is harsher
   and GrabCut will fail on some images. **First job: run spotcheck, then the
   dev eval, and see what actually happens.**
2. Run `fit_fusion.py`. It prints AUROC for Gemini-only, CV-only, and fused, and
   states whether fusion beats both. **This is the falsifiability check for the
   entire hybrid thesis.** In v1 the gap was +0.000. If it is still ~0, say so
   plainly and investigate which features are dead — a negative result honestly
   reported is publishable and is better than a fake positive.
3. If the hybrid holds up: freeze v1-hybrid-v2, tag it, then run the ablation
   grid (`hybrid_v2` / `gemini_only` / `opencv_only`) on **test**, once.
4. Report AUROC with bootstrap 95% CIs, not just point sensitivity/specificity.
   Reviewers expect threshold-free metrics.
5. `paper_log.md` tracks decisions and results — keep it updated.
6. Note for the paper: this task is normal vs (benign+malignant), which is NOT
   what most published BUSI papers report (usually benign vs malignant, or
   3-class). Do not compare accuracy numbers across those framings without
   flagging the difference loudly.

## Working style

Paul is a high school senior (grad 2027) building this for research publication
and college applications. He wants direct, specific guidance: exact file names,
exact commands, plain language. Tell him explicitly which files go in the repo
and whether each is committed or gitignored. Explain *why* a change works, not
just what it does. Push back honestly when something would undermine the
project's credibility — he has asked for that and it has been the right call
every time.
