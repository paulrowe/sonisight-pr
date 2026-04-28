"""
Train the lesion-level classifier.

We extract the same handcrafted features used at inference time and fit a
Random Forest on them. Features are computed using the GROUND-TRUTH mask
during training so that the classifier learns from ideal lesion delineation,
independent of segmentation quality.

At inference time, the segmenter's predicted mask is fed into the same
feature extractor. There's a known train-test mismatch here (training masks
are clean, inference masks are noisy). We surface this honestly in the eval
report below by also reporting metrics with the predicted segmenter mask.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import numpy as np

from pipeline.classifier import ClassifierMetadata, LesionClassifier
from pipeline.features import NUMERIC_FEATURES, extract_features, features_vector
from pipeline.preprocessing import PreprocessConfig, preprocess
from pipeline.segmentation import ClassicalSegmenter
from .dataset import BUSISample, Splits, load_pair


def _resize_mask(m: np.ndarray, size: int) -> np.ndarray:
    return cv2.resize(m, (size, size), interpolation=cv2.INTER_NEAREST)


def build_feature_table(samples: List[BUSISample],
                        pre_cfg: PreprocessConfig,
                        label_map: Dict[str, int],
                        use_segmenter: bool = False,
                        segmenter=None,
                        ) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """
    Extract features per sample.

    If `use_segmenter=True`, features come from the SEGMENTER's predicted mask
    (matches inference). Otherwise features come from the GROUND-TRUTH mask
    (cleaner training signal). We use ground-truth for training and segmenter
    output for honest test-time evaluation.
    """
    X_rows: List[np.ndarray] = []
    y_rows: List[int] = []
    stems: List[str] = []

    for s in samples:
        if s.klass not in label_map:
            continue
        img, mask = load_pair(s)
        gray = preprocess(img, pre_cfg)

        if use_segmenter:
            if segmenter is None:
                raise ValueError("use_segmenter=True but no segmenter provided")
            seg_out = segmenter.predict(gray)
            use_mask = seg_out.mask
        else:
            # Ground-truth path: resize the GT mask to the preprocessed size.
            use_mask = _resize_mask(mask, pre_cfg.target_size)

        # For the `normal` class, both masks are empty - that's fine. Features
        # become zeros and `mass_present` is False, which is the signal we want.
        feats = extract_features(gray, use_mask)
        X_rows.append(features_vector(feats))
        y_rows.append(label_map[s.klass])
        stems.append(s.stem)

    X = np.vstack(X_rows) if X_rows else np.zeros((0, len(NUMERIC_FEATURES)),
                                                  dtype=np.float32)
    y = np.asarray(y_rows, dtype=np.int64)
    return X, y, stems


def train_classifier(busi_root: str,
                     splits_path: str,
                     out_dir: str,
                     seg_config_path: str = None,
                     binary: bool = True) -> Dict:
    splits = Splits.load(splits_path)

    pre_cfg = PreprocessConfig(target_size=256)
    if binary:
        meta = ClassifierMetadata(
            classes=["normal", "suspicious"],
            label_map={"normal": 0, "benign": 1, "malignant": 1},
        )
    else:
        meta = ClassifierMetadata(
            classes=["normal", "benign", "malignant"],
            label_map={"normal": 0, "benign": 1, "malignant": 2},
        )

    # 1) Train features from GROUND-TRUTH masks.
    Xtr, ytr, _ = build_feature_table(splits.train, pre_cfg, meta.label_map,
                                      use_segmenter=False)
    print(f"[classifier] train rows: {len(Xtr)} | classes: {dict(zip(*np.unique(ytr, return_counts=True)))}")

    clf = LesionClassifier(meta=meta)
    cv_report = clf.fit(Xtr, ytr)
    print(f"[classifier] CV mean f1_macro: {cv_report['cv_mean_f1']}")

    # 2) Test features from BOTH ground-truth and segmenter predictions.
    Xte_gt, yte, _ = build_feature_table(splits.test, pre_cfg, meta.label_map,
                                         use_segmenter=False)

    seg = ClassicalSegmenter.load(seg_config_path)
    Xte_seg, yte2, _ = build_feature_table(splits.test, pre_cfg, meta.label_map,
                                           use_segmenter=True, segmenter=seg)
    assert (yte == yte2).all(), "test ordering drifted"

    test_gt  = clf.evaluate(Xte_gt, yte)
    test_seg = clf.evaluate(Xte_seg, yte)

    print(f"[classifier] test (GT mask)        f1={test_gt['f1']:.3f}  acc={test_gt['accuracy']:.3f}")
    print(f"[classifier] test (segmenter mask) f1={test_seg['f1']:.3f}  acc={test_seg['accuracy']:.3f}")

    out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    model_path = out_dir / "classifier.joblib"
    clf.save(model_path)

    report = {
        "cv": cv_report,
        "test_with_ground_truth_mask": test_gt,
        "test_with_predicted_mask": test_seg,
        "n_train": int(len(Xtr)), "n_test": int(len(yte)),
        "classes": meta.classes,
        "feature_names": list(meta.feature_names),
        "model_path": str(model_path),
    }
    (out_dir / "classifier_report.json").write_text(json.dumps(report, indent=2))
    return report


def _cli():
    ap = argparse.ArgumentParser(description="Train SoniSight classifier")
    ap.add_argument("--busi", required=True, help="BUSI root (unused; kept for symmetry)")
    ap.add_argument("--splits", required=True)
    ap.add_argument("--out", default="artifacts")
    ap.add_argument("--seg_config", default="artifacts/seg_config.json")
    ap.add_argument("--three_class", action="store_true",
                    help="Train 3-class normal/benign/malignant instead of binary.")
    args = ap.parse_args()
    train_classifier(args.busi, args.splits, args.out,
                     seg_config_path=args.seg_config,
                     binary=not args.three_class)


if __name__ == "__main__":
    _cli()
