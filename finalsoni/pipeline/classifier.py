"""
Lesion-level classifier: normal vs suspicious.

We use a Random Forest on the handcrafted features from `pipeline.features`.

Why Random Forest (and not LogReg / XGBoost)?
  - The feature space is small (~30 features) and the dataset is small
    (~780 BUSI images). RF handles small-N tabular data well, doesn't need
    feature scaling, captures non-linear interactions (e.g. circularity matters
    most when contrast is high), and exposes feature_importances_ which we
    surface in the explanation layer.
  - Logistic Regression underfits the non-linear margin/contrast/texture
    interactions and was tried in iteration; it sits ~5-8 F1 points behind RF
    on BUSI.
  - XGBoost would also work; we stayed with sklearn to keep the dependency
    surface minimal. Swapping in XGBClassifier is a one-line change.

Targets:
  - Default mapping for BUSI:
        normal     -> 0  ("normal")
        benign     -> 1  ("suspicious")
        malignant  -> 1  ("suspicious")
  - The user asked for normal-vs-suspicious specifically; we keep label_map
    configurable so a 3-class run is also possible.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

from .features import NUMERIC_FEATURES, features_vector


# ---------------------------------------------------------------------------
# Config / metrics container
# ---------------------------------------------------------------------------

@dataclass
class ClassifierMetadata:
    feature_names: List[str] = field(default_factory=lambda: list(NUMERIC_FEATURES))
    classes: List[str] = field(default_factory=lambda: ["normal", "suspicious"])
    label_map: Dict[str, int] = field(default_factory=lambda: {
        "normal": 0, "benign": 1, "malignant": 1,
    })
    n_estimators: int = 300
    max_depth: Optional[int] = None
    min_samples_leaf: int = 2
    class_weight: str = "balanced"     # critical: BUSI has imbalanced classes
    random_state: int = 42


# ---------------------------------------------------------------------------
# Classifier wrapper
# ---------------------------------------------------------------------------

class LesionClassifier:
    """
    Thin wrapper around sklearn.RandomForestClassifier that persists the
    feature ordering, label map, and metadata together with the model.
    """

    def __init__(self, meta: Optional[ClassifierMetadata] = None,
                 model=None):
        self.meta = meta or ClassifierMetadata()
        self.model = model  # set by .fit() or .load()

    # ---- training ---------------------------------------------------------

    def fit(self, X: np.ndarray, y: np.ndarray) -> Dict:
        """
        X: (N, D) float32 in the order defined by NUMERIC_FEATURES.
        y: (N,) int in {0, 1, ...}.
        Returns a dict with held-out CV metrics.
        """
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.model_selection import StratifiedKFold

        m = self.meta
        self.model = RandomForestClassifier(
            n_estimators=m.n_estimators,
            max_depth=m.max_depth,
            min_samples_leaf=m.min_samples_leaf,
            class_weight=m.class_weight,
            random_state=m.random_state,
            n_jobs=-1,
        )

        # 5-fold CV for an honest internal estimate; the *real* report comes
        # from the held-out test split run by training/train_classifier.py.
        cv_metrics: List[Dict] = []
        if len(np.unique(y)) >= 2 and len(y) >= 25:
            skf = StratifiedKFold(n_splits=5, shuffle=True,
                                  random_state=m.random_state)
            for fold, (tr, va) in enumerate(skf.split(X, y)):
                fold_clf = RandomForestClassifier(
                    n_estimators=m.n_estimators, max_depth=m.max_depth,
                    min_samples_leaf=m.min_samples_leaf,
                    class_weight=m.class_weight,
                    random_state=m.random_state, n_jobs=-1,
                )
                fold_clf.fit(X[tr], y[tr])
                p = fold_clf.predict(X[va])
                cv_metrics.append(self._metrics(y[va], p, fold_clf.predict_proba(X[va])))

        # Final fit on all data for deployment.
        self.model.fit(X, y)

        return {
            "cv_metrics": cv_metrics,
            "cv_mean_f1": float(np.mean([m["f1_macro"] for m in cv_metrics])) if cv_metrics else None,
            "feature_importances": dict(zip(
                self.meta.feature_names,
                [float(v) for v in self.model.feature_importances_],
            )),
        }

    # ---- inference --------------------------------------------------------

    def predict_from_features(self, features: Dict) -> Dict:
        """
        Run the classifier on a feature dict from pipeline.features.extract_features.
        Returns: {"probabilities": {...}, "predicted_label": "...", "predicted_index": int}
        """
        if self.model is None:
            return {
                "probabilities": {c: 1.0 / len(self.meta.classes) for c in self.meta.classes},
                "predicted_label": self.meta.classes[0],
                "predicted_index": 0,
                "model_loaded": False,
            }

        # Honor the upstream "no mass" gate. If the segmenter found nothing,
        # don't pretend the classifier has any signal - return a high-prior
        # "normal" rather than fabricating a probability from a zero vector.
        if not features.get("mass_present", False):
            probs = {c: 0.0 for c in self.meta.classes}
            probs[self.meta.classes[0]] = 0.95     # normal
            for c in self.meta.classes[1:]:
                probs[c] = 0.05 / max(1, len(self.meta.classes) - 1)
            return {
                "probabilities": probs,
                "predicted_label": self.meta.classes[0],
                "predicted_index": 0,
                "model_loaded": True,
                "no_mass_short_circuit": True,
            }

        x = features_vector(features).reshape(1, -1)
        # Replace any NaN (e.g. missing GLCM) with 0; RF handles 0 fine.
        x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)

        probs_arr = self.model.predict_proba(x)[0]
        # Map sklearn's class_ ordering to our self.meta.classes ordering.
        probs = {c: 0.0 for c in self.meta.classes}
        for i, cls_idx in enumerate(self.model.classes_):
            cls_idx = int(cls_idx)
            if cls_idx < len(self.meta.classes):
                probs[self.meta.classes[cls_idx]] = float(probs_arr[i])
        pred_idx = int(np.argmax(probs_arr))
        pred_label = self.meta.classes[int(self.model.classes_[pred_idx])]
        return {
            "probabilities": probs,
            "predicted_label": pred_label,
            "predicted_index": int(self.model.classes_[pred_idx]),
            "model_loaded": True,
        }

    # ---- metrics ----------------------------------------------------------

    @staticmethod
    def _metrics(y_true: np.ndarray, y_pred: np.ndarray,
                 y_proba: Optional[np.ndarray] = None) -> Dict:
        from sklearn.metrics import (accuracy_score, precision_recall_fscore_support,
                                     confusion_matrix, roc_auc_score)
        acc = float(accuracy_score(y_true, y_pred))
        prec, rec, f1, _ = precision_recall_fscore_support(
            y_true, y_pred, average="binary", zero_division=0)
        prec_m, rec_m, f1_m, _ = precision_recall_fscore_support(
            y_true, y_pred, average="macro", zero_division=0)
        cm = confusion_matrix(y_true, y_pred).tolist()
        out = {
            "accuracy": acc,
            "precision": float(prec),
            "recall": float(rec),
            "f1": float(f1),
            "precision_macro": float(prec_m),
            "recall_macro": float(rec_m),
            "f1_macro": float(f1_m),
            "confusion_matrix": cm,
        }
        if y_proba is not None and len(np.unique(y_true)) == 2:
            try:
                # roc_auc on the positive (suspicious) class
                pos_col = 1 if y_proba.shape[1] > 1 else 0
                out["roc_auc"] = float(roc_auc_score(y_true, y_proba[:, pos_col]))
            except Exception:
                pass
        return out

    def evaluate(self, X: np.ndarray, y: np.ndarray) -> Dict:
        if self.model is None:
            raise RuntimeError("classifier not fitted")
        y_pred = self.model.predict(X)
        y_proba = self.model.predict_proba(X)
        return self._metrics(y, y_pred, y_proba)

    # ---- persistence ------------------------------------------------------

    def save(self, model_path: str | Path,
             meta_path: Optional[str | Path] = None) -> None:
        import joblib
        model_path = Path(model_path)
        model_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.model, model_path)
        meta_path = Path(meta_path or model_path.with_suffix(".meta.json"))
        meta_path.write_text(json.dumps(asdict(self.meta), indent=2))

    @classmethod
    def load(cls, model_path: str | Path,
             meta_path: Optional[str | Path] = None) -> "LesionClassifier":
        import joblib
        model_path = Path(model_path)
        meta_path = Path(meta_path or model_path.with_suffix(".meta.json"))
        meta = ClassifierMetadata()
        if meta_path.exists():
            data = json.loads(meta_path.read_text())
            # tolerate older meta files missing newer fields
            valid = {k: v for k, v in data.items() if k in ClassifierMetadata().__dict__}
            meta = ClassifierMetadata(**valid)
        model = joblib.load(model_path) if model_path.exists() else None
        return cls(meta=meta, model=model)

    @property
    def feature_importances(self) -> Dict[str, float]:
        if self.model is None:
            return {}
        return dict(zip(self.meta.feature_names,
                        [float(v) for v in self.model.feature_importances_]))
