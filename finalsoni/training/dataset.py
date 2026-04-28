"""
BUSI (Breast Ultrasound Images) dataset loader and split manager.

Expected directory layout (matches the user's upload):

    Dataset_BUSI_with_GT/
        normal/
            normal (1).png
            normal (1)_mask.png
            ...
        benign/                 # optional
            benign (1).png
            benign (1)_mask.png
            ...
        malignant/
            malignant (1).png
            malignant (1)_mask.png
            ...

A few BUSI samples have multiple mask files per image (e.g. `..._mask_1.png`,
`..._mask_2.png`); we OR them into a single mask.

The split is made deterministically (seeded) and saved to JSON so segmentation
training, classifier training, and evaluation all see the same partitions.
"""

from __future__ import annotations

import json
import random
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import cv2
import numpy as np


# ---------------------------------------------------------------------------
# Sample dataclass
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class BUSISample:
    image_path: str
    mask_paths: Tuple[str, ...]   # one or more
    klass: str                    # "normal" | "benign" | "malignant"

    @property
    def stem(self) -> str:
        return Path(self.image_path).stem


# ---------------------------------------------------------------------------
# Discovery
# ---------------------------------------------------------------------------

# matches "..._mask.png" or "..._mask_1.png"
_MASK_RE = re.compile(r"_mask(_\d+)?$", re.IGNORECASE)


def _is_mask(stem: str) -> bool:
    return bool(_MASK_RE.search(stem))


def _strip_mask(stem: str) -> str:
    return _MASK_RE.sub("", stem)


def discover_busi(root: str | Path,
                  classes: Iterable[str] = ("normal", "benign", "malignant"),
                  ) -> List[BUSISample]:
    """
    Scan the BUSI root and return one BUSISample per image (with all matching
    masks grouped). Missing class folders are skipped silently.
    """
    root = Path(root)
    if not root.exists():
        raise FileNotFoundError(f"BUSI root not found: {root}")

    samples: List[BUSISample] = []
    for cls in classes:
        cdir = root / cls
        if not cdir.exists():
            continue

        # group files by their non-mask stem
        groups: Dict[str, Dict[str, List[Path]]] = {}
        for f in cdir.iterdir():
            if not f.is_file():
                continue
            if f.suffix.lower() not in {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}:
                continue
            stem = f.stem
            base = _strip_mask(stem) if _is_mask(stem) else stem
            g = groups.setdefault(base, {"image": [], "masks": []})
            (g["masks"] if _is_mask(stem) else g["image"]).append(f)

        for base, g in groups.items():
            if not g["image"]:
                # masks without an image - skip
                continue
            img = sorted(g["image"])[0]   # exactly one is expected
            masks = tuple(str(m) for m in sorted(g["masks"]))
            samples.append(BUSISample(image_path=str(img),
                                      mask_paths=masks, klass=cls))

    samples.sort(key=lambda s: s.image_path)
    return samples


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def load_image(path: str) -> np.ndarray:
    """Read as BGR (or grayscale), uint8."""
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"Could not read image: {path}")
    return img


def load_mask(paths: Iterable[str], shape: Tuple[int, int]) -> np.ndarray:
    """
    OR-combine one or more mask files into a single binary uint8 mask shaped
    like `shape` (H, W). Resizes any mask that doesn't match.
    """
    H, W = shape
    out = np.zeros((H, W), dtype=np.uint8)
    for p in paths:
        m = cv2.imread(p, cv2.IMREAD_GRAYSCALE)
        if m is None:
            continue
        if m.shape != (H, W):
            m = cv2.resize(m, (W, H), interpolation=cv2.INTER_NEAREST)
        out = np.maximum(out, (m > 127).astype(np.uint8) * 255)
    return out


def load_pair(sample: BUSISample) -> Tuple[np.ndarray, np.ndarray]:
    """Convenience: returns (bgr_image, binary_mask)."""
    img = load_image(sample.image_path)
    mask = load_mask(sample.mask_paths, img.shape[:2])
    return img, mask


# ---------------------------------------------------------------------------
# Splits
# ---------------------------------------------------------------------------

@dataclass
class Splits:
    train: List[BUSISample]
    val: List[BUSISample]
    test: List[BUSISample]

    def to_dict(self) -> Dict:
        def ser(s: BUSISample) -> Dict:
            return {"image_path": s.image_path,
                    "mask_paths": list(s.mask_paths),
                    "klass": s.klass}
        return {"train": [ser(s) for s in self.train],
                "val":   [ser(s) for s in self.val],
                "test":  [ser(s) for s in self.test]}

    def save(self, path: str | Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text(json.dumps(self.to_dict(), indent=2))

    @classmethod
    def load(cls, path: str | Path) -> "Splits":
        d = json.loads(Path(path).read_text())
        def deser(x): return BUSISample(image_path=x["image_path"],
                                        mask_paths=tuple(x["mask_paths"]),
                                        klass=x["klass"])
        return cls(train=[deser(x) for x in d["train"]],
                   val=[deser(x) for x in d["val"]],
                   test=[deser(x) for x in d["test"]])


def make_splits(samples: List[BUSISample],
                val_frac: float = 0.15, test_frac: float = 0.15,
                seed: int = 42) -> Splits:
    """
    Stratified split by class so each split has the same class balance.
    """
    by_class: Dict[str, List[BUSISample]] = {}
    for s in samples:
        by_class.setdefault(s.klass, []).append(s)

    rng = random.Random(seed)
    train: List[BUSISample] = []
    val:   List[BUSISample] = []
    test:  List[BUSISample] = []

    for cls, lst in by_class.items():
        lst = list(lst)
        rng.shuffle(lst)
        n = len(lst)
        n_test = int(round(n * test_frac))
        n_val = int(round(n * val_frac))
        n_train = n - n_test - n_val
        train += lst[:n_train]
        val   += lst[n_train:n_train + n_val]
        test  += lst[n_train + n_val:]

    rng.shuffle(train)
    rng.shuffle(val)
    rng.shuffle(test)
    return Splits(train=train, val=val, test=test)


def class_distribution(samples: List[BUSISample]) -> Dict[str, int]:
    out: Dict[str, int] = {}
    for s in samples:
        out[s.klass] = out.get(s.klass, 0) + 1
    return out
