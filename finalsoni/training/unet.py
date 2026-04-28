"""
Lightweight U-Net for breast ultrasound segmentation.

This file imports torch at top-level. Anything that wants to run *without*
torch should not import this module - the main pipeline only loads it through
UNetSegmenter.load() which is gated.

Architecture: standard U-Net, 4 down-blocks + 4 up-blocks, base channel=32.
That's ~7.7M parameters, small enough to train on a single GPU and BUSI-sized
data without overfitting catastrophically.

Loss: combined BCE + soft Dice. BCE alone misses the imbalance (most pixels
are background); Dice alone is unstable early. Mixing them is the standard
recipe for medical segmentation.

Run with:
    python -m training.unet --busi /path/Dataset_BUSI_with_GT \
        --splits artifacts/splits.json --out artifacts/unet.pt \
        --epochs 50 --batch 8
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.utils.data import Dataset, DataLoader
except ImportError as e:
    raise ImportError(
        "training.unet requires PyTorch. Install with `pip install torch`."
    ) from e

import cv2

from pipeline.preprocessing import PreprocessConfig, preprocess
from pipeline.segmentation import dice as np_dice, iou as np_iou
from .dataset import BUSISample, Splits, load_pair


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

class _DoubleConv(nn.Module):
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.net(x)


class UNet(nn.Module):
    def __init__(self, in_channels: int = 1, out_channels: int = 1,
                 base: int = 32):
        super().__init__()
        self.d1 = _DoubleConv(in_channels, base)
        self.d2 = _DoubleConv(base, base * 2)
        self.d3 = _DoubleConv(base * 2, base * 4)
        self.d4 = _DoubleConv(base * 4, base * 8)
        self.bottleneck = _DoubleConv(base * 8, base * 16)

        self.up4 = nn.ConvTranspose2d(base * 16, base * 8, 2, stride=2)
        self.u4 = _DoubleConv(base * 16, base * 8)
        self.up3 = nn.ConvTranspose2d(base * 8, base * 4, 2, stride=2)
        self.u3 = _DoubleConv(base * 8, base * 4)
        self.up2 = nn.ConvTranspose2d(base * 4, base * 2, 2, stride=2)
        self.u2 = _DoubleConv(base * 4, base * 2)
        self.up1 = nn.ConvTranspose2d(base * 2, base, 2, stride=2)
        self.u1 = _DoubleConv(base * 2, base)

        self.out = nn.Conv2d(base, out_channels, 1)
        self.pool = nn.MaxPool2d(2)

    def forward(self, x):
        d1 = self.d1(x)
        d2 = self.d2(self.pool(d1))
        d3 = self.d3(self.pool(d2))
        d4 = self.d4(self.pool(d3))
        b = self.bottleneck(self.pool(d4))
        u4 = self.u4(torch.cat([self.up4(b), d4], dim=1))
        u3 = self.u3(torch.cat([self.up3(u4), d3], dim=1))
        u2 = self.u2(torch.cat([self.up2(u3), d2], dim=1))
        u1 = self.u1(torch.cat([self.up1(u2), d1], dim=1))
        return self.out(u1)


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

class BUSITorchDataset(Dataset):
    def __init__(self, samples: List[BUSISample],
                 pre_cfg: PreprocessConfig,
                 augment: bool = False):
        self.samples = samples
        self.pre_cfg = pre_cfg
        self.augment = augment

    def __len__(self) -> int:
        return len(self.samples)

    def _augment(self, x: np.ndarray, m: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        # Horizontal flip (50%).
        if np.random.rand() < 0.5:
            x = x[:, ::-1].copy()
            m = m[:, ::-1].copy()
        # Small rotation (-10..+10 degrees).
        if np.random.rand() < 0.5:
            angle = float(np.random.uniform(-10, 10))
            h, w = x.shape
            M = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
            x = cv2.warpAffine(x, M, (w, h),
                               flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
            m = cv2.warpAffine(m, M, (w, h),
                               flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_REFLECT)
        return x, m

    def __getitem__(self, idx):
        s = self.samples[idx]
        img, mask = load_pair(s)
        gray = preprocess(img, self.pre_cfg)
        m = cv2.resize(mask, (self.pre_cfg.target_size, self.pre_cfg.target_size),
                       interpolation=cv2.INTER_NEAREST)
        m = (m > 127).astype(np.float32)

        if self.augment:
            gray, m = self._augment(gray, m)

        x = gray.astype(np.float32) / 255.0
        x = torch.from_numpy(x).unsqueeze(0)        # (1, H, W)
        y = torch.from_numpy(m).unsqueeze(0)        # (1, H, W)
        return x, y


# ---------------------------------------------------------------------------
# Loss
# ---------------------------------------------------------------------------

def soft_dice_loss(logits: torch.Tensor, target: torch.Tensor,
                   eps: float = 1e-6) -> torch.Tensor:
    p = torch.sigmoid(logits)
    inter = (p * target).sum(dim=(1, 2, 3))
    s = p.sum(dim=(1, 2, 3)) + target.sum(dim=(1, 2, 3))
    dice = (2 * inter + eps) / (s + eps)
    return 1.0 - dice.mean()


def combined_loss(logits, target, w_bce: float = 0.5):
    bce = F.binary_cross_entropy_with_logits(logits, target)
    d = soft_dice_loss(logits, target)
    return w_bce * bce + (1 - w_bce) * d


# ---------------------------------------------------------------------------
# Train / eval loops
# ---------------------------------------------------------------------------

@torch.no_grad()
def evaluate_torch(model: UNet, loader: DataLoader, device: str,
                   threshold: float = 0.5) -> Dict[str, float]:
    model.eval()
    dices, ious, losses = [], [], []
    for x, y in loader:
        x = x.to(device); y = y.to(device)
        logits = model(x)
        loss = combined_loss(logits, y).item()
        losses.append(loss)
        probs = torch.sigmoid(logits).cpu().numpy()
        gt = y.cpu().numpy()
        for p, g in zip(probs, gt):
            pm = (p[0] >= threshold).astype(np.uint8)
            gm = (g[0] > 0.5).astype(np.uint8)
            if gm.sum() == 0:
                continue
            dices.append(np_dice(pm, gm))
            ious.append(np_iou(pm, gm))
    return {"loss": float(np.mean(losses)) if losses else 0.0,
            "mean_dice": float(np.mean(dices)) if dices else 0.0,
            "mean_iou":  float(np.mean(ious))  if ious  else 0.0}


def train(busi_root: str, splits_path: str, out_path: str,
          epochs: int = 50, batch_size: int = 8, lr: float = 1e-3,
          target_size: int = 256, device: Optional[str] = None,
          patience: int = 10) -> Dict:
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    splits = Splits.load(splits_path)

    pre_cfg = PreprocessConfig(target_size=target_size)
    # Train on lesion-bearing samples only - normal images have empty masks
    # and would push the network toward always predicting empty.
    train_samples = [s for s in splits.train if s.klass in ("benign", "malignant")]
    val_samples   = [s for s in splits.val   if s.klass in ("benign", "malignant")]

    print(f"[unet] train={len(train_samples)} val={len(val_samples)} device={device}")

    tr_ds = BUSITorchDataset(train_samples, pre_cfg, augment=True)
    va_ds = BUSITorchDataset(val_samples,   pre_cfg, augment=False)
    tr_ld = DataLoader(tr_ds, batch_size=batch_size, shuffle=True, num_workers=2, drop_last=True)
    va_ld = DataLoader(va_ds, batch_size=batch_size, shuffle=False, num_workers=2)

    model = UNet().to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)

    history: List[Dict] = []
    best_dice = -1.0
    no_improve = 0

    for epoch in range(1, epochs + 1):
        model.train()
        t0 = time.time()
        losses = []
        for x, y in tr_ld:
            x = x.to(device); y = y.to(device)
            opt.zero_grad()
            logits = model(x)
            loss = combined_loss(logits, y)
            loss.backward()
            opt.step()
            losses.append(loss.item())
        sched.step()

        train_loss = float(np.mean(losses)) if losses else 0.0
        val = evaluate_torch(model, va_ld, device)
        history.append({"epoch": epoch, "train_loss": train_loss, **val,
                        "elapsed": time.time() - t0})
        print(f"  ep {epoch:>3}  train_loss={train_loss:.4f}  "
              f"val_dice={val['mean_dice']:.3f}  val_iou={val['mean_iou']:.3f}")

        if val["mean_dice"] > best_dice:
            best_dice = val["mean_dice"]
            no_improve = 0
            Path(out_path).parent.mkdir(parents=True, exist_ok=True)
            torch.save(model.state_dict(), out_path)
        else:
            no_improve += 1
            if no_improve >= patience:
                print(f"  (early stop at epoch {epoch})")
                break

    return {"best_val_dice": best_dice, "history": history,
            "checkpoint": out_path}


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _cli():
    ap = argparse.ArgumentParser(description="Train U-Net on BUSI")
    ap.add_argument("--busi", required=True)
    ap.add_argument("--splits", required=True)
    ap.add_argument("--out", required=True, help="output .pt checkpoint path")
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--size", type=int, default=256)
    ap.add_argument("--device", default=None)
    args = ap.parse_args()
    summary = train(args.busi, args.splits, args.out,
                    epochs=args.epochs, batch_size=args.batch, lr=args.lr,
                    target_size=args.size, device=args.device)
    out_json = Path(args.out).with_suffix(".history.json")
    out_json.write_text(json.dumps(summary, indent=2))
    print(f"saved checkpoint -> {args.out}")
    print(f"saved history    -> {out_json}")


if __name__ == "__main__":
    _cli()
