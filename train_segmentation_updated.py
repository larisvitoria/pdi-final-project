import argparse
import os
import random
from dataclasses import dataclass
from typing import Dict, Tuple, Optional, List

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import autocast, GradScaler
from torch.utils.data import Dataset, DataLoader

from preprocess_cbis import CBISDDSM_Preprocessor


# ----------------------------
# Repro
# ----------------------------
def set_seed(seed: int, deterministic: bool = False):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    else:
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = True


def _pad_to_multiple(x: torch.Tensor, multiple: int = 16) -> Tuple[torch.Tensor, Tuple[int, int]]:
    h, w = x.shape[-2:]
    pad_h = (multiple - h % multiple) % multiple
    pad_w = (multiple - w % multiple) % multiple
    if pad_h or pad_w:
        x = F.pad(x, (0, pad_w, 0, pad_h))
    return x, (h, w)


# ----------------------------
# Models (same as before; keep UNet + AttentionUNet)
# ----------------------------
class DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch):
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
    def __init__(self, in_channels=1, out_channels=1, base_channels=32):
        super().__init__()
        self.enc1 = DoubleConv(in_channels, base_channels)
        self.pool1 = nn.MaxPool2d(2)
        self.enc2 = DoubleConv(base_channels, base_channels * 2)
        self.pool2 = nn.MaxPool2d(2)
        self.enc3 = DoubleConv(base_channels * 2, base_channels * 4)
        self.pool3 = nn.MaxPool2d(2)
        self.enc4 = DoubleConv(base_channels * 4, base_channels * 8)
        self.pool4 = nn.MaxPool2d(2)

        self.bottleneck = DoubleConv(base_channels * 8, base_channels * 16)

        self.up4 = nn.ConvTranspose2d(base_channels * 16, base_channels * 8, 2, stride=2)
        self.dec4 = DoubleConv(base_channels * 16, base_channels * 8)
        self.up3 = nn.ConvTranspose2d(base_channels * 8, base_channels * 4, 2, stride=2)
        self.dec3 = DoubleConv(base_channels * 8, base_channels * 4)
        self.up2 = nn.ConvTranspose2d(base_channels * 4, base_channels * 2, 2, stride=2)
        self.dec2 = DoubleConv(base_channels * 4, base_channels * 2)
        self.up1 = nn.ConvTranspose2d(base_channels * 2, base_channels, 2, stride=2)
        self.dec1 = DoubleConv(base_channels * 2, base_channels)

        self.outc = nn.Conv2d(base_channels, out_channels, 1)

    def forward(self, x):
        x, orig_hw = _pad_to_multiple(x, multiple=16)
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool1(e1))
        e3 = self.enc3(self.pool2(e2))
        e4 = self.enc4(self.pool3(e3))
        b = self.bottleneck(self.pool4(e4))

        d4 = self.up4(b)
        d4 = self.dec4(torch.cat([d4, e4], dim=1))
        d3 = self.up3(d4)
        d3 = self.dec3(torch.cat([d3, e3], dim=1))
        d2 = self.up2(d3)
        d2 = self.dec2(torch.cat([d2, e2], dim=1))
        d1 = self.up1(d2)
        d1 = self.dec1(torch.cat([d1, e1], dim=1))

        out = self.outc(d1)
        h, w = orig_hw
        if out.shape[-2:] != (h, w):
            out = out[..., :h, :w]
        return out


class AttentionBlock(nn.Module):
    def __init__(self, F_g, F_l, F_int):
        super().__init__()
        self.W_g = nn.Sequential(nn.Conv2d(F_g, F_int, 1, bias=True), nn.BatchNorm2d(F_int))
        self.W_x = nn.Sequential(nn.Conv2d(F_l, F_int, 1, bias=True), nn.BatchNorm2d(F_int))
        self.psi = nn.Sequential(nn.Conv2d(F_int, 1, 1, bias=True), nn.BatchNorm2d(1), nn.Sigmoid())
        self.relu = nn.ReLU(inplace=True)

    def forward(self, g, x):
        psi = self.relu(self.W_g(g) + self.W_x(x))
        psi = self.psi(psi)
        return x * psi


class AttentionUNet(nn.Module):
    def __init__(self, in_channels=1, out_channels=1, base_channels=32):
        super().__init__()
        self.enc1 = DoubleConv(in_channels, base_channels)
        self.pool1 = nn.MaxPool2d(2)
        self.enc2 = DoubleConv(base_channels, base_channels * 2)
        self.pool2 = nn.MaxPool2d(2)
        self.enc3 = DoubleConv(base_channels * 2, base_channels * 4)
        self.pool3 = nn.MaxPool2d(2)
        self.enc4 = DoubleConv(base_channels * 4, base_channels * 8)
        self.pool4 = nn.MaxPool2d(2)

        self.bottleneck = DoubleConv(base_channels * 8, base_channels * 16)

        self.up4 = nn.ConvTranspose2d(base_channels * 16, base_channels * 8, 2, stride=2)
        self.att4 = AttentionBlock(F_g=base_channels * 8, F_l=base_channels * 8, F_int=base_channels * 4)
        self.dec4 = DoubleConv(base_channels * 16, base_channels * 8)

        self.up3 = nn.ConvTranspose2d(base_channels * 8, base_channels * 4, 2, stride=2)
        self.att3 = AttentionBlock(F_g=base_channels * 4, F_l=base_channels * 4, F_int=base_channels * 2)
        self.dec3 = DoubleConv(base_channels * 8, base_channels * 4)

        self.up2 = nn.ConvTranspose2d(base_channels * 4, base_channels * 2, 2, stride=2)
        self.att2 = AttentionBlock(F_g=base_channels * 2, F_l=base_channels * 2, F_int=base_channels)
        self.dec2 = DoubleConv(base_channels * 4, base_channels * 2)

        self.up1 = nn.ConvTranspose2d(base_channels * 2, base_channels, 2, stride=2)
        self.att1 = AttentionBlock(F_g=base_channels, F_l=base_channels, F_int=base_channels // 2)
        self.dec1 = DoubleConv(base_channels * 2, base_channels)

        self.outc = nn.Conv2d(base_channels, out_channels, 1)

    def forward(self, x):
        x, orig_hw = _pad_to_multiple(x, multiple=16)
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool1(e1))
        e3 = self.enc3(self.pool2(e2))
        e4 = self.enc4(self.pool3(e3))
        b = self.bottleneck(self.pool4(e4))

        d4 = self.up4(b)
        e4a = self.att4(d4, e4)
        d4 = self.dec4(torch.cat([d4, e4a], dim=1))

        d3 = self.up3(d4)
        e3a = self.att3(d3, e3)
        d3 = self.dec3(torch.cat([d3, e3a], dim=1))

        d2 = self.up2(d3)
        e2a = self.att2(d2, e2)
        d2 = self.dec2(torch.cat([d2, e2a], dim=1))

        d1 = self.up1(d2)
        e1a = self.att1(d1, e1)
        d1 = self.dec1(torch.cat([d1, e1a], dim=1))

        out = self.outc(d1)
        h, w = orig_hw
        if out.shape[-2:] != (h, w):
            out = out[..., :h, :w]
        return out


# ----------------------------
# Losses + metrics
# ----------------------------
def soft_dice_loss(logits: torch.Tensor, targets: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    probs = torch.sigmoid(logits)
    probs = probs.view(probs.size(0), -1)
    targets = targets.view(targets.size(0), -1)
    inter = (probs * targets).sum(dim=1)
    denom = probs.sum(dim=1) + targets.sum(dim=1)
    dice = (2 * inter + eps) / (denom + eps)
    return 1.0 - dice.mean()


@torch.no_grad()
def dice_iou_metrics(logits: torch.Tensor, targets: torch.Tensor, threshold: float = 0.5, eps: float = 1e-6) -> Tuple[float, float]:
    probs = torch.sigmoid(logits)
    preds = (probs > threshold).float()
    preds = preds.view(preds.size(0), -1)
    targets = targets.view(targets.size(0), -1)

    inter = (preds * targets).sum(dim=1)
    union = preds.sum(dim=1) + targets.sum(dim=1) - inter
    iou = (inter + eps) / (union + eps)

    dice = (2 * inter + eps) / (preds.sum(dim=1) + targets.sum(dim=1) + eps)
    return float(dice.mean().item()), float(iou.mean().item())


@torch.no_grad()
def find_best_threshold(model: nn.Module, loader: DataLoader, device: torch.device, thresholds: Optional[List[float]] = None) -> float:
    if thresholds is None:
        thresholds = [i / 100 for i in range(10, 91, 5)]  # 0.10 .. 0.90
    model.eval()

    # Accumulate all logits/targets once to avoid repeated forward passes
    all_logits = []
    all_targets = []
    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        logits = model(x)
        all_logits.append(logits.detach().cpu())
        all_targets.append(y.detach().cpu())
    logits = torch.cat(all_logits, dim=0)
    targets = torch.cat(all_targets, dim=0)

    best_t = 0.5
    best_d = -1.0
    for t in thresholds:
        d, _ = dice_iou_metrics(logits, targets, threshold=float(t))
        if d > best_d:
            best_d = d
            best_t = float(t)
    return best_t


# ----------------------------
# Dataset
# ----------------------------
class CBISSegDataset(Dataset):
    def __init__(self, df: pd.DataFrame, preprocessor: CBISDDSM_Preprocessor, patch_size: int, do_aug: bool):
        self.df = df.reset_index(drop=True)
        self.pre = preprocessor
        self.patch_size = patch_size
        self.do_aug = do_aug

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        # We assume empty masks have already been filtered in preprocessing.
        # Still keep a retry loop for robustness.
        for _ in range(3):
            try:
                img, msk = self.pre.preprocess_pair(row, patch_size=self.patch_size, do_aug=self.do_aug)
                break
            except Exception:
                idx = random.randint(0, len(self.df) - 1)
                row = self.df.iloc[idx]
        else:
            raise RuntimeError("Failed to load a valid sample after retries.")

        # Normalize for training stability (z-score per patch)
        mean = float(img.mean())
        std = float(img.std())
        img = (img - mean) / (std + 1e-6)

        x = torch.from_numpy(img).unsqueeze(0).float()         # [1,H,W]
        y = torch.from_numpy(msk).unsqueeze(0).float()         # [1,H,W]
        return x, y


# ----------------------------
# Pos-weight estimation (pixel imbalance)
# ----------------------------
@torch.no_grad()
def estimate_pos_weight(train_df: pd.DataFrame, pre: CBISDDSM_Preprocessor, patch_size: int, max_samples: int = 400) -> float:
    """
    Estimate pos_weight = neg_pixels / pos_pixels on preprocessed patches.
    This typically improves tiny-lesion segmentation a lot.
    """
    if len(train_df) == 0:
        return 1.0
    idxs = list(range(len(train_df)))
    random.shuffle(idxs)
    idxs = idxs[: min(max_samples, len(idxs))]

    pos = 0.0
    tot = 0.0
    for i in idxs:
        row = train_df.iloc[i]
        try:
            _, m = pre.preprocess_pair(row, patch_size=patch_size, do_aug=False)
        except Exception:
            continue
        pos += float(m.sum())
        tot += float(m.size)

    if pos <= 0.0:
        return 1.0
    neg = tot - pos
    pw = neg / pos
    # Keep it in a sane range (very small lesions can explode this)
    pw = float(np.clip(pw, 1.0, 50.0))
    return pw


# ----------------------------
# Train / Eval loops
# ----------------------------
def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scheduler: Optional[torch.optim.lr_scheduler._LRScheduler],
    bce_loss: nn.Module,
    device: torch.device,
    scaler: Optional[GradScaler],
    amp: bool,
    bce_w: float,
    dice_w: float,
    grad_clip: float,
):
    model.train()
    total_loss = 0.0

    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        if amp and scaler is not None:
            with autocast():
                logits = model(x)
                loss_bce = bce_loss(logits, y)
                loss_dice = soft_dice_loss(logits, y)
                loss = bce_w * loss_bce + dice_w * loss_dice
            scaler.scale(loss).backward()
            if grad_clip and grad_clip > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            scaler.step(optimizer)
            scaler.update()
        else:
            logits = model(x)
            loss_bce = bce_loss(logits, y)
            loss_dice = soft_dice_loss(logits, y)
            loss = bce_w * loss_bce + dice_w * loss_dice
            loss.backward()
            if grad_clip and grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()

        if scheduler is not None:
            scheduler.step()

        total_loss += float(loss.item())

    return total_loss / max(1, len(loader))


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    bce_loss: nn.Module,
    device: torch.device,
    threshold: float,
    bce_w: float,
    dice_w: float,
):
    model.eval()
    total_loss = 0.0
    total_dice = 0.0
    total_iou = 0.0

    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        logits = model(x)
        loss_bce = bce_loss(logits, y)
        loss_dice = soft_dice_loss(logits, y)
        loss = bce_w * loss_bce + dice_w * loss_dice

        d, i = dice_iou_metrics(logits, y, threshold=threshold)

        total_loss += float(loss.item())
        total_dice += d
        total_iou += i

    n = max(1, len(loader))
    return total_loss / n, total_dice / n, total_iou / n


def build_loaders(train_df, val_df, pre, patch_size, batch_size, num_workers):
    train_ds = CBISSegDataset(train_df, pre, patch_size=patch_size, do_aug=True)
    val_ds = CBISSegDataset(val_df, pre, patch_size=patch_size, do_aug=False)

    kwargs = dict(
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=True,
    )
    if num_workers > 0:
        kwargs["persistent_workers"] = True
        kwargs["prefetch_factor"] = 2

    train_loader = DataLoader(train_ds, shuffle=True, **kwargs)
    val_loader = DataLoader(val_ds, shuffle=False, batch_size=batch_size, num_workers=num_workers,
                            pin_memory=torch.cuda.is_available(), drop_last=False,
                            persistent_workers=(num_workers > 0), prefetch_factor=2 if num_workers > 0 else None)
    return train_loader, val_loader


# ----------------------------
# Main
# ----------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="preprocessed_output", help="Folder with manifest.csv and split manifests.")
    parser.add_argument("--base_path", type=str, default=None, help="CBIS-DDSM dataset root (optional; auto-detect).")

    parser.add_argument("--model", type=str, choices=["unet", "attunet"], default="unet")
    parser.add_argument("--base_channels", type=int, default=32)

    parser.add_argument("--patch_size", type=int, default=598)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=30)

    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)

    parser.add_argument("--bce_w", type=float, default=0.3)
    parser.add_argument("--dice_w", type=float, default=0.7)
    parser.add_argument("--grad_clip", type=float, default=1.0)

    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--deterministic", action="store_true")

    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--no_amp", action="store_true")

    parser.add_argument("--threshold", type=float, default=0.5, help="Eval threshold (will be overridden if --tune_threshold).")
    parser.add_argument("--tune_threshold", action="store_true", help="Find best threshold on val after training.")
    parser.add_argument("--pos_weight", type=float, default=0.0, help="If 0, estimate automatically from train patches.")

    args = parser.parse_args()

    set_seed(args.seed, deterministic=args.deterministic)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp = (device.type == "cuda") and (not args.no_amp)
    scaler = GradScaler() if amp else None

    # Preprocessor
    pre = CBISDDSM_Preprocessor(base_path=args.base_path, output_path=args.data_dir)

    # Load manifests (expects you ran preprocess_cbis.py first)
    train_csv = os.path.join(args.data_dir, "train_manifest.csv")
    val_csv = os.path.join(args.data_dir, "val_manifest.csv")
    test_csv = os.path.join(args.data_dir, "test_manifest.csv")

    if not (os.path.isfile(train_csv) and os.path.isfile(val_csv) and os.path.isfile(test_csv)):
        raise FileNotFoundError(
            f"Missing split manifests in {args.data_dir}. Run preprocess_cbis.py first to generate train/val/test CSVs."
        )

    train_df = pd.read_csv(train_csv)
    val_df = pd.read_csv(val_csv)
    test_df = pd.read_csv(test_csv)

    train_loader, val_loader = build_loaders(train_df, val_df, pre, args.patch_size, args.batch_size, args.num_workers)

    # Estimate pos_weight (pixel imbalance) unless user provided one
    pw = args.pos_weight if args.pos_weight and args.pos_weight > 0 else estimate_pos_weight(train_df, pre, args.patch_size)
    print(f">> Using BCE pos_weight={pw:.3f}")

    pos_weight = torch.tensor([pw], device=device)
    bce_loss = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    # Model
    if args.model == "unet":
        model = UNet(in_channels=1, out_channels=1, base_channels=args.base_channels)
    else:
        model = AttentionUNet(in_channels=1, out_channels=1, base_channels=args.base_channels)

    model.to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    # OneCycleLR generally works well for UNet-style training on limited data
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=args.lr,
        epochs=args.epochs,
        steps_per_epoch=len(train_loader),
        pct_start=0.1,
        div_factor=10.0,
        final_div_factor=100.0,
    )

    best_val_dice = -1.0
    best_path = os.path.join(args.data_dir, f"best_{args.model}.pt")
    os.makedirs(args.data_dir, exist_ok=True)

    for epoch in range(1, args.epochs + 1):
        tr_loss = train_one_epoch(
            model, train_loader, optimizer, scheduler, bce_loss, device, scaler, amp,
            bce_w=args.bce_w, dice_w=args.dice_w, grad_clip=args.grad_clip,
        )
        val_loss, val_dice, val_iou = evaluate(
            model, val_loader, bce_loss, device,
            threshold=args.threshold, bce_w=args.bce_w, dice_w=args.dice_w
        )

        lr_now = optimizer.param_groups[0]["lr"]
        print(f"Epoch {epoch:03d}/{args.epochs} | lr={lr_now:.2e} | train_loss={tr_loss:.4f} | val_loss={val_loss:.4f} | val_dice={val_dice:.4f} | val_iou={val_iou:.4f}")

        if val_dice > best_val_dice:
            best_val_dice = val_dice
            torch.save(
                {"model": model.state_dict(), "args": vars(args), "pos_weight": pw, "best_val_dice": best_val_dice},
                best_path,
            )

    # Load best
    ckpt = torch.load(best_path, map_location=device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    # Optionally tune threshold on val
    thr = args.threshold
    if args.tune_threshold:
        thr = find_best_threshold(model, val_loader, device)
        print(f">> Best threshold on val: {thr:.2f}")

    # Evaluate on test
    test_loader = DataLoader(
        CBISSegDataset(test_df, pre, patch_size=args.patch_size, do_aug=False),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=False,
        persistent_workers=(args.num_workers > 0),
        prefetch_factor=2 if args.num_workers > 0 else None,
    )
    test_loss, test_dice, test_iou = evaluate(model, test_loader, bce_loss, device, threshold=thr, bce_w=args.bce_w, dice_w=args.dice_w)
    print(f">> TEST @thr={thr:.2f} | loss={test_loss:.4f} | dice={test_dice:.4f} | iou={test_iou:.4f}")

    # Save final metrics
    metrics_path = os.path.join(args.data_dir, f"metrics_{args.model}.json")
    with open(metrics_path, "w") as f:
        import json
        json.dump(
            {
                "model": args.model,
                "best_val_dice": float(ckpt.get("best_val_dice", best_val_dice)),
                "pos_weight": float(ckpt.get("pos_weight", pw)),
                "threshold": float(thr),
                "test_loss": float(test_loss),
                "test_dice": float(test_dice),
                "test_iou": float(test_iou),
            },
            f,
            indent=2,
        )
    print(">> Saved:", best_path)
    print(">> Saved:", metrics_path)


if __name__ == "__main__":
    main()
