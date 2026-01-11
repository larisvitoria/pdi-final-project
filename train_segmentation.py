import argparse
import os
import random
import sys
from typing import Tuple, Dict

import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import Dataset, DataLoader

from preprocess_cbis import CBISDDSM_Preprocessor


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def dice_loss(logits: torch.Tensor, targets: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    probs = torch.sigmoid(logits)
    dims = (1, 2, 3)
    intersection = (probs * targets).sum(dims)
    union = probs.sum(dims) + targets.sum(dims)
    dice = (2 * intersection + eps) / (union + eps)
    return 1 - dice.mean()


def dice_iou_metrics(logits: torch.Tensor, targets: torch.Tensor, eps: float = 1e-6) -> Tuple[float, float]:
    probs = torch.sigmoid(logits)
    preds = (probs > 0.5).float()
    dims = (1, 2, 3)
    intersection = (preds * targets).sum(dims)
    union = preds.sum(dims) + targets.sum(dims)
    dice = (2 * intersection + eps) / (union + eps)
    iou = (intersection + eps) / (preds.sum(dims) + targets.sum(dims) - intersection + eps)
    return dice.mean().item(), iou.mean().item()


def dice_np(pred: np.ndarray, target: np.ndarray, eps: float = 1e-6) -> float:
    pred = pred.astype(np.float32)
    target = target.astype(np.float32)
    intersection = float((pred * target).sum())
    union = float(pred.sum() + target.sum())
    return (2 * intersection + eps) / (union + eps)


class CBISSegDataset(Dataset):
    def __init__(self, df: pd.DataFrame, preprocessor: CBISDDSM_Preprocessor, augment: bool, patch_size: int):
        self.df = df.reset_index(drop=True)
        self.preprocessor = preprocessor
        self.augment = augment
        self.patch_size = patch_size

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int):
        if len(self.df) == 0:
            raise IndexError("Empty dataset")

        for attempt in range(len(self.df)):
            row = self.df.iloc[(idx + attempt) % len(self.df)]
            try:
                img, mask = self.preprocessor.preprocess_pair(row, do_aug=self.augment, patch_size=self.patch_size)
                x = torch.from_numpy(img).float().unsqueeze(0)
                y = torch.from_numpy(mask).float().unsqueeze(0)
                return x, y
            except Exception:
                continue

        raise RuntimeError("Failed to load any sample from dataset")


class DoubleConv(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.seq = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.seq(x)


class UNet(nn.Module):
    def __init__(self, in_channels: int = 1, out_channels: int = 1, base_channels: int = 32):
        super().__init__()
        self.enc1 = DoubleConv(in_channels, base_channels)
        self.enc2 = DoubleConv(base_channels, base_channels * 2)
        self.enc3 = DoubleConv(base_channels * 2, base_channels * 4)
        self.enc4 = DoubleConv(base_channels * 4, base_channels * 8)
        self.pool = nn.MaxPool2d(2)
        self.bottleneck = DoubleConv(base_channels * 8, base_channels * 16)

        self.up4 = nn.ConvTranspose2d(base_channels * 16, base_channels * 8, kernel_size=2, stride=2)
        self.dec4 = DoubleConv(base_channels * 16, base_channels * 8)
        self.up3 = nn.ConvTranspose2d(base_channels * 8, base_channels * 4, kernel_size=2, stride=2)
        self.dec3 = DoubleConv(base_channels * 8, base_channels * 4)
        self.up2 = nn.ConvTranspose2d(base_channels * 4, base_channels * 2, kernel_size=2, stride=2)
        self.dec2 = DoubleConv(base_channels * 4, base_channels * 2)
        self.up1 = nn.ConvTranspose2d(base_channels * 2, base_channels, kernel_size=2, stride=2)
        self.dec1 = DoubleConv(base_channels * 2, base_channels)

        self.out_conv = nn.Conv2d(base_channels, out_channels, kernel_size=1)

    def _pad_input(self, x: torch.Tensor) -> Tuple[torch.Tensor, int, int]:
        _, _, h, w = x.shape
        pad_h = (16 - h % 16) % 16
        pad_w = (16 - w % 16) % 16
        if pad_h or pad_w:
            x = F.pad(x, (0, pad_w, 0, pad_h), mode="constant", value=0)
        return x, pad_h, pad_w

    def _match_size(self, src: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if src.shape[2:] != target.shape[2:]:
            src = F.interpolate(src, size=target.shape[2:], mode="bilinear", align_corners=False)
        return src

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        orig_h, orig_w = x.shape[2], x.shape[3]
        x, pad_h, pad_w = self._pad_input(x)

        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))
        b = self.bottleneck(self.pool(e4))

        d4 = self.up4(b)
        d4 = self._match_size(d4, e4)
        d4 = self.dec4(torch.cat([e4, d4], dim=1))

        d3 = self.up3(d4)
        d3 = self._match_size(d3, e3)
        d3 = self.dec3(torch.cat([e3, d3], dim=1))

        d2 = self.up2(d3)
        d2 = self._match_size(d2, e2)
        d2 = self.dec2(torch.cat([e2, d2], dim=1))

        d1 = self.up1(d2)
        d1 = self._match_size(d1, e1)
        d1 = self.dec1(torch.cat([e1, d1], dim=1))

        out = self.out_conv(d1)
        if pad_h or pad_w:
            out = out[..., :orig_h, :orig_w]
        return out


class AttentionGate(nn.Module):
    def __init__(self, F_g: int, F_l: int, F_int: int):
        super().__init__()
        self.W_g = nn.Sequential(nn.Conv2d(F_g, F_int, kernel_size=1), nn.BatchNorm2d(F_int))
        self.W_x = nn.Sequential(nn.Conv2d(F_l, F_int, kernel_size=1), nn.BatchNorm2d(F_int))
        self.psi = nn.Sequential(nn.Conv2d(F_int, 1, kernel_size=1), nn.BatchNorm2d(1), nn.Sigmoid())
        self.relu = nn.ReLU(inplace=True)

    def forward(self, g: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        if g.shape[2:] != x.shape[2:]:
            g = F.interpolate(g, size=x.shape[2:], mode="bilinear", align_corners=False)
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = self.relu(g1 + x1)
        psi = self.psi(psi)
        return x * psi


class AttentionUNet(nn.Module):
    def __init__(self, in_channels: int = 1, out_channels: int = 1, base_channels: int = 32):
        super().__init__()
        self.enc1 = DoubleConv(in_channels, base_channels)
        self.enc2 = DoubleConv(base_channels, base_channels * 2)
        self.enc3 = DoubleConv(base_channels * 2, base_channels * 4)
        self.enc4 = DoubleConv(base_channels * 4, base_channels * 8)
        self.pool = nn.MaxPool2d(2)
        self.bottleneck = DoubleConv(base_channels * 8, base_channels * 16)

        self.up4 = nn.ConvTranspose2d(base_channels * 16, base_channels * 8, kernel_size=2, stride=2)
        self.att4 = AttentionGate(base_channels * 8, base_channels * 8, base_channels * 4)
        self.dec4 = DoubleConv(base_channels * 16, base_channels * 8)

        self.up3 = nn.ConvTranspose2d(base_channels * 8, base_channels * 4, kernel_size=2, stride=2)
        self.att3 = AttentionGate(base_channels * 4, base_channels * 4, base_channels * 2)
        self.dec3 = DoubleConv(base_channels * 8, base_channels * 4)

        self.up2 = nn.ConvTranspose2d(base_channels * 4, base_channels * 2, kernel_size=2, stride=2)
        self.att2 = AttentionGate(base_channels * 2, base_channels * 2, base_channels)
        self.dec2 = DoubleConv(base_channels * 4, base_channels * 2)

        self.up1 = nn.ConvTranspose2d(base_channels * 2, base_channels, kernel_size=2, stride=2)
        self.att1 = AttentionGate(base_channels, base_channels, base_channels // 2)
        self.dec1 = DoubleConv(base_channels * 2, base_channels)

        self.out_conv = nn.Conv2d(base_channels, out_channels, kernel_size=1)

    def _pad_input(self, x: torch.Tensor) -> Tuple[torch.Tensor, int, int]:
        _, _, h, w = x.shape
        pad_h = (16 - h % 16) % 16
        pad_w = (16 - w % 16) % 16
        if pad_h or pad_w:
            x = F.pad(x, (0, pad_w, 0, pad_h), mode="constant", value=0)
        return x, pad_h, pad_w

    def _match_size(self, src: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if src.shape[2:] != target.shape[2:]:
            src = F.interpolate(src, size=target.shape[2:], mode="bilinear", align_corners=False)
        return src

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        orig_h, orig_w = x.shape[2], x.shape[3]
        x, pad_h, pad_w = self._pad_input(x)

        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))
        b = self.bottleneck(self.pool(e4))

        d4 = self.up4(b)
        d4 = self._match_size(d4, e4)
        e4_att = self.att4(d4, e4)
        d4 = self.dec4(torch.cat([e4_att, d4], dim=1))

        d3 = self.up3(d4)
        d3 = self._match_size(d3, e3)
        e3_att = self.att3(d3, e3)
        d3 = self.dec3(torch.cat([e3_att, d3], dim=1))

        d2 = self.up2(d3)
        d2 = self._match_size(d2, e2)
        e2_att = self.att2(d2, e2)
        d2 = self.dec2(torch.cat([e2_att, d2], dim=1))

        d1 = self.up1(d2)
        d1 = self._match_size(d1, e1)
        e1_att = self.att1(d1, e1)
        d1 = self.dec1(torch.cat([e1_att, d1], dim=1))

        out = self.out_conv(d1)
        if pad_h or pad_w:
            out = out[..., :orig_h, :orig_w]
        return out


def build_loaders(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    preprocessor: CBISDDSM_Preprocessor,
    patch_size: int,
    batch_size: int,
    num_workers: int,
    device: torch.device,
):
    pin_memory = device.type == "cuda"
    train_ds = CBISSegDataset(train_df, preprocessor, augment=True, patch_size=patch_size)
    val_ds = CBISSegDataset(val_df, preprocessor, augment=False, patch_size=patch_size)
    test_ds = CBISSegDataset(test_df, preprocessor, augment=False, patch_size=patch_size)

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    return train_loader, val_loader, test_loader


def train_one_epoch(model, loader, optimizer, scaler, bce_loss, device, use_amp: bool) -> float:
    model.train()
    total_loss = 0.0
    for x, y in loader:
        x = x.to(device)
        y = y.to(device)
        optimizer.zero_grad(set_to_none=True)
        with autocast(enabled=use_amp):
            logits = model(x)
            bce = bce_loss(logits, y)
            d_loss = dice_loss(logits, y)
            loss = 0.5 * bce + 0.5 * d_loss
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        total_loss += loss.item() * x.size(0)
    return total_loss / max(len(loader.dataset), 1)


def evaluate(model, loader, bce_loss, device, use_amp: bool) -> Tuple[float, float, float]:
    model.eval()
    total_loss = 0.0
    total_dice = 0.0
    total_iou = 0.0
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            y = y.to(device)
            with autocast(enabled=use_amp):
                logits = model(x)
                bce = bce_loss(logits, y)
                d_loss = dice_loss(logits, y)
                loss = 0.5 * bce + 0.5 * d_loss
            dice, iou = dice_iou_metrics(logits, y)
            total_loss += loss.item() * x.size(0)
            total_dice += dice * x.size(0)
            total_iou += iou * x.size(0)
    n = max(len(loader.dataset), 1)
    return total_loss / n, total_dice / n, total_iou / n


def save_overlay_triplet(img: np.ndarray, gt: np.ndarray, pred: np.ndarray, out_path: str):
    img_u8 = (img * 255).astype(np.uint8)
    base = np.stack([img_u8, img_u8, img_u8], axis=-1)

    gt_bgr = base.copy()
    gt_bgr[gt > 0.5, 2] = 255

    pred_bgr = base.copy()
    pred_bgr[pred > 0.5, 2] = 255

    triplet = np.concatenate([base, gt_bgr, pred_bgr], axis=1)
    cv2.imwrite(out_path, triplet)


def generate_predictions(
    model: nn.Module,
    df: pd.DataFrame,
    preprocessor: CBISDDSM_Preprocessor,
    device: torch.device,
    patch_size: int,
    out_dir: str,
    max_samples: int = 50,
    worst_k: int = 20,
):
    os.makedirs(out_dir, exist_ok=True)
    worst_dir = os.path.join(out_dir, "worst")
    os.makedirs(worst_dir, exist_ok=True)

    model.eval()
    worst_samples = []

    with torch.no_grad():
        for idx, row in df.iterrows():
            try:
                img, mask = preprocessor.preprocess_pair(row, do_aug=False, patch_size=patch_size)
            except Exception:
                continue

            x = torch.from_numpy(img).unsqueeze(0).unsqueeze(0).to(device)
            logits = model(x)
            probs = torch.sigmoid(logits).cpu().numpy()[0, 0]
            pred = (probs > 0.5).astype(np.float32)
            d = dice_np(pred, mask)

            if len(worst_samples) < worst_k:
                worst_samples.append((d, row, img, mask, pred))
            else:
                worst_samples.sort(key=lambda t: t[0])
                if d < worst_samples[-1][0]:
                    worst_samples[-1] = (d, row, img, mask, pred)

            if idx < max_samples:
                out_path = os.path.join(out_dir, f"pred_{idx:04d}_{row['patient_id']}.png")
                save_overlay_triplet(img, mask, pred, out_path)

    worst_samples.sort(key=lambda t: t[0])
    for rank, (d, row, img, mask, pred) in enumerate(worst_samples[:worst_k]):
        out_path = os.path.join(worst_dir, f"worst_{rank:02d}_dice_{d:.4f}_{row['patient_id']}.png")
        save_overlay_triplet(img, mask, pred, out_path)


def run_training(
    model_name: str,
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
    output_path: str,
    epochs: int,
):
    set_seed(42)
    model = model.to(device)
    use_amp = device.type == "cuda"
    scaler = GradScaler(enabled=use_amp)
    bce_loss = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", patience=5, factor=0.5)

    best_dice = -1.0
    ckpt_dir = os.path.join(output_path, "checkpoints")
    os.makedirs(ckpt_dir, exist_ok=True)
    best_path = os.path.join(ckpt_dir, f"best_{model_name}.pt")

    for epoch in range(1, epochs + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, scaler, bce_loss, device, use_amp)
        val_loss, val_dice, val_iou = evaluate(model, val_loader, bce_loss, device, use_amp)
        scheduler.step(val_dice)

        lr = optimizer.param_groups[0]["lr"]
        print(
            f"{model_name} | epoch {epoch:03d}/{epochs} "
            f"train_loss={train_loss:.4f} val_loss={val_loss:.4f} "
            f"val_dice={val_dice:.4f} val_iou={val_iou:.4f} lr={lr:.6f}"
        )

        if val_dice > best_dice:
            best_dice = val_dice
            torch.save(
                {
                    "model": model.state_dict(),
                    "val_dice": val_dice,
                    "epoch": epoch,
                },
                best_path,
            )

    return best_path


def load_checkpoint(model: nn.Module, checkpoint_path: str, device: torch.device):
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model"])


def evaluate_by_abnormality(
    model: nn.Module,
    df: pd.DataFrame,
    preprocessor: CBISDDSM_Preprocessor,
    patch_size: int,
    batch_size: int,
    num_workers: int,
    device: torch.device,
    abnormality: str,
) -> Tuple[float, float]:
    subset = df[df["abnormality_type"].str.lower() == abnormality]
    if subset.empty:
        return 0.0, 0.0
    ds = CBISSegDataset(subset, preprocessor, augment=False, patch_size=patch_size)
    loader = DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
    )
    bce_loss = nn.BCEWithLogitsLoss()
    _, dice, iou = evaluate(model, loader, bce_loss, device, use_amp=device.type == "cuda")
    return dice, iou


def main():
    parser = argparse.ArgumentParser(description="Train UNet and Attention UNet for CBIS-DDSM segmentation")
    parser.add_argument("--base_path", type=str, default=None)
    parser.add_argument("--output_path", type=str, default=None)
    parser.add_argument("--patch_size", type=int, default=598)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--base_channels", type=int, default=32)
    args = parser.parse_args()

    preprocessor = CBISDDSM_Preprocessor(base_path=args.base_path, output_path=args.output_path)
    output_path = preprocessor.output_path

    train_path = os.path.join(output_path, "train_manifest.csv")
    val_path = os.path.join(output_path, "val_manifest.csv")
    test_path = os.path.join(output_path, "test_manifest.csv")

    if not (os.path.isfile(train_path) and os.path.isfile(val_path) and os.path.isfile(test_path)):
        print("ERROR: Missing manifest splits. Run preprocess_cbis.py first to generate train/val/test CSVs.")
        sys.exit(1)

    train_df = pd.read_csv(train_path)
    val_df = pd.read_csv(val_path)
    test_df = pd.read_csv(test_path)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_loader, val_loader, test_loader = build_loaders(
        train_df,
        val_df,
        test_df,
        preprocessor,
        args.patch_size,
        args.batch_size,
        args.num_workers,
        device,
    )

    results: Dict[str, Dict[str, float]] = {}
    for model_name in ["unet", "att_unet"]:
        if model_name == "unet":
            model = UNet(in_channels=1, out_channels=1, base_channels=args.base_channels)
        else:
            model = AttentionUNet(in_channels=1, out_channels=1, base_channels=args.base_channels)

        print(f"\n>> Training {model_name}...")
        best_ckpt = run_training(
            model_name,
            model,
            train_loader,
            val_loader,
            device,
            output_path,
            args.epochs,
        )

        load_checkpoint(model, best_ckpt, device)
        bce_loss = nn.BCEWithLogitsLoss()
        test_loss, test_dice, test_iou = evaluate(model, test_loader, bce_loss, device, use_amp=device.type == "cuda")

        mass_dice, mass_iou = evaluate_by_abnormality(
            model,
            test_df,
            preprocessor,
            args.patch_size,
            args.batch_size,
            args.num_workers,
            device,
            abnormality="mass",
        )
        calc_dice, calc_iou = evaluate_by_abnormality(
            model,
            test_df,
            preprocessor,
            args.patch_size,
            args.batch_size,
            args.num_workers,
            device,
            abnormality="calcification",
        )

        results[model_name] = {
            "test_loss": test_loss,
            "test_dice": test_dice,
            "test_iou": test_iou,
            "mass_dice": mass_dice,
            "mass_iou": mass_iou,
            "calc_dice": calc_dice,
            "calc_iou": calc_iou,
        }

        preds_dir = os.path.join(output_path, "preds", model_name)
        print(f">> Saving predictions to {preds_dir}")
        generate_predictions(
            model,
            test_df,
            preprocessor,
            device,
            args.patch_size,
            preds_dir,
            max_samples=50,
            worst_k=20,
        )

    print("\nFinal comparison (test):")
    print("model\tDice\tIoU\tDice_mass\tIoU_mass\tDice_calc\tIoU_calc")
    for model_name, metrics in results.items():
        print(
            f"{model_name}\t"
            f"{metrics['test_dice']:.4f}\t"
            f"{metrics['test_iou']:.4f}\t"
            f"{metrics['mass_dice']:.4f}\t"
            f"{metrics['mass_iou']:.4f}\t"
            f"{metrics['calc_dice']:.4f}\t"
            f"{metrics['calc_iou']:.4f}"
        )


if __name__ == "__main__":
    main()
