import os
import json
import argparse
from dataclasses import dataclass
from typing import Tuple, Dict

import numpy as np
import pandas as pd
import cv2
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models as tv_models
from torch.utils.data import Dataset, DataLoader


# -----------------------------
# Models
# -----------------------------
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


class ResidualBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_ch)
        self.act = nn.ReLU(inplace=True)
        self.skip = None
        if in_ch != out_ch:
            self.skip = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 1, bias=False),
                nn.BatchNorm2d(out_ch),
            )

    def forward(self, x):
        identity = x if self.skip is None else self.skip(x)
        out = self.act(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = self.act(out + identity)
        return out


class UNetPlusPlus(nn.Module):
    def __init__(self, in_ch=1, base=32):
        super().__init__()
        self.pool = nn.MaxPool2d(2)

        self.conv0_0 = DoubleConv(in_ch, base)
        self.conv1_0 = DoubleConv(base, base * 2)
        self.conv2_0 = DoubleConv(base * 2, base * 4)
        self.conv3_0 = DoubleConv(base * 4, base * 8)
        self.conv4_0 = DoubleConv(base * 8, base * 16)

        self.conv0_1 = DoubleConv(base * 3, base)
        self.conv1_1 = DoubleConv(base * 6, base * 2)
        self.conv2_1 = DoubleConv(base * 12, base * 4)
        self.conv3_1 = DoubleConv(base * 24, base * 8)

        self.conv0_2 = DoubleConv(base * 4, base)
        self.conv1_2 = DoubleConv(base * 8, base * 2)
        self.conv2_2 = DoubleConv(base * 16, base * 4)

        self.conv0_3 = DoubleConv(base * 5, base)
        self.conv1_3 = DoubleConv(base * 10, base * 2)

        self.conv0_4 = DoubleConv(base * 6, base)

        self.out = nn.Conv2d(base, 1, 1)

    @staticmethod
    def _up(x: torch.Tensor, ref: torch.Tensor) -> torch.Tensor:
        return F.interpolate(x, size=ref.shape[2:], mode="bilinear", align_corners=False)

    def forward(self, x):
        x0_0 = self.conv0_0(x)
        x1_0 = self.conv1_0(self.pool(x0_0))
        x2_0 = self.conv2_0(self.pool(x1_0))
        x3_0 = self.conv3_0(self.pool(x2_0))
        x4_0 = self.conv4_0(self.pool(x3_0))

        x0_1 = self.conv0_1(torch.cat([x0_0, self._up(x1_0, x0_0)], dim=1))
        x1_1 = self.conv1_1(torch.cat([x1_0, self._up(x2_0, x1_0)], dim=1))
        x2_1 = self.conv2_1(torch.cat([x2_0, self._up(x3_0, x2_0)], dim=1))
        x3_1 = self.conv3_1(torch.cat([x3_0, self._up(x4_0, x3_0)], dim=1))

        x0_2 = self.conv0_2(torch.cat([x0_0, x0_1, self._up(x1_1, x0_0)], dim=1))
        x1_2 = self.conv1_2(torch.cat([x1_0, x1_1, self._up(x2_1, x1_0)], dim=1))
        x2_2 = self.conv2_2(torch.cat([x2_0, x2_1, self._up(x3_1, x2_0)], dim=1))

        x0_3 = self.conv0_3(torch.cat([x0_0, x0_1, x0_2, self._up(x1_2, x0_0)], dim=1))
        x1_3 = self.conv1_3(torch.cat([x1_0, x1_1, x1_2, self._up(x2_2, x1_0)], dim=1))

        x0_4 = self.conv0_4(torch.cat([x0_0, x0_1, x0_2, x0_3, self._up(x1_3, x0_0)], dim=1))

        return self.out(x0_4)


class ConvBNReLU(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size=3, stride=1, padding=1, dilation=1):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size, stride, padding, dilation=dilation, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class ASPP(nn.Module):
    def __init__(self, in_ch, out_ch, rates=(6, 12, 18)):
        super().__init__()
        self.conv1 = ConvBNReLU(in_ch, out_ch, kernel_size=1, padding=0)
        self.conv2 = ConvBNReLU(in_ch, out_ch, kernel_size=3, padding=rates[0], dilation=rates[0])
        self.conv3 = ConvBNReLU(in_ch, out_ch, kernel_size=3, padding=rates[1], dilation=rates[1])
        self.conv4 = ConvBNReLU(in_ch, out_ch, kernel_size=3, padding=rates[2], dilation=rates[2])
        self.image_pool = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            ConvBNReLU(in_ch, out_ch, kernel_size=1, padding=0),
        )
        self.project = ConvBNReLU(out_ch * 5, out_ch, kernel_size=1, padding=0)

    def forward(self, x):
        size = x.shape[2:]
        x1 = self.conv1(x)
        x2 = self.conv2(x)
        x3 = self.conv3(x)
        x4 = self.conv4(x)
        x5 = self.image_pool(x)
        x5 = F.interpolate(x5, size=size, mode="bilinear", align_corners=False)
        x = torch.cat([x1, x2, x3, x4, x5], dim=1)
        return self.project(x)


class DeepLabV3Plus(nn.Module):
    def __init__(self, in_ch=1, base_channels=256):
        super().__init__()
        self.backbone = tv_models.resnet50(weights=None)
        if in_ch != 3:
            self.backbone.conv1 = nn.Conv2d(in_ch, 64, kernel_size=7, stride=2, padding=3, bias=False)

        self.aspp = ASPP(2048, base_channels)
        self.low_proj = ConvBNReLU(256, 48, kernel_size=1, padding=0)
        self.decoder = nn.Sequential(
            ConvBNReLU(base_channels + 48, base_channels, kernel_size=3, padding=1),
            ConvBNReLU(base_channels, base_channels, kernel_size=3, padding=1),
        )
        self.out = nn.Conv2d(base_channels, 1, kernel_size=1)

    def forward(self, x):
        input_size = x.shape[2:]
        x = self.backbone.conv1(x)
        x = self.backbone.bn1(x)
        x = self.backbone.relu(x)
        x = self.backbone.maxpool(x)

        low = self.backbone.layer1(x)
        x = self.backbone.layer2(low)
        x = self.backbone.layer3(x)
        x = self.backbone.layer4(x)

        x = self.aspp(x)
        x = F.interpolate(x, size=low.shape[2:], mode="bilinear", align_corners=False)
        low = self.low_proj(low)
        x = torch.cat([x, low], dim=1)
        x = self.decoder(x)
        x = self.out(x)
        x = F.interpolate(x, size=input_size, mode="bilinear", align_corners=False)
        return x


class UNet(nn.Module):
    def __init__(self, in_ch=1, base=32):
        super().__init__()
        self.d1 = DoubleConv(in_ch, base)
        self.p1 = nn.MaxPool2d(2)
        self.d2 = DoubleConv(base, base * 2)
        self.p2 = nn.MaxPool2d(2)
        self.d3 = DoubleConv(base * 2, base * 4)
        self.p3 = nn.MaxPool2d(2)
        self.d4 = DoubleConv(base * 4, base * 8)
        self.p4 = nn.MaxPool2d(2)

        self.b = DoubleConv(base * 8, base * 16)

        self.u4 = nn.ConvTranspose2d(base * 16, base * 8, 2, stride=2)
        self.c4 = DoubleConv(base * 16, base * 8)
        self.u3 = nn.ConvTranspose2d(base * 8, base * 4, 2, stride=2)
        self.c3 = DoubleConv(base * 8, base * 4)
        self.u2 = nn.ConvTranspose2d(base * 4, base * 2, 2, stride=2)
        self.c2 = DoubleConv(base * 4, base * 2)
        self.u1 = nn.ConvTranspose2d(base * 2, base, 2, stride=2)
        self.c1 = DoubleConv(base * 2, base)

        self.out = nn.Conv2d(base, 1, 1)

    def forward(self, x):
        d1 = self.d1(x)
        d2 = self.d2(self.p1(d1))
        d3 = self.d3(self.p2(d2))
        d4 = self.d4(self.p3(d3))
        b = self.b(self.p4(d4))

        x = self.u4(b)
        x = self.c4(torch.cat([x, d4], dim=1))
        x = self.u3(x)
        x = self.c3(torch.cat([x, d3], dim=1))
        x = self.u2(x)
        x = self.c2(torch.cat([x, d2], dim=1))
        x = self.u1(x)
        x = self.c1(torch.cat([x, d1], dim=1))
        return self.out(x)


class AttentionGate(nn.Module):
    def __init__(self, g_ch, x_ch, inter_ch):
        super().__init__()
        self.Wg = nn.Sequential(nn.Conv2d(g_ch, inter_ch, 1, bias=False), nn.BatchNorm2d(inter_ch))
        self.Wx = nn.Sequential(nn.Conv2d(x_ch, inter_ch, 1, bias=False), nn.BatchNorm2d(inter_ch))
        self.psi = nn.Sequential(nn.Conv2d(inter_ch, 1, 1, bias=True), nn.BatchNorm2d(1), nn.Sigmoid())
        self.act = nn.ReLU(inplace=True)

    def forward(self, g, x):
        psi = self.act(self.Wg(g) + self.Wx(x))
        psi = self.psi(psi)
        return x * psi


class AttentionUNet(nn.Module):
    def __init__(self, in_ch=1, base=32):
        super().__init__()
        self.d1 = DoubleConv(in_ch, base)
        self.p1 = nn.MaxPool2d(2)
        self.d2 = DoubleConv(base, base * 2)
        self.p2 = nn.MaxPool2d(2)
        self.d3 = DoubleConv(base * 2, base * 4)
        self.p3 = nn.MaxPool2d(2)
        self.d4 = DoubleConv(base * 4, base * 8)
        self.p4 = nn.MaxPool2d(2)

        self.b = DoubleConv(base * 8, base * 16)

        self.u4 = nn.ConvTranspose2d(base * 16, base * 8, 2, stride=2)
        self.ag4 = AttentionGate(base * 8, base * 8, base * 4)
        self.c4 = DoubleConv(base * 16, base * 8)

        self.u3 = nn.ConvTranspose2d(base * 8, base * 4, 2, stride=2)
        self.ag3 = AttentionGate(base * 4, base * 4, base * 2)
        self.c3 = DoubleConv(base * 8, base * 4)

        self.u2 = nn.ConvTranspose2d(base * 4, base * 2, 2, stride=2)
        self.ag2 = AttentionGate(base * 2, base * 2, base)
        self.c2 = DoubleConv(base * 4, base * 2)

        self.u1 = nn.ConvTranspose2d(base * 2, base, 2, stride=2)
        self.ag1 = AttentionGate(base, base, max(base // 2, 1))
        self.c1 = DoubleConv(base * 2, base)

        self.out = nn.Conv2d(base, 1, 1)

    def forward(self, x):
        d1 = self.d1(x)
        d2 = self.d2(self.p1(d1))
        d3 = self.d3(self.p2(d2))
        d4 = self.d4(self.p3(d3))
        b = self.b(self.p4(d4))

        x = self.u4(b)
        d4a = self.ag4(x, d4)
        x = self.c4(torch.cat([x, d4a], dim=1))

        x = self.u3(x)
        d3a = self.ag3(x, d3)
        x = self.c3(torch.cat([x, d3a], dim=1))

        x = self.u2(x)
        d2a = self.ag2(x, d2)
        x = self.c2(torch.cat([x, d2a], dim=1))

        x = self.u1(x)
        d1a = self.ag1(x, d1)
        x = self.c1(torch.cat([x, d1a], dim=1))

        return self.out(x)


class ResUNet(nn.Module):
    """Residual U-Net (recommended default for MASS segmentation)."""
    def __init__(self, in_ch=1, base=32):
        super().__init__()
        self.d1 = ResidualBlock(in_ch, base)
        self.p1 = nn.MaxPool2d(2)
        self.d2 = ResidualBlock(base, base * 2)
        self.p2 = nn.MaxPool2d(2)
        self.d3 = ResidualBlock(base * 2, base * 4)
        self.p3 = nn.MaxPool2d(2)
        self.d4 = ResidualBlock(base * 4, base * 8)
        self.p4 = nn.MaxPool2d(2)

        self.b = ResidualBlock(base * 8, base * 16)

        self.u4 = nn.ConvTranspose2d(base * 16, base * 8, 2, stride=2)
        self.c4 = ResidualBlock(base * 16, base * 8)
        self.u3 = nn.ConvTranspose2d(base * 8, base * 4, 2, stride=2)
        self.c3 = ResidualBlock(base * 8, base * 4)
        self.u2 = nn.ConvTranspose2d(base * 4, base * 2, 2, stride=2)
        self.c2 = ResidualBlock(base * 4, base * 2)
        self.u1 = nn.ConvTranspose2d(base * 2, base, 2, stride=2)
        self.c1 = ResidualBlock(base * 2, base)

        self.out = nn.Conv2d(base, 1, 1)

    def forward(self, x):
        d1 = self.d1(x)
        d2 = self.d2(self.p1(d1))
        d3 = self.d3(self.p2(d2))
        d4 = self.d4(self.p3(d3))
        b = self.b(self.p4(d4))

        x = self.u4(b)
        x = self.c4(torch.cat([x, d4], dim=1))
        x = self.u3(x)
        x = self.c3(torch.cat([x, d3], dim=1))
        x = self.u2(x)
        x = self.c2(torch.cat([x, d2], dim=1))
        x = self.u1(x)
        x = self.c1(torch.cat([x, d1], dim=1))
        return self.out(x)


def build_model(name: str, base: int = 32) -> nn.Module:
    name = name.lower()
    if name == "unet":
        return UNet(in_ch=1, base=base)
    if name in ("attunet", "attention_unet", "attentionunet"):
        return AttentionUNet(in_ch=1, base=base)
    if name in ("resunet", "res_unet"):
        return ResUNet(in_ch=1, base=base)
    if name in ("unetpp", "unetplusplus", "unet++"):
        return UNetPlusPlus(in_ch=1, base=base)
    if name in ("deeplabv3p", "deeplabv3plus", "deeplabv3+"):
        return DeepLabV3Plus(in_ch=1)
    raise ValueError(f"Unknown model: {name}")


# -----------------------------
# Losses & metrics
# -----------------------------
def dice_coeff_from_logits(logits: torch.Tensor, targets: torch.Tensor, threshold: float = 0.5, eps: float = 1e-6) -> torch.Tensor:
    probs = torch.sigmoid(logits)
    preds = (probs > threshold).float()
    targets = targets.float()
    intersection = (preds * targets).sum(dim=(1, 2, 3))
    union = preds.sum(dim=(1, 2, 3)) + targets.sum(dim=(1, 2, 3))
    dice = (2.0 * intersection + eps) / (union + eps)
    return dice.mean()


def iou_from_logits(logits: torch.Tensor, targets: torch.Tensor, threshold: float = 0.5, eps: float = 1e-6) -> torch.Tensor:
    probs = torch.sigmoid(logits)
    preds = (probs > threshold).float()
    targets = targets.float()
    intersection = (preds * targets).sum(dim=(1, 2, 3))
    union = (preds + targets - preds * targets).sum(dim=(1, 2, 3))
    iou = (intersection + eps) / (union + eps)
    return iou.mean()


class FocalTverskyLoss(nn.Module):
    """Focal Tversky Loss."""
    def __init__(self, alpha: float = 0.3, beta: float = 0.7, gamma: float = 0.75, eps: float = 1e-6):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.eps = eps

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        probs = torch.sigmoid(logits)
        targets = targets.float()

        probs = probs.view(probs.size(0), -1)
        targets = targets.view(targets.size(0), -1)

        tp = (probs * targets).sum(dim=1)
        fp = (probs * (1 - targets)).sum(dim=1)
        fn = ((1 - probs) * targets).sum(dim=1)

        tversky = (tp + self.eps) / (tp + self.alpha * fp + self.beta * fn + self.eps)
        loss = torch.pow((1 - tversky), self.gamma)
        return loss.mean()


# -----------------------------
# Dataset
# -----------------------------
class MassSegDataset(Dataset):
    def __init__(
        self,
        df: pd.DataFrame,
        base_path: str,
        patch_size: int = 512,
        augment: bool = False,
        clahe: bool = True,
        drop_empty_masks: bool = True,
    ):
        self.df = df.reset_index(drop=True)
        self.base_path = base_path
        self.patch_size = patch_size
        self.augment = augment
        self.clahe = clahe
        self.drop_empty_masks = drop_empty_masks

        import albumentations as A
        self.aug = A.Compose(
            [
                A.HorizontalFlip(p=0.5),
                A.Rotate(limit=10, border_mode=0, p=0.5),
                A.GridDistortion(num_steps=5, distort_limit=0.15, border_mode=0, p=0.25),
                A.ElasticTransform(alpha=30, sigma=5, border_mode=0, p=0.15),
            ]
        )

        if self.drop_empty_masks:
            kept = []
            for i in range(len(self.df)):
                mp = os.path.join(self.base_path, self.df.loc[i, "mask_path"])
                m = cv2.imread(mp, cv2.IMREAD_GRAYSCALE)
                if m is None:
                    continue
                if (m > 127).any():
                    kept.append(i)
            self.df = self.df.iloc[kept].reset_index(drop=True)

    def __len__(self):
        return len(self.df)

    @staticmethod
    def _segment_breast_otsu(img_float01: np.ndarray) -> np.ndarray:
        img_uint8 = (img_float01 * 255).astype(np.uint8)
        _, binary = cv2.threshold(img_uint8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        kernel = np.ones((5, 5), np.uint8)
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return img_float01
        largest = max(contours, key=cv2.contourArea)
        mask = np.zeros_like(img_uint8)
        cv2.drawContours(mask, [largest], -1, 255, thickness=cv2.FILLED)
        out = cv2.bitwise_and(img_uint8, img_uint8, mask=mask)
        return out.astype(np.float32) / 255.0

    @staticmethod
    def _is_breast_on_left(img_uint8: np.ndarray) -> bool:
        h, w = img_uint8.shape
        return np.sum(img_uint8[:, : w // 2]) > np.sum(img_uint8[:, w // 2 :])

    @staticmethod
    def _suppress_pectoral(img_float01: np.ndarray, view: str) -> np.ndarray:
        if view != "MLO":
            return img_float01
        img_uint8 = (img_float01 * 255).astype(np.uint8)
        h, w = img_uint8.shape
        is_left = MassSegDataset._is_breast_on_left(img_uint8)

        edges = cv2.Canny(img_uint8, 30, 100)
        roi_mask = np.zeros_like(edges)
        if is_left:
            roi_mask[0 : h // 2, 0 : w // 2] = 255
        else:
            roi_mask[0 : h // 2, w // 2 : w] = 255
        edges_roi = cv2.bitwise_and(edges, edges, mask=roi_mask)

        lines = cv2.HoughLinesP(edges_roi, 1, np.pi / 180, threshold=25, minLineLength=30, maxLineGap=30)
        if lines is None:
            return img_float01

        best = None
        best_len = 0.0
        for line in lines:
            x1, y1, x2, y2 = line[0]
            if x2 == x1:
                continue
            length = float(np.hypot(x2 - x1, y2 - y1))
            angle = abs(np.arctan2(y2 - y1, x2 - x1) * 180 / np.pi)
            if 20 < angle < 85 and length > best_len:
                best_len = length
                best = (x1, y1, x2, y2)
        if best is None:
            return img_float01

        x1, y1, x2, y2 = best
        m = (y2 - y1) / (x2 - x1)
        b = y1 - m * x1

        tri = np.zeros_like(img_uint8)
        if is_left:
            x_int = int(-b / m) if m != 0 else 0
            y_int = int(b)
            pts = np.array([[[0, 0], [0, min(y_int, h)], [min(x_int, w), 0]]], dtype=np.int32)
        else:
            x_int = int(-b / m) if m != 0 else w
            y_int = int(m * w + b)
            pts = np.array([[[w, 0], [w, min(y_int, h)], [min(x_int, w), 0]]], dtype=np.int32)
        cv2.fillPoly(tri, pts, 255)
        out = cv2.bitwise_and(img_uint8, img_uint8, mask=cv2.bitwise_not(tri))
        return out.astype(np.float32) / 255.0

    @staticmethod
    def _apply_clahe(img_float01: np.ndarray) -> np.ndarray:
        img_uint8 = (img_float01 * 255).astype(np.uint8)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        out = clahe.apply(img_uint8)
        return out.astype(np.float32) / 255.0

    @staticmethod
    def _extract_patch(img: np.ndarray, mask: np.ndarray, patch_size: int) -> Tuple[np.ndarray, np.ndarray]:
        mask_uint8 = (mask * 255).astype(np.uint8)
        moms = cv2.moments(mask_uint8)
        if moms["m00"] == 0:
            cy, cx = img.shape[0] // 2, img.shape[1] // 2
        else:
            cx = int(moms["m10"] / moms["m00"])
            cy = int(moms["m01"] / moms["m00"])

        half = patch_size // 2
        pad = ((half, half), (half, half))
        img_p = np.pad(img, pad, mode="constant", constant_values=0)
        m_p = np.pad(mask, pad, mode="constant", constant_values=0)

        pcx, pcy = cx + half, cy + half
        sx, ex = pcx - half, pcx - half + patch_size
        sy, ey = pcy - half, pcy - half + patch_size

        img_patch = img_p[sy:ey, sx:ex]
        m_patch = m_p[sy:ey, sx:ex]

        if img_patch.shape != (patch_size, patch_size):
            img_patch = cv2.resize(img_patch, (patch_size, patch_size), interpolation=cv2.INTER_LINEAR)
            m_patch = cv2.resize(m_patch, (patch_size, patch_size), interpolation=cv2.INTER_NEAREST)

        m_patch = (m_patch > 0.5).astype(np.float32)
        return img_patch.astype(np.float32), m_patch

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        ip = os.path.join(self.base_path, row["image_path"])
        mp = os.path.join(self.base_path, row["mask_path"])
        view = str(row["view"])

        img = cv2.imread(ip, cv2.IMREAD_GRAYSCALE)
        mask = cv2.imread(mp, cv2.IMREAD_GRAYSCALE)
        if img is None or mask is None:
            raise RuntimeError(f"Failed to read: {ip} / {mp}")

        img = img.astype(np.float32) / 255.0
        mask = (mask > 127).astype(np.float32)
        if mask.shape != img.shape:
            mask = cv2.resize(mask, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_NEAREST)

        img = self._segment_breast_otsu(img)
        img = self._suppress_pectoral(img, view)

        # Force mask background to match cleaned image background
        mask = np.where(img > 0.0, mask, 0.0).astype(np.float32)

        if self.clahe:
            img = self._apply_clahe(img)

        img_patch, mask_patch = self._extract_patch(img, mask, self.patch_size)

        if self.augment:
            out = self.aug(image=img_patch, mask=mask_patch)
            img_patch = out["image"].astype(np.float32)
            mask_patch = (out["mask"] > 0.5).astype(np.float32)

        x = torch.from_numpy(img_patch[None, ...])
        y = torch.from_numpy(mask_patch[None, ...])
        return x, y


# -----------------------------
# Training utilities
# -----------------------------
def compute_pos_weight_from_loader(loader: DataLoader, device: torch.device, max_batches: int = 50) -> torch.Tensor:
    pos = 0.0
    neg = 0.0
    with torch.no_grad():
        for i, (_, y) in enumerate(loader):
            if i >= max_batches:
                break
            y = y.to(device)
            pos += float((y == 1).sum().item())
            neg += float((y == 0).sum().item())
    pos = max(pos, 1.0)
    pw = neg / pos
    return torch.tensor([pw], device=device)


def train_one(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
    epochs: int,
    lr: float,
    weight_decay: float,
    threshold: float,
    tversky_alpha: float,
    tversky_beta: float,
    tversky_gamma: float,
    out_dir: str,
    model_name: str,
    use_amp: bool = True,
) -> Dict:
    os.makedirs(out_dir, exist_ok=True)
    model = model.to(device)

    pos_weight = compute_pos_weight_from_loader(train_loader, device=device)
    bce = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    ftl = FocalTverskyLoss(alpha=tversky_alpha, beta=tversky_beta, gamma=tversky_gamma)

    def loss_fn(logits, targets):
        return 0.25 * bce(logits, targets) + 0.75 * ftl(logits, targets)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    steps_per_epoch = len(train_loader)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=lr,
        epochs=epochs,
        steps_per_epoch=steps_per_epoch,
        pct_start=0.1,
        div_factor=10.0,
        final_div_factor=100.0,
    )

    scaler = torch.cuda.amp.GradScaler(enabled=(use_amp and device.type == "cuda"))

    best_val = -1.0
    best_path = os.path.join(out_dir, f"{model_name}_best.pt")

    for epoch in range(1, epochs + 1):
        model.train()
        tr_losses = []

        for x, y in tqdm(train_loader, desc=f"[{model_name}] Train {epoch}/{epochs}", leave=False):
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=(use_amp and device.type == "cuda")):
                logits = model(x)
                loss = loss_fn(logits, y)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()

            scheduler.step()
            tr_losses.append(float(loss.item()))

        model.eval()
        val_losses = []
        dices = []
        empty_pred_masks = 0
        total_masks = 0
        with torch.no_grad():
            for x, y in tqdm(val_loader, desc=f"[{model_name}] Val {epoch}/{epochs}", leave=False):
                x = x.to(device, non_blocking=True)
                y = y.to(device, non_blocking=True)
                logits = model(x)
                loss = loss_fn(logits, y)
                val_losses.append(float(loss.item()))
                dices.append(float(dice_coeff_from_logits(logits, y, threshold=threshold).item()))
                preds = (torch.sigmoid(logits) > threshold).float()
                empty_pred_masks += int((preds.view(preds.size(0), -1).sum(dim=1) == 0).sum().item())
                total_masks += int(preds.size(0))

        mean_tr = float(np.mean(tr_losses)) if tr_losses else float("nan")
        mean_val = float(np.mean(val_losses)) if val_losses else float("nan")
        mean_dice = float(np.mean(dices)) if dices else 0.0
        empty_ratio = (empty_pred_masks / total_masks) if total_masks > 0 else 0.0

        print(
            f"[{model_name}] epoch={epoch:03d} "
            f"train_loss={mean_tr:.4f} val_loss={mean_val:.4f} "
            f"val_dice@{threshold:.2f}={mean_dice:.4f} "
            f"empty_pred_ratio={empty_ratio:.3f}"
        )

        if mean_dice > best_val:
            best_val = mean_dice
            torch.save(
                {
                    "model_name": model_name,
                    "state_dict": model.state_dict(),
                    "threshold": threshold,
                    "pos_weight": float(pos_weight.item()),
                    "tversky_alpha": tversky_alpha,
                    "tversky_beta": tversky_beta,
                    "tversky_gamma": tversky_gamma,
                },
                best_path,
            )

    return {"best_val_dice": best_val, "pos_weight": float(pos_weight.item()), "threshold": threshold, "best_ckpt": best_path}


def evaluate(
    model: nn.Module,
    ckpt_path: str,
    loader: DataLoader,
    device: torch.device,
    threshold: float,
    tversky_alpha: float,
    tversky_beta: float,
    tversky_gamma: float,
) -> Dict:
    ckpt = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt["state_dict"])
    model = model.to(device)
    model.eval()

    losses = []
    dices = []
    ious = []
    empty_pred_masks = 0
    total_masks = 0

    bce = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([ckpt.get("pos_weight", 1.0)], device=device))
    ftl = FocalTverskyLoss(alpha=tversky_alpha, beta=tversky_beta, gamma=tversky_gamma)

    def loss_fn(logits, targets):
        return 0.25 * bce(logits, targets) + 0.75 * ftl(logits, targets)

    with torch.no_grad():
        for x, y in tqdm(loader, desc="Test", leave=False):
            x = x.to(device)
            y = y.to(device)
            logits = model(x)
            losses.append(float(loss_fn(logits, y).item()))
            dices.append(float(dice_coeff_from_logits(logits, y, threshold=threshold).item()))
            ious.append(float(iou_from_logits(logits, y, threshold=threshold).item()))
            preds = (torch.sigmoid(logits) > threshold).float()
            empty_pred_masks += int((preds.view(preds.size(0), -1).sum(dim=1) == 0).sum().item())
            total_masks += int(preds.size(0))

    empty_ratio = (empty_pred_masks / total_masks) if total_masks > 0 else 0.0
    return {
        "test_loss": float(np.mean(losses)),
        "test_dice": float(np.mean(dices)),
        "test_iou": float(np.mean(ious)),
        "test_empty_pred_ratio": float(empty_ratio),
    }


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base_path", type=str, default="cbis-ddsm-breast-cancer-image-dataset")
    ap.add_argument("--manifest", type=str, default="preprocessed_output/manifest_mass.csv")
    ap.add_argument("--out_dir", type=str, default="seg_runs_mass")
    ap.add_argument("--models", type=str, default="resunet", help="Comma-separated: unet,attunet,resunet,unetpp,deeplabv3p")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--patch_size", type=int, default=512)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--weight_decay", type=float, default=1e-4)
    ap.add_argument("--num_workers", type=int, default=2)
    ap.add_argument("--threshold", type=float, default=0.5)
    ap.add_argument("--tversky_alpha", type=float, default=0.7)
    ap.add_argument("--tversky_beta", type=float, default=0.3)
    ap.add_argument("--tversky_gamma", type=float, default=0.75)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--base_channels", type=int, default=32)
    ap.add_argument("--no_amp", action="store_true")
    return ap.parse_args()


def set_seed(seed: int):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True


def main():
    args = parse_args()
    set_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f">> Device: {device}")

    if not os.path.isfile(args.manifest):
        raise FileNotFoundError(f"Missing manifest: {args.manifest}. Run preprocess_cbis_mass.py first.")

    df = pd.read_csv(args.manifest)

    # Participant-aware group split
    from sklearn.model_selection import GroupShuffleSplit
    gss = GroupShuffleSplit(n_splits=1, test_size=0.15, random_state=args.seed)
    train_val_idx, test_idx = next(gss.split(df, groups=df["participant_id"]))
    train_val = df.iloc[train_val_idx].reset_index(drop=True)
    test_df = df.iloc[test_idx].reset_index(drop=True)

    val_ratio = 0.15 / (1 - 0.15)
    gss2 = GroupShuffleSplit(n_splits=1, test_size=val_ratio, random_state=args.seed)
    train_idx, val_idx = next(gss2.split(train_val, groups=train_val["participant_id"]))
    train_df = train_val.iloc[train_idx].reset_index(drop=True)
    val_df = train_val.iloc[val_idx].reset_index(drop=True)

    print(f">> Split sizes: train={len(train_df)} val={len(val_df)} test={len(test_df)}")

    train_ds = MassSegDataset(train_df, base_path=args.base_path, patch_size=args.patch_size, augment=True, drop_empty_masks=True)
    val_ds = MassSegDataset(val_df, base_path=args.base_path, patch_size=args.patch_size, augment=False, drop_empty_masks=True)
    test_ds = MassSegDataset(test_df, base_path=args.base_path, patch_size=args.patch_size, augment=False, drop_empty_masks=True)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)

    models = [m.strip() for m in args.models.split(",") if m.strip()]
    os.makedirs(args.out_dir, exist_ok=True)

    all_results = {}

    for mname in models:
        model = build_model(mname, base=args.base_channels)
        run_dir = os.path.join(args.out_dir, mname)

        res = train_one(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            device=device,
            epochs=args.epochs,
            lr=args.lr,
            weight_decay=args.weight_decay,
            threshold=args.threshold,
            tversky_alpha=args.tversky_alpha,
            tversky_beta=args.tversky_beta,
            tversky_gamma=args.tversky_gamma,
            out_dir=run_dir,
            model_name=mname,
            use_amp=(not args.no_amp),
        )

        model2 = build_model(mname, base=args.base_channels)
        test_metrics = evaluate(
            model2,
            res["best_ckpt"],
            test_loader,
            device=device,
            threshold=args.threshold,
            tversky_alpha=args.tversky_alpha,
            tversky_beta=args.tversky_beta,
            tversky_gamma=args.tversky_gamma,
        )

        final = {"model": mname, **res, **test_metrics}
        all_results[mname] = final

        out_json = os.path.join(run_dir, "metrics.json")
        with open(out_json, "w") as f:
            json.dump(final, f, indent=2)
        print(f">> Saved metrics: {out_json}")

    summary_path = os.path.join(args.out_dir, "summary.json")
    with open(summary_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f">> Saved summary: {summary_path}")


if __name__ == "__main__":
    main()
