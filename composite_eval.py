import argparse
import os
import random
from typing import Tuple

import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import (
    roc_auc_score,
    accuracy_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
)
from sklearn.model_selection import GroupShuffleSplit
from torchvision import models as tv_models
from torchvision import models

from preprocess_cbis_mass import CBISDDSM_Preprocessor


# -----------------------------
# Segmentation models (from train_segmentation_mass.py)
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


# -----------------------------
# Classifier models (from classification.py)
# -----------------------------


def build_classifier(model_name: str, dropout: float = 0.5) -> nn.Module:
    name = model_name.lower()
    if name == "resnet50":
        model = models.resnet50(weights=None)
        model.fc = nn.Sequential(nn.Dropout(p=dropout), nn.Linear(model.fc.in_features, 1))
        return model
    if name == "resnext50_32x4d":
        model = models.resnext50_32x4d(weights=None)
        model.fc = nn.Sequential(nn.Dropout(p=dropout), nn.Linear(model.fc.in_features, 1))
        return model
    if name == "densenet121":
        model = models.densenet121(weights=None)
        model.classifier = nn.Sequential(nn.Dropout(p=dropout), nn.Linear(model.classifier.in_features, 1))
        return model
    if name == "densenet169":
        model = models.densenet169(weights=None)
        model.classifier = nn.Sequential(nn.Dropout(p=dropout), nn.Linear(model.classifier.in_features, 1))
        return model
    if name == "efficientnet_b0":
        model = models.efficientnet_b0(weights=None)
        in_features = model.classifier[1].in_features
        model.classifier[1] = nn.Sequential(nn.Dropout(p=dropout), nn.Linear(in_features, 1))
        return model
    if name == "efficientnet_v2_s":
        model = models.efficientnet_v2_s(weights=None)
        in_features = model.classifier[1].in_features
        model.classifier[1] = nn.Sequential(nn.Dropout(p=dropout), nn.Linear(in_features, 1))
        return model

    raise ValueError(f"Unsupported classifier model: {model_name}")


# -----------------------------
# Segmentation preprocessing helpers
# -----------------------------


def preprocess_for_segmentation(
    proc: CBISDDSM_Preprocessor,
    row: pd.Series,
    patch_size: int,
) -> Tuple[np.ndarray, np.ndarray]:
    img_path = os.path.join(proc.base_path, row["image_path"])
    mask_path = os.path.join(proc.base_path, row["mask_path"])

    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    if img is None or mask is None:
        raise RuntimeError(f"Failed to read: {img_path} / {mask_path}")

    img = img.astype(np.float32) / 255.0
    mask = (mask > 127).astype(np.float32)
    if mask.shape != img.shape:
        mask = cv2.resize(mask, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_NEAREST)

    img = proc.segment_breast_otsu(img)
    img = proc.suppress_pectoral_muscle(img, str(row["view"]))
    mask = np.where(img > 0.0, mask, 0.0).astype(np.float32)
    img = proc.apply_clahe(img)

    img_patch, mask_patch = proc.extract_lesion_patch(img, mask, patch_size=patch_size)
    img_patch = np.clip(img_patch, 0.0, 1.0).astype(np.float32)
    mask_patch = (mask_patch > 0.5).astype(np.float32)
    return img_patch, mask_patch


def load_full_image_and_mask(
    proc: CBISDDSM_Preprocessor,
    row: pd.Series,
) -> Tuple[np.ndarray, np.ndarray]:
    img_path = os.path.join(proc.base_path, row["image_path"])
    mask_path = os.path.join(proc.base_path, row["mask_path"])

    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    if img is None or mask is None:
        raise RuntimeError(f"Failed to read: {img_path} / {mask_path}")

    img = img.astype(np.float32) / 255.0
    mask = (mask > 127).astype(np.float32)
    if mask.shape != img.shape:
        mask = cv2.resize(mask, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_NEAREST)
    return img, mask


def crop_by_pred_mask(
    image: np.ndarray,
    pred_mask: np.ndarray,
    crop_size: int,
    final_size: int,
) -> np.ndarray:
    img_h, img_w = image.shape
    mask_uint8 = (pred_mask * 255).astype(np.uint8)
    contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if contours:
        largest = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(largest)
        center_x, center_y = x + w // 2, y + h // 2
        max_dim = max(w, h)
        dynamic_size = int(max_dim * 1.5)
        crop_dim = max(crop_size, dynamic_size)
    else:
        center_x, center_y = img_w // 2, img_h // 2
        crop_dim = crop_size

    crop_dim = min(crop_dim, img_w, img_h)
    half = crop_dim // 2

    x1 = max(0, center_x - half)
    y1 = max(0, center_y - half)
    x2 = min(img_w, center_x + half)
    y2 = min(img_h, center_y + half)

    roi = image[y1:y2, x1:x2]
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    roi = clahe.apply((roi * 255).astype(np.uint8))
    roi_resized = cv2.resize(roi, (final_size, final_size), interpolation=cv2.INTER_AREA)
    return roi_resized


def crop_by_center(
    image: np.ndarray,
    center_x: int,
    center_y: int,
    crop_size: int,
    final_size: int,
) -> np.ndarray:
    img_h, img_w = image.shape
    crop_dim = min(crop_size, img_w, img_h)
    half = crop_dim // 2

    x1 = max(0, center_x - half)
    y1 = max(0, center_y - half)
    x2 = min(img_w, center_x + half)
    y2 = min(img_h, center_y + half)

    roi = image[y1:y2, x1:x2]
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    roi = clahe.apply((roi * 255).astype(np.uint8))
    roi_resized = cv2.resize(roi, (final_size, final_size), interpolation=cv2.INTER_AREA)
    return roi_resized


def postprocess_mask(
    mask: np.ndarray,
    min_area_ratio: float = 0.001,
    morph_kernel: int = 3,
) -> np.ndarray:
    mask_uint8 = (mask > 0.5).astype(np.uint8) * 255
    if morph_kernel > 0:
        kernel = np.ones((morph_kernel, morph_kernel), np.uint8)
        mask_uint8 = cv2.morphologyEx(mask_uint8, cv2.MORPH_OPEN, kernel)
        mask_uint8 = cv2.morphologyEx(mask_uint8, cv2.MORPH_CLOSE, kernel)

    contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return np.zeros_like(mask, dtype=np.float32)

    largest = max(contours, key=cv2.contourArea)
    area = cv2.contourArea(largest)
    if area / float(mask.size) < min_area_ratio:
        return np.zeros_like(mask, dtype=np.float32)

    cleaned = np.zeros_like(mask_uint8)
    cv2.drawContours(cleaned, [largest], -1, 255, thickness=cv2.FILLED)
    return (cleaned > 0).astype(np.float32)


def prob_centroid(prob: np.ndarray) -> Tuple[int, int]:
    h, w = prob.shape
    total = float(prob.sum())
    if total <= 0:
        return w // 2, h // 2
    ys, xs = np.indices(prob.shape)
    cx = int((prob * xs).sum() / total)
    cy = int((prob * ys).sum() / total)
    return cx, cy


def to_classifier_tensor(gray_u8: np.ndarray) -> torch.Tensor:
    img = cv2.cvtColor(gray_u8, cv2.COLOR_GRAY2RGB).astype(np.float32) / 255.0
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    img = (img - mean) / std
    img = np.transpose(img, (2, 0, 1))
    return torch.from_numpy(img).float()


def load_seg_model(model_name: str, ckpt_path: str, device: torch.device) -> nn.Module:
    name = model_name.lower()
    if name == "unet":
        model = UNet()
    elif name in ("attunet", "attentionunet"):
        model = AttentionUNet()
    elif name == "resunet":
        model = ResUNet()
    elif name in ("unetpp", "unetplusplus", "unet++"):
        model = UNetPlusPlus()
    elif name in ("deeplabv3p", "deeplabv3plus", "deeplabv3+"):
        model = DeepLabV3Plus()
    else:
        raise ValueError(f"Unsupported segmentation model: {model_name}")

    state = torch.load(ckpt_path, map_location=device)
    if isinstance(state, dict):
        if "model" in state:
            model.load_state_dict(state["model"])
        elif "state_dict" in state:
            model.load_state_dict(state["state_dict"])
        else:
            model.load_state_dict(state)
    else:
        model.load_state_dict(state)
    model.to(device)
    model.eval()
    return model


def predict_segmentation_ensemble(
    seg_models: list[nn.Module],
    x: torch.Tensor,
    weights: np.ndarray,
) -> np.ndarray:
    probs_list = []
    with torch.no_grad():
        for model in seg_models:
            logits = model(x)
            probs = torch.sigmoid(logits).cpu().numpy()[0, 0]
            probs_list.append(probs)
    probs_stack = np.stack(probs_list, axis=0)
    probs_avg = np.tensordot(weights, probs_stack, axes=(0, 0))
    return probs_avg.astype(np.float32)


def load_classifier(model_name: str, ckpt_path: str, device: torch.device, dropout: float) -> nn.Module:
    model = build_classifier(model_name, dropout=dropout)
    state = torch.load(ckpt_path, map_location=device)
    if isinstance(state, dict) and "state_dict" in state:
        model.load_state_dict(state["state_dict"])
    else:
        model.load_state_dict(state)
    model.to(device)
    model.eval()
    return model


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate composite system: segmentation -> crop -> classification"
    )
    parser.add_argument("--base_path", type=str, default=None)
    parser.add_argument("--manifest", type=str, default="preprocessed_output/manifest_mass.csv")
    parser.add_argument("--seg_model", type=str, default="resunet")
    parser.add_argument("--seg_ckpt", type=str, default=None)
    parser.add_argument("--seg_models", type=str, nargs="+", default=None)
    parser.add_argument("--seg_ckpts", type=str, nargs="+", default=None)
    parser.add_argument("--seg_weights", type=float, nargs="*", default=None)
    parser.add_argument("--seg_threshold", type=float, default=0.5)
    parser.add_argument(
        "--sweep_seg_thresholds",
        type=float,
        nargs="*",
        default=None,
        help="Optional list of segmentation thresholds to sweep in a single run.",
    )
    parser.add_argument(
        "--crop_mode",
        type=str,
        default="centroid",
        choices=["centroid", "bbox"],
        help="How to crop ROI from predicted mask: centroid (probability centroid + fixed size) or bbox (mask bbox).",
    )
    parser.add_argument("--min_mask_area_ratio", type=float, default=0.001)
    parser.add_argument("--morph_kernel", type=int, default=3)
    parser.add_argument("--patch_size", type=int, default=512)
    parser.add_argument("--use_pred_mask", action="store_true", help="Use segmentation predictions for ROI crop.")
    parser.add_argument("--clf_models", type=str, nargs="+", required=True)
    parser.add_argument("--clf_ckpts", type=str, nargs="+", required=True)
    parser.add_argument("--ensemble_weights", type=float, nargs="*", default=None)
    parser.add_argument("--clf_dropout", type=float, default=0.5)
    parser.add_argument("--clf_threshold", type=float, default=0.52)
    parser.add_argument(
        "--sweep_clf_thresholds",
        type=float,
        nargs="*",
        default=None,
        help="Optional list of classifier thresholds to sweep in a single run.",
    )
    parser.add_argument("--crop_size", type=int, default=600)
    parser.add_argument("--final_size", type=int, default=512)
    parser.add_argument("--split", type=str, default="test")
    parser.add_argument("--test_size", type=float, default=0.2)
    parser.add_argument("--val_size", type=float, default=0.15)
    parser.add_argument("--max_samples", type=int, default=-1)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    proc = CBISDDSM_Preprocessor(base_path=args.base_path)
    if not os.path.isfile(args.manifest):
        raise FileNotFoundError(f"Missing manifest: {args.manifest}")

    df = pd.read_csv(args.manifest)

    if "split" in df.columns and args.split in {"train", "val", "test"}:
        df_eval = df[df["split"] == args.split].reset_index(drop=True)
    else:
        if "participant_id" not in df.columns:
            raise ValueError("Manifest missing participant_id; cannot create group split.")
        gss = GroupShuffleSplit(n_splits=1, test_size=args.test_size, random_state=42)
        train_val_idx, test_idx = next(gss.split(df, groups=df["participant_id"]))
        train_val = df.iloc[train_val_idx].reset_index(drop=True)
        test_df = df.iloc[test_idx].reset_index(drop=True)
        if args.split == "test":
            df_eval = test_df
        elif args.split == "train":
            df_eval = train_val
        else:
            val_ratio = args.val_size / (1 - args.test_size)
            gss2 = GroupShuffleSplit(n_splits=1, test_size=val_ratio, random_state=42)
            tr_idx, val_idx = next(gss2.split(train_val, groups=train_val["participant_id"]))
            df_eval = train_val.iloc[val_idx].reset_index(drop=True)

    if args.max_samples > 0:
        df_eval = df_eval.iloc[: args.max_samples].reset_index(drop=True)

    if len(args.clf_models) != len(args.clf_ckpts):
        raise ValueError("--clf_models and --clf_ckpts must have the same length.")

    if args.use_pred_mask:
        if args.seg_models is not None or args.seg_ckpts is not None:
            if not args.seg_models or not args.seg_ckpts:
                raise ValueError("--seg_models and --seg_ckpts must both be provided for segmentation ensembling.")
            if len(args.seg_models) != len(args.seg_ckpts):
                raise ValueError("--seg_models and --seg_ckpts must have the same length.")
            seg_models = [
                load_seg_model(m, p, device)
                for m, p in zip(args.seg_models, args.seg_ckpts)
            ]
        else:
            if not args.seg_ckpt:
                raise ValueError("--seg_ckpt is required when --use_pred_mask is set.")
            seg_models = [load_seg_model(args.seg_model, args.seg_ckpt, device)]

        if args.seg_weights is not None and len(args.seg_weights) != len(seg_models):
            raise ValueError("--seg_weights must be the same length as the segmentation ensemble.")
        if args.seg_weights is None:
            seg_weights = np.ones(len(seg_models), dtype=np.float32) / len(seg_models)
        else:
            seg_weights = np.array(args.seg_weights, dtype=np.float32)
            if seg_weights.sum() == 0:
                seg_weights = np.ones(len(seg_models), dtype=np.float32) / len(seg_models)
            else:
                seg_weights = seg_weights / seg_weights.sum()
    else:
        seg_models = []
        seg_weights = np.ones(1, dtype=np.float32)

    if args.sweep_seg_thresholds:
        seg_thresholds = list(args.sweep_seg_thresholds)
    else:
        seg_thresholds = [args.seg_threshold]
    clf_models = [
        load_classifier(m, p, device, dropout=args.clf_dropout)
        for m, p in zip(args.clf_models, args.clf_ckpts)
    ]

    if args.ensemble_weights is not None and len(args.ensemble_weights) != len(clf_models):
        raise ValueError("--ensemble_weights must be the same length as --clf_models.")
    if args.ensemble_weights is None:
        weights = np.ones(len(clf_models), dtype=np.float32) / len(clf_models)
    else:
        weights = np.array(args.ensemble_weights, dtype=np.float32)
        if weights.sum() == 0:
            weights = np.ones(len(clf_models), dtype=np.float32) / len(clf_models)
        else:
            weights = weights / weights.sum()

    y_true = []
    y_prob_by_thr = {thr: [] for thr in seg_thresholds}
    empty_preds_by_thr = {thr: 0 for thr in seg_thresholds}

    for _, row in df_eval.iterrows():
        try:
            if args.use_pred_mask:
                img_patch, _ = preprocess_for_segmentation(proc, row, patch_size=args.patch_size)
                x = torch.from_numpy(img_patch).unsqueeze(0).unsqueeze(0).to(device)
                probs = predict_segmentation_ensemble(seg_models, x, seg_weights)
                roi_u8_by_thr = {}
                for thr in seg_thresholds:
                    pred_mask = (probs > thr).astype(np.float32)
                    pred_mask = postprocess_mask(
                        pred_mask,
                        min_area_ratio=args.min_mask_area_ratio,
                        morph_kernel=args.morph_kernel,
                    )
                    if pred_mask.sum() == 0:
                        empty_preds_by_thr[thr] += 1
                    if args.crop_mode == "centroid":
                        cx, cy = prob_centroid(probs)
                        roi_u8_by_thr[thr] = crop_by_center(img_patch, cx, cy, args.crop_size, args.final_size)
                    else:
                        roi_u8_by_thr[thr] = crop_by_pred_mask(img_patch, pred_mask, args.crop_size, args.final_size)
            else:
                img_full, mask_full = load_full_image_and_mask(proc, row)
                roi_u8 = crop_by_pred_mask(img_full, mask_full, args.crop_size, args.final_size)
                roi_u8_by_thr = {thr: roi_u8 for thr in seg_thresholds}
        except Exception:
            continue

        y_true.append(int(row["label"]))
        for thr in seg_thresholds:
            x_clf = to_classifier_tensor(roi_u8_by_thr[thr]).unsqueeze(0).to(device)
            probs_clf = []
            with torch.no_grad():
                for clf in clf_models:
                    logit = clf(x_clf)
                    prob = torch.sigmoid(logit).cpu().numpy().reshape(-1)[0]
                    probs_clf.append(prob)
            prob = float(np.dot(weights, np.array(probs_clf, dtype=np.float32)))
            y_prob_by_thr[thr].append(prob)

    if not y_true:
        print("No samples were evaluated.")
        return

    y_true = np.array(y_true)
    if args.sweep_clf_thresholds:
        clf_thresholds = list(args.sweep_clf_thresholds)
    else:
        clf_thresholds = [args.clf_threshold]

    header = "seg -> clf" if args.use_pred_mask else "gt mask -> clf"
    print(f"\nComposite system results ({header}):")
    print(f"Samples: {len(y_true)}")

    for thr in seg_thresholds:
        y_prob = np.array(y_prob_by_thr[thr])
        thr_label = f"{thr:.2f}"
        if args.use_pred_mask:
            print(f"\n[seg_threshold={thr_label}] empty_pred_masks={empty_preds_by_thr[thr]}")
        else:
            print(f"\n[seg_threshold={thr_label}]")
        print(f"AUC:      {roc_auc_score(y_true, y_prob):.4f}")
        for clf_thr in clf_thresholds:
            y_pred = (y_prob > clf_thr).astype(int)
            print(f"\n[clf_threshold={clf_thr:.2f}]")
            print(f"Accuracy: {accuracy_score(y_true, y_pred):.4f}")
            print(f"Recall:   {recall_score(y_true, y_pred):.4f}")
            print(f"F1:       {f1_score(y_true, y_pred):.4f}")
            print("Confusion matrix:")
            print(confusion_matrix(y_true, y_pred))
            print("\nClassification report:")
            print(classification_report(y_true, y_pred, target_names=["Benign", "Malignant"]))


if __name__ == "__main__":
    main()
