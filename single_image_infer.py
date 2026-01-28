import argparse
import os
from typing import Tuple, List

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models as tv_models


# -----------------------------
# Segmentation models (same as composite_eval)
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


# -----------------------------
# Classifier models
# -----------------------------

def build_classifier(model_name: str, dropout: float = 0.5) -> nn.Module:
    name = model_name.lower()
    if name == "resnet50":
        model = tv_models.resnet50(weights=None)
        model.fc = nn.Sequential(nn.Dropout(p=dropout), nn.Linear(model.fc.in_features, 1))
        return model
    if name == "resnext50_32x4d":
        model = tv_models.resnext50_32x4d(weights=None)
        model.fc = nn.Sequential(nn.Dropout(p=dropout), nn.Linear(model.fc.in_features, 1))
        return model
    if name == "densenet121":
        model = tv_models.densenet121(weights=None)
        model.classifier = nn.Sequential(nn.Dropout(p=dropout), nn.Linear(model.classifier.in_features, 1))
        return model
    if name == "densenet169":
        model = tv_models.densenet169(weights=None)
        model.classifier = nn.Sequential(nn.Dropout(p=dropout), nn.Linear(model.classifier.in_features, 1))
        return model
    if name == "efficientnet_b0":
        model = tv_models.efficientnet_b0(weights=None)
        in_features = model.classifier[1].in_features
        model.classifier[1] = nn.Sequential(nn.Dropout(p=dropout), nn.Linear(in_features, 1))
        return model
    if name == "efficientnet_v2_s":
        model = tv_models.efficientnet_v2_s(weights=None)
        in_features = model.classifier[1].in_features
        model.classifier[1] = nn.Sequential(nn.Dropout(p=dropout), nn.Linear(in_features, 1))
        return model

    raise ValueError(f"Unsupported classifier model: {model_name}")


# -----------------------------
# Helpers
# -----------------------------

def otsu_mask(image_float01: np.ndarray) -> np.ndarray:
    img_uint8 = (image_float01 * 255).astype(np.uint8)
    _, binary = cv2.threshold(img_uint8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    kernel = np.ones((5, 5), np.uint8)
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return np.zeros_like(img_uint8)
    largest = max(contours, key=cv2.contourArea)
    mask = np.zeros_like(img_uint8)
    cv2.drawContours(mask, [largest], -1, 255, thickness=cv2.FILLED)
    return (mask > 0).astype(np.uint8)


def suppress_pectoral_muscle(image_float01: np.ndarray, view: str) -> np.ndarray:
    if view != "MLO":
        return image_float01
    img_uint8 = (image_float01 * 255).astype(np.uint8)
    h, w = img_uint8.shape
    is_left = np.sum(img_uint8[:, : w // 2]) > np.sum(img_uint8[:, w // 2 :])

    edges = cv2.Canny(img_uint8, 30, 100)
    roi_mask = np.zeros_like(edges)
    if is_left:
        roi_mask[0 : h // 2, 0 : w // 2] = 255
    else:
        roi_mask[0 : h // 2, w // 2 : w] = 255
    edges_roi = cv2.bitwise_and(edges, edges, mask=roi_mask)

    lines = cv2.HoughLinesP(edges_roi, 1, np.pi / 180, threshold=25, minLineLength=30, maxLineGap=30)
    if lines is None:
        return image_float01

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
        return image_float01

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


def apply_clahe(image_float01: np.ndarray) -> np.ndarray:
    img_uint8 = (image_float01 * 255).astype(np.uint8)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    out = clahe.apply(img_uint8)
    return out.astype(np.float32) / 255.0


def extract_patch_from_mask(image: np.ndarray, mask: np.ndarray, patch_size: int) -> np.ndarray:
    mask_uint8 = (mask * 255).astype(np.uint8)
    moments = cv2.moments(mask_uint8)
    if moments["m00"] == 0:
        cy, cx = image.shape[0] // 2, image.shape[1] // 2
    else:
        cx = int(moments["m10"] / moments["m00"])
        cy = int(moments["m01"] / moments["m00"])

    half = patch_size // 2
    pad = ((half, half), (half, half))
    img_p = np.pad(image, pad, mode="constant", constant_values=0)

    pcx, pcy = cx + half, cy + half
    sx, ex = pcx - half, pcx - half + patch_size
    sy, ey = pcy - half, pcy - half + patch_size

    patch = img_p[sy:ey, sx:ex]
    if patch.shape != (patch_size, patch_size):
        patch = cv2.resize(patch, (patch_size, patch_size), interpolation=cv2.INTER_LINEAR)
    return patch.astype(np.float32)


def postprocess_mask(mask: np.ndarray, min_area_ratio: float = 0.001, morph_kernel: int = 3) -> np.ndarray:
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


def crop_by_center(image: np.ndarray, center_x: int, center_y: int, crop_size: int, final_size: int) -> np.ndarray:
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


def predict_segmentation_ensemble(models: List[nn.Module], x: torch.Tensor, weights: np.ndarray) -> np.ndarray:
    probs_list = []
    with torch.no_grad():
        for model in models:
            logits = model(x)
            probs = torch.sigmoid(logits).cpu().numpy()[0, 0]
            probs_list.append(probs)
    probs_stack = np.stack(probs_list, axis=0)
    probs_avg = np.tensordot(weights, probs_stack, axes=(0, 0))
    return probs_avg.astype(np.float32)


def choose_image_path(path_arg: str) -> str:
    if path_arg:
        return path_arg
    try:
        import tkinter as tk
        from tkinter import filedialog
        root = tk.Tk()
        root.withdraw()
        file_path = filedialog.askopenfilename(title="Select mammogram image")
        return file_path
    except Exception as e:
        raise RuntimeError("No --image provided and file dialog failed. Provide --image.") from e


def main():
    parser = argparse.ArgumentParser(description="Single-image composite inference")
    parser.add_argument("--image", type=str, default=None, help="Path to a mammogram image (JPEG/PNG)")
    parser.add_argument("--view", type=str, default="CC", help="CC or MLO (affects pectoral suppression)")
    parser.add_argument("--seg_models", type=str, nargs="+", required=True)
    parser.add_argument("--seg_ckpts", type=str, nargs="+", required=True)
    parser.add_argument("--seg_weights", type=float, nargs="*", default=None)
    parser.add_argument("--seg_threshold", type=float, default=0.5)
    parser.add_argument("--patch_size", type=int, default=512)
    parser.add_argument("--min_mask_area_ratio", type=float, default=0.001)
    parser.add_argument("--morph_kernel", type=int, default=3)
    parser.add_argument("--crop_size", type=int, default=600)
    parser.add_argument("--final_size", type=int, default=512)
    parser.add_argument("--clf_models", type=str, nargs="+", required=True)
    parser.add_argument("--clf_ckpts", type=str, nargs="+", required=True)
    parser.add_argument("--clf_weights", type=float, nargs="*", default=None)
    parser.add_argument("--clf_threshold", type=float, default=0.70)
    parser.add_argument("--overlay_path", type=str, default="single_infer_overlay.png")
    parser.add_argument("--mask_path", type=str, default=None)
    args = parser.parse_args()

    if len(args.seg_models) != len(args.seg_ckpts):
        raise ValueError("--seg_models and --seg_ckpts must have the same length.")
    if len(args.clf_models) != len(args.clf_ckpts):
        raise ValueError("--clf_models and --clf_ckpts must have the same length.")

    if args.seg_weights is None:
        seg_weights = np.ones(len(args.seg_models), dtype=np.float32) / len(args.seg_models)
    else:
        seg_weights = np.array(args.seg_weights, dtype=np.float32)
        seg_weights = seg_weights / seg_weights.sum() if seg_weights.sum() > 0 else np.ones(len(args.seg_models), dtype=np.float32) / len(args.seg_models)

    if args.clf_weights is None:
        clf_weights = np.ones(len(args.clf_models), dtype=np.float32) / len(args.clf_models)
    else:
        clf_weights = np.array(args.clf_weights, dtype=np.float32)
        clf_weights = clf_weights / clf_weights.sum() if clf_weights.sum() > 0 else np.ones(len(args.clf_models), dtype=np.float32) / len(args.clf_models)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    image_path = choose_image_path(args.image)
    if not image_path or not os.path.isfile(image_path):
        raise FileNotFoundError(f"Image not found: {image_path}")

    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise RuntimeError(f"Failed to read image: {image_path}")

    img_float = img.astype(np.float32) / 255.0
    breast_mask = otsu_mask(img_float)
    img_clean = img_float * breast_mask
    img_clean = suppress_pectoral_muscle(img_clean, args.view)
    img_clean = apply_clahe(img_clean)

    patch = extract_patch_from_mask(img_clean, breast_mask, args.patch_size)
    patch = np.clip(patch, 0.0, 1.0).astype(np.float32)

    seg_models = [load_seg_model(m, p, device) for m, p in zip(args.seg_models, args.seg_ckpts)]
    x = torch.from_numpy(patch).unsqueeze(0).unsqueeze(0).to(device)
    probs = predict_segmentation_ensemble(seg_models, x, seg_weights)

    pred_mask = (probs > args.seg_threshold).astype(np.float32)
    pred_mask = postprocess_mask(pred_mask, min_area_ratio=args.min_mask_area_ratio, morph_kernel=args.morph_kernel)

    cx, cy = prob_centroid(probs)
    roi_u8 = crop_by_center(patch, cx, cy, args.crop_size, args.final_size)

    clf_models = [load_classifier(m, p, device, dropout=0.5) for m, p in zip(args.clf_models, args.clf_ckpts)]
    x_clf = to_classifier_tensor(roi_u8).unsqueeze(0).to(device)

    probs_clf = []
    with torch.no_grad():
        for clf in clf_models:
            logit = clf(x_clf)
            prob = torch.sigmoid(logit).cpu().numpy().reshape(-1)[0]
            probs_clf.append(prob)

    prob_final = float(np.dot(clf_weights, np.array(probs_clf, dtype=np.float32)))
    verdict = "MALIGNANT" if prob_final > args.clf_threshold else "BENIGN"

    print(f"\nImage: {image_path}")
    print(f"Seg prob threshold: {args.seg_threshold:.2f}")
    print(f"Classifier threshold: {args.clf_threshold:.2f}")
    print(f"Malignant probability: {prob_final:.4f}")
    print(f"Verdict: {verdict}")

    # Save overlay of seg mask on patch
    patch_u8 = (patch * 255).astype(np.uint8)
    overlay = np.stack([patch_u8, patch_u8, patch_u8], axis=-1)
    overlay[pred_mask > 0.5, 2] = 255
    cv2.imwrite(args.overlay_path, overlay)
    print(f"Saved overlay: {args.overlay_path}")

    if args.mask_path:
        mask_u8 = (pred_mask * 255).astype(np.uint8)
        cv2.imwrite(args.mask_path, mask_u8)
        print(f"Saved mask: {args.mask_path}")


if __name__ == "__main__":
    main()
