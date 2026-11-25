"""Flood segmentation model and masked losses for Sen1Floods11."""

from __future__ import annotations

import torch
from torch import Tensor, nn


class ConvBlock(nn.Module):
    def __init__(self, source: int, target: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.net = nn.Sequential(nn.Conv2d(source, target, 3, padding=1, bias=False), nn.BatchNorm2d(target), nn.ReLU(inplace=True), nn.Conv2d(target, target, 3, padding=1, bias=False), nn.BatchNorm2d(target), nn.ReLU(inplace=True), nn.Dropout2d(dropout) if dropout else nn.Identity())
    def forward(self, value: Tensor) -> Tensor: return self.net(value)


class FloodUNet(nn.Module):
    """Reliable dependency-light fallback U-Net with three SAR input features."""
    def __init__(self, channels: int = 3, base: int = 32, dropout: float = 0.15) -> None:
        super().__init__()
        self.e1, self.e2, self.e3 = ConvBlock(channels, base), ConvBlock(base, base * 2), ConvBlock(base * 2, base * 4)
        self.pool = nn.MaxPool2d(2); self.mid = ConvBlock(base * 4, base * 8, dropout)
        self.u3 = nn.ConvTranspose2d(base * 8, base * 4, 2, 2); self.d3 = ConvBlock(base * 8, base * 4, dropout)
        self.u2 = nn.ConvTranspose2d(base * 4, base * 2, 2, 2); self.d2 = ConvBlock(base * 4, base * 2)
        self.u1 = nn.ConvTranspose2d(base * 2, base, 2, 2); self.d1 = ConvBlock(base * 2, base)
        self.out = nn.Conv2d(base, 1, 1)
    def forward(self, value: Tensor) -> Tensor:
        e1 = self.e1(value); e2 = self.e2(self.pool(e1)); e3 = self.e3(self.pool(e2)); mid = self.mid(self.pool(e3))
        d3 = self.d3(torch.cat((self.u3(mid), e3), 1)); d2 = self.d2(torch.cat((self.u2(d3), e2), 1)); d1 = self.d1(torch.cat((self.u1(d2), e1), 1))
        return self.out(d1).squeeze(1)


def masked_bce_dice(logits: Tensor, labels: Tensor, ignore_index: int = -1, dice_weight: float = 1.0) -> Tensor:
    valid = labels != ignore_index
    if not bool(valid.any()): return logits.sum() * 0.0
    targets = labels.clamp_min(0).float()
    bce = nn.functional.binary_cross_entropy_with_logits(logits[valid], targets[valid])
    probabilities = torch.sigmoid(logits)
    intersection = (probabilities[valid] * targets[valid]).sum()
    dice = 1.0 - (2 * intersection + 1.0) / (probabilities[valid].sum() + targets[valid].sum() + 1.0)
    return bce + dice_weight * dice
