"""Metrics, uncertainty, calibration, and classical baselines without sklearn."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch import Tensor, nn


@dataclass(frozen=True)
class BinaryMetrics:
    iou: float; f1: float; precision: float; recall: float; pixel_accuracy: float
    tp: int; fp: int; fn: int; tn: int
    def to_dict(self) -> dict[str, float | int]: return self.__dict__.copy()


def binary_metrics(probability: np.ndarray, labels: np.ndarray, threshold: float = .5) -> BinaryMetrics:
    valid = labels >= 0; truth = labels[valid].astype(bool); predicted = probability[valid] >= threshold
    tp, fp, fn, tn = int(np.logical_and(predicted, truth).sum()), int(np.logical_and(predicted, ~truth).sum()), int(np.logical_and(~predicted, truth).sum()), int(np.logical_and(~predicted, ~truth).sum())
    return metrics_from_counts(tp, fp, fn, tn)


def metrics_from_counts(tp: int, fp: int, fn: int, tn: int) -> BinaryMetrics:
    precision = tp / max(tp + fp, 1); recall = tp / max(tp + fn, 1); f1 = 2 * precision * recall / max(precision + recall, 1e-8); iou = tp / max(tp + fp + fn, 1); accuracy = (tp + tn) / max(tp + fp + fn + tn, 1)
    return BinaryMetrics(iou, f1, precision, recall, accuracy, tp, fp, fn, tn)


def otsu_probability(vh_normalized: np.ndarray) -> np.ndarray:
    """Water is typically dark in VH; Otsu yields a deterministic baseline mask."""
    values = np.clip(vh_normalized * 255, 0, 255).astype(np.uint8)
    histogram = np.bincount(values.ravel(), minlength=256).astype(np.float64); weight = histogram / max(histogram.sum(), 1)
    omega = np.cumsum(weight); mean = np.cumsum(weight * np.arange(256)); total = mean[-1]
    score = (total * omega - mean) ** 2 / np.maximum(omega * (1 - omega), 1e-12); threshold = int(np.argmax(score)) / 255.0
    return (vh_normalized <= threshold).astype(np.float32)


def enable_mc_dropout(model: nn.Module) -> None:
    model.eval()
    for module in model.modules():
        if isinstance(module, (nn.Dropout, nn.Dropout2d, nn.Dropout3d)): module.train()


def mc_predict(model: nn.Module, features: Tensor, passes: int = 12) -> tuple[Tensor, Tensor]:
    enable_mc_dropout(model); outputs = []
    with torch.inference_mode():
        for _ in range(passes): outputs.append(torch.sigmoid(model(features)))
    stacked = torch.stack(outputs); return stacked.mean(0), stacked.var(0, unbiased=False)


def expected_calibration_error(probabilities: np.ndarray, labels: np.ndarray, bins: int = 15) -> float:
    valid = labels >= 0; confidence = probabilities[valid].reshape(-1); truth = labels[valid].reshape(-1)
    if not confidence.size: return 0.0
    ece = 0.0
    for low, high in zip(np.linspace(0, 1, bins, endpoint=False), np.linspace(1 / bins, 1, bins)):
        selected = (confidence >= low) & ((confidence < high) if high < 1 else (confidence <= high))
        if selected.any(): ece += selected.mean() * abs(confidence[selected].mean() - truth[selected].mean())
    return float(ece)


def fit_temperature(logits: Tensor, labels: Tensor, max_steps: int = 40) -> float:
    valid = labels != -1
    if not bool(valid.any()): return 1.0
    temperature = torch.ones((), device=logits.device, requires_grad=True)
    optimizer = torch.optim.LBFGS([temperature], lr=.1, max_iter=max_steps)
    targets = labels[valid].float()
    def closure() -> Tensor:
        optimizer.zero_grad(); loss = nn.functional.binary_cross_entropy_with_logits(logits[valid] / temperature.clamp_min(.05), targets); loss.backward(); return loss
    optimizer.step(closure); return float(temperature.detach().clamp_min(.05))
