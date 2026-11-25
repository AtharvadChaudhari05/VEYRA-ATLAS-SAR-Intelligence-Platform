from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np


MODEL_1_VERSION = "deterministic-sar-to-optical-v0"
TRAINED_MODEL_1_VERSION = "pix2pix-sar-to-optical-pilot-v0-experimental"
MODEL_2_VERSION = "deterministic-flood-segmentation-v0"
TRAINED_MODEL_2_VERSION = "flood-unet-sen1floods11-v1"
SUPPORTED_DISASTERS = {"flood"}


@dataclass(frozen=True)
class Model1Output:
    optical_like: np.ndarray
    uncertainty: np.ndarray
    preservation: np.ndarray
    confidence: float
    model_version: str


@dataclass(frozen=True)
class DisasterOutput:
    probability: np.ndarray
    mask: np.ndarray
    threshold: float
    coverage_percent: float
    confidence: float
    disaster_type: str
    model_version: str


def model1_runtime(checkpoint_path: Path | None, *, enable_experimental: bool = False) -> dict[str, str]:
    if checkpoint_path is not None and not enable_experimental:
        return {"runtime": "deterministic_baseline", "version": MODEL_1_VERSION, "status": "experimental_checkpoint_disabled"}
    if checkpoint_path is None:
        return {"runtime": "deterministic_baseline", "version": MODEL_1_VERSION, "status": "fallback"}
    if not checkpoint_path.exists():
        return {"runtime": "deterministic_baseline", "version": MODEL_1_VERSION, "status": "checkpoint_not_found"}
    try:
        _load_generator(str(checkpoint_path))
    except Exception:
        return {"runtime": "deterministic_baseline", "version": MODEL_1_VERSION, "status": "checkpoint_load_failed"}
    return {"runtime": "pytorch_pix2pix", "version": TRAINED_MODEL_1_VERSION, "status": "experimental"}


@lru_cache(maxsize=2)
def _load_generator(checkpoint_path: str):
    """Load a custom trained checkpoint only when Model 1 is invoked."""
    import torch

    from ml.sar_optical import UNetGenerator

    generator = UNetGenerator()
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    generator.load_state_dict(checkpoint["generator"])
    generator.eval()
    return generator


def predict_sar_to_optical(
    sar: np.ndarray,
    *,
    checkpoint_path: Path | None = None,
    enable_experimental: bool = False,
) -> Model1Output:
    # A checkpoint is never silently enabled: the pilot failed the quality gate.
    if enable_experimental and checkpoint_path is not None and checkpoint_path.exists():
        try:
            return _predict_with_checkpoint(sar, checkpoint_path)
        except Exception:
            # The API remains available if an optional experimental checkpoint is corrupt.
            pass
    return _predict_baseline(sar)


def _predict_with_checkpoint(sar: np.ndarray, checkpoint_path: Path) -> Model1Output:
    import torch
    import torch.nn.functional as functional

    values = _clean_unit(sar)
    original_height, original_width = values.shape
    tensor = torch.from_numpy(values).unsqueeze(0).unsqueeze(0)
    tensor = functional.interpolate(tensor, size=(256, 256), mode="bilinear", align_corners=False)
    tensor = tensor * 2.0 - 1.0
    with torch.inference_mode():
        generated = _load_generator(str(checkpoint_path))(tensor)
    generated = (generated.squeeze(0) + 1.0) / 2.0
    generated = functional.interpolate(generated.unsqueeze(0), size=(original_height, original_width), mode="bilinear", align_corners=False)
    optical_like = np.clip(generated.squeeze(0).permute(1, 2, 0).numpy(), 0.0, 1.0).astype(np.float32)

    smooth = _box_blur(values, radius=2)
    texture = np.clip(np.abs(values - smooth) * 3.0, 0.0, 1.0)
    uncertainty = np.clip(0.45 + 0.35 * texture, 0.0, 1.0).astype(np.float32)
    preservation = np.clip(1.0 - np.abs(_gradient_magnitude(optical_like.mean(axis=2)) - _gradient_magnitude(values)) * 2.0, 0.0, 1.0).astype(np.float32)
    return Model1Output(
        optical_like=optical_like,
        uncertainty=uncertainty,
        preservation=preservation,
        confidence=float(np.clip(1.0 - uncertainty.mean(), 0.0, 1.0)),
        model_version=TRAINED_MODEL_1_VERSION,
    )


def _predict_baseline(sar: np.ndarray) -> Model1Output:
    values = _clean_unit(sar)
    smooth = _box_blur(values, radius=2)
    texture = np.clip(np.abs(values - smooth) * 3.0, 0.0, 1.0)
    edges = _gradient_magnitude(values)

    red = np.clip(0.68 * values + 0.22 * edges + 0.10 * smooth, 0.0, 1.0)
    green = np.clip(0.60 * smooth + 0.25 * (1.0 - texture) + 0.15 * values, 0.0, 1.0)
    blue = np.clip(0.50 * (1.0 - values) + 0.35 * texture + 0.15 * edges, 0.0, 1.0)
    optical_like = np.stack([red, green, blue], axis=-1).astype(np.float32)

    local_variance = _box_blur(values * values, radius=3) - np.square(_box_blur(values, radius=3))
    local_variance = np.clip(local_variance * 10.0, 0.0, 1.0)
    uncertainty = np.clip(0.55 * local_variance + 0.30 * texture + 0.15 * _outlier_score(values), 0.0, 1.0)

    generated_luminance = optical_like.mean(axis=2)
    preservation_error = np.abs(_gradient_magnitude(generated_luminance) - edges)
    preservation = np.clip(1.0 - preservation_error * 2.0, 0.0, 1.0)
    confidence = float(np.clip(1.0 - uncertainty.mean(), 0.0, 1.0))
    return Model1Output(
        optical_like=optical_like,
        uncertainty=uncertainty.astype(np.float32),
        preservation=preservation.astype(np.float32),
        confidence=confidence,
        model_version=MODEL_1_VERSION,
    )


def analyze_disaster(
    sar: np.ndarray,
    model1: Model1Output,
    *,
    disaster_type: str = "flood",
    threshold: float = 0.55,
) -> DisasterOutput:
    disaster = disaster_type.lower().strip()
    if disaster not in SUPPORTED_DISASTERS:
        supported = ", ".join(sorted(SUPPORTED_DISASTERS))
        raise ValueError(f"Unsupported disaster_type '{disaster_type}'. Configured types: {supported}")
    if not 0.0 < threshold < 1.0:
        raise ValueError("threshold must be between 0.0 and 1.0")

    values = _clean_unit(sar)
    smoothness = 1.0 - np.clip(np.abs(values - _box_blur(values, radius=3)) * 4.0, 0.0, 1.0)
    dark_backscatter = np.power(1.0 - values, 1.35)
    uncertainty_gate = 1.0 - 0.35 * model1.uncertainty
    probability = np.clip(0.72 * dark_backscatter + 0.20 * smoothness + 0.08 * model1.preservation, 0.0, 1.0)
    probability = np.clip(probability * uncertainty_gate, 0.0, 1.0).astype(np.float32)

    mask = probability >= threshold
    coverage = float(mask.mean() * 100.0)
    boundary_band = np.abs(probability - threshold) < 0.08
    ambiguity_penalty = float(boundary_band.mean() * 0.25)
    confidence = float(np.clip(model1.confidence - ambiguity_penalty, 0.0, 1.0))
    return DisasterOutput(
        probability=probability,
        mask=mask,
        threshold=float(threshold),
        coverage_percent=coverage,
        confidence=confidence,
        disaster_type=disaster,
        model_version=MODEL_2_VERSION,
    )


def analyze_learned_flood(
    probability: np.ndarray,
    *,
    threshold: float = 0.55,
) -> DisasterOutput:
    """Build the same API response as the baseline from learned probabilities."""
    if not 0.0 < threshold < 1.0:
        raise ValueError("threshold must be between 0.0 and 1.0")
    values = np.clip(np.asarray(probability, dtype=np.float32), 0.0, 1.0)
    mask = values >= threshold
    # Confidence is probability separation from the decision boundary, not an
    # unsupported claim of ground-truth accuracy for a user upload.
    separation = np.abs(values - threshold) / max(threshold, 1.0 - threshold)
    confidence = float(np.clip(separation.mean(), 0.0, 1.0))
    return DisasterOutput(
        probability=values,
        mask=mask,
        threshold=float(threshold),
        coverage_percent=float(mask.mean() * 100.0),
        confidence=confidence,
        disaster_type="flood",
        model_version=TRAINED_MODEL_2_VERSION,
    )


def _clean_unit(array: np.ndarray) -> np.ndarray:
    values = array.astype(np.float32)
    values[~np.isfinite(values)] = 0.0
    return np.clip(values, 0.0, 1.0)


def _box_blur(array: np.ndarray, *, radius: int) -> np.ndarray:
    values = _clean_unit(array)
    padded = np.pad(values, radius, mode="edge")
    total = np.zeros_like(values, dtype=np.float32)
    count = 0
    for row_offset in range(2 * radius + 1):
        for col_offset in range(2 * radius + 1):
            total += padded[row_offset : row_offset + values.shape[0], col_offset : col_offset + values.shape[1]]
            count += 1
    return total / float(count)


def _gradient_magnitude(array: np.ndarray) -> np.ndarray:
    values = _clean_unit(array)
    grad_y, grad_x = np.gradient(values)
    magnitude = np.sqrt(grad_x * grad_x + grad_y * grad_y)
    high = float(np.percentile(magnitude, 98.0))
    if high <= 0:
        return np.zeros_like(values, dtype=np.float32)
    return np.clip(magnitude / high, 0.0, 1.0).astype(np.float32)


def _outlier_score(array: np.ndarray) -> np.ndarray:
    values = _clean_unit(array)
    median = float(np.median(values))
    deviation = np.abs(values - median)
    scale = float(np.percentile(deviation, 95.0))
    if scale <= 0:
        return np.zeros_like(values, dtype=np.float32)
    return np.clip(deviation / scale, 0.0, 1.0).astype(np.float32)
