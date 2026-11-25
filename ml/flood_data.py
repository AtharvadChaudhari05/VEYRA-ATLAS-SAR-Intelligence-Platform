"""Dataset and preprocessing for the official Sen1Floods11 hand-labelled task."""

from __future__ import annotations

import csv
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
from torch import Tensor
from torch.utils.data import Dataset


@dataclass(frozen=True)
class NormalizationStats:
    low: tuple[float, float] = (-32.5, -36.5)
    high: tuple[float, float] = (-2.0, -5.0)

    def to_dict(self) -> dict[str, list[float]]:
        return {"low": list(self.low), "high": list(self.high)}


def db_from_linear(values: np.ndarray) -> np.ndarray:
    return 10.0 * np.log10(np.maximum(values, 1e-10))


def to_db(values: np.ndarray) -> np.ndarray:
    """Accept dB chips or linear intensity chips without corrupting dB inputs."""
    finite = values[np.isfinite(values)]
    if finite.size and float(np.nanpercentile(finite, 95)) > 5.0:
        return db_from_linear(values)
    return values.astype(np.float32)


def percentile_normalize(vv_vh_db: np.ndarray, stats: NormalizationStats) -> np.ndarray:
    low = np.asarray(stats.low, dtype=np.float32)[:, None, None]
    high = np.asarray(stats.high, dtype=np.float32)[:, None, None]
    return np.clip((vv_vh_db - low) / np.maximum(high - low, 1e-6), 0.0, 1.0).astype(np.float32)


def lee_filter(image: np.ndarray, kernel: int = 5) -> np.ndarray:
    """Lightweight Lee filter; use only when explicitly enabled in config."""
    if kernel < 3 or kernel % 2 == 0:
        raise ValueError("Lee filter kernel must be an odd integer >= 3")
    pad = kernel // 2
    padded = np.pad(image, ((0, 0), (pad, pad), (pad, pad)), mode="reflect")
    filtered = np.empty_like(image, dtype=np.float32)
    for channel in range(image.shape[0]):
        values = padded[channel]
        windows = np.lib.stride_tricks.sliding_window_view(values, (kernel, kernel))
        mean = windows.mean(axis=(-1, -2))
        variance = windows.var(axis=(-1, -2))
        noise = float(np.median(variance))
        weight = variance / np.maximum(variance + noise, 1e-7)
        filtered[channel] = mean + weight * (image[channel] - mean)
    return filtered


def prepare_features(raw: np.ndarray, stats: NormalizationStats, use_lee_filter: bool = False) -> np.ndarray:
    """Return VV, VH, VV-VH features in [0, 1] from a two-band Sentinel-1 chip."""
    if raw.ndim != 3:
        raise ValueError(f"Expected [bands,height,width] SAR array, got {raw.shape}")
    if raw.shape[0] < 2:
        raise ValueError("Sen1Floods11 S1Hand image must contain VV and VH bands")
    db = to_db(raw[:2].astype(np.float32))
    # Sentinel-1 chips may carry NaN/Inf nodata values. Labels mark these
    # pixels as -1 for loss/metrics; fill model inputs with the low clipping
    # bound so no non-finite values can contaminate convolutions.
    low_values = np.asarray(stats.low, dtype=np.float32)[:, None, None]
    db = np.where(np.isfinite(db), db, low_values)
    if use_lee_filter:
        db = lee_filter(db)
    polarizations = percentile_normalize(db, stats)
    ratio = db[0:1] - db[1:2]
    ratio = np.clip((ratio + 15.0) / 30.0, 0.0, 1.0)
    return np.concatenate((polarizations, ratio), axis=0).astype(np.float32)


def _read_raster(path: Path) -> np.ndarray:
    try:
        import rasterio
    except ImportError as exc:
        raise RuntimeError("rasterio is required for Sen1Floods11 GeoTIFF chips") from exc
    with rasterio.open(path) as dataset:
        return dataset.read().astype(np.float32)


def _split_entries(csv_path: Path) -> list[str]:
    with csv_path.open(newline="", encoding="utf-8-sig") as handle:
        rows = list(csv.reader(handle))
    values = [cell.strip() for row in rows for cell in row if cell.strip()]
    return [value for value in values if value.lower() not in {"filename", "image", "chips", "chip"}]


def _find_split(root: Path, split: str) -> Path:
    aliases = {"train": ("train",), "val": ("val", "validation"), "test": ("test",), "bolivia": ("bolivia",)}[split]
    candidates = list((root / "splits" / "flood_handlabeled").glob("*.csv"))
    for candidate in candidates:
        name = candidate.stem.lower()
        if any(alias in name for alias in aliases):
            return candidate
    raise FileNotFoundError(f"Could not find official {split} CSV under {root}/splits/flood_handlabeled")


def _chip_stem(name: str) -> str:
    return Path(name).stem.replace("_S1Hand", "").replace("_LabelHand", "")


class Sen1Floods11HandDataset(Dataset[tuple[Tensor, Tensor, str]]):
    """Official S1Hand / LabelHand dataset; label -1 is kept as ignore index."""

    def __init__(self, root: str | Path, split: str, *, stats: NormalizationStats = NormalizationStats(), use_lee_filter: bool = False, augment: bool = False, patch_size: int | None = None) -> None:
        self.root, self.split, self.stats, self.use_lee_filter, self.augment, self.patch_size = Path(root), split, stats, use_lee_filter, augment, patch_size
        csv_path = _find_split(self.root, split)
        s1_root = self.root / "data" / "flood_events" / "HandLabeled" / "S1Hand"
        label_root = self.root / "data" / "flood_events" / "HandLabeled" / "LabelHand"
        all_s1 = { _chip_stem(path.name): path for path in s1_root.glob("*.tif") }
        all_labels = { _chip_stem(path.name): path for path in label_root.glob("*.tif") }
        # Official CSV rows contain both S1Hand and LabelHand filenames; retain
        # each chip exactly once while preserving the published ordering.
        names = list(dict.fromkeys(_chip_stem(name) for name in _split_entries(csv_path)))
        self.items = [(all_s1[name], all_labels[name], name) for name in names if name in all_s1 and name in all_labels]
        if not self.items:
            raise ValueError(f"No matched S1Hand/LabelHand chips for split {split}")

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, index: int) -> tuple[Tensor, Tensor, str]:
        sar_path, label_path, chip_id = self.items[index]
        features = prepare_features(_read_raster(sar_path), self.stats, self.use_lee_filter)
        label = _read_raster(label_path)[0]
        label = np.where(label < 0, -1, np.where(label > 0, 1, 0)).astype(np.int64)
        if self.patch_size:
            features, label = _crop(features, label, self.patch_size, random_crop=self.augment)
        if self.augment:
            features, label = _augment(features, label)
        return torch.from_numpy(features.copy()), torch.from_numpy(label.copy()), chip_id


def _augment(features: np.ndarray, label: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    rotations = int(torch.randint(0, 4, ()).item())
    if rotations:
        features, label = np.rot90(features, rotations, axes=(1, 2)), np.rot90(label, rotations)
    if bool(torch.randint(0, 2, ()).item()):
        features, label = features[:, :, ::-1], label[:, ::-1]
    if bool(torch.randint(0, 2, ()).item()):
        features, label = features[:, ::-1, :], label[::-1, :]
    # SAR-safe mild speckle-like perturbation, not optical color jitter.
    features = np.clip(features + np.random.normal(0, 0.012, features.shape).astype(np.float32), 0, 1)
    return features, label


def _crop(features: np.ndarray, label: np.ndarray, size: int, *, random_crop: bool) -> tuple[np.ndarray, np.ndarray]:
    height, width = label.shape
    if size > height or size > width:
        raise ValueError(f"patch_size {size} exceeds chip shape {width}x{height}")
    if random_crop:
        row = int(torch.randint(0, height - size + 1, ()).item()); col = int(torch.randint(0, width - size + 1, ()).item())
    else:
        row, col = (height - size) // 2, (width - size) // 2
    return features[:, row : row + size, col : col + size], label[row : row + size, col : col + size]
