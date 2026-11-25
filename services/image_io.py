from __future__ import annotations

import base64
import io
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image


try:
    import rasterio  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    rasterio = None


@dataclass(frozen=True)
class SceneMetadata:
    width: int
    height: int
    crs: str | None = None
    transform: list[float] | None = None
    bounds: list[float] | None = None
    source_driver: str = "PIL"
    notes: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "width": self.width,
            "height": self.height,
            "crs": self.crs,
            "transform": self.transform,
            "bounds": self.bounds,
            "source_driver": self.source_driver,
            "notes": list(self.notes),
        }


@dataclass(frozen=True)
class RasterScene:
    sar: np.ndarray
    metadata: SceneMetadata
    # Un-normalised VV/VH samples, retained only for a two-band GeoTIFF.
    # The trained flood model must never receive the display-normalised image.
    sar_bands: np.ndarray | None = None


def load_raster_from_bytes(
    data: bytes,
    filename: str,
    *,
    max_side: int = 512,
    crs: str | None = None,
    transform: list[float] | None = None,
    bounds: list[float] | None = None,
) -> RasterScene:
    suffix = Path(filename).suffix.lower()
    if rasterio is not None and suffix in {".tif", ".tiff"}:
        loaded = _load_with_rasterio(data, max_side=max_side)
        metadata = loaded.metadata
        return RasterScene(
            sar=loaded.sar,
            sar_bands=loaded.sar_bands,
            metadata=SceneMetadata(
                width=metadata.width,
                height=metadata.height,
                crs=crs or metadata.crs,
                transform=transform or metadata.transform,
                bounds=bounds or metadata.bounds,
                source_driver=metadata.source_driver,
                notes=metadata.notes,
            ),
        )

    image = Image.open(io.BytesIO(data))
    image.load()
    if image.mode not in {"L", "I;16", "I", "F", "RGB", "RGBA"}:
        image = image.convert("RGB")
    source_width, source_height = image.size
    array = np.asarray(image)
    if array.ndim == 3:
        array = array[..., :3].astype(np.float32).mean(axis=2)
    else:
        array = array.astype(np.float32)
    array = _resize_array(array, max_side=max_side)
    normalized = normalize(array)
    adjusted_transform = _scale_transform(transform, source_width, source_height, normalized.shape[1], normalized.shape[0])
    metadata = SceneMetadata(
        width=int(normalized.shape[1]),
        height=int(normalized.shape[0]),
        crs=crs,
        transform=adjusted_transform,
        bounds=bounds,
        source_driver="PIL",
        notes=("No embedded geospatial metadata was read; pass CRS/transform for exact area.",) if not transform else (),
    )
    return RasterScene(sar=normalized, metadata=metadata, sar_bands=None)


def normalize(array: np.ndarray) -> np.ndarray:
    values = array.astype(np.float32)
    finite = np.isfinite(values)
    if not finite.any():
        return np.zeros_like(values, dtype=np.float32)
    valid = values[finite]
    low, high = np.percentile(valid, [1.0, 99.0])
    if high <= low:
        high = float(valid.max())
        low = float(valid.min())
    if high <= low:
        return np.zeros_like(values, dtype=np.float32)
    values = np.clip((values - low) / (high - low), 0.0, 1.0)
    values[~finite] = 0.0
    return values.astype(np.float32)


def to_uint8(array: np.ndarray) -> np.ndarray:
    return np.clip(array * 255.0, 0, 255).astype(np.uint8)


def save_png(array: np.ndarray, path: Path, *, heatmap: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    image = Image.fromarray(to_heatmap(array) if heatmap else _png_ready(array))
    image.save(path, format="PNG")


def png_bytes(array: np.ndarray, *, heatmap: bool = False) -> bytes:
    buffer = io.BytesIO()
    Image.fromarray(to_heatmap(array) if heatmap else _png_ready(array)).save(buffer, format="PNG")
    return buffer.getvalue()


def data_uri_png(array: np.ndarray, *, heatmap: bool = False) -> str:
    encoded = base64.b64encode(png_bytes(array, heatmap=heatmap)).decode("ascii")
    return f"data:image/png;base64,{encoded}"


def to_heatmap(array: np.ndarray) -> np.ndarray:
    values = np.clip(array.astype(np.float32), 0.0, 1.0)
    red = np.clip(1.8 * values - 0.35, 0.0, 1.0)
    green = np.clip(1.6 - np.abs(values - 0.55) * 2.8, 0.0, 1.0)
    blue = np.clip(1.15 - 1.7 * values, 0.0, 1.0)
    return to_uint8(np.stack([red, green, blue], axis=-1))


def _png_ready(array: np.ndarray) -> np.ndarray:
    values = np.clip(array.astype(np.float32), 0.0, 1.0)
    if values.ndim == 2:
        return to_uint8(values)
    if values.ndim == 3 and values.shape[2] == 3:
        return to_uint8(values)
    raise ValueError("PNG arrays must be HxW or HxWx3")


def _resize_array(array: np.ndarray, *, max_side: int) -> np.ndarray:
    height, width = array.shape[:2]
    largest = max(height, width)
    if largest <= max_side:
        return array.astype(np.float32)
    scale = max_side / float(largest)
    new_size = (max(1, int(round(width * scale))), max(1, int(round(height * scale))))
    image = Image.fromarray(normalize(array) * 255.0).convert("L")
    resized = image.resize(new_size, Image.Resampling.BILINEAR)
    return np.asarray(resized).astype(np.float32) / 255.0


def _scale_transform(transform: list[float] | None, source_width: int, source_height: int, target_width: int, target_height: int) -> list[float] | None:
    if transform is None:
        return None
    if source_width == target_width and source_height == target_height:
        return transform
    scale_x = source_width / float(target_width)
    scale_y = source_height / float(target_height)
    a, b, c, d, e, f = transform
    return [a * scale_x, b, c, d, e * scale_y, f]


def _load_with_rasterio(data: bytes, *, max_side: int) -> RasterScene:
    with rasterio.MemoryFile(data) as memory_file:  # type: ignore[union-attr]
        with memory_file.open() as dataset:
            source_height = dataset.height
            source_width = dataset.width
            transform = dataset.transform
            crs = str(dataset.crs) if dataset.crs else None
            bounds = [float(dataset.bounds.left), float(dataset.bounds.bottom), float(dataset.bounds.right), float(dataset.bounds.top)]
            largest = max(source_width, source_height)
            scale = min(1.0, max_side / float(largest))
            target_width = max(1, int(round(source_width * scale)))
            target_height = max(1, int(round(source_height * scale)))
            indexes = list(range(1, min(dataset.count, 2) + 1))
            raw_bands = dataset.read(
                indexes=indexes,
                out_shape=(len(indexes), target_height, target_width),
                resampling=rasterio.enums.Resampling.bilinear,
            ).astype(np.float32)
            resized = raw_bands[0]
            scale_x = source_width / target_width
            scale_y = source_height / target_height
            adjusted_transform = [
                float(transform.a * scale_x),
                float(transform.b),
                float(transform.c),
                float(transform.d),
                float(transform.e * scale_y),
                float(transform.f),
            ]
            metadata = SceneMetadata(
                width=int(resized.shape[1]),
                height=int(resized.shape[0]),
                crs=crs,
                transform=adjusted_transform,
                bounds=bounds,
                source_driver="rasterio",
            )
            return RasterScene(
                sar=normalize(resized),
                metadata=metadata,
                sar_bands=raw_bands if raw_bands.shape[0] >= 2 else None,
            )
