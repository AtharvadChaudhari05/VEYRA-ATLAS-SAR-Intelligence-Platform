from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass
from typing import Any

import numpy as np

from app.services.image_io import SceneMetadata


EARTH_RADIUS_M = 6_371_008.8


@dataclass(frozen=True)
class AreaResult:
    affected_area_km2: float | None
    coverage_percent: float
    pixel_count: int
    total_pixels: int
    method: str
    warning: str | None = None


def calculate_affected_area(mask: np.ndarray, metadata: SceneMetadata) -> AreaResult:
    binary = mask.astype(bool)
    pixel_count = int(binary.sum())
    total_pixels = int(binary.size)
    coverage = 0.0 if total_pixels == 0 else pixel_count / total_pixels * 100.0
    if pixel_count == 0:
        return AreaResult(0.0, coverage, pixel_count, total_pixels, "empty-mask")

    if metadata.transform:
        if _is_geographic(metadata.crs):
            area_m2 = _area_geographic(binary, metadata.transform)
            return AreaResult(area_m2 / 1_000_000.0, coverage, pixel_count, total_pixels, "geographic-transform")
        pixel_area_m2 = _projected_pixel_area(metadata.transform)
        return AreaResult(pixel_count * pixel_area_m2 / 1_000_000.0, coverage, pixel_count, total_pixels, "projected-transform")

    warning = "Exact area requires CRS and raster transform metadata."
    return AreaResult(None, coverage, pixel_count, total_pixels, "missing-georeference", warning)


def mask_to_geojson(mask: np.ndarray, metadata: SceneMetadata, *, min_pixels: int = 8, max_features: int = 256) -> dict[str, Any]:
    binary = mask.astype(bool)
    components = _components(binary, min_pixels=min_pixels, max_features=max_features)
    features = []
    for index, component in enumerate(components, start=1):
        min_row, min_col, max_row, max_col, pixels = component
        ring = [
            _pixel_to_world(min_col, min_row, metadata),
            _pixel_to_world(max_col + 1, min_row, metadata),
            _pixel_to_world(max_col + 1, max_row + 1, metadata),
            _pixel_to_world(min_col, max_row + 1, metadata),
            _pixel_to_world(min_col, min_row, metadata),
        ]
        features.append(
            {
                "type": "Feature",
                "properties": {
                    "component": index,
                    "pixel_count": int(pixels),
                    "boundary_type": "component_bbox",
                },
                "geometry": {"type": "Polygon", "coordinates": [ring]},
            }
        )
    return {
        "type": "FeatureCollection",
        "name": "affected_area_boundaries",
        "crs": {"type": "name", "properties": {"name": metadata.crs or "pixel-coordinates"}},
        "features": features,
    }


def _projected_pixel_area(transform: list[float]) -> float:
    a, b, _c, d, e, _f = transform
    return abs(a * e - b * d)


def _is_geographic(crs: str | None) -> bool:
    if not crs:
        return False
    normalized = crs.upper().replace(" ", "")
    return normalized in {"EPSG:4326", "OGC:CRS84"} or "WGS84" in normalized


def _area_geographic(mask: np.ndarray, transform: list[float]) -> float:
    a, _b, c, _d, e, f = transform
    dlon = abs(a)
    area = 0.0
    for row in range(mask.shape[0]):
        count = int(mask[row].sum())
        if count == 0:
            continue
        lat_top = f + e * row
        lat_bottom = f + e * (row + 1)
        strip = (
            EARTH_RADIUS_M
            * EARTH_RADIUS_M
            * math.radians(dlon)
            * abs(math.sin(math.radians(lat_top)) - math.sin(math.radians(lat_bottom)))
        )
        area += count * strip
    return area


def _pixel_to_world(col: int, row: int, metadata: SceneMetadata) -> list[float]:
    if not metadata.transform:
        return [float(col), float(row)]
    a, b, c, d, e, f = metadata.transform
    x = a * col + b * row + c
    y = d * col + e * row + f
    return [float(x), float(y)]


def _components(mask: np.ndarray, *, min_pixels: int, max_features: int) -> list[tuple[int, int, int, int, int]]:
    visited = np.zeros(mask.shape, dtype=bool)
    components: list[tuple[int, int, int, int, int]] = []
    height, width = mask.shape
    for row in range(height):
        for col in range(width):
            if not mask[row, col] or visited[row, col]:
                continue
            queue: deque[tuple[int, int]] = deque([(row, col)])
            visited[row, col] = True
            min_row = max_row = row
            min_col = max_col = col
            pixels = 0
            while queue:
                current_row, current_col = queue.popleft()
                pixels += 1
                min_row = min(min_row, current_row)
                max_row = max(max_row, current_row)
                min_col = min(min_col, current_col)
                max_col = max(max_col, current_col)
                for next_row, next_col in (
                    (current_row - 1, current_col),
                    (current_row + 1, current_col),
                    (current_row, current_col - 1),
                    (current_row, current_col + 1),
                ):
                    if 0 <= next_row < height and 0 <= next_col < width and mask[next_row, next_col] and not visited[next_row, next_col]:
                        visited[next_row, next_col] = True
                        queue.append((next_row, next_col))
            if pixels >= min_pixels:
                components.append((min_row, min_col, max_row, max_col, pixels))
                components.sort(key=lambda item: item[4], reverse=True)
                components = components[:max_features]
    return components

