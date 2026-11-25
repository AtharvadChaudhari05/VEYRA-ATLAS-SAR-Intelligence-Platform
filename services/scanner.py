from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from PIL import Image


IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}
SAR_HINTS = {"sar", "s1", "sentinel1", "sentinel-1", "vv", "vh"}
OPTICAL_HINTS = {"optical", "s2", "sentinel2", "sentinel-2", "rgb", "msi"}


@dataclass(frozen=True)
class PairRecord:
    scene_id: str
    sar_path: str
    optical_path: str
    width: int | None
    height: int | None
    channels: int | None
    valid: bool
    errors: tuple[str, ...]
    region_key: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "scene_id": self.scene_id,
            "sar_path": self.sar_path,
            "optical_path": self.optical_path,
            "width": self.width,
            "height": self.height,
            "channels": self.channels,
            "valid": self.valid,
            "errors": list(self.errors),
            "region_key": self.region_key,
        }


def scan_dataset(root_path: str | Path) -> dict[str, Any]:
    root = Path(root_path).expanduser().resolve()
    if not root.exists():
        raise FileNotFoundError(f"Dataset root does not exist: {root}")
    files = [path for path in root.rglob("*") if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS]
    sar_files = [_file for _file in files if _looks_like(_file, SAR_HINTS)]
    optical_files = [_file for _file in files if _looks_like(_file, OPTICAL_HINTS)]
    optical_by_key = {_scene_key(path): path for path in optical_files}

    pairs: list[PairRecord] = []
    missing_optical: list[str] = []
    used_optical: set[Path] = set()
    for sar_path in sorted(sar_files):
        key = _scene_key(sar_path)
        optical_path = optical_by_key.get(key)
        if optical_path is None:
            missing_optical.append(str(sar_path))
            continue
        used_optical.add(optical_path)
        pairs.append(_validate_pair(sar_path, optical_path, key))

    orphan_optical = [str(path) for path in optical_files if path not in used_optical]
    valid_pairs = [pair for pair in pairs if pair.valid]
    manifests = generate_manifests(valid_pairs)
    return {
        "root_path": str(root),
        "sar_files": len(sar_files),
        "optical_files": len(optical_files),
        "pair_count": len(pairs),
        "valid_pair_count": len(valid_pairs),
        "invalid_pair_count": len(pairs) - len(valid_pairs),
        "missing_optical": missing_optical,
        "orphan_optical": orphan_optical,
        "pairs": [pair.to_dict() for pair in pairs],
        "manifests": manifests,
    }


def generate_manifests(pairs: list[PairRecord]) -> dict[str, list[str]]:
    by_region: dict[str, list[str]] = {}
    for pair in pairs:
        by_region.setdefault(pair.region_key, []).append(pair.scene_id)
    regions = sorted(by_region)
    train_regions, val_regions, test_regions = _region_split(regions)
    return {
        "phase0_train": _take_scene_ids(by_region, train_regions, limit=2000),
        "phase1_train": _take_scene_ids(by_region, train_regions, limit=20000),
        "validation": _take_scene_ids(by_region, val_regions),
        "test": _take_scene_ids(by_region, test_regions),
        "train_regions": train_regions,
        "validation_regions": val_regions,
        "test_regions": test_regions,
    }


def _validate_pair(sar_path: Path, optical_path: Path, scene_id: str) -> PairRecord:
    errors: list[str] = []
    width: int | None = None
    height: int | None = None
    channels: int | None = None
    try:
        with Image.open(sar_path) as sar_image:
            sar_width, sar_height = sar_image.size
    except Exception as exc:
        errors.append(f"SAR unreadable: {exc}")
        sar_width = sar_height = 0
    try:
        with Image.open(optical_path) as optical_image:
            optical_width, optical_height = optical_image.size
            channels = len(optical_image.getbands())
    except Exception as exc:
        errors.append(f"Optical unreadable: {exc}")
        optical_width = optical_height = 0
        channels = None

    if sar_width and optical_width:
        width = sar_width
        height = sar_height
        if (sar_width, sar_height) != (optical_width, optical_height):
            errors.append(f"Dimension mismatch: SAR {sar_width}x{sar_height}, optical {optical_width}x{optical_height}")
    return PairRecord(
        scene_id=scene_id,
        sar_path=str(sar_path),
        optical_path=str(optical_path),
        width=width,
        height=height,
        channels=channels,
        valid=not errors,
        errors=tuple(errors),
        region_key=_region_key(sar_path),
    )


def _looks_like(path: Path, hints: set[str]) -> bool:
    parts = {part.lower().replace("_", "").replace("-", "") for part in path.parts}
    stem = path.stem.lower().replace("_", "").replace("-", "")
    return bool(parts.intersection({hint.replace("-", "") for hint in hints})) or any(hint.replace("-", "") in stem for hint in hints)


def _scene_key(path: Path) -> str:
    stem = path.stem.lower()
    for token in ("_sar", "-sar", "_s1", "-s1", "_vv", "-vv", "_vh", "-vh", "_optical", "-optical", "_s2", "-s2", "_rgb", "-rgb"):
        stem = stem.replace(token, "")
    return hashlib.sha1(stem.encode("utf-8")).hexdigest()[:16]


def _region_key(path: Path) -> str:
    parts = [part for part in path.parts if part]
    for index, part in enumerate(parts):
        normalized = part.lower()
        if normalized in {"train", "validation", "val", "test", "raw", "sar", "optical"} and index > 0:
            return parts[index - 1].lower()
    return parts[-3].lower() if len(parts) >= 3 else "unknown-region"


def _region_split(regions: list[str]) -> tuple[list[str], list[str], list[str]]:
    if not regions:
        return [], [], []
    count = len(regions)
    train_count = max(1, round(count * 0.70))
    val_count = max(1 if count >= 3 else 0, round(count * 0.15))
    if train_count + val_count >= count and count > 1:
        train_count = max(1, count - 1)
        val_count = 0 if count == 2 else 1
    train_regions = regions[:train_count]
    val_regions = regions[train_count : train_count + val_count]
    test_regions = regions[train_count + val_count :]
    return train_regions, val_regions, test_regions


def _take_scene_ids(by_region: dict[str, list[str]], regions: list[str], limit: int | None = None) -> list[str]:
    scene_ids: list[str] = []
    for region in regions:
        scene_ids.extend(sorted(by_region.get(region, [])))
    return scene_ids if limit is None else scene_ids[:limit]

