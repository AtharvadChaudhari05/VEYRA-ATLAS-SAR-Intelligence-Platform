from __future__ import annotations

import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path


def _optional_path_from_env(name: str) -> Path | None:
    value = os.getenv(name, "").strip()
    return Path(value) if value else None


@dataclass(frozen=True)
class Settings:
    app_name: str
    app_version: str
    storage_dir: Path
    upload_dir: Path
    result_dir: Path
    database_path: Path
    max_image_side: int
    default_disaster_type: str
    default_threshold: float
    model1_checkpoint: Path | None
    enable_experimental_model1: bool
    flood_model_onnx: Path | None
    flood_model_promotion: Path | None


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    project_dir = Path(__file__).resolve().parents[1]
    storage_dir = Path(os.getenv("RSAI_STORAGE_DIR", str(project_dir / "storage")))
    return Settings(
        app_name="Veyra Atlas",
        app_version="0.2.0",
        storage_dir=storage_dir,
        upload_dir=storage_dir / "uploads",
        result_dir=storage_dir / "results",
        database_path=storage_dir / "remote_sensing.db",
        max_image_side=int(os.getenv("RSAI_MAX_IMAGE_SIDE", "512")),
        default_disaster_type="flood",
        default_threshold=float(os.getenv("RSAI_DEFAULT_THRESHOLD", "0.55")),
        model1_checkpoint=_optional_path_from_env("RSAI_MODEL1_CHECKPOINT"),
        enable_experimental_model1=os.getenv("RSAI_ENABLE_EXPERIMENTAL_MODEL1", "false").lower() in {"1", "true", "yes"},
        flood_model_onnx=_optional_path_from_env("RSAI_FLOOD_MODEL_ONNX"),
        flood_model_promotion=_optional_path_from_env("RSAI_FLOOD_MODEL_PROMOTION"),
    )


def ensure_storage(settings: Settings) -> None:
    settings.upload_dir.mkdir(parents=True, exist_ok=True)
    settings.result_dir.mkdir(parents=True, exist_ok=True)
