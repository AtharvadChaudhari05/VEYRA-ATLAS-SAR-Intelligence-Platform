from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class HealthResponse(BaseModel):
    status: str
    app_name: str
    version: str
    models: dict[str, Any]


class SceneMetadataResponse(BaseModel):
    id: str
    filename: str
    width: int
    height: int
    crs: str | None
    transform: list[float] | None
    bounds: list[float] | None
    metadata: dict[str, Any]
    created_at: str


class PredictionResponse(BaseModel):
    scene_id: str
    prediction_id: str
    job_id: str
    model_version: str
    confidence: float = Field(ge=0.0, le=1.0)
    metadata: dict[str, Any]
    layers: dict[str, str]
    created_at: str


class AnalysisResponse(BaseModel):
    scene_id: str
    prediction_id: str
    analysis_id: str
    job_id: str
    disaster_type: str
    threshold: float
    affected_area_km2: float | None
    coverage_percent: float
    confidence: float = Field(ge=0.0, le=1.0)
    area_method: str
    area_warning: str | None
    layers: dict[str, str]
    boundary_geojson_path: str
    created_at: str


class DatasetScanRequest(BaseModel):
    root_path: str


class DatasetScanResponse(BaseModel):
    root_path: str
    sar_files: int
    optical_files: int
    pair_count: int
    valid_pair_count: int
    invalid_pair_count: int
    missing_optical: list[str]
    orphan_optical: list[str]
    pairs: list[dict[str, Any]]
    manifests: dict[str, Any]


class JobResponse(BaseModel):
    id: str
    kind: str
    status: str
    message: str
    result_id: str | None
    created_at: str
    updated_at: str
