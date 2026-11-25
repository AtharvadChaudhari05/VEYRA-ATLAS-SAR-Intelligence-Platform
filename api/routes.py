from __future__ import annotations

import json
from pathlib import Path
from typing import Annotated

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse

from app.core.config import Settings, ensure_storage, get_settings
from app.core.database import Database
from app.core.repository import Repository
from app.models.schemas import (
    AnalysisResponse,
    DatasetScanRequest,
    DatasetScanResponse,
    HealthResponse,
    JobResponse,
    PredictionResponse,
    SceneMetadataResponse,
)
from app.services.inference import MODEL_1_VERSION, MODEL_2_VERSION, SUPPORTED_DISASTERS, model1_runtime
from app.services.learned_flood import flood_runtime
from app.services.pipeline import RemoteSensingPipeline
from app.services.scanner import scan_dataset


router = APIRouter()


def get_repository(settings: Settings = Depends(get_settings)) -> Repository:
    database = Database(settings.database_path)
    database.initialize()
    return Repository(database)


def get_pipeline(
    settings: Settings = Depends(get_settings),
    repository: Repository = Depends(get_repository),
) -> RemoteSensingPipeline:
    ensure_storage(settings)
    return RemoteSensingPipeline(settings, repository)


@router.get("/health", response_model=HealthResponse)
def health(settings: Settings = Depends(get_settings)) -> HealthResponse:
    model1 = model1_runtime(settings.model1_checkpoint, enable_experimental=settings.enable_experimental_model1)
    model2 = flood_runtime(settings.flood_model_onnx, settings.flood_model_promotion)
    return HealthResponse(
        status="ok",
        app_name=settings.app_name,
        version=settings.app_version,
        models={
            "sar_to_optical": model1["version"],
            "disaster_segmentation": model2["version"],
            "supported_disasters": sorted(SUPPORTED_DISASTERS),
        },
    )


@router.get("/models/status")
def model_status(settings: Settings = Depends(get_settings)) -> dict[str, object]:
    model1 = model1_runtime(settings.model1_checkpoint, enable_experimental=settings.enable_experimental_model1)
    model2 = flood_runtime(settings.flood_model_onnx, settings.flood_model_promotion)
    return {
        "model_1": {
            "name": "SAR-to-optical representation",
            "version": model1["version"],
            "outputs": ["optical_like", "uncertainty", "sar_preservation"],
            "runtime": model1["runtime"],
            "status": model1["status"],
        },
        "model_2": {
            "name": "Disaster segmentation",
            "version": model2["version"],
            "supported_disasters": sorted(SUPPORTED_DISASTERS),
            "runtime": model2["runtime"],
            "status": model2["status"],
            "activation_rule": "ONNX model activates only after it beats the deterministic baseline on official test and Bolivia hold-out data.",
        },
    }


@router.post("/predict", response_model=PredictionResponse)
async def predict(
    file: Annotated[UploadFile, File(description="SAR image upload. GeoTIFF is preferred.")],
    crs: Annotated[str | None, Form()] = None,
    transform: Annotated[str | None, Form(description="JSON list [a,b,c,d,e,f] for x=a*col+b*row+c, y=d*col+e*row+f.")] = None,
    bounds: Annotated[str | None, Form(description="JSON list [minx,miny,maxx,maxy].")] = None,
    pipeline: RemoteSensingPipeline = Depends(get_pipeline),
) -> PredictionResponse:
    try:
        file_bytes = await file.read()
        stored = pipeline.run_prediction(
            file_bytes=file_bytes,
            filename=file.filename or "upload.bin",
            content_type=file.content_type,
            crs=crs,
            transform=_parse_float_list(transform, "transform", expected=6),
            bounds=_parse_float_list(bounds, "bounds", expected=4),
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {exc}") from exc
    return _prediction_response(stored.scene, stored.prediction, stored.job)


@router.post("/analyze", response_model=AnalysisResponse)
async def analyze(
    file: Annotated[UploadFile, File(description="SAR image upload. GeoTIFF is preferred.")],
    disaster_type: Annotated[str | None, Form()] = None,
    threshold: Annotated[float | None, Form()] = None,
    crs: Annotated[str | None, Form()] = None,
    transform: Annotated[str | None, Form(description="JSON list [a,b,c,d,e,f] for x=a*col+b*row+c, y=d*col+e*row+f.")] = None,
    bounds: Annotated[str | None, Form(description="JSON list [minx,miny,maxx,maxy].")] = None,
    pipeline: RemoteSensingPipeline = Depends(get_pipeline),
) -> AnalysisResponse:
    try:
        file_bytes = await file.read()
        stored = pipeline.run_analysis(
            file_bytes=file_bytes,
            filename=file.filename or "upload.bin",
            content_type=file.content_type,
            disaster_type=disaster_type,
            threshold=threshold,
            crs=crs,
            transform=_parse_float_list(transform, "transform", expected=6),
            bounds=_parse_float_list(bounds, "bounds", expected=4),
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Analysis failed: {exc}") from exc
    return _analysis_response(stored.scene, stored.prediction, stored.analysis, stored.job)


@router.get("/metadata", response_model=list[SceneMetadataResponse])
def list_metadata(repository: Repository = Depends(get_repository)) -> list[SceneMetadataResponse]:
    return [_scene_response(scene) for scene in repository.list_scenes()]


@router.get("/metadata/{scene_id}", response_model=SceneMetadataResponse)
def get_metadata(scene_id: str, repository: Repository = Depends(get_repository)) -> SceneMetadataResponse:
    try:
        scene = repository.get_scene(scene_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    return _scene_response(scene)


@router.get("/jobs/{job_id}", response_model=JobResponse)
def get_job(job_id: str, repository: Repository = Depends(get_repository)) -> JobResponse:
    try:
        return JobResponse(**repository.get_job(job_id))
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@router.get("/layers/{scene_id}/{layer_name}")
def get_layer(
    scene_id: str,
    layer_name: str,
    settings: Settings = Depends(get_settings),
    repository: Repository = Depends(get_repository),
) -> FileResponse:
    try:
        scene = repository.get_scene(scene_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    result_dir = settings.result_dir / scene_id
    candidates = {
        "optical_like": result_dir / "optical_like.png",
        "uncertainty": result_dir / "uncertainty.png",
        "sar_preservation": result_dir / "sar_preservation.png",
        "probability": result_dir / "disaster_probability.png",
        "segmentation_mask": result_dir / "segmentation_mask.png",
        "affected_boundary": result_dir / "affected_boundary.geojson",
    }
    path = candidates.get(layer_name)
    if path is None:
        raise HTTPException(status_code=404, detail=f"Unknown layer: {layer_name}")
    if not path.exists():
        raise HTTPException(status_code=404, detail=f"Layer has not been generated: {layer_name}")
    media_type = "application/geo+json" if path.suffix == ".geojson" else "image/png"
    return FileResponse(path, media_type=media_type, filename=path.name)


@router.post("/datasets/scan", response_model=DatasetScanResponse)
def scan_dataset_endpoint(request: DatasetScanRequest) -> DatasetScanResponse:
    try:
        return DatasetScanResponse(**scan_dataset(request.root_path))
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Dataset scan failed: {exc}") from exc


def _parse_float_list(value: str | None, field_name: str, *, expected: int) -> list[float] | None:
    if not value:
        return None
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError as exc:
        raise ValueError(f"{field_name} must be valid JSON") from exc
    if not isinstance(parsed, list) or len(parsed) != expected:
        raise ValueError(f"{field_name} must be a JSON list with {expected} numbers")
    try:
        return [float(item) for item in parsed]
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field_name} must contain only numbers") from exc


def _scene_response(scene: dict[str, object]) -> SceneMetadataResponse:
    return SceneMetadataResponse(
        id=str(scene["id"]),
        filename=str(scene["filename"]),
        width=int(scene["width"]),
        height=int(scene["height"]),
        crs=scene.get("crs") if isinstance(scene.get("crs"), str) else None,
        transform=scene.get("transform") if isinstance(scene.get("transform"), list) else None,
        bounds=scene.get("bounds") if isinstance(scene.get("bounds"), list) else None,
        metadata=scene.get("metadata") if isinstance(scene.get("metadata"), dict) else {},
        created_at=str(scene["created_at"]),
    )


def _prediction_response(scene: dict[str, object], prediction: dict[str, object], job: dict[str, object]) -> PredictionResponse:
    summary = prediction.get("summary") if isinstance(prediction.get("summary"), dict) else {}
    return PredictionResponse(
        scene_id=str(scene["id"]),
        prediction_id=str(prediction["id"]),
        job_id=str(job["id"]),
        model_version=str(prediction["model_version"]),
        confidence=float(summary.get("confidence", 0.0)),
        metadata=scene.get("metadata") if isinstance(scene.get("metadata"), dict) else {},
        layers={
            "optical_like": f"/layers/{scene['id']}/optical_like",
            "uncertainty": f"/layers/{scene['id']}/uncertainty",
            "sar_preservation": f"/layers/{scene['id']}/sar_preservation",
        },
        created_at=str(prediction["created_at"]),
    )


def _analysis_response(scene: dict[str, object], prediction: dict[str, object], analysis: dict[str, object], job: dict[str, object]) -> AnalysisResponse:
    summary = analysis.get("summary") if isinstance(analysis.get("summary"), dict) else {}
    return AnalysisResponse(
        scene_id=str(scene["id"]),
        prediction_id=str(prediction["id"]),
        analysis_id=str(analysis["id"]),
        job_id=str(job["id"]),
        disaster_type=str(analysis["disaster_type"]),
        threshold=float(analysis["threshold"]),
        affected_area_km2=analysis["affected_area_km2"] if analysis["affected_area_km2"] is None else float(analysis["affected_area_km2"]),
        coverage_percent=float(analysis["coverage_percent"]),
        confidence=float(analysis["confidence"]),
        area_method=str(summary.get("area_method", "unknown")),
        area_warning=summary.get("area_warning") if isinstance(summary.get("area_warning"), str) else None,
        layers={
            "optical_like": f"/layers/{scene['id']}/optical_like",
            "uncertainty": f"/layers/{scene['id']}/uncertainty",
            "sar_preservation": f"/layers/{scene['id']}/sar_preservation",
            "probability": f"/layers/{scene['id']}/probability",
            "segmentation_mask": f"/layers/{scene['id']}/segmentation_mask",
            "affected_boundary": f"/layers/{scene['id']}/affected_boundary",
        },
        boundary_geojson_path=str(analysis["boundary_geojson_path"]),
        created_at=str(analysis["created_at"]),
    )
