from __future__ import annotations

import json
import shutil
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from app.core.config import Settings
from app.core.repository import Repository
from app.services.geospatial import calculate_affected_area, mask_to_geojson
from app.services.image_io import RasterScene, load_raster_from_bytes, save_png
from app.services.inference import DisasterOutput, Model1Output, analyze_disaster, analyze_learned_flood, predict_sar_to_optical
from app.services.learned_flood import flood_runtime, predict_probability


@dataclass(frozen=True)
class StoredPrediction:
    scene: dict[str, Any]
    prediction: dict[str, Any]
    job: dict[str, Any]
    model1: Model1Output
    raster: RasterScene


@dataclass(frozen=True)
class StoredAnalysis:
    scene: dict[str, Any]
    prediction: dict[str, Any]
    analysis: dict[str, Any]
    job: dict[str, Any]
    model1: Model1Output
    model2: DisasterOutput


class RemoteSensingPipeline:
    def __init__(self, settings: Settings, repository: Repository):
        self.settings = settings
        self.repository = repository

    def run_prediction(
        self,
        *,
        file_bytes: bytes,
        filename: str,
        content_type: str | None,
        crs: str | None = None,
        transform: list[float] | None = None,
        bounds: list[float] | None = None,
    ) -> StoredPrediction:
        scene_id = uuid.uuid4().hex
        prediction_id = uuid.uuid4().hex
        safe_name = _safe_filename(filename)
        upload_path = self.settings.upload_dir / f"{scene_id}_{safe_name}"
        upload_path.parent.mkdir(parents=True, exist_ok=True)
        upload_path.write_bytes(file_bytes)

        raster = load_raster_from_bytes(
            file_bytes,
            filename,
            max_side=self.settings.max_image_side,
            crs=crs,
            transform=transform,
            bounds=bounds,
        )
        model1 = predict_sar_to_optical(
            raster.sar,
            checkpoint_path=self.settings.model1_checkpoint,
            enable_experimental=self.settings.enable_experimental_model1,
        )

        result_dir = self.settings.result_dir / scene_id
        optical_path = result_dir / "optical_like.png"
        uncertainty_path = result_dir / "uncertainty.png"
        preservation_path = result_dir / "sar_preservation.png"
        save_png(model1.optical_like, optical_path)
        save_png(model1.uncertainty, uncertainty_path, heatmap=True)
        save_png(model1.preservation, preservation_path, heatmap=True)

        metadata = raster.metadata.to_dict()
        scene = self.repository.create_scene(
            scene_id=scene_id,
            filename=filename,
            content_type=content_type,
            upload_path=upload_path,
            width=raster.metadata.width,
            height=raster.metadata.height,
            crs=raster.metadata.crs,
            transform=raster.metadata.transform,
            bounds=raster.metadata.bounds,
            metadata=metadata,
        )
        summary = {
            "confidence": model1.confidence,
            "width": raster.metadata.width,
            "height": raster.metadata.height,
            "notes": list(raster.metadata.notes),
            "layer_paths": {
                "optical_like": str(optical_path),
                "uncertainty": str(uncertainty_path),
                "sar_preservation": str(preservation_path),
            },
        }
        prediction = self.repository.create_prediction(
            prediction_id=prediction_id,
            scene_id=scene_id,
            model_version=model1.model_version,
            optical_image_path=optical_path,
            uncertainty_path=uncertainty_path,
            preservation_path=preservation_path,
            summary=summary,
        )
        job = self.repository.create_job(job_id=uuid.uuid4().hex, kind="predict", status="completed", message="Prediction completed", result_id=prediction_id)
        return StoredPrediction(scene=scene, prediction=prediction, job=job, model1=model1, raster=raster)

    def run_analysis(
        self,
        *,
        file_bytes: bytes,
        filename: str,
        content_type: str | None,
        disaster_type: str | None = None,
        threshold: float | None = None,
        crs: str | None = None,
        transform: list[float] | None = None,
        bounds: list[float] | None = None,
    ) -> StoredAnalysis:
        stored = self.run_prediction(
            file_bytes=file_bytes,
            filename=filename,
            content_type=content_type,
            crs=crs,
            transform=transform,
            bounds=bounds,
        )
        requested_disaster = disaster_type or self.settings.default_disaster_type
        requested_threshold = threshold if threshold is not None else self.settings.default_threshold
        learned = flood_runtime(self.settings.flood_model_onnx, self.settings.flood_model_promotion)
        if (
            requested_disaster.lower().strip() == "flood"
            and learned["runtime"] == "onnxruntime"
            and stored.raster.sar_bands is not None
            and self.settings.flood_model_onnx is not None
        ):
            model2 = analyze_learned_flood(
                predict_probability(stored.raster.sar_bands, self.settings.flood_model_onnx),
                threshold=requested_threshold,
            )
        else:
            model2 = analyze_disaster(
                stored.raster.sar,
                stored.model1,
                disaster_type=requested_disaster,
                threshold=requested_threshold,
            )
        area = calculate_affected_area(model2.mask, stored.raster.metadata)
        boundaries = mask_to_geojson(model2.mask, stored.raster.metadata)

        analysis_id = uuid.uuid4().hex
        result_dir = self.settings.result_dir / stored.scene["id"]
        probability_path = result_dir / "disaster_probability.png"
        mask_path = result_dir / "segmentation_mask.png"
        boundary_path = result_dir / "affected_boundary.geojson"
        save_png(model2.probability, probability_path, heatmap=True)
        save_png(model2.mask.astype(np.float32), mask_path)
        boundary_path.write_text(json.dumps(boundaries, indent=2), encoding="utf-8")

        summary = {
            "model_version": model2.model_version,
            "area_method": area.method,
            "area_warning": area.warning,
            "pixel_count": area.pixel_count,
            "total_pixels": area.total_pixels,
            "boundary_feature_count": len(boundaries["features"]),
            "layer_paths": {
                "probability": str(probability_path),
                "segmentation_mask": str(mask_path),
                "affected_boundary": str(boundary_path),
            },
        }
        analysis = self.repository.create_analysis(
            analysis_id=analysis_id,
            scene_id=stored.scene["id"],
            prediction_id=stored.prediction["id"],
            disaster_type=model2.disaster_type,
            threshold=model2.threshold,
            affected_area_km2=area.affected_area_km2,
            coverage_percent=model2.coverage_percent,
            confidence=model2.confidence,
            probability_path=probability_path,
            mask_path=mask_path,
            boundary_geojson_path=boundary_path,
            summary=summary,
        )
        job = self.repository.create_job(job_id=uuid.uuid4().hex, kind="analyze", status="completed", message="Analysis completed", result_id=analysis_id)
        return StoredAnalysis(
            scene=stored.scene,
            prediction=stored.prediction,
            analysis=analysis,
            job=job,
            model1=stored.model1,
            model2=model2,
        )


def copy_result_file(path: str, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(path, destination)


def _safe_filename(filename: str) -> str:
    safe = "".join(char if char.isalnum() or char in {".", "_", "-"} else "_" for char in filename)
    return safe or "upload.bin"
