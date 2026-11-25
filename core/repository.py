from __future__ import annotations

from pathlib import Path
from typing import Any

from app.core.database import Database, json_dumps, json_loads, utc_now


class Repository:
    def __init__(self, database: Database):
        self.database = database

    def create_scene(
        self,
        *,
        scene_id: str,
        filename: str,
        content_type: str | None,
        upload_path: Path,
        width: int,
        height: int,
        crs: str | None,
        transform: list[float] | None,
        bounds: list[float] | None,
        metadata: dict[str, Any],
    ) -> dict[str, Any]:
        created_at = utc_now()
        with self.database.connect() as connection:
            connection.execute(
                """
                INSERT INTO scenes (
                    id, filename, content_type, upload_path, width, height, crs,
                    transform_json, bounds_json, metadata_json, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    scene_id,
                    filename,
                    content_type,
                    str(upload_path),
                    width,
                    height,
                    crs,
                    json_dumps(transform) if transform else None,
                    json_dumps(bounds) if bounds else None,
                    json_dumps(metadata),
                    created_at,
                ),
            )
        return self.get_scene(scene_id)

    def get_scene(self, scene_id: str) -> dict[str, Any]:
        with self.database.connect() as connection:
            row = connection.execute("SELECT * FROM scenes WHERE id = ?", (scene_id,)).fetchone()
        if row is None:
            raise KeyError(f"Scene not found: {scene_id}")
        return self._scene_from_row(row)

    def list_scenes(self) -> list[dict[str, Any]]:
        with self.database.connect() as connection:
            rows = connection.execute("SELECT * FROM scenes ORDER BY created_at DESC").fetchall()
        return [self._scene_from_row(row) for row in rows]

    def create_prediction(
        self,
        *,
        prediction_id: str,
        scene_id: str,
        model_version: str,
        optical_image_path: Path,
        uncertainty_path: Path,
        preservation_path: Path,
        summary: dict[str, Any],
    ) -> dict[str, Any]:
        created_at = utc_now()
        with self.database.connect() as connection:
            connection.execute(
                """
                INSERT INTO predictions (
                    id, scene_id, model_version, optical_image_path, uncertainty_path,
                    preservation_path, summary_json, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    prediction_id,
                    scene_id,
                    model_version,
                    str(optical_image_path),
                    str(uncertainty_path),
                    str(preservation_path),
                    json_dumps(summary),
                    created_at,
                ),
            )
        return self.get_prediction(prediction_id)

    def get_prediction(self, prediction_id: str) -> dict[str, Any]:
        with self.database.connect() as connection:
            row = connection.execute("SELECT * FROM predictions WHERE id = ?", (prediction_id,)).fetchone()
        if row is None:
            raise KeyError(f"Prediction not found: {prediction_id}")
        return self._prediction_from_row(row)

    def create_analysis(
        self,
        *,
        analysis_id: str,
        scene_id: str,
        prediction_id: str,
        disaster_type: str,
        threshold: float,
        affected_area_km2: float | None,
        coverage_percent: float,
        confidence: float,
        probability_path: Path,
        mask_path: Path,
        boundary_geojson_path: Path,
        summary: dict[str, Any],
    ) -> dict[str, Any]:
        created_at = utc_now()
        with self.database.connect() as connection:
            connection.execute(
                """
                INSERT INTO analyses (
                    id, scene_id, prediction_id, disaster_type, threshold,
                    affected_area_km2, coverage_percent, confidence,
                    probability_path, mask_path, boundary_geojson_path,
                    summary_json, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    analysis_id,
                    scene_id,
                    prediction_id,
                    disaster_type,
                    threshold,
                    affected_area_km2,
                    coverage_percent,
                    confidence,
                    str(probability_path),
                    str(mask_path),
                    str(boundary_geojson_path),
                    json_dumps(summary),
                    created_at,
                ),
            )
        return self.get_analysis(analysis_id)

    def get_analysis(self, analysis_id: str) -> dict[str, Any]:
        with self.database.connect() as connection:
            row = connection.execute("SELECT * FROM analyses WHERE id = ?", (analysis_id,)).fetchone()
        if row is None:
            raise KeyError(f"Analysis not found: {analysis_id}")
        return self._analysis_from_row(row)

    def create_job(self, *, job_id: str, kind: str, status: str, message: str, result_id: str | None = None) -> dict[str, Any]:
        now = utc_now()
        with self.database.connect() as connection:
            connection.execute(
                "INSERT INTO jobs (id, kind, status, message, result_id, created_at, updated_at) VALUES (?, ?, ?, ?, ?, ?, ?)",
                (job_id, kind, status, message, result_id, now, now),
            )
        return self.get_job(job_id)

    def get_job(self, job_id: str) -> dict[str, Any]:
        with self.database.connect() as connection:
            row = connection.execute("SELECT * FROM jobs WHERE id = ?", (job_id,)).fetchone()
        if row is None:
            raise KeyError(f"Job not found: {job_id}")
        return dict(row)

    @staticmethod
    def _scene_from_row(row: Any) -> dict[str, Any]:
        item = dict(row)
        item["transform"] = json_loads(item.pop("transform_json"), None)
        item["bounds"] = json_loads(item.pop("bounds_json"), None)
        item["metadata"] = json_loads(item.pop("metadata_json"), {})
        return item

    @staticmethod
    def _prediction_from_row(row: Any) -> dict[str, Any]:
        item = dict(row)
        item["summary"] = json_loads(item.pop("summary_json"), {})
        return item

    @staticmethod
    def _analysis_from_row(row: Any) -> dict[str, Any]:
        item = dict(row)
        item["summary"] = json_loads(item.pop("summary_json"), {})
        return item

