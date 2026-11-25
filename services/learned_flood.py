"""Guarded ONNX Runtime adapter for the trained Sen1Floods11 flood model.

Weights alone are deliberately insufficient for production use.  A promotion
record produced by ``ml/finalize_flood.py`` must say that the checkpoint beat
the deterministic baseline on both official held-out splits.
"""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np

from ml.flood_data import NormalizationStats, prepare_features


LEARNED_MODEL_VERSION = "flood-unet-sen1floods11-v1"
PROMOTED_STATUS = "eligible_for_manual_backend_integration"


def flood_runtime(onnx_path: Path | None, promotion_path: Path | None) -> dict[str, str]:
    """Return runtime status without silently enabling unvalidated weights."""
    if onnx_path is None and promotion_path is None:
        return {"runtime": "deterministic_baseline", "version": "deterministic-flood-segmentation-v0", "status": "training_or_not_configured"}
    if onnx_path is None or not onnx_path.exists():
        return {"runtime": "deterministic_baseline", "version": "deterministic-flood-segmentation-v0", "status": "onnx_weights_not_found"}
    if promotion_path is None or not promotion_path.exists():
        return {"runtime": "deterministic_baseline", "version": "deterministic-flood-segmentation-v0", "status": "awaiting_official_evaluation_gate"}
    try:
        promotion = json.loads(promotion_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {"runtime": "deterministic_baseline", "version": "deterministic-flood-segmentation-v0", "status": "promotion_record_invalid"}
    if promotion.get("status") != PROMOTED_STATUS:
        return {"runtime": "deterministic_baseline", "version": "deterministic-flood-segmentation-v0", "status": "evaluation_gate_not_passed"}
    try:
        _load_session(str(onnx_path))
    except Exception:
        return {"runtime": "deterministic_baseline", "version": "deterministic-flood-segmentation-v0", "status": "onnx_runtime_load_failed"}
    return {"runtime": "onnxruntime", "version": LEARNED_MODEL_VERSION, "status": "validated_active"}


@lru_cache(maxsize=2)
def _load_session(onnx_path: str) -> Any:
    import onnxruntime as ort

    providers = ["CPUExecutionProvider"]
    return ort.InferenceSession(onnx_path, providers=providers)


def predict_probability(sar_bands: np.ndarray, onnx_path: Path) -> np.ndarray:
    """Run the promoted model on raw VV/VH, returning one probability per pixel."""
    features = prepare_features(sar_bands[:2], NormalizationStats())[None, ...]
    session = _load_session(str(onnx_path))
    input_name = session.get_inputs()[0].name
    logits = np.asarray(session.run(None, {input_name: features})[0], dtype=np.float32)
    if logits.ndim == 3:
        logits = logits[0]
    if logits.ndim != 2:
        raise ValueError(f"Unexpected ONNX flood output shape: {logits.shape}")
    return (1.0 / (1.0 + np.exp(-np.clip(logits, -60.0, 60.0)))).astype(np.float32)
