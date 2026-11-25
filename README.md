# Veyra Atlas — SAR Intelligence Platform

Veyra Atlas is a full-stack remote-sensing application that helps users inspect Synthetic Aperture Radar (SAR) scenes. It turns SAR inputs into an optical-like visual interpretation, provides flood-analysis layers, and preserves geospatial metadata for defensible area and boundary reporting.

## The basic idea

Optical satellite images are easy to understand, but clouds, night, and severe weather often obscure them. SAR is available day and night and can see through clouds, but it is grayscale and difficult to interpret. Veyra Atlas provides a clear workflow for students, researchers, and disaster-response users:

1. Upload a SAR image, preferably a two-band Sentinel-1 GeoTIFF.
2. Inspect an optical-like representation and trust layers.
3. Run flood analysis to produce a probability raster, segmentation mask, boundary, coverage, and affected-area estimate.
4. Use the map only when the file contains real georeferencing; otherwise, retain the image in local scene coordinates instead of inventing a location.

## What is included

### Frontend

The Next.js interface provides:

- A light, responsive mission dashboard.
- Separate Predict and Analyze workflows.
- Upload support for PNG, JPEG, WebP, BMP, GIF, TIFF, and GeoTIFF.
- Layer controls for optical-like imagery, probability, segmentation, and boundaries.
- Satellite basemaps for WGS84-georeferenced scenes only.
- Model runtime information so users can see which model is actually active.

### Backend

The FastAPI service handles file upload, raster processing, model inference, result storage, and geospatial calculations.

Important endpoints include:

| Endpoint | Purpose |
| --- | --- |
| `GET /health` | Application health and active model versions |
| `GET /models/status` | Runtime, validation, and deployment status for both models |
| `POST /predict` | SAR-to-optical-like interpretation |
| `POST /analyze` | Flood probability, mask, boundary, and area analysis |
| `GET /layers/{scene_id}/{layer_name}` | Generated raster and GeoJSON layers |
| `POST /datasets/scan` | Dataset folder diagnostics |

### Model 1 — SAR representation

Model 1 creates an **optical-like representation**, uncertainty field, and signal-preservation layer from the uploaded SAR image. The deployed default is a deterministic, explainable baseline. It is intended for interpretation, not as a claim that it recreates a ground-truth optical photograph.

### Model 2 — flood segmentation

Model 2 is a U-Net flood-segmentation pipeline trained for two-band Sentinel-1 VV/VH SAR input. Its preprocessing creates VV, VH, and VV-minus-VH features. Its outputs are a flood probability field and a thresholded segmentation mask.

The application uses a strict deployment gate:

1. Train on official Sen1Floods11 hand-labelled SAR data.
2. Evaluate on the official test split.
3. Evaluate separately on the Bolivia geographic hold-out.
4. Compare against the deterministic baseline on both splits.
5. Activate the exported ONNX model only when it wins both comparisons.

Until that gate passes, the production API intentionally uses the deterministic flood baseline. The status is visible at `/models/status`; this prevents presenting an unvalidated ML checkpoint as a reliable live model.

## Geospatial behaviour

For a GeoTIFF with CRS and affine transform information, Veyra Atlas calculates an area estimate and displays the boundary in its true map location. For image-only uploads, exact physical area and real-world placement are unavailable. In that case the platform reports the limitation rather than drawing a false map location.

## Technology stack

- **Frontend:** Next.js 14, React, TypeScript, Tailwind CSS, TanStack Query, MapLibre
- **Backend:** FastAPI, Pydantic, SQLite
- **Raster processing:** NumPy, Pillow, Rasterio
- **Machine learning:** PyTorch training pipeline, ONNX export, ONNX Runtime deployment adapter
- **Dataset:** official Sen1Floods11 hand-labelled Sentinel-1 SAR split

## Local setup

### Backend

Use Python 3.9+ and install the application/training dependencies as required by your environment. Start the API from the directory that contains the `app` package:

```powershell
$env:PYTHONPATH = ".runtime;..;."
python -m uvicorn app.main:app --host 127.0.0.1 --port 8000
```

### Frontend

```powershell
cd web
npm install
npm run dev
```

Open `http://127.0.0.1:3000`.

## Training Model 2

Training configurations, official-dataset downloader, evaluation code, ONNX exporter, and promotion gate are versioned in this repository:

- `scripts/download_sen1floods11.py` — dataset download helper
- `configs/flood_sen1floods11*.yaml` — smoke, full CPU, and continuation configurations
- `ml/train_flood.py` — training loop with validation and early stopping
- `ml/eval_flood.py` — official test and Bolivia hold-out evaluation
- `ml/finalize_flood.py` — mandatory deployment decision
- `ml/export_flood_onnx.py` — ONNX export
- `scripts/deploy_validated_flood_model.ps1` — guarded local activation helper

See [TRAINING.md](TRAINING.md) and [docs/EVALUATION.md](docs/EVALUATION.md) for the reproducible workflow and currently measured results.

## Repository layout

```text
api/          FastAPI HTTP routes
core/         Configuration, SQLite database, repository layer
services/     Raster I/O, inference, pipeline, and geospatial logic
models/       API schemas and model registry
ml/           Data preparation, training, metrics, evaluation, and export
configs/      Training configurations
scripts/      Dataset, evaluation, and deployment helpers
web/          Next.js frontend
docs/         Audit and evaluation notes
```

## Development roadmap

1. Complete the active Model 2 training run and retain the best validation checkpoint.
2. Run official test and Bolivia hold-out evaluation, then promote only if both baseline comparisons pass.
3. Deploy the promoted ONNX flood model through the guarded runtime adapter.
4. Add calibrated uncertainty layers and per-scene quality checks for the learned flood model.
5. Extend supported event types only after obtaining dedicated labelled data and independent evaluations for each class.
6. Add automated tests, CI, model-version tracking, and a reproducible release artifact.
7. Add authenticated cloud storage and production deployment configuration for multi-user operation.

## Current limitations

- Flood is the only supported disaster type.
- Model 1 produces an optical-like visual aid, not a guaranteed real optical reconstruction.
- Exact area and real-world map placement require valid CRS and affine transform metadata.
- The trained Model 2 is not enabled until it passes both held-out evaluation gates.

## License and data

This source repository excludes downloaded datasets, local SQLite databases, generated outputs, Python runtime folders, and model checkpoints. Review the source and dataset licenses before redistribution or commercial use.
