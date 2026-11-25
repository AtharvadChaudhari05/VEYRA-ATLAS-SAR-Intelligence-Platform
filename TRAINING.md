# Custom SAR-to-optical training

This is the route for a defensible accuracy target. It trains a Pix2Pix U-Net on **aligned, paired** SAR and optical images. Do not use unrelated photographs as training pairs.

## Compact real-data pilot

Use two public 445 MB shards (about **0.89 GB** total) from the CC BY 4.0 [Project-AgML Sentinel SAR/RGB dataset](https://huggingface.co/datasets/Project-AgML/sentinel_SAR_NDVI). They contain co-registered Sentinel-1 SAR and Sentinel-2 RGB imagery. Download `train-00000-of-00010.parquet` and `train-00001-of-00010.parquet` into `E:\VeyraAtlas\source`, then prepare the folder layout:

```powershell
python -m ml.prepare_agml_subset --parquet E:\VeyraAtlas\source\train-00000-of-00010.parquet E:\VeyraAtlas\source\train-00001-of-00010.parquet --output E:\VeyraAtlas\pilot
```

The split is deterministic 80/10/10. This pilot validates data loading, training, and the quality gate, but cannot honestly guarantee SSIM 0.85. A final release must pass evaluation on geographically held-out representative scenes.

## Required dataset layout

```text
dataset/
  train/sar/<matching relative paths>.png
  train/optical/<matching relative paths>.png
  validation/sar/<matching relative paths>.png
  validation/optical/<matching relative paths>.png
  test/sar/<matching relative paths>.png
  test/optical/<matching relative paths>.png
```

SAR input is converted to one band and both images are resized to 256×256. Keep test scenes geographically distinct from training scenes.

## Install and train

```powershell
py -3.11 -m venv .venv-train
.venv-train\Scripts\Activate.ps1
pip install -r requirements-training.txt
python -m ml.train --data C:\datasets\sen1-2 --epochs 200 --output checkpoints\sar2optical --target-ssim 0.85
python -m ml.evaluate --data C:\datasets\sen1-2 --checkpoint checkpoints\sar2optical\best.pt --target-ssim 0.85
```

Training stops with a non-zero exit code when validation SSIM does not reach the requested target. Release requires the independent `test` evaluation to pass too.

## Phase 1: real flood model (Sen1Floods11)

The flood model is **not yet trained** in this repository. Download the public, official hand-labelled data (S1Hand + LabelHand) with a Google Cloud CLI installation, then run this on a GPU machine or Colab:

```powershell
python scripts/download_sen1floods11.py --output data/sen1floods11 --resume
pip install -r requirements-training.txt
python -m ml.train_flood --config configs/flood_sen1floods11.yaml
python -m ml.eval_flood --data data/sen1floods11 --checkpoint artifacts/flood_sen1floods11/best.pt
python -m ml.export_flood_onnx --checkpoint artifacts/flood_sen1floods11/best.pt --output artifacts/flood_sen1floods11/model.onnx
```

The evaluator writes test and Bolivia geographic hold-out results only after a real run. It reports IoU, F1, precision, recall, pixel accuracy, calibration ECE, and comparisons with deterministic and Otsu baselines.

`configs/flood_sen1floods11_full_cpu.yaml` uses all 252 official train chips and 89 validation chips. It is a genuine full-data configuration, but a CPU-only machine will take substantially longer than a CUDA GPU or Colab runtime. A model is promoted only when the held-out test metrics beat the deterministic baseline.
