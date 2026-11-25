"""Train the Phase 1 flood U-Net on official Sen1Floods11 hand labels."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
import yaml
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader, Subset

from ml.flood_data import NormalizationStats, Sen1Floods11HandDataset
from ml.flood_metrics import binary_metrics, mc_predict, metrics_from_counts
from ml.flood_model import FloodUNet, masked_bce_dice


def set_seed(seed: int) -> None:
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed); torch.backends.cudnn.deterministic = True


def validate(model: FloodUNet, loader: DataLoader, device: torch.device) -> dict[str, float]:
    model.eval(); tp = fp = fn = tn = 0
    with torch.inference_mode():
        for features, target, _ in loader:
            result = binary_metrics(torch.sigmoid(model(features.to(device))).cpu().numpy(), target.numpy())
            tp += result.tp; fp += result.fp; fn += result.fn; tn += result.tn
    result = metrics_from_counts(tp, fp, fn, tn)
    return {key: float(value) for key, value in result.to_dict().items() if isinstance(value, float)}


def main() -> None:
    parser = argparse.ArgumentParser(); parser.add_argument("--config", default="configs/flood_sen1floods11.yaml"); args = parser.parse_args()
    config = yaml.safe_load(Path(args.config).read_text(encoding="utf-8")); set_seed(int(config["seed"]))
    if config.get("torch_num_threads"):
        torch.set_num_threads(int(config["torch_num_threads"]))
    device = torch.device("cuda" if config.get("device", "auto") == "auto" and torch.cuda.is_available() else "cpu")
    output = Path(config["output_dir"]); output.mkdir(parents=True, exist_ok=True); (output / "config.yaml").write_text(Path(args.config).read_text(encoding="utf-8"), encoding="utf-8")
    stats = NormalizationStats(); data_args = dict(stats=stats, use_lee_filter=bool(config.get("use_lee_filter", False)))
    train = Sen1Floods11HandDataset(config["data_root"], "train", augment=True, patch_size=config.get("train_patch_size"), **data_args); validation = Sen1Floods11HandDataset(config["data_root"], "val", **data_args)
    if config.get("max_train_samples"):
        train = Subset(train, range(min(int(config["max_train_samples"]), len(train))))
    if config.get("max_val_samples"):
        validation = Subset(validation, range(min(int(config["max_val_samples"]), len(validation))))
    train_loader = DataLoader(train, int(config["batch_size"]), shuffle=True, num_workers=int(config["num_workers"]), pin_memory=device.type == "cuda"); val_loader = DataLoader(validation, int(config["batch_size"]), num_workers=int(config["num_workers"]))
    base_channels = int(config.get("base_channels", 32)); model = FloodUNet(base=base_channels).to(device); optimizer = torch.optim.AdamW(model.parameters(), lr=float(config["learning_rate"]), weight_decay=1e-4); scaler = GradScaler(enabled=device.type == "cuda")
    best, remaining, start_epoch = -1.0, int(config["patience"]), 1; history: list[dict[str, float | int]] = []
    if config.get("resume_checkpoint"):
        checkpoint = torch.load(config["resume_checkpoint"], map_location=device); model.load_state_dict(checkpoint["state_dict"]); best = float(checkpoint.get("validation", {}).get("iou", -1.0)); start_epoch = int(checkpoint.get("epoch", 0)) + 1
        print(f"Resumed model weights from {config['resume_checkpoint']} at epoch {start_epoch - 1} with best IoU={best:.4f}", flush=True)
    for epoch in range(start_epoch, int(config["epochs"]) + 1):
        model.train(); losses = []
        for features, target, _ in train_loader:
            optimizer.zero_grad(set_to_none=True); features, target = features.to(device), target.to(device)
            with autocast(enabled=device.type == "cuda"):
                loss = masked_bce_dice(model(features), target)
            scaler.scale(loss).backward(); scaler.step(optimizer); scaler.update(); losses.append(float(loss.detach()))
        metrics = validate(model, val_loader, device); record = {"epoch": epoch, "train_loss": float(np.mean(losses)), **metrics}; history.append(record); print(json.dumps(record), flush=True)
        if metrics["iou"] > best:
            best, remaining = metrics["iou"], int(config["patience"])
            torch.save({"architecture": "FloodUNet", "base_channels": base_channels, "input_features": ["VV_dB", "VH_dB", "VV_minus_VH_dB"], "normalization": stats.to_dict(), "use_lee_filter": data_args["use_lee_filter"], "epoch": epoch, "validation": metrics, "state_dict": model.state_dict()}, output / "best.pt")
        else:
            remaining -= 1
            if remaining <= 0: break
    (output / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
    print(f"Training complete. Best validation IoU={best:.4f}; run ml.eval_flood before publishing any claim.")


if __name__ == "__main__": main()
