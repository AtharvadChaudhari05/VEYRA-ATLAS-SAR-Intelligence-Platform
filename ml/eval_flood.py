"""Evaluate trained and deterministic flood baselines on official splits.

Writes machine-readable results. It never invents a metric if a checkpoint or
dataset is unavailable.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from ml.flood_data import NormalizationStats, Sen1Floods11HandDataset
from ml.flood_metrics import binary_metrics, mc_predict, metrics_from_counts, otsu_probability
from ml.flood_model import FloodUNet


def _add_counts(total: list[int], result: object) -> None:
    for index, key in enumerate(("tp", "fp", "fn", "tn")):
        total[index] += int(getattr(result, key))


def _ece_bins(probability: np.ndarray, labels: np.ndarray, count: np.ndarray, confidence: np.ndarray, truth: np.ndarray) -> None:
    valid = labels >= 0; values = probability[valid]; targets = labels[valid]
    bins = np.minimum((values * len(count)).astype(int), len(count) - 1)
    for index in range(len(count)):
        selected = bins == index
        if selected.any():
            count[index] += int(selected.sum()); confidence[index] += float(values[selected].sum()); truth[index] += float(targets[selected].sum())


def _finalize(counts: list[int], ece_count: np.ndarray | None = None, ece_confidence: np.ndarray | None = None, ece_truth: np.ndarray | None = None) -> dict[str, float | int]:
    result = metrics_from_counts(*counts).to_dict()
    if ece_count is not None and ece_confidence is not None and ece_truth is not None:
        total = max(int(ece_count.sum()), 1); ece = 0.0
        for count, confidence, truth in zip(ece_count, ece_confidence, ece_truth):
            if count: ece += (count / total) * abs(confidence / count - truth / count)
        result["ece"] = float(ece)
    return result


def deterministic_probability(features: np.ndarray) -> np.ndarray:
    """Explicit reproduction of the production v0 dark/smooth rule for comparison."""
    vh = features[:, 1]
    ratio = features[:, 2]
    return np.clip(.72 * (1.0 - vh) + .28 * (1.0 - np.abs(ratio - .5) * 2), 0, 1)


def main() -> None:
    parser = argparse.ArgumentParser(); parser.add_argument("--data", required=True); parser.add_argument("--checkpoint", required=True); parser.add_argument("--output", default="artifacts/flood_sen1floods11/evaluation.json"); parser.add_argument("--batch-size", type=int, default=4); parser.add_argument("--mc-passes", type=int, default=12); args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu"); checkpoint = torch.load(args.checkpoint, map_location=device)
    stats = NormalizationStats(tuple(checkpoint["normalization"]["low"]), tuple(checkpoint["normalization"]["high"]))
    model = FloodUNet(base=int(checkpoint.get("base_channels", 32))).to(device); model.load_state_dict(checkpoint["state_dict"])
    results: dict[str, object] = {"checkpoint": str(args.checkpoint), "status": "measured", "splits": {}}
    for split in ("test", "bolivia"):
        dataset = Sen1Floods11HandDataset(args.data, split, stats=stats, use_lee_filter=bool(checkpoint.get("use_lee_filter", False)))
        loader = DataLoader(dataset, args.batch_size)
        learned_counts, deterministic_counts, otsu_counts = [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]
        ece_count, ece_confidence, ece_truth = np.zeros(15, dtype=np.int64), np.zeros(15), np.zeros(15)
        for features, labels, _ in loader:
            probability, _variance = mc_predict(model, features.to(device), args.mc_passes); prediction = probability.cpu().numpy(); target = labels.numpy(); source = features.numpy()
            _add_counts(learned_counts, binary_metrics(prediction, target)); _ece_bins(prediction, target, ece_count, ece_confidence, ece_truth)
            _add_counts(deterministic_counts, binary_metrics(deterministic_probability(source), target))
            _add_counts(otsu_counts, binary_metrics(np.stack([otsu_probability(item[1]) for item in source]), target))
        results["splits"][split] = {"learned": _finalize(learned_counts, ece_count, ece_confidence, ece_truth), "deterministic_v0_proxy": _finalize(deterministic_counts), "otsu_vh": _finalize(otsu_counts)}
    output = Path(args.output); output.parent.mkdir(parents=True, exist_ok=True); output.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(json.dumps(results, indent=2))


if __name__ == "__main__": main()
