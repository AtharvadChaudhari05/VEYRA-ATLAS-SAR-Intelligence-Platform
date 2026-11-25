"""Apply the non-negotiable deployment gate to a measured flood evaluation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(); parser.add_argument("--evaluation", required=True); parser.add_argument("--output", required=True); args = parser.parse_args()
    evaluation = json.loads(Path(args.evaluation).read_text(encoding="utf-8")); splits = evaluation["splits"]
    comparisons = {}
    for split in ("test", "bolivia"):
        learned = float(splits[split]["learned"]["iou"]); baseline = float(splits[split]["deterministic_v0_proxy"]["iou"])
        comparisons[split] = {"learned_iou": learned, "baseline_iou": baseline, "beats_baseline": learned > baseline}
    eligible = all(item["beats_baseline"] for item in comparisons.values())
    result = {"status": "eligible_for_manual_backend_integration" if eligible else "rejected_keep_deterministic_baseline", "criterion": "Learned IoU must exceed the deterministic baseline on both official test and Bolivia hold-out.", "comparisons": comparisons}
    output = Path(args.output); output.parent.mkdir(parents=True, exist_ok=True); output.write_text(json.dumps(result, indent=2), encoding="utf-8"); print(json.dumps(result, indent=2))


if __name__ == "__main__": main()
