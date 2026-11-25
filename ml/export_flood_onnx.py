from __future__ import annotations

import argparse
from pathlib import Path

import torch

from ml.flood_model import FloodUNet


def main() -> None:
    parser = argparse.ArgumentParser(); parser.add_argument("--checkpoint", required=True); parser.add_argument("--output", required=True); parser.add_argument("--size", type=int, default=512); args = parser.parse_args()
    checkpoint = torch.load(args.checkpoint, map_location="cpu"); model = FloodUNet(base=int(checkpoint.get("base_channels", 32))); model.load_state_dict(checkpoint["state_dict"]); model.eval(); output = Path(args.output); output.parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(model, torch.zeros(1, 3, args.size, args.size), output, input_names=["sar_features"], output_names=["flood_logits"], dynamic_axes={"sar_features": {0: "batch", 2: "height", 3: "width"}, "flood_logits": {0: "batch", 1: "height", 2: "width"}}, opset_version=17)
    print(f"Exported {output}")


if __name__ == "__main__": main()
