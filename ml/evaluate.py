from __future__ import annotations

import argparse
import torch
from torch.utils.data import DataLoader
from ml.sar_optical import PairedSarOpticalDataset, UNetGenerator, ssim

def main() -> None:
    p = argparse.ArgumentParser(description="Evaluate SAR -> optical checkpoint on an unseen test split."); p.add_argument("--data", required=True); p.add_argument("--checkpoint", required=True); p.add_argument("--batch-size", type=int, default=8); p.add_argument("--image-size", type=int, default=256); p.add_argument("--target-ssim", type=float, default=.85); a = p.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu"); dataset = PairedSarOpticalDataset(a.data, "test", a.image_size); loader = DataLoader(dataset, a.batch_size); generator = UNetGenerator().to(device); checkpoint = torch.load(a.checkpoint, map_location=device); generator.load_state_dict(checkpoint["generator"]); generator.eval(); scores=[]
    with torch.inference_mode():
        for sar, optical in loader: scores.append(float(ssim(generator(sar.to(device)), optical.to(device))))
    score=sum(scores)/max(len(scores),1); print(f"test_ssim={score:.4f}")
    if score < a.target_ssim: raise SystemExit(f"FAIL: SSIM {score:.4f} is below target {a.target_ssim:.4f}")
    print("PASS: target met on unseen test data")
if __name__ == "__main__": main()
