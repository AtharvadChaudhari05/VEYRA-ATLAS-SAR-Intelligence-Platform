from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader

from ml.sar_optical import PairedSarOpticalDataset, PatchDiscriminator, UNetGenerator, save_checkpoint, ssim


def evaluate(generator: UNetGenerator, loader: DataLoader, device: torch.device) -> float:
    generator.eval(); scores: list[float] = []
    with torch.inference_mode():
        for sar, optical in loader: scores.append(float(ssim(generator(sar.to(device)), optical.to(device))))
    return sum(scores) / max(len(scores), 1)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train paired SAR -> optical Pix2Pix.")
    parser.add_argument("--data", required=True); parser.add_argument("--epochs", type=int, default=200); parser.add_argument("--batch-size", type=int, default=8); parser.add_argument("--workers", type=int, default=2); parser.add_argument("--image-size", type=int, default=256); parser.add_argument("--output", default="checkpoints/sar2optical"); parser.add_argument("--lambda-l1", type=float, default=100.0); parser.add_argument("--target-ssim", type=float, default=.85)
    args = parser.parse_args(); device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train = PairedSarOpticalDataset(args.data, "train", args.image_size); validation = PairedSarOpticalDataset(args.data, "validation", args.image_size)
    train_loader = DataLoader(train, args.batch_size, shuffle=True, num_workers=args.workers, pin_memory=device.type == "cuda"); val_loader = DataLoader(validation, args.batch_size, num_workers=args.workers)
    generator, discriminator = UNetGenerator().to(device), PatchDiscriminator().to(device); bce, l1 = nn.BCEWithLogitsLoss(), nn.L1Loss(); gen_opt = torch.optim.Adam(generator.parameters(), lr=.0002, betas=(.5,.999)); disc_opt = torch.optim.Adam(discriminator.parameters(), lr=.0002, betas=(.5,.999)); best = -1.0; output = Path(args.output)
    for epoch in range(1, args.epochs + 1):
        generator.train(); discriminator.train()
        for sar, optical in train_loader:
            sar, optical = sar.to(device), optical.to(device); fake = generator(sar)
            disc_opt.zero_grad(); real_loss = bce(discriminator(sar, optical), torch.ones_like(discriminator(sar, optical))); fake_loss = bce(discriminator(sar, fake.detach()), torch.zeros_like(discriminator(sar, fake.detach()))); (real_loss + fake_loss).mul(.5).backward(); disc_opt.step()
            gen_opt.zero_grad(); fake = generator(sar); adversarial = bce(discriminator(sar, fake), torch.ones_like(discriminator(sar, fake))); (adversarial + args.lambda_l1 * l1(fake, optical)).backward(); gen_opt.step()
        score = evaluate(generator, val_loader, device); print(f"epoch={epoch} validation_ssim={score:.4f}")
        if score > best: best = score; save_checkpoint(output / "best.pt", generator, epoch, score)
    print(f"Best validation SSIM: {best:.4f}")
    if best < args.target_ssim: raise SystemExit(f"Target not met: {best:.4f} < {args.target_ssim:.4f}. Do not claim {args.target_ssim:.0%} accuracy.")
    print(f"Target met: validation SSIM >= {args.target_ssim:.2f}. Evaluate separately on the unseen test split before release.")


if __name__ == "__main__": main()
