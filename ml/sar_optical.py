from __future__ import annotations

from pathlib import Path
from typing import Iterable

import torch
from PIL import Image
from torch import Tensor, nn
from torch.utils.data import Dataset


IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".webp"}


class Down(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, *, normalize: bool = True) -> None:
        super().__init__()
        layers: list[nn.Module] = [nn.Conv2d(in_channels, out_channels, 4, 2, 1, bias=not normalize)]
        if normalize:
            layers.append(nn.BatchNorm2d(out_channels))
        layers.append(nn.LeakyReLU(0.2, inplace=True))
        self.block = nn.Sequential(*layers)

    def forward(self, value: Tensor) -> Tensor:
        return self.block(value)


class Up(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, *, dropout: bool = False) -> None:
        super().__init__()
        layers: list[nn.Module] = [
            nn.ConvTranspose2d(in_channels, out_channels, 4, 2, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        ]
        if dropout:
            layers.append(nn.Dropout(0.5))
        self.block = nn.Sequential(*layers)

    def forward(self, value: Tensor) -> Tensor:
        return self.block(value)


class UNetGenerator(nn.Module):
    """256px Pix2Pix generator: one SAR band -> RGB optical-like output."""

    def __init__(self) -> None:
        super().__init__()
        self.d1 = Down(1, 64, normalize=False)
        self.d2 = Down(64, 128)
        self.d3 = Down(128, 256)
        self.d4 = Down(256, 512)
        self.d5 = Down(512, 512)
        self.d6 = Down(512, 512)
        self.d7 = Down(512, 512)
        self.d8 = nn.Sequential(nn.Conv2d(512, 512, 4, 2, 1), nn.ReLU(inplace=True))
        self.u1 = Up(512, 512, dropout=True)
        self.u2 = Up(1024, 512, dropout=True)
        self.u3 = Up(1024, 512, dropout=True)
        self.u4 = Up(1024, 512)
        self.u5 = Up(1024, 256)
        self.u6 = Up(512, 128)
        self.u7 = Up(256, 64)
        self.out = nn.Sequential(nn.ConvTranspose2d(128, 3, 4, 2, 1), nn.Tanh())

    def forward(self, source: Tensor) -> Tensor:
        d1 = self.d1(source); d2 = self.d2(d1); d3 = self.d3(d2); d4 = self.d4(d3)
        d5 = self.d5(d4); d6 = self.d6(d5); d7 = self.d7(d6); d8 = self.d8(d7)
        u1 = self.u1(d8); u2 = self.u2(torch.cat((u1, d7), 1)); u3 = self.u3(torch.cat((u2, d6), 1))
        u4 = self.u4(torch.cat((u3, d5), 1)); u5 = self.u5(torch.cat((u4, d4), 1)); u6 = self.u6(torch.cat((u5, d3), 1))
        u7 = self.u7(torch.cat((u6, d2), 1)); return self.out(torch.cat((u7, d1), 1))


class PatchDiscriminator(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        def block(a: int, b: int, norm: bool = True) -> list[nn.Module]:
            result: list[nn.Module] = [nn.Conv2d(a, b, 4, 2, 1, bias=not norm)]
            if norm: result.append(nn.BatchNorm2d(b))
            result.append(nn.LeakyReLU(0.2, inplace=True)); return result
        self.net = nn.Sequential(*block(4, 64, False), *block(64, 128), *block(128, 256), nn.Conv2d(256, 1, 4, 1, 1))

    def forward(self, sar: Tensor, optical: Tensor) -> Tensor:
        return self.net(torch.cat((sar, optical), dim=1))


def _unit(image: Image.Image, size: int, mode: str) -> Tensor:
    image = image.convert(mode).resize((size, size), Image.Resampling.BICUBIC)
    raw = torch.from_numpy(__import__("numpy").asarray(image).copy()).float() / 255.0
    if raw.ndim == 2: raw = raw.unsqueeze(0)
    else: raw = raw.permute(2, 0, 1)
    return raw * 2.0 - 1.0


class PairedSarOpticalDataset(Dataset[tuple[Tensor, Tensor]]):
    """Pairs root/sar and root/optical recursively by matching relative paths/stems."""
    def __init__(self, root: str | Path, split: str, image_size: int = 256) -> None:
        root = Path(root); self.image_size = image_size
        sar_root, optical_root = root / split / "sar", root / split / "optical"
        if not sar_root.exists(): sar_root, optical_root = root / "sar", root / "optical"
        if not sar_root.exists() or not optical_root.exists():
            raise FileNotFoundError("Expected dataset/<split>/sar + optical, or dataset/sar + optical.")
        optical = {path.relative_to(optical_root).with_suffix("").as_posix(): path for path in optical_root.rglob("*") if path.suffix.lower() in IMAGE_EXTENSIONS}
        self.pairs = [(path, optical[path.relative_to(sar_root).with_suffix("").as_posix()]) for path in sar_root.rglob("*") if path.suffix.lower() in IMAGE_EXTENSIONS and path.relative_to(sar_root).with_suffix("").as_posix() in optical]
        if not self.pairs: raise ValueError(f"No matching pairs in {sar_root} and {optical_root}.")

    def __len__(self) -> int: return len(self.pairs)
    def __getitem__(self, index: int) -> tuple[Tensor, Tensor]:
        sar, optical = self.pairs[index]
        with Image.open(sar) as source, Image.open(optical) as target:
            return _unit(source, self.image_size, "L"), _unit(target, self.image_size, "RGB")


def ssim(predicted: Tensor, target: Tensor) -> Tensor:
    """Differentiable global SSIM over RGB tensors in [-1, 1]."""
    x, y = (predicted + 1) / 2, (target + 1) / 2
    mu_x, mu_y = x.mean((2, 3), keepdim=True), y.mean((2, 3), keepdim=True)
    var_x = ((x - mu_x) ** 2).mean((2, 3), keepdim=True); var_y = ((y - mu_y) ** 2).mean((2, 3), keepdim=True)
    cov = ((x - mu_x) * (y - mu_y)).mean((2, 3), keepdim=True); c1, c2 = .01 ** 2, .03 ** 2
    return (((2 * mu_x * mu_y + c1) * (2 * cov + c2)) / ((mu_x**2 + mu_y**2 + c1) * (var_x + var_y + c2))).mean()


def save_checkpoint(path: Path, generator: nn.Module, epoch: int, val_ssim: float) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"architecture": "pix2pix-unet-256", "epoch": epoch, "val_ssim": val_ssim, "generator": generator.state_dict()}, path)
