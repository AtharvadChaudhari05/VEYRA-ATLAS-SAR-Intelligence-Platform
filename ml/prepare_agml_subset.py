"""Turn compact public Sentinel-1/Sentinel-2 parquet shards into Pix2Pix folders."""

from __future__ import annotations

import argparse
import io
from pathlib import Path

import numpy as np
from PIL import Image


def image_bytes(value: object) -> bytes:
    if isinstance(value, dict) and value.get("bytes") is not None:
        return value["bytes"]
    if isinstance(value, (bytes, bytearray)):
        return bytes(value)
    raise TypeError(f"Unsupported RGB image representation: {type(value)!r}")


def sar_to_luminance(raw: bytes, shape: list[int], dtype: str) -> Image.Image:
    """Combine the real VV/VH SAR tile into a normalized single-band image."""
    array = np.frombuffer(raw, dtype=np.dtype(dtype)).reshape(tuple(shape)).astype(np.float32)
    if array.ndim == 3:
        array = np.nanmean(array, axis=0)  # source tiles are channel-first VV/VH
    if array.ndim != 2:
        raise ValueError(f"Expected 2D or 3D SAR tile, got {array.shape}")
    finite = array[np.isfinite(array)]
    if finite.size == 0:
        raise ValueError("SAR tile contains no finite values")
    low, high = np.percentile(finite, (1, 99))
    scaled = np.clip((array - low) / max(high - low, 1e-6), 0, 1)
    return Image.fromarray((scaled * 255).astype(np.uint8), mode="L")


def split_for(index: int) -> str:
    """Deterministic 80/10/10 split, with no random leakage."""
    return "test" if index % 10 == 0 else "validation" if index % 10 == 1 else "train"


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare the roughly 0.9 GB SAR/RGB training pilot.")
    parser.add_argument("--parquet", nargs="+", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=0, help="Optional pair cap; 0 exports all records")
    args = parser.parse_args()
    try:
        import pyarrow.parquet as pq
    except ImportError as exc:
        raise SystemExit("Install pyarrow first: pip install -r requirements-training.txt") from exc

    written = 0
    columns = ["rgb", "sar_bands", "sar_bands_shape", "sar_bands_dtype"]
    for shard in args.parquet:
        for batch in pq.ParquetFile(shard).iter_batches(columns=columns):
            for row in batch.to_pylist():
                if args.limit and written >= args.limit:
                    print(f"Prepared {written} aligned pairs in {args.output}")
                    return
                try:
                    sar = sar_to_luminance(row["sar_bands"], row["sar_bands_shape"], row["sar_bands_dtype"])
                    optical = Image.open(io.BytesIO(image_bytes(row["rgb"]))).convert("RGB")
                    stem, split = f"pair_{written:05d}", split_for(written)
                    sar_path = args.output / split / "sar" / f"{stem}.png"
                    optical_path = args.output / split / "optical" / f"{stem}.png"
                    sar_path.parent.mkdir(parents=True, exist_ok=True)
                    optical_path.parent.mkdir(parents=True, exist_ok=True)
                    sar.save(sar_path, optimize=True)
                    optical.save(optical_path, optimize=True)
                    written += 1
                except Exception as exc:
                    print(f"Skipping malformed record {written}: {exc}")
    print(f"Prepared {written} aligned pairs in {args.output}")


if __name__ == "__main__":
    main()
