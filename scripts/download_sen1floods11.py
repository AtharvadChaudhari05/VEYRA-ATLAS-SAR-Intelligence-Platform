"""Download the official Sen1Floods11 hand-labelled benchmark with resume support.

This script intentionally downloads only the S1Hand, LabelHand, and official
split CSVs needed for the supervised flood benchmark.  The source bucket is
public: gs://sen1floods11/v1.1.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import json
import shutil
import subprocess
import urllib.parse
import urllib.request
from pathlib import Path


PREFIXES = (
    "splits/flood_handlabeled",
    "data/flood_events/HandLabeled/S1Hand",
    "data/flood_events/HandLabeled/LabelHand",
)
BUCKET = "gs://sen1floods11/v1.1"


def command_for(source: str, destination: Path) -> list[str]:
    """Prefer gcloud; gsutil is an equally supported official alternative."""
    if shutil.which("gcloud"):
        return ["gcloud", "storage", "rsync", "--recursive", source, str(destination)]
    if shutil.which("gsutil"):
        return ["gsutil", "-m", "rsync", "-r", source, str(destination)]
    raise RuntimeError("Install Google Cloud CLI (gcloud) or gsutil before downloading Sen1Floods11.")


def _public_objects(prefix: str) -> list[dict[str, object]]:
    """List objects from the public bucket without requiring cloud credentials."""
    page_token = ""; objects: list[dict[str, object]] = []
    while True:
        query = urllib.parse.urlencode({"prefix": f"v1.1/{prefix}/", "pageToken": page_token})
        with urllib.request.urlopen(f"https://storage.googleapis.com/storage/v1/b/sen1floods11/o?{query}", timeout=60) as response:
            payload = json.load(response)
        objects.extend(payload.get("items", [])); page_token = str(payload.get("nextPageToken", ""))
        if not page_token: return objects


def _download_object(item: dict[str, object], output: Path, dry_run: bool) -> None:
        name = str(item["name"]); relative = Path(name).relative_to("v1.1")
        destination = output / relative; destination.parent.mkdir(parents=True, exist_ok=True)
        size = int(item.get("size", 0))
        if destination.exists() and destination.stat().st_size == size:
            return
        url = f"https://storage.googleapis.com/download/storage/v1/b/sen1floods11/o/{urllib.parse.quote(name, safe='')}?alt=media"
        if dry_run:
            print(f"GET {url} -> {destination}"); return
        offset = destination.stat().st_size if destination.exists() else 0
        request = urllib.request.Request(url, headers={"Range": f"bytes={offset}-"} if offset else {})
        print(f"Downloading {relative} ({offset}/{size} bytes)")
        try:
            with urllib.request.urlopen(request, timeout=120) as response:
                # Public GCS normally honours Range with 206; if it does not,
                # overwrite instead of corrupting a partial object by appending.
                mode = "ab" if offset and response.status == 206 else "wb"
                with destination.open(mode) as handle:
                    shutil.copyfileobj(response, handle, length=1024 * 1024)
        except Exception:
            # A later --resume run continues from the partial size.
            raise


def _http_sync(prefix: str, output: Path, dry_run: bool, workers: int) -> None:
    objects = _public_objects(prefix)
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = [executor.submit(_download_object, item, output, dry_run) for item in objects]
        for future in futures:
            future.result()


def main() -> None:
    parser = argparse.ArgumentParser(description="Download official Sen1Floods11 S1Hand + LabelHand data.")
    parser.add_argument("--output", type=Path, default=Path("data/sen1floods11"))
    parser.add_argument("--resume", action="store_true", help="Use rsync to skip files already downloaded.")
    parser.add_argument("--workers", type=int, default=6, help="Parallel HTTP downloads when no Google CLI is installed.")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    for prefix in PREFIXES:
        source = f"{BUCKET}/{prefix}"
        destination = args.output / prefix
        destination.parent.mkdir(parents=True, exist_ok=True)
        command = command_for(source, destination) if (shutil.which("gcloud") or shutil.which("gsutil")) else None
        if command is None:
            print("No Google Cloud CLI found; using public GCS HTTPS fallback.")
            _http_sync(prefix, args.output, args.dry_run, args.workers)
            continue
        if args.dry_run:
            print(" ".join(command))
            continue
        # Both rsync implementations are resumable: existing matching files are skipped.
        print(f"Syncing {source} -> {destination}")
        subprocess.run(command, check=True)
    print(f"Completed Sen1Floods11 sync at {args.output}")


if __name__ == "__main__":
    main()
