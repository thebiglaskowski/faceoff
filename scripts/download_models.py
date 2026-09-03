"""Download FaceOff model weights from their upstream sources.

The weight files are intentionally NOT stored in this git repo (GitHub
bandwidth). Run this once after cloning::

    python scripts/download_models.py

What it fetches (all from upstream, nothing from this repo):

- ``inswapper_128.onnx`` (~530 MB) -> project root
- ``buffalo_l`` face-analysis bundle (~330 MB) -> ``models/buffalo_l/``

GFPGAN / CodeFormer / Real-ESRGAN weights are handled separately: those
libraries self-download their own weights on first use, so this script
does not fetch them.
"""

from __future__ import annotations

import argparse
import sys
import urllib.request
import zipfile
from pathlib import Path

# ---------------------------------------------------------------------------
# Upstream sources (public releases, not this repo).
# ---------------------------------------------------------------------------

INSWAPPER_URL = (
    "https://huggingface.co/thebiglaskowski/inswapper_128.onnx"
    "/resolve/main/inswapper_128.onnx"
)
INSWAPPER_SHA256 = None  # Fill in with the published checksum if desired.
# Sanity floor: the real file is ~530 MB; anything far smaller is an
# error page, not a model.
INSWAPPER_MIN_BYTES = 400 * 1024 * 1024

BUFFALO_L_URL = (
    "https://github.com/deepinsight/insightface" "/releases/download/v0.7/buffalo_l.zip"
)
BUFFALO_L_MIN_BYTES = 200 * 1024 * 1024

CHUNK_SIZE = 4 * 1024 * 1024


def _project_root() -> Path:
    return Path(__file__).resolve().parent.parent


def _download(url: str, dest: Path, min_bytes: int) -> None:
    """Stream *url* to *dest* with a progress line; validate the size."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    req = urllib.request.Request(url, headers={"User-Agent": "FaceOff-model-setup"})
    with urllib.request.urlopen(req) as resp, open(tmp, "wb") as fh:  # noqa: S310
        total = int(resp.headers.get("Content-Length", 0) or 0)
        done = 0
        while True:
            chunk = resp.read(CHUNK_SIZE)
            if not chunk:
                break
            fh.write(chunk)
            done += len(chunk)
            if total:
                print(
                    f"\r  {dest.name}: {done / 1024 / 1024:.0f} / "
                    f"{total / 1024 / 1024:.0f} MB",
                    end="",
                    flush=True,
                )
        print()
    size = tmp.stat().st_size
    if size < min_bytes:
        tmp.unlink(missing_ok=True)
        raise ValueError(
            f"Downloaded {dest.name} is only {size} bytes "
            f"(expected at least {min_bytes}); the URL probably "
            "returned an error page instead of the model."
        )
    tmp.replace(dest)
    print(f"  saved {dest} ({size / 1024 / 1024:.0f} MB)")


def fetch_inswapper(root: Path, url: str, force: bool = False) -> Path:
    """Download inswapper_128.onnx into the project root (unless present)."""
    dest = root / "inswapper_128.onnx"
    if dest.exists() and not force:
        print(f"  {dest.name} already present, skipping.")
        return dest
    print(f"Downloading inswapper_128.onnx from:\n  {url}")
    _download(url, dest, INSWAPPER_MIN_BYTES)
    return dest


def fetch_buffalo_l(root: Path, url: str, force: bool = False) -> Path:
    """Download + unzip the buffalo_l bundle into models/buffalo_l/."""
    dest_dir = root / "models" / "buffalo_l"
    marker = dest_dir / "det_10g.onnx"
    if marker.exists() and not force:
        print("  models/buffalo_l already present, skipping.")
        return dest_dir
    archive = dest_dir / "buffalo_l.zip"
    print(f"Downloading buffalo_l from:\n  {url}")
    _download(url, archive, BUFFALO_L_MIN_BYTES)
    print("  extracting...")
    with zipfile.ZipFile(archive) as zf:
        zf.extractall(dest_dir)
    # The zip contains a top-level buffalo_l/ folder; flatten if needed.
    nested = dest_dir / "buffalo_l"
    if nested.is_dir():
        for item in nested.iterdir():
            item.rename(dest_dir / item.name)
        nested.rmdir()
    archive.unlink(missing_ok=True)
    if not marker.exists():
        raise ValueError(
            f"Extraction finished but {marker} is missing; "
            "the archive layout may have changed."
        )
    print(f"  saved {dest_dir}/ ({len(list(dest_dir.glob('*.onnx')))} .onnx files)")
    return dest_dir


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Download FaceOff model weights from upstream sources."
    )
    parser.add_argument(
        "--models",
        choices=["all", "inswapper", "buffalo_l"],
        default="all",
        help="Which weights to fetch (default: all).",
    )
    parser.add_argument(
        "--force", action="store_true", help="Re-download even if files exist."
    )
    parser.add_argument(
        "--inswapper-url",
        default=INSWAPPER_URL,
        help="Override the inswapper download URL.",
    )
    parser.add_argument(
        "--buffalo-url",
        default=BUFFALO_L_URL,
        help="Override the buffalo_l download URL.",
    )
    args = parser.parse_args(argv)
    root = _project_root()

    try:
        if args.models in ("all", "inswapper"):
            fetch_inswapper(root, args.inswapper_url, force=args.force)
        if args.models in ("all", "buffalo_l"):
            fetch_buffalo_l(root, args.buffalo_url, force=args.force)
    except Exception as exc:  # noqa: BLE001 - report cleanly for CLI use
        print(f"\nERROR: {exc}", file=sys.stderr)
        print(
            "\nIf the download URL is stale, grab the files manually:\n"
            "  inswapper_128.onnx -> project root\n"
            "    https://huggingface.co/thebiglaskowski/inswapper_128.onnx\n"
            "  buffalo_l.zip -> extract into models/buffalo_l/\n"
            "    https://github.com/deepinsight/insightface/releases/download/v0.7/buffalo_l.zip",
            file=sys.stderr,
        )
        return 1
    print("\nAll requested models are in place.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
