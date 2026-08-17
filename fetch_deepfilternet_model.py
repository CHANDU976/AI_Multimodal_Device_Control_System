"""
fetch_deepfilternet_model.py

Downloads a DeepFilterNet model straight from the upstream repo into
Models/DFN3 (or wherever you point --dest), so the speech pipeline runs fully
offline from our own copy — no ~/.cache lookup, no network call at startup.

Usage
-----
    python fetch_deepfilternet_model.py                  # DFN3 → Models/DFN3
    python fetch_deepfilternet_model.py --force          # re-download
    python fetch_deepfilternet_model.py --model DeepFilterNet2 --dest Models/DFN2

Deliberately depends on nothing but the standard library. The obvious
alternative — calling df.enhance.init_df() and copying whatever it cached —
requires the whole DeepFilterNet stack (and therefore Python ≤3.11 and
numpy<2) just to fetch a 8 MB zip. This runs on any interpreter, so you can
populate Models/ before setting up the runtime venv.

Layout produced (this is exactly what init_df(model_base_dir=...) expects):

    Models/DFN3/
      config.ini
      checkpoints/model_120.ckpt.best
"""

from __future__ import annotations

import argparse
import hashlib
import io
import shutil
import sys
import urllib.request
import zipfile
from pathlib import Path

MODEL_URL = "https://github.com/Rikorose/DeepFilterNet/raw/main/models/{model}.zip"

# Repo root — this file lives at the top level of the project.
PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_DEST = PROJECT_ROOT / "Models" / "DFN3"


def download(model: str) -> bytes:
    url = MODEL_URL.format(model=model)
    print(f"Downloading {url} …")
    request = urllib.request.Request(url, headers={"User-Agent": "python-urllib"})
    with urllib.request.urlopen(request) as response:
        data = response.read()
    print(f"  {len(data):,} bytes  (sha256 {hashlib.sha256(data).hexdigest()[:16]}…)")
    return data


def extract(data: bytes, model: str, dest: Path) -> None:
    """
    Unpack the archive into `dest`, stripping the leading `<model>/` directory
    so `dest` itself becomes the model_base_dir.
    """
    archive = zipfile.ZipFile(io.BytesIO(data))
    prefix = f"{model}/"
    dest.mkdir(parents=True, exist_ok=True)

    members = [m for m in archive.infolist() if m.filename.startswith(prefix)]
    if not members:
        raise RuntimeError(
            f"archive has no '{prefix}' directory — got {archive.namelist()[:5]}"
        )

    for member in members:
        relative = member.filename[len(prefix):]
        if not relative:
            continue
        # Guard against path traversal in the archive.
        target = (dest / relative).resolve()
        if not str(target).startswith(str(dest.resolve())):
            raise RuntimeError(f"archive entry escapes destination: {member.filename}")

        if member.is_dir():
            target.mkdir(parents=True, exist_ok=True)
            continue

        target.parent.mkdir(parents=True, exist_ok=True)
        with archive.open(member) as source, open(target, "wb") as out:
            shutil.copyfileobj(source, out)
        print(f"  -> {target.relative_to(PROJECT_ROOT)}  ({target.stat().st_size:,} bytes)")


def verify(dest: Path) -> None:
    config = dest / "config.ini"
    checkpoints = dest / "checkpoints"
    if not config.is_file():
        raise RuntimeError(f"missing {config}")
    if not checkpoints.is_dir() or not any(checkpoints.iterdir()):
        raise RuntimeError(f"missing or empty {checkpoints}")

    # Surface the rate the pipeline has to capture at — it must match
    # AUDIO["capture_rate"] in speech_module/config.py.
    for line in config.read_text(encoding="utf-8").splitlines():
        if line.strip().startswith("sr ="):
            print(f"\nModel sample rate: {line.split('=', 1)[1].strip()} Hz")
            break


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--model", default="DeepFilterNet3",
        help="Model to fetch: DeepFilterNet, DeepFilterNet2, DeepFilterNet3 (default).",
    )
    parser.add_argument(
        "--dest", type=Path, default=DEFAULT_DEST,
        help=f"Directory to populate. Default: {DEFAULT_DEST}",
    )
    parser.add_argument(
        "--force", action="store_true",
        help="Re-download even if the destination already looks populated.",
    )
    args = parser.parse_args()

    dest: Path = args.dest
    if (dest / "config.ini").is_file() and not args.force:
        print(f"{dest} is already populated — nothing to do (use --force to re-download).")
        verify(dest)
        return 0

    if dest.exists() and args.force:
        print(f"Clearing {dest} …")
        shutil.rmtree(dest)

    extract(download(args.model), args.model, dest)
    verify(dest)

    print(f"\nModel ready at: {dest}")
    print("speech_module/config.py already points DENOISE['model_dir'] here.")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as exc:      # noqa: BLE001 — this is a CLI, print don't traceback
        print(f"\nERROR: {exc}", file=sys.stderr)
        sys.exit(1)
