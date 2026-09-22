"""Repository locations and Git-LFS awareness for tests."""

from __future__ import annotations

from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
CENTRAL = ROOT / "central_module"
SPEECH = ROOT / "speech_module"
GESTURE = ROOT / "Mousegesture"
MODELS = ROOT / "Models"
NLU_DIR = MODELS / "NLU" / "joint-distilbert"

_LFS_HEADER = b"version https://git-lfs.github.com/spec/v1"


def is_lfs_pointer(path: Path) -> bool:
    """True when *path* is a Git-LFS pointer rather than the real content.

    CI checks out without LFS (the models are ~1 GB and LFS bandwidth is
    metered), so every weight file there is a ~130-byte text stub.
    """
    try:
        with open(path, "rb") as fh:
            return fh.read(len(_LFS_HEADER)) == _LFS_HEADER
    except OSError:
        return False


def require_real_file(path: Path) -> Path:
    """Skip the calling test unless *path* holds real (non-pointer) content."""
    if not path.exists():
        pytest.skip(f"{path.relative_to(ROOT)} is not present")
    if is_lfs_pointer(path):
        pytest.skip(f"{path.relative_to(ROOT)} is a Git-LFS pointer "
                    f"(run `git lfs pull` to fetch it)")
    return path
