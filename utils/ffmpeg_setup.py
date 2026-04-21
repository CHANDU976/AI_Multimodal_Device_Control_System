"""
ffmpeg_setup.py
Platform-aware FFmpeg setup — the only file you need to import.

Usage
-----
    # Option A: explicit call
    from utils.ffmpeg_setup import setup_ffmpeg
    setup_ffmpeg()

    # Option B: auto-runs on import
    import utils.ffmpeg_setup

Supported platforms
-------------------
    win32   → ffmpeg_setup_windows.py   (gyan.dev build)
    linux   → ffmpeg_setup_linux.py     (johnvansickle.com static build)
    darwin  → ffmpeg_setup_macos.py     (evermeet.cx build, arm64 or x86_64)

Folder layout expected under MyProject/Tools/FFMPEG/
-----------------------------------------------------
    windows/bin/         ffmpeg.exe, ffprobe.exe
    linux/bin/           ffmpeg, ffprobe
    macos/arm64/bin/     ffmpeg, ffprobe   ← Apple Silicon
    macos/x86_64/bin/    ffmpeg, ffprobe   ← Intel
"""

import sys
import warnings


def setup_ffmpeg() -> str | None:
    """
    Detect the current platform, delegate to the appropriate
    platform-specific setup module, and return the resolved
    FFmpeg bin path (or None if setup failed).
    """
    platform = sys.platform

    if platform == "win32":
        from utils.ffmpeg_setup_windows import setup_ffmpeg as _setup

    elif platform == "linux":
        from utils.ffmpeg_setup_linux import setup_ffmpeg as _setup

    elif platform == "darwin":
        from utils.ffmpeg_setup_macos import setup_ffmpeg as _setup

    else:
        warnings.warn(
            f"[ffmpeg_setup] Unsupported platform '{platform}'. "
            "FFmpeg PATH not configured — audio features may fail.",
            RuntimeWarning,
            stacklevel=2,
        )
        return None

    return _setup()


# ── Auto-run when imported ─────────────────────────────────────────────────────
_ffmpeg_path = setup_ffmpeg()

if _ffmpeg_path:
    print(f"[ffmpeg_setup] FFmpeg ready ({sys.platform}): {_ffmpeg_path}")
else:
    print(f"[ffmpeg_setup] WARNING: FFmpeg setup failed on '{sys.platform}'. "
          "Check Tools/FFMPEG/ layout — see ffmpeg_setup.py docstring.")
