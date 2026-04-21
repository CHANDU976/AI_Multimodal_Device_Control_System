"""
ffmpeg_setup_macos.py
Prepends the bundled FFmpeg bin directory to PATH on macOS and sets
executable permissions. Automatically detects Apple Silicon vs Intel.

Expected layout:
    MyProject/
    └── Tools/FFMPEG/macos/
        ├── arm64/bin/          ← Apple Silicon (M1/M2/M3/M4)
        │   ├── ffmpeg
        │   └── ffprobe
        └── x86_64/bin/         ← Intel
            ├── ffmpeg
            └── ffprobe

Download from: https://evermeet.cx/ffmpeg/
  - Apple Silicon : ffmpeg-<version>-arm64.zip  + ffprobe-<version>-arm64.zip
  - Intel         : ffmpeg-<version>.zip         + ffprobe-<version>.zip
Extract each binary into the corresponding bin/ folder above.
"""

import os
import platform
import sys
import warnings


def setup_ffmpeg() -> str | None:
    """
    Detect CPU architecture, chmod the appropriate bundled macOS FFmpeg
    binaries, and add their directory to PATH.

    Returns the resolved bin path, or None if setup failed.
    """
    if getattr(sys, "frozen", False):
        base_path = sys._MEIPASS          # PyInstaller temp extraction dir
    else:
        base_path = os.path.dirname(os.path.abspath(__file__))

    base_path = os.path.dirname(base_path)    # utils/ → MyProject/

    # ── Detect architecture ────────────────────────────────────────────────────
    machine = platform.machine().lower()      # 'arm64' on M-series, 'x86_64' on Intel
    if machine == "arm64":
        arch = "arm64"
    elif machine == "x86_64":
        arch = "x86_64"
    else:
        warnings.warn(
            f"[ffmpeg_setup_macos] Unrecognised Mac architecture '{machine}'. "
            "Falling back to x86_64 build — this may not work.",
            RuntimeWarning,
            stacklevel=2,
        )
        arch = "x86_64"

    ffmpeg_bin = os.path.join(base_path, "Tools", "FFMPEG", "macos", arch, "bin")

    if not os.path.isdir(ffmpeg_bin):
        warnings.warn(
            f"[ffmpeg_setup_macos] Bin directory not found: '{ffmpeg_bin}'. "
            f"Download the {arch} build from https://evermeet.cx/ffmpeg/ and "
            f"place ffmpeg + ffprobe in Tools/FFMPEG/macos/{arch}/bin/",
            RuntimeWarning,
            stacklevel=2,
        )
        return None

    # ── Ensure binaries are executable ────────────────────────────────────────
    for binary in ("ffmpeg", "ffprobe"):
        binary_path = os.path.join(ffmpeg_bin, binary)
        if os.path.isfile(binary_path):
            current_mode = os.stat(binary_path).st_mode
            if not (current_mode & 0o111):      # skip if already executable
                try:
                    os.chmod(binary_path, current_mode | 0o755)
                except PermissionError as exc:
                    warnings.warn(
                        f"[ffmpeg_setup_macos] Cannot chmod '{binary_path}': {exc}",
                        RuntimeWarning,
                        stacklevel=2,
                    )
        else:
            warnings.warn(
                f"[ffmpeg_setup_macos] Binary not found: '{binary_path}'.",
                RuntimeWarning,
                stacklevel=2,
            )

    os.environ["PATH"] = ffmpeg_bin + os.pathsep + os.environ.get("PATH", "")
    return ffmpeg_bin
