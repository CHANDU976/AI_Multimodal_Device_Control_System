"""
ffmpeg_setup_linux.py
Prepends the bundled FFmpeg bin directory to PATH on Linux and sets
executable permissions on the binaries.

Expected layout:
    MyProject/
    └── Tools/FFMPEG/linux/bin/
        ├── ffmpeg
        └── ffprobe

Download from: https://johnvansickle.com/ffmpeg/
(ffmpeg-release-amd64-static.tar.xz — extract ffmpeg + ffprobe into Tools/FFMPEG/linux/bin/)
"""

import os
import sys
import warnings


def setup_ffmpeg() -> str | None:
    """
    Chmod the bundled Linux FFmpeg binaries and add their directory to PATH.

    Returns the resolved bin path, or None if the folder was not found.
    """
    if getattr(sys, "frozen", False):
        base_path = sys._MEIPASS          # PyInstaller temp extraction dir
    else:
        base_path = os.path.dirname(os.path.abspath(__file__))

    base_path = os.path.dirname(base_path)    # utils/ → MyProject/

    ffmpeg_bin = os.path.join(base_path, "Tools", "FFMPEG", "linux", "bin")

    if not os.path.isdir(ffmpeg_bin):
        warnings.warn(
            f"[ffmpeg_setup_linux] Bin directory not found: '{ffmpeg_bin}'. "
            "Download from https://johnvansickle.com/ffmpeg/ and extract "
            "ffmpeg + ffprobe into Tools/FFMPEG/linux/bin/",
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
                        f"[ffmpeg_setup_linux] Cannot chmod '{binary_path}': {exc}",
                        RuntimeWarning,
                        stacklevel=2,
                    )
        else:
            warnings.warn(
                f"[ffmpeg_setup_linux] Binary not found: '{binary_path}'.",
                RuntimeWarning,
                stacklevel=2,
            )

    os.environ["PATH"] = ffmpeg_bin + os.pathsep + os.environ.get("PATH", "")
    return ffmpeg_bin
