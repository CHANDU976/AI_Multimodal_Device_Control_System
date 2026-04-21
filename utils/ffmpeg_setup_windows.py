"""
ffmpeg_setup_windows.py
Prepends the bundled FFmpeg bin directory to PATH on Windows.

Expected layout:
    MyProject/
    └── Tools/FFMPEG/windows/bin/
        ├── ffmpeg.exe
        └── ffprobe.exe

Download from: https://gyan.dev/ffmpeg/builds/
(ffmpeg-release-essentials.zip — extract bin/ contents into Tools/FFMPEG/windows/bin/)
"""

import os
import sys
import warnings


def setup_ffmpeg() -> str | None:
    """
    Add the bundled Windows FFmpeg bin folder to PATH.

    Returns the resolved bin path, or None if the folder was not found.
    """
    if getattr(sys, "frozen", False):
        base_path = sys._MEIPASS          # PyInstaller temp extraction dir
    else:
        base_path = os.path.dirname(os.path.abspath(__file__))

    base_path = os.path.dirname(base_path)    # utils/ → MyProject/

    ffmpeg_bin = os.path.join(base_path, "Tools", "FFMPEG", "windows", "bin")

    if not os.path.isdir(ffmpeg_bin):
        warnings.warn(
            f"[ffmpeg_setup_windows] Bin directory not found: '{ffmpeg_bin}'. "
            "Download from https://gyan.dev/ffmpeg/builds/ and extract into "
            "Tools/FFMPEG/windows/bin/",
            RuntimeWarning,
            stacklevel=2,
        )
        return None

    os.environ["PATH"] = ffmpeg_bin + os.pathsep + os.environ.get("PATH", "")
    return ffmpeg_bin
