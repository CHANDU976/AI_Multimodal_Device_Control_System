# Make sure to set executable permissions on ffmpeg binary after extraction
# For example, you can run the following command in your setup script:
# os.chmod +x MyApp/ffmpeg/bin/ffmpeg


import os
import sys

def add_ffmpeg_to_path():
    
    if getattr(sys, 'frozen', False):
        base_path = sys._MEIPASS   # PyInstaller temp dir
    else:
        base_path = os.path.dirname(os.path.abspath(__file__))
    base_path = os.path.dirname(base_path)  # Move up one level to MyApp
    ffmpeg_bin = os.path.join(base_path, "Tools", "FFMPEG", "ffmpeg-8.0.1-essentials_build", "bin")
    os.chmod(os.path.join(ffmpeg_bin, "ffmpeg"), 0o755)

    os.environ["PATH"] = ffmpeg_bin + os.pathsep + os.environ.get("PATH", "")

add_ffmpeg_to_path()
