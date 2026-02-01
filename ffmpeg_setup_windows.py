import os
import sys

if getattr(sys, 'frozen', False):
    base_path = sys._MEIPASS
else:
    base_path = os.path.dirname(__file__)

ffmpeg_path = os.path.join(base_path, "Tools", "FFMPEG", "ffmpeg-8.0.1-essentials_build", "bin")
os.environ["PATH"] = ffmpeg_path + os.pathsep + os.environ["PATH"]
