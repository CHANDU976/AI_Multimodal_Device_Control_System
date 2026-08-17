# AI Multimodal Device Control System

## 📋 Project Overview

A Laptop/Desktop Control System Which used Speech, Hand gestures and User Context and preferences wisely Using AI Technilogies

## 👥 Team Members

- **Chandan B L** - Designer, Developer - [GitHub: @CHANDU976]
- **[Team Member 2 Name]** - [Role/Responsibilities] - [GitHub: @username]
- **[Team Member 3 Name]** - [Role/Responsibilities] - [GitHub: @username]
- **[Team Member 4 Name]** - [Role/Responsibilities] - [GitHub: @username]

## 🎯 Project Goals

[List the main objectives and goals of this project]

- Goal 1: [Description]
- Goal 2: [Description]
- Goal 3: [Description]

## ✨ Features

[List the key features of your system]

- **Feature 1**: [Description]
- **Feature 2**: [Description]
- **Feature 3**: [Description]
- **Feature 4**: [Description]

## 🏗️ System Architecture

[Provide an overview of the system architecture, including major components and how they interact]

### Components

- **Component 1**: [Description]
- **Component 2**: [Description]
- **Component 3**: [Description]

## 🛠️ Technologies Used

[List the main technologies, frameworks, and tools used in the project]

- **Programming Languages**: [e.g., Python, JavaScript]
- **AI/ML Frameworks**: [e.g., TensorFlow, PyTorch, OpenCV]
- **Multimodal Processing**: [e.g., Speech Recognition, Computer Vision, NLP]
- **Hardware Integration**: [e.g., Raspberry Pi, Arduino, Sensors]
- **Other Tools**: [e.g., Docker, Git, etc.]

## 📦 Installation

### ⚠️ Python 3.8 – 3.11 only

**Not 3.12 or 3.13.** DeepFilterNet's native library (`DeepFilterLib`) publishes
wheels for `cp38`–`cp311` only, and DeepFilterNet pins `numpy<2`, which the whole
stack has to respect. On a newer interpreter the denoiser silently degrades to
pass-through — set `DENOISE["required"] = True` in `speech_module/config.py` to
turn that into a startup error instead.

`python3.11` is what you want, even if a newer Python is your system default.

### Platform support

| | Windows x64 | Linux x86_64 | macOS Apple Silicon | macOS Intel |
|---|---|---|---|---|
| DeepFilterNet3 | ✅ | ✅ | ✅ | ✅ |
| VOSK | `0.3.45` | `0.3.45` | `0.3.44` ¹ | `0.3.44` ¹ |
| torch / torchaudio | `2.13.0+cpu` | `2.13.0+cpu` | `2.13.0` ² | `2.2.2` ³ |
| PortAudio (mic) | bundled | **install it** ⁴ | bundled | bundled |
| Verified end-to-end | ✅ | — | — | — |

- ¹ vosk `0.3.45` shipped no macOS wheel; `0.3.44` is `universal2` and covers both Macs.
- ² PyTorch publishes no `+cpu` variant for macOS — the PyPI wheels are already native. The `2.13.0` wheel is built for macOS 14+; on macOS 13 or older, pin `torch==2.2.2` / `torchaudio==2.2.2` instead.
- ³ PyTorch **stopped building macOS x86_64 wheels after 2.2.2**. Intel Macs are capped there. Everything else in the stack still works.
- ⁴ `sounddevice` bundles PortAudio on Windows and macOS but not on Linux.

`requirements.txt` encodes all of this with environment markers, so the same
file installs the right thing on every platform.

---

### 1. Git LFS — before cloning

The VOSK and Whisper models are LFS objects. Clone without LFS and you get
text pointer files, and every model load fails.

```bash
# Windows  : winget install GitHub.GitLFS      (or: choco install git-lfs)
# Debian   : sudo apt install git-lfs
# Fedora   : sudo dnf install git-lfs
# macOS    : brew install git-lfs

git lfs install
git clone https://github.com/CHANDU976/AI-Multimodal-Device-Control-System-.git
cd AI-Multimodal-Device-Control-System-
```

Already cloned without it? `git lfs install && git lfs pull`.

### 2. System audio library

Only Linux needs anything here:

```bash
# Debian / Ubuntu
sudo apt install libportaudio2 python3.11-venv

# Fedora
sudo dnf install portaudio

# Windows / macOS — nothing to do, the sounddevice wheel bundles PortAudio.
# (macOS: `brew install portaudio` only if sounddevice fails to find a device.)
```

### 3. Virtual environment + dependencies

**Windows** (PowerShell):

```powershell
py -3.11 -m venv .venv
.venv\Scripts\python.exe -m pip install -r requirements.txt
```

**Linux / macOS** (Apple Silicon and Intel alike):

```bash
python3.11 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
```

Always invoke the venv interpreter explicitly (`.venv\Scripts\python.exe` /
`.venv/bin/python`) rather than a bare `python`, or you will silently hit the
system 3.13 and lose the denoiser.

### 4. Speech models

VOSK and Whisper arrive with the clone via LFS. DeepFilterNet3 (~8 MB) is
fetched separately — it is not in the repo:

```bash
.venv/bin/python fetch_deepfilternet_model.py        # Windows: .venv\Scripts\python.exe
```

This populates `Models/DFN3/` with `config.ini` + `checkpoints/`. It depends on
nothing but the standard library, so it also runs on your system Python.

Expected layout:

```
Models/
  DFN3/        config.ini, checkpoints/model_120.ckpt.best
  VOSK/        vosk-model-small-{en-in,en-us,hi}-*
  WHISPER/     base.pt, tiny.pt
```

### 5. FFmpeg (not needed for the live pipeline)

Whisper's own docs list ffmpeg as required, and for its documented usage it is —
`whisper clip.mp3` on the CLI and `model.transcribe("clip.wav")` both go through
`whisper.load_audio()`, which shells out to the `ffmpeg` binary.

**This pipeline never takes that path.** `WhisperEngine.transcribe()` is handed a
numpy array of already-decoded mic PCM, and `log_mel_spectrogram()` only calls
`load_audio()` when its input is a *string path*. Verified: the full pipeline
transcribes successfully on a machine with no ffmpeg on PATH.

So install it when you want to transcribe **files** — evaluating accuracy against
recorded test clips, for instance. `utils/ffmpeg_setup.py` is platform-aware and
downloads a static build into `Tools/FFMPEG/` on first import (Windows, Linux,
and both macOS architectures):

```python
import utils.ffmpeg_setup      # auto-configures PATH
```

Or use a system install (`winget install ffmpeg` / `sudo apt install ffmpeg` /
`brew install ffmpeg`).

### 6. Verify

```bash
.venv/bin/python speech_module/denoiser.py      # loads DFN3, records 5 s, writes before/after wavs
.venv/bin/python speech_module/speech_manager.py  # full pipeline, speak into the mic
```

If `denoiser.py` logs `DFN3 loaded from …`, the denoiser is live. A
`DeepFilterNet3 unavailable: …` warning means you are on the wrong interpreter
or `Models/DFN3` is empty.

## 🚀 Usage

[Provide instructions on how to use the system]

```bash
# Example commands to run the system
# python main.py
# [Add your commands here]
```

## 📖 Documentation

[Link to additional documentation, API references, or user guides]

- [User Guide](docs/user-guide.md)
- [API Documentation](docs/api.md)
- [Developer Guide](docs/developer-guide.md)

## 🧪 Testing

[Explain how to run tests]

```bash
# Example:
# pytest tests/
# [Add your test commands]
```

## 🗓️ Project Timeline

[Outline major milestones and deadlines]

- **Week 1-2**: [Milestone description]
- **Week 3-4**: [Milestone description]
- **Week 5-6**: [Milestone description]
- **Week 7-8**: [Milestone description]

## 📊 Project Status

[Current status of the project]

- [ ] Project Setup
- [ ] Component 1 Implementation
- [ ] Component 2 Implementation
- [ ] Component 3 Implementation
- [ ] Testing & Debugging
- [ ] Documentation
- [ ] Final Presentation

## 🤝 Contributing

[Guidelines for team members on how to contribute]

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📝 License

[Specify the license for your project, e.g., MIT, Apache 2.0, or All Rights Reserved]

## 📧 Contact

For questions or feedback, please contact:

- [Team Member Name] - [email@example.com]
- Project Link: [https://github.com/CHANDU976/AI-Multimodal-Device-Control-System-](https://github.com/CHANDU976/AI-Multimodal-Device-Control-System-)

## 🙏 Acknowledgments

[Credit any resources, libraries, or people that helped with the project]

- [Resource/Person 1]
- [Resource/Person 2]
- [Resource/Person 3]

---

*This project is part of [Course Name/Organization] - [Year]*
