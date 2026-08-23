"""
calibrate.py
Routing-threshold calibration harness.

The ROUTING thresholds in config.py decide when a VOSK result is trusted as a
command and when the audio escalates to Whisper. This tool replaces guessed
thresholds with measured ones: run your own voice, mic and room through the
exact segment-decode path and look at where the confidence distributions
actually sit.

Usage
-----
    python calibrate.py clips/                 # analyse a folder of wav clips
    python calibrate.py clips/ --denoise       # DFN3 first, like the live pipeline
    python calibrate.py clips/ --record 10     # record 10 clips into the folder

Each clip is a mono wav at any sample rate. An optional sidecar with the same
stem (clip_01.txt next to clip_01.wav) holds the expected text; clips that
have one contribute to the threshold suggestions, clips without one are just
decoded and shown.

Every clip is decoded twice — free and grammar-constrained — so the report
also shows what the command grammar does to confidences and which utterances
it maps to [unk].
"""

from __future__ import annotations
import path_setup  # noqa: F401 — adds project root to sys.path

import argparse
import re
import sys
from pathlib import Path

import numpy as np
import soundfile as sf

from audio_utils import resample_int16
from config import AUDIO, ROUTING, command_grammar
from router import Router
from vosk_engine import VoskEngine, VoskResult

_WORD_RE = re.compile(r"[a-z']+")


def _normalize(text: str) -> list[str]:
    """Lowercase word list, punctuation stripped — for decode comparison."""
    return _WORD_RE.findall(text.lower())


def _load_clip(path: Path) -> np.ndarray:
    """Read a wav → mono int16 @ 16 kHz, whatever it was stored as."""
    data, rate = sf.read(path, dtype="int16", always_2d=True)
    mono = data.mean(axis=1).astype(np.int16) if data.shape[1] > 1 else data[:, 0]
    if rate != AUDIO["target_rate"]:
        mono = resample_int16(mono, rate, AUDIO["target_rate"])
    return mono


def _describe(result: VoskResult) -> str:
    unk = f"  unk {result.unk_count}/{len(result.words)}" if result.unk_count else ""
    return (f"conf {result.mean_confidence:.2f}/min {result.min_confidence:.2f}"
            f"{unk}  {result.text!r}")


def analyse(folder: Path, denoise: bool) -> None:
    clips = sorted(folder.glob("*.wav"))
    if not clips:
        sys.exit(f"No .wav files in {folder}")

    engine = VoskEngine(audio_queue=None)   # segment decoding only, no stream
    engine.load_model()
    grammar = command_grammar()
    router = Router()

    denoiser = None
    if denoise:
        from denoiser import DeepFilterDenoiser
        denoiser = DeepFilterDenoiser()
        denoiser.load()

    # Per-clip rows; word/mean confidences split by whether the decode was right.
    word_confs = {True: [], False: []}
    mean_confs = {True: [], False: []}

    # Two passes (all free, then all grammar) so the engine's cached segment
    # recognizer is built twice total instead of twice per clip.
    audio = {}
    for wav in clips:
        pcm = _load_clip(wav)
        if denoiser is not None:
            pcm48 = resample_int16(pcm, AUDIO["target_rate"], AUDIO["capture_rate"])
            pcm = resample_int16(denoiser.enhance_int16(pcm48),
                                 AUDIO["capture_rate"], AUDIO["target_rate"])
        audio[wav] = pcm
    free_results = {wav: engine.recognize_segment(pcm) for wav, pcm in audio.items()}
    grammar_results = {wav: engine.recognize_segment(pcm, grammar=grammar)
                       for wav, pcm in audio.items()}

    print(f"\n{len(clips)} clips  (denoise {'on' if denoiser else 'off'})\n")
    for wav in clips:
        free = free_results[wav]
        constrained = grammar_results[wav]
        decision = router.route(constrained)

        expected_file = wav.with_suffix(".txt")
        expected = expected_file.read_text(encoding="utf-8").strip() if expected_file.exists() else None

        print(f"── {wav.name}")
        if expected is not None:
            correct = _normalize(free.text) == _normalize(expected)
            print(f"   expected : {expected!r}")
            print(f"   free     : {_describe(free)}   {'OK' if correct else 'WRONG'}")
            word_confs[correct] += [float(w["conf"]) for w in free.words if "conf" in w]
            mean_confs[correct].append(free.mean_confidence)
        else:
            print(f"   free     : {_describe(free)}")
        print(f"   grammar  : {_describe(constrained)}")
        print(f"   route    : {decision.destination}  ({decision.reason})")

    # ── Threshold suggestions ──────────────────────────────────────────────────
    print("\n──── summary " + "─" * 47)
    scored = len(mean_confs[True]) + len(mean_confs[False])
    if not scored:
        print("No .txt sidecars — add expected text next to clips to get "
              "threshold suggestions.")
        return
    print(f"clips with expected text: {scored}  "
          f"(correct {len(mean_confs[True])}, wrong {len(mean_confs[False])})")

    if word_confs[True]:
        w = np.array(word_confs[True])
        m = np.array(mean_confs[True])
        print(f"correct decodes  word conf: min {w.min():.2f}  p5 {np.percentile(w, 5):.2f}"
              f"   mean conf: min {m.min():.2f}  p5 {np.percentile(m, 5):.2f}")
        # 5th percentile keeps ~95% of correct decodes on the fast path.
        print(f"suggested  min_word_confidence ≈ {np.floor(np.percentile(w, 5) * 20) / 20:.2f}"
              f"   (config: {ROUTING['min_word_confidence']})")
        print(f"suggested  min_mean_confidence ≈ {np.floor(np.percentile(m, 5) * 20) / 20:.2f}"
              f"   (config: {ROUTING['min_mean_confidence']})")
    if word_confs[False]:
        w = np.array(word_confs[False])
        print(f"wrong decodes    word conf: max {w.max():.2f}  "
              f"median {np.median(w):.2f}  — thresholds must sit above the bulk of these")


def record(folder: Path, count: int) -> None:
    """Capture clips through the real mic → VAD path and save wav + txt."""
    from audio_stream import AudioStream
    from vad import create_vad

    folder.mkdir(parents=True, exist_ok=True)
    existing = len(list(folder.glob("*.wav")))
    vad = create_vad()

    with AudioStream() as stream:
        for i in range(existing + 1, existing + count + 1):
            input(f"\n[{i - existing}/{count}] Press Enter, then speak one command …")
            # Drain stale audio so the clip starts now.
            while not stream.chunk_queue.empty():
                stream.chunk_queue.get_nowait()
            vad.reset()
            segment = None
            while segment is None:
                chunk = stream.chunk_queue.get()
                if chunk is None:
                    return
                segment = vad.feed(chunk)
            wav = folder / f"clip_{i:02d}.wav"
            sf.write(wav, segment.pcm, AUDIO["capture_rate"])
            print(f"   saved {wav.name} ({segment.duration_s:.1f}s)")
            expected = input("   what did you say? (Enter to skip) ").strip()
            if expected:
                wav.with_suffix(".txt").write_text(expected + "\n", encoding="utf-8")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("folder", type=Path, help="folder of .wav clips (+ .txt sidecars)")
    ap.add_argument("--record", type=int, metavar="N", help="record N clips first")
    ap.add_argument("--denoise", action="store_true",
                    help="run DFN3 before decoding, like the live pipeline")
    args = ap.parse_args()

    if args.record:
        record(args.folder, args.record)
    analyse(args.folder, args.denoise)
