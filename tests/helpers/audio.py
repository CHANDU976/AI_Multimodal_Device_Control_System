"""Deterministic synthetic audio (int16 PCM) for the speech-pipeline tests."""

from __future__ import annotations

import numpy as np

RATE = 48000


def noise(seconds: float, rms: float = 200.0, rate: int = RATE, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return (rng.standard_normal(int(rate * seconds)) * rms).astype(np.int16)


def voice(seconds: float, amp: float = 6000.0, rate: int = RATE, seed: int = 1) -> np.ndarray:
    """A two-partial tone over light noise — loud, voiced, speech-band."""
    t = np.arange(int(rate * seconds)) / rate
    tone = (np.sin(2 * np.pi * 200 * t) + 0.5 * np.sin(2 * np.pi * 400 * t)) * amp
    return (tone + noise(seconds, 200.0, rate, seed)).clip(-32768, 32767).astype(np.int16)


def silence(seconds: float, rate: int = RATE) -> np.ndarray:
    return np.zeros(int(rate * seconds), dtype=np.int16)


def chunks(signal: np.ndarray, size: int = 4800):
    for i in range(0, len(signal), size):
        yield signal[i:i + size]


def feed_all(vad, signal: np.ndarray, size: int = 4800, as_bytes: bool = False):
    """Push a whole signal through a VAD chunk by chunk; returns all segments."""
    found = []
    for chunk in chunks(signal, size):
        seg = vad.feed(chunk.tobytes() if as_bytes else chunk)
        if seg is not None:
            found.append(seg)
    tail = vad.flush()
    if tail is not None:
        found.append(tail)
    return found
