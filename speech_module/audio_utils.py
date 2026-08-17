"""
audio_utils.py
Sample-rate conversion helpers for the speech pipeline.

The microphone is now captured at 48 kHz because that is DeepFilterNet3's
native rate. VOSK and Whisper both want 16 kHz, so every branch that feeds
them passes through one of the converters here:

    Mic (48k) ─┬─ StreamResampler ──────────────────────→ VOSK   (16k)
               └─ buffer → DFN3 → resample_int16 ───────→ Whisper (16k)

Backed by `soxr`, which is already installed as a librosa dependency and is
roughly an order of magnitude faster than librosa's own resampler.

Person A — Real-Time Engine + Audio Pipeline
"""

from __future__ import annotations
import path_setup  # noqa: F401 — adds project root to sys.path

import logging

import numpy as np
import soxr

from config import AUDIO

logger = logging.getLogger(__name__)

# "HQ" is transparent for speech and costs ~1 ms of CPU per second of audio.
_QUALITY = "HQ"


class StreamResampler:
    """
    Stateful 48 kHz → 16 kHz converter for a continuous chunk stream.

    Feed it the raw int16 PCM chunks coming off AudioStream; it returns raw
    int16 PCM at `out_rate`, ready for KaldiRecognizer.AcceptWaveform().

    Statefulness matters here: resampling each chunk independently would clip
    the polyphase filter tail and inject a click at every chunk boundary,
    which VOSK hears as spurious phonemes. Keeping one instance alive for the
    whole session avoids that. Chunk lengths out are therefore *not* exactly
    len_in * out_rate / in_rate — the first chunk is short by the filter delay
    and flush() returns the remainder.

    Usage
    -----
        rs = StreamResampler()
        pcm16k = rs.process(pcm48k)     # per chunk
        pcm16k += rs.flush()            # once, at end of stream
    """

    def __init__(
        self,
        in_rate: int = AUDIO["capture_rate"],
        out_rate: int = AUDIO["target_rate"],
        channels: int = AUDIO["channels"],
    ) -> None:
        self.in_rate = in_rate
        self.out_rate = out_rate
        self.channels = channels

        self._stream: soxr.ResampleStream | None = None
        if in_rate != out_rate:
            self._stream = soxr.ResampleStream(
                in_rate, out_rate, channels, dtype="int16", quality=_QUALITY
            )
            logger.info("StreamResampler: %d Hz → %d Hz (soxr %s)", in_rate, out_rate, _QUALITY)

    @property
    def passthrough(self) -> bool:
        """True when in_rate == out_rate and process() is a no-op."""
        return self._stream is None

    def process(self, pcm: bytes) -> bytes:
        """Resample one raw int16 PCM chunk. Returns raw int16 PCM bytes."""
        if self._stream is None:
            return pcm
        samples = np.frombuffer(pcm, dtype=np.int16)
        return self._stream.resample_chunk(samples).tobytes()

    def flush(self) -> bytes:
        """Drain the filter tail. Call once when the stream ends."""
        if self._stream is None:
            return b""
        return self._stream.resample_chunk(np.zeros(0, dtype=np.int16), last=True).tobytes()

    def reset(self) -> None:
        """Clear filter state — use when the audio source restarts."""
        if self._stream is not None:
            self._stream.clear()


def resample_int16(audio: np.ndarray, in_rate: int, out_rate: int) -> np.ndarray:
    """
    One-shot resample of a complete int16 buffer — used on the dictation
    branch, where the whole utterance is already in memory.
    """
    if in_rate == out_rate or audio.size == 0:
        return audio
    return soxr.resample(audio, in_rate, out_rate, quality=_QUALITY)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    src = AUDIO["capture_rate"]
    dst = AUDIO["target_rate"]
    chunk = AUDIO["chunk_size"]

    # One second of a 440 Hz tone, fed through in realistic chunk sizes.
    t = np.arange(src, dtype=np.float32) / src
    tone = (np.sin(2 * np.pi * 440 * t) * 12000).astype(np.int16)

    rs = StreamResampler()
    out = b"".join(rs.process(tone[i:i + chunk].tobytes()) for i in range(0, len(tone), chunk))
    out += rs.flush()
    streamed = np.frombuffer(out, dtype=np.int16)

    oneshot = resample_int16(tone, src, dst)

    print(f"in      : {len(tone)} samples @ {src} Hz")
    print(f"streamed: {len(streamed)} samples @ {dst} Hz")
    print(f"oneshot : {len(oneshot)} samples @ {dst} Hz")
