"""
vad.py
Voice activity detection — the segmenter at the head of the speech pipeline.

    Mic (48k) → VAD → voiced segment → DFN3 → VOSK → command | Whisper

Two implementations behind one interface (create_vad() picks at runtime):

  SileroVAD  — preferred. A ~2 MB neural model (onnxruntime, sub-ms per
               frame) that scores *speech probability*, so keyboard clatter,
               music and coughs that clear an energy threshold do not open a
               segment. Fewer false segments = fewer wasted VOSK/Whisper
               decodes and less garbage sent to the central module.
  EnergyVAD  — adaptive-threshold RMS gate. Zero extra dependencies; the
               fallback when the Silero model can't load, the same way DFN3
               degrades to pass-through.

Why segments rather than a chunk stream
---------------------------------------
DeepFilterNet's Python API has no streaming mode: `df.enhance()` resets the
model's recurrent state on every call and pads each input, so denoising 100 ms
chunks independently would both discard temporal context and inject a boundary
artefact ten times a second. Handing it one complete utterance is its best
case, and a VAD is what produces those utterances.

Why a relative threshold
------------------------
A fixed RMS gate (what DictationBuffer used) works in a quiet room and fails
next to a fan — the noise sits above the threshold and speech never "ends".
This tracks a running percentile of recent quiet frames as the noise floor and
triggers on multiples of it, so the same settings hold across environments.
Separate start/end factors give hysteresis, so a level hovering near the
threshold does not chatter the state machine.

Person A — Real-Time Engine + Audio Pipeline
"""

from __future__ import annotations
import path_setup  # noqa: F401 — adds project root to sys.path

import collections
import logging
import time
from dataclasses import dataclass

import numpy as np

from audio_utils import StreamResampler
from config import AUDIO, SILERO as SILERO_CFG, VAD as VAD_CFG

logger = logging.getLogger(__name__)


@dataclass
class Segment:
    """One voiced stretch of audio, at the capture rate."""
    pcm: np.ndarray             # int16, mono, AUDIO["capture_rate"]
    duration_s: float
    peak_rms: float
    noise_floor: float
    truncated: bool = False     # hit max_segment_s rather than ending naturally
    # Wall-clock epoch seconds of the segment's first sample (preroll included),
    # derived as close-time minus duration — so it can lag true onset by any
    # queueing delay ahead of the VAD. Word offsets from the recognisers are
    # relative to this, letting the central module place words on its timeline.
    wall_start: float = 0.0
    # SileroVAD only: mean speech probability over the voiced frames.
    # 0.0 = segment came from EnergyVAD, which has no such notion.
    speech_prob: float = 0.0

    def __repr__(self) -> str:
        return (f"Segment({self.duration_s:.2f}s  peak_rms={self.peak_rms:.0f}  "
                f"floor={self.noise_floor:.0f}"
                f"{'  TRUNCATED' if self.truncated else ''})")


class EnergyVAD:
    """
    Frame-level energy VAD with an adaptive noise floor.

    Feed it raw int16 PCM chunks; it returns a Segment when one completes and
    None otherwise. One chunk can only ever close one segment, because a chunk
    is far shorter than the hangover.

    Usage
    -----
        vad = EnergyVAD()
        seg = vad.feed(chunk)
        if seg:
            ...                 # whole utterance, ready for DFN3
    """

    def __init__(
        self,
        sample_rate: int = AUDIO["capture_rate"],
        frame_ms: int = VAD_CFG["frame_ms"],
        start_factor: float = VAD_CFG["start_factor"],
        end_factor: float = VAD_CFG["end_factor"],
        min_speech_ms: int = VAD_CFG["min_speech_ms"],
        hangover_ms: int = VAD_CFG["hangover_ms"],
        preroll_ms: int = VAD_CFG["preroll_ms"],
        max_segment_s: float = VAD_CFG["max_segment_s"],
        absolute_floor_rms: float = VAD_CFG["absolute_floor_rms"],
        noise_percentile: float = VAD_CFG["noise_floor_percentile"],
        noise_window_s: float = VAD_CFG["noise_floor_window_s"],
    ) -> None:
        self.sample_rate = sample_rate
        self.frame_len = int(sample_rate * frame_ms / 1000)
        self.start_factor = start_factor
        self.end_factor = end_factor
        self.absolute_floor_rms = absolute_floor_rms
        self.noise_percentile = noise_percentile

        frames_per_s = 1000 / frame_ms
        self._min_speech_frames = max(1, int(min_speech_ms / frame_ms))
        self._hangover_frames = max(1, int(hangover_ms / frame_ms))
        self._max_frames = int(max_segment_s * frames_per_s)
        self._noise_hist: collections.deque[float] = collections.deque(
            maxlen=max(8, int(noise_window_s * frames_per_s))
        )
        self._preroll: collections.deque[np.ndarray] = collections.deque(
            maxlen=max(1, int(preroll_ms / frame_ms))
        )

        # Odd-sized chunks are carried across feed() calls so framing stays exact.
        self._residue = np.zeros(0, dtype=np.int16)

        self._in_speech = False
        self._voiced_frames = 0
        self._silence_frames = 0
        self._peak_rms = 0.0
        self._buffer: list[np.ndarray] = []

    # ── Public API ─────────────────────────────────────────────────────────────

    @property
    def noise_floor(self) -> float:
        """Current noise-floor estimate in int16 RMS."""
        if not self._noise_hist:
            return self.absolute_floor_rms
        return max(
            float(np.percentile(self._noise_hist, self.noise_percentile)),
            self.absolute_floor_rms,
        )

    @property
    def in_speech(self) -> bool:
        return self._in_speech

    def feed(self, chunk: bytes | np.ndarray) -> Segment | None:
        """Process one PCM chunk. Returns a Segment when one just completed."""
        samples = (np.frombuffer(chunk, dtype=np.int16)
                   if isinstance(chunk, (bytes, bytearray)) else chunk)
        if samples.size == 0:
            return None

        data = np.concatenate((self._residue, samples)) if self._residue.size else samples
        n_frames = len(data) // self.frame_len
        self._residue = data[n_frames * self.frame_len:].copy()

        completed: Segment | None = None
        for i in range(n_frames):
            frame = data[i * self.frame_len:(i + 1) * self.frame_len]
            seg = self._process_frame(frame)
            if seg is not None:
                completed = seg      # at most one per chunk in practice
        return completed

    def flush(self) -> Segment | None:
        """Close any in-progress segment — call when the stream ends."""
        if self._in_speech and self._voiced_frames >= self._min_speech_frames:
            return self._close_segment(truncated=False)
        self._reset_segment()
        return None

    def reset(self) -> None:
        """Drop all state, including the learned noise floor."""
        self._noise_hist.clear()
        self._preroll.clear()
        self._residue = np.zeros(0, dtype=np.int16)
        self._reset_segment()

    # ── Internals ──────────────────────────────────────────────────────────────

    @staticmethod
    def _rms(frame: np.ndarray) -> float:
        return float(np.sqrt(np.mean(frame.astype(np.float32) ** 2)) + 1e-9)

    def _process_frame(self, frame: np.ndarray) -> Segment | None:
        rms = self._rms(frame)
        floor = self.noise_floor

        if not self._in_speech:
            # Only quiet frames update the floor, or speech would raise it and
            # the VAD would slowly go deaf to the speaker.
            self._noise_hist.append(rms)
            self._preroll.append(frame)

            if rms > floor * self.start_factor:
                self._in_speech = True
                self._voiced_frames = 1
                self._silence_frames = 0
                self._peak_rms = rms
                # Seed with pre-roll so the onset consonant is not clipped —
                # energy only crosses the threshold after speech has begun.
                self._buffer = list(self._preroll)
                self._buffer.append(frame)
            return None

        # ── in speech ──────────────────────────────────────────────────────────
        self._buffer.append(frame)
        self._peak_rms = max(self._peak_rms, rms)

        if rms > floor * self.end_factor:
            self._voiced_frames += 1
            self._silence_frames = 0
        else:
            self._silence_frames += 1

        if len(self._buffer) >= self._max_frames:
            logger.warning("VAD: segment hit max_segment_s — forcing a cut.")
            return self._close_segment(truncated=True)

        if self._silence_frames >= self._hangover_frames:
            if self._voiced_frames >= self._min_speech_frames:
                return self._close_segment(truncated=False)
            # Too short to be speech — discard and relearn from these frames.
            logger.debug("VAD: discarded %d-frame blip.", self._voiced_frames)
            for f in self._buffer[-self._silence_frames:]:
                self._noise_hist.append(self._rms(f))
            self._reset_segment()
        return None

    def _close_segment(self, truncated: bool) -> Segment:
        pcm = np.concatenate(self._buffer) if self._buffer else np.zeros(0, np.int16)
        duration_s = len(pcm) / self.sample_rate
        seg = Segment(
            pcm=pcm,
            duration_s=duration_s,
            peak_rms=self._peak_rms,
            noise_floor=self.noise_floor,
            truncated=truncated,
            wall_start=time.time() - duration_s,
        )
        self._reset_segment()
        logger.debug("VAD: %r", seg)
        return seg

    def _reset_segment(self) -> None:
        self._in_speech = False
        self._voiced_frames = 0
        self._silence_frames = 0
        self._peak_rms = 0.0
        self._buffer = []
        self._preroll.clear()


# ── Silero VAD ─────────────────────────────────────────────────────────────────

class SileroVAD:
    """
    Neural VAD with EnergyVAD's exact interface: feed 48 kHz int16 chunks,
    get a Segment (still at 48 kHz, preroll included) when one completes.

    The model scores 512-sample frames of 16 kHz audio (32 ms), so the chunk
    stream is resampled down for *scoring only* — segment audio is cut from
    the original 48 kHz stream, which is what DFN3 wants. The two streams
    stay aligned frame-for-frame (one 512-sample frame ↔ 1536 capture
    samples); the resampler's few-ms filter delay shifts boundaries slightly,
    absorbed by preroll and hangover.

    Supports both Silero ONNX generations, detected from the session inputs:
    v5 ('input'/'state'/'sr') and v6 ('input'/'h'/'c'). Both take 64 samples
    of carried context + 512 new samples per call.
    """

    _FRAME16 = 512      # model frame @ 16 kHz
    _CONTEXT = 64       # carried tail samples the model expects prepended

    def __init__(
        self,
        sample_rate: int = AUDIO["capture_rate"],
        model_path: str = SILERO_CFG["model_path"],
        start_prob: float = SILERO_CFG["start_prob"],
        end_prob: float = SILERO_CFG["end_prob"],
        min_speech_ms: int = SILERO_CFG["min_speech_ms"],
        hangover_ms: int = SILERO_CFG["hangover_ms"],
        preroll_ms: int = SILERO_CFG["preroll_ms"],
        max_segment_s: float = SILERO_CFG["max_segment_s"],
    ) -> None:
        import onnxruntime  # deferred: EnergyVAD users don't need it

        if sample_rate % 16000:
            raise ValueError(f"sample_rate {sample_rate} not a multiple of 16 kHz")
        self.sample_rate = sample_rate
        self.frame_in = self._FRAME16 * (sample_rate // 16000)   # 1536 @ 48 kHz

        frame_ms = 32
        self._min_speech_frames = max(1, int(min_speech_ms / frame_ms))
        self._hangover_frames = max(1, int(hangover_ms / frame_ms))
        self._max_frames = int(max_segment_s * 1000 / frame_ms)
        self._preroll: collections.deque[np.ndarray] = collections.deque(
            maxlen=max(1, int(preroll_ms / frame_ms))
        )
        self.start_prob = start_prob
        self.end_prob = end_prob

        opts = onnxruntime.SessionOptions()
        opts.inter_op_num_threads = 1
        opts.intra_op_num_threads = 1
        opts.log_severity_level = 3
        self._sess = onnxruntime.InferenceSession(
            model_path, providers=["CPUExecutionProvider"], sess_options=opts
        )
        inputs = {i.name for i in self._sess.get_inputs()}
        if {"input", "state", "sr"} <= inputs:
            self._layout = "v5"
        elif {"input", "h", "c"} <= inputs:
            self._layout = "v6"
        else:
            raise RuntimeError(f"Unrecognised Silero ONNX inputs: {sorted(inputs)}")

        self._resampler = StreamResampler(in_rate=sample_rate, out_rate=16000)
        self._fifo16 = np.zeros(0, dtype=np.float32)
        self._fifo48 = np.zeros(0, dtype=np.int16)
        self._reset_model_state()
        self._reset_segment()
        logger.info("SileroVAD ready (%s layout, %s).", self._layout, model_path)

    # ── Public API (mirrors EnergyVAD) ─────────────────────────────────────────

    @property
    def noise_floor(self) -> float:
        """Interface parity with EnergyVAD — probabilities have no floor."""
        return 0.0

    @property
    def in_speech(self) -> bool:
        return self._in_speech

    def feed(self, chunk: bytes | np.ndarray) -> Segment | None:
        """Process one PCM chunk. Returns a Segment when one just completed."""
        samples = (np.frombuffer(chunk, dtype=np.int16)
                   if isinstance(chunk, (bytes, bytearray)) else chunk)
        if samples.size == 0:
            return None

        self._fifo48 = np.concatenate((self._fifo48, samples))
        res16 = np.frombuffer(self._resampler.process(samples.tobytes()), dtype=np.int16)
        if res16.size:
            self._fifo16 = np.concatenate(
                (self._fifo16, res16.astype(np.float32) / 32768.0)
            )

        completed: Segment | None = None
        while self._fifo16.size >= self._FRAME16 and self._fifo48.size >= self.frame_in:
            frame16, self._fifo16 = self._fifo16[:self._FRAME16], self._fifo16[self._FRAME16:]
            frame48, self._fifo48 = self._fifo48[:self.frame_in], self._fifo48[self.frame_in:]
            seg = self._process_frame(frame16, frame48)
            if seg is not None:
                completed = seg
        return completed

    def flush(self) -> Segment | None:
        """Close any in-progress segment — call when the stream ends."""
        if self._in_speech and self._voiced_frames >= self._min_speech_frames:
            return self._close_segment(truncated=False)
        self._reset_segment()
        return None

    def reset(self) -> None:
        """Drop all state, including the model's recurrent memory."""
        self._resampler.reset()
        self._fifo16 = np.zeros(0, dtype=np.float32)
        self._fifo48 = np.zeros(0, dtype=np.int16)
        self._preroll.clear()
        self._reset_model_state()
        self._reset_segment()

    # ── Internals ──────────────────────────────────────────────────────────────

    def _reset_model_state(self) -> None:
        self._context = np.zeros(self._CONTEXT, dtype=np.float32)
        self._state = np.zeros((2, 1, 128), dtype=np.float32)     # v5
        self._h = np.zeros((1, 1, 128), dtype=np.float32)         # v6
        self._c = np.zeros((1, 1, 128), dtype=np.float32)

    def _speech_prob(self, frame16: np.ndarray) -> float:
        x = np.concatenate((self._context, frame16))[np.newaxis, :]
        if self._layout == "v5":
            out, self._state = self._sess.run(
                None, {"input": x, "state": self._state,
                       "sr": np.array(16000, dtype=np.int64)},
            )
        else:
            out, self._h, self._c = self._sess.run(
                None, {"input": x, "h": self._h, "c": self._c},
            )
        self._context = frame16[-self._CONTEXT:]
        return float(out.reshape(-1)[0])

    def _process_frame(self, frame16: np.ndarray, frame48: np.ndarray) -> Segment | None:
        prob = self._speech_prob(frame16)

        if not self._in_speech:
            self._preroll.append(frame48)
            if prob >= self.start_prob:
                self._in_speech = True
                self._voiced_frames = 1
                self._silence_frames = 0
                self._prob_sum = prob
                self._peak_rms = EnergyVAD._rms(frame48)
                # Pre-roll (current frame included) so the onset consonant —
                # spoken before the probability crossed — is not clipped.
                self._buffer = list(self._preroll)
            return None

        # ── in speech ──────────────────────────────────────────────────────────
        self._buffer.append(frame48)
        self._peak_rms = max(self._peak_rms, EnergyVAD._rms(frame48))

        if prob > self.end_prob:
            self._voiced_frames += 1
            self._silence_frames = 0
            self._prob_sum += prob
        else:
            self._silence_frames += 1

        if len(self._buffer) >= self._max_frames:
            logger.warning("SileroVAD: segment hit max_segment_s — forcing a cut.")
            return self._close_segment(truncated=True)

        if self._silence_frames >= self._hangover_frames:
            if self._voiced_frames >= self._min_speech_frames:
                return self._close_segment(truncated=False)
            logger.debug("SileroVAD: discarded %d-frame blip.", self._voiced_frames)
            self._reset_segment()
        return None

    def _close_segment(self, truncated: bool) -> Segment:
        pcm = np.concatenate(self._buffer) if self._buffer else np.zeros(0, np.int16)
        duration_s = len(pcm) / self.sample_rate
        seg = Segment(
            pcm=pcm,
            duration_s=duration_s,
            peak_rms=self._peak_rms,
            noise_floor=0.0,
            truncated=truncated,
            wall_start=time.time() - duration_s,
            speech_prob=self._prob_sum / max(1, self._voiced_frames),
        )
        self._reset_segment()
        logger.debug("SileroVAD: %r", seg)
        return seg

    def _reset_segment(self) -> None:
        self._in_speech = False
        self._voiced_frames = 0
        self._silence_frames = 0
        self._peak_rms = 0.0
        self._prob_sum = 0.0
        self._buffer: list[np.ndarray] = []
        self._preroll.clear()


# ── Factory ────────────────────────────────────────────────────────────────────

def create_vad(sample_rate: int = AUDIO["capture_rate"]):
    """
    The segmenter the pipeline should use: SileroVAD when enabled and its
    model loads, EnergyVAD otherwise (unless SILERO["required"]).
    """
    if SILERO_CFG["enabled"]:
        try:
            return SileroVAD(sample_rate=sample_rate)
        except Exception as exc:
            if SILERO_CFG["required"]:
                raise
            logger.warning("Silero VAD unavailable (%s) — using EnergyVAD.", exc)
    return EnergyVAD(sample_rate=sample_rate)


# ── Smoke-test ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    logging.basicConfig(level=logging.DEBUG, stream=sys.stdout,
                        format="%(levelname)s: %(message)s")

    sr = AUDIO["capture_rate"]
    chunk = AUDIO["chunk_size"]
    rng = np.random.default_rng(0)

    # 1 s noise, 1.5 s "speech", 1.5 s noise — over a noticeable noise floor.
    noise = lambda n: (rng.standard_normal(n) * 200).astype(np.int16)   # noqa: E731
    t = np.arange(int(sr * 1.5)) / sr
    speech = ((np.sin(2 * np.pi * 200 * t) + 0.5 * np.sin(2 * np.pi * 400 * t))
              * 6000).astype(np.int16) + noise(len(t))
    stream = np.concatenate([noise(sr), speech, noise(int(sr * 1.5))])

    vad = EnergyVAD()
    found = []
    for i in range(0, len(stream), chunk):
        seg = vad.feed(stream[i:i + chunk])
        if seg:
            found.append(seg)
    tail = vad.flush()
    if tail:
        found.append(tail)

    print(f"\nnoise floor learned : {vad.noise_floor:.0f}")
    print(f"segments detected   : {len(found)}  (expect 1)")
    for s in found:
        print(f"   {s}")
    print(f"\nspeech was 1.50s; detected {found[0].duration_s:.2f}s "
          f"(pre-roll + hangover make it longer)" if found else "NO SEGMENT")
