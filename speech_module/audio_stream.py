"""
audio_stream.py
Streams raw PCM audio from the microphone into a thread-safe queue.

Person A — Real-Time Engine + Audio Pipeline
"""

from __future__ import annotations
import path_setup  # noqa: F401 — adds project root to sys.path

import queue
import threading
import logging
from typing import Callable

import numpy as np

from config import AUDIO, CONCURRENCY

logger = logging.getLogger(__name__)

# sounddevice is imported on first use, not at module import — see _sounddevice().
_sd = None


def _sounddevice():
    """
    Import sounddevice lazily so PortAudio is only initialised in processes
    that actually record.

    On Windows, multiprocessing spawn makes every child re-import the __main__
    module. When speech_manager.py is the entry point that pulls in this
    module, so the Whisper worker — which never touches a microphone — would
    otherwise enumerate audio devices on startup. That enumeration can stall
    for seconds or hang outright when a headset is plugged in, and the worker
    then never reaches its READY signal, freezing SpeechManager.start().
    """
    global _sd
    if _sd is None:
        import sounddevice
        _sd = sounddevice
    return _sd


class AudioStream:
    """
    Opens a microphone input stream and feeds raw int16 PCM chunks
    into `chunk_queue` for downstream consumers (DFN3, VOSK, Whisper, etc.).

    Captures at AUDIO["capture_rate"] (48 kHz — DeepFilterNet3's native rate),
    *not* at the 16 kHz the recognisers want. Downsampling happens per-branch
    in audio_utils.StreamResampler / resample_int16, so the denoiser gets the
    full-band signal it was trained on.

    Usage
    -----
    stream = AudioStream()
    stream.start()
    chunk = stream.chunk_queue.get()   # blocks until audio arrives
    stream.stop()
    """

    def __init__(
        self,
        sample_rate: int = AUDIO["capture_rate"],
        channels: int = AUDIO["channels"],
        chunk_size: int = AUDIO["chunk_size"],
        device_index: int | None = AUDIO["device_index"],
        on_error: Callable[[Exception], None] | None = None,
    ) -> None:
        self.sample_rate = sample_rate
        self.channels = channels
        self.chunk_size = chunk_size
        self.device_index = device_index
        self.on_error = on_error or self._default_error_handler

        self.chunk_queue: queue.Queue[bytes | None] = queue.Queue(
            maxsize=CONCURRENCY["vosk_queue_maxsize"]
        )
        self._stream: sd.RawInputStream | None = None
        self._running = threading.Event()

    # ── Public API ─────────────────────────────────────────────────────────────

    def start(self) -> None:
        """Open the microphone and begin pushing chunks to the queue."""
        if self._running.is_set():
            logger.warning("AudioStream is already running.")
            return

        sd = _sounddevice()
        self._stream = sd.RawInputStream(
            samplerate=self.sample_rate,
            blocksize=self.chunk_size,
            device=self.device_index,
            dtype=AUDIO["dtype"],
            channels=self.channels,
            callback=self._audio_callback,
        )
        self._stream.start()
        self._running.set()
        logger.info(
            "AudioStream started — device=%s  rate=%d Hz  chunk=%d frames",
            self.device_index,
            self.sample_rate,
            self.chunk_size,
        )

    def stop(self) -> None:
        """Stop the stream and signal consumers with a sentinel None."""
        if not self._running.is_set():
            return
        self._running.clear()
        self._close_stream()
        self.chunk_queue.put(None)   # sentinel — tells consumers to exit
        logger.info("AudioStream stopped.")

    _KEEP = object()   # restart(): "leave device_index as it is"

    def restart(self, device_index: int | None | object = _KEEP) -> None:
        """
        Reopen the capture stream, re-initialising PortAudio in between so the
        reopened stream sees the current device world — this is how a change
        of the system default input device is actually picked up.

        Unlike stop(), no None sentinel is pushed: downstream consumers
        (VAD, preview recognizer) just see a short gap in the chunk stream
        and keep running.

        Parameters
        ----------
        device_index : int | None, optional
            New device to capture from; None = system default. Omitted =
            keep the current setting.
        """
        was_running = self._running.is_set()
        self._running.clear()
        self._close_stream()
        if device_index is not self._KEEP:
            self.device_index = device_index

        # Fresh PortAudio state. Safe here: this process holds no open
        # streams between _close_stream() and start().
        sd = _sounddevice()
        try:
            sd._terminate()
            sd._initialize()
        except Exception as exc:
            logger.warning("PortAudio re-init failed (%s) — reopening anyway.", exc)

        if was_running:
            self.start()

    def _close_stream(self) -> None:
        if self._stream:
            self._stream.stop()
            self._stream.close()
            self._stream = None

    def is_running(self) -> bool:
        return self._running.is_set()

    def clear_queue(self) -> None:
        """Drain pending chunks (useful when switching modes)."""
        while not self.chunk_queue.empty():
            try:
                self.chunk_queue.get_nowait()
            except queue.Empty:
                break

    # ── Internal ───────────────────────────────────────────────────────────────

    def _audio_callback(
        self,
        indata: bytes,
        frames: int,
        time,          # noqa: ANN001 (CData from sounddevice)
        status: sd.CallbackFlags,
    ) -> None:
        if status:
            logger.warning("Audio callback status: %s", status)
        if self._running.is_set():
            try:
                self.chunk_queue.put_nowait(bytes(indata))
            except queue.Full:
                try:
                    self.chunk_queue.get_nowait()
                except queue.Empty:
                    pass
                try:
                    self.chunk_queue.put_nowait(bytes(indata))
                except queue.Full:
                    pass
                #logger.warning("que full old chunk dropped")

    @staticmethod
    def _default_error_handler(exc: Exception) -> None:
        logger.error("AudioStream error: %s", exc)

    # ── Context-manager support ────────────────────────────────────────────────

    def __enter__(self) -> "AudioStream":
        self.start()
        return self

    def __exit__(self, *_) -> None:
        self.stop()


# ── Helpers ────────────────────────────────────────────────────────────────────

def list_input_devices() -> list[dict]:
    """Return a list of available audio input devices for diagnostics."""
    devices = _sounddevice().query_devices()
    return [
        {"index": i, "name": d["name"], "channels": d["max_input_channels"]}
        for i, d in enumerate(devices)
        if d["max_input_channels"] > 0
    ]


if __name__ == "__main__":
    # Quick smoke-test: print 5 chunk sizes then exit
    logging.basicConfig(level=logging.DEBUG)
    print("Available input devices:")
    for dev in list_input_devices():
        print(f"  [{dev['index']}] {dev['name']}  ({dev['channels']} ch)")

    stream = AudioStream()
    stream.start()
    for i in range(5):
        chunk = stream.chunk_queue.get()
        print(f"Chunk {i+1}: {len(chunk)} bytes")
    stream.stop()
