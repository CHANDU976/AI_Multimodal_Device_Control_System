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

import sounddevice as sd
import numpy as np

from config import AUDIO, CONCURRENCY

logger = logging.getLogger(__name__)


class AudioStream:
    """
    Opens a microphone input stream and feeds raw int16 PCM chunks
    into `chunk_queue` for downstream consumers (VOSK, Whisper, etc.).

    Usage
    -----
    stream = AudioStream()
    stream.start()
    chunk = stream.chunk_queue.get()   # blocks until audio arrives
    stream.stop()
    """

    def __init__(
        self,
        sample_rate: int = AUDIO["sample_rate"],
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
        if self._stream:
            self._stream.stop()
            self._stream.close()
            self._stream = None
        self.chunk_queue.put(None)   # sentinel — tells consumers to exit
        logger.info("AudioStream stopped.")

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
    devices = sd.query_devices()
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
