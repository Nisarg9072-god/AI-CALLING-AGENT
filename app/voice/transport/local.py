"""
LocalTransport -- microphone input + speaker output via sounddevice.

Requires:
    pip install sounddevice

This is the default development transport when VOICE_TRANSPORT=local.
No telephony provider, no WebSocket -- pure local audio I/O.

Flow:
    Mic (sounddevice) -> AudioFrame -> VoiceSession -> STT -> Agent -> TTS -> AudioFrame -> Speaker
"""

from __future__ import annotations

import queue
import threading
import time
from typing import Any

from app.voice.audio import (
    AudioFrame,
    AudioConverter,
    INTERNAL_SAMPLE_RATE,
    INTERNAL_SAMPLE_WIDTH,
    INTERNAL_CHANNELS,
    compute_rms,
)
from app.voice.transport.base import VoiceTransport, TransportStatus


class LocalTransport(VoiceTransport):
    """
    Local microphone + speaker transport using sounddevice.

    Records audio in chunks, applies basic silence detection,
    and returns AudioFrames to the VoiceSession.

    Args:
        chunk_duration: Duration of each audio chunk in seconds.
        silence_threshold: RMS energy below which audio is considered silent.
        min_speech_duration: Minimum speech duration to trigger STT.
    """

    def __init__(
        self,
        chunk_duration: float = 0.5,
        silence_threshold: float = 300.0,
        min_speech_duration: float = 0.5,
        playback_sample_rate: int = 22050,
    ) -> None:
        self._chunk_duration = chunk_duration
        self._silence_threshold = silence_threshold
        self._min_speech_duration = min_speech_duration
        self._playback_sample_rate = playback_sample_rate

        self._status = TransportStatus.IDLE
        self._audio_queue: queue.Queue[AudioFrame | None] = queue.Queue()
        self._recording_thread: threading.Thread | None = None
        self._stop_event = threading.Event()

        # Speech buffer (collects chunks until silence detected)
        self._speech_buffer: list[AudioFrame] = []
        self._speech_started = False
        self._silence_chunks = 0
        self._max_silence_chunks = 6   # ~3 seconds of silence ends utterance

    def start_session(self) -> None:
        self._status = TransportStatus.ACTIVE
        self._stop_event.clear()
        print("\n[LocalTransport] Voice session started.")
        print("[LocalTransport] Speak into your microphone. Press Ctrl+C to end.")

    def receive_audio(self, timeout: float | None = 30.0) -> AudioFrame | None:
        """
        Record from microphone until an utterance is complete (silence detected).
        Returns the complete utterance as a single AudioFrame.
        """
        try:
            import sounddevice as sd
            import numpy as np
        except ImportError:
            raise RuntimeError(
                "sounddevice not installed.\n"
                "Install with:  pip install sounddevice\n"
                "Or:            uv pip install -e '.[voice]'"
            )

        self._speech_buffer.clear()
        self._speech_started = False
        self._silence_chunks = 0
        chunk_samples = int(self._chunk_duration * INTERNAL_SAMPLE_RATE)

        deadline = time.monotonic() + timeout if timeout else None

        print("[LocalTransport] Listening...", end="", flush=True)

        while True:
            if timeout and time.monotonic() > deadline:
                return None

            if self._stop_event.is_set():
                return None

            # Record a chunk
            try:
                raw = sd.rec(
                    chunk_samples,
                    samplerate=INTERNAL_SAMPLE_RATE,
                    channels=INTERNAL_CHANNELS,
                    dtype="int16",
                    blocking=True,
                )
            except Exception:
                return None

            chunk_bytes = raw.tobytes()
            frame = AudioFrame(
                samples=chunk_bytes,
                sample_rate=INTERNAL_SAMPLE_RATE,
                channels=INTERNAL_CHANNELS,
                sample_width=INTERNAL_SAMPLE_WIDTH,
            )
            rms = compute_rms(frame)

            if rms > self._silence_threshold:
                # Speech detected
                if not self._speech_started:
                    self._speech_started = True
                    print(" [speaking]", end="", flush=True)
                self._speech_buffer.append(frame)
                self._silence_chunks = 0
            elif self._speech_started:
                # Silence after speech
                self._speech_buffer.append(frame)
                self._silence_chunks += 1
                if self._silence_chunks >= self._max_silence_chunks:
                    # End of utterance
                    print(" [done]")
                    combined = b"".join(f.samples for f in self._speech_buffer)
                    return AudioFrame(
                        samples=combined,
                        sample_rate=INTERNAL_SAMPLE_RATE,
                        channels=INTERNAL_CHANNELS,
                        sample_width=INTERNAL_SAMPLE_WIDTH,
                    )

    def send_audio(self, audio: AudioFrame) -> None:
        """Play audio through the speaker."""
        try:
            import sounddevice as sd
            import numpy as np

            # Convert to numpy for playback
            arr = np.frombuffer(audio.samples, dtype=np.int16)

            # Resample to playback rate if needed
            if audio.sample_rate != self._playback_sample_rate:
                from app.voice.audio import AudioConverter
                resampled = AudioConverter.resample(audio, self._playback_sample_rate)
                arr = np.frombuffer(resampled.samples, dtype=np.int16)
                playback_rate = self._playback_sample_rate
            else:
                playback_rate = audio.sample_rate

            sd.play(arr, samplerate=playback_rate, blocking=True)

        except ImportError:
            pass    # No speaker -- degraded mode, text only
        except Exception:
            pass

    def end_session(self) -> None:
        self._stop_event.set()
        self._status = TransportStatus.ENDED
        print("\n[LocalTransport] Session ended.")

    def get_status(self) -> dict:
        return {
            "transport": "local",
            "status": self._status.value,
        }
