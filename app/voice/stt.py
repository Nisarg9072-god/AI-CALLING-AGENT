"""
STT (Speech-to-Text) providers.

Interface:  STTProvider.transcribe(AudioFrame) -> str
Impl:       FasterWhisperProvider -- local, free, no cloud
Mock:       MockSTTProvider -- scripted responses for tests

Configuration (.env):
    WHISPER_MODEL=small          # tiny | base | small | medium | large-v3
    WHISPER_DEVICE=auto          # auto | cpu | cuda
    WHISPER_COMPUTE_TYPE=auto    # auto | int8 | float16 | float32
    WHISPER_LANGUAGE=en          # ISO 639-1 language code
"""

from __future__ import annotations

import time
from abc import ABC, abstractmethod
from typing import Any

from app.voice.audio import AudioFrame, INTERNAL_SAMPLE_RATE


# ── Abstract interface ─────────────────────────────────────────────────────────


class STTProvider(ABC):
    """
    Speech-to-Text provider interface.

    The agent core depends ONLY on this interface.
    Swap implementations without changing any agent code.
    """

    @abstractmethod
    def transcribe(self, audio: AudioFrame) -> str:
        """
        Convert audio to text.

        Args:
            audio: Normalized AudioFrame (PCM mono 16kHz 16-bit).

        Returns:
            Transcribed text, or "" if nothing detected.
            NEVER raises -- returns "" on any error.
        """

    @property
    def provider_name(self) -> str:
        return type(self).__name__


# ── FasterWhisper (local, free) ────────────────────────────────────────────────


class FasterWhisperProvider(STTProvider):
    """
    Local STT using faster-whisper (CTranslate2 backend).

    Fully local -- no audio ever sent to an external API.
    Supports CPU and CUDA inference.

    Install:
        pip install faster-whisper
        # OR
        uv pip install -e ".[voice]"

    Models are downloaded automatically on first use from HuggingFace.
    Subsequent uses load from local cache.

    Configuration:
        WHISPER_MODEL=small          (tiny/base/small/medium/large-v3)
        WHISPER_DEVICE=auto          (auto/cpu/cuda)
        WHISPER_COMPUTE_TYPE=auto    (auto/int8/float16/float32)
        WHISPER_LANGUAGE=en
    """

    def __init__(
        self,
        model_size: str | None = None,
        device: str | None = None,
        compute_type: str | None = None,
        language: str | None = None,
    ) -> None:
        from app.config import settings

        self._model_size = model_size or getattr(settings, "whisper_model", "small")
        self._language = language or getattr(settings, "whisper_language", "en")

        # Resolve auto device/compute_type
        raw_device = device or getattr(settings, "whisper_device", "auto")
        raw_compute = compute_type or getattr(settings, "whisper_compute_type", "auto")

        self._device, self._compute_type = self._resolve_device(raw_device, raw_compute)
        self._model = None      # lazy loaded
        self._load_time_ms: float = 0.0

    def _resolve_device(self, device: str, compute_type: str) -> tuple[str, str]:
        """Resolve 'auto' device and compute_type based on available hardware."""
        if device == "auto":
            try:
                import torch
                device = "cuda" if torch.cuda.is_available() else "cpu"
            except ImportError:
                device = "cpu"

        if compute_type == "auto":
            compute_type = "int8" if device == "cpu" else "float16"

        return device, compute_type

    def _load(self) -> None:
        """Lazy-load the Whisper model on first use."""
        if self._model is not None:
            return
        try:
            from faster_whisper import WhisperModel
            t0 = time.monotonic()
            self._model = WhisperModel(
                self._model_size,
                device=self._device,
                compute_type=self._compute_type,
            )
            self._load_time_ms = (time.monotonic() - t0) * 1000
        except ImportError:
            raise RuntimeError(
                "faster-whisper is not installed.\n"
                "Install it with:  pip install faster-whisper\n"
                "Or:               uv pip install -e '.[voice]'"
            )

    def transcribe(self, audio: AudioFrame) -> str:
        """Transcribe audio to text using faster-whisper."""
        if not audio.samples:
            return ""

        t0 = time.monotonic()
        try:
            self._load()
            import numpy as np

            # Convert PCM16 bytes to float32 numpy array (required by whisper)
            audio_array = (
                np.frombuffer(audio.samples, dtype=np.int16).astype(np.float32)
                / 32768.0
            )

            segments, info = self._model.transcribe(
                audio_array,
                language=self._language,
                vad_filter=True,            # filter out silence
                vad_parameters={"min_silence_duration_ms": 500},
            )

            text = " ".join(seg.text.strip() for seg in segments if seg.text.strip())
            latency_ms = (time.monotonic() - t0) * 1000

            return text

        except Exception as exc:
            # STT must never crash the voice pipeline
            return ""

    @property
    def provider_name(self) -> str:
        return f"FasterWhisperProvider(model={self._model_size}, device={self._device})"

    @property
    def model_loaded(self) -> bool:
        return self._model is not None

    @property
    def load_time_ms(self) -> float:
        return self._load_time_ms


# ── Mock (for tests) ───────────────────────────────────────────────────────────


class MockSTTProvider(STTProvider):
    """
    Scripted STT provider for deterministic testing.

    Returns pre-programmed transcriptions in order.
    When exhausted, returns "goodbye".

    Usage:
        stt = MockSTTProvider(["Where is my order?", "Schedule a callback."])
        text = stt.transcribe(audio_frame)   # -> "Where is my order?"
    """

    def __init__(self, responses: list[str] | None = None) -> None:
        self._responses: list[str] = list(
            responses or ["Hello, I need help with my order."]
        )
        self._index: int = 0
        self.call_count: int = 0
        self.last_audio: AudioFrame | None = None

    def queue(self, text: str) -> None:
        """Append a scripted response."""
        self._responses.append(text)

    def transcribe(self, audio: AudioFrame) -> str:
        self.call_count += 1
        self.last_audio = audio

        if self._index < len(self._responses):
            text = self._responses[self._index]
            self._index += 1
            return text
        return "goodbye"

    def reset(self) -> None:
        self._index = 0
        self.call_count = 0

    @property
    def provider_name(self) -> str:
        return f"MockSTTProvider(queued={len(self._responses)}, used={self._index})"


# ── Factory ────────────────────────────────────────────────────────────────────


def build_stt_provider(
    provider_name: str | None = None,
    mock_responses: list[str] | None = None,
) -> STTProvider:
    """
    Build an STTProvider from configuration.

    Args:
        provider_name: Override ('faster_whisper' | 'mock').
        mock_responses: Pre-programmed responses for MockSTTProvider.

    Returns:
        An STTProvider ready to use.
    """
    from app.config import settings

    name = provider_name or getattr(settings, "stt_provider", "faster_whisper")

    if name == "mock":
        return MockSTTProvider(responses=mock_responses)

    if name in ("faster_whisper", "whisper"):
        return FasterWhisperProvider()

    raise ValueError(
        f"Unknown STT provider: '{name}'. Choose 'faster_whisper' or 'mock'."
    )
