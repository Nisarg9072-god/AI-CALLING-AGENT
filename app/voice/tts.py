"""
TTS (Text-to-Speech) providers.

Interface:  TTSProvider.synthesize(text) -> AudioFrame
Impl:       PiperProvider -- local, free, no cloud
Mock:       MockTTSProvider -- silent audio for tests

Configuration (.env):
    PIPER_MODEL_PATH=./data/voices/en_US-amy-medium.onnx
    PIPER_VOICE=en_US-amy-medium   # alternative to model path

Install Piper:
    pip install piper-tts
    # Then download a voice model:
    # https://github.com/rhasspy/piper/releases
    # Example:
    #   mkdir -p data/voices
    #   wget -O data/voices/en_US-amy-medium.onnx \\
    #     https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_US/amy/medium/en_US-amy-medium.onnx
"""

from __future__ import annotations

import io
import time
import wave
from abc import ABC, abstractmethod

from app.voice.audio import AudioFrame, INTERNAL_SAMPLE_RATE


# ── Abstract interface ─────────────────────────────────────────────────────────


class TTSProvider(ABC):
    """
    Text-to-Speech provider interface.

    The agent core depends ONLY on this interface.
    """

    @abstractmethod
    def synthesize(self, text: str) -> AudioFrame:
        """
        Convert text to spoken audio.

        Args:
            text: Text to speak.

        Returns:
            AudioFrame in internal format (PCM mono 16kHz 16-bit).
            Returns AudioFrame.silence() on any error -- never raises.
        """

    @property
    def provider_name(self) -> str:
        return type(self).__name__


# ── Piper TTS (local, free) ────────────────────────────────────────────────────


class PiperProvider(TTSProvider):
    """
    Local TTS using Piper (https://github.com/rhasspy/piper).

    Fully local -- no audio ever sent to an external API.
    Runs on CPU, produces high-quality natural speech.

    Two installation modes:
      1. piper-tts Python package (preferred):
             pip install piper-tts
      2. Piper binary (fallback):
             Download from https://github.com/rhasspy/piper/releases
             Place 'piper' binary in your PATH

    Voice models:
        Download from https://github.com/rhasspy/piper/blob/master/VOICES.md
        Place .onnx + .onnx.json in ./data/voices/

    Configuration:
        PIPER_MODEL_PATH=./data/voices/en_US-amy-medium.onnx
    """

    def __init__(self, model_path: str | None = None) -> None:
        from app.config import settings
        self._model_path = model_path or settings.piper_model_path
        self._voice = None      # lazy-loaded piper_tts.PiperVoice
        self._sample_rate: int | None = None

    def _load(self) -> None:
        """Lazy-load the Piper voice model."""
        if self._voice is not None:
            return

        import os
        if not os.path.exists(self._model_path):
            raise RuntimeError(
                f"Piper voice model not found: {self._model_path!r}\n"
                "Download from https://github.com/rhasspy/piper/releases\n"
                "Example:\n"
                "  mkdir -p data/voices\n"
                "  # Download en_US-amy-medium.onnx + en_US-amy-medium.onnx.json\n"
                "  # Place both in ./data/voices/"
            )

        try:
            from piper.voice import PiperVoice
            self._voice = PiperVoice.load(self._model_path)
            self._sample_rate = self._voice.config.sample_rate
        except ImportError:
            self._voice = None      # fall through to subprocess mode
            self._sample_rate = 22050

    def synthesize(self, text: str) -> AudioFrame:
        """Synthesize speech from text using Piper."""
        if not text.strip():
            return AudioFrame.silence(0.1)

        t0 = time.monotonic()
        try:
            self._load()

            if self._voice is not None:
                # Python package mode (preferred)
                return self._synthesize_python(text)
            else:
                # Subprocess fallback
                return self._synthesize_subprocess(text)

        except RuntimeError:
            raise       # re-raise model not found errors
        except Exception:
            return AudioFrame.silence(0.5)

    def _synthesize_python(self, text: str) -> AudioFrame:
        """Use piper-tts Python package directly."""
        buf = io.BytesIO()
        with wave.open(buf, "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)   # 16-bit
            wf.setframerate(self._sample_rate)
            self._voice.synthesize(text, wf)
        buf.seek(0)
        return AudioFrame.from_wav_bytes(buf.read())

    def _synthesize_subprocess(self, text: str) -> AudioFrame:
        """Use the piper binary via subprocess (fallback)."""
        import subprocess
        result = subprocess.run(
            ["piper", "--model", self._model_path, "--output_raw"],
            input=text.encode("utf-8"),
            capture_output=True,
            timeout=30,
        )
        if result.returncode != 0:
            raise RuntimeError(f"Piper binary failed: {result.stderr.decode()}")

        # Raw PCM from piper is 16kHz 16-bit mono
        from app.voice.audio import AudioConverter
        frame = AudioFrame(
            samples=result.stdout,
            sample_rate=self._sample_rate or 22050,
            channels=1,
            sample_width=2,
        )
        return AudioConverter.to_internal(frame)

    @property
    def provider_name(self) -> str:
        return f"PiperProvider(model={self._model_path!r})"

    @property
    def model_loaded(self) -> bool:
        return self._voice is not None


# ── Mock (for tests) ───────────────────────────────────────────────────────────


class MockTTSProvider(TTSProvider):
    """
    No-op TTS for tests.

    Records what was synthesized. Returns silent audio frames.
    """

    def __init__(self) -> None:
        self.spoken: list[str] = []
        self.call_count: int = 0

    def synthesize(self, text: str) -> AudioFrame:
        self.spoken.append(text)
        self.call_count += 1
        # Return 0.5s of silence -- real enough for tests
        return AudioFrame.silence(0.5)

    def reset(self) -> None:
        self.spoken.clear()
        self.call_count = 0

    @property
    def last_spoken(self) -> str | None:
        return self.spoken[-1] if self.spoken else None

    @property
    def provider_name(self) -> str:
        return f"MockTTSProvider(calls={self.call_count})"


# ── Factory ────────────────────────────────────────────────────────────────────


def build_tts_provider(
    provider_name: str | None = None,
) -> TTSProvider:
    """
    Build a TTSProvider from configuration.

    Args:
        provider_name: Override ('piper' | 'mock').

    Returns:
        A TTSProvider ready to use.
    """
    from app.config import settings

    name = provider_name or settings.tts_provider

    if name == "mock":
        return MockTTSProvider()

    if name == "piper":
        return PiperProvider()

    raise ValueError(
        f"Unknown TTS provider: '{name}'. Choose 'piper' or 'mock'."
    )
