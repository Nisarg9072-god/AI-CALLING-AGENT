"""
Voice Pipeline — Phase 13-15.

Abstractions for STT (Speech-to-Text) and TTS (Text-to-Speech).

Phase 13: faster-whisper STT (local, free)
Phase 14: Piper TTS (local, free)
Phase 15: Full local voice pipeline (mic → STT → AgentLoop → TTS → speaker)

Installation for voice mode:
  uv pip install -e ".[voice]"   # installs faster-whisper + sounddevice
  # Download Piper voice model:
  # https://github.com/rhasspy/piper/releases → en_US-amy-medium.onnx
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from app.config import settings


# ── Abstract interfaces ────────────────────────────────────────────────────────


class STTProvider(ABC):
    @abstractmethod
    def transcribe(self, audio_bytes: bytes, sample_rate: int = 16000) -> str:
        """Convert audio bytes to text. Returns empty string on failure."""


class TTSProvider(ABC):
    @abstractmethod
    def synthesize(self, text: str) -> bytes:
        """Convert text to audio bytes (WAV format). Returns empty bytes on failure."""


# ── faster-whisper STT (Phase 13) ─────────────────────────────────────────────


class FasterWhisperSTT(STTProvider):
    """
    Local STT using faster-whisper (free, runs on CPU or CUDA).

    Install: uv pip install -e ".[voice]"
    """

    def __init__(self, model_size: str | None = None, device: str | None = None) -> None:
        self._model_size = model_size or settings.stt_model
        self._device = device or settings.stt_device
        self._model = None  # lazy loaded on first use

    def _load(self):
        if self._model is None:
            try:
                from faster_whisper import WhisperModel
                self._model = WhisperModel(
                    self._model_size,
                    device=self._device,
                    compute_type="int8",
                )
            except ImportError:
                raise RuntimeError(
                    "faster-whisper not installed. Run: uv pip install -e '.[voice]'"
                )

    def transcribe(self, audio_bytes: bytes, sample_rate: int = 16000) -> str:
        self._load()
        import io
        import numpy as np
        audio_array = np.frombuffer(audio_bytes, dtype=np.int16).astype(np.float32) / 32768.0
        segments, _ = self._model.transcribe(audio_array, language="en")
        return " ".join(s.text.strip() for s in segments)


# ── Piper TTS (Phase 14) ───────────────────────────────────────────────────────


class PiperTTS(TTSProvider):
    """
    Local TTS using Piper (free, runs on CPU).

    Install piper-tts binary and download a voice model:
    https://github.com/rhasspy/piper/releases
    """

    def __init__(self, model_path: str | None = None) -> None:
        self._model_path = model_path or settings.piper_model_path

    def synthesize(self, text: str) -> bytes:
        import subprocess
        import tempfile
        import os

        # Write text to temp file, pipe through piper binary
        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as tf:
            tf.write(text)
            tf_path = tf.name

        try:
            result = subprocess.run(
                ["piper", "--model", self._model_path, "--output_raw"],
                input=text.encode(),
                capture_output=True,
                timeout=30,
            )
            if result.returncode != 0:
                raise RuntimeError(f"Piper failed: {result.stderr.decode()}")
            return result.stdout
        except FileNotFoundError:
            raise RuntimeError(
                "Piper TTS binary not found. "
                "Download from https://github.com/rhasspy/piper/releases"
            )
        finally:
            os.unlink(tf_path)


# ── Mock providers (for testing) ───────────────────────────────────────────────


class MockSTT(STTProvider):
    """Scripted STT for tests — returns predefined transcriptions."""

    def __init__(self, responses: list[str] | None = None) -> None:
        self._responses = list(responses or ["Hello", "I need help with my order"])
        self._index = 0

    def transcribe(self, audio_bytes: bytes, sample_rate: int = 16000) -> str:
        if self._index < len(self._responses):
            text = self._responses[self._index]
            self._index += 1
            return text
        return "goodbye"


class MockTTS(TTSProvider):
    """No-op TTS for tests — records what was synthesized."""

    def __init__(self) -> None:
        self.spoken: list[str] = []

    def synthesize(self, text: str) -> bytes:
        self.spoken.append(text)
        return b""   # empty audio — tests don't need real audio


# ── Local voice pipeline (Phase 15) ───────────────────────────────────────────


class LocalVoicePipeline:
    """
    Full local voice loop (Phase 15):
    Microphone → faster-whisper → AgentLoop → Piper → Speaker

    Usage:
        pipeline = LocalVoicePipeline(agent_loop, state)
        pipeline.run()   # blocking — listens and responds until call ends
    """

    def __init__(
        self,
        loop,           # AgentLoop
        state,          # CallState
        stt: STTProvider | None = None,
        tts: TTSProvider | None = None,
    ) -> None:
        from app.agent.loop import AgentLoop
        from app.agent.state import CallState
        self._loop: AgentLoop = loop
        self._state: CallState = state
        self._stt = stt or FasterWhisperSTT()
        self._tts = tts or PiperTTS()

    def run(self) -> None:
        """
        Main voice loop.
        Phase 15 full implementation: capture mic → transcribe → run agent → speak.
        Requires: sounddevice, faster-whisper, piper-tts
        """
        try:
            import sounddevice as sd
        except ImportError:
            raise RuntimeError(
                "sounddevice not installed. Run: uv pip install -e '.[voice]'"
            )

        from app.agent.state import MessageRole
        print(">>> Voice pipeline started. Speak to the agent.")
        print("    Press Ctrl+C to end the call.")

        while not self._state.finished:
            # Record audio chunk (3 seconds)
            sample_rate = 16000
            duration = 3
            audio = sd.rec(
                int(duration * sample_rate),
                samplerate=sample_rate,
                channels=1,
                dtype="int16",
            )
            sd.wait()
            audio_bytes = audio.tobytes()

            # STT
            user_text = self._stt.transcribe(audio_bytes, sample_rate)
            if not user_text.strip():
                continue

            print(f"[YOU]   {user_text}")
            self._state.current_user_message = user_text
            self._state.add_message(MessageRole.USER, user_text)

            # Run agent loop
            self._loop.run(self._state)

            # TTS — speak the agent response
            agent_text = self._state.current_agent_message
            if agent_text:
                print(f"[AGENT] {agent_text}")
                audio_out = self._tts.synthesize(agent_text)
                if audio_out:
                    import numpy as np
                    import io
                    arr = np.frombuffer(audio_out, dtype=np.int16)
                    sd.play(arr, samplerate=22050)
                    sd.wait()


def build_voice_pipeline(loop, state, mock: bool = False):
    """Factory for voice pipeline."""
    if mock:
        return LocalVoicePipeline(loop, state, stt=MockSTT(), tts=MockTTS())
    return LocalVoicePipeline(loop, state)
