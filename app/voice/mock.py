"""
Voice layer — abstract interfaces and mock implementations.

In the CLI demo, MockCallProvider wraps stdin/stdout.
In production, swap in TwilioCallProvider without changing the agent core.
"""

from __future__ import annotations

from abc import ABC, abstractmethod


# ── Abstract interfaces ────────────────────────────────────────────────────────


class STTProvider(ABC):
    """Speech-to-Text: converts audio bytes → text."""

    @abstractmethod
    def transcribe(self, audio_bytes: bytes) -> str: ...


class TTSProvider(ABC):
    """Text-to-Speech: converts text → audio bytes."""

    @abstractmethod
    def synthesize(self, text: str) -> bytes: ...


class CallProvider(ABC):
    """Manages the call transport layer."""

    @abstractmethod
    def start_call(self) -> None: ...

    @abstractmethod
    def receive_input(self) -> str:
        """Receive customer input (audio → transcribed text)."""
        ...

    @abstractmethod
    def send_output(self, text: str) -> None:
        """Send agent response (text → TTS → audio)."""
        ...

    @abstractmethod
    def end_call(self) -> None: ...


# ── Mock implementations (used by CLI) ────────────────────────────────────────


class MockSTTProvider(STTProvider):
    """Passthrough — returns text directly (no audio processing)."""

    def transcribe(self, audio_bytes: bytes) -> str:
        return audio_bytes.decode("utf-8", errors="replace")


class MockTTSProvider(TTSProvider):
    """Prints text — no audio synthesis."""

    def synthesize(self, text: str) -> bytes:
        return text.encode("utf-8")


class MockCallProvider(CallProvider):
    """
    Wraps CLI stdin/stdout as a call transport.
    Used for local demo and testing.
    """

    def __init__(
        self,
        stt: STTProvider | None = None,
        tts: TTSProvider | None = None,
    ) -> None:
        self._stt = stt or MockSTTProvider()
        self._tts = tts or MockTTSProvider()

    def start_call(self) -> None:
        pass  # No-op for CLI

    def receive_input(self) -> str:
        """Read from stdin — simulates customer audio input."""
        try:
            return input().strip()
        except (EOFError, KeyboardInterrupt):
            return ""

    def send_output(self, text: str) -> None:
        """Write to stdout — simulates TTS audio output."""
        pass  # CLI handles display via Rich; this is a no-op

    def end_call(self) -> None:
        pass  # No-op for CLI
