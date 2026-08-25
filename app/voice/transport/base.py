"""
VoiceTransport -- abstract base for all audio transport implementations.

The Agent Core is COMPLETELY decoupled from any transport.
Only VoiceSession knows about transports.

Implementations:
    LocalTransport      -- microphone + speaker via sounddevice
    WebSocketTransport  -- browser audio via WebSocket
    SIPTransport        -- SIP/VoIP via pjsua2 or equivalent [STUB]
    GSMTransport        -- GSM gateway [STUB]

Transport contract:
    - start_session()               -- open/initialize the transport
    - receive_audio() -> AudioFrame -- block until audio available
    - send_audio(AudioFrame)        -- play/stream audio to the caller
    - end_session()                 -- clean up
    - get_status() -> dict          -- current transport state
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from enum import Enum

from app.voice.audio import AudioFrame


class TransportStatus(str, Enum):
    IDLE = "idle"
    ACTIVE = "active"
    ENDED = "ended"
    ERROR = "error"


class VoiceTransport(ABC):
    """
    Abstract voice transport.

    Responsible for:
        - Receiving audio input from any source (mic, WebSocket, SIP, GSM)
        - Sending audio output to any destination
        - Session lifecycle management

    Does NOT:
        - Understand the conversation
        - Call the agent
        - Process STT/TTS
    """

    @abstractmethod
    def start_session(self) -> None:
        """Initialize the transport and open audio streams."""

    @abstractmethod
    def receive_audio(self, timeout: float | None = None) -> AudioFrame | None:
        """
        Block until an audio frame is available.

        Args:
            timeout: Max seconds to wait. None = wait forever.

        Returns:
            AudioFrame in internal format, or None on timeout/end.
        """

    @abstractmethod
    def send_audio(self, audio: AudioFrame) -> None:
        """
        Send audio to the caller.

        Args:
            audio: AudioFrame in internal format.
        """

    @abstractmethod
    def end_session(self) -> None:
        """Close audio streams and release resources."""

    @abstractmethod
    def get_status(self) -> dict:
        """Return current transport status as a dict."""

    @property
    def transport_name(self) -> str:
        return type(self).__name__
