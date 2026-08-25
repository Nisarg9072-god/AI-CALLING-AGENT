"""
WebSocketTransport -- audio transport over WebSocket.

Receives audio from a browser or client via WebSocket messages.
Sends synthesized audio back to the client.

Protocol (JSON messages):
    Client -> Server:
        {"type": "audio_chunk", "data": "<base64 PCM16>", "sample_rate": 16000}
        {"type": "audio_end"}           -- signals end of user utterance
        {"type": "ping"}

    Server -> Client:
        {"type": "session_started", "session_id": "..."}
        {"type": "transcription", "text": "..."}
        {"type": "agent_decision", "action": "...", "tool_name": "..."}
        {"type": "tool_started", "tool_name": "..."}
        {"type": "tool_result", "tool_name": "...", "success": true, "data": {...}}
        {"type": "agent_response", "text": "..."}
        {"type": "audio", "data": "<base64 PCM16>", "sample_rate": 16000}
        {"type": "error", "message": "..."}
        {"type": "session_ended", "reason": "..."}

Audio format:
    PCM16, mono, 16kHz, base64-encoded
"""

from __future__ import annotations

import asyncio
import base64
import json
from typing import Any, Callable, Awaitable

from app.voice.audio import (
    AudioFrame,
    AudioConverter,
    INTERNAL_SAMPLE_RATE,
    INTERNAL_SAMPLE_WIDTH,
    INTERNAL_CHANNELS,
    concat_frames,
)
from app.voice.transport.base import VoiceTransport, TransportStatus


# Async callback type for sending messages to the client
SendCallback = Callable[[dict], Awaitable[None]]


class WebSocketTransport(VoiceTransport):
    """
    WebSocket-based audio transport for browser voice.

    This is the async transport used by the WebSocket endpoint.
    It is NOT a standard blocking VoiceTransport -- it works with
    asyncio queues and callbacks for the WebSocket handler.

    Usage (inside a FastAPI WebSocket handler):
        transport = WebSocketTransport(send_fn=ws.send_json)
        await transport.handle_message(data)  # called for each incoming WS message
        audio_frame = await transport.get_utterance()  # get complete utterance
    """

    def __init__(self, send_fn: SendCallback) -> None:
        self._send_fn = send_fn
        self._audio_chunks: list[AudioFrame] = []
        self._utterance_ready: asyncio.Event = asyncio.Event()
        self._complete_utterance: AudioFrame | None = None
        self._status = TransportStatus.IDLE

    async def handle_message(self, message: dict) -> None:
        """
        Process an incoming WebSocket message from the client.
        Called by the WebSocket handler for each received message.
        """
        msg_type = message.get("type", "")

        if msg_type == "audio_chunk":
            # Accumulate audio chunks
            data_b64 = message.get("data", "")
            if not data_b64:
                return
            try:
                raw_bytes = base64.b64decode(data_b64)
                sample_rate = message.get("sample_rate", INTERNAL_SAMPLE_RATE)
                frame = AudioFrame(
                    samples=raw_bytes,
                    sample_rate=sample_rate,
                    channels=INTERNAL_CHANNELS,
                    sample_width=INTERNAL_SAMPLE_WIDTH,
                )
                # Normalize to internal format
                frame = AudioConverter.to_internal(frame)
                self._audio_chunks.append(frame)
            except Exception:
                pass

        elif msg_type == "audio_end":
            # User finished speaking -- combine all chunks into one utterance
            if self._audio_chunks:
                combined = concat_frames(self._audio_chunks)
                self._audio_chunks.clear()
                self._complete_utterance = combined
                self._utterance_ready.set()
            else:
                self._complete_utterance = None
                self._utterance_ready.set()

        elif msg_type == "ping":
            await self._send_fn({"type": "pong"})

    async def get_utterance(self, timeout: float = 30.0) -> AudioFrame | None:
        """
        Wait for a complete utterance from the client.

        Returns the complete audio frame, or None on timeout.
        """
        self._utterance_ready.clear()
        try:
            await asyncio.wait_for(self._utterance_ready.wait(), timeout=timeout)
        except asyncio.TimeoutError:
            return None
        result = self._complete_utterance
        self._complete_utterance = None
        return result

    async def send_event(self, event_type: str, **kwargs: Any) -> None:
        """Send a structured event to the client."""
        try:
            await self._send_fn({"type": event_type, **kwargs})
        except Exception:
            pass

    async def send_audio_frame(self, audio: AudioFrame) -> None:
        """Send an audio frame to the client as base64 PCM16."""
        try:
            encoded = base64.b64encode(audio.samples).decode("ascii")
            await self._send_fn({
                "type": "audio",
                "data": encoded,
                "sample_rate": audio.sample_rate,
                "channels": audio.channels,
                "format": "pcm16",
            })
        except Exception:
            pass

    # ── VoiceTransport interface (sync wrappers -- not used in async mode) ─────

    def start_session(self) -> None:
        self._status = TransportStatus.ACTIVE

    def receive_audio(self, timeout: float | None = None) -> AudioFrame | None:
        # Not used in async WebSocket mode -- use get_utterance() instead
        raise NotImplementedError("Use get_utterance() for WebSocket transport")

    def send_audio(self, audio: AudioFrame) -> None:
        # Not used in async WebSocket mode -- use send_audio_frame() instead
        raise NotImplementedError("Use send_audio_frame() for WebSocket transport")

    def end_session(self) -> None:
        self._status = TransportStatus.ENDED
        self._utterance_ready.set()   # unblock any waiting coroutines

    def get_status(self) -> dict:
        return {
            "transport": "websocket",
            "status": self._status.value,
            "chunks_buffered": len(self._audio_chunks),
        }
