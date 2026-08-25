"""
WebSocket voice endpoint: /ws/voice/{session_id}

Protocol:
    Client connects to ws://host/ws/voice/{session_id}
    Client sends audio chunks as base64 PCM16
    Server sends structured JSON events

Message types (client → server):
    audio_chunk  -- audio data (base64 PCM16)
    audio_end    -- signals user finished speaking
    ping         -- keepalive

Message types (server → client):
    session_started   -- session is active
    transcription     -- user speech recognized
    agent_observation -- agent is observing
    agent_decision    -- agent decided action
    tool_started      -- tool call initiated
    tool_result       -- tool call completed
    agent_response    -- agent text response
    audio             -- synthesized speech (base64 PCM16)
    error             -- error occurred
    session_ended     -- call ended
"""

from __future__ import annotations

import asyncio
import base64
import json
import time
import uuid
from typing import Any

from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from app.voice.session import CallSession, SessionStatus, VoiceSession
from app.voice.transport.websocket_transport import WebSocketTransport
from app.voice.stt import build_stt_provider
from app.voice.tts import build_tts_provider

router = APIRouter(tags=["voice-websocket"])

# In-memory session store (keyed by session_id)
# In production, use Redis or a DB
_sessions: dict[str, CallSession] = {}
_voice_sessions: dict[str, VoiceSession] = {}


def get_session_store() -> dict[str, CallSession]:
    return _sessions


def get_voice_session(session_id: str) -> VoiceSession | None:
    return _voice_sessions.get(session_id)


@router.websocket("/ws/voice/{session_id}")
async def websocket_voice_endpoint(websocket: WebSocket, session_id: str) -> None:
    """
    WebSocket voice endpoint.

    The client connects here after creating a session via POST /calls.
    Audio flows:
        Browser mic → base64 PCM16 → WebSocket → STT → Agent → TTS → base64 PCM16 → WebSocket → Browser speaker
    """
    await websocket.accept()

    # Get or create call session
    session = _sessions.get(session_id)
    if session is None:
        # Auto-create for direct WS connections (e.g., browser demo)
        session = CallSession(
            session_id=session_id,
            call_id=str(uuid.uuid4()),
            transport="websocket",
        )
        _sessions[session_id] = session

    # Build WebSocket transport with send callback
    async def send_json(data: dict) -> None:
        try:
            await websocket.send_json(data)
        except Exception:
            pass

    transport = WebSocketTransport(send_fn=send_json)
    transport.start_session()

    # Build STT and TTS providers
    stt = build_stt_provider()
    tts = build_tts_provider()

    # Event callback: relay agent events to client
    def on_event(event_type: str, payload: dict) -> None:
        asyncio.ensure_future(send_json({
            "type": event_type.lower(),
            **_sanitize_payload(payload),
        }))

    # Build VoiceSession (uses the same AgentLoop as the text CLI)
    voice_session = VoiceSession(
        session=session,
        stt=stt,
        tts=tts,
        on_event=on_event,
    )
    _voice_sessions[session_id] = voice_session
    session.status = SessionStatus.ACTIVE

    # Notify client: session ready
    await send_json({
        "type": "session_started",
        "session_id": session_id,
        "call_id": session.call_id,
        "phone_number": session.phone_number,
    })

    # Main WebSocket loop
    try:
        while not voice_session.is_finished:
            # Receive message from client
            try:
                raw = await asyncio.wait_for(websocket.receive_json(), timeout=60.0)
            except asyncio.TimeoutError:
                # Timeout -- ping client
                await send_json({"type": "ping"})
                continue
            except WebSocketDisconnect:
                break

            msg_type = raw.get("type", "")

            if msg_type == "audio_chunk":
                # Accumulate audio from browser
                await transport.handle_message(raw)

            elif msg_type == "audio_end":
                # User finished speaking -- process the utterance
                await transport.handle_message(raw)
                session.status = SessionStatus.PROCESSING

                audio_frame = await transport.get_utterance(timeout=5.0)
                if audio_frame is None or not audio_frame.samples:
                    await send_json({"type": "transcription", "text": ""})
                    continue

                # Run full pipeline in executor (CPU-bound STT/TTS)
                loop = asyncio.get_event_loop()

                user_text, agent_text, audio_out = await loop.run_in_executor(
                    None,
                    lambda: voice_session.process_audio(audio_frame),
                )

                # Send transcription
                await send_json({"type": "transcription", "text": user_text})

                if not user_text.strip():
                    continue

                # Send agent response text
                if agent_text:
                    await send_json({"type": "agent_response", "text": agent_text})

                # Send synthesized audio
                if audio_out:
                    encoded = base64.b64encode(audio_out).decode("ascii")
                    await send_json({
                        "type": "audio",
                        "data": encoded,
                        "sample_rate": 16000,
                        "format": "pcm16",
                    })

                # Check if agent ended the call
                if voice_session.is_finished:
                    break

            elif msg_type == "text_message":
                # Text-only message (no audio -- for testing)
                user_text = raw.get("text", "").strip()
                if not user_text:
                    continue

                session.status = SessionStatus.PROCESSING
                loop = asyncio.get_event_loop()
                agent_text = await loop.run_in_executor(
                    None,
                    lambda: voice_session.process_text(user_text),
                )

                await send_json({"type": "transcription", "text": user_text})
                if agent_text:
                    await send_json({"type": "agent_response", "text": agent_text})

                    # TTS for text messages too
                    try:
                        audio_frame_out = await loop.run_in_executor(
                            None, lambda: tts.synthesize(agent_text)
                        )
                        if audio_frame_out.samples:
                            encoded = base64.b64encode(audio_frame_out.samples).decode("ascii")
                            await send_json({
                                "type": "audio",
                                "data": encoded,
                                "sample_rate": audio_frame_out.sample_rate,
                                "format": "pcm16",
                            })
                    except Exception:
                        pass

                if voice_session.is_finished:
                    break

            elif msg_type == "end_session":
                break

            elif msg_type == "ping":
                await send_json({"type": "pong"})

    except WebSocketDisconnect:
        pass
    except Exception as exc:
        try:
            await send_json({"type": "error", "message": str(exc)})
        except Exception:
            pass
    finally:
        transport.end_session()
        reason = (
            voice_session.call_state.termination_reason.value
            if voice_session.call_state.termination_reason else "completed"
        )
        session.end(SessionStatus.COMPLETED, reason)

        try:
            await send_json({
                "type": "session_ended",
                "reason": reason,
                "turns": voice_session.call_state.current_iteration,
                "tool_calls": voice_session.call_state.tool_call_count,
            })
        except Exception:
            pass


def _sanitize_payload(payload: dict) -> dict:
    """Remove large/internal fields from event payloads sent to client."""
    exclude = {"call_id", "session_id"}
    return {k: v for k, v in payload.items() if k not in exclude and _is_serializable(v)}


def _is_serializable(value: Any) -> bool:
    try:
        json.dumps(value)
        return True
    except (TypeError, ValueError):
        return False
