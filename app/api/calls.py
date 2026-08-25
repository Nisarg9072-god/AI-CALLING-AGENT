"""
Calls API -- REST endpoints for managing voice call sessions.

POST   /calls           -- create a session (phone number → session)
GET    /calls/{id}      -- get session state
DELETE /calls/{id}      -- end a session
GET    /calls           -- list active sessions
"""

from __future__ import annotations

import re
import uuid
from typing import Any

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, field_validator

from app.voice.session import CallSession, SessionStatus
from app.api.voice_ws import get_session_store

router = APIRouter(prefix="/calls", tags=["calls"])

# ── E.164 validation ────────────────────────────────────────────────────────────
E164_PATTERN = re.compile(r"^\+[1-9]\d{6,14}$")


def validate_e164(phone: str) -> str:
    """Validate and normalize E.164 phone number."""
    normalized = re.sub(r"[\s\-\(\)]", "", phone.strip())
    if not E164_PATTERN.match(normalized):
        raise ValueError(
            f"Invalid phone number: {phone!r}. "
            "Must be in E.164 format, e.g. +919537266092 or +15551234567"
        )
    return normalized


# ── Request/Response models ────────────────────────────────────────────────────


class CreateCallRequest(BaseModel):
    phone_number: str
    transport: str = "local"    # local | websocket | sip | gsm

    @field_validator("phone_number")
    @classmethod
    def validate_phone(cls, v: str) -> str:
        return validate_e164(v)

    @field_validator("transport")
    @classmethod
    def validate_transport(cls, v: str) -> str:
        allowed = {"local", "websocket", "sip", "gsm"}
        if v not in allowed:
            raise ValueError(f"transport must be one of {sorted(allowed)}")
        return v


class CreateCallResponse(BaseModel):
    call_id: str
    session_id: str
    status: str
    transport: str
    phone_number: str
    websocket_url: str
    message: str


class CallStateResponse(BaseModel):
    call_id: str
    session_id: str
    phone_number: str
    transport: str
    status: str
    started_at: str
    ended_at: str | None
    duration_seconds: float | None
    outcome: str | None
    termination_reason: str | None
    tool_calls_made: int
    iterations: int
    transcript: list[dict[str, str]]


# ── Endpoints ──────────────────────────────────────────────────────────────────


@router.post("", response_model=CreateCallResponse, status_code=201)
async def create_call(request: CreateCallRequest) -> CreateCallResponse:
    """
    Create a new voice call session.

    When VOICE_TRANSPORT=local or transport='local':
        - No real phone call is made
        - A local voice session is created
        - Connect via the returned WebSocket URL to interact

    When transport='sip' or 'gsm':
        - Requires configured external infrastructure
        - See docs/sip.md or docs/gsm.md
    """
    from app.config import settings

    session = CallSession(
        session_id=str(uuid.uuid4()),
        call_id=str(uuid.uuid4()),
        phone_number=request.phone_number,
        transport=request.transport,
        status=SessionStatus.CREATED,
    )

    # Validate transport is available
    if request.transport in ("sip", "gsm"):
        # Warn but don't block -- let the WS connection fail gracefully
        pass

    store = get_session_store()
    store[session.session_id] = session

    ws_url = f"ws://localhost:{settings.port}/ws/voice/{session.session_id}"

    return CreateCallResponse(
        call_id=session.call_id,
        session_id=session.session_id,
        status=session.status.value,
        transport=session.transport,
        phone_number=session.phone_number,
        websocket_url=ws_url,
        message=(
            f"Session created. "
            f"Connect to {ws_url} to start the voice conversation."
            if request.transport != "local"
            else f"Local session created (no real phone call). "
                 f"Connect to {ws_url} or use python -m cli.voice_test"
        ),
    )


@router.get("/{session_id}", response_model=CallStateResponse)
async def get_call(session_id: str) -> CallStateResponse:
    """Get the current state of a call session."""
    store = get_session_store()
    session = store.get(session_id)
    if session is None:
        raise HTTPException(status_code=404, detail=f"Session '{session_id}' not found.")

    return CallStateResponse(
        call_id=session.call_id,
        session_id=session.session_id,
        phone_number=session.phone_number,
        transport=session.transport,
        status=session.status.value,
        started_at=session.started_at.isoformat(),
        ended_at=session.ended_at.isoformat() if session.ended_at else None,
        duration_seconds=session.duration_seconds,
        outcome=session.outcome,
        termination_reason=session.termination_reason,
        tool_calls_made=session.tool_calls_made,
        iterations=session.iterations,
        transcript=session.transcript,
    )


@router.delete("/{session_id}", status_code=204)
async def end_call(session_id: str) -> None:
    """End an active call session."""
    store = get_session_store()
    session = store.get(session_id)
    if session is None:
        raise HTTPException(status_code=404, detail=f"Session '{session_id}' not found.")

    if session.status not in (SessionStatus.COMPLETED, SessionStatus.FAILED):
        session.end(SessionStatus.COMPLETED, "ended_by_api")

    from app.api.voice_ws import get_voice_session
    voice_session = get_voice_session(session_id)
    if voice_session:
        voice_session.call_state.terminate(
            __import__("app.agent.state", fromlist=["TerminationReason"]).TerminationReason.AGENT_END,
            outcome="ended by API"
        )


@router.get("", response_model=list[dict])
async def list_calls() -> list[dict]:
    """List all active and recent call sessions."""
    store = get_session_store()
    return [s.to_dict() for s in store.values()]
