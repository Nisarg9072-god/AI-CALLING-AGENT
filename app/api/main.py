"""
FastAPI application — Phase 16.

Endpoints:
  POST /calls/start        - Start a new call session
  POST /calls/{id}/message - Send a user message to an active session
  GET  /calls/{id}/state   - Get current call state
  POST /calls/{id}/end     - End a call session
  GET  /health             - Health check

For Twilio webhook handlers see app/api/webhooks.py (Phase 18).
"""

from __future__ import annotations

from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from app.agent.agent import Agent
from app.agent.loop import AgentLoop
from app.agent.state import CallState, MessageRole
from app.config import settings
from app.llm.factory import build_llm_provider
from app.observability.logger import make_event_callback
from app.tools.factory import build_tool_registry


# ── In-memory session store (Phase 4 replaces with SQLite) ────────────────────
_sessions: dict[str, CallState] = {}
_loops: dict[str, AgentLoop] = {}


# ── App lifecycle ──────────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: build registry once
    app.state.registry = build_tool_registry()
    app.state.llm = build_llm_provider()
    yield
    # Shutdown: flush any pending traces
    _sessions.clear()
    _loops.clear()


app = FastAPI(
    title="AI Calling Agent v2",
    description="Genuinely agentic AI calling system with Observe→Decide→Act loop",
    version="2.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Request/Response models ────────────────────────────────────────────────────


class StartCallRequest(BaseModel):
    phone_number: str
    customer_id: str | None = None
    initial_message: str = "hello"


class StartCallResponse(BaseModel):
    call_id: str
    message: str


class SendMessageRequest(BaseModel):
    message: str


class SendMessageResponse(BaseModel):
    agent_response: str
    finished: bool
    tool_calls: int
    turns: int


class CallStateResponse(BaseModel):
    call_id: str
    finished: bool
    turns: int
    tool_calls: int
    is_verified: bool
    outcome: str | None
    termination_reason: str | None


# ── Endpoints ──────────────────────────────────────────────────────────────────


@app.get("/health")
async def health() -> dict[str, Any]:
    return {
        "status": "ok",
        "version": "2.0.0",
        "llm_provider": settings.llm_provider,
        "mistral_configured": settings.mistral_configured,
        "telephony_provider": settings.telephony_provider,
        "active_sessions": len(_sessions),
    }


@app.post("/calls/start", response_model=StartCallResponse)
async def start_call(request: StartCallRequest) -> StartCallResponse:
    """Start a new call session. Returns call_id for subsequent requests."""
    state = CallState(
        phone_number=request.phone_number,
        customer_id=request.customer_id,
        max_iterations=settings.max_turns,
        max_tool_calls=settings.max_tool_calls,
    )
    state.current_user_message = request.initial_message
    state.add_message(MessageRole.USER, request.initial_message)

    registry = app.state.registry
    llm = app.state.llm

    logger, on_event = make_event_callback(state.call_id)
    agent = Agent(llm=llm, registry=registry)
    loop = AgentLoop(agent=agent, registry=registry, on_event=on_event)

    _sessions[state.call_id] = state
    _loops[state.call_id] = loop

    return StartCallResponse(
        call_id=state.call_id,
        message="Call session started. POST /calls/{call_id}/message to interact.",
    )


@app.post("/calls/{call_id}/message", response_model=SendMessageResponse)
async def send_message(call_id: str, request: SendMessageRequest) -> SendMessageResponse:
    """Send a user message and get the agent's response."""
    state = _sessions.get(call_id)
    loop = _loops.get(call_id)

    if not state or not loop:
        raise HTTPException(status_code=404, detail=f"Session '{call_id}' not found.")

    if state.finished:
        raise HTTPException(status_code=400, detail="Call session has already ended.")

    # Set user message and add to history
    state.current_user_message = request.message
    state.add_message(MessageRole.USER, request.message)

    # Run ONE pass of the loop (single-turn interaction)
    loop.run(state)

    return SendMessageResponse(
        agent_response=state.current_agent_message,
        finished=state.finished,
        tool_calls=state.tool_call_count,
        turns=state.current_iteration,
    )


@app.get("/calls/{call_id}/state", response_model=CallStateResponse)
async def get_call_state(call_id: str) -> CallStateResponse:
    """Get the current state of a call session."""
    state = _sessions.get(call_id)
    if not state:
        raise HTTPException(status_code=404, detail=f"Session '{call_id}' not found.")

    return CallStateResponse(
        call_id=state.call_id,
        finished=state.finished,
        turns=state.current_iteration,
        tool_calls=state.tool_call_count,
        is_verified=state.is_verified,
        outcome=state.outcome,
        termination_reason=state.termination_reason.value if state.termination_reason else None,
    )


@app.post("/calls/{call_id}/end")
async def end_call(call_id: str) -> dict[str, str]:
    """Forcefully end a call session."""
    state = _sessions.get(call_id)
    if not state:
        raise HTTPException(status_code=404, detail=f"Session '{call_id}' not found.")

    from app.agent.state import TerminationReason
    if not state.finished:
        state.terminate(TerminationReason.USER_HANGUP, "Session ended via API.")

    _sessions.pop(call_id, None)
    _loops.pop(call_id, None)

    return {"status": "ended", "call_id": call_id}
