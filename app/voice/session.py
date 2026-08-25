"""
VoiceSession and CallSession -- voice conversation lifecycle management.

VoiceSession:
    The single component that connects voice transport to Agent Core.
    It does NOT contain business logic -- only the pipeline:
        Audio in → STT → Agent → TTS → Audio out

CallSession:
    Tracks state/metadata for a call (call_id, phone, status, transcript).

SessionStatus:
    Enum of all possible call states.

The voice pipeline uses the SAME AgentLoop as the text CLI.
No VoiceAgent -- just a different input/output adapter.
"""

from __future__ import annotations

import asyncio
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Callable

from app.agent.loop import AgentLoop
from app.agent.state import CallState, MessageRole, TerminationReason
from app.agent.observation import build_observation
from app.harness.policies import ALL_TOOL_NAMES, VERIFICATION_REQUIRED_TOOLS
from app.llm.factory import build_llm_provider
from app.tools.registry import ToolRegistry


# ── SessionStatus ──────────────────────────────────────────────────────────────


class SessionStatus(str, Enum):
    CREATED = "CREATED"
    CONNECTING = "CONNECTING"
    ACTIVE = "ACTIVE"
    SPEAKING = "SPEAKING"      # Agent is speaking (TTS)
    LISTENING = "LISTENING"    # Waiting for user input
    PROCESSING = "PROCESSING"  # Agent loop running
    ENDING = "ENDING"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"


# ── CallSession ────────────────────────────────────────────────────────────────


@dataclass
class CallSession:
    """
    Metadata and state for a single call session.
    Serializable for API responses and persistence.
    """

    call_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    session_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    phone_number: str = ""
    transport: str = "local"
    status: SessionStatus = SessionStatus.CREATED
    started_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    ended_at: datetime | None = None
    duration_seconds: float | None = None
    transcript: list[dict[str, str]] = field(default_factory=list)
    outcome: str | None = None
    termination_reason: str | None = None
    tool_calls_made: int = 0
    iterations: int = 0

    def add_transcript_entry(self, role: str, text: str) -> None:
        self.transcript.append({
            "role": role,
            "text": text,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        })

    def end(self, status: SessionStatus, reason: str | None = None) -> None:
        self.status = status
        self.ended_at = datetime.now(timezone.utc)
        if self.ended_at and self.started_at:
            self.duration_seconds = (self.ended_at - self.started_at).total_seconds()
        self.termination_reason = reason

    def to_dict(self) -> dict[str, Any]:
        return {
            "call_id": self.call_id,
            "session_id": self.session_id,
            "phone_number": self.phone_number,
            "transport": self.transport,
            "status": self.status.value,
            "started_at": self.started_at.isoformat(),
            "ended_at": self.ended_at.isoformat() if self.ended_at else None,
            "duration_seconds": self.duration_seconds,
            "outcome": self.outcome,
            "termination_reason": self.termination_reason,
            "tool_calls_made": self.tool_calls_made,
            "iterations": self.iterations,
            "transcript": self.transcript,
        }


# ── VoiceSession ───────────────────────────────────────────────────────────────


class VoiceSession:
    """
    Connects voice transport to Agent Core.

    Responsibilities:
        - Receive audio from transport
        - Convert to text (STT)
        - Run the Agent Loop (Observe→Decide→Act)
        - Convert response to audio (TTS)
        - Send audio back via transport
        - Track session state
        - Handle errors, silence, termination

    Does NOT:
        - Contain business logic
        - Know about telephony providers
        - Know about WebSocket or SIP specifically
        - Duplicate AgentLoop logic

    Voice uses the EXACT same AgentLoop as the text CLI.
    """

    def __init__(
        self,
        session: CallSession,
        stt,        # STTProvider
        tts,        # TTSProvider
        on_event: Callable[[str, dict], None] | None = None,
        silence_timeout: float = 5.0,
        llm_provider=None,
    ) -> None:
        from app.config import settings

        self._session = session
        self._stt = stt
        self._tts = tts
        self._on_event = on_event or (lambda t, d: None)
        self._silence_timeout = silence_timeout

        # Build the same components as AgentRuntime, but keep state across turns
        self._llm = llm_provider or build_llm_provider()
        self._registry = self._build_registry()

        # Single CallState maintained across ALL turns of the conversation
        self._state = CallState(
            phone_number=session.phone_number,
            max_iterations=settings.max_turns,
            max_tool_calls=settings.max_tool_calls,
        )
        self._state.available_tools = ALL_TOOL_NAMES

        from app.agent.agent import Agent
        from app.observability.logger import make_event_callback

        self._rich_logger, _on_event_internal = make_event_callback(
            self._state.call_id
        )

        # Wire events: internal logger + external callback
        def combined_on_event(event_type: str, payload: dict) -> None:
            _on_event_internal(event_type, payload)
            self._on_event(event_type, payload)

        agent = Agent(llm=self._llm, registry=self._registry)
        self._loop = AgentLoop(agent, self._registry, on_event=combined_on_event)

        self._consecutive_silences = 0
        self._max_consecutive_silences = 2

    def _build_registry(self) -> ToolRegistry:
        from app.tools.registry import ToolRegistry
        from app.tools.customer import GetCustomerTool, VerifyCustomerTool
        from app.tools.orders import GetOrderTool, GetOrderStatusTool, GetCustomerOrdersTool
        from app.tools.calendar import ScheduleCallbackTool
        from app.tools.support import CreateSupportTicketTool, GetCustomerTicketsTool
        from app.tools.escalation import EndCallTool, TransferToHumanTool

        registry = ToolRegistry(sensitive_tools=VERIFICATION_REQUIRED_TOOLS)
        for tool_cls in [
            GetCustomerTool, VerifyCustomerTool, GetOrderTool, GetOrderStatusTool,
            GetCustomerOrdersTool, ScheduleCallbackTool, CreateSupportTicketTool,
            GetCustomerTicketsTool, TransferToHumanTool, EndCallTool,
        ]:
            registry.register(tool_cls())
        return registry

    def process_text(self, user_text: str) -> str:
        """
        Process a user text turn through the agent.
        Used by both local and WebSocket sessions.

        Returns the agent's text response.
        """
        self._session.status = SessionStatus.PROCESSING
        self._session.add_transcript_entry("user", user_text)
        self._emit("USER_TRANSCRIPTION", {"text": user_text, "call_id": self._state.call_id})

        # Update state for this turn
        self._state.latest_user_message = user_text
        self._state.add_message(MessageRole.USER, user_text)

        # Run the agent loop -- same loop used by the CLI
        try:
            loop_result = self._loop.run(self._state)
            final_state = loop_result.final_state
        except Exception as exc:
            self._emit("error", {"message": str(exc)})
            return "I'm sorry, I encountered an error. Please try again."

        # Get the agent's response
        agent_text = final_state.latest_agent_message or ""

        if agent_text:
            self._session.add_transcript_entry("agent", agent_text)

        # Sync session metrics
        self._session.tool_calls_made = final_state.tool_call_count
        self._session.iterations = final_state.current_iteration

        return agent_text

    def process_audio(self, audio_frame) -> tuple[str, str, bytes]:
        """
        Full audio → STT → Agent → TTS pipeline.

        Returns:
            (user_text, agent_text, audio_bytes)
        """
        from app.voice.audio import AudioFrame as AF

        self._session.status = SessionStatus.LISTENING

        # STT
        self._emit("STT_STARTED", {"call_id": self._state.call_id})
        t0 = time.monotonic()
        user_text = self._stt.transcribe(audio_frame)
        stt_latency_ms = (time.monotonic() - t0) * 1000
        self._emit("STT_COMPLETED", {"text": user_text, "latency_ms": stt_latency_ms})

        if not user_text.strip():
            return "", "", b""

        # Agent
        agent_text = self.process_text(user_text)

        # TTS
        audio_out = b""
        if agent_text:
            self._session.status = SessionStatus.SPEAKING
            self._emit("TTS_STARTED", {"text": agent_text, "call_id": self._state.call_id})
            t0 = time.monotonic()
            audio_frame_out = self._tts.synthesize(agent_text)
            tts_latency_ms = (time.monotonic() - t0) * 1000
            audio_out = audio_frame_out.samples
            self._emit("TTS_COMPLETED", {"latency_ms": tts_latency_ms})

        return user_text, agent_text, audio_out

    def run_local(self, transport) -> CallSession:
        """
        Run a complete local voice session (blocking).

        Args:
            transport: A LocalTransport instance.

        Returns:
            Completed CallSession.
        """
        from app.voice.audio import AudioFrame as AF

        self._session.status = SessionStatus.ACTIVE
        self._emit("CALL_STARTED", {
            "call_id": self._state.call_id,
            "session_id": self._session.session_id,
            "phone": self._session.phone_number,
        })

        transport.start_session()

        try:
            while not self._state.finished:
                self._session.status = SessionStatus.LISTENING

                # Get audio from transport
                audio_frame = transport.receive_audio(timeout=self._silence_timeout * 3)

                if audio_frame is None:
                    # Timeout
                    self._consecutive_silences += 1
                    if self._consecutive_silences >= self._max_consecutive_silences:
                        print("\n[VoiceSession] Silence timeout. Ending session.")
                        break
                    # Ask if still there
                    still_there_audio = self._tts.synthesize("Are you still there?")
                    transport.send_audio(still_there_audio)
                    continue

                self._consecutive_silences = 0

                # Full audio pipeline
                user_text, agent_text, audio_out = self.process_audio(audio_frame)

                if not user_text:
                    continue

                print(f"\n[YOU]   {user_text}")
                if agent_text:
                    print(f"[AGENT] {agent_text}")

                # Play response
                if audio_out:
                    from app.voice.audio import AudioFrame as AF2
                    response_frame = AF2(
                        samples=audio_out,
                        sample_rate=16000,
                        channels=1,
                        sample_width=2,
                    )
                    transport.send_audio(response_frame)

        except KeyboardInterrupt:
            print("\n[VoiceSession] Interrupted by user.")
        except Exception as exc:
            self._session.status = SessionStatus.FAILED
            self._emit("error", {"message": str(exc)})
        finally:
            transport.end_session()

        reason = (
            self._state.termination_reason.value
            if self._state.termination_reason else "completed"
        )
        self._session.end(SessionStatus.COMPLETED, reason)
        self._emit("CALL_ENDED", {
            "call_id": self._state.call_id,
            "termination_reason": reason,
            "turns": self._state.current_iteration,
            "tool_calls": self._state.tool_call_count,
        })
        return self._session

    @property
    def call_state(self) -> CallState:
        return self._state

    @property
    def is_finished(self) -> bool:
        return self._state.finished

    def _emit(self, event_type: str, payload: dict) -> None:
        try:
            self._on_event(event_type, payload)
        except Exception:
            pass
