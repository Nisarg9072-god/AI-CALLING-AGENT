"""
Twilio webhook handlers — Phase 18.

Handles incoming Twilio events:
  POST /webhooks/twilio/voice    - Incoming call (TwiML response)
  POST /webhooks/twilio/gather   - Customer DTMF/speech input
  POST /webhooks/twilio/status   - Call status update

REQUIRES: TWILIO_CONFIGURED=True + ngrok or public URL.
"""

from __future__ import annotations

from fastapi import APIRouter, Form, Request
from fastapi.responses import HTMLResponse

router = APIRouter(prefix="/webhooks/twilio", tags=["twilio"])


@router.post("/voice", response_class=HTMLResponse)
async def twilio_incoming_call(request: Request) -> str:
    """
    Handle incoming Twilio call.
    Returns TwiML that greets the caller and starts the agent session.
    """
    # Phase 18 full implementation:
    # 1. Create new CallState
    # 2. Run agent greeting
    # 3. Use <Gather> to capture customer speech
    # 4. Stream response via TTS

    twiml = """<?xml version="1.0" encoding="UTF-8"?>
<Response>
  <Say voice="Polly.Joanna">
    Hello! Thank you for calling. I am your AI assistant.
    How can I help you today?
  </Say>
  <Gather input="speech" action="/webhooks/twilio/gather"
          language="en-US" speechTimeout="auto" timeout="5">
    <Say>Please speak after the tone.</Say>
  </Gather>
  <Say>I didn't catch that. Goodbye!</Say>
  <Hangup/>
</Response>"""
    return twiml


@router.post("/gather", response_class=HTMLResponse)
async def twilio_gather(
    SpeechResult: str = Form(default=""),
    CallSid: str = Form(default=""),
) -> str:
    """
    Handle customer speech input from Twilio Gather.

    In Phase 18 full implementation:
    1. Get call session by CallSid
    2. Run agent loop with SpeechResult as user message
    3. Convert agent response to speech via TwiML <Say>
    """
    # Placeholder — Phase 18 will wire this to AgentLoop
    agent_response = "I heard you say: " + (SpeechResult or "nothing.")

    twiml = f"""<?xml version="1.0" encoding="UTF-8"?>
<Response>
  <Say voice="Polly.Joanna">{agent_response}</Say>
  <Gather input="speech" action="/webhooks/twilio/gather"
          language="en-US" speechTimeout="auto" timeout="5">
  </Gather>
  <Hangup/>
</Response>"""
    return twiml


@router.post("/status")
async def twilio_status(
    CallSid: str = Form(default=""),
    CallStatus: str = Form(default=""),
) -> dict:
    """Receive call status updates from Twilio."""
    return {"received": True, "call_sid": CallSid, "status": CallStatus}
