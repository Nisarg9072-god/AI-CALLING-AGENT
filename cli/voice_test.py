"""
CLI voice test — end-to-end local voice session.

Usage:
    python -m cli.voice_test

Tests:
    - Microphone capture (sounddevice)
    - STT (faster-whisper)
    - Agent Loop (Observe->Decide->Act)
    - Tool execution
    - TTS (Piper)
    - Speaker playback

Requirements:
    pip install faster-whisper sounddevice piper-tts
    # Download Piper voice model (see docs/local-voice.md)

Press Ctrl+C to end the session.
"""

from __future__ import annotations

import sys


def main() -> None:
    print("""
============================================================
  AI Calling Agent - Local Voice Test
  faster-whisper STT | Piper TTS | Mistral AI
============================================================
""")

    # Check dependencies
    missing = []
    try:
        import sounddevice
    except ImportError:
        missing.append("sounddevice (pip install sounddevice)")

    try:
        from faster_whisper import WhisperModel
    except ImportError:
        missing.append("faster-whisper (pip install faster-whisper)")

    if missing:
        print("Missing required packages for local voice:")
        for m in missing:
            print(f"  - {m}")
        print("\nInstall with: uv pip install -e '.[voice]'")
        print("\nFalling back to TEXT-ONLY mode (STT/TTS will be mocked).\n")
        text_only_mode()
        return

    from app.config import settings
    from app.voice.stt import build_stt_provider
    from app.voice.tts import build_tts_provider
    from app.voice.session import CallSession, SessionStatus, VoiceSession
    from app.voice.transport.local import LocalTransport

    # Check for Piper model
    import os
    tts_provider_name = "piper"
    if not os.path.exists(settings.piper_model_path):
        print(f"Piper voice model not found: {settings.piper_model_path}")
        print("TTS will be text-only (no audio output).")
        print("Download a model: see docs/local-voice.md\n")
        tts_provider_name = "mock"

    # Build providers
    stt = build_stt_provider("faster_whisper")
    tts = build_stt_provider("mock") if tts_provider_name == "mock" else build_tts_provider("piper")
    if tts_provider_name == "mock":
        tts = build_tts_provider("mock")

    print(f"STT: {stt.provider_name}")
    print(f"TTS: {tts.provider_name if hasattr(tts, 'provider_name') else tts.__class__.__name__}")
    print(f"LLM: {settings.llm_provider} / {settings.mistral_model}")
    print()

    # Create session
    phone = input("Phone number (or press Enter for default): ").strip() or "+919537266092"
    session = CallSession(phone_number=phone, transport="local")

    def on_event(event_type: str, payload: dict) -> None:
        """Print key events to console."""
        if event_type in ("USER_TRANSCRIPTION",):
            pass    # already printed by VoiceSession
        elif event_type == "agent_decision":
            action = payload.get("action", "")
            tool = f" -> {payload.get('tool_name', '')}" if payload.get("tool_name") else ""
            print(f"  [AGENT] DECIDE: {action}{tool}")
        elif event_type in ("tool_requested", "TOOL_REQUESTED"):
            print(f"  [TOOL]  {payload.get('tool_name', '')}({payload.get('arguments', {})})")
        elif event_type in ("tool_completed", "TOOL_COMPLETED"):
            print(f"  [TOOL]  OK: {str(payload.get('data', ''))[:80]}")
        elif event_type in ("tool_failed", "TOOL_FAILED"):
            print(f"  [TOOL]  FAILED: {payload.get('error', '')}")

    voice_session = VoiceSession(
        session=session,
        stt=stt,
        tts=tts,
        on_event=on_event,
    )

    transport = LocalTransport(
        silence_threshold=300.0,
        min_speech_duration=0.5,
    )

    print("\n[Voice test ready. Speak when prompted. Ctrl+C to stop.]\n")

    try:
        voice_session.run_local(transport)
    except KeyboardInterrupt:
        print("\n[Voice test ended by user]")

    print(f"\nSession summary:")
    print(f"  Turns:      {session.iterations}")
    print(f"  Tool calls: {session.tool_calls_made}")
    print(f"  Status:     {session.status.value}")
    if session.transcript:
        print(f"\nTranscript ({len(session.transcript)} messages):")
        for msg in session.transcript:
            role = msg['role'].upper()
            text = msg['text'][:80]
            print(f"  [{role}] {text}")


def text_only_mode() -> None:
    """Fallback mode when audio deps are missing -- text input/output only."""
    from app.voice.session import CallSession, VoiceSession
    from app.voice.stt import MockSTTProvider
    from app.voice.tts import MockTTSProvider
    from app.config import settings

    print("Text-only mode. Type messages to interact with the agent.")
    print(f"LLM: {settings.llm_provider} / {settings.mistral_model}\n")

    session = CallSession(phone_number="+919537266092", transport="local")

    voice_session = VoiceSession(
        session=session,
        stt=MockSTTProvider(),
        tts=MockTTSProvider(),
    )

    print("Type your messages (Ctrl+C to quit):\n")
    try:
        while not voice_session.is_finished:
            try:
                user_text = input("YOU > ").strip()
            except EOFError:
                break
            if not user_text:
                continue
            if user_text.lower() in ("quit", "exit", "q"):
                break

            response = voice_session.process_text(user_text)
            print(f"AGENT > {response}\n")

    except KeyboardInterrupt:
        pass

    print("\n[Session ended]")


if __name__ == "__main__":
    main()
