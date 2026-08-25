"""
Tests for VoiceSession and CallSession.
"""
import pytest
from app.voice.session import CallSession, VoiceSession, SessionStatus
from app.voice.audio import AudioFrame
from app.voice.stt import MockSTTProvider
from app.voice.tts import MockTTSProvider
from app.llm.mock_provider import MockLLMProvider


class TestCallSession:
    def test_create_and_end(self):
        session = CallSession(phone_number="+15550100")
        assert session.status == SessionStatus.CREATED
        assert session.phone_number == "+15550100"
        assert session.call_id is not None
        assert session.duration_seconds is None
        
        session.end(SessionStatus.COMPLETED, "ended_by_user")
        assert session.status == SessionStatus.COMPLETED
        assert session.termination_reason == "ended_by_user"
        assert session.duration_seconds is not None
        
    def test_add_transcript(self):
        session = CallSession()
        session.add_transcript_entry("user", "Hello")
        session.add_transcript_entry("agent", "Hi there")
        
        assert len(session.transcript) == 2
        assert session.transcript[0]["role"] == "user"
        assert session.transcript[0]["text"] == "Hello"
        assert "timestamp" in session.transcript[0]


class TestVoiceSession:
    @pytest.fixture
    def mock_session(self):
        session = CallSession()
        stt = MockSTTProvider(["Hello agent"])
        tts = MockTTSProvider()
        
        # Use a mock LLM that just echoes to avoid calling Mistral during tests
        llm = MockLLMProvider()
        llm.queue_response("Hello human")
        
        voice_session = VoiceSession(
            session=session,
            stt=stt,
            tts=tts,
            llm_provider=llm
        )
        return voice_session, session, stt, tts, llm

    def test_process_text(self, mock_session):
        voice_session, session, stt, tts, llm = mock_session
        
        response = voice_session.process_text("I need help")
        
        assert response == "Hello human"
        assert session.status == SessionStatus.PROCESSING
        assert len(session.transcript) == 2
        assert session.transcript[0]["text"] == "I need help"
        assert session.transcript[1]["text"] == "Hello human"

    def test_process_audio(self, mock_session):
        voice_session, session, stt, tts, llm = mock_session
        
        audio_in = AudioFrame.silence(0.1)
        user_text, agent_text, audio_out = voice_session.process_audio(audio_in)
        
        assert user_text == "Hello agent"
        assert agent_text == "Hello human"
        assert len(audio_out) > 0
        assert session.status == SessionStatus.SPEAKING
        
        # Check that STT and TTS were actually called
        assert stt.call_count == 1
        assert tts.call_count == 1
        assert tts.last_spoken == "Hello human"
        
    def test_process_audio_silence(self, mock_session):
        voice_session, session, stt, tts, llm = mock_session
        
        # Make STT return empty (silence/no speech)
        stt._responses = [""]
        
        audio_in = AudioFrame.silence(0.1)
        user_text, agent_text, audio_out = voice_session.process_audio(audio_in)
        
        # Should short-circuit
        assert user_text == ""
        assert agent_text == ""
        assert audio_out == b""
        
        # LLM and TTS should NOT have been called
        assert tts.call_count == 0
