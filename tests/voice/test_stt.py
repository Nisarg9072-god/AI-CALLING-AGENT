"""
Tests for STT providers.
"""
import pytest
from app.voice.audio import AudioFrame
from app.voice.stt import build_stt_provider, MockSTTProvider, FasterWhisperProvider


def test_build_stt_provider_mock():
    stt = build_stt_provider("mock", mock_responses=["hello"])
    assert isinstance(stt, MockSTTProvider)
    assert stt.provider_name.startswith("MockSTTProvider")


def test_build_stt_provider_invalid():
    with pytest.raises(ValueError):
        build_stt_provider("invalid_provider")


class TestMockSTTProvider:
    def test_transcribe(self):
        stt = MockSTTProvider(["First", "Second"])
        audio = AudioFrame.silence(0.1)
        
        assert stt.transcribe(audio) == "First"
        assert stt.call_count == 1
        
        assert stt.transcribe(audio) == "Second"
        assert stt.call_count == 2
        
        # Falls back to goodbye when exhausted
        assert stt.transcribe(audio) == "goodbye"
        
    def test_queue_and_reset(self):
        stt = MockSTTProvider(["One"])
        stt.queue("Two")
        audio = AudioFrame.silence(0.1)
        
        assert stt.transcribe(audio) == "One"
        assert stt.transcribe(audio) == "Two"
        
        stt.reset()
        assert stt.transcribe(audio) == "One"
        assert stt.call_count == 1


# We don't comprehensively test FasterWhisperProvider here to avoid requiring
# the model to be downloaded during normal test runs, but we can test
# its configuration parsing and empty transcription.

class TestFasterWhisperProvider:
    def test_init_and_properties(self):
        stt = FasterWhisperProvider(model_size="tiny", device="cpu", compute_type="int8")
        assert "tiny" in stt.provider_name
        assert "cpu" in stt.provider_name
        assert not stt.model_loaded
        
    def test_empty_audio(self):
        stt = FasterWhisperProvider()
        empty = AudioFrame(samples=b"")
        assert stt.transcribe(empty) == ""
