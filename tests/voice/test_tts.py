"""
Tests for TTS providers.
"""
import pytest
from app.voice.audio import AudioFrame
from app.voice.tts import build_tts_provider, MockTTSProvider, PiperProvider


def test_build_tts_provider_mock():
    tts = build_tts_provider("mock")
    assert isinstance(tts, MockTTSProvider)


def test_build_tts_provider_invalid():
    with pytest.raises(ValueError):
        build_tts_provider("invalid_provider")


class TestMockTTSProvider:
    def test_synthesize(self):
        tts = MockTTSProvider()
        
        frame = tts.synthesize("Hello world")
        assert isinstance(frame, AudioFrame)
        assert tts.call_count == 1
        assert tts.last_spoken == "Hello world"
        assert len(tts.spoken) == 1
        
        # AudioFrame from Mock is silence but valid
        assert len(frame.samples) > 0
        
    def test_reset(self):
        tts = MockTTSProvider()
        tts.synthesize("test")
        tts.reset()
        assert tts.call_count == 0
        assert len(tts.spoken) == 0


class TestPiperProvider:
    def test_init_and_properties(self):
        tts = PiperProvider(model_path="dummy.onnx")
        assert "dummy.onnx" in tts.provider_name
        assert not tts.model_loaded
        
    def test_empty_text(self):
        tts = PiperProvider()
        frame = tts.synthesize("")
        # Should return short silence without failing
        assert isinstance(frame, AudioFrame)
        assert len(frame.samples) > 0
