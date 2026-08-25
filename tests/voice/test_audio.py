"""
Tests for audio frame and format conversion utilities.
"""
import struct
import pytest
from app.voice.audio import (
    AudioFrame, AudioConverter,
    INTERNAL_SAMPLE_RATE, INTERNAL_CHANNELS, INTERNAL_SAMPLE_WIDTH,
    compute_rms, concat_frames,
)


class TestAudioFrame:
    def test_silence_frame(self):
        frame = AudioFrame.silence(0.5)
        assert frame.sample_rate == INTERNAL_SAMPLE_RATE
        assert frame.channels == INTERNAL_CHANNELS
        assert frame.sample_width == INTERNAL_SAMPLE_WIDTH
        # 0.5s * 16000 * 2 bytes = 16000 bytes
        assert len(frame.samples) == 16000

    def test_duration(self):
        # 1 second of silence
        frame = AudioFrame.silence(1.0)
        assert abs(frame.duration_seconds - 1.0) < 0.01

    def test_wav_roundtrip(self):
        """AudioFrame -> WAV bytes -> AudioFrame should preserve samples."""
        original = AudioFrame(
            samples=b"\x01\x00" * 100,
            sample_rate=16000,
            channels=1,
            sample_width=2,
        )
        wav = original.to_wav_bytes()
        restored = AudioFrame.from_wav_bytes(wav)
        assert restored.samples == original.samples
        assert restored.sample_rate == 16000

    def test_len(self):
        frame = AudioFrame(samples=b"\x00" * 200)
        assert len(frame) == 200


class TestAudioConverter:
    def test_to_internal_noop(self):
        """Already internal format should pass through unchanged."""
        frame = AudioFrame.silence(0.1)
        result = AudioConverter.to_internal(frame)
        assert result.sample_rate == INTERNAL_SAMPLE_RATE
        assert result.channels == INTERNAL_CHANNELS
        assert result.sample_width == INTERNAL_SAMPLE_WIDTH

    def test_stereo_to_mono(self):
        """Stereo frame should be converted to mono."""
        # Create stereo PCM: alternate L/R samples
        stereo_samples = struct.pack("<" + "hh" * 100, *([100, 200] * 100))
        frame = AudioFrame(
            samples=stereo_samples,
            sample_rate=16000,
            channels=2,
            sample_width=2,
        )
        result = AudioConverter.to_internal(frame)
        assert result.channels == 1
        assert len(result.samples) == 100 * 2  # 100 samples * 2 bytes

    def test_float32_to_pcm16_roundtrip(self):
        """float32 -> pcm16 -> float32 should be close."""
        import struct
        floats = [0.0, 0.5, -0.5, 1.0, -1.0]
        float32_bytes = struct.pack(f"<{len(floats)}f", *floats)
        pcm16_bytes = AudioConverter.float32_to_pcm16(float32_bytes)
        assert len(pcm16_bytes) == len(floats) * 2

    def test_resample_changes_rate(self):
        """Resampling should produce correct sample rate."""
        frame = AudioFrame.silence(1.0)  # 16kHz
        result = AudioConverter.resample(frame, 8000)
        assert result.sample_rate == 8000
        # Duration should be approximately preserved
        assert abs(result.duration_seconds - 1.0) < 0.1


class TestComputeRms:
    def test_silence_has_low_rms(self):
        frame = AudioFrame.silence(0.1)
        assert compute_rms(frame) == 0.0

    def test_nonzero_signal_has_positive_rms(self):
        # Create a constant signal
        samples = struct.pack("<" + "h" * 100, *([1000] * 100))
        frame = AudioFrame(samples=samples, sample_rate=16000, channels=1, sample_width=2)
        assert compute_rms(frame) > 0


class TestConcatFrames:
    def test_concat_two_frames(self):
        a = AudioFrame(samples=b"\x01" * 100)
        b = AudioFrame(samples=b"\x02" * 50)
        result = concat_frames([a, b])
        assert len(result.samples) == 150
        assert result.samples[:100] == b"\x01" * 100
        assert result.samples[100:] == b"\x02" * 50

    def test_concat_empty(self):
        result = concat_frames([])
        assert len(result.samples) == 0
