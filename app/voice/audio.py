"""
AudioFrame -- normalized internal audio representation.

Internal format (non-negotiable):
    PCM, mono, 16 kHz, 16-bit signed little-endian

All transports convert to/from this format at the boundary.
The Agent Core, STT, and TTS never see raw telephony codecs.
"""

from __future__ import annotations

import audioop
import io
import struct
import wave
from dataclasses import dataclass, field
from typing import Literal


# ── Constants ──────────────────────────────────────────────────────────────────

INTERNAL_SAMPLE_RATE = 16_000    # Hz
INTERNAL_CHANNELS = 1             # mono
INTERNAL_SAMPLE_WIDTH = 2         # bytes (16-bit)
INTERNAL_FORMAT = "pcm16"         # label


# ── AudioFrame ─────────────────────────────────────────────────────────────────


@dataclass
class AudioFrame:
    """
    Normalized audio chunk passed through the voice pipeline.

    Always in internal format:
        PCM, mono, 16 kHz, 16-bit signed little-endian.

    Use AudioConverter.to_internal() to convert from external formats.
    """

    samples: bytes                              # raw PCM bytes
    sample_rate: int = INTERNAL_SAMPLE_RATE
    channels: int = INTERNAL_CHANNELS
    sample_width: int = INTERNAL_SAMPLE_WIDTH   # bytes per sample

    @property
    def duration_seconds(self) -> float:
        """Duration of this audio frame in seconds."""
        n_samples = len(self.samples) // (self.channels * self.sample_width)
        return n_samples / self.sample_rate

    @property
    def is_silent(self, threshold: int = 200) -> bool:
        """True if the frame contains only silence (RMS below threshold)."""
        if not self.samples:
            return True
        try:
            rms = audioop.rms(self.samples, self.sample_width)
            return rms < threshold
        except Exception:
            return False

    @classmethod
    def silence(cls, duration_seconds: float = 0.5) -> "AudioFrame":
        """Create a silent AudioFrame of the given duration."""
        n_samples = int(duration_seconds * INTERNAL_SAMPLE_RATE)
        return cls(samples=b"\x00" * n_samples * INTERNAL_SAMPLE_WIDTH)

    def to_wav_bytes(self) -> bytes:
        """Encode this frame as a WAV file (in memory)."""
        buf = io.BytesIO()
        with wave.open(buf, "wb") as wf:
            wf.setnchannels(self.channels)
            wf.setsampwidth(self.sample_width)
            wf.setframerate(self.sample_rate)
            wf.writeframes(self.samples)
        return buf.getvalue()

    @classmethod
    def from_wav_bytes(cls, wav_bytes: bytes) -> "AudioFrame":
        """Decode a WAV file into an AudioFrame (normalizing to internal format)."""
        buf = io.BytesIO(wav_bytes)
        with wave.open(buf, "rb") as wf:
            sample_rate = wf.getframerate()
            channels = wf.getnchannels()
            sample_width = wf.getsampwidth()
            samples = wf.readframes(wf.getnframes())
        frame = cls(
            samples=samples,
            sample_rate=sample_rate,
            channels=channels,
            sample_width=sample_width,
        )
        return AudioConverter.to_internal(frame)

    def __len__(self) -> int:
        return len(self.samples)


# ── AudioConverter ─────────────────────────────────────────────────────────────


class AudioConverter:
    """
    Converts AudioFrames between formats.

    Handles:
        - Sample rate conversion (via audioop.ratecv)
        - Channel conversion (stereo → mono via audioop.tomono)
        - Sample width conversion (audioop.lin2lin)
    """

    @staticmethod
    def to_internal(frame: AudioFrame) -> AudioFrame:
        """
        Convert any AudioFrame to the internal format:
        PCM mono 16kHz 16-bit.
        """
        samples = frame.samples
        channels = frame.channels
        sample_width = frame.sample_width
        sample_rate = frame.sample_rate

        # 1. Convert to 16-bit if needed
        if sample_width != INTERNAL_SAMPLE_WIDTH:
            samples = audioop.lin2lin(samples, sample_width, INTERNAL_SAMPLE_WIDTH)
            sample_width = INTERNAL_SAMPLE_WIDTH

        # 2. Convert to mono if needed
        if channels == 2:
            samples = audioop.tomono(samples, sample_width, 0.5, 0.5)
            channels = 1
        elif channels > 2:
            # Take first channel only for >2 channels
            n_frames = len(samples) // (channels * sample_width)
            mono = bytearray()
            for i in range(n_frames):
                offset = i * channels * sample_width
                mono.extend(samples[offset:offset + sample_width])
            samples = bytes(mono)
            channels = 1

        # 3. Resample if needed
        if sample_rate != INTERNAL_SAMPLE_RATE:
            samples, _ = audioop.ratecv(
                samples,
                sample_width,
                channels,
                sample_rate,
                INTERNAL_SAMPLE_RATE,
                None,
            )
            sample_rate = INTERNAL_SAMPLE_RATE

        return AudioFrame(
            samples=samples,
            sample_rate=sample_rate,
            channels=channels,
            sample_width=sample_width,
        )

    @staticmethod
    def resample(frame: AudioFrame, target_rate: int) -> AudioFrame:
        """Resample an AudioFrame to a different sample rate."""
        if frame.sample_rate == target_rate:
            return frame
        samples, _ = audioop.ratecv(
            frame.samples,
            frame.sample_width,
            frame.channels,
            frame.sample_rate,
            target_rate,
            None,
        )
        return AudioFrame(
            samples=samples,
            sample_rate=target_rate,
            channels=frame.channels,
            sample_width=frame.sample_width,
        )

    @staticmethod
    def float32_to_pcm16(float32_bytes: bytes) -> bytes:
        """
        Convert float32 PCM (from browser AudioContext) to int16 PCM.
        Browser sends float32 LE, we need int16.
        """
        import struct
        n_samples = len(float32_bytes) // 4
        float_samples = struct.unpack(f"<{n_samples}f", float32_bytes)
        int16_samples = [
            max(-32768, min(32767, int(s * 32767)))
            for s in float_samples
        ]
        return struct.pack(f"<{n_samples}h", *int16_samples)

    @staticmethod
    def pcm16_to_float32(pcm16_bytes: bytes) -> bytes:
        """Convert int16 PCM to float32 (for playback in browser)."""
        import struct
        n_samples = len(pcm16_bytes) // 2
        int16_samples = struct.unpack(f"<{n_samples}h", pcm16_bytes)
        float_samples = [s / 32767.0 for s in int16_samples]
        return struct.pack(f"<{n_samples}f", *float_samples)


# ── Audio utilities ────────────────────────────────────────────────────────────


def compute_rms(frame: AudioFrame) -> float:
    """Compute RMS energy of an audio frame (0.0 to 32767.0 for 16-bit)."""
    if not frame.samples:
        return 0.0
    try:
        return float(audioop.rms(frame.samples, frame.sample_width))
    except Exception:
        return 0.0


def concat_frames(frames: list[AudioFrame]) -> AudioFrame:
    """Concatenate multiple AudioFrames into one."""
    if not frames:
        return AudioFrame.silence(0.0)
    combined = b"".join(f.samples for f in frames)
    return AudioFrame(samples=combined)
