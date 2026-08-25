"""
Configuration — all settings loaded from environment variables / .env file.

Design principles:
  - Every tuneable parameter is defined here. No magic constants anywhere else.
  - Deterministic limits (max_turns, max_tool_calls) live here, not in LLM prompts.
  - Secrets are never logged, printed, or committed.
  - A single Settings singleton is imported everywhere via: from app.config import settings
"""

from __future__ import annotations

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # ── LLM Provider ──────────────────────────────────────────────────────────
    # Supported: "mistral" | "mock"
    # OpenAI is intentionally NOT a default to minimize cost.
    llm_provider: str = Field(default="mistral", description="LLM backend (mistral | mock)")

    # Mistral AI
    mistral_api_key: str = Field(default="", description="Mistral API key — never log this")
    mistral_model: str = Field(
        default="mistral-small-latest",
        description="Mistral model (mistral-small-latest | mistral-medium-latest | mistral-large-latest)",
    )
    llm_temperature: float = Field(default=0.2, ge=0.0, le=2.0)
    llm_max_tokens: int = Field(default=1024, ge=64, le=8192)

    # ── Deterministic Agent Limits (enforced by harness, NOT the LLM) ─────────
    max_turns: int = Field(default=20, ge=1, le=100, description="Max conversation turns before forced termination")
    max_tool_calls: int = Field(default=10, ge=1, le=50, description="Max total tool calls per call session")
    max_retries: int = Field(default=3, ge=0, le=10, description="Max retries for failed tool calls")
    tool_timeout_seconds: int = Field(default=10, ge=1, le=60)

    # ── Guardrail Policies ────────────────────────────────────────────────────
    # Tools in this list require verified customer identity before execution.
    # This is enforced deterministically by the harness — the LLM cannot bypass it.
    sensitive_tools: list[str] = Field(
        default=["update_delivery_address", "cancel_order", "update_account"],
        description="Tools requiring prior customer verification",
    )

    # ── Database ──────────────────────────────────────────────────────────────
    database_url: str = Field(
        default="sqlite+aiosqlite:///./data/agent.db",
        description="SQLAlchemy async database URL",
    )

    # ── FastAPI ───────────────────────────────────────────────────────────────
    host: str = Field(default="0.0.0.0")
    port: int = Field(default=8000, ge=1024, le=65535)
    reload: bool = Field(default=True)

    # ── Telephony ─────────────────────────────────────────────────────────────
    # Supported: "mock" | "twilio"
    # "mock" means no real calls are made — used for local development.
    telephony_provider: str = Field(default="mock", description="Telephony provider (mock | twilio)")

    # Twilio — REQUIRES PROVIDER CONFIGURATION
    # Trial accounts can only call verified numbers.
    twilio_account_sid: str = Field(default="", description="Twilio Account SID — never log this")
    twilio_auth_token: str = Field(default="", description="Twilio Auth Token — never log this")
    twilio_phone_number: str = Field(default="", description="Twilio source phone number (E.164)")
    twilio_to_number: str = Field(default="", description="Default destination number (E.164)")

    # ── Voice Pipeline ─────────────────────────────────────────────────────
    # Transport: local | websocket | sip | gsm
    # 'local' = no telephony provider required (default for development)
    voice_mode: str = Field(default="text", description="Voice mode (text | local | websocket)")
    voice_transport: str = Field(default="local", description="Voice transport (local | websocket | sip | gsm)")

    # Silence handling
    silence_timeout_seconds: float = Field(default=5.0, ge=1.0, le=60.0)

    # STT — faster-whisper (local, free, no cloud)
    stt_provider: str = Field(default="faster_whisper", description="STT provider (faster_whisper | mock)")
    whisper_model: str = Field(default="small", description="Whisper model (tiny|base|small|medium|large-v3)")
    whisper_device: str = Field(default="auto", description="Inference device (auto|cpu|cuda)")
    whisper_compute_type: str = Field(default="auto", description="Compute type (auto|int8|float16|float32)")
    whisper_language: str = Field(default="en", description="Language ISO 639-1 code")

    # Legacy STT aliases (kept for backward compatibility)
    stt_model: str = Field(default="small", description="Deprecated: use whisper_model")
    stt_device: str = Field(default="cpu", description="Deprecated: use whisper_device")

    # TTS — Piper (local, free, no cloud)
    tts_provider: str = Field(default="piper", description="TTS provider (piper | mock)")
    piper_model_path: str = Field(
        default="./data/voices/en_US-amy-medium.onnx",
        description="Path to Piper .onnx voice model file",
    )
    piper_voice: str = Field(default="en_US-amy-medium", description="Piper voice name")

    # ── SIP (optional — requires external SIP infrastructure) ─────────────────────
    # See docs/sip.md for setup instructions.
    sip_host: str = Field(default="", description="SIP server hostname")
    sip_port: int = Field(default=5060, description="SIP server port")
    sip_username: str = Field(default="", description="SIP username")
    sip_password: str = Field(default="", description="SIP password -- never log this")
    sip_trunk: str = Field(default="", description="SIP trunk name")

    # ── GSM Gateway (optional — requires hardware) ────────────────────────────
    # See docs/gsm.md and REAL_PHONE_SETUP.md for setup instructions.
    gsm_gateway_host: str = Field(default="", description="GSM gateway hostname/IP")
    gsm_gateway_port: int = Field(default=5060, description="GSM gateway port")

    # ── Observability ─────────────────────────────────────────────────────────
    log_level: str = Field(default="INFO")
    trace_to_file: bool = Field(default=True)
    trace_dir: str = Field(default="./data/traces")

    # ── Development ───────────────────────────────────────────────────────────
    debug: bool = Field(default=False)
    environment: str = Field(default="development")

    # ── Derived properties ────────────────────────────────────────────────────

    @property
    def is_development(self) -> bool:
        return self.environment == "development"

    @property
    def is_production(self) -> bool:
        return self.environment == "production"

    @property
    def mistral_configured(self) -> bool:
        """True if a real Mistral API key is present."""
        key = self.mistral_api_key
        return bool(key) and key != "your-mistral-api-key-here"

    @property
    def twilio_configured(self) -> bool:
        """True if all required Twilio credentials are present."""
        return bool(
            self.twilio_account_sid
            and self.twilio_auth_token
            and self.twilio_phone_number
            and not self.twilio_account_sid.startswith("your-")
        )

    @field_validator("log_level")
    @classmethod
    def validate_log_level(cls, v: str) -> str:
        allowed = {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}
        upper = v.upper()
        if upper not in allowed:
            raise ValueError(f"log_level must be one of {allowed}")
        return upper

    @field_validator("voice_mode")
    @classmethod
    def validate_voice_mode(cls, v: str) -> str:
        allowed = {"text", "local", "twilio"}
        if v not in allowed:
            raise ValueError(f"voice_mode must be one of {allowed}")
        return v

    @field_validator("telephony_provider")
    @classmethod
    def validate_telephony_provider(cls, v: str) -> str:
        allowed = {"mock", "twilio"}
        if v not in allowed:
            raise ValueError(f"telephony_provider must be one of {allowed}")
        return v

    def __repr__(self) -> str:
        """Safe repr — NEVER exposes secrets (API keys, tokens, auth)."""
        return (
            f"Settings("
            f"llm_provider={self.llm_provider!r}, "
            f"mistral_model={self.mistral_model!r}, "
            f"mistral_configured={self.mistral_configured}, "
            f"voice_mode={self.voice_mode!r}, "
            f"telephony_provider={self.telephony_provider!r}, "
            f"twilio_configured={self.twilio_configured}, "
            f"max_turns={self.max_turns}, "
            f"max_tool_calls={self.max_tool_calls}, "
            f"environment={self.environment!r}"
            f")"
        )

    def __str__(self) -> str:
        return self.__repr__()


# ── Singleton — import this everywhere ────────────────────────────────────────
# Usage: from app.config import settings
settings = Settings()
