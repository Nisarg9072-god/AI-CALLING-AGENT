"""
Configuration — typed settings loaded from environment variables / .env file.

All deterministic limits and policy parameters are defined here so they can
be inspected, tested, and overridden without touching agent code.
"""

from __future__ import annotations

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # ── LLM Provider ──────────────────────────────────────────────────────────
    llm_provider: str = Field(default="openai", description="LLM backend provider name")
    llm_api_key: str = Field(default="", description="API key for the LLM provider")
    llm_model: str = Field(default="gpt-4o", description="Model name to use")
    llm_temperature: float = Field(default=0.2, description="Sampling temperature (lower = more deterministic)")
    llm_max_tokens: int = Field(default=1024, description="Maximum tokens in LLM response")

    # ── Deterministic Agent Limits ─────────────────────────────────────────────
    max_turns: int = Field(default=20, description="Maximum conversation turns before forced termination")
    max_tool_calls: int = Field(default=10, description="Maximum total tool calls per call session")
    max_retries: int = Field(default=3, description="Maximum retries for failed tool calls")
    tool_timeout_seconds: int = Field(default=10, description="Tool execution timeout in seconds")

    # ── Guardrail Policy ──────────────────────────────────────────────────────
    sensitive_tools: list[str] = Field(
        default=["update_delivery_address", "cancel_order", "update_account"],
        description="Tools that require verified customer identity",
    )

    # ── Voice ──────────────────────────────────────────────────────────────────
    enable_voice: bool = Field(default=False, description="Enable real voice integration")
    voice_provider: str = Field(default="mock", description="Voice provider (mock | twilio | ...)")

    # ── Observability ─────────────────────────────────────────────────────────
    log_level: str = Field(default="INFO", description="Logging level")
    trace_to_file: bool = Field(default=True, description="Write execution traces to JSON files")
    trace_dir: str = Field(default="./traces", description="Directory for trace files")

    # ── Database ──────────────────────────────────────────────────────────────
    db_url: str = Field(default="memory", description="SQLite URL or 'memory' for in-memory")

    # ── Development ───────────────────────────────────────────────────────────
    debug: bool = Field(default=False, description="Enable debug output")


# Singleton settings instance — import this everywhere
settings = Settings()
