"""
Twilio telephony provider — Phase 17.

Handles outbound call initiation and call management via Twilio REST API.

Configuration (in .env):
    TELEPHONY_PROVIDER=twilio
    TWILIO_ACCOUNT_SID=ACxxxxxxxxxx
    TWILIO_AUTH_TOKEN=your-token
    TWILIO_PHONE_NUMBER=+1xxxxxxxxxx

REQUIRES PROVIDER CONFIGURATION:
    - Twilio trial account can only call verified numbers
    - For production: upgrade account + purchase phone number
    - ngrok or public URL required for webhooks
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any


# ── Abstract telephony interface ───────────────────────────────────────────────


@dataclass
class CallRecord:
    """Result of initiating a call."""
    call_sid: str
    status: str
    from_number: str
    to_number: str
    provider: str
    extra: dict[str, Any] | None = None


class TelephonyProvider(ABC):
    @abstractmethod
    def make_call(self, to: str, webhook_url: str) -> CallRecord:
        """Initiate an outbound call."""

    @abstractmethod
    def hang_up(self, call_sid: str) -> bool:
        """Hang up an active call."""


# ── Mock provider (default, no real calls) ────────────────────────────────────


class MockTelephonyProvider(TelephonyProvider):
    """No-op provider — logs calls without making real ones."""

    def __init__(self) -> None:
        self.calls_made: list[dict] = []

    def make_call(self, to: str, webhook_url: str) -> CallRecord:
        record = CallRecord(
            call_sid=f"MOCK-{len(self.calls_made)+1:04d}",
            status="mock_initiated",
            from_number="MOCK",
            to_number=to,
            provider="mock",
            extra={"webhook_url": webhook_url},
        )
        self.calls_made.append(record.__dict__)
        print(
            f"[MOCK TELEPHONY] Would call {to} -> webhook: {webhook_url}"
        )
        return record

    def hang_up(self, call_sid: str) -> bool:
        print(f"[MOCK TELEPHONY] Would hang up {call_sid}")
        return True


# ── Twilio provider (Phase 17) ────────────────────────────────────────────────


class TwilioProvider(TelephonyProvider):
    """
    Real Twilio outbound call provider.

    REQUIRES PROVIDER CONFIGURATION — see .env.example for all required fields.
    """

    def __init__(self) -> None:
        from app.config import settings
        if not settings.twilio_configured:
            raise RuntimeError(
                "Twilio is not configured. Set TWILIO_ACCOUNT_SID, "
                "TWILIO_AUTH_TOKEN, and TWILIO_PHONE_NUMBER in .env"
            )
        from twilio.rest import Client
        self._client = Client(settings.twilio_account_sid, settings.twilio_auth_token)
        from app.config import settings as s
        self._from_number = s.twilio_phone_number

    def make_call(self, to: str, webhook_url: str) -> CallRecord:
        call = self._client.calls.create(
            to=to,
            from_=self._from_number,
            url=webhook_url,
            method="POST",
        )
        return CallRecord(
            call_sid=call.sid,
            status=call.status,
            from_number=self._from_number,
            to_number=to,
            provider="twilio",
            extra={"direction": call.direction},
        )

    def hang_up(self, call_sid: str) -> bool:
        try:
            call = self._client.calls(call_sid).update(status="completed")
            return call.status == "completed"
        except Exception:
            return False


# ── Factory ───────────────────────────────────────────────────────────────────


def build_telephony_provider() -> TelephonyProvider:
    from app.config import settings
    if settings.telephony_provider == "twilio" and settings.twilio_configured:
        return TwilioProvider()
    return MockTelephonyProvider()
