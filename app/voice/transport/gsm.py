"""
GSMTransport -- GSM/mobile network transport abstraction.

STATUS: ARCHITECTURE STUB
This file defines the interface and documents the integration requirements.
Full implementation requires external GSM hardware or gateway.

============================================================
WHAT IS REQUIRED FOR REAL GSM CALLING
============================================================

Hardware approach (GSM gateway):
    - A GSM gateway device (e.g., OpenVox, Dinstar, ALLO)
    - Price range: $100-$500 for hardware
    - Physical SIM card(s) from a mobile carrier
    - Ongoing SIM plan costs (minutes, data)
    - The gateway exposes a SIP or ISDN interface to your server

Software-only approach (Android + AGI):
    - An Android phone with Linphone or similar VoIP app
    - The phone must register to your Asterisk server
    - Modern Android restrictions make audio injection unreliable
    - NOT RECOMMENDED for production use

GSM gateway → Asterisk:
    GSM phone → Mobile network → GSM gateway → SIP → Asterisk → Agent

============================================================
LIMITATIONS ON ANDROID
============================================================

Modern Android (API 26+) DOES NOT allow:
    - Background apps to inject audio into calls
    - Programmatic answer/hangup without MODIFY_PHONE_STATE (requires system app)
    - Direct access to cellular audio streams from normal apps

This means you CANNOT use an ordinary Android phone as a simple
GSM gateway without:
    - Rooting the device, OR
    - A purpose-built GSM gateway device, OR
    - A carrier-grade solution like a dedicated SIM server

============================================================
RECOMMENDED GSM GATEWAY SETUP
============================================================

1. Purchase a GSM gateway (OpenVox VS-GW1600 or similar)
2. Insert SIM cards with a voice plan
3. Configure the gateway to connect to your Asterisk server via SIP
4. Configure Asterisk to route calls to/from the gateway
5. The AI agent connects to Asterisk via SIP (see SIPTransport)
6. All call audio flows: Phone → GSM gateway → Asterisk → Agent

DO NOT claim GSM calling works until tested with actual hardware.
"""

from __future__ import annotations

from app.voice.audio import AudioFrame
from app.voice.transport.base import VoiceTransport, TransportStatus


class GSMTransport(VoiceTransport):
    """
    GSM/mobile network transport -- ARCHITECTURE STUB.

    The interface is defined and ready. Implementation requires
    a GSM gateway device and SIM card(s).

    See module docstring for full requirements.
    """

    NOT_IMPLEMENTED_MSG = (
        "GSMTransport is not yet implemented.\n\n"
        "Requirements:\n"
        "  1. GSM gateway hardware (OpenVox, Dinstar, or similar)\n"
        "  2. Physical SIM card(s) with a voice plan\n"
        "  3. Asterisk PBX to bridge GSM ↔ SIP\n"
        "  4. Configure gateway in .env:\n"
        "       GSM_GATEWAY_HOST=192.168.1.100\n"
        "       GSM_GATEWAY_PORT=5060\n\n"
        "See docs/gsm.md for complete setup instructions.\n"
        "See REAL_PHONE_SETUP.md for cost breakdown."
    )

    def __init__(
        self,
        gateway_host: str | None = None,
        gateway_port: int | None = None,
    ) -> None:
        from app.config import settings
        self._gateway_host = gateway_host or getattr(settings, "gsm_gateway_host", "")
        self._gateway_port = gateway_port or getattr(settings, "gsm_gateway_port", 5060)
        self._status = TransportStatus.IDLE

    def start_session(self) -> None:
        raise NotImplementedError(self.NOT_IMPLEMENTED_MSG)

    def receive_audio(self, timeout: float | None = None) -> AudioFrame | None:
        raise NotImplementedError(self.NOT_IMPLEMENTED_MSG)

    def send_audio(self, audio: AudioFrame) -> None:
        raise NotImplementedError(self.NOT_IMPLEMENTED_MSG)

    def end_session(self) -> None:
        self._status = TransportStatus.ENDED

    def get_status(self) -> dict:
        return {
            "transport": "gsm",
            "status": self._status.value,
            "gateway_host": self._gateway_host,
            "implemented": False,
            "note": "GSM transport requires hardware gateway + SIM. See docs/gsm.md",
        }
