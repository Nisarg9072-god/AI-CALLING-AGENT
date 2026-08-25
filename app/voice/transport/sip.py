"""
SIPTransport -- SIP/VoIP transport abstraction.

STATUS: ARCHITECTURE STUB
This file defines the interface and documents the integration requirements.
Full implementation requires external SIP infrastructure.

============================================================
WHAT IS REQUIRED FOR REAL SIP CALLING
============================================================

Software (free/open-source):
    - Asterisk PBX (https://www.asterisk.org) -- free, open-source
    - FreeSWITCH (https://freeswitch.org) -- free, open-source
    - pjsua2 Python bindings for SIP (https://www.pjsip.org)

Carrier (requires payment):
    - A SIP trunk provider (e.g., Twilio SIP, VoIP.ms, Vonage SIP)
    - OR a PSTN gateway connected to a phone line
    - Monthly fees typically $1-20/month + per-minute rates

Phone number:
    - A DID (Direct Inward Dialing) number from your SIP provider
    - NOT free -- rental cost applies

Codecs:
    - G.711 ulaw/alaw (8kHz, 8-bit) -- convert to/from internal PCM16
    - G.722 (16kHz) -- closer to internal format
    - Opus (variable) -- modern, high quality

Audio path:
    Real phone → PSTN → SIP provider → Asterisk → pjsua2 → SIPTransport → Agent

============================================================
INTEGRATION INSTRUCTIONS (when you have a SIP provider)
============================================================

1. Install Asterisk or FreeSWITCH on a Linux server
2. Configure a SIP trunk with your provider
3. Install pjsip Python bindings: pip install pjsua2
4. Set in .env:
    SIP_HOST=sip.yourprovider.com
    SIP_PORT=5060
    SIP_USERNAME=your_username
    SIP_PASSWORD=your_password
    SIP_TRUNK=your_trunk_name
5. Implement _connect() and _handle_audio() below
6. Remove NotImplementedError from the methods

DO NOT claim SIP calling works until step 5-6 are complete AND
tested with a real phone call.
"""

from __future__ import annotations

from app.voice.audio import AudioFrame
from app.voice.transport.base import VoiceTransport, TransportStatus


class SIPTransport(VoiceTransport):
    """
    SIP/VoIP transport -- ARCHITECTURE STUB.

    The interface is defined and ready. Implementation requires
    a SIP library (pjsua2) and external SIP infrastructure.

    See module docstring for full requirements.
    """

    NOT_IMPLEMENTED_MSG = (
        "SIPTransport is not yet implemented.\n\n"
        "Requirements:\n"
        "  1. SIP library: pip install pjsua2\n"
        "  2. SIP provider account (Twilio SIP, VoIP.ms, etc.)\n"
        "  3. SIP credentials in .env:\n"
        "       SIP_HOST=sip.yourprovider.com\n"
        "       SIP_PORT=5060\n"
        "       SIP_USERNAME=your_username\n"
        "       SIP_PASSWORD=your_password\n"
        "  4. Real phone number (DID) from SIP provider\n\n"
        "See docs/sip.md for complete setup instructions."
    )

    def __init__(
        self,
        host: str | None = None,
        port: int = 5060,
        username: str | None = None,
        password: str | None = None,
    ) -> None:
        from app.config import settings
        self._host = host or getattr(settings, "sip_host", "")
        self._port = port or getattr(settings, "sip_port", 5060)
        self._username = username or getattr(settings, "sip_username", "")
        self._password = password or getattr(settings, "sip_password", "")
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
            "transport": "sip",
            "status": self._status.value,
            "host": self._host,
            "implemented": False,
            "note": "SIP transport requires external SIP infrastructure. See docs/sip.md",
        }
