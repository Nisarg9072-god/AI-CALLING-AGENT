# SIP Architecture

To connect the AI Calling Agent to real phone networks via SIP (Session Initiation Protocol), you need to bridge the gap between Python and the PSTN (Public Switched Telephone Network).

## The SIPTransport Interface
The code contains a stub interface at `app/voice/transport/sip.py`. It is not functional out-of-the-box because SIP requires a heavy C-library backend.

## Requirements for SIP Integration

1. **SIP Provider**: You need an account with a SIP Trunking provider (e.g., Twilio SIP, VoIP.ms, Bandwidth, Telnyx).
2. **DID Number**: You must rent a phone number from the provider.
3. **PBX (Optional but Recommended)**: Asterisk or FreeSWITCH to handle NAT traversal, transcoding, and SIP registration.
4. **SIP Library for Python**: `pjsua2` (PJSIP) is the industry standard.

## Implementation Steps

1. Install PJSIP and compile the Python bindings (`pjsua2`).
   ```bash
   # On Ubuntu
   sudo apt install libpjproject-dev
   pip install pjsua2
   ```

2. Implement `app/voice/transport/sip.py`:
   - Initialize a `pjsua2.Endpoint`.
   - Register a `pjsua2.Account` using credentials from `.env`.
   - Override `pjsua2.Call` to handle incoming calls.
   - Override `pjsua2.AudioMediaPort` to intercept PCM audio frames and pass them to `receive_audio()`.
   - Take `AudioFrame`s from `send_audio()` and write them to the PJSIP media port.

3. Update `.env`:
   ```env
   SIP_HOST=sip.yourprovider.com
   SIP_PORT=5060
   SIP_USERNAME=your_username
   SIP_PASSWORD=your_password
   ```

## Why isn't this included by default?
Compiling `pjsua2` is complex and highly platform-dependent. Including it as a default dependency would break the simple `pip install` experience.
