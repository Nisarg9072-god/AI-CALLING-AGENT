# GSM Architecture

Connecting the AI Calling Agent directly to cellular networks (GSM) allows you to bypass VoIP providers entirely, making calls using a standard SIM card.

## The GSMTransport Interface
The code contains a stub interface at `app/voice/transport/gsm.py`. 

## How NOT to do it
**You cannot reliably use an Android phone as a GSM gateway.**
Modern Android (API 26+) heavily restricts background apps from injecting audio into active cellular phone calls, and prevents programmatic answering/hanging up without being a privileged system app. 

## The Correct Approach: Hardware GSM Gateway

You need dedicated hardware that holds SIM cards and bridges them to an IP network (usually via SIP).

### Requirements
1. **GSM Gateway**: e.g., OpenVox VS-GW1600, Dinstar UC2000, or a cheap GoIP gateway. ($100-$500).
2. **SIM Card(s)**: Physical SIM cards with an active voice plan (e.g., unlimited calling).
3. **PBX (Asterisk)**: To bridge the gateway to the Python agent.

### Architecture

```text
Cellular Network (GSM/4G)
       │
       ▼
┌───────────────┐
│ GSM Gateway   │  (Holds SIM card)
└──────┬────────┘
       │ SIP
       ▼
┌───────────────┐
│ Asterisk PBX  │  (Handles codecs, routing)
└──────┬────────┘
       │ SIP
       ▼
┌───────────────┐
│ AI Agent      │  (SIPTransport -> AgentCore)
└───────────────┘
```

### Implementation Steps
1. Purchase and configure the GSM gateway to register to an Asterisk server.
2. Implement the `SIPTransport` (since the GSM gateway talks SIP to Asterisk).
3. When a call comes in, Asterisk routes it to the Python agent.
4. The Python agent processes the audio exactly as it would in Local or WebSocket mode.
