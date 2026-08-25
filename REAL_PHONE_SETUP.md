# Real Phone Network Setup

If you want the AI Calling Agent to make and receive calls on the actual telephone network (PSTN), you must bridge the gap between the internet and the phone system.

## Option 1: VoIP / SIP Provider (Easiest & Most Common)

This is the standard approach for cloud telephony.

1. **Sign up for a SIP Trunking provider.**
   - Twilio (SIP Trunking, not just the REST API)
   - VoIP.ms (Very cheap, great for hobbyists)
   - Telnyx
2. **Rent a phone number (DID).**
   - Cost: ~$1.00 to $2.00 per month.
3. **Configure your PBX (Asterisk) to connect to the provider.**
4. **Implement the `SIPTransport`** in the agent to connect to Asterisk.
5. **Cost per minute:** Usually around $0.005 to $0.01 per minute for inbound/outbound calls.

## Option 2: Hardware GSM Gateway (No VoIP Provider)

If you have unlimited calling on a physical SIM card, you can use a hardware gateway.

1. **Buy a GSM Gateway.**
   - Example: OpenVox VS-GW1600 or a cheaper GoIP gateway.
   - Cost: $100 - $500 upfront.
2. **Insert an active SIM card.**
   - Monthly cost: Whatever your carrier charges for the unlimited plan ($10 - $50/mo).
3. **Connect the Gateway to your PBX (Asterisk) over your local network.**
4. **Implement the `SIPTransport`** (because Asterisk talks SIP to the gateway).
5. **Cost per minute:** $0.00 (included in your unlimited carrier plan).

## Option 3: Twilio REST API (Included, but requires cloud)

The codebase includes `TwilioProvider` which uses Twilio's REST API and Webhooks.

1. Configure `.env` with `TELEPHONY_PROVIDER=twilio` and your Twilio credentials.
2. Use ngrok to expose your local server.
3. Set the Twilio webhook URL for your rented number to your ngrok address.
4. **Drawback:** You have to pay Twilio's per-minute rates, and audio routing is entirely managed by Twilio rather than the raw `VoiceTransport` interface.
