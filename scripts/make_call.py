"""
Initiate a real outbound call via Twilio — Phase 19.

Run:
    uv run python scripts/make_call.py --to "+1234567890"

Requires TWILIO_CONFIGURED=True and ngrok running (or public URL) pointing to FastAPI server.
"""

from __future__ import annotations

import argparse
import sys
from rich.console import Console

from app.config import settings
from app.voice.telephony import build_telephony_provider

console = Console()

def main() -> None:
    parser = argparse.ArgumentParser(description="Make an outbound AI call")
    parser.add_argument("--to", help="Phone number to call (E.164 format)")
    parser.add_argument("--webhook", help="Public URL of your FastAPI server (e.g. https://your-ngrok.ngrok.io)")
    args = parser.parse_args()

    to_number = args.to or settings.twilio_to_number
    if not to_number:
        console.print("[red]Error: Must provide --to number or set TWILIO_TO_NUMBER in .env[/red]")
        sys.exit(1)

    webhook_base = args.webhook or "http://localhost:8000"
    voice_url = f"{webhook_base}/webhooks/twilio/voice"

    provider = build_telephony_provider()
    
    console.print(f"[cyan]Initiating call to {to_number}...[/cyan]")
    console.print(f"[cyan]Webhook URL: {voice_url}[/cyan]")
    console.print(f"[cyan]Provider: {provider.__class__.__name__}[/cyan]")

    try:
        record = provider.make_call(to=to_number, webhook_url=voice_url)
        console.print(f"\n[green]Call initiated successfully![/green]")
        console.print(f"Call SID: {record.call_sid}")
        console.print(f"Status:   {record.status}")
    except Exception as e:
        console.print(f"\n[bold red]Call failed:[/bold red] {e}")
        if not settings.twilio_configured and settings.telephony_provider == "twilio":
            console.print("[yellow]Twilio is not configured correctly in .env[/yellow]")

if __name__ == "__main__":
    main()
