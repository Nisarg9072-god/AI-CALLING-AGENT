"""
CLI Simulator — the interactive terminal demo for the AI Calling Agent.

Run with:  python -m app.main
Or:        python -m app.main --scenario order_status

Uses Rich for beautiful terminal output showing the full agent loop trace:
  [USER]     Customer message
  [THINK]    Agent reasoning
  [TOOL->]   Tool call dispatched
  [←TOOL]   Tool result received
  [AGENT]    Agent response spoken
  [END]      Call terminated
"""

from __future__ import annotations

import sys
from typing import Optional

import typer
from rich.console import Console
from rich.panel import Panel
from rich.rule import Rule
from rich.style import Style
from rich.table import Table
from rich.text import Text

from app.agent.decision import ActionType
from app.agent.llm_provider import build_llm_provider
from app.agent.state import TerminationReason
from app.config import settings
from app.harness.runtime import AgentRuntime
from app.observability.trace import EventType

app = typer.Typer(
    name="ai-calling-agent",
    help="Agentic AI Customer Service Calling Agent",
    add_completion=False,
)

console = Console()

# ── Color palette ──────────────────────────────────────────────────────────────
COLORS = {
    "user": "bold cyan",
    "agent": "bold green",
    "tool_out": "bold yellow",
    "tool_in": "bold magenta",
    "think": "dim italic white",
    "block": "bold red",
    "system": "bold blue",
    "end": "bold white on dark_green",
}


# ── Display helpers ────────────────────────────────────────────────────────────


def print_header() -> None:
    console.print()
    console.print(Panel.fit(
        "[bold white]  AI CALLING AGENT[/bold white]\n"
        "[dim]Agentic  Observe -> Decide -> Act  Loop Demo[/dim]\n"
        "[dim]Type your message and press Enter. Type 'quit' to exit.[/dim]",
        border_style="bright_blue",
        padding=(1, 4),
    ))
    console.print()


def print_event(tag: str, content: str, color: str, icon: str = "") -> None:
    label = Text(f" {icon}{tag} ", style=f"bold {color}")
    console.print(label, Text(content, style=color))


def print_separator(label: str = "") -> None:
    console.print(Rule(label, style="dim"))


def print_trace_summary(trace_events: list) -> None:
    """Print a compact event trace after each agent loop cycle."""
    table = Table(show_header=True, header_style="bold dim", box=None, padding=(0, 2))
    table.add_column("Turn", style="dim", width=4)
    table.add_column("Event", width=24)
    table.add_column("Detail", overflow="fold")

    for event in trace_events:
        etype = event.event_type.value
        payload = event.payload
        detail = ""

        if etype == "AGENT_DECISION":
            detail = f"action={payload.get('action')} | {payload.get('reasoning', '')[:60]}"
        elif etype in ("TOOL_REQUESTED", "TOOL_COMPLETED"):
            detail = f"{payload.get('tool_name', '')} — {str(payload.get('data', payload.get('arguments', '')))[:60]}"
        elif etype == "TOOL_FAILED":
            detail = f"[red]{payload.get('error', '')[:60]}[/red]"
        elif etype == "GUARDRAIL_BLOCKED":
            detail = f"[red]BLOCKED: {payload.get('reason', '')[:60]}[/red]"
        elif etype == "AGENT_RESPONSE":
            detail = payload.get("text", "")[:60]
        elif etype == "VERIFICATION_SUCCESS":
            detail = "[green]✓ Customer verified[/green]"
        elif etype == "VERIFICATION_FAILED":
            detail = f"[red]✗ Attempt {payload.get('attempt', '?')}[/red]"
        elif etype == "ESCALATION":
            detail = payload.get("reason", payload.get("text", ""))[:60]
        elif etype == "CALL_ENDED":
            detail = f"outcome={payload.get('outcome', '')}"

        table.add_row(str(event.iteration), etype, detail)

    if table.row_count:
        console.print()
        console.print("[dim]── Execution Trace ──[/dim]")
        console.print(table)


# ── Interactive mode ───────────────────────────────────────────────────────────


def run_interactive(phone: str = "+1-555-0101") -> None:
    """Run the agent in interactive CLI mode."""
    print_header()

    console.print(f"[dim]Phone:[/dim] [cyan]{phone}[/cyan]")
    console.print()

    # Get first message from user
    try:
        console.print("[bold cyan]YOU ->[/bold cyan] ", end="")
        first_message = input().strip()
    except (EOFError, KeyboardInterrupt):
        console.print("\n[dim]Call ended.[/dim]")
        return

    if first_message.lower() in ("quit", "exit", "q"):
        console.print("[dim]Goodbye.[/dim]")
        return

    console.print()
    print_separator("Starting call")

    # Build runtime
    runtime = AgentRuntime(customer_phone=phone)

    # Run the agent loop
    with console.status("[bold green]Agent processing...[/bold green]", spinner="dots"):
        final_state, trace = runtime.run(first_message, customer_phone=phone)

    # Display trace
    print_trace_summary(trace.events)
    console.print()
    print_separator("Call complete")

    # Show conversation
    console.print("\n[bold]📞 Full Conversation:[/bold]\n")
    for msg in final_state.conversation_history:
        role = msg.role.value.upper()
        if role == "USER":
            console.print(f"  [bold cyan]CUSTOMER ▶[/bold cyan] {msg.content}")
        elif role == "ASSISTANT":
            console.print(f"  [bold green]  AGENT ▶[/bold green] {msg.content}")
        elif role == "TOOL":
            console.print(f"  [dim yellow]   TOOL ▶[/dim yellow] {msg.content}")

    # Show outcome
    console.print()
    outcome_color = {
        TerminationReason.RESOLVED: "green",
        TerminationReason.AGENT_END: "green",
        TerminationReason.ESCALATED: "yellow",
        TerminationReason.MAX_TURNS_REACHED: "red",
        TerminationReason.ERROR: "red",
    }.get(final_state.termination_reason, "white")

    console.print(Panel.fit(
        f"[bold]Termination:[/bold] [{outcome_color}]{final_state.termination_reason.value}[/{outcome_color}]\n"
        f"[bold]Outcome:[/bold] {final_state.call_outcome or 'N/A'}\n"
        f"[bold]Turns:[/bold] {final_state.current_iteration} / {final_state.max_iterations}\n"
        f"[bold]Tool calls:[/bold] {final_state.tool_call_count} / {final_state.max_tool_calls}\n"
        f"[bold]Verified:[/bold] {'✓' if final_state.is_verified else '✗'}",
        title="[bold]Call Summary[/bold]",
        border_style=outcome_color,
    ))
    console.print()


# ── Typer commands ─────────────────────────────────────────────────────────────


@app.command()
def demo(
    phone: str = typer.Option("+1-555-0101", "--phone", "-p", help="Customer phone number"),
    message: Optional[str] = typer.Option(None, "--message", "-m", help="Run with this message (non-interactive)"),
) -> None:
    """Run the AI calling agent in interactive CLI demo mode."""
    if message:
        # Non-interactive single-shot mode
        runtime = AgentRuntime(customer_phone=phone)
        console.print(f"\n[dim]Running single-shot: '{message}'[/dim]\n")
        final_state, trace = runtime.run(message, customer_phone=phone)
        print_trace_summary(trace.events)
        console.print(f"\n[bold]Outcome:[/bold] {final_state.termination_reason.value}")
        console.print(f"[bold]Call outcome:[/bold] {final_state.call_outcome}")
    else:
        run_interactive(phone=phone)


@app.command()
def info() -> None:
    """Show current configuration and available tools."""
    table = Table(title="Agent Configuration", show_header=True, header_style="bold magenta")
    table.add_column("Setting", style="cyan")
    table.add_column("Value", style="white")

    table.add_row("LLM Provider", settings.llm_provider)
    table.add_row("LLM Model", settings.mistral_model)
    table.add_row("Max Turns", str(settings.max_turns))
    table.add_row("Max Tool Calls", str(settings.max_tool_calls))
    table.add_row("Sensitive Tools", ", ".join(settings.sensitive_tools))
    table.add_row("Trace Dir", settings.trace_dir)

    console.print()
    console.print(table)
    console.print()


if __name__ == "__main__":
    app()
