"""
CLI Text Simulator — Phase 10.

Run:
    uv run python -m cli.simulator

Demonstrates the full Observe→Decide→Act loop in text mode.
Uses the SAME AgentLoop and tools as the voice and Twilio modes.

The simulator shows:
    [OBSERVE]  what the agent currently knows
    [DECIDE]   what Mistral decided
    [VALIDATE] harness check result
    [TOOL]     tool execution
    [RESULT]   tool output
    [AGENT]    what the agent said

IMPORTANT: Requires a valid MISTRAL_API_KEY in .env.
           Set LLM_PROVIDER=mock in .env for testing without API key.
"""

from __future__ import annotations

import sys

from rich.console import Console
from rich.panel import Panel
from rich.prompt import Prompt
from rich.text import Text

from app.agent.agent import Agent
from app.agent.loop import AgentLoop
from app.agent.state import CallState, MessageRole
from app.config import settings
from app.llm.factory import build_llm_provider
from app.observability.logger import make_event_callback
from app.tools.factory import build_tool_registry

console = Console()


def _print_header() -> None:
    console.print(Panel(
        Text.assemble(
            ("AI CALLING AGENT", "bold cyan"),
            "\n",
            ("Observe → Decide → Act Loop", "dim"),
            "\n",
            (f"LLM: {settings.llm_provider} / {settings.mistral_model}", "dim green"),
            "\n",
            ("Type your message and press Enter. 'quit' to exit.", "dim"),
        ),
        border_style="cyan",
        padding=(1, 4),
    ))


def run_text_session(
    phone: str = "+919537266092",
    customer_id: str | None = None,
    verbose: bool = True,
) -> None:
    """
    Run an interactive text-mode agent session.

    Args:
        phone: Simulated caller phone number.
        customer_id: Pre-set customer ID (optional).
        verbose: Show full trace output.
    """
    _print_header()

    if not settings.mistral_configured and settings.llm_provider != "mock":
        console.print(
            "[yellow]WARNING: MISTRAL_API_KEY not set. "
            "Set LLM_PROVIDER=mock in .env to test without API key.[/yellow]\n"
        )

    console.print(f"[dim]Phone: {phone}[/dim]\n")

    # ── Build the wired agent ─────────────────────────────────────────────────
    registry = build_tool_registry()
    llm = build_llm_provider()
    agent = Agent(llm=llm, registry=registry)

    # ── Initialize call state ─────────────────────────────────────────────────
    state = CallState(
        phone_number=phone,
        customer_id=customer_id,
        max_iterations=settings.max_turns,
        max_tool_calls=settings.max_tool_calls,
    )
    state.available_tools = registry.list_available(is_verified=False)

    # ── Observability ─────────────────────────────────────────────────────────
    logger, on_event = make_event_callback(state.call_id, silent=not verbose)
    loop = AgentLoop(agent=agent, registry=registry, on_event=on_event)

    console.print(
        f"[dim]Call ID: {state.call_id[:8]}... | "
        f"Max turns: {state.max_iterations} | "
        f"Max tools: {state.max_tool_calls}[/dim]\n"
    )

    # ── Agent greeting (first turn) ───────────────────────────────────────────
    state.current_user_message = "<<call_started>>"
    state.add_message(MessageRole.SYSTEM,
                      "Call started. Greet the customer and ask how you can help.")

    # Run ONE iteration for the greeting
    from app.agent.decision import ActionType
    from app.agent.observation import build_observation
    obs = build_observation(state)
    state.available_tools = registry.list_available(is_verified=False)
    state.current_iteration += 1
    greeting_decision = agent.decide(obs)
    if greeting_decision.action in (ActionType.SPEAK, ActionType.ASK_CLARIFICATION):
        text = greeting_decision.response_text or "Hello! How can I help you today?"
        state.add_message(MessageRole.ASSISTANT, text)
        console.print(f"\n[bold green]AGENT[/bold green]  {text}\n")

    # ── Interactive loop ──────────────────────────────────────────────────────
    while not state.finished:
        try:
            user_input = Prompt.ask("[bold yellow]YOU[/bold yellow]").strip()
        except (EOFError, KeyboardInterrupt):
            console.print("\n[dim]Session ended by user.[/dim]")
            break

        if user_input.lower() in ("quit", "exit", "bye", "q"):
            console.print("[dim]Exiting simulator.[/dim]")
            break

        if not user_input:
            continue

        # Record user message and run the loop for this turn
        state.current_user_message = user_input
        state.add_message(MessageRole.USER, user_input)
        console.print()

        result = loop.run(state)

        if state.finished:
            break

        console.print()  # spacing between turns

    # ── Summary ───────────────────────────────────────────────────────────────
    trace_path = logger.save()
    console.print(Panel(
        f"[bold]Call Summary[/bold]\n"
        f"Termination : {state.termination_reason.value if state.termination_reason else 'N/A'}\n"
        f"Outcome     : {state.outcome or 'N/A'}\n"
        f"Turns       : {state.current_iteration}\n"
        f"Tool calls  : {state.tool_call_count}\n"
        f"Verified    : {'Yes' if state.is_verified else 'No'}\n"
        + (f"Trace       : {trace_path}" if trace_path else ""),
        border_style="dim",
        title="Session Complete",
    ))


def main() -> None:
    """Entry point: python -m cli.simulator [--mock] [--phone PHONE]"""
    import argparse
    parser = argparse.ArgumentParser(description="AI Calling Agent CLI Simulator")
    parser.add_argument("--mock", action="store_true",
                        help="Force MockLLMProvider (no API key needed)")
    parser.add_argument("--phone", default="+919537266092",
                        help="Simulated caller phone number")
    parser.add_argument("--customer", default=None,
                        help="Pre-set customer ID")
    parser.add_argument("--quiet", action="store_true",
                        help="Suppress trace output (show only agent responses)")
    args = parser.parse_args()

    if args.mock:
        import os
        os.environ["LLM_PROVIDER"] = "mock"

    run_text_session(
        phone=args.phone,
        customer_id=args.customer,
        verbose=not args.quiet,
    )


if __name__ == "__main__":
    main()
