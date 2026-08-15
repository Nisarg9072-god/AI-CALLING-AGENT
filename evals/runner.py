"""
Evaluation framework — runs scripted scenarios through the real AgentLoop
and verifies behavioral requirements.

Each scenario tests that the agent:
  - Calls the right tools
  - Does NOT call forbidden tools
  - Reaches the expected outcome
  - Respects guardrails

All scenarios use MockLLMProvider — no API key required.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

from rich.console import Console
from rich.table import Table

from app.agent.agent import Agent
from app.agent.decision import ActionType, AgentDecision
from app.agent.loop import AgentLoop, LoopResult
from app.agent.state import CallState, TerminationReason
from app.llm.mock_provider import MockLLMProvider
from app.tools.factory import build_tool_registry

console = Console()


# ── Scenario definition ────────────────────────────────────────────────────────


@dataclass
class EvalScenario:
    """A single evaluation scenario."""
    name: str
    description: str
    initial_message: str
    llm_responses: list[AgentDecision]

    # Expected behaviors
    expected_tools: list[str] = field(default_factory=list)      # must have been called
    forbidden_tools: list[str] = field(default_factory=list)     # must NOT have been called
    expected_termination: TerminationReason | None = None
    expect_verified: bool | None = None                          # None = don't check
    max_tool_calls: int = 10
    max_turns: int = 20
    customer_id: str | None = "C001"

    # Behavioral assertions (optional callables)
    custom_assertions: list[Any] = field(default_factory=list)


@dataclass
class EvalResult:
    """Result of running a single scenario."""
    scenario_name: str
    passed: bool
    failures: list[str]
    turns: int
    tool_calls_made: list[str]
    termination_reason: str
    duration_ms: float


# ── Eval runner ────────────────────────────────────────────────────────────────


class EvalRunner:
    """Runs EvalScenarios and reports results."""

    def run_scenario(self, scenario: EvalScenario) -> EvalResult:
        registry = build_tool_registry()
        provider = MockLLMProvider(scenario.llm_responses)
        agent = Agent(llm=provider, registry=registry)
        loop = AgentLoop(agent=agent, registry=registry)

        state = CallState(
            customer_id=scenario.customer_id,
            max_iterations=scenario.max_turns,
            max_tool_calls=scenario.max_tool_calls,
        )
        from app.agent.state import MessageRole
        state.current_user_message = scenario.initial_message
        state.add_message(MessageRole.USER, scenario.initial_message)

        start = time.monotonic()
        loop.run(state)
        duration_ms = (time.monotonic() - start) * 1000

        # Collect which tools were actually called
        tools_called = [r.tool_name for r in state.tool_calls_made]
        failures: list[str] = []

        # Check expected tools
        for tool in scenario.expected_tools:
            if tool not in tools_called:
                failures.append(f"Expected tool '{tool}' to be called, but it wasn't.")

        # Check forbidden tools
        for tool in scenario.forbidden_tools:
            if tool in tools_called:
                failures.append(f"Forbidden tool '{tool}' was called.")

        # Check termination reason
        if scenario.expected_termination is not None:
            if state.termination_reason != scenario.expected_termination:
                failures.append(
                    f"Expected termination={scenario.expected_termination.value}, "
                    f"got={state.termination_reason}"
                )

        # Check verification
        if scenario.expect_verified is not None:
            if state.is_verified != scenario.expect_verified:
                failures.append(
                    f"Expected is_verified={scenario.expect_verified}, "
                    f"got={state.is_verified}"
                )

        # Custom assertions
        for assertion in scenario.custom_assertions:
            try:
                assertion(state)
            except AssertionError as e:
                failures.append(f"Custom assertion failed: {e}")

        return EvalResult(
            scenario_name=scenario.name,
            passed=len(failures) == 0,
            failures=failures,
            turns=state.current_iteration,
            tool_calls_made=tools_called,
            termination_reason=state.termination_reason.value if state.termination_reason else "N/A",
            duration_ms=duration_ms,
        )

    def run_all(self, scenarios: list[EvalScenario], verbose: bool = False) -> list[EvalResult]:
        results = []
        for scenario in scenarios:
            result = self.run_scenario(scenario)
            results.append(result)
        return results

    def print_results(self, results: list[EvalResult]) -> None:
        table = Table(title=f"Eval Results — {len(results)} scenarios", show_lines=True)
        table.add_column("Scenario", style="cyan", min_width=30)
        table.add_column("Result", min_width=6)
        table.add_column("Turns", justify="right")
        table.add_column("Tools Called")
        table.add_column("Termination")
        table.add_column("ms", justify="right")

        passed = sum(1 for r in results if r.passed)

        for r in results:
            status = "[bold green]PASS[/bold green]" if r.passed else "[bold red]FAIL[/bold red]"
            tools = ", ".join(r.tool_calls_made) if r.tool_calls_made else "-"
            table.add_row(
                r.scenario_name,
                status,
                str(r.turns),
                tools[:40],
                r.termination_reason,
                f"{r.duration_ms:.0f}",
            )

        console.print(table)

        # Print failures
        for r in results:
            if not r.passed:
                console.print(f"\n[bold red]FAILURES in '{r.scenario_name}':[/bold red]")
                for f in r.failures:
                    console.print(f"  [red]x[/red] {f}")

        console.print(
            f"\n[bold]{'[green]' if passed == len(results) else '[red]'}"
            f"{passed}/{len(results)} scenarios passed[/bold]"
        )
