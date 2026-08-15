"""
Eval runner — runs all 9 scenarios and reports pass/fail with traces.

Usage:
  python -m evals.runner
  python -m evals.runner --scenario s1_order_status
  python -m evals.runner --verbose
"""

from __future__ import annotations

import sys
import time
from dataclasses import dataclass, field
from typing import Optional

import typer
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

# Force UTF-8 on Windows (legacy cp1252 terminal can't render Unicode symbols)
if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

from app.agent.llm_provider import MockLLMProvider
from app.agent.state import TerminationReason
from app.harness.runtime import AgentRuntime
from evals.scenarios import EvalScenario, get_all_scenarios

app = typer.Typer(name="evals", help="Eval runner for AI Calling Agent", add_completion=False)
console = Console()


# ── EvalResult ─────────────────────────────────────────────────────────────────


@dataclass
class EvalResult:
    scenario_name: str
    passed: bool
    failures: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    actual_turns: int = 0
    actual_tools_called: list[str] = field(default_factory=list)
    actual_termination: Optional[TerminationReason] = None
    duration_ms: float = 0.0


# ── Runner ─────────────────────────────────────────────────────────────────────


class EvalRunner:
    def run_scenario(self, scenario: EvalScenario) -> EvalResult:
        """Run a single scenario and evaluate the result."""
        start = time.perf_counter()

        # Build mock LLM with scripted responses
        mock_llm = MockLLMProvider(responses=list(scenario.mock_responses))
        runtime = AgentRuntime(
            llm_provider=mock_llm,
            customer_phone=scenario.customer_phone,
            trace_dir="./traces/evals",
        )

        # Override max_turns for the max_turns guardrail test
        final_state, trace = runtime.run(
            initial_message=scenario.initial_message,
            customer_phone=scenario.customer_phone,
        )

        elapsed = (time.perf_counter() - start) * 1000
        tools_called = [r.tool_name for r in final_state.tool_calls_made]

        result = EvalResult(
            scenario_name=scenario.name,
            passed=True,
            actual_turns=final_state.current_iteration,
            actual_tools_called=tools_called,
            actual_termination=final_state.termination_reason,
            duration_ms=elapsed,
        )

        # ── Assertion checks ───────────────────────────────────────────────────

        # G1: Expected tools were called
        for expected_tool in scenario.expected_tools:
            if expected_tool not in tools_called:
                result.failures.append(
                    f"Expected tool '{expected_tool}' was NOT called. Called: {tools_called}"
                )

        # G2: Forbidden tools were not called
        for forbidden_tool in scenario.forbidden_tools:
            if forbidden_tool in tools_called:
                result.failures.append(
                    f"Forbidden tool '{forbidden_tool}' WAS called. Should not have been."
                )

        # G3: Expected termination reason
        if scenario.expected_termination is not None:
            if final_state.termination_reason != scenario.expected_termination:
                result.failures.append(
                    f"Expected termination '{scenario.expected_termination.value}' "
                    f"but got '{final_state.termination_reason.value}'"
                )

        # G4: Minimum turns
        if final_state.current_iteration < scenario.min_turns:
            result.failures.append(
                f"Expected at least {scenario.min_turns} turns, got {final_state.current_iteration}"
            )

        # G5: State is terminated (no infinite loops)
        if not final_state.finished:
            result.failures.append("Call did not terminate - potential infinite loop!")

        result.passed = len(result.failures) == 0
        return result

    def run_all(
        self,
        filter_name: str | None = None,
        verbose: bool = False,
    ) -> list[EvalResult]:
        scenarios = get_all_scenarios()
        if filter_name:
            scenarios = [s for s in scenarios if filter_name in s.name]
        results = []
        for scenario in scenarios:
            result = self.run_scenario(scenario)
            results.append(result)
        return results


# ── CLI ────────────────────────────────────────────────────────────────────────


def _print_results(results: list[EvalResult], verbose: bool) -> None:
    table = Table(
        title=f"Eval Results - {len(results)} scenarios",
        show_header=True,
        header_style="bold white",
        border_style="dim",
    )
    table.add_column("Scenario", style="cyan", width=32)
    table.add_column("Result", width=8, justify="center")
    table.add_column("Turns", width=6, justify="right")
    table.add_column("Tools Called", overflow="fold")
    table.add_column("Termination", width=22)
    table.add_column("ms", width=6, justify="right")

    for r in results:
        status = "[bold green]PASS[/bold green]" if r.passed else "[bold red]FAIL[/bold red]"
        table.add_row(
            r.scenario_name,
            status,
            str(r.actual_turns),
            ", ".join(r.actual_tools_called) or "-",
            r.actual_termination.value if r.actual_termination else "?",
            f"{r.duration_ms:.0f}",
        )

    console.print()
    console.print(table)

    if verbose:
        for r in results:
            if r.failures or r.warnings:
                console.print(f"\n[bold]{'FAIL' if r.failures else 'WARN'}[/bold]: {r.scenario_name}")
                for f in r.failures:
                    console.print(f"  [red]FAIL: {f}[/red]")
                for w in r.warnings:
                    console.print(f"  [yellow]WARN: {w}[/yellow]")

    passed = sum(1 for r in results if r.passed)
    failed = len(results) - passed
    color = "green" if failed == 0 else "red"
    console.print()
    console.print(Panel.fit(
        f"[bold {color}]{passed}/{len(results)} scenarios passed[/bold {color}]"
        + (f"  •  [red]{failed} failed[/red]" if failed else ""),
        border_style=color,
    ))
    console.print()


@app.command()
def run(
    scenario: Optional[str] = typer.Option(None, "--scenario", "-s", help="Run a specific scenario by name"),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Show failure details"),
) -> None:
    """Run all eval scenarios and report results."""
    console.print("[bold]>> Running AI Calling Agent Evals...[/bold]\n")
    runner = EvalRunner()
    results = runner.run_all(filter_name=scenario, verbose=verbose)
    _print_results(results, verbose=verbose)

    # Exit with non-zero code if any failed (useful for CI)
    failed = sum(1 for r in results if not r.passed)
    if failed:
        raise typer.Exit(code=1)


if __name__ == "__main__":
    app()
