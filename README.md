# 🤖 AI Calling Agent

A production-grade **agentic AI customer service calling agent** demonstrating a genuine **Observe → Decide → Act** loop architecture — not a chatbot wrapped in a procedural script.

## Architecture

```
Customer Input
      ↓
AgentRuntime (Harness)
┌─────────────────────────────────────────────────────────┐
│  while not state.finished:                              │
│    OBSERVE  → build_observation(state, last_tool_result)│
│    DECIDE   → agent.decide(observation)   ← LLM        │
│    VALIDATE → guardrails.validate(decision)             │
│    ACT      → tool_registry.execute(validated)         │
│    RECORD   → trace_logger.record(event)               │
└─────────────────────────────────────────────────────────┘
      ↓
Agent Response
```

| Layer | Role |
|---|---|
| **Agent** | Probabilistic — LLM decides intent, tool selection, response |
| **GuardrailEngine** | Deterministic — enforces limits, allowlists, auth, idempotency |
| **ToolRegistry** | Dispatches tool calls with schema validation |
| **CompanyService** | Only gateway to company data — never bypassed |
| **TraceLogger** | Full execution log of every event |

## Quick Start

```bash
# Install
pip install -e ".[dev]"

# Configure
copy .env.example .env
# Set OPENAI_API_KEY in .env

# Run interactive demo
python -m app.main demo

# Run evals (no API key needed)
python -m evals.runner run

# Run tests
pytest
```

## Available Tools

| Tool | Description | Requires Verification |
|---|---|---|
| `get_customer` | Look up customer by ID | No |
| `verify_customer` | Verify identity via PIN | No |
| `get_order` | Full order details | No |
| `get_order_status` | Status + ETA only | No |
| `get_customer_orders` | All orders for a customer | No |
| `schedule_callback` | Book a callback slot | No |
| `create_support_ticket` | Open a support ticket | No |
| `get_customer_tickets` | List customer tickets | No |
| `transfer_to_human` | Escalate to human agent | No |
| `end_call` | Gracefully end the call | No |

## Eval Scenarios (9 total)

| Scenario | What it tests |
|---|---|
| S1: Order status | 3-step loop, tool result drives response |
| S2: Customer + orders | Multi-tool multi-step reasoning |
| S3: Support ticket | Issue creation flow |
| S4: Callback | Scheduling side-effect |
| S5: Escalation | Human transfer path |
| S6: Max turns | Guardrail enforcement |
| S7: Verification | Auth before sensitive ops |
| S8: Tool→decision | Tool result changes next decision |
| S9: Complex multi-tool | 6-step reasoning chain |

## Key Design Principles

1. **Structured decisions** — LLM always outputs valid `AgentDecision` JSON, never free-form text
2. **Agent never writes state** — only the harness mutates `CallState`
3. **Tools never see CallState** — receive only a minimal context dict
4. **Voice-provider independent** — swap `MockCallProvider` → `TwilioCallProvider` without changing the agent
5. **Idempotent sensitive ops** — SHA-256 keyed duplicate detection

## Stack

- **Python 3.11+** + **Pydantic v2** (all models/schemas)
- **OpenAI GPT-4o** with structured JSON output (provider-agnostic interface)
- **Rich** for terminal output
- **Pytest** for unit + integration + agent tests
- **Typer** for CLI

## Project Structure

```
app/
├── agent/          # LLM reasoning core + loop + state + decisions
├── company/        # Repository + service (data layer)
├── harness/        # Runtime + guardrails + policies
├── observability/  # Event types + trace logger
├── tools/          # All tool implementations + registry
├── voice/          # Abstract interfaces + mock providers
└── main.py         # CLI entrypoint
evals/              # 9 eval scenarios + runner
tests/              # Unit + integration tests
docs/               # Architecture + quickstart docs
```
