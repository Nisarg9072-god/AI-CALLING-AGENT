# AI Calling Agent — Architecture Overview

## The Agentic Loop

This system implements a genuine **Observe → Decide → Act** loop. This is NOT a chatbot
wrapped in a procedural script — every decision is made by the LLM at runtime, with the
harness enforcing deterministic constraints.

```
Customer Speech / Text Input
          ↓
   [Voice Layer / CLI]
          ↓
    AgentRuntime (Harness)
    ┌──────────────────────────────────────────────────────┐
    │  while not state.finished:                           │
    │    1. OBSERVE  → build_observation(state, last_tool) │
    │    2. DECIDE   → agent.decide(observation)  ← LLM   │
    │    3. VALIDATE → guardrails.validate(decision)       │
    │    4. ACT      → execute(validated_decision)         │
    │    5. RECORD   → trace_logger.record(event)          │
    └──────────────────────────────────────────────────────┘
          ↓
    [TTS / CLI Output]
          ↓
   Customer Hears Response
```

## Separation of Concerns

| Component | Role | Deterministic? |
|---|---|---|
| `Agent` | Probabilistic reasoning — intent, tool selection, response generation | ❌ LLM |
| `GuardrailEngine` | Policy enforcement — limits, allowlists, auth, idempotency | ✅ Yes |
| `ToolRegistry` | Tool dispatch, schema validation, permission checks | ✅ Yes |
| `CompanyService` | Business logic — only gateway to company data | ✅ Yes |
| `CompanyRepository` | Data storage (in-memory, swap to SQLite/Postgres) | ✅ Yes |
| `TraceLogger` | Full execution log — every event recorded | ✅ Yes |

## Data Flow

```
Agent
  ↕ (AgentDecision — structured Pydantic model)
GuardrailEngine
  ↕ (ValidatedDecision — post-guardrail)
ToolRegistry
  ↕ (ToolResult)
BaseTool implementations
  ↕ (service calls)
CompanyService
  ↕ (CRUD)
CompanyRepository (in-memory)
```

## Key Design Decisions

### 1. Structured Decisions — Not Free-Form Text

Every LLM response must be a valid `AgentDecision` JSON object. The harness validates
it before anything executes. This prevents prompt injection, hallucinated tool names,
and invalid arguments from reaching the execution layer.

### 2. Agent Never Writes State

The `Agent` class is read-only. Only the `AgentLoop` (owned by the harness) mutates
`CallState`. This makes reasoning about the system much easier.

### 3. Tools Never See CallState

Tools receive only a minimal `context` dict (call_id, customer_id, is_verified). They
cannot read conversation history, inspection execution events, or other tools' results.

### 4. Voice-Provider Independence

The agent core (Agent, AgentLoop, AgentRuntime) has zero imports from `app.voice`.
Swap `MockCallProvider` for `TwilioCallProvider` and nothing in the agent changes.

### 5. Idempotency for Sensitive Operations

Sensitive tools (those requiring verification) use SHA-256 hashed (tool_name, arguments)
as an idempotency key. Re-calling the same tool with the same args is a no-op.

## File Structure

```
app/
├── config.py              # All settings (env vars)
├── main.py                # CLI entrypoint
├── agent/
│   ├── agent.py           # Probabilistic reasoning core
│   ├── decision.py        # AgentDecision schema
│   ├── llm_provider.py    # OpenAI + Mock LLM providers
│   ├── loop.py            # Observe→Decide→Act loop
│   ├── prompts.py         # System prompt + observation builder
│   └── state.py           # CallState model
├── company/
│   ├── repository.py      # In-memory data store
│   └── service.py         # Business logic layer
├── harness/
│   ├── guardrails.py      # Deterministic policy engine
│   ├── policies.py        # Tool lists and limits
│   └── runtime.py         # AgentRuntime (top-level wiring)
├── observability/
│   └── trace.py           # EventType, TraceLogger
├── tools/
│   ├── base.py            # BaseTool, ToolResult
│   ├── registry.py        # ToolRegistry
│   ├── customer.py        # get_customer, verify_customer
│   ├── orders.py          # get_order, get_order_status, get_customer_orders
│   ├── calendar.py        # schedule_callback
│   ├── support.py         # create_support_ticket, get_customer_tickets
│   └── escalation.py      # transfer_to_human, end_call
└── voice/
    └── mock.py            # MockCallProvider, MockSTTProvider, MockTTSProvider
evals/
├── scenarios.py           # 9 scripted eval scenarios
└── runner.py              # Eval runner with pass/fail assertions
tests/
├── test_state_and_decision.py
├── test_tools_and_service.py
└── test_agent_integration.py
```
