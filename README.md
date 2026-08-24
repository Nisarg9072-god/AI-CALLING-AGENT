# AI Calling Agent v2 !!

A **genuinely agentic** AI calling system built with a real **Observe → Decide → Act** loop.

> This is NOT a chatbot. Every decision is made by Mistral AI observing the full context including previous tool results. The harness deterministically enforces all policies, limits, and guardrails.

## The Agent Loop

```
while not state.finished:

    observation = observe(state)           # What does the agent know right now?
    decision    = decide(observation)      # Mistral reasons → structured decision
    validated   = harness.validate(decision)  # Deterministic policy enforcement
    result      = harness.execute(validated)  # Tool runs / response emitted
    state       = update_state(state, result) # Next iteration sees the result
```

The **next decision depends on the previous tool result**. This is what makes it agentic.

## Architecture

```
PHONE USER
    │
    ▼
 TWILIO / LOCAL MIC
    │  (audio)
    ▼
   STT (faster-whisper)
    │  (text)
    ▼
┌──────────────────────────────┐
│       AGENT HARNESS          │
│  State · Policies · Limits   │
│  Guardrails · Observability  │
└──────────────┬───────────────┘
               │
    ┌──────────▼──────────┐
    │     AGENT LOOP      │
    │  Observe            │
    │  Decide (Mistral)   │
    │  Validate           │
    │  Act                │
    │  Update state       │
    └──────────┬──────────┘
               │
         TOOL REGISTRY
               │
    ┌──────────┼──────────┐
    ▼          ▼          ▼
 Customer   Orders    Calendar
  Tool       Tool       Tool
    │          │          │
    └──────────┼──────────┘
               ▼
        COMPANY SERVICE
               ▼
          REPOSITORY
               ▼
            SQLite
               │
    Agent Response (text)
               │
    TTS (Piper) → audio
               │
    TWILIO / LOCAL SPEAKER
```

## Separation of Concerns

| Component | Role | Deterministic? |
|---|---|---|
| **Mistral** | Intent, tool selection, response generation | No — probabilistic |
| **Harness** | Policy enforcement, limits, auth, idempotency | Yes — always |
| **Tool Registry** | Tool dispatch, schema validation, permissions | Yes — always |
| **Company Service** | Only gateway to business data | Yes — always |
| **SQLite** | Persistence — never accessed directly by agent | Yes — always |

## Phases

| Phase | Status |
|---|---|
| 1 — Project structure, config, dependencies | ✅ DONE |
| 2 — State, Observation, Decision models | ✅ DONE |
| 3 — Tool abstraction + tool registry | ✅ DONE |
| 4 — SQLite, repositories, company services | ✅ DONE |
| 5 — Mistral LLM provider | ✅ DONE |
| 6 — Real Agent Loop | ✅ DONE |
| 7 — Harness (runtime + policies) | ✅ DONE |
| 8 — Guardrails + authorization + idempotency | ✅ DONE |
| 9 — Observability + execution traces | ✅ DONE |
| 10 — CLI text simulator | ✅ DONE |
| 11 — Evaluation framework | ✅ DONE |
| 12 — Test suite | ✅ DONE |
| 13 — faster-whisper STT | ✅ DONE |
| 14 — Piper TTS | ✅ DONE |
| 15 — Local voice pipeline | ✅ DONE |
| 16 — FastAPI call/session API | ✅ DONE |
| 17 — Twilio telephony provider | ✅ DONE |
| 18 — Twilio webhooks | ✅ DONE |
| 19 — Real outbound call | ✅ DONE |
| 20 — Full end-to-end test | ✅ DONE |

## Quick Start (Phase 1 — text mode only)

```bash
# 1. Create venv and install
uv venv --python 3.12
uv pip install -e ".[dev]"

# 2. Configure
copy .env.example .env
# Edit .env — set MISTRAL_API_KEY=your-real-key

# 3. Verify config
uv run python -c "from app.config import settings; print(settings)"

# 4. Run checks
uv run pytest tests/ -v
```

## Stack

- **Python 3.12+** with full type hints
- **Mistral AI** — sole LLM (cost-efficient, no OpenAI dependency)
- **FastAPI** — REST API + webhook handling
- **SQLAlchemy + aiosqlite** — async SQLite
- **faster-whisper** — local STT (free, no cloud)
- **Piper TTS** — local TTS (free, no cloud)
- **Twilio** — telephony (isolated behind abstraction)
- **Pydantic v2** — all models and validation
- **Rich** — terminal output
- **pytest** — testing

## Design Principles

1. **Agent never directly accesses SQLite** — always through Tool → Service → Repository
2. **LLM never bypasses authorization** — harness validates every decision
3. **All limits are deterministic** — max_turns, max_tool_calls enforced by harness
4. **Idempotency is harness responsibility** — not LLM responsibility
5. **Voice mode is swappable** — same Agent Loop for text, local voice, and Twilio

## NOT VERIFIED (Phase 1)

- Mistral API calls (require real API key)
- SQLite persistence (Phase 4)
- FastAPI endpoints (Phase 16)
- Twilio calls (Phase 17-19, REQUIRES PROVIDER CONFIGURATION)
- faster-whisper STT (Phase 13, requires separate install)
- Piper TTS (Phase 14, requires separate install + model download)
