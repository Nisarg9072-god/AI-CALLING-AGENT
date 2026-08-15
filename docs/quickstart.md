# Quick Start Guide

## Prerequisites

- Python 3.11+
- An OpenAI API key (for real LLM calls) OR use mock mode

## Installation

```bash
# 1. Clone / open in your IDE
cd "AI CALLING AGENT"

# 2. Install dependencies
pip install -e ".[dev]"

# 3. Configure environment
copy .env.example .env
# Edit .env and set OPENAI_API_KEY=sk-...
```

## Running the CLI Demo

### With a real OpenAI key

```bash
# Interactive mode — type your message and press Enter
python -m app.main demo

# Or with a custom phone number
python -m app.main demo --phone "+1-555-0202"

# Non-interactive (for scripts/CI)
python -m app.main demo --message "Where is my order ORD-1001?"
```

### Without an API key (mock mode)

```bash
LLM_PROVIDER=mock python -m app.main demo --message "Where is my order ORD-1001?"
```

## Running Evals (No API Key Required)

```bash
# Run all 9 scenarios
python -m evals.runner run

# Run a specific scenario
python -m evals.runner run --scenario s1_order_status

# Verbose output (shows failure details)
python -m evals.runner run --verbose
```

## Running Tests

```bash
# All tests
pytest

# With coverage
pytest --cov=app --cov-report=term-missing

# Specific test file
pytest tests/test_agent_integration.py -v
```

## Makefile Shortcuts

```bash
make install    # pip install -e ".[dev]"
make test       # pytest tests/
make evals      # python -m evals.runner run
make demo       # python -m app.main demo
make info       # Show current configuration
```

## Sample Conversation

```
YOU → Hi, I need to know the status of my order ORD-1001

── Starting call ──────────────────────────────────────────

── Execution Trace ────────────────────────────────────────
Turn  Event             Detail
1     AGENT_DECISION    action=tool_call | Looking up order ORD-1001
1     TOOL_REQUESTED    get_order_status — {'order_id': 'ORD-1001'}
1     TOOL_COMPLETED    get_order_status — {'status': 'shipped', ...}
2     AGENT_DECISION    action=speak | Reporting status after tool result
2     AGENT_RESPONSE    Your order ORD-1001 is currently shipped...
3     AGENT_DECISION    action=end_call | Issue resolved
3     CALL_ENDED        outcome=resolved

📞 Full Conversation:
  CUSTOMER ▶ Hi, I need to know the status of my order ORD-1001
    AGENT ▶ Your order ORD-1001 is currently shipped with tracking TRK-ALPHA-7823...
    AGENT ▶ Thank you for calling ACME Corp. Have a great day!

╭─ Call Summary ───────────────────────────────────────────╮
│ Termination: agent_end                                   │
│ Outcome: Order status provided                           │
│ Turns: 3 / 20                                            │
│ Tool calls: 2 / 10                                       │
│ Verified: ✗                                              │
╰──────────────────────────────────────────────────────────╯
```

## Available Mock Customers (for demo)

| Customer ID | Name | Phone | PIN |
|---|---|---|---|
| C001 | Alice Johnson | +1-555-0101 | 1234 |
| C002 | Bob Martinez | +1-555-0202 | 5678 |
| C003 | Carol Williams | +1-555-0303 | 9012 |
| C004 | David Chen | +1-555-0404 | 3456 |

## Available Mock Orders

| Order ID | Customer | Status |
|---|---|---|
| ORD-1001 | C001 | shipped |
| ORD-1002 | C001 | delivered |
| ORD-2001 | C002 | delayed |
| ORD-3001 | C003 | processing |
| ORD-4001 | C004 | cancelled |
