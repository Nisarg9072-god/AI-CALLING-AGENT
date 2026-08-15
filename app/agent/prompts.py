"""
Agent prompts — system prompt builder and observation formatter.

These are the only place where natural language instructions live.
Keeping prompts here makes them easy to iterate on without touching logic.
"""

from __future__ import annotations

from typing import Any

from app.agent.state import CallState


SYSTEM_PROMPT_TEMPLATE = """\
You are an AI customer service agent for ACME Corp, a consumer electronics company.
You handle inbound customer calls with professionalism, empathy, and efficiency.

## Your Role
- Understand the customer's request through active listening
- Use the available tools to look up information and take actions
- Never guess or fabricate data — always use tools to get real information
- Escalate to a human when you cannot resolve the issue

## Agentic Loop Rules
You will be called repeatedly. Each call you receive:
1. The current conversation history
2. Results of any tools you called in the previous turn
3. The current call state

You MUST respond with a JSON decision following the exact schema provided.
Never break out of the JSON format.

## Decision Actions
- `speak`: Say something to the customer
- `tool_call`: Call a company tool to get data or take action
- `ask_clarification`: Ask the customer for more information
- `escalate`: Transfer to a human agent
- `end_call`: Gracefully close the call

## Verification Rules
- You MUST verify the customer's identity with their 4-digit PIN before accessing sensitive data
- Use `verify_customer` before any account changes or sensitive queries
- If verification fails after 3 attempts, escalate to a human

## Tool Usage Rules
- Only call tools that are listed in the AVAILABLE TOOLS section below
- Always examine tool results before deciding what to do next
- If a tool fails, explain to the customer and try an alternative approach
- Never call the same tool with the same arguments twice

## Current Call State
Call ID: {call_id}
Customer Phone: {customer_phone}
Customer ID: {customer_id}
Verification Status: {verification_status}
Current Turn: {current_iteration}

## Available Tools
{tools_list}

## Guidelines
- Be concise but warm — this is a phone call, not a chat
- If unsure, ask one clarifying question at a time
- Always reason step by step (put reasoning in reasoning_summary)
- confidence should reflect how certain you are about this decision
"""


def build_system_prompt(state: CallState, tool_schemas: list[dict[str, Any]]) -> str:
    """Build the system prompt with current call state injected."""
    tools_str = _format_tools(tool_schemas)
    return SYSTEM_PROMPT_TEMPLATE.format(
        call_id=state.call_id,
        customer_phone=state.customer_phone or "unknown",
        customer_id=state.customer_id or "not yet identified",
        verification_status=state.verification_status.value,
        current_iteration=state.current_iteration,
        tools_list=tools_str,
    )


def _format_tools(schemas: list[dict[str, Any]]) -> str:
    if not schemas:
        return "No tools available."
    lines = []
    for schema in schemas:
        params = schema.get("parameters", {})
        props = params.get("properties", {})
        required = params.get("required", [])
        param_strs = []
        for k, v in props.items():
            req = " (required)" if k in required else " (optional)"
            desc = v.get("description", "")
            param_strs.append(f"    - {k}{req}: {desc}")
        params_block = "\n".join(param_strs) if param_strs else "    (no parameters)"
        lines.append(f"- **{schema['name']}**: {schema['description']}\n{params_block}")
    return "\n".join(lines)


def build_observation(state: CallState, last_tool_result: dict[str, Any] | None = None) -> str:
    """
    Build the 'observation' message that the agent sees at the start of each iteration.
    This is what 'Observe' means in the Observe→Decide→Act loop.
    """
    parts = []

    if last_tool_result is not None:
        if last_tool_result.get("success"):
            parts.append(f"TOOL RESULT: {last_tool_result['data']}")
        else:
            parts.append(f"TOOL ERROR: {last_tool_result['error']}")

    # Include recent conversation context
    recent = state.recent_messages(8)
    if recent:
        history = []
        for msg in recent:
            history.append(f"{msg.role.upper()}: {msg.content}")
        parts.append("CONVERSATION:\n" + "\n".join(history))

    parts.append(f"STATE: verification={state.verification_status.value}, "
                 f"turn={state.current_iteration}, "
                 f"tools_used={state.tool_call_count}")

    return "\n\n".join(parts)
