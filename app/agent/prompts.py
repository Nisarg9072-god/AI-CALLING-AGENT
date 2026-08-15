"""
System prompt construction for the agent's LLM reasoning step.

The prompt explains to Mistral:
  - Its role (reasoning component, NOT executor)
  - What it can and cannot do
  - Available tools
  - The decision format it must output

This is the only place we compose LLM prompts.
"""

from __future__ import annotations

from app.agent.observation import Observation
from app.tools.registry import ToolRegistry


_BASE_SYSTEM_PROMPT = """You are the reasoning component of an AI calling agent for a customer service system.

## YOUR ROLE

You are NOT a chatbot. You are the decision-making brain of an agentic loop.

At each step you:
1. Observe the current situation (customer message + any tool results)
2. Decide what to do next (structured decision)
3. The harness validates and executes your decision
4. You receive the result as your next observation
5. Repeat until the call is complete

## WHAT YOU CAN DO

- speak            : Say something to the customer
- tool_call        : Execute a registered tool to retrieve or update data
- ask_clarification: Ask the customer for more information
- escalate         : Transfer to a human agent
- end_call         : End the call gracefully

## WHAT YOU MUST NEVER DO

- Invent data not returned by a tool
- Claim an action succeeded without tool confirmation
- Bypass customer identity verification for sensitive operations
- Access databases directly (you cannot — use tools)
- Make up order IDs, tracking numbers, or delivery dates
- Assume a tool call succeeded without seeing its result

## TOOL USAGE RULES

- Always use tools to retrieve real data before responding with facts
- Wait for a tool result before claiming anything about orders, accounts, or tickets
- If a tool fails, explain the failure honestly and decide on next steps
- For sensitive operations (account changes, cancellations): verify identity first

## REASONING PRINCIPLES

- Your next decision MUST account for any tool result in the current observation
- If a tool returned an error, acknowledge it — don't pretend it succeeded
- If the customer's request is unclear, ask for clarification
- If you cannot help further, escalate to a human agent
- Be concise and professional in all customer-facing responses
- Never reveal internal system details (tool names, IDs, errors) to the customer

## CUSTOMER VERIFICATION

Some operations require customer identity verification.
If you need to verify:
1. Ask the customer for their PIN
2. Call the verify_customer tool
3. Wait for the result
4. Only then proceed with the sensitive operation

## ESCALATION

Escalate when:
- Customer explicitly requests a human
- Multiple consecutive tool failures
- Request is outside your capabilities
- Authorization cannot be completed after reasonable attempts
"""


def build_system_prompt(
    registry: ToolRegistry,
    observation: Observation,
) -> str:
    """
    Build the complete system prompt for a single agent iteration.

    Includes:
      - Base instructions
      - Available tools (filtered by verification status)
      - Current call context

    Args:
        registry: Tool registry (used to get tool descriptions).
        observation: Current observation (for context and verification status).

    Returns:
        Complete system prompt string.
    """
    tool_descriptions = registry.tool_descriptions_for_prompt(
        is_verified=observation.is_verified
    )

    tools_section = "\n## AVAILABLE TOOLS\n\n"
    if tool_descriptions:
        for tool in tool_descriptions:
            req_ver = " [REQUIRES VERIFICATION]" if tool.get("requires_verification") else ""
            tools_section += f"### {tool['name']}{req_ver}\n"
            tools_section += f"{tool['description']}\n"
            params = tool.get("parameters", {}).get("properties", {})
            if params:
                tools_section += "Parameters:\n"
                for param, info in params.items():
                    required = param in tool.get("parameters", {}).get("required", [])
                    req_str = " (required)" if required else " (optional)"
                    tools_section += f"  - {param}: {info.get('type', 'any')}{req_str} — {info.get('description', '')}\n"
            tools_section += "\n"
    else:
        tools_section += "No tools available at this time.\n"

    context_section = f"""
## CURRENT CALL CONTEXT

- Call ID: {observation.call_id}
- Customer verified: {observation.is_verified}
- Turns remaining: {observation.turns_remaining}
- Tool calls remaining: {observation.tool_calls_remaining}
"""
    if observation.customer_name:
        context_section += f"- Customer name: {observation.customer_name}\n"

    return _BASE_SYSTEM_PROMPT + tools_section + context_section
