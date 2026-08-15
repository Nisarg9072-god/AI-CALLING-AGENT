"""
MistralProvider — real Mistral AI LLM implementation.

Uses JSON mode + schema-in-system-prompt since Mistral doesn't support
full JSON Schema mode. The DECISION_JSON_SCHEMA is injected into the
system prompt so Mistral knows exactly what structure to output.

Configuration (from .env):
    MISTRAL_API_KEY=your-key
    MISTRAL_MODEL=mistral-small-latest
    LLM_TEMPERATURE=0.2
    LLM_MAX_TOKENS=1024
"""

from __future__ import annotations

import json
import re

from app.agent.decision import DECISION_JSON_SCHEMA, ActionType, AgentDecision
from app.agent.observation import Observation
from app.config import settings
from app.llm.base import LLMProvider


class MistralProvider(LLMProvider):
    """
    Mistral AI provider using the official mistralai SDK (v2.x).

    Prompt strategy:
      - JSON mode (response_format={"type": "json_object"}) to constrain output
      - DECISION_JSON_SCHEMA embedded in system prompt
      - _strip_fences() handles accidental markdown wrapping

    Error handling:
      - Any SDK/network error → returns ASK_CLARIFICATION fallback
      - JSON parse error → returns ASK_CLARIFICATION fallback
      - Validation error → returns ASK_CLARIFICATION fallback
      - NEVER raises — the agent loop must always get a decision
    """

    _SCHEMA_ADDENDUM = f"""
## REQUIRED OUTPUT FORMAT

You MUST respond with a valid JSON object ONLY. No markdown, no code fences, no explanation.

The JSON must contain exactly these fields:
```
{{
  "action": one of {[a.value for a in ActionType]},
  "tool_name": string or null,
  "arguments": object or null,
  "response_text": string or null,
  "reasoning_summary": "one-sentence explanation of your decision",
  "confidence": 0.0 to 1.0
}}
```

Rules:
- action="tool_call"  → tool_name and arguments are REQUIRED
- action="speak"      → response_text is REQUIRED
- action="ask_clarification" → response_text is REQUIRED
- action="escalate"   → response_text is REQUIRED
- action="end_call"   → response_text is REQUIRED
- reasoning_summary is ALWAYS required
- Respond with ONLY the JSON. Nothing else.
"""

    def __init__(
        self,
        api_key: str | None = None,
        model: str | None = None,
    ) -> None:
        from mistralai.client import Mistral
        self._client = Mistral(api_key=api_key or settings.mistral_api_key)
        self._model = model or settings.mistral_model

    def structured_decision(
        self,
        system_prompt: str,
        observation: Observation,
    ) -> AgentDecision:
        full_system = system_prompt + self._SCHEMA_ADDENDUM
        messages = self._build_messages(observation)
        all_messages = [{"role": "system", "content": full_system}] + messages

        try:
            response = self._client.chat.complete(
                model=self._model,
                messages=all_messages,  # type: ignore[arg-type]
                temperature=settings.llm_temperature,
                max_tokens=settings.llm_max_tokens,
                response_format={"type": "json_object"},
            )
            raw = response.choices[0].message.content or "{}"
            content = _strip_fences(raw)
            data = json.loads(content)
            return AgentDecision(**data)

        except (json.JSONDecodeError, ValueError) as exc:
            return _fallback(f"Parse/validation error: {exc}")
        except Exception as exc:
            return _fallback(f"Mistral API error: {exc}")

    def _build_messages(self, observation: Observation) -> list[dict[str, str]]:
        """Convert observation into the message list for the chat API."""
        messages = list(observation.recent_messages)

        # Append the current observation context as a user message
        context_parts = []

        if observation.last_tool_result:
            tr = observation.last_tool_result
            status = "succeeded" if tr.success else "FAILED"
            context_parts.append(
                f"[TOOL RESULT] '{tr.tool_name}' {status}: {json.dumps(tr.data)}"
            )

        if observation.latest_user_message:
            context_parts.append(f"[CUSTOMER] {observation.latest_user_message}")

        context_parts.append(
            f"[CONTEXT] Iteration {observation.iteration} | "
            f"Turns left: {observation.turns_remaining} | "
            f"Tool calls left: {observation.tool_calls_remaining} | "
            f"Verified: {observation.is_verified}"
        )

        context_parts.append("[DECISION] What is your next action?")

        messages.append({"role": "user", "content": "\n".join(context_parts)})
        return messages

    @property
    def provider_name(self) -> str:
        return f"MistralProvider({self._model})"


# ── Helpers ────────────────────────────────────────────────────────────────────


def _strip_fences(text: str) -> str:
    """Remove ```json ... ``` or ``` ... ``` wrappers if present."""
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```[a-z]*\n?", "", text)
        text = re.sub(r"\n?```$", "", text)
    return text.strip()


def _fallback(reason: str) -> AgentDecision:
    """Return a safe fallback decision when the LLM fails."""
    return AgentDecision(
        action=ActionType.ASK_CLARIFICATION,
        response_text="I'm sorry, I had a technical issue. Could you please repeat that?",
        reasoning_summary=reason,
        confidence=0.0,
    )
