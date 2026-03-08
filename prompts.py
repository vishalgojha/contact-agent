"""Prompt templates for the Contact Name Recovery Agent."""
# contact_name_agent/prompts.py

from __future__ import annotations

import json
from collections.abc import Sequence
from typing import Any

SYSTEM_PROMPT = """You are a careful, privacy-first personal contact recovery assistant.

Your job is to recover the most likely name for a phone number while minimizing privacy risk.

Rules:
- Prefer user-provided local files over any web-derived evidence.
- Only choose `search_web` when the caller explicitly allows web access.
- Never overstate confidence. Unknown is acceptable when evidence is weak.
- Ask the user when evidence is conflicting or incomplete.
- Output strict JSON only.
"""

NAME_EXTRACTION_SYSTEM_PROMPT = """You extract likely human contact names from noisy public snippets.

Rules:
- Return only plausible person or business contact names.
- Ignore generic phrases like Mobile Number, Contact Details, WhatsApp, India, Broker, Agent, and property jargon unless it is part of an actual name.
- Prefer concise names exactly as they appear in the snippet.
- Output strict JSON only.
"""


def render_tool_catalog(tool_catalog: Sequence[dict[str, Any]]) -> str:
    """Render the tool catalog for prompt inclusion."""
    return json.dumps(list(tool_catalog), indent=2, ensure_ascii=True)


def build_planner_prompt(
    *,
    masked_phone: str,
    allow_web: bool,
    state: dict[str, Any],
    tool_catalog: Sequence[dict[str, Any]],
) -> str:
    """Build the next-action planning prompt for the agent loop."""
    return f"""Current contact: {masked_phone}
Web search allowed: {str(allow_web).lower()}

Current state:
{json.dumps(state, indent=2, ensure_ascii=True)}

Available tools:
{render_tool_catalog(tool_catalog)}

Choose the next single best action.

Allowed action values:
- search_web
- extract_names_from_snippets
- ask_user
- finalize

Return JSON with exactly these keys:
{{
  "thought": "short internal reasoning",
  "action": "one allowed action",
  "action_input": {{}},
  "reasoning": "why this action is appropriate",
  "tentative_confidence": 0.0
}}

Guidance:
- Use `finalize` immediately for a direct local match.
- If the evidence is insufficient and web search is disallowed or already exhausted, either ask the user or finalize as UNKNOWN.
- `finalize` action_input may include `name`, `confidence`, and `source`.
"""


def build_reflection_prompt(
    *,
    masked_phone: str,
    state: dict[str, Any],
    last_action: str,
    last_result: str,
) -> str:
    """Build a brief reflection prompt after a tool action."""
    return f"""Reflect on the last action for contact {masked_phone}.

State now:
{json.dumps(state, indent=2, ensure_ascii=True)}

Last action: {last_action}
Last result: {last_result}

Return strict JSON:
{{
  "reflection": "one or two short sentences",
  "should_continue": true
}}

Set `should_continue` to false if the agent should finalize now.
"""


def build_name_extraction_prompt(phone: str, snippets: Sequence[str]) -> str:
    """Build the candidate-name extraction prompt for snippet parsing."""
    return f"""Phone: {phone}

Snippets:
{json.dumps(list(snippets), indent=2, ensure_ascii=True)}

Extract likely contact names from the snippets.

Return strict JSON:
{{
  "candidates": ["Name One", "Name Two"]
}}

Return an empty list when no plausible names exist.
"""
