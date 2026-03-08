"""Main CLI and ReAct-style loop for the Contact Name Recovery Agent."""
# contact_name_agent/agent.py

from __future__ import annotations

import argparse
import json
import logging
import re
import sys
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field, ValidationError

from memory import ContactRecord, MemoryStore
from tui import ContactAgentTUI
from prompts import (
    NAME_EXTRACTION_SYSTEM_PROMPT,
    SYSTEM_PROMPT,
    build_name_extraction_prompt,
    build_planner_prompt,
    build_reflection_prompt,
)
from tools import (
    LoadedContact,
    TruecallerSearchResult,
    WebSnippet,
    ask_user,
    build_tool_catalog,
    export_resolved,
    extract_names_from_snippets,
    finalize,
    get_truecaller_status,
    load_contacts,
    match_google_contacts,
    run_truecaller_register,
    save_truecaller_installation_id,
    search_web,
    search_truecaller,
)
from utils import clamp_confidence, configure_logging, dedupe_strings, mask_phone
from utils import country_code_for_region, normalize_region_input

try:
    from ollama import Client as OllamaClient
except ImportError:  # pragma: no cover - optional dependency at runtime
    OllamaClient = None


class AgentAction(BaseModel):
    """Structured planner output for the next tool step."""

    thought: str = ""
    action: str
    action_input: dict[str, Any] = Field(default_factory=dict)
    reasoning: str = ""
    tentative_confidence: float = Field(default=0.0, ge=0.0, le=1.0)


class ReflectionResult(BaseModel):
    """Structured reflection output after a tool step."""

    reflection: str = ""
    should_continue: bool = True


class ContactState(BaseModel):
    """Mutable state tracked while resolving one contact."""

    phone: str
    original_value: str
    google_match: str | None = None
    memory_name: str | None = None
    memory_confidence: float | None = None
    searched_web: bool = False
    web_results: list[WebSnippet] = Field(default_factory=list)
    candidate_names: list[str] = Field(default_factory=list)
    user_response: str | None = None
    user_selected_name: str | None = None
    user_marked_unknown: bool = False
    last_action: str | None = None
    last_result: str | None = None
    reflections: list[str] = Field(default_factory=list)
    final_record: ContactRecord | None = None


class OllamaReasoner:
    """Thin wrapper around Ollama JSON-mode calls with safe fallback behavior."""

    def __init__(self, *, model: str, host: str, logger: logging.Logger):
        self.model = model
        self.host = host
        self.logger = logger
        self.client = OllamaClient(host=host) if OllamaClient is not None else None
        self._enabled = self.client is not None
        self._probe_complete = False
        self.disabled_reason = ""

    def is_available(self) -> bool:
        """Return true when the configured Ollama model is ready for use."""
        if not self._probe_complete:
            self.probe()
        return self._enabled

    def probe(self) -> bool:
        """Validate that Ollama is installed and the requested model is available."""
        if self._probe_complete:
            return self._enabled

        self._probe_complete = True
        if self.client is None:
            self._enabled = False
            self.disabled_reason = "Ollama client library is not installed."
            return False

        try:
            self.client.show(self.model)
        except Exception as exc:  # pragma: no cover - depends on local Ollama runtime
            self._enabled = False
            self.disabled_reason = str(exc)
            return False

        self._enabled = True
        self.disabled_reason = ""
        return True

    def plan_next_action(
        self,
        *,
        masked_phone: str,
        allow_web: bool,
        state: dict[str, Any],
        tool_catalog: list[dict[str, Any]],
    ) -> AgentAction | None:
        """Ask the local LLM for the next action."""
        payload = self._chat_json(
            system_prompt=SYSTEM_PROMPT,
            user_prompt=build_planner_prompt(
                masked_phone=masked_phone,
                allow_web=allow_web,
                state=state,
                tool_catalog=tool_catalog,
            ),
        )
        if payload is None:
            return None
        try:
            return AgentAction.model_validate(payload)
        except ValidationError:
            return None

    def reflect(
        self,
        *,
        masked_phone: str,
        state: dict[str, Any],
        last_action: str,
        last_result: str,
    ) -> ReflectionResult | None:
        """Ask the local LLM whether another step is needed."""
        payload = self._chat_json(
            system_prompt=SYSTEM_PROMPT,
            user_prompt=build_reflection_prompt(
                masked_phone=masked_phone,
                state=state,
                last_action=last_action,
                last_result=last_result,
            ),
        )
        if payload is None:
            return None
        try:
            return ReflectionResult.model_validate(payload)
        except ValidationError:
            return None

    def extract_name_candidates(self, phone: str, snippets: list[str]) -> list[str]:
        """Use the local LLM to extract candidate names from snippets."""
        payload = self._chat_json(
            system_prompt=NAME_EXTRACTION_SYSTEM_PROMPT,
            user_prompt=build_name_extraction_prompt(phone, snippets),
        )
        if not payload:
            return []
        raw_candidates = payload.get("candidates", [])
        if not isinstance(raw_candidates, list):
            return []
        return dedupe_strings(str(candidate) for candidate in raw_candidates)

    def _chat_json(self, *, system_prompt: str, user_prompt: str) -> dict[str, Any] | None:
        if not self.is_available() or self.client is None:
            return None

        try:
            response = self.client.chat(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                options={"temperature": 0.1},
                format="json",
            )
        except Exception as exc:  # pragma: no cover - depends on local Ollama runtime
            self._enabled = False
            self.disabled_reason = str(exc)
            self.logger.warning("Ollama disabled for this run after a failed call: %s", exc)
            return None

        content = self._extract_response_content(response)
        return self._parse_json(content)

    @staticmethod
    def _extract_response_content(response: Any) -> str:
        if isinstance(response, dict):
            message = response.get("message", {})
            return str(message.get("content") or "")
        message = getattr(response, "message", None)
        if message is not None:
            return str(getattr(message, "content", ""))
        return ""

    @staticmethod
    def _parse_json(content: str) -> dict[str, Any] | None:
        try:
            payload = json.loads(content)
            return payload if isinstance(payload, dict) else None
        except json.JSONDecodeError:
            match = re.search(r"\{.*\}", content, flags=re.DOTALL)
            if not match:
                return None
            try:
                payload = json.loads(match.group(0))
                return payload if isinstance(payload, dict) else None
            except json.JSONDecodeError:
                return None


class ContactRecoveryAgent:
    """Coordinate tool execution, short-term state, and persistent memory."""

    def __init__(
        self,
        *,
        input_path: Path,
        model: str,
        allow_web: bool,
        google_contacts_path: Path | None,
        default_region: str,
        max_iterations: int,
        memory_root: Path,
        output_root: Path,
        ollama_host: str,
        allow_truecaller: bool,
        truecaller_repo: Path,
    ):
        self.project_root = Path(__file__).resolve().parent
        self.input_path = input_path
        self.model = model
        self.allow_web = allow_web
        self.google_contacts_path = google_contacts_path
        self.default_region = default_region
        self.max_iterations = max(1, max_iterations)
        self.memory = MemoryStore(memory_root)
        self.output_root = output_root
        self.logger = configure_logging()
        self.reasoner = OllamaReasoner(model=model, host=ollama_host, logger=self.logger)
        self.tool_catalog = build_tool_catalog()
        self.session_id = ""
        self.interactive = sys.stdin.isatty()
        self.allow_truecaller = allow_truecaller
        self.truecaller_repo = truecaller_repo
        self.truecaller_ready = False
        self.truecaller_reason = ""
        self.last_resolved_records: list[ContactRecord] = []
        self.last_export_path: Path | None = None
        self.last_summary = ""

    def run(self) -> int:
        """Run the full agent workflow end-to-end."""
        contacts = load_contacts(str(self.input_path), self.default_region)
        if not contacts:
            self.logger.error("No valid phone numbers were found in %s", self.input_path)
            return 1

        google_index = self._load_google_index()
        self.session_id = self.memory.start_session(
            model=self.model,
            allow_web=self.allow_web,
            input_path=str(self.input_path),
        )

        if self.reasoner.is_available():
            self.logger.info("Using Ollama model %s for local reasoning.", self.model)
        else:
            reason = self.reasoner.disabled_reason or "Ollama is unavailable."
            self.logger.warning("Falling back to heuristic planning. Ollama unavailable: %s", reason)

        self._configure_truecaller()

        resolved: list[ContactRecord] = []
        for contact in contacts:
            try:
                resolved.append(self.resolve_contact(contact, google_index))
            except Exception as exc:  # pragma: no cover - safety net
                self.logger.exception("Unexpected error while resolving %s", mask_phone(contact.phone))
                fallback = finalize(
                    phone=contact.phone,
                    name="UNKNOWN",
                    confidence=0.0,
                    source="unknown",
                    reasoning=f"Resolution failed safely after an unexpected error: {exc}",
                    memory_store=self.memory,
                )
                resolved.append(fallback)
                self.memory.append_session_event(
                    self.session_id,
                    action="error_fallback",
                    outcome="safe_unknown",
                    note="Unexpected error. Stored UNKNOWN instead of crashing.",
                    phone=contact.phone,
                )

        export_path = export_resolved(resolved, self.output_root / "recovered.csv")
        known = sum(1 for record in resolved if record.name != "UNKNOWN")
        summary = f"Resolved {known} of {len(resolved)} contacts."
        self.last_resolved_records = resolved
        self.last_export_path = export_path
        self.last_summary = summary
        self.memory.finish_session(self.session_id, exported_path=str(export_path), summary=summary)
        self.logger.info(summary)
        self.logger.info("Exported results to %s", export_path)
        return 0

    def resolve_contact(self, contact: LoadedContact, google_index: dict[str, str]) -> ContactRecord:
        """Resolve one contact through a short ReAct loop."""
        masked_phone = mask_phone(contact.phone)
        self.logger.info("Resolving %s", masked_phone)

        memory_hit = self.memory.get_contact(contact.phone)
        state = ContactState(
            phone=contact.phone,
            original_value=contact.original_value,
            google_match=google_index.get(contact.phone),
            memory_name=memory_hit.name if memory_hit else None,
            memory_confidence=memory_hit.confidence if memory_hit else None,
            candidate_names=[memory_hit.name] if memory_hit and memory_hit.name != "UNKNOWN" else [],
        )

        if memory_hit and memory_hit.name != "UNKNOWN" and memory_hit.confidence >= 0.95:
            self.memory.append_session_event(
                self.session_id,
                action="reuse_memory",
                outcome="resolved",
                note=f"Reused existing high-confidence memory for {memory_hit.name}.",
                phone=contact.phone,
            )
            return memory_hit

        if state.google_match:
            record = finalize(
                phone=contact.phone,
                name=state.google_match,
                confidence=0.99,
                source="google_contacts",
                reasoning="Direct match from the user-provided Google Contacts file.",
                memory_store=self.memory,
            )
            self.memory.append_session_event(
                self.session_id,
                action="direct_google_match",
                outcome="resolved",
                note=f"Matched local Google Contacts entry for {record.name}.",
                phone=contact.phone,
            )
            return record

        truecaller_match = self._lookup_truecaller(contact.phone)
        if truecaller_match is not None:
            reasoning = "Matched via the vendored truecaller-cli lookup."
            if truecaller_match.email:
                reasoning += f" Truecaller also returned email {truecaller_match.email}."
            record = finalize(
                phone=contact.phone,
                name=truecaller_match.name,
                confidence=0.88,
                source="truecaller_cli",
                reasoning=reasoning,
                memory_store=self.memory,
            )
            self.memory.append_session_event(
                self.session_id,
                action="truecaller_lookup",
                outcome="resolved",
                note=f"Resolved with truecaller-cli as {record.name}.",
                phone=contact.phone,
            )
            return record

        for _ in range(self.max_iterations):
            if self._should_use_heuristic_planning(state):
                self.logger.info("Skipping LLM planning for %s because the next step is deterministic.", masked_phone)
                action = self._heuristic_action(state)
            else:
                self.logger.info("Planning next action for %s with Ollama.", masked_phone)
                action = self.reasoner.plan_next_action(
                    masked_phone=masked_phone,
                    allow_web=self.allow_web,
                    state=self._planner_state(state),
                    tool_catalog=self.tool_catalog,
                )
            action = self._sanitize_action(action, state)
            result_note = self._execute_action(state, action)
            if state.final_record is not None:
                state.last_action = action.action
                state.last_result = result_note
                self.memory.append_session_event(
                    self.session_id,
                    action=action.action,
                    outcome="completed",
                    note=result_note,
                    phone=contact.phone,
                )
                return state.final_record
            if self._should_use_heuristic_reflection(state):
                reflection = None
                reflection_text = self._heuristic_reflection(state, result_note)
            else:
                self.logger.info("Reflecting on the last step for %s with Ollama.", masked_phone)
                reflection = self.reasoner.reflect(
                    masked_phone=masked_phone,
                    state=self._planner_state(state),
                    last_action=action.action,
                    last_result=result_note,
                )
                reflection_text = (
                    reflection.reflection if reflection is not None else self._heuristic_reflection(state, result_note)
                )
            state.reflections.append(reflection_text)
            state.last_action = action.action
            state.last_result = result_note
            self.memory.append_session_event(
                self.session_id,
                action=action.action,
                outcome="completed",
                note=f"{result_note} Reflection: {reflection_text}",
                phone=contact.phone,
            )
            if reflection is not None and not reflection.should_continue:
                break

        return self._fallback_finalize(state)

    def _execute_action(self, state: ContactState, action: AgentAction) -> str:
        if action.action == "search_web":
            if not self.allow_web:
                return "Skipped web search because the CLI flag disabled it."
            try:
                max_results = int(action.action_input.get("max_results", 5))
                state.web_results = search_web(state.phone, max_results=max_results)
                state.searched_web = True
            except Exception as exc:
                state.searched_web = True
                return f"Web search failed safely: {exc}"
            if not state.web_results:
                return "Web search returned no usable snippets."
            return f"Web search returned {len(state.web_results)} snippets."

        if action.action == "extract_names_from_snippets":
            snippets = [f"{item.title} {item.snippet}".strip() for item in state.web_results]
            if not snippets:
                return "No snippets were available to parse."
            candidates = extract_names_from_snippets(
                snippets,
                phone=state.phone,
                llm_callable=self.reasoner.extract_name_candidates if self.reasoner.is_available() else None,
            )
            state.candidate_names = dedupe_strings([*state.candidate_names, *candidates])
            if not candidates:
                return "Snippet parsing found no plausible names."
            return f"Extracted {len(candidates)} candidate names."

        if action.action == "ask_user":
            if not self.interactive:
                return "Interactive input is unavailable in this session."
            question = str(action.action_input.get("question") or self._build_user_question(state))
            answer = ask_user(question)
            return self._apply_user_answer(state, answer)

        if action.action == "finalize":
            name = str(action.action_input.get("name") or self._best_candidate(state) or "UNKNOWN").strip()
            if not name:
                name = "UNKNOWN"
            if name == "UNKNOWN":
                confidence = 0.0
                source = "unknown"
            else:
                confidence = clamp_confidence(
                    action.action_input.get("confidence", self._estimate_candidate_confidence(state, name))
                )
                source = str(action.action_input.get("source") or self._infer_source(state, name))
            reasoning = str(
                action.reasoning
                or action.action_input.get("reasoning")
                or self._compose_reasoning(state, name=name, confidence=confidence, source=source)
            )
            state.final_record = finalize(
                phone=state.phone,
                name=name,
                confidence=confidence,
                source=source,
                reasoning=reasoning,
                memory_store=self.memory,
            )
            return f"Finalized as {name} with confidence {confidence:.2f} from {source}."

        return "Planner requested an unknown action; no tool was executed."

    def _sanitize_action(self, action: AgentAction | None, state: ContactState) -> AgentAction:
        if action is None:
            return self._heuristic_action(state)

        valid_actions = {"search_web", "extract_names_from_snippets", "ask_user", "finalize"}
        if action.action not in valid_actions:
            return self._heuristic_action(state)
        if action.action == "search_web" and not self.allow_web:
            return self._heuristic_action(state)
        return action

    def _heuristic_action(self, state: ContactState) -> AgentAction:
        if state.user_marked_unknown:
            return AgentAction(
                action="finalize",
                action_input={
                    "name": "UNKNOWN",
                    "confidence": 0.0,
                    "source": "unknown",
                },
                reasoning="The user explicitly marked this contact as unknown.",
                tentative_confidence=0.0,
            )

        if state.user_selected_name:
            return AgentAction(
                action="finalize",
                action_input={
                    "name": state.user_selected_name,
                    "confidence": 0.9,
                    "source": "user_input",
                },
                reasoning="The user supplied or confirmed the name directly.",
                tentative_confidence=0.9,
            )

        if state.memory_name and state.memory_name != "UNKNOWN" and (state.memory_confidence or 0.0) >= 0.85:
            return AgentAction(
                action="finalize",
                action_input={
                    "name": state.memory_name,
                    "confidence": state.memory_confidence,
                    "source": "memory",
                },
                reasoning="A prior high-confidence local memory match already exists.",
                tentative_confidence=state.memory_confidence or 0.85,
            )

        if not state.searched_web and self.allow_web:
            return AgentAction(
                action="search_web",
                action_input={"max_results": 5},
                reasoning="No strong local signal exists yet, so opt-in public search is the next step.",
                tentative_confidence=0.35,
            )

        if state.web_results and not state.candidate_names:
            return AgentAction(
                action="extract_names_from_snippets",
                action_input={},
                reasoning="The agent has snippets but no extracted candidate names yet.",
                tentative_confidence=0.4,
            )

        if len(state.candidate_names) == 1:
            candidate = state.candidate_names[0]
            return AgentAction(
                action="finalize",
                action_input={
                    "name": candidate,
                    "confidence": self._estimate_candidate_confidence(state, candidate),
                    "source": self._infer_source(state, candidate),
                },
                reasoning="One candidate remains after applying the available signals.",
                tentative_confidence=self._estimate_candidate_confidence(state, candidate),
            )

        if len(state.candidate_names) > 1 and self.interactive:
            return AgentAction(
                action="ask_user",
                action_input={"question": self._build_user_question(state)},
                reasoning="Multiple candidates remain, so user disambiguation is safer than guessing.",
                tentative_confidence=0.25,
            )

        if not state.candidate_names and self.interactive:
            return AgentAction(
                action="ask_user",
                action_input={"question": self._build_user_question(state)},
                reasoning="No reliable evidence exists; the user may know the contact directly.",
                tentative_confidence=0.1,
            )

        return AgentAction(
            action="finalize",
            action_input={"name": "UNKNOWN", "confidence": 0.0, "source": "unknown"},
            reasoning="The available evidence is too weak to justify a name.",
            tentative_confidence=0.0,
        )

    def _should_use_heuristic_planning(self, state: ContactState) -> bool:
        """Skip the LLM when the next step is obvious from local state."""
        if not self.reasoner.is_available():
            return True
        if state.user_marked_unknown:
            return True
        if state.user_selected_name:
            return True
        if state.memory_name and state.memory_name != "UNKNOWN" and (state.memory_confidence or 0.0) >= 0.85:
            return True
        if not self.allow_web and not state.google_match and not state.web_results and not state.candidate_names:
            return True
        return False

    def _should_use_heuristic_reflection(self, state: ContactState) -> bool:
        """Skip reflective LLM calls when the agent has little or no evidence state."""
        if not self.reasoner.is_available():
            return True
        if not self.allow_web and not state.google_match and not state.web_results:
            return True
        return False

    def _heuristic_reflection(self, state: ContactState, result_note: str) -> str:
        if state.final_record is not None:
            return "Resolution is complete."
        if state.user_marked_unknown:
            return "The user marked this contact as unknown, so finalization should happen next."
        if state.user_selected_name:
            return "A direct user answer is available, so finalization should happen next."
        if len(state.candidate_names) > 1:
            return "Conflicting candidates remain. Ask the user or avoid overconfident guesses."
        if len(state.candidate_names) == 1:
            return "A single candidate exists. Finalization is reasonable if the source is acceptable."
        if state.searched_web and not state.web_results:
            return "Public search produced no usable evidence."
        return result_note

    def _fallback_finalize(self, state: ContactState) -> ContactRecord:
        candidate = self._best_candidate(state)
        if candidate:
            confidence = self._estimate_candidate_confidence(state, candidate)
            source = self._infer_source(state, candidate)
            reasoning = self._compose_reasoning(state, name=candidate, confidence=confidence, source=source)
            record = finalize(
                phone=state.phone,
                name=candidate,
                confidence=confidence,
                source=source,
                reasoning=reasoning,
                memory_store=self.memory,
            )
        else:
            record = finalize(
                phone=state.phone,
                name="UNKNOWN",
                confidence=0.0,
                source="unknown",
                reasoning="No reliable local match, user confirmation, or public evidence was available.",
                memory_store=self.memory,
            )
        state.final_record = record
        return record

    def _build_user_question(self, state: ContactState) -> str:
        masked_phone = mask_phone(state.phone)
        if len(state.candidate_names) == 1:
            candidate = state.candidate_names[0]
            return (
                f"For {masked_phone}, press Enter to accept '{candidate}', "
                "type a different name, or type unknown."
            )
        if len(state.candidate_names) > 1:
            candidates = ", ".join(state.candidate_names[:4])
            return f"For {masked_phone}, possible names are: {candidates}. Type the best name or type unknown."
        return (
            f"I could not recover a reliable name for {masked_phone}. "
            "Type a name if you know it, or press Enter to mark unknown."
        )

    def _apply_user_answer(self, state: ContactState, answer: str) -> str:
        normalized = answer.strip()
        state.user_response = normalized
        if not normalized and state.candidate_names:
            state.user_marked_unknown = False
            state.user_selected_name = state.candidate_names[0]
            return f"User accepted the suggested name {state.user_selected_name}."
        if not normalized:
            state.user_marked_unknown = True
            state.user_selected_name = None
            return "User left the name blank, so the contact was marked unknown."

        lowered = normalized.lower()
        if lowered in {"unknown", "skip", "n/a", "none", "no"}:
            state.user_marked_unknown = True
            state.user_selected_name = None
            return "User marked the contact as unknown."

        if lowered in {"yes", "y"} and state.candidate_names:
            state.user_marked_unknown = False
            state.user_selected_name = state.candidate_names[0]
            return f"User confirmed the suggested name {state.user_selected_name}."

        state.user_marked_unknown = False
        state.user_selected_name = normalized
        state.candidate_names = dedupe_strings([normalized, *state.candidate_names])
        return f"User supplied the name {normalized}."

    def _planner_state(self, state: ContactState) -> dict[str, Any]:
        return {
            "memory_name": state.memory_name,
            "memory_confidence": state.memory_confidence,
            "google_match": state.google_match,
            "searched_web": state.searched_web,
            "candidate_names": state.candidate_names,
            "user_response": state.user_response,
            "user_selected_name": state.user_selected_name,
            "web_results": [
                {"title": result.title, "snippet": result.snippet, "url": result.url}
                for result in state.web_results[:5]
            ],
            "recent_reflections": state.reflections[-3:],
            "last_action": state.last_action,
            "last_result": state.last_result,
        }

    def _best_candidate(self, state: ContactState) -> str | None:
        if state.user_selected_name:
            return state.user_selected_name
        if not state.candidate_names:
            return state.memory_name if state.memory_name and state.memory_name != "UNKNOWN" else None
        ranked = sorted(
            state.candidate_names,
            key=lambda candidate: self._estimate_candidate_confidence(state, candidate),
            reverse=True,
        )
        return ranked[0]

    def _estimate_candidate_confidence(self, state: ContactState, name: str) -> float:
        if state.user_selected_name == name:
            return 0.9
        if state.google_match == name:
            return 0.99
        if state.memory_name == name and state.memory_confidence is not None:
            return clamp_confidence(max(state.memory_confidence, 0.75))

        score = 0.2
        if state.web_results:
            mentions = sum(
                1
                for result in state.web_results
                if name.lower() in f"{result.title} {result.snippet}".lower()
            )
            if mentions >= 3:
                score = 0.75
            elif mentions == 2:
                score = 0.65
            elif mentions == 1:
                score = 0.55
        if len(state.candidate_names) > 1:
            score -= 0.1
        return clamp_confidence(score)

    def _infer_source(self, state: ContactState, name: str) -> str:
        if state.user_selected_name == name:
            return "user_input"
        if state.google_match == name:
            return "google_contacts"
        if state.memory_name == name:
            return "memory"
        if state.web_results:
            return "web_search"
        return "unknown"

    def _compose_reasoning(self, state: ContactState, *, name: str, confidence: float, source: str) -> str:
        if name == "UNKNOWN":
            return "The agent did not find enough reliable evidence to recover a name safely."
        fragments = [f"Selected '{name}' with confidence {confidence:.2f}."]
        if source == "google_contacts":
            fragments.append("A direct match existed in the user-provided Google Contacts file.")
        elif source == "memory":
            fragments.append("A prior local memory record already matched this number.")
        elif source == "user_input":
            fragments.append("The user directly supplied or confirmed the name.")
        elif source == "web_search":
            fragments.append("Public snippets consistently suggested this name after quoted-number search.")
        if state.candidate_names:
            fragments.append(f"Candidate pool: {', '.join(state.candidate_names[:4])}.")
        return " ".join(fragments)

    def _load_google_index(self) -> dict[str, str]:
        if self.google_contacts_path is None:
            return {}
        if not self.google_contacts_path.exists():
            self.logger.warning("Google Contacts file not found: %s", self.google_contacts_path)
            return {}
        matches = match_google_contacts(str(self.google_contacts_path), self.default_region)
        self.logger.info("Loaded %d local Google Contacts matches.", len(matches))
        return matches

    def _configure_truecaller(self) -> None:
        if not self.allow_truecaller:
            return
        status = get_truecaller_status(self.truecaller_repo)
        self.truecaller_ready = status.ready
        self.truecaller_reason = status.reason
        if self.truecaller_ready:
            self.logger.info("Truecaller CLI lookup is enabled.")
        else:
            self.logger.warning("Truecaller CLI lookup requested but unavailable: %s", status.reason)

    def _lookup_truecaller(self, phone: str) -> TruecallerSearchResult | None:
        if not self.allow_truecaller or not self.truecaller_ready:
            return None
        try:
            return search_truecaller(phone, self.truecaller_repo)
        except Exception as exc:
            self.truecaller_ready = False
            self.truecaller_reason = str(exc)
            self.logger.warning("Disabling Truecaller CLI for this run: %s", exc)
            return None


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(description="Local-first Contact Name Recovery Agent")
    subparsers = parser.add_subparsers(dest="command", required=True)

    subparsers.add_parser("onboard", help="Launch the interactive setup wizard and then run recovery.")

    run_parser = subparsers.add_parser("run", help="Recover names for phone numbers.")
    run_parser.add_argument("--input", required=True, help="Path to numbers.csv or an input directory.")
    run_parser.add_argument("--model", default="llama3.2:latest", help="Ollama model name.")
    run_parser.add_argument("--allow-web", choices=["yes", "no"], default="no", help="Allow opt-in public search.")
    run_parser.add_argument("--google-contacts", help="Optional Google Contacts CSV or VCF file.")
    run_parser.add_argument(
        "--default-region",
        default="",
        help="Default region for parsing local numbers, for example IN, US, or +91.",
    )
    run_parser.add_argument("--max-iterations", type=int, default=4, help="Maximum agent loop steps per contact.")
    run_parser.add_argument("--ollama-host", default="http://127.0.0.1:11434", help="Ollama host URL.")
    run_parser.add_argument(
        "--allow-truecaller",
        choices=["yes", "no"],
        default="yes",
        help="Allow lookups via the vendored truecaller-cli after setup.",
    )

    setup_parser = subparsers.add_parser(
        "setup-truecaller",
        help="Register the vendored truecaller-cli with your own 10-digit Indian number.",
    )
    setup_parser.add_argument("--number", required=True, help="Your 10-digit Indian mobile number for OTP setup.")

    token_parser = subparsers.add_parser(
        "setup-truecaller-token",
        help="Save a manual Truecaller installationId for the vendored truecaller-cli.",
    )
    token_parser.add_argument("--installation-id", required=True, help="Manual Truecaller installationId token.")
    return parser.parse_args()


def resolve_default_region(memory_store: MemoryStore, supplied_region: str, interactive: bool) -> str:
    """Resolve the default parsing region from CLI input, memory, or one-time prompt."""
    configured = normalize_region_input(supplied_region)
    if configured is not None:
        memory_store.set_default_region(configured)
        return configured

    saved = memory_store.get_default_region()
    if saved is not None:
        return saved

    if interactive:
        while True:
            answer = input(
                "Enter your default country once for phone parsing (examples: IN, US, +91):\n> "
            ).strip()
            configured = normalize_region_input(answer)
            if configured is None:
                print("Please enter a valid ISO country code like IN or US, or a dialing code like +91.")
                continue
            memory_store.set_default_region(configured)
            return configured

    memory_store.set_default_region("IN")
    return "IN"


def resolve_google_contacts_path(raw_path: str | None, project_root: Path, input_path: Path) -> Path | None:
    """Resolve the optional Google Contacts file path."""
    if raw_path:
        return Path(raw_path).resolve()

    candidates = [
        project_root / "inputs" / "google_contacts.xlsx",
        project_root / "inputs" / "google_contacts.csv",
        project_root / "inputs" / "google_contacts.vcf",
    ]
    if input_path.is_dir():
        candidates.extend(
            [
                input_path / "google_contacts.xlsx",
                input_path / "google_contacts.csv",
                input_path / "google_contacts.vcf",
            ]
        )

    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()
    return None


def build_agent(
    *,
    project_root: Path,
    input_path: Path,
    model: str,
    allow_web: bool,
    google_contacts_path: Path | None,
    default_region: str,
    max_iterations: int,
    ollama_host: str,
    allow_truecaller: bool,
) -> ContactRecoveryAgent:
    """Construct the contact recovery agent from resolved configuration."""
    return ContactRecoveryAgent(
        input_path=input_path,
        model=model,
        allow_web=allow_web,
        google_contacts_path=google_contacts_path,
        default_region=default_region,
        max_iterations=max_iterations,
        memory_root=project_root / "memory",
        output_root=project_root / "output",
        ollama_host=ollama_host,
        allow_truecaller=allow_truecaller,
        truecaller_repo=project_root / "_vendor" / "truecaller-cli",
    )


def main() -> int:
    """CLI entrypoint."""
    args = parse_args()
    project_root = Path(__file__).resolve().parent
    truecaller_repo = project_root / "_vendor" / "truecaller-cli"
    memory_root = project_root / "memory"
    memory_store = MemoryStore(memory_root)
    interactive = sys.stdin.isatty()

    if args.command == "onboard":
        logger = configure_logging()
        if not interactive:
            logger.error("The onboarding wizard requires an interactive terminal session.")
            return 1
        truecaller_status = get_truecaller_status(truecaller_repo)
        tui = ContactAgentTUI(project_root)
        config = tui.run_onboarding(
            memory_store=memory_store,
            truecaller_ready=truecaller_status.ready,
            truecaller_reason=truecaller_status.reason,
        )
        if config is None:
            return 0
        memory_store.set_default_region(config.default_region)
        agent = build_agent(
            project_root=project_root,
            input_path=config.input_path,
            model=config.model,
            allow_web=config.allow_web,
            google_contacts_path=config.google_contacts_path,
            default_region=config.default_region,
            max_iterations=config.max_iterations,
            ollama_host=config.ollama_host,
            allow_truecaller=config.allow_truecaller,
        )
        exit_code = agent.run()
        tui.show_run_summary(records=agent.last_resolved_records, export_path=agent.last_export_path)
        return exit_code

    default_region = resolve_default_region(memory_store, getattr(args, "default_region", ""), interactive)

    if args.command == "setup-truecaller":
        logger = configure_logging()
        status = get_truecaller_status(truecaller_repo)
        if "dependencies are not installed" in status.reason:
            logger.error(status.reason)
            return 1
        if default_region != "IN":
            dialing_code = country_code_for_region(default_region)
            logger.error(
                "The vendored truecaller-cli is hardcoded for India. Your saved region is %s (+%s). "
                "Use an Indian region or skip Truecaller for this project.",
                default_region,
                dialing_code or "?",
            )
            return 1
        return run_truecaller_register(args.number, truecaller_repo)

    if args.command == "setup-truecaller-token":
        logger = configure_logging()
        status = get_truecaller_status(truecaller_repo)
        if "dependencies are not installed" in status.reason:
            logger.error(status.reason)
            return 1
        try:
            config_path = save_truecaller_installation_id(args.installation_id, truecaller_repo)
        except ValueError as exc:
            logger.error(str(exc))
            return 1
        logger.info("Saved manual Truecaller installationId to %s", config_path)
        return 0

    if args.command != "run":
        return 1

    input_path = Path(args.input).resolve()
    google_contacts_path = resolve_google_contacts_path(args.google_contacts, project_root, input_path)

    agent = build_agent(
        project_root=project_root,
        input_path=input_path,
        model=args.model,
        allow_web=args.allow_web == "yes",
        google_contacts_path=google_contacts_path,
        default_region=default_region,
        max_iterations=args.max_iterations,
        ollama_host=args.ollama_host,
        allow_truecaller=args.allow_truecaller == "yes",
    )
    return agent.run()


if __name__ == "__main__":
    raise SystemExit(main())
