"""Persistent local memory for resolved contacts and masked session history."""
# contact_name_agent/memory.py

from __future__ import annotations

import json
import uuid
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from utils import mask_phone, utc_now_iso


class ContactRecord(BaseModel):
    """Resolved contact data stored across runs."""

    phone: str
    name: str
    confidence: float = Field(ge=0.0, le=1.0)
    source: str
    reasoning: str
    last_updated: str


class SessionEvent(BaseModel):
    """Single masked event entry for session history."""

    timestamp: str
    phone_masked: str
    action: str
    outcome: str
    note: str


class SessionRecord(BaseModel):
    """High-level summary of one agent run."""

    session_id: str
    started_at: str
    completed_at: str | None = None
    model: str
    allow_web: bool
    input_path: str
    exported_path: str | None = None
    summary: str = ""
    events: list[SessionEvent] = Field(default_factory=list)


class MemoryStore:
    """Manage JSON-backed contact memory and session history."""

    def __init__(self, root: str | Path):
        self.root = Path(root)
        self.resolved_path = self.root / "resolved_contacts.json"
        self.history_path = self.root / "session_history.json"
        self.preferences_path = self.root / "preferences.json"
        self._ensure_files()

    def _ensure_files(self) -> None:
        self.root.mkdir(parents=True, exist_ok=True)
        if not self.resolved_path.exists():
            self._write_json(self.resolved_path, [])
        if not self.history_path.exists():
            self._write_json(self.history_path, [])
        if not self.preferences_path.exists():
            self._write_json(self.preferences_path, {})

    def _read_json(self, path: Path, default: Any) -> Any:
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except (FileNotFoundError, json.JSONDecodeError):
            return default

    def _write_json(self, path: Path, payload: Any) -> None:
        path.write_text(json.dumps(payload, indent=2, ensure_ascii=True), encoding="utf-8")

    def all_contacts(self) -> list[ContactRecord]:
        """Load all stored contact resolutions."""
        return [ContactRecord.model_validate(item) for item in self._read_json(self.resolved_path, [])]

    def get_contact(self, phone: str) -> ContactRecord | None:
        """Get a stored resolution by E.164 number."""
        for record in self.all_contacts():
            if record.phone == phone:
                return record
        return None

    def save_contact(self, record: ContactRecord) -> ContactRecord:
        """Insert or update a resolved contact."""
        contacts = self.all_contacts()
        replaced = False
        for index, existing in enumerate(contacts):
            if existing.phone == record.phone:
                contacts[index] = record
                replaced = True
                break
        if not replaced:
            contacts.append(record)
        self._write_json(self.resolved_path, [item.model_dump() for item in contacts])
        return record

    def start_session(self, *, model: str, allow_web: bool, input_path: str) -> str:
        """Create and persist a new session entry."""
        session_id = f"session-{uuid.uuid4().hex[:10]}"
        sessions = self._load_sessions()
        sessions.append(
            SessionRecord(
                session_id=session_id,
                started_at=utc_now_iso(),
                model=model,
                allow_web=allow_web,
                input_path=input_path,
            )
        )
        self._save_sessions(sessions)
        return session_id

    def append_session_event(
        self,
        session_id: str,
        *,
        action: str,
        outcome: str,
        note: str,
        phone: str | None = None,
    ) -> None:
        """Append a masked event to an existing session."""
        sessions = self._load_sessions()
        for session in sessions:
            if session.session_id != session_id:
                continue
            session.events.append(
                SessionEvent(
                    timestamp=utc_now_iso(),
                    phone_masked=mask_phone(phone or ""),
                    action=action,
                    outcome=outcome,
                    note=note,
                )
            )
            break
        self._save_sessions(sessions)

    def finish_session(self, session_id: str, *, exported_path: str, summary: str) -> None:
        """Mark a session as completed."""
        sessions = self._load_sessions()
        for session in sessions:
            if session.session_id != session_id:
                continue
            session.completed_at = utc_now_iso()
            session.exported_path = exported_path
            session.summary = summary
            break
        self._save_sessions(sessions)

    def get_default_region(self) -> str | None:
        """Load the persisted default region preference."""
        preferences = self._read_json(self.preferences_path, {})
        region = str(preferences.get("default_region") or "").strip().upper()
        return region or None

    def set_default_region(self, region: str) -> str:
        """Persist the default region preference."""
        preferences = self._read_json(self.preferences_path, {})
        preferences["default_region"] = region.strip().upper()
        self._write_json(self.preferences_path, preferences)
        return preferences["default_region"]

    def _load_sessions(self) -> list[SessionRecord]:
        return [SessionRecord.model_validate(item) for item in self._read_json(self.history_path, [])]

    def _save_sessions(self, sessions: list[SessionRecord]) -> None:
        self._write_json(self.history_path, [session.model_dump() for session in sessions])
