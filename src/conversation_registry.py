"""Persisted registry of active Genesys conversations for cleanup."""

import json
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from pydantic import BaseModel


DEFAULT_REGISTRY_PATH = ".gc-tester/active_conversations.json"


class ConversationEntry(BaseModel):
    """A tracked active conversation."""

    conversation_id: str
    scenario_name: str
    attempt_number: int
    registered_at: str


class ConversationRegistry:
    """Thread-safe registry of active conversations backed by a JSON file."""

    def __init__(self, file_path: Optional[str] = None):
        self._path = Path(file_path or DEFAULT_REGISTRY_PATH)
        self._lock = threading.Lock()
        self._entries: list[ConversationEntry] = []
        self._load()

    def _load(self) -> None:
        if not self._path.exists():
            self._entries = []
            return

        try:
            raw = json.loads(self._path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            self._entries = []
            return

        items = raw.get("conversations", []) if isinstance(raw, dict) else []
        self._entries = [
            ConversationEntry(**item)
            for item in items
            if isinstance(item, dict)
        ]

    def _save(self) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "conversations": [entry.model_dump() for entry in self._entries]
        }
        self._path.write_text(
            json.dumps(payload, indent=2) + "\n",
            encoding="utf-8",
        )

    def add(
        self,
        conversation_id: str,
        scenario_name: str,
        attempt_number: int,
    ) -> None:
        """Register an active conversation."""
        entry = ConversationEntry(
            conversation_id=conversation_id,
            scenario_name=scenario_name,
            attempt_number=attempt_number,
            registered_at=datetime.now(timezone.utc).isoformat(),
        )
        with self._lock:
            self._entries = [
                e for e in self._entries if e.conversation_id != conversation_id
            ]
            self._entries.append(entry)
            self._save()

    def remove(self, conversation_id: str) -> bool:
        """Remove a conversation from the registry. Returns True if it was present."""
        with self._lock:
            before = len(self._entries)
            self._entries = [
                e for e in self._entries if e.conversation_id != conversation_id
            ]
            removed = len(self._entries) < before
            if removed:
                self._save()
            return removed

    def list_entries(self) -> list[ConversationEntry]:
        """Return a copy of all tracked conversations."""
        with self._lock:
            return list(self._entries)

    def remove_all(self) -> list[ConversationEntry]:
        """Clear the registry and return the entries that were removed."""
        with self._lock:
            entries = list(self._entries)
            self._entries = []
            self._save()
            return entries
