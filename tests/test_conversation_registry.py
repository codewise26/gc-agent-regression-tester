"""Unit tests for the conversation registry."""

import json

from src.conversation_registry import ConversationRegistry


class TestConversationRegistry:
    def test_add_and_list(self, tmp_path):
        registry = ConversationRegistry(file_path=str(tmp_path / "active.json"))

        registry.add("conv-1", "Scenario A", 1)
        registry.add("conv-2", "Scenario B", 2)

        entries = registry.list_entries()
        assert len(entries) == 2
        assert entries[0].conversation_id == "conv-1"
        assert entries[1].scenario_name == "Scenario B"

    def test_remove(self, tmp_path):
        registry = ConversationRegistry(file_path=str(tmp_path / "active.json"))
        registry.add("conv-1", "Scenario A", 1)

        assert registry.remove("conv-1") is True
        assert registry.list_entries() == []
        assert registry.remove("conv-1") is False

    def test_persistence(self, tmp_path):
        path = tmp_path / "active.json"
        registry = ConversationRegistry(file_path=str(path))
        registry.add("conv-1", "Scenario A", 1)

        reloaded = ConversationRegistry(file_path=str(path))
        entries = reloaded.list_entries()
        assert len(entries) == 1
        assert entries[0].conversation_id == "conv-1"

    def test_remove_all(self, tmp_path):
        registry = ConversationRegistry(file_path=str(tmp_path / "active.json"))
        registry.add("conv-1", "Scenario A", 1)
        registry.add("conv-2", "Scenario B", 2)

        removed = registry.remove_all()
        assert len(removed) == 2
        assert registry.list_entries() == []

    def test_add_replaces_existing_id(self, tmp_path):
        registry = ConversationRegistry(file_path=str(tmp_path / "active.json"))
        registry.add("conv-1", "Scenario A", 1)
        registry.add("conv-1", "Scenario A", 2)

        entries = registry.list_entries()
        assert len(entries) == 1
        assert entries[0].attempt_number == 2

    def test_load_invalid_file(self, tmp_path):
        path = tmp_path / "active.json"
        path.write_text("not-json", encoding="utf-8")

        registry = ConversationRegistry(file_path=str(path))
        assert registry.list_entries() == []

    def test_saved_file_format(self, tmp_path):
        path = tmp_path / "active.json"
        registry = ConversationRegistry(file_path=str(path))
        registry.add("conv-1", "Scenario A", 1)

        data = json.loads(path.read_text(encoding="utf-8"))
        assert "conversations" in data
        assert data["conversations"][0]["conversation_id"] == "conv-1"
