"""Unit tests for conversation disconnect lifecycle in ConversationRunner."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.conversation_registry import ConversationRegistry
from src.conversation_runner import ConversationRunner
from src.judge_llm import GoalEvaluation
from src.models import TestScenario
from src.platform_api_client import PlatformApiClient
from src.web_messaging_client import AgentMessage


@pytest.fixture
def scenario():
    return TestScenario(
        name="Test Scenario",
        persona="A customer",
        goal="Get help",
        first_message="Hello",
        attempts=1,
    )


@pytest.fixture
def mock_platform_client():
    client = MagicMock(spec=PlatformApiClient)
    client.resolve_conversation_id.return_value = "conv-123"
    client.disconnect_conversation.return_value = None
    return client


@pytest.fixture
def registry(tmp_path):
    return ConversationRegistry(file_path=str(tmp_path / "active.json"))


def _build_runner(mock_platform_client, registry):
    judge = MagicMock()
    judge.model = "test-model"
    return ConversationRunner(
        judge=judge,
        web_msg_config={
            "region": "mypurecloud.com",
            "deployment_id": "dep-1",
            "timeout": 5,
            "origin": "https://localhost",
        },
        max_turns=1,
        platform_client=mock_platform_client,
        conversation_registry=registry,
    )


@pytest.mark.asyncio
async def test_disconnect_called_on_success(
    scenario, mock_platform_client, registry
):
    runner = _build_runner(mock_platform_client, registry)
    mock_client = AsyncMock()
    mock_client.wait_for_welcome.return_value = AgentMessage(
        text="Welcome!",
        message_id="msg-1",
    )
    mock_client.receive_response.return_value = AgentMessage(
        text="Agent reply",
        message_id="msg-2",
    )

    with patch(
        "src.conversation_runner.WebMessagingClient",
        return_value=mock_client,
    ), patch.object(
        runner,
        "_run_judge",
        new=AsyncMock(
            return_value=GoalEvaluation(success=True, explanation="Done")
        ),
    ):
        result = await runner.run_attempt(scenario, 1)

    assert result.success is True
    assert result.conversation_id == "conv-123"
    mock_platform_client.resolve_conversation_id.assert_called_once_with("msg-1")
    mock_platform_client.disconnect_conversation.assert_called_once_with("conv-123")
    assert registry.list_entries() == []


@pytest.mark.asyncio
async def test_disconnect_called_on_failure(
    scenario, mock_platform_client, registry
):
    runner = _build_runner(mock_platform_client, registry)
    mock_client = AsyncMock()
    mock_client.wait_for_welcome.return_value = AgentMessage(
        text="Welcome!",
        message_id="msg-1",
    )
    mock_client.receive_response.return_value = AgentMessage(
        text="Agent reply",
        message_id="msg-2",
    )

    with patch(
        "src.conversation_runner.WebMessagingClient",
        return_value=mock_client,
    ), patch.object(
        runner,
        "_run_judge",
        new=AsyncMock(
            return_value=GoalEvaluation(success=False, explanation="Not done")
        ),
    ):
        result = await runner.run_attempt(scenario, 1)

    assert result.success is False
    mock_platform_client.disconnect_conversation.assert_called_once_with("conv-123")
    assert registry.list_entries() == []


@pytest.mark.asyncio
async def test_registry_kept_when_disconnect_fails(
    scenario, mock_platform_client, registry
):
    from src.platform_api_client import PlatformApiError

    mock_platform_client.disconnect_conversation.side_effect = PlatformApiError(
        "disconnect failed"
    )
    runner = _build_runner(mock_platform_client, registry)
    mock_client = AsyncMock()
    mock_client.wait_for_welcome.return_value = AgentMessage(
        text="Welcome!",
        message_id="msg-1",
    )
    mock_client.receive_response.return_value = AgentMessage(
        text="Agent reply",
        message_id="msg-2",
    )

    with patch(
        "src.conversation_runner.WebMessagingClient",
        return_value=mock_client,
    ), patch.object(
        runner,
        "_run_judge",
        new=AsyncMock(
            return_value=GoalEvaluation(success=True, explanation="Done")
        ),
    ):
        result = await runner.run_attempt(scenario, 1)

    assert result.success is True
    entries = registry.list_entries()
    assert len(entries) == 1
    assert entries[0].conversation_id == "conv-123"
