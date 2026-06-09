"""Shared CLI command helpers for conversation management."""

from typing import Optional

from .app_config import load_app_config, validate_platform_config
from .conversation_registry import ConversationEntry, ConversationRegistry
from .platform_api_client import PlatformApiClient, PlatformApiError


def get_registry(config) -> ConversationRegistry:
    """Build a ConversationRegistry from app config."""
    return ConversationRegistry(file_path=config.gc_conversations_file)


def get_platform_client(config) -> PlatformApiClient:
    """Build a PlatformApiClient from app config."""
    return PlatformApiClient(
        region=config.gc_region or "",
        client_id=config.gc_client_id or "",
        client_secret=config.gc_client_secret or "",
        timeout=config.response_timeout,
    )


def list_conversations(config) -> list[ConversationEntry]:
    """Return active conversations from the registry."""
    return get_registry(config).list_entries()


def disconnect_conversation(
    config,
    conversation_id: str,
    *,
    registry: Optional[ConversationRegistry] = None,
    platform_client: Optional[PlatformApiClient] = None,
) -> None:
    """Disconnect one conversation and remove it from the registry."""
    missing = validate_platform_config(config)
    if missing:
        raise PlatformApiError(
            f"Missing required Platform API configuration: {', '.join(missing)}"
        )

    registry = registry or get_registry(config)
    platform_client = platform_client or get_platform_client(config)

    platform_client.disconnect_conversation(conversation_id)
    registry.remove(conversation_id)


def disconnect_all_conversations(
    config,
    *,
    registry: Optional[ConversationRegistry] = None,
    platform_client: Optional[PlatformApiClient] = None,
) -> tuple[int, list[str]]:
    """Disconnect all tracked conversations.

    Returns:
        Tuple of (success_count, error_messages).
    """
    missing = validate_platform_config(config)
    if missing:
        raise PlatformApiError(
            f"Missing required Platform API configuration: {', '.join(missing)}"
        )

    registry = registry or get_registry(config)
    platform_client = platform_client or get_platform_client(config)

    entries = registry.remove_all()
    success_count = 0
    errors: list[str] = []

    for entry in entries:
        try:
            platform_client.disconnect_conversation(entry.conversation_id)
            success_count += 1
        except PlatformApiError as e:
            registry.add(
                entry.conversation_id,
                entry.scenario_name,
                entry.attempt_number,
            )
            errors.append(f"{entry.conversation_id}: {e}")

    return success_count, errors


def load_config_for_management():
    """Load app config for conversation management commands."""
    return load_app_config()
