"""Genesys Cloud Platform API client for conversation resolution and disconnect."""

import time
from typing import Any, Optional

import requests


class PlatformApiError(Exception):
    """Custom exception for Genesys Cloud Platform API errors."""

    pass


class PlatformApiClient:
    """OAuth-authenticated client for Genesys Cloud Platform API calls."""

    def __init__(
        self,
        region: str,
        client_id: str,
        client_secret: str,
        timeout: int = 30,
    ):
        """Initialize with OAuth credentials and region.

        Args:
            region: Genesys Cloud region (e.g. 'mypurecloud.com').
            client_id: OAuth client ID for client credentials grant.
            client_secret: OAuth client secret.
            timeout: HTTP request timeout in seconds.
        """
        self.region = region
        self.client_id = client_id
        self.client_secret = client_secret
        self.timeout = timeout
        self._access_token: Optional[str] = None
        self._token_expires_at: float = 0.0

    @property
    def _login_url(self) -> str:
        return f"https://login.{self.region}/oauth/token"

    @property
    def _api_base_url(self) -> str:
        return f"https://api.{self.region}"

    def _get_access_token(self) -> str:
        """Return a valid access token, refreshing if needed."""
        if self._access_token and time.time() < self._token_expires_at:
            return self._access_token

        try:
            response = requests.post(
                self._login_url,
                data={"grant_type": "client_credentials"},
                auth=(self.client_id, self.client_secret),
                timeout=self.timeout,
            )
            response.raise_for_status()
        except requests.RequestException as e:
            raise PlatformApiError(
                f"Failed to obtain OAuth token for region={self.region}: {e}"
            ) from e

        data = response.json()
        token = data.get("access_token")
        if not token:
            raise PlatformApiError(
                f"OAuth response missing access_token for region={self.region}"
            )

        expires_in = int(data.get("expires_in", 3600))
        self._access_token = token
        self._token_expires_at = time.time() + max(expires_in - 60, 0)
        return token

    def _request(
        self,
        method: str,
        path: str,
        *,
        params: Optional[dict[str, Any]] = None,
        json_body: Optional[dict[str, Any]] = None,
    ) -> requests.Response:
        """Make an authenticated Platform API request."""
        token = self._get_access_token()
        url = f"{self._api_base_url}{path}"
        headers = {"Authorization": f"Bearer {token}"}

        try:
            response = requests.request(
                method,
                url,
                headers=headers,
                params=params,
                json=json_body,
                timeout=self.timeout,
            )
        except requests.RequestException as e:
            raise PlatformApiError(
                f"Platform API request failed: {method} {path}: {e}"
            ) from e

        return response

    @staticmethod
    def _extract_conversation_id(data: dict[str, Any]) -> Optional[str]:
        """Extract conversationId from message details response."""
        for key in ("conversationId", "conversation_id"):
            value = data.get(key)
            if value:
                return str(value)

        conversation = data.get("conversation")
        if isinstance(conversation, dict):
            conv_id = conversation.get("id")
            if conv_id:
                return str(conv_id)

        return None

    def resolve_conversation_id(self, message_id: str) -> str:
        """Resolve conversationId from a Web Messaging messageId.

        Args:
            message_id: Message ID from a WebSocket agent message payload.

        Returns:
            The Genesys conversation ID.

        Raises:
            PlatformApiError: If the API call fails or conversationId is missing.
        """
        response = self._request(
            "GET",
            f"/api/v2/conversations/messages/{message_id}/details",
            params={"useNormalizedMessage": "true"},
        )

        if response.status_code == 404:
            raise PlatformApiError(
                f"Message not found: message_id={message_id}"
            )

        if not response.ok:
            raise PlatformApiError(
                f"Failed to resolve conversationId for message_id={message_id}: "
                f"HTTP {response.status_code} — {response.text[:200]}"
            )

        data = response.json()
        conversation_id = self._extract_conversation_id(data)
        if not conversation_id:
            raise PlatformApiError(
                f"Message details response missing conversationId: message_id={message_id}"
            )

        return conversation_id

    def disconnect_conversation(self, conversation_id: str) -> None:
        """Disconnect a conversation via Platform API teardown.

        Args:
            conversation_id: Genesys conversation ID to disconnect.

        Raises:
            PlatformApiError: If the API call fails (404 is treated as success).
        """
        response = self._request(
            "POST",
            f"/api/v2/conversations/{conversation_id}/disconnect",
        )

        if response.status_code in (404, 410):
            return

        if not response.ok:
            raise PlatformApiError(
                f"Failed to disconnect conversation_id={conversation_id}: "
                f"HTTP {response.status_code} — {response.text[:200]}"
            )
