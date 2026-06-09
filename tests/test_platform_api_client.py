"""Unit tests for the Platform API client."""

from unittest.mock import MagicMock, patch

import pytest

from src.platform_api_client import PlatformApiClient, PlatformApiError


@pytest.fixture
def client():
    return PlatformApiClient(
        region="mypurecloud.com",
        client_id="client-id",
        client_secret="client-secret",
        timeout=5,
    )


class TestPlatformApiClientAuth:
    @patch("src.platform_api_client.requests.post")
    def test_get_access_token(self, mock_post, client):
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "access_token": "token-123",
            "expires_in": 3600,
        }
        mock_response.raise_for_status = MagicMock()
        mock_post.return_value = mock_response

        token = client._get_access_token()

        assert token == "token-123"
        mock_post.assert_called_once()

    @patch("src.platform_api_client.requests.post")
    def test_get_access_token_missing_token_raises(self, mock_post, client):
        mock_response = MagicMock()
        mock_response.json.return_value = {"expires_in": 3600}
        mock_response.raise_for_status = MagicMock()
        mock_post.return_value = mock_response

        with pytest.raises(PlatformApiError, match="missing access_token"):
            client._get_access_token()


class TestPlatformApiClientResolve:
    @patch.object(PlatformApiClient, "_request")
    def test_resolve_conversation_id(self, mock_request, client):
        mock_response = MagicMock()
        mock_response.ok = True
        mock_response.status_code = 200
        mock_response.json.return_value = {"conversationId": "conv-abc"}
        mock_request.return_value = mock_response

        conversation_id = client.resolve_conversation_id("msg-123")

        assert conversation_id == "conv-abc"
        mock_request.assert_called_once_with(
            "GET",
            "/api/v2/conversations/messages/msg-123/details",
            params={"useNormalizedMessage": "true"},
        )

    @patch.object(PlatformApiClient, "_request")
    def test_resolve_conversation_id_nested(self, mock_request, client):
        mock_response = MagicMock()
        mock_response.ok = True
        mock_response.status_code = 200
        mock_response.json.return_value = {"conversation": {"id": "conv-nested"}}
        mock_request.return_value = mock_response

        assert client.resolve_conversation_id("msg-456") == "conv-nested"

    @patch.object(PlatformApiClient, "_request")
    def test_resolve_conversation_id_not_found(self, mock_request, client):
        mock_response = MagicMock()
        mock_response.status_code = 404
        mock_request.return_value = mock_response

        with pytest.raises(PlatformApiError, match="Message not found"):
            client.resolve_conversation_id("missing-msg")


class TestPlatformApiClientDisconnect:
    @patch.object(PlatformApiClient, "_request")
    def test_disconnect_conversation_success(self, mock_request, client):
        mock_response = MagicMock()
        mock_response.ok = True
        mock_response.status_code = 200
        mock_request.return_value = mock_response

        client.disconnect_conversation("conv-abc")

        mock_request.assert_called_once_with(
            "POST",
            "/api/v2/conversations/conv-abc/disconnect",
        )

    @patch.object(PlatformApiClient, "_request")
    def test_disconnect_conversation_404_is_idempotent(self, mock_request, client):
        mock_response = MagicMock()
        mock_response.ok = False
        mock_response.status_code = 404
        mock_request.return_value = mock_response

        client.disconnect_conversation("conv-gone")

    @patch.object(PlatformApiClient, "_request")
    def test_disconnect_conversation_failure(self, mock_request, client):
        mock_response = MagicMock()
        mock_response.ok = False
        mock_response.status_code = 403
        mock_response.text = "Forbidden"
        mock_request.return_value = mock_response

        with pytest.raises(PlatformApiError, match="Failed to disconnect"):
            client.disconnect_conversation("conv-denied")
