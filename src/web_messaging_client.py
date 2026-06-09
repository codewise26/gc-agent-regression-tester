"""Web Messaging Client for Genesys Cloud Web Messaging Guest API."""

import asyncio
import json
import uuid
from dataclasses import dataclass
from typing import Optional

import websockets


class WebMessagingError(Exception):
    """Custom exception for Web Messaging connection and protocol errors."""

    pass


@dataclass(frozen=True)
class AgentMessage:
    """Text and optional message ID from an agent WebSocket message."""

    text: str
    message_id: Optional[str] = None


class WebMessagingClient:
    """Client for communicating with Genesys Cloud agents via the Web Messaging Guest API.

    Manages WebSocket connections, session lifecycle, and message exchange
    with the Genesys Cloud Web Messaging protocol.
    """

    def __init__(self, region: str, deployment_id: str, timeout: int = 30, origin: str = "https://localhost"):
        """Initialize with Genesys Cloud connection details.

        Args:
            region: The Genesys Cloud region (e.g., 'mypurecloud.com').
            deployment_id: The Web Messaging deployment ID.
            timeout: Timeout in seconds for waiting on messages (default 30).
            origin: The origin header value (must match an allowed origin on the deployment).
        """
        self.region = region
        self.deployment_id = deployment_id
        self.timeout = timeout
        self.origin = origin
        self._ws: Optional[websockets.WebSocketClientProtocol] = None
        self._token: Optional[str] = None

    @property
    def ws_url(self) -> str:
        """Construct the WebSocket URL for the Web Messaging Guest API."""
        return f"wss://webmessaging.{self.region}/v1?deploymentId={self.deployment_id}"

    async def connect(self) -> None:
        """Establish a new WebSocket session with the Genesys Cloud Web Messaging Guest API.

        Sends a configureSession message to initialize the session.

        Raises:
            WebMessagingError: If the connection cannot be established.
                The error message includes both the deployment ID and region.
        """
        try:
            self._ws = await websockets.connect(
                self.ws_url,
                additional_headers={"Origin": self.origin},
            )
        except Exception as e:
            raise WebMessagingError(
                f"Failed to connect to Web Messaging API: deployment_id={self.deployment_id}, "
                f"region={self.region}. Error: {e}"
            ) from e

        # Send configureSession to initialize the session
        self._token = str(uuid.uuid4())
        configure_message = {
            "action": "configureSession",
            "deploymentId": self.deployment_id,
            "token": self._token,
        }
        try:
            await self._ws.send(json.dumps(configure_message))
        except Exception as e:
            raise WebMessagingError(
                f"Failed to configure session: deployment_id={self.deployment_id}, "
                f"region={self.region}. Error: {e}"
            ) from e

        # Wait for session confirmation
        try:
            response = await asyncio.wait_for(self._ws.recv(), timeout=self.timeout)
            data = json.loads(response)
            if data.get("type") == "SessionResponse":
                # Session established successfully
                pass
            # Accept other initial messages as well (protocol may vary)
        except asyncio.TimeoutError:
            raise WebMessagingError(
                f"Timed out waiting for session confirmation: deployment_id={self.deployment_id}, "
                f"region={self.region}"
            )
        except Exception as e:
            raise WebMessagingError(
                f"Error during session setup: deployment_id={self.deployment_id}, "
                f"region={self.region}. Error: {e}"
            ) from e

    async def wait_for_welcome(self) -> AgentMessage:
        """Wait for the agent's welcome message.

        Returns:
            The agent's welcome message with optional message ID.

        Raises:
            TimeoutError: If no welcome message is received within the configured timeout.
            WebMessagingError: If the connection is not established or a protocol error occurs.
        """
        if self._ws is None:
            raise WebMessagingError(
                f"Not connected: deployment_id={self.deployment_id}, region={self.region}"
            )

        try:
            return await self._receive_agent_message()
        except asyncio.TimeoutError:
            raise TimeoutError(
                f"Timed out waiting for welcome message after {self.timeout}s"
            )

    async def send_join(self) -> None:
        """Send a Join presence event to start the conversation.

        This triggers the bot flow to begin and send a welcome message.

        Raises:
            WebMessagingError: If the connection is not established or sending fails.
        """
        if self._ws is None:
            raise WebMessagingError(
                f"Not connected: deployment_id={self.deployment_id}, region={self.region}"
            )

        join_message = {
            "action": "onMessage",
            "token": self._token,
            "message": {
                "type": "Event",
                "events": [
                    {
                        "eventType": "Presence",
                        "presence": {
                            "type": "Join"
                        }
                    }
                ]
            }
        }
        try:
            await self._ws.send(json.dumps(join_message))
        except Exception as e:
            raise WebMessagingError(
                f"Failed to send join event: deployment_id={self.deployment_id}, "
                f"region={self.region}. Error: {e}"
            ) from e

    async def send_message(self, text: str) -> None:
        """Send a user message through the active session.

        Args:
            text: The message text to send.

        Raises:
            WebMessagingError: If the connection is not established or sending fails.
        """
        if self._ws is None:
            raise WebMessagingError(
                f"Not connected: deployment_id={self.deployment_id}, region={self.region}"
            )

        message = {
            "action": "onMessage",
            "token": self._token,
            "message": {
                "type": "Text",
                "text": text,
            },
        }
        try:
            await self._ws.send(json.dumps(message))
        except Exception as e:
            raise WebMessagingError(
                f"Failed to send message: deployment_id={self.deployment_id}, "
                f"region={self.region}. Error: {e}"
            ) from e

    async def receive_response(self) -> AgentMessage:
        """Wait for and return the next agent response.

        Returns:
            The agent's response message with optional message ID.

        Raises:
            TimeoutError: If no response is received within the configured timeout.
            WebMessagingError: If the connection is not established or a protocol error occurs.
        """
        if self._ws is None:
            raise WebMessagingError(
                f"Not connected: deployment_id={self.deployment_id}, region={self.region}"
            )

        try:
            return await self._receive_agent_message()
        except asyncio.TimeoutError:
            raise TimeoutError(
                f"Timed out waiting for agent response after {self.timeout}s"
            )

    async def disconnect(self) -> None:
        """Close the WebSocket session.

        Safe to call even if not connected.
        """
        if self._ws is not None:
            try:
                await self._ws.close()
            except Exception:
                pass  # Best-effort close
            finally:
                self._ws = None
                self._token = None

    @staticmethod
    def _extract_message_id(body: dict) -> Optional[str]:
        """Extract message ID from a structured message body."""
        message_id = body.get("id")
        if message_id:
            return str(message_id)

        channel = body.get("channel")
        if isinstance(channel, dict):
            channel_id = channel.get("messageId")
            if channel_id:
                return str(channel_id)

        return None

    @staticmethod
    def _build_agent_message(body: dict) -> Optional[AgentMessage]:
        """Build an AgentMessage from a message body dict."""
        text = body.get("text", "")
        if not text:
            return None
        return AgentMessage(text=text, message_id=WebMessagingClient._extract_message_id(body))

    async def _receive_agent_message(self) -> AgentMessage:
        """Wait for and extract text from the next agent message.

        Parses incoming WebSocket messages according to the Genesys Cloud
        Web Messaging protocol, filtering for structured message types
        that contain agent text content.

        Returns:
            The extracted agent message with optional message ID.

        Raises:
            asyncio.TimeoutError: If no agent message arrives within timeout.
            WebMessagingError: If a protocol error occurs.
        """
        deadline = asyncio.get_event_loop().time() + self.timeout

        while True:
            remaining = deadline - asyncio.get_event_loop().time()
            if remaining <= 0:
                raise asyncio.TimeoutError()

            try:
                raw = await asyncio.wait_for(self._ws.recv(), timeout=remaining)
            except asyncio.TimeoutError:
                raise

            try:
                data = json.loads(raw)
            except json.JSONDecodeError:
                continue  # Skip non-JSON messages

            # Handle structured message types from the Web Messaging protocol
            msg_type = data.get("type", "")
            msg_class = data.get("class", "")

            # Skip echo of our own inbound messages and events
            body = data.get("body", {})
            if isinstance(body, dict):
                direction = body.get("direction", "")
                if direction == "Inbound":
                    continue
                # Skip typing indicators and presence events
                body_type = body.get("type", "")
                if body_type == "Event":
                    continue

            # Look for agent messages in the "message" type responses
            if msg_type == "message" and msg_class == "StructuredMessage":
                if isinstance(body, dict):
                    agent_message = self._build_agent_message(body)
                    if agent_message:
                        return agent_message

            # Also handle simpler response format
            if msg_type == "message":
                # Try to extract text from body directly
                body = data.get("body", "")
                if isinstance(body, str) and body:
                    return AgentMessage(text=body)
                # Try nested text field
                if isinstance(body, dict):
                    agent_message = self._build_agent_message(body)
                    if agent_message:
                        return agent_message

            # Handle "response" type messages (some protocol variants)
            if msg_type == "response":
                body = data.get("body", {})
                if isinstance(body, dict):
                    agent_message = self._build_agent_message(body)
                    if agent_message:
                        return agent_message
                elif isinstance(body, str) and body:
                    return AgentMessage(text=body)
