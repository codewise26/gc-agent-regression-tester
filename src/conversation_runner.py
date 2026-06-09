"""Conversation Runner for executing single test attempts."""

import asyncio
from typing import Callable, Optional

from .conversation_registry import ConversationRegistry
from .judge_llm import JudgeLLMClient, JudgeLLMError
from .models import (
    AttemptResult,
    GoalEvaluation,
    Message,
    MessageRole,
    ProgressEvent,
    ProgressEventType,
    TestScenario,
)
from .platform_api_client import PlatformApiClient, PlatformApiError
from .progress import ProgressEmitter
from .web_messaging_client import WebMessagingClient, WebMessagingError


class ConversationRunner:
    """Manages a single conversation attempt between the Judge LLM and a Genesys Cloud agent.

    Creates a new WebMessagingClient per attempt for test isolation,
    drives the conversation loop via the Judge LLM, and evaluates the goal.
    """

    def __init__(
        self,
        judge: JudgeLLMClient,
        web_msg_config: dict,
        max_turns: int = 20,
        progress_emitter: Optional[ProgressEmitter] = None,
        platform_client: Optional[PlatformApiClient] = None,
        conversation_registry: Optional[ConversationRegistry] = None,
    ):
        """Initialize with judge client and web messaging configuration.

        Args:
            judge: The JudgeLLMClient instance for generating messages and evaluating goals.
            web_msg_config: Dict with keys: region, deployment_id, timeout.
            max_turns: Maximum number of user-agent message pairs before stopping.
            progress_emitter: Optional emitter for granular debug progress events.
            platform_client: Optional Platform API client for conversation resolution/disconnect.
            conversation_registry: Optional registry of active conversations.
        """
        self.judge = judge
        self.web_msg_config = web_msg_config
        self.max_turns = max_turns
        self.progress_emitter = progress_emitter
        self.platform_client = platform_client
        self.conversation_registry = conversation_registry

    def _emit_debug(
        self,
        scenario_name: str,
        attempt_number: int,
        message: str,
        detail: Optional[str] = None,
    ) -> None:
        if self.progress_emitter is None:
            return
        self.progress_emitter.emit(
            ProgressEvent(
                event_type=ProgressEventType.DEBUG_STEP,
                scenario_name=scenario_name,
                attempt_number=attempt_number,
                message=message,
                detail=detail,
            )
        )

    async def _run_judge(self, fn: Callable, *args, **kwargs):
        """Run blocking judge HTTP calls off the asyncio event loop."""
        return await asyncio.to_thread(fn, *args, **kwargs)

    async def _resolve_and_register_conversation(
        self,
        scenario_name: str,
        attempt_number: int,
        message_id: Optional[str],
    ) -> Optional[str]:
        """Resolve conversationId from messageId and register it."""
        if not message_id:
            self._emit_debug(
                scenario_name,
                attempt_number,
                "Welcome message missing messageId",
                detail="Skipping Platform API conversation resolution",
            )
            return None

        if self.platform_client is None:
            return None

        try:
            conversation_id = await asyncio.to_thread(
                self.platform_client.resolve_conversation_id,
                message_id,
            )
        except PlatformApiError as e:
            self._emit_debug(
                scenario_name,
                attempt_number,
                "Failed to resolve conversationId",
                detail=str(e),
            )
            return None

        if self.conversation_registry is not None:
            self.conversation_registry.add(
                conversation_id,
                scenario_name,
                attempt_number,
            )

        self._emit_debug(
            scenario_name,
            attempt_number,
            "Conversation ID resolved",
            detail=conversation_id,
        )
        return conversation_id

    async def _disconnect_conversation(
        self,
        scenario_name: str,
        attempt_number: int,
        conversation_id: Optional[str],
    ) -> None:
        """Disconnect a conversation via Platform API and remove from registry."""
        if not conversation_id or self.platform_client is None:
            return

        try:
            await asyncio.to_thread(
                self.platform_client.disconnect_conversation,
                conversation_id,
            )
            if self.conversation_registry is not None:
                self.conversation_registry.remove(conversation_id)
            self._emit_debug(
                scenario_name,
                attempt_number,
                "Conversation disconnected via Platform API",
                detail=conversation_id,
            )
        except PlatformApiError as e:
            self._emit_debug(
                scenario_name,
                attempt_number,
                "Failed to disconnect conversation via Platform API",
                detail=f"{conversation_id}: {e}",
            )

    async def run_attempt(
        self, scenario: TestScenario, attempt_number: int
    ) -> AttemptResult:
        """Execute a single conversation attempt for a scenario.

        Creates a new WebMessagingClient (test isolation), connects, waits for
        the agent welcome message, drives the conversation via the Judge LLM,
        and returns the result.

        Args:
            scenario: The test scenario to execute.
            attempt_number: The attempt number (1-based).

        Returns:
            AttemptResult with conversation history, success/failure, and explanation.
        """
        scenario_name = scenario.name
        conversation: list[Message] = []
        conversation_id: Optional[str] = None
        result: Optional[AttemptResult] = None
        region = self.web_msg_config["region"]
        deployment_id = self.web_msg_config["deployment_id"]
        timeout = self.web_msg_config.get("timeout", 30)

        client = WebMessagingClient(
            region=region,
            deployment_id=deployment_id,
            timeout=timeout,
            origin=self.web_msg_config.get("origin", "https://localhost"),
        )

        try:
            self._emit_debug(
                scenario_name,
                attempt_number,
                "Connecting to Genesys Web Messaging",
                detail=f"wss://webmessaging.{region}/v1 (deployment {deployment_id}, timeout {timeout}s)",
            )
            await client.connect()
            self._emit_debug(scenario_name, attempt_number, "WebSocket connected, sending Join event")
            await client.send_join()

            self._emit_debug(
                scenario_name,
                attempt_number,
                "Waiting for agent welcome message",
                detail=f"Up to {timeout}s",
            )
            welcome = await client.wait_for_welcome()
            self._emit_debug(
                scenario_name,
                attempt_number,
                "Welcome message received",
                detail=welcome.text[:120]
                + ("…" if len(welcome.text) > 120 else ""),
            )

            conversation_id = await self._resolve_and_register_conversation(
                scenario_name,
                attempt_number,
                welcome.message_id,
            )
            conversation.append(Message(role=MessageRole.AGENT, content=welcome.text))

            turn_count = 0
            first_turn = True
            early_success = False
            evaluation: Optional[GoalEvaluation] = None

            while turn_count < self.max_turns:
                turn_label = turn_count + 1

                if first_turn and scenario.first_message:
                    user_message = scenario.first_message
                    first_turn = False
                    self._emit_debug(
                        scenario_name,
                        attempt_number,
                        f"Turn {turn_label}: using scripted first_message",
                        detail=user_message[:120]
                        + ("…" if len(user_message) > 120 else ""),
                    )
                else:
                    first_turn = False
                    self._emit_debug(
                        scenario_name,
                        attempt_number,
                        f"Turn {turn_label}: generating user message (Ollama)",
                        detail=f"model={self.judge.model}",
                    )
                    user_message = await self._run_judge(
                        self.judge.generate_user_message,
                        scenario.persona,
                        scenario.goal,
                        conversation,
                    )
                    self._emit_debug(
                        scenario_name,
                        attempt_number,
                        f"Turn {turn_label}: user message ready",
                        detail=user_message[:120]
                        + ("…" if len(user_message) > 120 else ""),
                    )

                conversation.append(Message(role=MessageRole.USER, content=user_message))

                self._emit_debug(
                    scenario_name,
                    attempt_number,
                    f"Turn {turn_label}: sent to agent, waiting for reply",
                    detail=f"Up to {timeout}s",
                )
                await client.send_message(user_message)
                agent_response = await client.receive_response()
                conversation.append(
                    Message(role=MessageRole.AGENT, content=agent_response.text)
                )
                self._emit_debug(
                    scenario_name,
                    attempt_number,
                    f"Turn {turn_label}: agent replied",
                    detail=agent_response.text[:120]
                    + ("…" if len(agent_response.text) > 120 else ""),
                )

                turn_count += 1

                self._emit_debug(
                    scenario_name,
                    attempt_number,
                    f"Turn {turn_label}: evaluating goal (Ollama)",
                )
                try:
                    evaluation = await self._run_judge(
                        self.judge.evaluate_goal,
                        scenario.persona,
                        scenario.goal,
                        conversation,
                    )
                    if evaluation.success:
                        self._emit_debug(
                            scenario_name,
                            attempt_number,
                            f"Turn {turn_label}: goal achieved",
                            detail=evaluation.explanation[:200],
                        )
                        early_success = True
                        break
                    self._emit_debug(
                        scenario_name,
                        attempt_number,
                        f"Turn {turn_label}: goal not yet met, continuing",
                    )
                except JudgeLLMError as e:
                    self._emit_debug(
                        scenario_name,
                        attempt_number,
                        f"Turn {turn_label}: goal evaluation failed, continuing",
                        detail=str(e),
                    )

            if early_success and evaluation is not None:
                result = AttemptResult(
                    attempt_number=attempt_number,
                    success=True,
                    conversation=conversation,
                    explanation=f"Goal achieved after {turn_count} turn(s). {evaluation.explanation}",
                    conversation_id=conversation_id,
                )
            else:
                self._emit_debug(
                    scenario_name,
                    attempt_number,
                    "Final goal evaluation (Ollama)",
                )
                evaluation = await self._run_judge(
                    self.judge.evaluate_goal,
                    scenario.persona,
                    scenario.goal,
                    conversation,
                )

                if evaluation.success:
                    result = AttemptResult(
                        attempt_number=attempt_number,
                        success=True,
                        conversation=conversation,
                        explanation=f"Goal achieved at max turns ({turn_count}). {evaluation.explanation}",
                        conversation_id=conversation_id,
                    )
                else:
                    result = AttemptResult(
                        attempt_number=attempt_number,
                        success=False,
                        conversation=conversation,
                        explanation=f"Goal NOT achieved after {turn_count} turn(s) (max: {self.max_turns}). {evaluation.explanation}",
                        conversation_id=conversation_id,
                    )

        except TimeoutError as e:
            timeout_stage = "welcome message" if not conversation else "agent response"
            self._emit_debug(
                scenario_name,
                attempt_number,
                f"Timed out waiting for {timeout_stage}",
                detail=str(e),
            )
            result = AttemptResult(
                attempt_number=attempt_number,
                success=False,
                conversation=conversation,
                explanation=f"Timed out waiting for {timeout_stage} after {len(conversation)} message(s) in the conversation.",
                error=str(e),
                conversation_id=conversation_id,
            )
        except WebMessagingError as e:
            self._emit_debug(
                scenario_name,
                attempt_number,
                "Genesys Web Messaging error",
                detail=str(e),
            )
            result = AttemptResult(
                attempt_number=attempt_number,
                success=False,
                conversation=conversation,
                explanation=f"Connection error with Genesys Cloud after {len(conversation)} message(s). Check deployment ID and region.",
                error=str(e),
                conversation_id=conversation_id,
            )
        except JudgeLLMError as e:
            self._emit_debug(
                scenario_name,
                attempt_number,
                "Ollama judge LLM error",
                detail=str(e),
            )
            result = AttemptResult(
                attempt_number=attempt_number,
                success=False,
                conversation=conversation,
                explanation=f"The judge LLM failed to produce a valid response after {len(conversation)} message(s). This is usually a model issue — try a larger model.",
                error=str(e),
                conversation_id=conversation_id,
            )
        finally:
            await client.disconnect()
            self._emit_debug(scenario_name, attempt_number, "WebSocket disconnected")
            await self._disconnect_conversation(
                scenario_name,
                attempt_number,
                conversation_id,
            )

        if result is None:
            result = AttemptResult(
                attempt_number=attempt_number,
                success=False,
                conversation=conversation,
                explanation="Attempt ended without a result.",
                conversation_id=conversation_id,
            )

        return result
