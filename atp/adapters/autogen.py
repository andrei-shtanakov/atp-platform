"""AutoGen adapter for agents implemented with Microsoft AutoGen."""

import importlib
import time
from collections.abc import AsyncIterator
from datetime import datetime
from typing import Any

from pydantic import Field

from atp.protocol import (
    ATPEvent,
    ATPRequest,
    ATPResponse,
    EventType,
    Metrics,
    ResponseStatus,
)

from .base import AdapterConfig, AgentAdapter
from .exceptions import (
    AdapterConnectionError,
    AdapterError,
    AdapterTimeoutError,
)


class AutoGenAdapterConfig(AdapterConfig):
    """Configuration for AutoGen adapter."""

    module: str = Field(..., description="Python module path containing the agent(s)")
    agent: str = Field(
        ..., description="Name of the agent variable or factory function"
    )
    is_factory: bool = Field(False, description="Whether 'agent' is a factory function")
    factory_args: dict[str, Any] = Field(
        default_factory=dict, description="Arguments to pass to factory function"
    )
    user_proxy: str | None = Field(
        None, description="Name of user proxy agent for conversation initiation"
    )
    max_consecutive_auto_reply: int = Field(
        10, description="Maximum consecutive auto-replies"
    )
    human_input_mode: str = Field(
        "NEVER", description="Human input mode (NEVER, TERMINATE, ALWAYS)"
    )


class AutoGenAdapter(AgentAdapter):
    """
    Adapter for agents implemented with Microsoft AutoGen.

    Supports both AutoGen v0.2 (legacy) and v0.4+ patterns.
    Loads agents from Python modules and executes conversations
    with ATP Protocol translation.
    """

    def __init__(self, config: AutoGenAdapterConfig) -> None:
        """
        Initialize AutoGen adapter.

        Args:
            config: AutoGen adapter configuration.
        """
        super().__init__(config)
        self._config: AutoGenAdapterConfig = config
        self._agent: Any = None
        self._user_proxy: Any = None
        self._module: Any = None

    @property
    def adapter_type(self) -> str:
        """Return the adapter type identifier."""
        return "autogen"

    def _load_agent(self) -> tuple[Any, Any | None]:
        """
        Load the AutoGen agent from the configured module.

        Returns:
            Tuple of (main agent, user proxy agent or None).

        Raises:
            AdapterConnectionError: If module or agent cannot be loaded.
        """
        if self._agent is not None:
            return self._agent, self._user_proxy

        try:
            self._module = importlib.import_module(self._config.module)
        except ImportError as e:
            raise AdapterConnectionError(
                f"Failed to import module '{self._config.module}': {e}",
                endpoint=self._config.module,
                adapter_type=self.adapter_type,
                cause=e,
            ) from e

        try:
            agent_or_factory = getattr(self._module, self._config.agent)
        except AttributeError as e:
            raise AdapterConnectionError(
                f"Agent '{self._config.agent}' not found in module "
                f"'{self._config.module}'",
                endpoint=f"{self._config.module}.{self._config.agent}",
                adapter_type=self.adapter_type,
                cause=e,
            ) from e

        if self._config.is_factory:
            try:
                self._agent = agent_or_factory(**self._config.factory_args)
            except Exception as e:
                raise AdapterConnectionError(
                    f"Failed to create agent from factory '{self._config.agent}': {e}",
                    endpoint=f"{self._config.module}.{self._config.agent}",
                    adapter_type=self.adapter_type,
                    cause=e,
                ) from e
        else:
            self._agent = agent_or_factory

        # Load user proxy if specified
        if self._config.user_proxy:
            try:
                self._user_proxy = getattr(self._module, self._config.user_proxy)
            except AttributeError as e:
                raise AdapterConnectionError(
                    f"User proxy '{self._config.user_proxy}' not found in module "
                    f"'{self._config.module}'",
                    endpoint=f"{self._config.module}.{self._config.user_proxy}",
                    adapter_type=self.adapter_type,
                    cause=e,
                ) from e

        return self._agent, self._user_proxy

    def _create_user_proxy(self) -> Any:
        """
        Create a default user proxy agent for initiating conversations.

        Returns:
            UserProxyAgent instance.

        Raises:
            AdapterError: If AutoGen is not available.
        """
        try:
            # Try AutoGen v0.2 import
            from autogen import UserProxyAgent  # pyrefly: ignore[missing-import]
        except ImportError:
            try:
                # Try AutoGen v0.4+ import
                from autogen_agentchat.agents import (
                    UserProxyAgent,  # pyrefly: ignore[missing-import]
                )
            except ImportError:
                raise AdapterError(
                    "AutoGen is not installed. "
                    "Install with: pip install autogen-agentchat",
                    adapter_type=self.adapter_type,
                )

        return UserProxyAgent(
            name="atp_user_proxy",
            human_input_mode=self._config.human_input_mode,
            max_consecutive_auto_reply=self._config.max_consecutive_auto_reply,
            is_termination_msg=lambda msg: (
                msg.get("content", "").rstrip().endswith("TERMINATE")
                if msg.get("content")
                else False
            ),
        )

    def _build_message(self, request: ATPRequest) -> str:
        """
        Build the initial message from an ATP request.

        Args:
            request: ATP Request with task specification.

        Returns:
            Message string to send to the agent.
        """
        message = request.task.description

        # Append any additional context from input_data
        if request.task.input_data:
            for key, value in request.task.input_data.items():
                if key not in ("task", "description", "message"):
                    message += f"\n\n{key}: {value}"

        return message

    def _create_event(
        self,
        request: ATPRequest,
        event_type: EventType,
        payload: dict[str, Any],
        sequence: int,
    ) -> ATPEvent:
        """Create an ATP event."""
        return ATPEvent(
            task_id=request.task_id,
            timestamp=datetime.now(),
            sequence=sequence,
            event_type=event_type,
            payload=payload,
        )

    def _extract_chat_history(self, agent: Any) -> list[dict[str, Any]]:
        """
        Extract chat history from an AutoGen agent.

        Args:
            agent: AutoGen agent with chat history.

        Returns:
            List of message dictionaries.
        """
        messages = []

        # Try to get chat history from different AutoGen versions
        if hasattr(agent, "chat_messages"):
            for recipient, msgs in agent.chat_messages.items():
                for msg in msgs:
                    messages.append(
                        {
                            "role": msg.get("role", "unknown"),
                            "content": msg.get("content", ""),
                            "name": msg.get("name", str(recipient)),
                        }
                    )
        elif hasattr(agent, "_oai_messages"):
            for recipient, msgs in agent._oai_messages.items():
                for msg in msgs:
                    messages.append(
                        {
                            "role": msg.get("role", "unknown"),
                            "content": msg.get("content", ""),
                        }
                    )

        return messages

    async def execute(self, request: ATPRequest) -> ATPResponse:
        """
        Execute a task via AutoGen.

        Args:
            request: ATP Request with task specification.

        Returns:
            ATPResponse from the agent conversation.

        Raises:
            AdapterConnectionError: If agent cannot be loaded.
            AdapterTimeoutError: If execution times out.
            AdapterError: If execution fails.
        """
        import asyncio

        agent, user_proxy = self._load_agent()

        # Create default user proxy if not provided
        if user_proxy is None:
            user_proxy = self._create_user_proxy()

        message = self._build_message(request)

        start_time = time.time()
        timeout = request.constraints.get(
            "timeout_seconds", self._config.timeout_seconds
        )

        try:
            # Check for async initiate_chat
            if hasattr(user_proxy, "a_initiate_chat"):
                try:
                    chat_result = await asyncio.wait_for(
                        user_proxy.a_initiate_chat(
                            agent,
                            message=message,
                            clear_history=True,
                        ),
                        timeout=timeout,
                    )
                except TimeoutError as e:
                    raise AdapterTimeoutError(
                        f"AutoGen execution timed out after {timeout}s",
                        timeout_seconds=timeout,
                        adapter_type=self.adapter_type,
                    ) from e
            elif hasattr(user_proxy, "initiate_chat"):
                # Sync initiate_chat - run in executor
                loop = asyncio.get_running_loop()
                try:
                    chat_result = await asyncio.wait_for(
                        loop.run_in_executor(
                            None,
                            lambda: user_proxy.initiate_chat(
                                agent,
                                message=message,
                                clear_history=True,
                            ),
                        ),
                        timeout=timeout,
                    )
                except TimeoutError as e:
                    raise AdapterTimeoutError(
                        f"AutoGen execution timed out after {timeout}s",
                        timeout_seconds=timeout,
                        adapter_type=self.adapter_type,
                    ) from e
            else:
                raise AdapterError(
                    "Agent does not support initiate_chat",
                    adapter_type=self.adapter_type,
                )

            wall_time = time.time() - start_time

            # Extract results
            output, metrics = self._extract_result(
                chat_result, agent, user_proxy, wall_time
            )

            return ATPResponse(
                task_id=request.task_id,
                status=ResponseStatus.COMPLETED,
                artifacts=[
                    {
                        "type": "structured",
                        "name": "output",
                        "data": {
                            "content": output,
                            "chat_history": self._extract_chat_history(user_proxy),
                        },
                    }
                ],
                metrics=metrics,
            )

        except AdapterTimeoutError:
            raise
        except AdapterError:
            raise
        except Exception as e:
            wall_time = time.time() - start_time
            return ATPResponse(
                task_id=request.task_id,
                status=ResponseStatus.FAILED,
                error=str(e),
                metrics=Metrics(wall_time_seconds=wall_time),
            )

    def _extract_result(
        self,
        chat_result: Any,
        agent: Any,
        user_proxy: Any,
        wall_time: float,
    ) -> tuple[str, Metrics]:
        """
        Extract result and metrics from AutoGen chat.

        Args:
            chat_result: Result from initiate_chat.
            agent: The main agent.
            user_proxy: The user proxy agent.
            wall_time: Wall clock time in seconds.

        Returns:
            Tuple of (output string, Metrics).
        """
        output = ""
        total_tokens = None
        llm_calls = 0
        total_steps = 0

        # Extract output from chat result
        if chat_result is not None:
            if hasattr(chat_result, "summary"):
                output = str(chat_result.summary)
            elif hasattr(chat_result, "chat_history"):
                # Get last non-empty message
                for msg in reversed(chat_result.chat_history):
                    content = msg.get("content", "")
                    if content and content.strip() and "TERMINATE" not in content:
                        output = content
                        break
                total_steps = len(chat_result.chat_history)

            # Extract cost/token info if available
            if hasattr(chat_result, "cost"):
                cost_data = chat_result.cost
                if isinstance(cost_data, dict):
                    for model_cost in cost_data.values():
                        if isinstance(model_cost, dict):
                            tokens = model_cost.get("total_tokens", 0)
                            if tokens:
                                total_tokens = (total_tokens or 0) + tokens

        # Fallback: get last message from agent
        if not output:
            if hasattr(agent, "last_message"):
                last_msg = agent.last_message()
                if last_msg:
                    output = str(last_msg.get("content", ""))

        # Count messages as LLM calls
        messages = self._extract_chat_history(user_proxy)
        total_steps = len(messages)
        llm_calls = sum(1 for m in messages if m.get("role") == "assistant")

        return output, Metrics(
            total_tokens=total_tokens,
            total_steps=total_steps,
            llm_calls=llm_calls,
            wall_time_seconds=wall_time,
        )

    async def stream_events(
        self, request: ATPRequest
    ) -> AsyncIterator[ATPEvent | ATPResponse]:
        """
        Execute a task with event streaming via AutoGen.

        AutoGen supports streaming through callbacks. This implementation
        uses a message queue to yield events.

        Args:
            request: ATP Request with task specification.

        Yields:
            ATPEvent objects during execution.
            Final ATPResponse when complete.

        Raises:
            AdapterConnectionError: If agent cannot be loaded.
            AdapterTimeoutError: If execution times out.
            AdapterError: If execution fails.
        """
        import asyncio
        import queue

        agent, user_proxy = self._load_agent()

        # Create default user proxy if not provided
        if user_proxy is None:
            user_proxy = self._create_user_proxy()

        message = self._build_message(request)

        start_time = time.time()
        sequence = 0
        timeout = request.constraints.get(
            "timeout_seconds", self._config.timeout_seconds
        )

        # Event queue for streaming
        event_queue: queue.Queue[ATPEvent | None] = queue.Queue()

        # Emit start event
        yield self._create_event(
            request,
            EventType.PROGRESS,
            {"message": "Starting AutoGen conversation", "current_step": 0},
            sequence,
        )
        sequence += 1

        # Create callback handler for streaming
        def message_callback(
            sender: Any,
            message: dict[str, Any] | str,
            recipient: Any,
            silent: bool,
        ) -> None:
            nonlocal sequence
            if isinstance(message, str):
                content = message
                role = "unknown"
            else:
                content = message.get("content", "")
                role = message.get("role", "unknown")

            sender_name = getattr(sender, "name", str(sender))
            recipient_name = getattr(recipient, "name", str(recipient))

            if role == "assistant" or sender_name != "atp_user_proxy":
                event = self._create_event(
                    request,
                    EventType.LLM_REQUEST,
                    {
                        "model": "autogen",
                        "sender": sender_name,
                        "recipient": recipient_name,
                        "content": content[:500],  # Truncate for events
                    },
                    sequence,
                )
                event_queue.put(event)
                sequence += 1

        # Register callback if supported
        if hasattr(agent, "register_reply"):
            # AutoGen v0.2 style callback
            try:
                agent.register_reply(
                    [type(user_proxy)],
                    lambda *args, **kwargs: message_callback(*args, **kwargs) or False,
                )
            except Exception:
                pass

        try:
            # Execute in background
            loop = asyncio.get_running_loop()

            async def run_chat():
                if hasattr(user_proxy, "a_initiate_chat"):
                    return await user_proxy.a_initiate_chat(
                        agent,
                        message=message,
                        clear_history=True,
                    )
                elif hasattr(user_proxy, "initiate_chat"):
                    return await loop.run_in_executor(
                        None,
                        lambda: user_proxy.initiate_chat(
                            agent,
                            message=message,
                            clear_history=True,
                        ),
                    )
                else:
                    raise AdapterError(
                        "Agent does not support initiate_chat",
                        adapter_type=self.adapter_type,
                    )

            # Create task for chat execution
            chat_task = asyncio.create_task(run_chat())

            # Yield events while chat is running
            deadline = time.time() + timeout
            while not chat_task.done():
                if time.time() > deadline:
                    chat_task.cancel()
                    raise AdapterTimeoutError(
                        f"AutoGen execution timed out after {timeout}s",
                        timeout_seconds=timeout,
                        adapter_type=self.adapter_type,
                    )

                # Check for events in queue
                try:
                    event = event_queue.get_nowait()
                    if event:
                        yield event
                except queue.Empty:
                    pass

                await asyncio.sleep(0.1)

            # Get result
            chat_result = await chat_task

            # Drain remaining events
            while not event_queue.empty():
                try:
                    event = event_queue.get_nowait()
                    if event:
                        yield event
                except queue.Empty:
                    break

            wall_time = time.time() - start_time

            # Extract results
            output, metrics = self._extract_result(
                chat_result, agent, user_proxy, wall_time
            )

            yield ATPResponse(
                task_id=request.task_id,
                status=ResponseStatus.COMPLETED,
                artifacts=[
                    {
                        "type": "structured",
                        "name": "output",
                        "data": {
                            "content": output,
                            "chat_history": self._extract_chat_history(user_proxy),
                        },
                    }
                ],
                metrics=metrics,
            )

        except AdapterTimeoutError:
            raise
        except AdapterError:
            raise
        except asyncio.CancelledError:
            raise AdapterTimeoutError(
                f"AutoGen execution cancelled after {timeout}s",
                timeout_seconds=timeout,
                adapter_type=self.adapter_type,
            )
        except Exception as e:
            wall_time = time.time() - start_time
            yield ATPResponse(
                task_id=request.task_id,
                status=ResponseStatus.FAILED,
                error=str(e),
                metrics=Metrics(wall_time_seconds=wall_time),
            )

    async def health_check(self) -> bool:
        """
        Check if the AutoGen agent can be loaded.

        Returns:
            True if agent loads successfully, False otherwise.
        """
        try:
            self._load_agent()
            return True
        except AdapterConnectionError:
            return False

    async def cleanup(self) -> None:
        """Release any resources."""
        # Clear chat history if possible
        if self._user_proxy is not None:
            if hasattr(self._user_proxy, "reset"):
                self._user_proxy.reset()
            elif hasattr(self._user_proxy, "clear_history"):
                self._user_proxy.clear_history()

        if self._agent is not None:
            if hasattr(self._agent, "reset"):
                self._agent.reset()
            elif hasattr(self._agent, "clear_history"):
                self._agent.clear_history()

        self._agent = None
        self._user_proxy = None
        self._module = None
