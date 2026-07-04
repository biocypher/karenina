"""Claude Agent SDK Agent adapter implementing the AgentPort interface.

This module provides the ClaudeSDKAgentAdapter class that implements the AgentPort
interface using the Claude Agent SDK's ClaudeSDKClient for agent loops with MCP
tool support.

IMPORTANT: Uses ClaudeSDKClient (NOT query()) because:
- ClaudeSDKClient provides MCP server integration
- Supports hooks and interrupts for complex workflows
- Manages multi-turn conversation state

Key differences from LangChain:
- Claude SDK uses string prompts, not message arrays
- System prompts go in ClaudeAgentOptions.system_prompt
- MCP servers are configured in ClaudeAgentOptions.mcp_servers
- Uses ClaudeSDKClient context manager pattern
"""

from __future__ import annotations

import asyncio
import logging
import os
from pathlib import Path
from typing import TYPE_CHECKING, Any

from karenina.adapters.agent_runtime import (
    CONTAINER_BACKEND,
    get_agent_runtime_access_mode,
    get_agent_runtime_capabilities,
    get_agent_runtime_option,
    get_claude_sdk_backend,
    get_container_runtime_config,
    preflight_container_runtime,
)
from karenina.ports import (
    AdapterUnavailableError,
    AgentConfig,
    AgentPort,
    AgentResponseError,
    AgentResult,
    AgentTimeoutError,
    MCPServerConfig,
    Message,
    Role,
    Tool,
)
from karenina.ports.capabilities import PortCapabilities

from .auth import subscription_auth_env
from .mcp import convert_mcp_config
from .messages import ClaudeSDKMessageConverter
from .trace import sdk_messages_to_raw_trace
from .usage import (
    backfill_assistant_output_tokens,
    collapse_partial_assistant_messages,
    extract_sdk_usage,
    extract_sdk_usage_from_messages,
)

if TYPE_CHECKING:
    from claude_agent_sdk import ClaudeAgentOptions, ResultMessage

    from karenina.schemas.config import ModelConfig

logger = logging.getLogger(__name__)

CLAUDE_SDK_READ_ONLY_TOOLS = ("Read", "Grep", "Glob", "LS")
CLAUDE_SDK_CONTAINER_WRAPPER = Path(__file__).with_name("docker_cli_wrapper.py")
CLAUDE_SDK_DOCKER_WRAPPER = CLAUDE_SDK_CONTAINER_WRAPPER
ZAI_ANTHROPIC_BASE_URL_MARKER = "api.z.ai/api/anthropic"


def _is_zai_anthropic_endpoint(base_url: str | None) -> bool:
    return bool(base_url and ZAI_ANTHROPIC_BASE_URL_MARKER in base_url)


class ClaudeSDKAgentAdapter:
    """Agent adapter using Claude Agent SDK's ClaudeSDKClient.

    This adapter implements the AgentPort Protocol for agent execution with
    MCP tools and multi-turn conversation support. Uses ClaudeSDKClient for
    full agent capabilities including tool invocation and hooks.

    The adapter handles:
    - Message conversion from unified Message to prompt string
    - MCP server configuration conversion to SDK format
    - Agent loop execution with turn limit handling
    - Dual trace output (raw_trace string and trace_messages list)
    - Usage metadata extraction from SDK responses

    Example:
        >>> from karenina.schemas.config import ModelConfig
        >>> config = ModelConfig(
        ...     id="claude-sonnet",
        ...     model_name="claude-sonnet-4-20250514",
        ...     model_provider="anthropic",
        ...     interface="claude_agent_sdk"
        ... )
        >>> adapter = ClaudeSDKAgentAdapter(config)
        >>> result = await adapter.arun(
        ...     messages=[Message.user("What files are in /tmp?")],
        ...     mcp_servers={
        ...         "filesystem": {
        ...             "type": "http",
        ...             "url": "https://mcp.example.com/filesystem",
        ...         }
        ...     }
        ... )
        >>> print(result.final_response)
    """

    def __init__(self, model_config: ModelConfig) -> None:
        """Initialize the Claude SDK Agent adapter.

        Args:
            model_config: Configuration specifying model, provider, and interface.
        """
        self._config = model_config
        self._converter = ClaudeSDKMessageConverter()

    @property
    def capabilities(self) -> PortCapabilities:
        """Declare adapter capabilities.

        Returns:
            PortCapabilities with system_prompt=True.
        """
        return get_agent_runtime_capabilities(self._config)

    def _build_sandbox_settings(self) -> dict[str, Any] | None:
        """Build Claude Code sandbox settings for Bash isolation."""

        if get_claude_sdk_backend(self._config) == CONTAINER_BACKEND:
            return None
        if not get_agent_runtime_option(
            self._config,
            "sandbox_enabled",
            True,
            legacy_attr="claude_sdk_sandbox_enabled",
        ):
            return None

        return {
            "enabled": True,
            "failIfUnavailable": bool(
                get_agent_runtime_option(
                    self._config,
                    "sandbox_fail_if_unavailable",
                    True,
                    legacy_attr="claude_sdk_sandbox_fail_if_unavailable",
                )
            ),
            "autoAllowBashIfSandboxed": True,
            "allowUnsandboxedCommands": bool(
                get_agent_runtime_option(
                    self._config,
                    "allow_unsandboxed_commands",
                    False,
                    legacy_attr="claude_sdk_allow_unsandboxed_commands",
                )
            ),
        }

    def _workspace_local_env(self, workspace_path: Path | None) -> dict[str, str]:
        """Return environment defaults that keep tool state inside a workspace."""

        if workspace_path is None:
            return {}
        return {
            "UV_CACHE_DIR": str(workspace_path / ".uv-cache"),
            "XDG_CACHE_HOME": str(workspace_path / ".cache"),
            "UV_PROJECT_ENVIRONMENT": str(workspace_path / ".venv"),
        }

    def _container_wrapper_env(self, workspace_path: Path | None) -> dict[str, str]:
        """Return environment required by the container CLI wrapper."""

        if workspace_path is None:
            raise AdapterUnavailableError(
                "agent_runtime backend='container' requires an AgentConfig.workspace_path",
                reason="missing_workspace",
            )
        container_config = get_container_runtime_config(self._config)
        if not container_config.image:
            raise AdapterUnavailableError(
                "agent_runtime container_image is required when claude_agent_sdk backend='container'",
                reason="missing_claude_sdk_container_image",
            )
        preflight_container_runtime(container_config)
        env = {
            "KARENINA_CLAUDE_CONTAINER_WORKSPACE": str(workspace_path.resolve()),
            "KARENINA_CLAUDE_CONTAINER_RUNTIME": container_config.runtime,
            "KARENINA_CLAUDE_CONTAINER_IMAGE": container_config.image,
            "KARENINA_CLAUDE_CONTAINER_NETWORK": container_config.network,
            "CLAUDE_CONFIG_DIR": "/tmp/claude-config",
        }
        if container_config.add_hosts:
            env["KARENINA_CLAUDE_CONTAINER_ADD_HOSTS"] = ",".join(container_config.add_hosts)
        if container_config.runtime == "docker":
            env.update(
                {
                    "KARENINA_CLAUDE_DOCKER_WORKSPACE": str(workspace_path.resolve()),
                    "KARENINA_CLAUDE_DOCKER_IMAGE": container_config.image,
                    "KARENINA_CLAUDE_DOCKER_NETWORK": container_config.network,
                }
            )
            if container_config.add_hosts:
                env["KARENINA_CLAUDE_DOCKER_ADD_HOSTS"] = ",".join(container_config.add_hosts)
        return env

    def _build_options(
        self,
        system_prompt: str | None,
        mcp_servers: dict[str, Any] | None,
        config: AgentConfig,
        tools: list[Tool] | None = None,
    ) -> ClaudeAgentOptions:
        """Build ClaudeAgentOptions for the agent run.

        Args:
            system_prompt: System prompt extracted from messages.
            mcp_servers: SDK-formatted MCP server configuration.
            config: Agent execution configuration.
            tools: Optional list of tool definitions.

        Returns:
            Configured ClaudeAgentOptions.
        """
        from claude_agent_sdk import ClaudeAgentOptions

        runtime_backend = get_claude_sdk_backend(self._config)
        access_mode = get_agent_runtime_access_mode(self._config)
        default_permission_mode = (
            "bypassPermissions"
            if runtime_backend == CONTAINER_BACKEND and access_mode == "read_write"
            else "acceptEdits"
        )

        options_kwargs: dict[str, Any] = {
            # In native mode, use sandbox-aware permissions by default. For
            # Container read-write mode, the container is the execution boundary,
            # so non-interactive runs need Bash to execute without prompts.
            # Keep read-only mode on acceptEdits because bypassPermissions does
            # not respect allowed_tools.
            "permission_mode": str(
                get_agent_runtime_option(
                    self._config,
                    "permission_mode",
                    default_permission_mode,
                    legacy_attr="claude_sdk_permission_mode",
                )
            ),
            # Map AgentConfig.max_turns to SDK
            "max_turns": config.max_turns,
            # Emit raw stream events alongside AssistantMessage objects so the
            # caller can recover per-turn output_tokens from message_delta
            # events. AssistantMessage.usage reflects only message_start state
            # (output_tokens=0 on backends that defer the count to the delta),
            # so without partials we cannot account for output tokens on a
            # wall-clock timeout where no ResultMessage is ever emitted.
            "include_partial_messages": True,
        }

        # Add system prompt if provided
        if system_prompt:
            options_kwargs["system_prompt"] = system_prompt
        elif self._config.system_prompt:
            options_kwargs["system_prompt"] = self._config.system_prompt

        # Add model specification if provided. Z.ai maps Claude Code's internal
        # Sonnet/Opus model names to GLM through ANTHROPIC_DEFAULT_* env vars;
        # passing glm-* directly to Claude Code is rejected before the endpoint
        # can apply that mapping.
        configured_model_name = self._config.model_name or ""
        model_name = configured_model_name
        if _is_zai_anthropic_endpoint(self._config.anthropic_base_url) and configured_model_name.startswith("glm-"):
            model_name = str((config.extra or {}).get("claude_sdk_model_name", "claude-sonnet-4-5"))
        if model_name:
            options_kwargs["model"] = model_name

        # Add MCP servers if provided
        if mcp_servers:
            options_kwargs["mcp_servers"] = mcp_servers

            # IMPORTANT: When MCP servers are configured, restrict tools to ONLY MCP tools.
            # This prevents the agent from using default Claude Code tools (Read, Grep, Bash, etc.)
            # which can distract from the actual MCP tool usage.
            #
            # Build allowed_tools list from MCP server names and tool filter.
            # MCP tools follow naming convention: mcp__<server_name>__<tool_name>
            mcp_tool_filter = self._config.mcp_tool_filter
            if mcp_tool_filter:
                # User specified specific tools - build full MCP tool names
                allowed_tools = []
                for server_name in mcp_servers:
                    for tool_name in mcp_tool_filter:
                        allowed_tools.append(f"mcp__{server_name}__{tool_name}")
                options_kwargs["allowed_tools"] = allowed_tools
                logger.info(f"Restricting SDK to MCP tools: {allowed_tools}")
            # If no tool filter specified but MCP servers are configured,
            # we cannot pre-filter (don't know tool names yet).
            # The agent will have access to all tools including defaults.
            # TODO: Consider fetching tool list from MCP servers to build filter.

        # Add tool filter if specific tools are requested (non-MCP case)
        if tools and "allowed_tools" not in options_kwargs:
            # SDK supports tool filtering via allowed_tools
            options_kwargs["allowed_tools"] = [t.name for t in tools]

        if get_agent_runtime_access_mode(self._config) == "read_only":
            read_only_tools = list(CLAUDE_SDK_READ_ONLY_TOOLS)
            options_kwargs["tools"] = read_only_tools
            options_kwargs["allowed_tools"] = read_only_tools

        sandbox_settings = self._build_sandbox_settings()
        if sandbox_settings:
            options_kwargs["sandbox"] = sandbox_settings

        # Apply any extra options from config
        if config.extra:
            extra_permission_mode = config.extra.get("claude_sdk_permission_mode")
            if extra_permission_mode:
                options_kwargs["permission_mode"] = extra_permission_mode
            elif config.extra.get("allow_unsafe_permission_mode_override") and "permission_mode" in config.extra:
                options_kwargs["permission_mode"] = config.extra["permission_mode"]

            for key, value in config.extra.items():
                # Don't override critical options
                if key not in (
                    "permission_mode",
                    "claude_sdk_permission_mode",
                    "allow_unsafe_permission_mode_override",
                    "max_turns",
                    "mcp_servers",
                    "allowed_tools",
                    "claude_sdk_model_name",
                ):
                    options_kwargs[key] = value

        # Deny Claude Code's bundled Skill tool by default. Newer CLIs
        # (>= 2.1.170, observed against 2.1.146 which does not) advertise their
        # bundled skills by appending a system-role message to the messages
        # array of every API request. That message contaminates benchmark
        # conversations with skill listings, and OpenAI-compatible Anthropic
        # endpoints such as vLLM's /v1/messages reject it with HTTP 400 because
        # only user/assistant roles are accepted inside messages. Disallowing
        # the Skill tool suppresses the system-role message entirely (verified
        # empirically against Claude Code 2.1.170). Callers can re-enable
        # skills via AgentConfig(extra={"disallowed_tools": []}); the extra
        # loop above already applied any caller-provided value, so setdefault
        # respects it.
        options_kwargs.setdefault("disallowed_tools", ["Skill"])

        # Build env dict for Anthropic settings (api_key, base_url).
        # The Claude Agent SDK reads ANTHROPIC_API_KEY and ANTHROPIC_BASE_URL from env.
        env_vars: dict[str, str] = dict(options_kwargs.get("env") or {})
        workspace_path = config.workspace_path
        if self._config.anthropic_api_key:
            env_vars["ANTHROPIC_API_KEY"] = self._config.anthropic_api_key.get_secret_value()
            if _is_zai_anthropic_endpoint(self._config.anthropic_base_url):
                env_vars["ANTHROPIC_AUTH_TOKEN"] = self._config.anthropic_api_key.get_secret_value()
        else:
            env_vars.update(subscription_auth_env())
        if self._config.anthropic_base_url:
            env_vars["ANTHROPIC_BASE_URL"] = self._config.anthropic_base_url
        if _is_zai_anthropic_endpoint(self._config.anthropic_base_url) and configured_model_name.startswith("glm-"):
            env_vars.setdefault("ANTHROPIC_DEFAULT_SONNET_MODEL", configured_model_name)
            env_vars.setdefault("ANTHROPIC_DEFAULT_OPUS_MODEL", configured_model_name)
            env_vars.setdefault("API_TIMEOUT_MS", "3000000")
        env_vars.update(self._workspace_local_env(workspace_path))
        # Forward CLAUDE_CONFIG_DIR from the parent process so the subprocess does
        # not load the user's personal MCP servers from ~/.claude/. Combined with
        # setting_sources=[] below, this mitigates issue 089: 30-40s startup
        # overhead per spawn and tool-namespace contamination.
        parent_claude_config_dir = os.environ.get("CLAUDE_CONFIG_DIR")
        if parent_claude_config_dir:
            env_vars["CLAUDE_CONFIG_DIR"] = parent_claude_config_dir

        if get_claude_sdk_backend(self._config) == CONTAINER_BACKEND:
            env_vars.update(self._container_wrapper_env(workspace_path))
            options_kwargs["cli_path"] = str(CLAUDE_SDK_CONTAINER_WRAPPER)

        if env_vars:
            options_kwargs["env"] = env_vars

        # Disable the Claude Code settings chain. Without this, the SDK still
        # loads personal settings even when CLAUDE_CONFIG_DIR points at an
        # empty directory. See issue 089.
        # Use setdefault so a caller passing AgentConfig(extra={"setting_sources": [...]})
        # can still override this default.
        options_kwargs.setdefault("setting_sources", [])

        # Wire cwd to the workspace boundary used by Claude Code permissions
        # and native Bash sandboxing. When no workspace is supplied, use the
        # current process directory instead of letting the SDK choose implicitly.
        options_kwargs["cwd"] = str(config.workspace_path or Path.cwd())

        return ClaudeAgentOptions(**options_kwargs)

    def _build_raw_trace(self, messages: list[Any]) -> str:
        """Build raw_trace string from SDK messages.

        Delegates to sdk_messages_to_raw_trace() from trace module.
        Matches the format used by LangChain adapter's harmonize_agent_response().

        Args:
            messages: List of SDK message objects.

        Returns:
            Formatted trace string compatible with existing infrastructure.
        """
        return sdk_messages_to_raw_trace(messages, include_user_messages=False)

    def _extract_final_response(self, messages: list[Any], result_msg: Any) -> str:
        """Extract final text response from SDK messages.

        Args:
            messages: List of SDK message objects.
            result_msg: ResultMessage from SDK.

        Returns:
            The final text response.
        """
        # First try ResultMessage.result (SDK's extraction)
        if hasattr(result_msg, "result") and result_msg.result:
            return str(result_msg.result)

        # Fall back to last AssistantMessage text content
        try:
            from claude_agent_sdk import AssistantMessage
            from claude_agent_sdk.types import TextBlock
        except ImportError:
            return "[Unable to extract response - SDK types not available]"

        for msg in reversed(messages):
            if isinstance(msg, AssistantMessage):
                text_parts = []
                for block in msg.content:
                    if isinstance(block, TextBlock):
                        text_parts.append(block.text)
                if text_parts:
                    return "\n".join(text_parts)

        return "[No final response extracted]"

    def _extract_actual_model(self, messages: list[Any]) -> str | None:
        """Extract actual model name from AssistantMessage.model field.

        The SDK includes the actual model that generated each response
        in the AssistantMessage.model field.

        Args:
            messages: List of SDK message objects.

        Returns:
            The model name from the last AssistantMessage, or None.
        """
        try:
            from claude_agent_sdk import AssistantMessage
        except ImportError:
            return None

        for msg in reversed(messages):
            if isinstance(msg, AssistantMessage) and hasattr(msg, "model") and msg.model:
                return msg.model

        return None

    def _build_agent_result(
        self,
        collected_messages: list[Any],
        result_message: Any | None,
        *,
        limit_reached: bool,
        timeout_reached: bool = False,
    ) -> AgentResult:
        """Build an AgentResult from completed or partial SDK messages."""

        # Collapse partial AssistantMessages to one per turn (the SDK emits
        # one per content-block-stop, all sharing a message_id) so iterations
        # and per-message usage reflect real LLM calls rather than partial
        # snapshots. Then backfill output_tokens from message_delta
        # StreamEvents into the surviving AssistantMessage.usage dicts BEFORE
        # downstream consumers run. Both the trace converter and the aggregate
        # usage summer read AssistantMessage.usage, so a single in-place patch
        # fixes both.
        collected_messages = collapse_partial_assistant_messages(collected_messages)
        backfill_assistant_output_tokens(collected_messages)

        raw_trace = self._build_raw_trace(collected_messages)
        if limit_reached:
            raw_trace += "\n\n[Note: Recursion limit reached - partial response shown]"
        if timeout_reached:
            raw_trace += "\n\n[Note: Agent timed out - partial trace shown]"

        # Exclude user and system messages: system prompts are captured
        # separately via tagged_messages injection.
        trace_messages = [
            m for m in self._converter.from_provider(collected_messages) if m.role not in (Role.USER, Role.SYSTEM)
        ]

        final_response = self._extract_final_response(collected_messages, result_message)
        # Prefer the SDK-aggregated usage on clean exit; fall back to summing
        # per-AssistantMessage usage when no ResultMessage was emitted (e.g.
        # mid-stream cancellation via asyncio.wait_for). Without the fallback,
        # wall-clock timeouts lose all token accounting even though the
        # collected messages still carry per-call usage.
        if result_message:
            usage = extract_sdk_usage(result_message, model=self._config.model_name)
        else:
            usage = extract_sdk_usage_from_messages(collected_messages, model=self._config.model_name)
        turns = result_message.num_turns if result_message and hasattr(result_message, "num_turns") else 0
        actual_model = self._extract_actual_model(collected_messages) or self._config.model_name
        session_id = result_message.session_id if result_message and hasattr(result_message, "session_id") else None

        return AgentResult(
            final_response=final_response,
            raw_trace=raw_trace,
            trace_messages=trace_messages,
            usage=usage,
            turns=turns or len([m for m in trace_messages if m.role.value == "assistant"]),
            limit_reached=limit_reached,
            session_id=session_id,
            actual_model=actual_model,
            timeout_reached=timeout_reached,
        )

    def _convert_mcp_servers(
        self,
        mcp_servers: dict[str, MCPServerConfig] | None,
    ) -> dict[str, Any] | None:
        """Convert MCPServerConfig to SDK format.

        Handles both already-SDK-formatted configs and karenina's simplified format.

        Args:
            mcp_servers: MCP server configuration.

        Returns:
            SDK-formatted MCP configuration, or None.
        """
        if not mcp_servers:
            return None

        first_config: Any = next(iter(mcp_servers.values()), {})
        if isinstance(first_config, dict):
            # Detect karenina's MCPServerConfig format by checking for
            # required TypedDict fields rather than generic key sniffing.
            is_karenina_stdio = "command" in first_config
            is_karenina_http = "url" in first_config and first_config.get("type") in (
                "http",
                "sse",
            )

            if is_karenina_stdio or is_karenina_http:
                # Karenina simplified format: convert to SDK format
                return mcp_servers

        # Assume simplified format {"name": "url"} - convert
        # This handles the case where mcp_servers is still in karenina's
        # simplified URL dict format
        url_dict: dict[str, str] = {}
        for name, config in mcp_servers.items():
            if isinstance(config, str):
                url_dict[name] = config
            elif isinstance(config, dict):
                url = config.get("url")
                if url and isinstance(url, str):
                    url_dict[name] = url

        if url_dict:
            return convert_mcp_config(url_dict)

        return mcp_servers

    async def arun(
        self,
        messages: list[Message],
        tools: list[Tool] | None = None,
        mcp_servers: dict[str, MCPServerConfig] | None = None,
        config: AgentConfig | None = None,
    ) -> AgentResult:
        """Execute an agent loop with optional tools and MCP servers.

        Args:
            messages: Initial conversation messages.
            tools: Optional list of Tool definitions the agent can invoke.
            mcp_servers: Optional dict of MCP server configurations.
            config: Optional AgentConfig for execution parameters.

        Returns:
            AgentResult with final response, traces, usage, and metadata.

        Raises:
            AgentExecutionError: If the agent fails during execution.
            AgentTimeoutError: If execution exceeds the timeout.
            AgentResponseError: If the response is malformed or invalid.
        """
        from claude_agent_sdk import ClaudeSDKClient, ResultMessage

        config = config or AgentConfig()

        # Convert messages to SDK format
        prompt_string = self._converter.to_prompt_string(messages)
        system_prompt = self._converter.extract_system_prompt(messages)

        # Convert MCP servers to SDK format
        sdk_mcp_servers = self._convert_mcp_servers(mcp_servers)

        # Build options
        options = self._build_options(system_prompt, sdk_mcp_servers, config, tools)

        # Execute agent loop using ClaudeSDKClient
        collected_messages: list[Any] = []
        result_message: ResultMessage | None = None
        limit_reached = False

        async def execute_agent() -> None:
            nonlocal collected_messages, result_message, limit_reached

            async with ClaudeSDKClient(options) as client:
                # Send initial query
                await client.query(prompt_string)

                # Collect all messages from response
                # IMPORTANT: Let iteration complete naturally - no break
                async for msg in client.receive_response():
                    collected_messages.append(msg)

                    if isinstance(msg, ResultMessage):
                        result_message = msg
                        # Check if limit was reached
                        if hasattr(msg, "subtype"):
                            subtype = msg.subtype or ""
                            if "limit" in subtype.lower() or "max" in subtype.lower():
                                limit_reached = True

        try:
            if config.timeout:
                await asyncio.wait_for(execute_agent(), timeout=config.timeout)
            else:
                await execute_agent()

        except TimeoutError as e:
            if collected_messages:
                logger.warning(
                    "Claude SDK agent timed out after %ss with %d partial messages",
                    config.timeout,
                    len(collected_messages),
                )
                return self._build_agent_result(
                    collected_messages,
                    result_message,
                    limit_reached=limit_reached,
                    timeout_reached=True,
                )
            raise AgentTimeoutError(f"Agent execution timed out after {config.timeout}s") from e
        except Exception as e:
            from .errors import wrap_sdk_error

            translated = wrap_sdk_error(e)
            if getattr(translated, "limit_reached", False):
                limit_reached = True
                logger.warning("Agent hit turn limit: %s", e)
            else:
                raise translated from e

        if result_message is None and not collected_messages:
            raise AgentResponseError("No messages received from SDK agent")

        return self._build_agent_result(
            collected_messages,
            result_message,
            limit_reached=limit_reached,
        )

    def run(
        self,
        messages: list[Message],
        tools: list[Tool] | None = None,
        mcp_servers: dict[str, MCPServerConfig] | None = None,
        config: AgentConfig | None = None,
    ) -> AgentResult:
        """Synchronous wrapper for arun().

        Always uses a dedicated thread with asyncio.run() to give the
        Claude Agent SDK a fresh event loop, avoiding cancel scope
        conflicts with BlockingPortal. See _run_in_fresh_loop in
        the llm module for the full rationale.

        Args:
            messages: Initial conversation messages.
            tools: Optional list of Tool definitions.
            mcp_servers: Optional MCP server configurations.
            config: Optional AgentConfig for execution parameters.

        Returns:
            AgentResult from the agent execution.

        Raises:
            AgentExecutionError: If the agent fails during execution.
            AgentTimeoutError: If execution exceeds the timeout.
            AgentResponseError: If the response is malformed or invalid.
        """
        from .llm import _run_in_fresh_loop

        timeout = (config.timeout + 30) if config and config.timeout else 600
        return _run_in_fresh_loop(
            self.arun,
            messages,
            tools,
            mcp_servers,
            config,
            timeout=timeout,
        )

    async def aclose(self) -> None:
        """Close underlying resources.

        The Claude SDK agent adapter uses ClaudeSDKClient as a context manager
        which handles its own cleanup, so this is a no-op. Provided for
        interface consistency with other adapters that do require cleanup.
        """
        pass


# Verify protocol compliance at import time
def _verify_protocol_compliance() -> None:
    """Verify ClaudeSDKAgentAdapter implements AgentPort protocol."""
    adapter_instance: AgentPort = None  # type: ignore[assignment]
    _ = adapter_instance
