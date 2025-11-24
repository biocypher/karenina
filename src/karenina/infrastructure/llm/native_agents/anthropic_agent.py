"""Anthropic native tool-calling agent implementation.

This module provides the AnthropicNativeAgent class that uses the Anthropic SDK
directly for tool calling instead of going through LangChain/LangGraph.
"""

from __future__ import annotations

import os
from typing import Any

from .base import NativeAgentBase, NativeAgentResponse
from .usage_tracker import NativeUsageTracker


class AnthropicNativeAgent(NativeAgentBase):
    """Native Anthropic tool-calling agent.

    Uses the Anthropic SDK directly to implement an agentic tool-calling loop.
    This bypasses LangChain/LangGraph while maintaining compatibility with
    MCP tools fetched via langchain-mcp-adapters.

    Key differences from OpenAI:
    - System prompt is passed separately, not in messages
    - Responses contain content blocks (text + tool_use)
    - Tool results are sent as user messages with tool_result content type
    """

    def __init__(self, **kwargs: Any) -> None:
        """Initialize the Anthropic native agent.

        Args:
            **kwargs: Arguments passed to NativeAgentBase
        """
        super().__init__(**kwargs)

        # Import anthropic here to make it an optional dependency
        try:
            import anthropic
        except ImportError as e:
            raise ImportError(
                "The 'anthropic' package is required for native Anthropic tool calling. "
                "Install it with: pip install anthropic>=0.25.0"
            ) from e

        # Initialize Anthropic client
        api_key = self.api_key or os.getenv("ANTHROPIC_API_KEY")
        if not api_key:
            raise ValueError(
                "Anthropic API key not found. Set ANTHROPIC_API_KEY environment variable or pass api_key explicitly."
            )

        self.client = anthropic.AsyncAnthropic(api_key=api_key)

    async def ainvoke(self, messages: list[dict[str, Any]]) -> NativeAgentResponse:
        """Run the async agent loop.

        Args:
            messages: List of message dicts with 'role' and 'content' keys

        Returns:
            NativeAgentResponse with content, messages, usage, and metrics
        """
        # Anthropic requires system prompt to be separate from messages
        system_content = self.system_prompt or ""
        conversation: list[dict[str, Any]] = []

        for msg in messages:
            if msg.get("role") == "system":
                system_content = msg.get("content", "")
            else:
                conversation.append(msg)

        usage_tracker = NativeUsageTracker(self.model)

        # Metrics tracking
        iterations = 0
        tool_calls_count = 0
        tools_used: set[str] = set()
        suspect_failed_tool_calls = 0
        suspect_failed_tools: set[str] = set()
        recursion_limit_reached = False

        for iteration in range(self.max_iterations):
            iterations += 1

            # Build request kwargs
            request_kwargs: dict[str, Any] = {
                "model": self.model,
                "max_tokens": self.extra_kwargs.get("max_tokens", 4096),
                "messages": conversation,
                "temperature": self.temperature,
            }

            # Add system prompt if present
            if system_content:
                request_kwargs["system"] = system_content

            # Add tools if available
            if self.tools:
                request_kwargs["tools"] = self.tools

            # Add extra kwargs (excluding max_tokens which we handle above)
            for key, value in self.extra_kwargs.items():
                if key not in request_kwargs and key != "max_tokens":
                    request_kwargs[key] = value

            # Call Anthropic API
            response = await self.client.messages.create(**request_kwargs)

            # Track usage
            usage_tracker.track_anthropic_response(response)

            # Process content blocks
            assistant_content: list[dict[str, Any]] = []
            tool_uses: list[Any] = []

            for block in response.content:
                if block.type == "text":
                    assistant_content.append({"type": "text", "text": block.text})
                elif block.type == "tool_use":
                    tool_uses.append(block)
                    assistant_content.append(
                        {
                            "type": "tool_use",
                            "id": block.id,
                            "name": block.name,
                            "input": block.input,
                        }
                    )

            # Add assistant message to conversation
            conversation.append({"role": "assistant", "content": assistant_content})

            # Check if done (no tool uses or end_turn)
            if not tool_uses or response.stop_reason == "end_turn":
                break

            # Check if at max iterations
            if iteration == self.max_iterations - 1:
                recursion_limit_reached = True
                break

            # Execute tools and collect results
            tool_results: list[dict[str, Any]] = []
            for tool_use in tool_uses:
                tool_calls_count += 1
                tool_name = tool_use.name
                tools_used.add(tool_name)

                try:
                    result = await self._execute_tool(tool_name, tool_use.input)

                    if self._is_suspect_failure(result):
                        suspect_failed_tool_calls += 1
                        suspect_failed_tools.add(tool_name)

                except Exception as e:
                    result = f"Error executing tool: {e!s}"
                    suspect_failed_tool_calls += 1
                    suspect_failed_tools.add(tool_name)

                tool_results.append(
                    {
                        "type": "tool_result",
                        "tool_use_id": tool_use.id,
                        "content": result,
                    }
                )

            # Add tool results as user message (Anthropic convention)
            conversation.append({"role": "user", "content": tool_results})

        # Extract final text content from last assistant message
        final_content = ""
        for msg in reversed(conversation):
            if msg.get("role") == "assistant":
                content = msg.get("content", [])
                for block in content:
                    if isinstance(block, dict) and block.get("type") == "text":
                        final_content = block.get("text", "")
                        break
                if final_content:
                    break

        return NativeAgentResponse(
            content=final_content,
            messages=conversation,
            usage=usage_tracker.accumulated_usage.copy(),
            agent_metrics={
                "iterations": iterations,
                "tool_calls": tool_calls_count,
                "tools_used": list(tools_used),
                "suspect_failed_tool_calls": suspect_failed_tool_calls,
                "suspect_failed_tools": list(suspect_failed_tools),
            },
            recursion_limit_reached=recursion_limit_reached,
        )

    async def _execute_tool(self, tool_name: str, tool_args: dict[str, Any]) -> str:
        """Execute a tool using the original LangChain tool from tool_map.

        Args:
            tool_name: Name of the tool to execute
            tool_args: Arguments to pass to the tool

        Returns:
            Tool execution result as string
        """
        return await self._execute_tool_with_langchain(tool_name, tool_args)
