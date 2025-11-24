"""OpenAI native tool-calling agent implementation.

This module provides the OpenAINativeAgent class that uses the OpenAI SDK
directly for tool calling instead of going through LangChain/LangGraph.
"""

from __future__ import annotations

import json
import os
from typing import Any

from .base import NativeAgentBase, NativeAgentResponse
from .usage_tracker import NativeUsageTracker


class OpenAINativeAgent(NativeAgentBase):
    """Native OpenAI tool-calling agent.

    Uses the OpenAI SDK directly to implement an agentic tool-calling loop.
    This bypasses LangChain/LangGraph while maintaining compatibility with
    MCP tools fetched via langchain-mcp-adapters.

    The agent loop:
    1. Sends messages to OpenAI with tool definitions
    2. If the model returns tool_calls, executes them
    3. Adds tool results to conversation and repeats
    4. Stops when no more tool calls or max_iterations reached
    """

    def __init__(self, **kwargs: Any) -> None:
        """Initialize the OpenAI native agent.

        Args:
            **kwargs: Arguments passed to NativeAgentBase
        """
        super().__init__(**kwargs)

        # Import openai here to make it an optional dependency
        try:
            import openai
        except ImportError as e:
            raise ImportError(
                "The 'openai' package is required for native OpenAI tool calling. "
                "Install it with: pip install openai>=1.0.0"
            ) from e

        # Initialize OpenAI clients
        api_key = self.api_key or os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError(
                "OpenAI API key not found. Set OPENAI_API_KEY environment variable or pass api_key explicitly."
            )

        self.client = openai.AsyncOpenAI(api_key=api_key)

    async def ainvoke(self, messages: list[dict[str, Any]]) -> NativeAgentResponse:
        """Run the async agent loop.

        Args:
            messages: List of message dicts with 'role' and 'content' keys

        Returns:
            NativeAgentResponse with content, messages, usage, and metrics
        """
        conversation = list(messages)
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
                "messages": conversation,
                "temperature": self.temperature,
            }

            # Add tools if available
            if self.tools:
                request_kwargs["tools"] = self.tools
                request_kwargs["tool_choice"] = "auto"

            # Add extra kwargs (e.g., max_tokens)
            for key, value in self.extra_kwargs.items():
                if key not in request_kwargs:
                    request_kwargs[key] = value

            # Call OpenAI API
            response = await self.client.chat.completions.create(**request_kwargs)

            # Track usage
            usage_tracker.track_openai_response(response)

            # Get assistant message
            assistant_message = response.choices[0].message

            # Build message dict for conversation history
            message_dict: dict[str, Any] = {
                "role": "assistant",
                "content": assistant_message.content or "",
            }

            # Add tool calls if present
            if assistant_message.tool_calls:
                message_dict["tool_calls"] = [
                    {
                        "id": tc.id,
                        "type": "function",
                        "function": {
                            "name": tc.function.name,
                            "arguments": tc.function.arguments,
                        },
                    }
                    for tc in assistant_message.tool_calls
                ]

            conversation.append(message_dict)

            # Check if done (no tool calls)
            if not assistant_message.tool_calls:
                break

            # Check if at max iterations
            if iteration == self.max_iterations - 1:
                recursion_limit_reached = True
                break

            # Execute each tool call
            for tool_call in assistant_message.tool_calls:
                tool_calls_count += 1
                tool_name = tool_call.function.name
                tools_used.add(tool_name)

                try:
                    # Parse arguments
                    args = json.loads(tool_call.function.arguments)
                    result = await self._execute_tool(tool_name, args)

                    if self._is_suspect_failure(result):
                        suspect_failed_tool_calls += 1
                        suspect_failed_tools.add(tool_name)

                except json.JSONDecodeError as e:
                    result = f"Error parsing tool arguments: {e!s}"
                    suspect_failed_tool_calls += 1
                    suspect_failed_tools.add(tool_name)
                except Exception as e:
                    result = f"Error executing tool: {e!s}"
                    suspect_failed_tool_calls += 1
                    suspect_failed_tools.add(tool_name)

                # Add tool result to conversation
                conversation.append(
                    {
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "content": result,
                    }
                )

        # Extract final content from last assistant message
        final_content = ""
        for msg in reversed(conversation):
            if msg.get("role") == "assistant" and msg.get("content"):
                final_content = msg["content"]
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
