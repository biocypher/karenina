"""Tests for agent tool call failure detection.

Tests verify that:
1. Clean tool outputs are not counted as failures
2. Tool outputs with error keywords are detected as suspected failures
3. HTTP error codes are detected (404, 500, etc.)
4. Exception indicators are detected
5. Case-insensitive matching works correctly
6. Multiple failures from the same tool are counted correctly
7. Status field failures are detected
"""

from langchain_core.messages import AIMessage, ToolMessage

from karenina.benchmark.verification.verification_utils import _extract_agent_metrics


class TestAgentFailureDetection:
    """Test suite for tool call failure detection."""

    def test_clean_tool_output_not_flagged(self) -> None:
        """Tool call with clean output should not be counted as suspect failure."""
        # Create agent response with clean tool output
        response = {
            "messages": [
                AIMessage(content="I'll search for that"),
                ToolMessage(content="Found 5 results", name="search_tool", tool_call_id="call_1"),
                AIMessage(content="Here are the results"),
            ]
        }

        metrics = _extract_agent_metrics(response)

        assert metrics is not None
        assert metrics["tool_calls"] == 1
        assert metrics["tools_used"] == ["search_tool"]
        assert metrics["suspect_failed_tool_calls"] == 0
        assert metrics["suspect_failed_tools"] == []

    def test_error_keyword_detected(self) -> None:
        """Tool call with 'error' in output should be counted as suspect failure."""
        response = {
            "messages": [
                AIMessage(content="Let me try that"),
                ToolMessage(content="Error: Connection refused", name="api_tool", tool_call_id="call_id"),
                AIMessage(content="I encountered an error"),
            ]
        }

        metrics = _extract_agent_metrics(response)

        assert metrics is not None
        assert metrics["tool_calls"] == 1
        assert metrics["tools_used"] == ["api_tool"]
        assert metrics["suspect_failed_tool_calls"] == 1
        assert metrics["suspect_failed_tools"] == ["api_tool"]

    def test_failed_keyword_detected(self) -> None:
        """Tool call with 'failed' in output should be counted as suspect failure."""
        response = {
            "messages": [
                AIMessage(content="Executing command"),
                ToolMessage(content="Operation failed due to timeout", name="exec_tool", tool_call_id="call_id"),
                AIMessage(content="Command failed"),
            ]
        }

        metrics = _extract_agent_metrics(response)

        assert metrics is not None
        assert metrics["suspect_failed_tool_calls"] == 1
        assert metrics["suspect_failed_tools"] == ["exec_tool"]

    def test_exception_keyword_detected(self) -> None:
        """Tool call with 'exception' in output should be counted as suspect failure."""
        response = {
            "messages": [
                AIMessage(content="Processing request"),
                ToolMessage(content="ValueError exception occurred", name="process_tool", tool_call_id="call_id"),
                AIMessage(content="Exception encountered"),
            ]
        }

        metrics = _extract_agent_metrics(response)

        assert metrics is not None
        assert metrics["suspect_failed_tool_calls"] == 1
        assert metrics["suspect_failed_tools"] == ["process_tool"]

    def test_http_404_detected(self) -> None:
        """Tool call with '404' should be counted as suspect failure."""
        response = {
            "messages": [
                AIMessage(content="Fetching resource"),
                ToolMessage(content="404 Not Found", name="http_tool", tool_call_id="call_id"),
                AIMessage(content="Resource not available"),
            ]
        }

        metrics = _extract_agent_metrics(response)

        assert metrics is not None
        assert metrics["suspect_failed_tool_calls"] == 1
        assert metrics["suspect_failed_tools"] == ["http_tool"]

    def test_http_500_detected(self) -> None:
        """Tool call with '500' should be counted as suspect failure."""
        response = {
            "messages": [
                AIMessage(content="Making request"),
                ToolMessage(
                    content="Server returned 500 Internal Server Error", name="api_call", tool_call_id="call_id"
                ),
                AIMessage(content="Server error occurred"),
            ]
        }

        metrics = _extract_agent_metrics(response)

        assert metrics is not None
        assert metrics["suspect_failed_tool_calls"] == 1
        assert metrics["suspect_failed_tools"] == ["api_call"]

    def test_timeout_keyword_detected(self) -> None:
        """Tool call with 'timeout' should be counted as suspect failure."""
        response = {
            "messages": [
                AIMessage(content="Waiting for response"),
                ToolMessage(content="Request timeout after 30 seconds", name="slow_tool", tool_call_id="call_id"),
                AIMessage(content="Operation timed out"),
            ]
        }

        metrics = _extract_agent_metrics(response)

        assert metrics is not None
        assert metrics["suspect_failed_tool_calls"] == 1
        assert metrics["suspect_failed_tools"] == ["slow_tool"]

    def test_unauthorized_keyword_detected(self) -> None:
        """Tool call with 'unauthorized' should be counted as suspect failure."""
        response = {
            "messages": [
                AIMessage(content="Authenticating"),
                ToolMessage(content="401 Unauthorized - Invalid API key", name="auth_tool", tool_call_id="call_id"),
                AIMessage(content="Authentication failed"),
            ]
        }

        metrics = _extract_agent_metrics(response)

        assert metrics is not None
        assert metrics["suspect_failed_tool_calls"] == 1
        assert metrics["suspect_failed_tools"] == ["auth_tool"]

    def test_forbidden_keyword_detected(self) -> None:
        """Tool call with 'forbidden' should be counted as suspect failure."""
        response = {
            "messages": [
                AIMessage(content="Accessing resource"),
                ToolMessage(content="403 Forbidden - Access denied", name="access_tool", tool_call_id="call_id"),
                AIMessage(content="Access was denied"),
            ]
        }

        metrics = _extract_agent_metrics(response)

        assert metrics is not None
        assert metrics["suspect_failed_tool_calls"] == 1
        assert metrics["suspect_failed_tools"] == ["access_tool"]

    def test_not_found_phrase_detected(self) -> None:
        """Tool call with 'not found' phrase should be counted as suspect failure."""
        response = {
            "messages": [
                AIMessage(content="Looking up file"),
                ToolMessage(content="File not found in directory", name="file_tool", tool_call_id="call_id"),
                AIMessage(content="Could not locate file"),
            ]
        }

        metrics = _extract_agent_metrics(response)

        assert metrics is not None
        assert metrics["suspect_failed_tool_calls"] == 1
        assert metrics["suspect_failed_tools"] == ["file_tool"]

    def test_cannot_keyword_detected(self) -> None:
        """Tool call with 'cannot' should be counted as suspect failure."""
        response = {
            "messages": [
                AIMessage(content="Trying operation"),
                ToolMessage(content="Cannot connect to database", name="db_tool", tool_call_id="call_id"),
                AIMessage(content="Connection issue"),
            ]
        }

        metrics = _extract_agent_metrics(response)

        assert metrics is not None
        assert metrics["suspect_failed_tool_calls"] == 1
        assert metrics["suspect_failed_tools"] == ["db_tool"]

    def test_unable_to_phrase_detected(self) -> None:
        """Tool call with 'unable to' phrase should be counted as suspect failure."""
        response = {
            "messages": [
                AIMessage(content="Processing"),
                ToolMessage(content="Unable to parse the response", name="parse_tool", tool_call_id="call_id"),
                AIMessage(content="Parsing failed"),
            ]
        }

        metrics = _extract_agent_metrics(response)

        assert metrics is not None
        assert metrics["suspect_failed_tool_calls"] == 1
        assert metrics["suspect_failed_tools"] == ["parse_tool"]

    def test_case_insensitive_matching(self) -> None:
        """Pattern matching should be case-insensitive."""
        # Test various case combinations
        test_cases = [
            "ERROR occurred",
            "Error occurred",
            "error occurred",
            "ErRoR occurred",
        ]

        for error_text in test_cases:
            response = {
                "messages": [
                    AIMessage(content="Testing"),
                    ToolMessage(content=error_text, name="test_tool", tool_call_id="call_id"),
                    AIMessage(content="Done"),
                ]
            }

            metrics = _extract_agent_metrics(response)
            assert metrics is not None
            assert metrics["suspect_failed_tool_calls"] == 1, f"Failed to detect: {error_text}"
            assert metrics["suspect_failed_tools"] == ["test_tool"]

    def test_multiple_failures_from_same_tool(self) -> None:
        """Multiple failures from same tool should count all calls but list tool once."""
        response = {
            "messages": [
                AIMessage(content="First attempt"),
                ToolMessage(content="Error: Connection failed", name="network_tool", tool_call_id="call_id"),
                AIMessage(content="Retrying"),
                ToolMessage(content="Timeout occurred", name="network_tool", tool_call_id="call_id"),
                AIMessage(content="Failed again"),
                ToolMessage(content="Success: Connected", name="network_tool", tool_call_id="call_id"),
                AIMessage(content="Finally worked"),
            ]
        }

        metrics = _extract_agent_metrics(response)

        assert metrics is not None
        assert metrics["tool_calls"] == 3  # Total tool calls
        assert metrics["tools_used"] == ["network_tool"]
        assert metrics["suspect_failed_tool_calls"] == 2  # Two failures
        assert metrics["suspect_failed_tools"] == ["network_tool"]  # Tool appears once

    def test_multiple_tools_with_failures(self) -> None:
        """Multiple different tools with failures should be tracked separately."""
        response = {
            "messages": [
                AIMessage(content="Using tool A"),
                ToolMessage(content="Error in tool A", name="tool_a", tool_call_id="call_id"),
                AIMessage(content="Switching to tool B"),
                ToolMessage(content="Success from tool B", name="tool_b", tool_call_id="call_id"),
                AIMessage(content="Trying tool C"),
                ToolMessage(content="Exception in tool C", name="tool_c", tool_call_id="call_id"),
                AIMessage(content="Done"),
            ]
        }

        metrics = _extract_agent_metrics(response)

        assert metrics is not None
        assert metrics["tool_calls"] == 3
        assert set(metrics["tools_used"]) == {"tool_a", "tool_b", "tool_c"}
        assert metrics["suspect_failed_tool_calls"] == 2
        assert set(metrics["suspect_failed_tools"]) == {"tool_a", "tool_c"}

    def test_status_field_failure_detection(self) -> None:
        """Tool message with error status field should be detected."""
        # Create mock ToolMessage with status attribute
        tool_msg = ToolMessage(content="Some content", name="status_tool", tool_call_id="call_id")
        # Manually set status attribute (simulating what some tools might do)
        tool_msg.status = "error"

        response = {"messages": [AIMessage(content="Testing"), tool_msg, AIMessage(content="Done")]}

        metrics = _extract_agent_metrics(response)

        assert metrics is not None
        assert metrics["suspect_failed_tool_calls"] == 1
        assert metrics["suspect_failed_tools"] == ["status_tool"]

    def test_traceback_keyword_detected(self) -> None:
        """Tool call with 'traceback' should be counted as suspect failure."""
        response = {
            "messages": [
                AIMessage(content="Executing"),
                ToolMessage(
                    content="Traceback (most recent call last):\n  File ...", name="python_tool", tool_call_id="call_id"
                ),
                AIMessage(content="Error occurred"),
            ]
        }

        metrics = _extract_agent_metrics(response)

        assert metrics is not None
        assert metrics["suspect_failed_tool_calls"] == 1
        assert metrics["suspect_failed_tools"] == ["python_tool"]

    def test_stack_trace_phrase_detected(self) -> None:
        """Tool call with 'stack trace' phrase should be counted as suspect failure."""
        response = {
            "messages": [
                AIMessage(content="Running code"),
                ToolMessage(
                    content="Stack trace:\n  at function1()\n  at function2()",
                    name="debug_tool",
                    tool_call_id="call_id",
                ),
                AIMessage(content="Encountered error"),
            ]
        }

        metrics = _extract_agent_metrics(response)

        assert metrics is not None
        assert metrics["suspect_failed_tool_calls"] == 1
        assert metrics["suspect_failed_tools"] == ["debug_tool"]

    def test_word_boundary_matching(self) -> None:
        """Patterns should use word boundaries to avoid false positives."""
        # "error" should not match in words like "terrorist" or "territory"
        response = {
            "messages": [
                AIMessage(content="Searching"),
                ToolMessage(
                    content="Found information about the territory and its borders",
                    name="search_tool",
                    tool_call_id="call_id",
                ),
                AIMessage(content="Done"),
            ]
        }

        metrics = _extract_agent_metrics(response)

        assert metrics is not None
        assert metrics["suspect_failed_tool_calls"] == 0
        assert metrics["suspect_failed_tools"] == []

    def test_no_tool_messages(self) -> None:
        """Agent with no tool calls should return zero failures."""
        response = {
            "messages": [
                AIMessage(content="I can answer this directly"),
                AIMessage(content="The answer is 42"),
            ]
        }

        metrics = _extract_agent_metrics(response)

        assert metrics is not None
        assert metrics["iterations"] == 2
        assert metrics["tool_calls"] == 0
        assert metrics["tools_used"] == []
        assert metrics["suspect_failed_tool_calls"] == 0
        assert metrics["suspect_failed_tools"] == []

    def test_tool_without_name(self) -> None:
        """Tool message without name should still be counted but not in tools list."""
        # Create ToolMessage without name attribute
        tool_msg = ToolMessage(content="Error occurred", tool_call_id="call_id")
        # Ensure name is None
        tool_msg.name = None

        response = {"messages": [AIMessage(content="Testing"), tool_msg, AIMessage(content="Done")]}

        metrics = _extract_agent_metrics(response)

        assert metrics is not None
        assert metrics["tool_calls"] == 1
        assert metrics["tools_used"] == []  # No name, so not in list
        assert metrics["suspect_failed_tool_calls"] == 1  # But failure detected
        assert metrics["suspect_failed_tools"] == []  # No name, so not in list

    def test_empty_response(self) -> None:
        """Empty response should return None."""
        metrics = _extract_agent_metrics({})
        assert metrics is None

    def test_none_response(self) -> None:
        """None response should return None."""
        metrics = _extract_agent_metrics(None)
        assert metrics is None

    def test_invalid_response_type(self) -> None:
        """Invalid response type should return None."""
        metrics = _extract_agent_metrics("not a dict")
        assert metrics is None

    def test_mixed_success_and_failure(self) -> None:
        """Complex scenario with mix of successes and failures."""
        response = {
            "messages": [
                AIMessage(content="Starting task"),
                ToolMessage(content="Successfully fetched data", name="fetch_tool", tool_call_id="call_id"),
                AIMessage(content="Processing data"),
                ToolMessage(content="Error: Invalid format", name="process_tool", tool_call_id="call_id"),
                AIMessage(content="Trying alternative approach"),
                ToolMessage(content="Success: Alternative method worked", name="process_tool", tool_call_id="call_id"),
                AIMessage(content="Saving results"),
                ToolMessage(content="Saved to database", name="save_tool", tool_call_id="call_id"),
                AIMessage(content="Task complete"),
            ]
        }

        metrics = _extract_agent_metrics(response)

        assert metrics is not None
        assert metrics["iterations"] == 5  # 5 AI messages
        assert metrics["tool_calls"] == 4  # 4 tool calls total
        assert set(metrics["tools_used"]) == {"fetch_tool", "process_tool", "save_tool"}
        assert metrics["suspect_failed_tool_calls"] == 1  # Only one failure
        assert metrics["suspect_failed_tools"] == ["process_tool"]

    def test_http_502_detected(self) -> None:
        """Tool call with '502' should be counted as suspect failure."""
        response = {
            "messages": [
                AIMessage(content="Connecting"),
                ToolMessage(content="502 Bad Gateway", name="proxy_tool", tool_call_id="call_id"),
                AIMessage(content="Gateway error"),
            ]
        }

        metrics = _extract_agent_metrics(response)

        assert metrics is not None
        assert metrics["suspect_failed_tool_calls"] == 1
        assert metrics["suspect_failed_tools"] == ["proxy_tool"]

    def test_http_503_detected(self) -> None:
        """Tool call with '503' should be counted as suspect failure."""
        response = {
            "messages": [
                AIMessage(content="Requesting"),
                ToolMessage(content="503 Service Unavailable", name="service_tool", tool_call_id="call_id"),
                AIMessage(content="Service down"),
            ]
        }

        metrics = _extract_agent_metrics(response)

        assert metrics is not None
        assert metrics["suspect_failed_tool_calls"] == 1
        assert metrics["suspect_failed_tools"] == ["service_tool"]

    def test_invalid_keyword_detected(self) -> None:
        """Tool call with 'invalid' should be counted as suspect failure."""
        response = {
            "messages": [
                AIMessage(content="Validating"),
                ToolMessage(content="Invalid input provided", name="validate_tool", tool_call_id="call_id"),
                AIMessage(content="Validation failed"),
            ]
        }

        metrics = _extract_agent_metrics(response)

        assert metrics is not None
        assert metrics["suspect_failed_tool_calls"] == 1
        assert metrics["suspect_failed_tools"] == ["validate_tool"]
