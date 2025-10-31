"""Demo test to show the new failure detection functionality."""

from langchain_core.messages import AIMessage, ToolMessage

from karenina.benchmark.verification.verification_utils import _extract_agent_metrics


def test_demo_clean_tool_calls() -> None:
    """Demo: Clean tool calls (no failures)."""
    print("\n" + "=" * 50)
    print("Example 1: Clean tool calls (no failures)")
    print("=" * 50)

    response = {
        "messages": [
            AIMessage(content="Searching for information"),
            ToolMessage(content="Found 10 results", name="search_tool", tool_call_id="call_1"),
            AIMessage(content="Now calculating"),
            ToolMessage(content="Result is 42", name="calculator", tool_call_id="call_2"),
            AIMessage(content="Done!"),
        ]
    }

    metrics = _extract_agent_metrics(response)

    print(f"Iterations: {metrics['iterations']}")
    print(f"Tool calls: {metrics['tool_calls']}")
    print(f"Tools used: {metrics['tools_used']}")
    print(f"Suspect failed calls: {metrics['suspect_failed_tool_calls']}")
    print(f"Suspect failed tools: {metrics['suspect_failed_tools']}")

    assert metrics["iterations"] == 3
    assert metrics["tool_calls"] == 2
    assert metrics["suspect_failed_tool_calls"] == 0


def test_demo_mixed_success_and_failures() -> None:
    """Demo: Mixed success and failures."""
    print("\n" + "=" * 50)
    print("Example 2: Mixed success and failures")
    print("=" * 50)

    response = {
        "messages": [
            AIMessage(content="Trying API"),
            ToolMessage(content="Error: Connection refused", name="api_tool", tool_call_id="call_1"),
            AIMessage(content="Retrying with fallback"),
            ToolMessage(content="404 Not Found", name="http_tool", tool_call_id="call_2"),
            AIMessage(content="Using cache"),
            ToolMessage(content="Cache hit: returned data successfully", name="cache_tool", tool_call_id="call_3"),
            AIMessage(content="Success!"),
        ]
    }

    metrics = _extract_agent_metrics(response)

    print(f"Iterations: {metrics['iterations']}")
    print(f"Tool calls: {metrics['tool_calls']}")
    print(f"Tools used: {metrics['tools_used']}")
    print(f"Suspect failed calls: {metrics['suspect_failed_tool_calls']}")
    print(f"Suspect failed tools: {metrics['suspect_failed_tools']}")

    assert metrics["iterations"] == 4
    assert metrics["tool_calls"] == 3
    assert metrics["suspect_failed_tool_calls"] == 2
    assert set(metrics["suspect_failed_tools"]) == {"api_tool", "http_tool"}


def test_demo_multiple_failures_same_tool() -> None:
    """Demo: Multiple failures from same tool."""
    print("\n" + "=" * 50)
    print("Example 3: Multiple failures from same tool")
    print("=" * 50)

    response = {
        "messages": [
            AIMessage(content="First attempt"),
            ToolMessage(content="Timeout after 30s", name="network", tool_call_id="call_1"),
            AIMessage(content="Second attempt"),
            ToolMessage(content="Error: Cannot connect", name="network", tool_call_id="call_2"),
            AIMessage(content="Third attempt"),
            ToolMessage(content="Connected successfully!", name="network", tool_call_id="call_3"),
            AIMessage(content="Done"),
        ]
    }

    metrics = _extract_agent_metrics(response)

    print(f"Iterations: {metrics['iterations']}")
    print(f"Tool calls: {metrics['tool_calls']}")
    print(f"Tools used: {metrics['tools_used']}")
    print(f"Suspect failed calls: {metrics['suspect_failed_tool_calls']}")
    print(f"Suspect failed tools: {metrics['suspect_failed_tools']}")

    assert metrics["iterations"] == 4
    assert metrics["tool_calls"] == 3
    assert metrics["suspect_failed_tool_calls"] == 2  # 2 failures
    assert metrics["suspect_failed_tools"] == ["network"]  # Tool appears once in list

    print("\n✅ All examples completed successfully!")
