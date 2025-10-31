"""Tests for LLM usage tracking functionality.

This module tests the UsageTracker utility class which aggregates
token usage metadata across verification stages.
"""

from karenina.benchmark.verification.utils import UsageTracker


class TestUsageTracker:
    """Test suite for UsageTracker class."""

    def test_track_single_call(self):
        """Test tracking a single LLM call."""
        tracker = UsageTracker()

        # Sample metadata from LangChain callback
        metadata = {
            "gpt-4o-mini": {
                "input_tokens": 100,
                "output_tokens": 50,
                "total_tokens": 150,
            }
        }

        tracker.track_call("answer_generation", "openai/gpt-4o-mini", metadata)

        # Get stage summary
        summary = tracker.get_stage_summary("answer_generation")

        assert summary is not None
        # Model name comes from metadata dict key, not the model_name parameter
        assert summary["model"] == "gpt-4o-mini"
        assert summary["input_tokens"] == 100
        assert summary["output_tokens"] == 50
        assert summary["total_tokens"] == 150

    def test_track_multiple_calls_same_stage(self):
        """Test tracking multiple calls to the same stage aggregates correctly."""
        tracker = UsageTracker()

        # First call
        metadata1 = {
            "gpt-4o-mini": {
                "input_tokens": 100,
                "output_tokens": 50,
                "total_tokens": 150,
            }
        }
        tracker.track_call("parsing", "openai/gpt-4o-mini", metadata1)

        # Second call (e.g., retry)
        metadata2 = {
            "gpt-4o-mini": {
                "input_tokens": 80,
                "output_tokens": 40,
                "total_tokens": 120,
            }
        }
        tracker.track_call("parsing", "openai/gpt-4o-mini", metadata2)

        # Get aggregated summary
        summary = tracker.get_stage_summary("parsing")

        assert summary is not None
        assert summary["input_tokens"] == 180  # 100 + 80
        assert summary["output_tokens"] == 90  # 50 + 40
        assert summary["total_tokens"] == 270  # 150 + 120

    def test_track_multiple_stages(self):
        """Test tracking calls across different stages."""
        tracker = UsageTracker()

        # Answer generation
        metadata1 = {
            "gpt-4o-mini": {
                "input_tokens": 200,
                "output_tokens": 100,
                "total_tokens": 300,
            }
        }
        tracker.track_call("answer_generation", "openai/gpt-4o-mini", metadata1)

        # Parsing
        metadata2 = {
            "gpt-4o-mini": {
                "input_tokens": 150,
                "output_tokens": 80,
                "total_tokens": 230,
            }
        }
        tracker.track_call("parsing", "openai/gpt-4o-mini", metadata2)

        # Rubric evaluation
        metadata3 = {
            "gpt-4o-mini": {
                "input_tokens": 180,
                "output_tokens": 90,
                "total_tokens": 270,
            }
        }
        tracker.track_call("rubric_evaluation", "openai/gpt-4o-mini", metadata3)

        # Verify each stage tracked separately
        answer_summary = tracker.get_stage_summary("answer_generation")
        parsing_summary = tracker.get_stage_summary("parsing")
        rubric_summary = tracker.get_stage_summary("rubric_evaluation")

        assert answer_summary["total_tokens"] == 300
        assert parsing_summary["total_tokens"] == 230
        assert rubric_summary["total_tokens"] == 270

    def test_get_total_summary(self):
        """Test getting aggregated summary across all stages with totals."""
        tracker = UsageTracker()

        # Track multiple stages
        tracker.track_call(
            "answer_generation",
            "openai/gpt-4o-mini",
            {"gpt-4o-mini": {"input_tokens": 200, "output_tokens": 100, "total_tokens": 300}},
        )

        tracker.track_call(
            "parsing",
            "openai/gpt-4o-mini",
            {"gpt-4o-mini": {"input_tokens": 150, "output_tokens": 80, "total_tokens": 230}},
        )

        tracker.track_call(
            "rubric_evaluation",
            "openai/gpt-4o-mini",
            {"gpt-4o-mini": {"input_tokens": 180, "output_tokens": 90, "total_tokens": 270}},
        )

        # Get total summary
        total = tracker.get_total_summary()

        # Should have stage breakdowns + total
        assert "answer_generation" in total
        assert "parsing" in total
        assert "rubric_evaluation" in total
        assert "total" in total

        # Verify totals
        total_summary = total["total"]
        assert total_summary["input_tokens"] == 530  # 200 + 150 + 180
        assert total_summary["output_tokens"] == 270  # 100 + 80 + 90
        assert total_summary["total_tokens"] == 800  # 300 + 230 + 270

    def test_track_agent_metrics(self):
        """Test tracking agent execution metrics."""
        tracker = UsageTracker()

        # Create mock message objects with proper types
        class MockAIMessage:
            __class__ = type("AIMessage", (), {})

        class MockToolMessage:
            __class__ = type("ToolMessage", (), {})

            def __init__(self, tool_name):
                self.name = tool_name

        # Simulate agent response with messages
        agent_response = {
            "messages": [
                MockAIMessage(),  # Iteration 1
                MockToolMessage("web_search"),
                MockToolMessage("calculator"),
                MockAIMessage(),  # Iteration 2
                MockToolMessage("file_read"),
                MockToolMessage("calculator"),  # Duplicate tool
                MockAIMessage(),  # Iteration 3
                MockToolMessage("web_search"),  # Duplicate tool
            ]
        }

        # Track agent metrics
        tracker.track_agent_metrics(agent_response)

        # Retrieve agent metrics
        retrieved_metrics = tracker.get_agent_metrics()

        assert retrieved_metrics is not None
        assert retrieved_metrics["iterations"] == 3
        assert retrieved_metrics["tool_calls"] == 5
        assert len(retrieved_metrics["tools_used"]) == 3  # Unique tools
        assert "web_search" in retrieved_metrics["tools_used"]
        assert "calculator" in retrieved_metrics["tools_used"]
        assert "file_read" in retrieved_metrics["tools_used"]

    def test_get_stage_summary_nonexistent(self):
        """Test getting summary for a stage that wasn't tracked."""
        tracker = UsageTracker()

        summary = tracker.get_stage_summary("nonexistent_stage")

        assert summary is None

    def test_empty_tracker_total_summary(self):
        """Test getting total summary from empty tracker."""
        tracker = UsageTracker()

        total = tracker.get_total_summary()

        # Should be empty dict if no calls tracked
        assert total == {}
        assert "total" not in total

    def test_track_with_empty_metadata(self):
        """Test tracking with empty metadata doesn't crash."""
        tracker = UsageTracker()

        # Empty metadata
        tracker.track_call("answer_generation", "openai/gpt-4o-mini", {})

        summary = tracker.get_stage_summary("answer_generation")

        # Should exist but with zero tokens
        assert summary is not None
        assert summary["input_tokens"] == 0
        assert summary["output_tokens"] == 0
        assert summary["total_tokens"] == 0

    def test_track_with_partial_metadata(self):
        """Test tracking with partial/incomplete metadata."""
        tracker = UsageTracker()

        # Metadata missing some fields
        metadata = {"gpt-4o-mini": {"input_tokens": 100}}

        tracker.track_call("parsing", "openai/gpt-4o-mini", metadata)

        summary = tracker.get_stage_summary("parsing")

        assert summary is not None
        assert summary["input_tokens"] == 100
        assert summary["output_tokens"] == 0  # Should default to 0
        assert summary["total_tokens"] == 0

    def test_track_with_token_details(self):
        """Test tracking metadata with detailed token breakdown."""
        tracker = UsageTracker()

        # Metadata with token details (e.g., cache hits)
        metadata = {
            "gpt-4o-mini": {
                "input_tokens": 200,
                "output_tokens": 100,
                "total_tokens": 300,
                "input_token_details": {"cache_read": 50, "audio": 0},
                "output_token_details": {"reasoning": 20, "audio": 0},
            }
        }

        tracker.track_call("answer_generation", "openai/gpt-4o-mini", metadata)

        summary = tracker.get_stage_summary("answer_generation")

        assert summary is not None
        assert "input_token_details" in summary
        assert summary["input_token_details"]["cache_read"] == 50

    def test_multiple_models_same_stage(self):
        """Test tracking different models in same stage (edge case)."""
        tracker = UsageTracker()

        # First call with gpt-4o-mini
        metadata1 = {
            "gpt-4o-mini": {
                "input_tokens": 100,
                "output_tokens": 50,
                "total_tokens": 150,
            }
        }
        tracker.track_call("answer_generation", "openai/gpt-4o-mini", metadata1)

        # Second call with different model (e.g., after retry with different model)
        metadata2 = {
            "gpt-4o": {
                "input_tokens": 120,
                "output_tokens": 60,
                "total_tokens": 180,
            }
        }
        tracker.track_call("answer_generation", "openai/gpt-4o", metadata2)

        # Should aggregate tokens, model name comes from first call
        summary = tracker.get_stage_summary("answer_generation")

        assert summary is not None
        assert summary["input_tokens"] == 220  # 100 + 120
        assert summary["output_tokens"] == 110  # 50 + 60
        assert summary["total_tokens"] == 330  # 150 + 180
        # Model name from first call's metadata key
        assert summary["model"] == "gpt-4o-mini"

    def test_get_agent_metrics_when_none(self):
        """Test getting agent metrics when none were tracked."""
        tracker = UsageTracker()

        metrics = tracker.get_agent_metrics()

        assert metrics is None

    def test_stage_summary_includes_model(self):
        """Test that stage summary includes model information."""
        tracker = UsageTracker()

        metadata = {
            "gpt-4o-mini": {
                "input_tokens": 100,
                "output_tokens": 50,
                "total_tokens": 150,
            }
        }

        tracker.track_call("answer_generation", "openai/gpt-4o-mini", metadata)

        summary = tracker.get_stage_summary("answer_generation")

        assert "model" in summary
        # Model name comes from metadata dict key
        assert summary["model"] == "gpt-4o-mini"
