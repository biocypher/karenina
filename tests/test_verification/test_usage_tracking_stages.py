"""Integration tests for usage tracking across verification stages.

Tests verify that:
1. Each stage properly captures usage metadata from LLM calls
2. UsageTracker is passed through context correctly
3. Agent metrics are tracked when MCP agents are used
4. End-to-end pipeline aggregates usage correctly
"""

from unittest.mock import MagicMock, Mock, patch

from langchain_core.messages import AIMessage

from karenina.benchmark.verification.stage import VerificationContext
from karenina.benchmark.verification.stages.abstention_check import AbstentionCheckStage
from karenina.benchmark.verification.stages.generate_answer import GenerateAnswerStage
from karenina.benchmark.verification.stages.parse_template import ParseTemplateStage
from karenina.benchmark.verification.stages.rubric_evaluation import RubricEvaluationStage
from karenina.benchmark.verification.utils import UsageTracker
from karenina.schemas.domain import BaseAnswer, LLMRubricTrait, Rubric


class TestGenerateAnswerStageUsageTracking:
    """Test usage tracking in GenerateAnswerStage."""

    @patch("karenina.benchmark.verification.stages.generate_answer._invoke_llm_with_retry")
    @patch("karenina.benchmark.verification.stages.generate_answer.init_chat_model_unified")
    def test_tracks_usage_metadata(
        self,
        mock_init_llm: Mock,
        mock_invoke: Mock,
        basic_context: VerificationContext,
    ) -> None:
        """Test that GenerateAnswerStage tracks usage metadata."""

        # Set up Answer artifact
        class MockAnswer(BaseAnswer):
            def verify(self) -> bool:
                return True

        basic_context.set_artifact("Answer", MockAnswer)

        # Mock LLM initialization
        mock_llm = MagicMock()
        mock_init_llm.return_value = mock_llm

        # Mock LLM response with usage metadata (4-tuple)
        mock_response = AIMessage(content="The answer is 4")
        usage_metadata = {
            "gpt-4.1-mini": {
                "input_tokens": 100,
                "output_tokens": 50,
                "total_tokens": 150,
                "input_token_details": {"audio": 0, "cache_read": 0},
                "output_token_details": {"audio": 0, "reasoning": 0},
            }
        }
        mock_invoke.return_value = (mock_response, False, usage_metadata, None)

        # Execute stage
        stage = GenerateAnswerStage()
        stage.execute(basic_context)

        # Verify no error
        assert basic_context.error is None

        # Verify usage tracker was created and stored in context
        assert basic_context.has_artifact("usage_tracker")
        tracker = basic_context.get_artifact("usage_tracker")
        assert isinstance(tracker, UsageTracker)

        # Verify usage was tracked for answer_generation stage
        summary = tracker.get_stage_summary("answer_generation")
        assert summary is not None
        assert summary["total_tokens"] == 150
        assert summary["input_tokens"] == 100
        assert summary["output_tokens"] == 50

    @patch("karenina.benchmark.verification.stages.generate_answer._invoke_llm_with_retry")
    @patch("karenina.benchmark.verification.stages.generate_answer.init_chat_model_unified")
    def test_tracks_agent_metrics(
        self,
        mock_init_llm: Mock,
        mock_invoke: Mock,
        basic_context: VerificationContext,
    ) -> None:
        """Test that GenerateAnswerStage tracks agent metrics for MCP agents."""
        # Configure context for MCP
        basic_context.answering_model.mcp_urls_dict = {"test-server": "http://localhost:8000"}

        # Set up Answer artifact
        class MockAnswer(BaseAnswer):
            def verify(self) -> bool:
                return True

        basic_context.set_artifact("Answer", MockAnswer)

        # Mock LLM with MCP support
        mock_llm = MagicMock()
        mock_init_llm.return_value = mock_llm

        # Mock LLM response with agent metrics (4-tuple)
        mock_response = "Agent response after harmonization"
        usage_metadata = {
            "gpt-4.1-mini": {
                "input_tokens": 200,
                "output_tokens": 100,
                "total_tokens": 300,
            }
        }
        agent_metrics = {
            "iterations": 3,
            "tool_calls": 5,
            "tools_used": ["mcp__brave_search", "mcp__read_resource"],
        }
        mock_invoke.return_value = (mock_response, False, usage_metadata, agent_metrics)

        # Execute stage
        stage = GenerateAnswerStage()
        stage.execute(basic_context)

        # Verify agent metrics were stored in UsageTracker
        tracker = basic_context.get_artifact("usage_tracker")
        assert tracker is not None
        stored_metrics = tracker.get_agent_metrics()
        assert stored_metrics is not None
        assert stored_metrics == agent_metrics
        assert stored_metrics["iterations"] == 3
        assert stored_metrics["tool_calls"] == 5
        assert "mcp__brave_search" in stored_metrics["tools_used"]


class TestParseTemplateStageUsageTracking:
    """Test usage tracking in ParseTemplateStage."""

    @patch("karenina.benchmark.verification.stages.parse_template.init_chat_model_unified")
    def test_tracks_usage_metadata_standard_parsing(
        self,
        mock_init_llm: Mock,
        basic_context: VerificationContext,
    ) -> None:
        """Test that ParseTemplateStage tracks usage metadata for standard parsing."""

        # Set up required artifacts
        class MockAnswer(BaseAnswer):
            result: int
            correct: dict

            def verify(self) -> bool:
                return True

        basic_context.set_artifact("RawAnswer", MockAnswer)
        basic_context.set_artifact("Answer", MockAnswer)
        basic_context.set_artifact("raw_llm_response", '{"result": 4, "correct": {"value": 4}}')

        # Initialize usage tracker (as if coming from previous stage)
        tracker = UsageTracker()
        tracker.track_call(
            "answer_generation",
            "gpt-4.1-mini",
            {
                "gpt-4.1-mini": {
                    "input_tokens": 100,
                    "output_tokens": 50,
                    "total_tokens": 150,
                }
            },
        )
        basic_context.set_artifact("usage_tracker", tracker)

        # Mock parsing LLM
        mock_llm = MagicMock()
        mock_init_llm.return_value = mock_llm

        # Mock parsing response with usage metadata callback
        mock_parsed_response = AIMessage(content='{"result": 4, "correct": {"value": 4}}')
        mock_llm.invoke.return_value = mock_parsed_response

        # We need to mock the usage metadata callback
        with patch("karenina.benchmark.verification.stages.parse_template.get_usage_metadata_callback") as mock_cb:
            # Create a mock callback context manager
            mock_cb_instance = MagicMock()
            mock_cb_instance.usage_metadata = {
                "gpt-4.1-mini": {
                    "input_tokens": 50,
                    "output_tokens": 30,
                    "total_tokens": 80,
                }
            }
            mock_cb.return_value.__enter__.return_value = mock_cb_instance
            mock_cb.return_value.__exit__.return_value = None

            # Execute stage
            stage = ParseTemplateStage()
            stage.execute(basic_context)

        # Verify usage tracker was updated
        tracker = basic_context.get_artifact("usage_tracker")
        assert tracker is not None

        # Verify parsing stage usage was tracked
        parsing_summary = tracker.get_stage_summary("parsing")
        assert parsing_summary is not None
        assert parsing_summary["total_tokens"] == 80
        assert parsing_summary["input_tokens"] == 50
        assert parsing_summary["output_tokens"] == 30

        # Verify total includes both stages
        total_summary = tracker.get_total_summary()
        assert total_summary["total"]["total_tokens"] == 230  # 150 + 80


class TestRubricEvaluationStageUsageTracking:
    """Test usage tracking in RubricEvaluationStage."""

    @patch("karenina.benchmark.verification.stages.rubric_evaluation.RubricEvaluator")
    def test_tracks_usage_metadata(
        self,
        mock_evaluator_class: Mock,
        basic_context: VerificationContext,
    ) -> None:
        """Test that RubricEvaluationStage tracks usage metadata from rubric evaluation."""

        # Set up required artifacts
        basic_context.set_artifact("raw_llm_response", "The answer is 4")
        basic_context.set_artifact("parsed_answer", {"result": 4})

        # Create rubric with traits
        rubric = Rubric(
            llm_traits=[
                LLMRubricTrait(
                    name="Accuracy",
                    description="Is the answer accurate?",
                    kind="score",
                    min_score=1,
                    max_score=10,
                ),
            ],
        )
        basic_context.rubric = rubric

        # Initialize usage tracker (as if coming from previous stages)
        tracker = UsageTracker()
        tracker.track_call(
            "answer_generation",
            "gpt-4.1-mini",
            {"gpt-4.1-mini": {"input_tokens": 100, "output_tokens": 50, "total_tokens": 150}},
        )
        basic_context.set_artifact("usage_tracker", tracker)

        # Mock rubric evaluator
        mock_evaluator = MagicMock()
        mock_evaluator_class.return_value = mock_evaluator

        # Mock evaluate_rubric to return results with usage metadata (tuple)
        rubric_results = {"Accuracy": 10}
        usage_metadata_list = [
            {
                "gpt-4.1-mini": {
                    "input_tokens": 80,
                    "output_tokens": 20,
                    "total_tokens": 100,
                }
            }
        ]
        mock_evaluator.evaluate_rubric.return_value = (rubric_results, usage_metadata_list)
        mock_evaluator.evaluate_metric_traits.return_value = ({}, [])

        # Execute stage
        stage = RubricEvaluationStage()
        stage.execute(basic_context)

        # Verify usage tracker was updated
        tracker = basic_context.get_artifact("usage_tracker")
        assert tracker is not None

        # Verify rubric evaluation usage was tracked
        rubric_summary = tracker.get_stage_summary("rubric_evaluation")
        assert rubric_summary is not None
        assert rubric_summary["total_tokens"] == 100
        assert rubric_summary["input_tokens"] == 80
        assert rubric_summary["output_tokens"] == 20

        # Verify total includes all stages
        total_summary = tracker.get_total_summary()
        assert total_summary["total"]["total_tokens"] == 250  # 150 + 100


class TestAbstentionCheckStageUsageTracking:
    """Test usage tracking in AbstentionCheckStage."""

    @patch("karenina.benchmark.verification.stages.abstention_check.detect_abstention")
    def test_tracks_usage_metadata(
        self,
        mock_detect: Mock,
        basic_context: VerificationContext,
    ) -> None:
        """Test that AbstentionCheckStage tracks usage metadata."""

        # Enable abstention checking
        basic_context.abstention_enabled = True

        # Set up required artifacts
        basic_context.set_artifact("raw_llm_response", "The answer is 4")

        # Initialize usage tracker (as if coming from previous stages)
        tracker = UsageTracker()
        tracker.track_call(
            "answer_generation",
            "gpt-4.1-mini",
            {"gpt-4.1-mini": {"input_tokens": 100, "output_tokens": 50, "total_tokens": 150}},
        )
        basic_context.set_artifact("usage_tracker", tracker)

        # Mock detect_abstention to return 4-tuple with usage metadata
        usage_metadata = {
            "gpt-4.1-mini": {
                "input_tokens": 40,
                "output_tokens": 10,
                "total_tokens": 50,
            }
        }
        mock_detect.return_value = (False, True, "No abstention detected", usage_metadata)

        # Execute stage
        stage = AbstentionCheckStage()
        stage.execute(basic_context)

        # Verify usage tracker was updated
        tracker = basic_context.get_artifact("usage_tracker")
        assert tracker is not None

        # Verify abstention check usage was tracked
        abstention_summary = tracker.get_stage_summary("abstention_check")
        assert abstention_summary is not None
        assert abstention_summary["total_tokens"] == 50
        assert abstention_summary["input_tokens"] == 40
        assert abstention_summary["output_tokens"] == 10

        # Verify total includes all stages
        total_summary = tracker.get_total_summary()
        assert total_summary["total"]["total_tokens"] == 200  # 150 + 50


class TestEndToEndUsageTracking:
    """Test end-to-end usage tracking through multiple stages."""

    @patch("karenina.benchmark.verification.stages.abstention_check.detect_abstention")
    @patch("karenina.benchmark.verification.stages.rubric_evaluation.RubricEvaluator")
    @patch("karenina.benchmark.verification.stages.parse_template.init_chat_model_unified")
    @patch("karenina.benchmark.verification.stages.generate_answer._invoke_llm_with_retry")
    @patch("karenina.benchmark.verification.stages.generate_answer.init_chat_model_unified")
    def test_usage_tracking_across_all_stages(
        self,
        mock_init_answer_llm: Mock,
        mock_invoke: Mock,
        mock_init_parsing_llm: Mock,
        mock_evaluator_class: Mock,
        mock_detect_abstention: Mock,
        basic_context: VerificationContext,
    ) -> None:
        """Test that usage metadata is tracked and aggregated across all stages."""

        # Enable all features
        basic_context.abstention_enabled = True
        rubric = Rubric(
            llm_traits=[
                LLMRubricTrait(
                    name="Accuracy",
                    description="Is the answer accurate?",
                    kind="score",
                    min_score=1,
                    max_score=10,
                ),
            ],
        )
        basic_context.rubric = rubric

        # Set up Answer artifact
        class MockAnswer(BaseAnswer):
            result: int
            correct: dict

            def verify(self) -> bool:
                return True

        basic_context.set_artifact("Answer", MockAnswer)

        # === Stage 1: GenerateAnswer ===
        mock_answer_llm = MagicMock()
        mock_init_answer_llm.return_value = mock_answer_llm

        mock_response = AIMessage(content="The answer is 4")
        answer_usage = {
            "gpt-4.1-mini": {
                "input_tokens": 100,
                "output_tokens": 50,
                "total_tokens": 150,
            }
        }
        mock_invoke.return_value = (mock_response, False, answer_usage, None)

        generate_stage = GenerateAnswerStage()
        generate_stage.execute(basic_context)

        # Verify tracker was created
        assert basic_context.has_artifact("usage_tracker")
        tracker = basic_context.get_artifact("usage_tracker")
        assert tracker.get_stage_summary("answer_generation") is not None

        # === Stage 2: ParseTemplate ===
        basic_context.set_artifact("RawAnswer", MockAnswer)
        basic_context.set_artifact("raw_llm_response", '{"result": 4, "correct": {"value": 4}}')

        mock_parsing_llm = MagicMock()
        mock_init_parsing_llm.return_value = mock_parsing_llm
        mock_parsing_response = AIMessage(content='{"result": 4, "correct": {"value": 4}}')
        mock_parsing_llm.invoke.return_value = mock_parsing_response

        with patch("karenina.benchmark.verification.stages.parse_template.get_usage_metadata_callback") as mock_cb:
            mock_cb_instance = MagicMock()
            mock_cb_instance.usage_metadata = {
                "gpt-4.1-mini": {
                    "input_tokens": 50,
                    "output_tokens": 30,
                    "total_tokens": 80,
                }
            }
            mock_cb.return_value.__enter__.return_value = mock_cb_instance
            mock_cb.return_value.__exit__.return_value = None

            parse_stage = ParseTemplateStage()
            parse_stage.execute(basic_context)

        # Verify parsing was tracked
        tracker = basic_context.get_artifact("usage_tracker")
        assert tracker.get_stage_summary("parsing") is not None

        # === Stage 3: RubricEvaluation ===
        basic_context.set_artifact("parsed_answer", {"result": 4})

        mock_evaluator = MagicMock()
        mock_evaluator_class.return_value = mock_evaluator
        rubric_results = {"Accuracy": 10}
        rubric_usage = [
            {
                "gpt-4.1-mini": {
                    "input_tokens": 80,
                    "output_tokens": 20,
                    "total_tokens": 100,
                }
            }
        ]
        mock_evaluator.evaluate_rubric.return_value = (rubric_results, rubric_usage)
        mock_evaluator.evaluate_metric_traits.return_value = ({}, [])

        rubric_stage = RubricEvaluationStage()
        rubric_stage.execute(basic_context)

        # Verify rubric evaluation was tracked
        tracker = basic_context.get_artifact("usage_tracker")
        assert tracker.get_stage_summary("rubric_evaluation") is not None

        # === Stage 4: AbstentionCheck ===
        abstention_usage = {
            "gpt-4.1-mini": {
                "input_tokens": 40,
                "output_tokens": 10,
                "total_tokens": 50,
            }
        }
        mock_detect_abstention.return_value = (False, True, "No abstention", abstention_usage)

        abstention_stage = AbstentionCheckStage()
        abstention_stage.execute(basic_context)

        # === Verify Final Aggregation ===
        tracker = basic_context.get_artifact("usage_tracker")
        total_summary = tracker.get_total_summary()

        # Verify all stages are present
        assert "answer_generation" in total_summary
        assert "parsing" in total_summary
        assert "rubric_evaluation" in total_summary
        assert "abstention_check" in total_summary
        assert "total" in total_summary

        # Verify totals are correct
        assert total_summary["answer_generation"]["total_tokens"] == 150
        assert total_summary["parsing"]["total_tokens"] == 80
        assert total_summary["rubric_evaluation"]["total_tokens"] == 100
        assert total_summary["abstention_check"]["total_tokens"] == 50

        # Verify grand total
        assert total_summary["total"]["total_tokens"] == 380  # 150 + 80 + 100 + 50
        assert total_summary["total"]["input_tokens"] == 270  # 100 + 50 + 80 + 40
        assert total_summary["total"]["output_tokens"] == 110  # 50 + 30 + 20 + 10

    @patch("karenina.benchmark.verification.stages.generate_answer._invoke_llm_with_retry")
    @patch("karenina.benchmark.verification.stages.generate_answer.init_chat_model_unified")
    def test_usage_tracking_with_mcp_agent(
        self,
        mock_init_llm: Mock,
        mock_invoke: Mock,
        basic_context: VerificationContext,
    ) -> None:
        """Test that agent metrics are properly tracked in end-to-end flow."""
        # Configure context for MCP
        basic_context.answering_model.mcp_urls_dict = {"test-server": "http://localhost:8000"}

        # Set up Answer artifact
        class MockAnswer(BaseAnswer):
            def verify(self) -> bool:
                return True

        basic_context.set_artifact("Answer", MockAnswer)

        # Mock LLM with MCP support
        mock_llm = MagicMock()
        mock_init_llm.return_value = mock_llm

        # Mock agent response with both usage and agent metrics
        mock_response = "Agent response"
        usage_metadata = {
            "gpt-4.1-mini": {
                "input_tokens": 200,
                "output_tokens": 150,
                "total_tokens": 350,
            }
        }
        agent_metrics = {
            "iterations": 4,
            "tool_calls": 8,
            "tools_used": ["mcp__brave_search", "mcp__read_resource", "mcp__write_file"],
        }
        mock_invoke.return_value = (mock_response, False, usage_metadata, agent_metrics)

        # Execute stage
        stage = GenerateAnswerStage()
        stage.execute(basic_context)

        # Verify both usage and agent metrics were tracked
        tracker = basic_context.get_artifact("usage_tracker")
        assert tracker is not None

        # Verify usage tracking
        summary = tracker.get_stage_summary("answer_generation")
        assert summary["total_tokens"] == 350

        # Verify agent metrics were stored separately
        stored_agent_metrics = tracker.get_agent_metrics()
        assert stored_agent_metrics is not None
        assert stored_agent_metrics["iterations"] == 4
        assert stored_agent_metrics["tool_calls"] == 8
        assert len(stored_agent_metrics["tools_used"]) == 3

    def test_usage_tracker_created_even_without_prior_tracker(
        self,
        basic_context: VerificationContext,
    ) -> None:
        """Test that stages create UsageTracker if not present in context."""

        # Set up minimal artifacts for AbstentionCheckStage
        basic_context.abstention_enabled = True
        basic_context.set_artifact("raw_llm_response", "The answer is 4")

        # Ensure no tracker exists
        basic_context.artifacts.pop("usage_tracker", None)

        with patch("karenina.benchmark.verification.stages.abstention_check.detect_abstention") as mock_detect:
            usage_metadata = {
                "gpt-4.1-mini": {
                    "input_tokens": 40,
                    "output_tokens": 10,
                    "total_tokens": 50,
                }
            }
            mock_detect.return_value = (False, True, "No abstention", usage_metadata)

            # Execute stage
            stage = AbstentionCheckStage()
            stage.execute(basic_context)

            # Verify tracker was created
            assert basic_context.has_artifact("usage_tracker")
            tracker = basic_context.get_artifact("usage_tracker")
            assert isinstance(tracker, UsageTracker)
            assert tracker.get_stage_summary("abstention_check") is not None
