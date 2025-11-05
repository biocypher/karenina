"""Integration test for the complete ManualTraces workflow."""

import hashlib

import pytest
from langchain_core.messages import AIMessage, ToolMessage
from pydantic import Field

from karenina.benchmark import Benchmark
from karenina.infrastructure.llm.manual_traces import (
    ManualTraces,
    clear_manual_traces,
    get_manual_trace,
)
from karenina.schemas import ModelConfig, VerificationConfig
from karenina.schemas.domain import BaseAnswer


class SimpleAnswer(BaseAnswer):
    """Simple answer template for testing."""

    value: str = Field(description="The answer value")

    def verify(self) -> bool:
        """Verify the answer."""
        return len(self.value) > 0


class TestManualTracesIntegration:
    """Integration tests for complete ManualTraces workflow."""

    @pytest.fixture(autouse=True)
    def cleanup_traces(self):
        """Clean up traces before and after each test."""
        clear_manual_traces()
        yield
        clear_manual_traces()

    def test_complete_workflow_with_string_traces(self):
        """Test complete workflow with string traces - based on example usage."""
        # 1. Create benchmark
        benchmark = Benchmark("integration_test_benchmark")
        benchmark.add_question(question="What is 2+2?", raw_answer="4", answer_template=SimpleAnswer)
        benchmark.add_question(
            question="What is the capital of France?", raw_answer="Paris", answer_template=SimpleAnswer
        )

        # 2. Create ManualTraces
        manual_traces = ManualTraces(benchmark)

        # 3. Register traces using question text (string format)
        manual_traces.register_trace(
            "What is 2+2?", "The answer is 4. I computed this by adding 2 and 2.", map_to_id=True
        )

        manual_traces.register_trace(
            "What is the capital of France?",
            "The capital of France is Paris. It is a major European city.",
            map_to_id=True,
        )

        # 4. Create ModelConfig with manual_traces
        manual_config = ModelConfig(interface="manual", manual_traces=manual_traces)

        # Verify config was created correctly
        assert manual_config.interface == "manual"
        assert manual_config.id == "manual"
        assert manual_config.model_name == "manual"
        assert manual_config.manual_traces == manual_traces

        # 5. Create judge config
        judge_config = ModelConfig(
            id="test-judge",
            model_provider="openai",
            model_name="gpt-4o-mini",
            temperature=0.0,
            interface="langchain",
            system_prompt="You are an expert judge. Parse responses carefully.",
        )

        # 6. Create VerificationConfig
        config = VerificationConfig(answering_models=[manual_config], parsing_models=[judge_config])

        # Verify config structure
        assert len(config.answering_models) == 1
        assert len(config.parsing_models) == 1
        assert config.answering_models[0].interface == "manual"

        # 7. Verify traces are accessible
        question1_hash = hashlib.md5(b"What is 2+2?").hexdigest()
        question2_hash = hashlib.md5(b"What is the capital of France?").hexdigest()

        trace1 = get_manual_trace(question1_hash)
        trace2 = get_manual_trace(question2_hash)

        assert trace1 == "The answer is 4. I computed this by adding 2 and 2."
        assert trace2 == "The capital of France is Paris. It is a major European city."

    def test_complete_workflow_with_message_traces(self):
        """Test complete workflow with LangChain message list traces."""
        # 1. Create benchmark
        benchmark = Benchmark("integration_test_messages")
        benchmark.add_question(question="What is 2+2?", raw_answer="4", answer_template=SimpleAnswer)

        # 2. Create ManualTraces
        manual_traces = ManualTraces(benchmark)

        # 3. Register trace with LangChain message list format
        messages = [
            AIMessage(content="Let me calculate this step by step"),
            ToolMessage(name="calculator", content="Result: 4", tool_call_id="call_calc_123"),
            AIMessage(content="The answer is 4. This is the sum of 2 and 2."),
        ]

        manual_traces.register_trace("What is 2+2?", messages, map_to_id=True)

        # 4. Create ModelConfig
        ModelConfig(interface="manual", manual_traces=manual_traces)

        # 5. Verify trace was preprocessed correctly
        question_hash = hashlib.md5(b"What is 2+2?").hexdigest()
        trace = get_manual_trace(question_hash)

        # Trace should be harmonized to string
        assert trace is not None
        assert isinstance(trace, str)
        # Should contain content from the messages
        assert "calculator" in trace or "The answer is 4" in trace

        # 6. Verify agent metrics were extracted
        from karenina.infrastructure.llm.manual_traces import get_manual_trace_with_metrics

        trace, metrics = get_manual_trace_with_metrics(question_hash)
        assert metrics is not None
        assert "tool_calls" in metrics
        assert metrics["tool_calls"] >= 1  # Should have detected the tool call

    def test_batch_registration_workflow(self):
        """Test batch registration of traces."""
        # 1. Create benchmark
        benchmark = Benchmark("batch_test")
        benchmark.add_question(question="Question 1?", raw_answer="Answer 1")
        benchmark.add_question(question="Question 2?", raw_answer="Answer 2")
        benchmark.add_question(question="Question 3?", raw_answer="Answer 3")

        # 2. Create ManualTraces
        manual_traces = ManualTraces(benchmark)

        # 3. Batch register using question texts
        traces_dict = {
            "Question 1?": "This is the answer to question 1",
            "Question 2?": "This is the answer to question 2",
            "Question 3?": "This is the answer to question 3",
        }

        manual_traces.register_traces(traces_dict, map_to_id=True)

        # 4. Verify all traces were registered
        for question_text, expected_trace in traces_dict.items():
            question_hash = hashlib.md5(question_text.encode("utf-8")).hexdigest()
            retrieved_trace = get_manual_trace(question_hash)
            assert retrieved_trace == expected_trace

        # 5. Create config and verify
        config = ModelConfig(interface="manual", manual_traces=manual_traces)
        assert config.manual_traces == manual_traces

    def test_populate_traces_after_config_creation(self):
        """Test registering traces after ModelConfig is created."""
        # 1. Create benchmark and ManualTraces
        benchmark = Benchmark("delayed_registration")
        benchmark.add_question(question="What is Python?", raw_answer="A programming language")

        manual_traces = ManualTraces(benchmark)

        # 2. Create ModelConfig BEFORE registering traces
        manual_config = ModelConfig(interface="manual", manual_traces=manual_traces)

        # 3. Register trace AFTER config creation
        manual_traces.register_trace(
            "What is Python?", "Python is a high-level programming language known for its readability.", map_to_id=True
        )

        # 4. Verify trace is accessible
        question_hash = hashlib.md5(b"What is Python?").hexdigest()
        trace = get_manual_trace(question_hash)
        assert trace == "Python is a high-level programming language known for its readability."

        # 5. Verify config still has reference to manual_traces
        assert manual_config.manual_traces == manual_traces

    def test_mixed_trace_formats(self):
        """Test mixing string and message list trace formats."""
        # 1. Create benchmark
        benchmark = Benchmark("mixed_formats")
        benchmark.add_question(question="String question?", raw_answer="String answer")
        benchmark.add_question(question="Messages question?", raw_answer="Messages answer")

        # 2. Create ManualTraces
        manual_traces = ManualTraces(benchmark)

        # 3. Register one trace as string
        manual_traces.register_trace("String question?", "This is a simple string trace", map_to_id=True)

        # 4. Register another as message list
        messages = [
            AIMessage(content="Processing..."),
            ToolMessage(name="tool", content="Result", tool_call_id="call_1"),
            AIMessage(content="Final answer"),
        ]
        manual_traces.register_trace("Messages question?", messages, map_to_id=True)

        # 5. Verify both traces exist and are in correct format
        hash1 = hashlib.md5(b"String question?").hexdigest()
        hash2 = hashlib.md5(b"Messages question?").hexdigest()

        trace1 = get_manual_trace(hash1)
        trace2 = get_manual_trace(hash2)

        assert trace1 == "This is a simple string trace"
        assert isinstance(trace2, str)  # Should be harmonized from messages

        # 6. Verify metrics only exist for message trace
        from karenina.infrastructure.llm.manual_traces import get_manual_trace_with_metrics

        _, metrics1 = get_manual_trace_with_metrics(hash1)
        _, metrics2 = get_manual_trace_with_metrics(hash2)

        assert metrics1 is None  # String trace has no metrics
        assert metrics2 is not None  # Message list has metrics

    def test_manual_config_validation(self):
        """Test that ModelConfig properly validates manual interface requirements."""
        benchmark = Benchmark("validation_test")
        benchmark.add_question(question="Test?", raw_answer="Test answer")

        manual_traces = ManualTraces(benchmark)

        # Should succeed with manual_traces
        config = ModelConfig(interface="manual", manual_traces=manual_traces)
        assert config.interface == "manual"

        # Should fail without manual_traces
        with pytest.raises(ValueError, match="manual_traces is required"):
            ModelConfig(interface="manual")

        # Should fail with MCP configuration
        with pytest.raises(ValueError, match="MCP tools are not supported"):
            ModelConfig(interface="manual", manual_traces=manual_traces, mcp_urls_dict={"server": "http://localhost"})
