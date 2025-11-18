"""Tests for TaskEval integration with MetricRubricTrait.

This test file validates that the refactored TaskEval properly supports
MetricRubricTrait evaluation, including confusion matrices and computed metrics
(precision, recall, F1, accuracy, specificity).
"""

import os

import pytest

from karenina.benchmark.task_eval import TaskEval
from karenina.schemas.domain import MetricRubricTrait, Rubric
from karenina.schemas.workflow import ModelConfig, VerificationConfig


@pytest.mark.skipif(not os.getenv("OPENAI_API_KEY"), reason="OpenAI API key required for metric trait tests")
class TestTaskEvalMetricTraits:
    """Test TaskEval with MetricRubricTrait evaluation."""

    def test_metric_trait_basic(self) -> None:
        """Test basic MetricRubricTrait evaluation with TaskEval."""
        task = TaskEval(task_id="metric_trait_test")

        # Create answer template that extracts lists
        answer_template_code = '''
from karenina.schemas.domain import BaseAnswer
from pydantic import Field

class Answer(BaseAnswer):
    """Answer that extracts entity lists."""

    mentioned_entities: list[str] = Field(description="List of entities mentioned")

    def model_post_init(self, __context):
        # Ground truth: the correct entities
        self.correct = {"mentioned_entities": ["Alice", "Bob", "Charlie"]}

    def verify(self) -> bool:
        # Verification checks if all ground truth entities are found
        correct_set = set(self.correct["mentioned_entities"])
        mentioned_set = set(self.mentioned_entities)
        return correct_set.issubset(mentioned_set)
'''

        question = {
            "id": "entity_q1",
            "question": "Who are mentioned in the text?",
            "raw_answer": "Alice, Bob, Charlie",
            "answer_template": answer_template_code,
        }
        task.add_question(question)

        # Add MetricRubricTrait for entity extraction
        metric_rubric = Rubric(
            metric_traits=[
                MetricRubricTrait(
                    name="entity_extraction",
                    description="Entity extraction quality",
                    metrics=["precision", "recall", "f1"],
                    tp_instructions=["Alice", "Bob", "Charlie"],
                )
            ]
        )
        task.add_rubric(metric_rubric)

        # Log agent output with some entities
        # This will have: TP=2 (Alice, Bob), FP=1 (Dave), FN=1 (Charlie)
        task.log("The text mentions Alice and Bob, as well as Dave.")

        config = VerificationConfig(
            parsing_models=[
                ModelConfig(
                    id="parser",
                    model_provider="openai",
                    model_name="gpt-4.1-mini",
                    system_prompt="Parse and extract entities",
                )
            ],
            parsing_only=True,
        )

        # Evaluate
        result = task.evaluate(config)

        # Verify results
        assert result.global_eval is not None
        verification_results = result.global_eval.verification_results
        assert "entity_q1" in verification_results

        vr = verification_results["entity_q1"][0]

        # Check that metric trait evaluation happened
        assert vr.metric_trait_confusion_lists is not None
        assert "entity_extraction" in vr.metric_trait_confusion_lists

        confusion = vr.metric_trait_confusion_lists["entity_extraction"]

        # Verify confusion matrix structure
        assert "tp" in confusion
        assert "fp" in confusion
        assert "fn" in confusion
        assert "tn" in confusion

        # Check metrics were computed
        assert vr.metric_trait_metrics is not None
        assert "entity_extraction" in vr.metric_trait_metrics

        metrics = vr.metric_trait_metrics["entity_extraction"]
        assert "precision" in metrics
        assert "recall" in metrics
        assert "f1" in metrics

        # Metrics should be between 0 and 1
        for metric_name, value in metrics.items():
            assert 0 <= value <= 1, f"{metric_name} should be between 0 and 1"

        # Note: accuracy and specificity require full_matrix mode
        # In tp_only mode, only precision, recall, f1 are available

    def test_metric_trait_perfect_match(self) -> None:
        """Test MetricRubricTrait with perfect entity extraction (all TP)."""
        task = TaskEval(task_id="perfect_metric_test")

        answer_template_code = '''
from karenina.schemas.domain import BaseAnswer
from pydantic import Field

class Answer(BaseAnswer):
    """Answer that extracts entity lists."""

    extracted_names: list[str] = Field(description="List of names")

    def model_post_init(self, __context):
        self.correct = {"extracted_names": ["Alice", "Bob"]}

    def verify(self) -> bool:
        return set(self.correct["extracted_names"]) == set(self.extracted_names)
'''

        question = {
            "id": "perfect_q",
            "question": "Extract names",
            "raw_answer": "Alice, Bob",
            "answer_template": answer_template_code,
        }
        task.add_question(question)

        metric_rubric = Rubric(
            metric_traits=[
                MetricRubricTrait(
                    name="name_extraction",
                    description="Name extraction",
                    metrics=["precision", "recall", "f1"],
                    tp_instructions=["Alice", "Bob"],
                )
            ]
        )
        task.add_rubric(metric_rubric)

        # Perfect match - log exactly the correct entities
        task.log("extracted_names: [Alice, Bob]")

        config = VerificationConfig(
            parsing_models=[
                ModelConfig(
                    id="parser",
                    model_provider="openai",
                    model_name="gpt-4.1-mini",
                    system_prompt="Parse names",
                )
            ],
            parsing_only=True,
        )

        result = task.evaluate(config)
        vr = result.global_eval.verification_results["perfect_q"][0]

        # Perfect match should have:
        # - All TPs, no FPs or FNs
        # - Precision = Recall = F1 = 1.0
        confusion = vr.metric_trait_confusion_lists["name_extraction"]
        assert len(confusion["tp"]) == 2  # Alice and Bob
        assert len(confusion["fp"]) == 0  # No false positives
        assert len(confusion["fn"]) == 0  # No false negatives

        metrics = vr.metric_trait_metrics["name_extraction"]
        assert metrics["precision"] == 1.0
        assert metrics["recall"] == 1.0
        assert metrics["f1"] == 1.0

    def test_metric_trait_with_other_rubric_types(self) -> None:
        """Test that MetricRubricTrait works alongside RubricTrait and ManualRubricTrait."""
        task = TaskEval(task_id="combined_rubric_test")

        answer_template_code = '''
from karenina.schemas.domain import BaseAnswer
from pydantic import Field

class Answer(BaseAnswer):
    """Combined answer with entities and verification."""

    entities: list[str] = Field(description="Extracted entities")

    def model_post_init(self, __context):
        self.correct = {"entities": ["Alice", "Bob"]}

    def verify(self) -> bool:
        return set(self.correct["entities"]).issubset(set(self.entities))
'''

        question = {
            "id": "combined_q",
            "question": "Extract entities and evaluate",
            "raw_answer": "Alice, Bob",
            "answer_template": answer_template_code,
        }
        task.add_question(question)

        # Combine all three rubric trait types
        from karenina.schemas.domain import LLMRubricTrait, RegexTrait

        combined_rubric = Rubric(
            llm_traits=[
                LLMRubricTrait(name="completeness", description="Is answer complete?", kind="boolean"),
            ],
            regex_traits=[
                RegexTrait(
                    name="has_alice",
                    description="Contains Alice",
                    pattern=r"Alice",
                )
            ],
            metric_traits=[
                MetricRubricTrait(
                    name="entity_metrics",
                    description="Entity extraction metrics",
                    metrics=["precision", "recall", "f1"],
                    tp_instructions=["Alice", "Bob"],
                )
            ],
        )
        task.add_rubric(combined_rubric)

        task.log("The answer mentions Alice and Bob.")

        config = VerificationConfig(
            parsing_models=[
                ModelConfig(
                    id="parser",
                    model_provider="openai",
                    model_name="gpt-4.1-mini",
                    system_prompt="Parse and evaluate",
                )
            ],
            parsing_only=True,
        )

        result = task.evaluate(config)
        vr = result.global_eval.verification_results["combined_q"][0]

        # All three types should be evaluated
        assert vr.verify_rubric is not None  # LLM and manual traits
        assert "completeness" in vr.verify_rubric  # RubricTrait
        assert "has_alice" in vr.verify_rubric  # ManualRubricTrait

        assert vr.metric_trait_metrics is not None  # MetricRubricTrait
        assert "entity_metrics" in vr.metric_trait_metrics

    def test_metric_trait_display_formatting(self) -> None:
        """Test that metric trait results are properly formatted in display output."""
        task = TaskEval(task_id="display_test")

        answer_template_code = """
from karenina.schemas.domain import BaseAnswer
from pydantic import Field

class Answer(BaseAnswer):
    items: list[str] = Field(description="Items")

    def model_post_init(self, __context):
        self.correct = {"items": ["A", "B", "C"]}

    def verify(self) -> bool:
        return True
"""

        question = {
            "id": "display_q",
            "question": "Extract items",
            "raw_answer": "A, B, C",
            "answer_template": answer_template_code,
        }
        task.add_question(question)

        metric_rubric = Rubric(
            metric_traits=[
                MetricRubricTrait(
                    name="item_extraction",
                    description="Item extraction quality",
                    metrics=["precision", "recall", "f1"],
                    tp_instructions=["A", "B", "C"],
                )
            ]
        )
        task.add_rubric(metric_rubric)

        task.log("items: [A, B]")

        config = VerificationConfig(
            parsing_models=[
                ModelConfig(
                    id="parser",
                    model_provider="openai",
                    model_name="gpt-4.1-mini",
                    system_prompt="Parse items",
                )
            ],
            parsing_only=True,
        )

        result = task.evaluate(config)

        # Test display output includes metric trait results
        display_output = result.display()

        # Should contain confusion matrix counts
        assert "TP=" in display_output or "tp=" in display_output.lower()

        # Should contain metrics
        assert "precision" in display_output.lower()
        assert "recall" in display_output.lower()
        assert "f1" in display_output.lower()

        # Should show metric trait name
        assert "item_extraction" in display_output

    def test_metric_trait_summary_stats(self) -> None:
        """Test that metric traits are included in summary statistics."""
        task = TaskEval(task_id="stats_test")

        answer_template_code = """
from karenina.schemas.domain import BaseAnswer
from pydantic import Field

class Answer(BaseAnswer):
    tags: list[str] = Field(description="Tags")

    def model_post_init(self, __context):
        self.correct = {"tags": ["tag1", "tag2"]}

    def verify(self) -> bool:
        return True
"""

        question = {
            "id": "stats_q",
            "question": "Extract tags",
            "raw_answer": "tag1, tag2",
            "answer_template": answer_template_code,
        }
        task.add_question(question)

        metric_rubric = Rubric(
            metric_traits=[
                MetricRubricTrait(
                    name="tag_extraction",
                    description="Tag extraction",
                    metrics=["precision", "recall", "f1"],
                    tp_instructions=["tag1", "tag2"],
                )
            ]
        )
        task.add_rubric(metric_rubric)

        task.log("tags: [tag1]")

        config = VerificationConfig(
            parsing_models=[
                ModelConfig(
                    id="parser",
                    model_provider="openai",
                    model_name="gpt-4.1-mini",
                    system_prompt="Parse tags",
                )
            ],
            parsing_only=True,
        )

        result = task.evaluate(config)

        # Get summary stats
        stats = result.global_eval.get_summary_stats()

        # Metric traits should be counted in rubric totals
        assert stats["rubric_traits_total"] >= 1  # At least the metric trait
        assert stats["rubric_traits_passed"] >= 1  # Metric traits count as "passed"

    def test_multiple_metric_traits(self) -> None:
        """Test evaluation with multiple metric traits."""
        task = TaskEval(task_id="multi_metric_test")

        answer_template_code = """
from karenina.schemas.domain import BaseAnswer
from pydantic import Field

class Answer(BaseAnswer):
    names: list[str] = Field(description="Person names")
    places: list[str] = Field(description="Place names")

    def model_post_init(self, __context):
        self.correct = {
            "names": ["Alice", "Bob"],
            "places": ["London", "Paris"]
        }

    def verify(self) -> bool:
        return True
"""

        question = {
            "id": "multi_q",
            "question": "Extract names and places",
            "raw_answer": "Names: Alice, Bob; Places: London, Paris",
            "answer_template": answer_template_code,
        }
        task.add_question(question)

        metric_rubric = Rubric(
            metric_traits=[
                MetricRubricTrait(
                    name="name_extraction",
                    description="Name extraction",
                    metrics=["precision", "recall", "f1"],
                    tp_instructions=["Alice", "Bob"],
                ),
                MetricRubricTrait(
                    name="place_extraction",
                    description="Place extraction",
                    metrics=["precision", "recall", "f1"],
                    tp_instructions=["London", "Paris"],
                ),
            ]
        )
        task.add_rubric(metric_rubric)

        task.log("Names: Alice, Bob; Places: London")

        config = VerificationConfig(
            parsing_models=[
                ModelConfig(
                    id="parser",
                    model_provider="openai",
                    model_name="gpt-4.1-mini",
                    system_prompt="Parse names and places",
                )
            ],
            parsing_only=True,
        )

        result = task.evaluate(config)
        vr = result.global_eval.verification_results["multi_q"][0]

        # Both metric traits should be evaluated
        assert vr.metric_trait_metrics is not None
        assert "name_extraction" in vr.metric_trait_metrics
        assert "place_extraction" in vr.metric_trait_metrics

        # Both should have confusion matrices
        assert vr.metric_trait_confusion_lists is not None
        assert "name_extraction" in vr.metric_trait_confusion_lists
        assert "place_extraction" in vr.metric_trait_confusion_lists
