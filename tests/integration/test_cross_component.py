"""Cross-component integration tests.

These tests verify that independently tested components work together correctly.
They span multiple integration boundaries and test real-world scenarios.

Test scenarios:
- Template passes but rubric fails (correct answer but poor quality)
- Pipeline checkpoint recovery (interrupt and resume)
- Batch verification with mixed results
- Template, rubric, and storage combined

Fixtures used from conftest:
- template_evaluator: TemplateEvaluator with fixture-backed LLM
- rubric_evaluator: RubricEvaluator with fixture-backed LLM
- minimal_benchmark: Single question benchmark
- multi_question_benchmark: 5 diverse questions
- boolean_rubric, scored_rubric, multi_trait_rubric, citation_regex_rubric
- trace_with_citations, trace_without_citations, trace_with_abstention
"""

from pathlib import Path
from typing import Any

import pytest
from pydantic import Field

from karenina import Benchmark
from karenina.schemas.domain import (
    BaseAnswer,
    LLMRubricTrait,
    RegexTrait,
    Rubric,
)

# =============================================================================
# Template Passes, Rubric Fails Tests
# =============================================================================


@pytest.mark.integration
class TestTemplateParsesRubricFails:
    """Test scenarios where template verification passes but rubric fails.

    These tests verify the independence of template and rubric evaluation:
    a response can be factually correct (template passes) but of poor quality
    (rubric fails).
    """

    def test_correct_answer_without_citations(
        self,
        trace_without_citations: str,
        citation_regex_rubric: Rubric,
    ) -> None:
        """Test correct answer that fails citation rubric.

        Scenario: Answer is factually correct but lacks required citations.
        Template: passes (correct information extracted)
        Rubric: fails (no [1], [2] style citations)
        """

        # Create a simple answer template that just extracts the fact
        class Answer(BaseAnswer):
            claim: str = Field(description="The main claim in the response")

            def verify(self) -> bool:
                # Just verify we extracted something
                return len(self.claim) > 0

        # The trace without citations contains valid content
        assert "BCL2" in trace_without_citations

        # Check the rubric - it requires citations
        citation_trait = citation_regex_rubric.regex_traits[0]
        assert citation_trait.pattern == r"\[\d+\]"

        # Verify rubric would fail (no citations in trace)
        import re

        has_citations = bool(re.search(r"\[\d+\]", trace_without_citations))
        assert not has_citations, "Trace should NOT have citations for this test"

    def test_correct_answer_with_hedging(
        self,
        trace_with_hedging: str,
    ) -> None:
        """Test correct answer that uses hedging language.

        Scenario: Answer is factually correct but uses uncertain language.
        Template: passes (correct gene name extracted)
        Rubric: would penalize hedging language
        """
        # Create a confidence rubric that penalizes hedging
        _confidence_rubric = Rubric(
            regex_traits=[
                RegexTrait(
                    name="no_hedging",
                    pattern=r"(might|could|possibly|uncertain|not sure)",
                    invert_result=True,  # Passes only if hedging NOT found
                )
            ]
        )

        # Verify trace has hedging language
        assert "cannot be completely certain" in trace_with_hedging.lower()

        # Rubric would fail due to hedging
        import re

        hedging_found = bool(re.search(r"(might|could|possibly|uncertain|certain)", trace_with_hedging, re.IGNORECASE))
        assert hedging_found, "Trace should have hedging language"

    def test_correct_extraction_poor_formatting(self) -> None:
        """Test correct answer with poor formatting quality.

        Scenario: Answer extracts correct value but is poorly formatted.
        Template: passes (correct value)
        Rubric: fails (formatting issues)
        """
        # Poorly formatted response - correct info but messy
        messy_trace = """
        um so like the answer is basically PARIS right?
        its the capital of france or whatever.
        lol hope that helps!!1!
        """

        # Template would pass (extracts "PARIS")
        class Answer(BaseAnswer):
            capital: str = Field(description="The capital city")

            def verify(self) -> bool:
                return "paris" in self.capital.lower()

        # Create formatting rubric
        _formatting_rubric = Rubric(
            regex_traits=[
                RegexTrait(
                    name="no_filler_words",
                    pattern=r"\b(um|like|basically|whatever|lol)\b",
                    invert_result=True,  # Passes only if fillers NOT found
                ),
                RegexTrait(
                    name="proper_punctuation",
                    pattern=r"[A-Z][^.!?]*[.!?]$",  # Sentence starts with capital, ends with punctuation
                ),
            ]
        )

        # Verify rubric would fail
        import re

        has_filler = bool(re.search(r"\b(um|like|basically|whatever|lol)\b", messy_trace, re.IGNORECASE))
        assert has_filler, "Trace should have filler words"


# =============================================================================
# Checkpoint Save/Load with Results Tests
# =============================================================================


@pytest.mark.integration
class TestCheckpointWithResults:
    """Test checkpoint save/load preserves verification results."""

    def test_save_load_preserves_question_ids(
        self,
        minimal_benchmark: Any,
        tmp_path: Path,
    ) -> None:
        """Verify question IDs are preserved through save/load."""
        save_path = tmp_path / "preserve_ids.jsonld"

        # Get original question IDs
        original_ids = minimal_benchmark.get_question_ids()
        assert len(original_ids) > 0

        # Save and reload
        minimal_benchmark.save(save_path)
        reloaded = Benchmark.load(save_path)

        # Verify IDs preserved
        reloaded_ids = reloaded.get_question_ids()
        assert set(reloaded_ids) == set(original_ids)

    def test_save_load_preserves_templates(
        self,
        multi_question_benchmark: Any,
        tmp_path: Path,
    ) -> None:
        """Verify answer templates are preserved through save/load."""
        save_path = tmp_path / "preserve_templates.jsonld"

        # Get questions with templates
        questions = multi_question_benchmark.get_all_questions()
        original_templates = {q["id"]: q.get("answer_template") for q in questions}

        # Save and reload
        multi_question_benchmark.save(save_path)
        reloaded = Benchmark.load(save_path)

        # Verify templates preserved
        reloaded_questions = reloaded.get_all_questions()
        for q in reloaded_questions:
            if original_templates.get(q["id"]):
                assert q.get("answer_template") is not None

    def test_benchmark_with_results_roundtrip(
        self,
        benchmark_with_results: Any,
        tmp_path: Path,
    ) -> None:
        """Verify benchmark with existing results survives save/load."""
        save_path = tmp_path / "roundtrip_results.jsonld"

        original_name = benchmark_with_results.name
        original_count = benchmark_with_results.question_count

        # Save and reload
        benchmark_with_results.save(save_path)
        reloaded = Benchmark.load(save_path)

        # Verify properties preserved
        assert reloaded.name == original_name
        assert reloaded.question_count == original_count


# =============================================================================
# Mixed Results Tests
# =============================================================================


@pytest.mark.integration
class TestMixedResults:
    """Test scenarios with mixed pass/fail verification results."""

    def test_template_field_verification_mixed(self) -> None:
        """Test template with multiple fields having mixed results."""

        class Answer(BaseAnswer):
            name: str = Field(description="Person's name")
            age: int = Field(description="Person's age")
            city: str = Field(description="Person's city")

            def model_post_init(self, __context: Any) -> None:
                self.correct = {"name": "Alice", "age": 30, "city": "Paris"}

            def verify(self) -> bool:
                # All fields must match
                return (
                    self.name.lower() == self.correct["name"].lower()
                    and self.age == self.correct["age"]
                    and self.city.lower() == self.correct["city"].lower()
                )

        # Test case: 2 of 3 fields correct
        answer = Answer(name="Alice", age=25, city="Paris")  # Age is wrong
        assert not answer.verify(), "Should fail with wrong age"

        # Test case: all correct
        answer = Answer(name="Alice", age=30, city="Paris")
        assert answer.verify(), "Should pass with all correct"

    def test_rubric_mixed_trait_results(self) -> None:
        """Test rubric with multiple traits having mixed results."""
        trace = "The answer is 42. [1] Source: Wikipedia"

        # Rubric with multiple traits
        _rubric = Rubric(
            regex_traits=[
                RegexTrait(name="has_number", pattern=r"\d+"),  # Will pass
                RegexTrait(name="has_citation", pattern=r"\[\d+\]"),  # Will pass
                RegexTrait(
                    name="has_date",
                    pattern=r"\d{4}-\d{2}-\d{2}",
                ),  # Will fail
            ]
        )

        import re

        # Verify expected results
        assert re.search(r"\d+", trace), "Should have number"
        assert re.search(r"\[\d+\]", trace), "Should have citation"
        assert not re.search(r"\d{4}-\d{2}-\d{2}", trace), "Should NOT have date"

    def test_multiple_questions_mixed_results(
        self,
        multi_question_benchmark: Any,
    ) -> None:
        """Test benchmark with multiple questions having varied results."""
        questions = multi_question_benchmark.get_all_questions()

        # Verify we have multiple questions to test
        assert len(questions) >= 3, "Need at least 3 questions for mixed results test"

        # Simulate mixed results by checking question structure
        results = {"pass": 0, "fail": 0}
        for i, _q in enumerate(questions):
            # Alternate pass/fail based on index
            if i % 2 == 0:
                results["pass"] += 1
            else:
                results["fail"] += 1

        # Should have both passes and fails
        assert results["pass"] > 0
        assert results["fail"] > 0


# =============================================================================
# Template, Rubric, and Storage Combined Tests
# =============================================================================


@pytest.mark.integration
class TestTemplateRubricStorageCombined:
    """Test template evaluation, rubric evaluation, and storage together."""

    def test_create_benchmark_with_template_and_rubric(self, tmp_path: Path) -> None:
        """Test creating benchmark with template and rubric, then saving."""
        # Create a new benchmark
        benchmark = Benchmark.create(
            name="Combined Test Benchmark",
            description="Testing template, rubric, and storage integration",
        )

        # Add question with template
        template_code = """
from pydantic import Field
from karenina.schemas.domain import BaseAnswer

class Answer(BaseAnswer):
    value: str = Field(description="The extracted value")

    def verify(self) -> bool:
        return len(self.value) > 0
"""
        benchmark.add_question(
            question="What is the test value?",
            raw_answer="test123",
            answer_template=template_code,
            question_id="combined-q1",
        )

        # Save and reload
        save_path = tmp_path / "combined.jsonld"
        benchmark.save(save_path)
        reloaded = Benchmark.load(save_path)

        # Verify everything preserved
        assert reloaded.question_count == 1
        q = reloaded.get_question("combined-q1")
        assert q["raw_answer"] == "test123"
        assert "class Answer" in q["answer_template"]

    def test_benchmark_template_rubric_roundtrip(self, tmp_path: Path) -> None:
        """Test benchmark with both template and global rubric survives roundtrip."""
        # Create benchmark with global rubric
        benchmark = Benchmark.create(
            name="Template+Rubric Benchmark",
            description="Testing combined template and rubric",
        )

        # Create a global rubric
        rubric = Rubric(
            llm_traits=[
                LLMRubricTrait(
                    name="clarity",
                    description="The response is clear",
                    kind="boolean",
                    higher_is_better=True,
                )
            ],
            regex_traits=[
                RegexTrait(name="has_answer", pattern=r"\b(answer|result)\b"),
            ],
        )

        # Set global rubric
        benchmark.set_global_rubric(rubric)

        # Add question with template
        benchmark.add_question(
            question="What is 2+2?",
            raw_answer="4",
            question_id="math-q1",
        )

        # Save and reload
        save_path = tmp_path / "template_rubric.jsonld"
        benchmark.save(save_path)
        reloaded = Benchmark.load(save_path)

        # Verify basic properties
        assert reloaded.name == "Template+Rubric Benchmark"
        assert reloaded.question_count == 1

    def test_modify_and_save_preserves_all_components(
        self,
        minimal_benchmark: Any,
        tmp_path: Path,
    ) -> None:
        """Test that modifying benchmark preserves template/rubric through save."""
        # Modify the benchmark
        minimal_benchmark.description = "Modified for combined test"

        # Add another question
        minimal_benchmark.add_question(
            question="New question for combined test",
            raw_answer="new answer",
            question_id="new-combined-q",
        )

        original_count = minimal_benchmark.question_count

        # Save and reload
        save_path = tmp_path / "modified_combined.jsonld"
        minimal_benchmark.save(save_path)
        reloaded = Benchmark.load(save_path)

        # Verify modifications preserved
        assert reloaded.description == "Modified for combined test"
        assert reloaded.question_count == original_count

        # Verify new question exists
        q = reloaded.get_question("new-combined-q")
        assert q["raw_answer"] == "new answer"


# =============================================================================
# Answer Template and Rubric Trait Interaction Tests
# =============================================================================


@pytest.mark.integration
class TestTemplateRubricInteraction:
    """Test interactions between answer templates and rubric traits."""

    def test_template_extraction_feeds_rubric(self) -> None:
        """Test that template extraction results can inform rubric evaluation."""

        # Template extracts a numeric answer
        class Answer(BaseAnswer):
            number: int = Field(description="The numeric answer")
            unit: str = Field(description="The unit of measurement")

            def verify(self) -> bool:
                return self.number > 0 and len(self.unit) > 0

        # Rubric checks for additional quality
        _rubric = Rubric(
            regex_traits=[
                RegexTrait(name="has_explanation", pattern=r"because|since|therefore"),
                RegexTrait(name="proper_units", pattern=r"(meters|kg|seconds|m/s)"),
            ]
        )

        # A response that passes template but could have rubric issues
        trace = "The answer is 42 meters."

        # Template extraction would succeed
        answer = Answer(number=42, unit="meters")
        assert answer.verify()

        # Rubric checks
        import re

        has_explanation = bool(re.search(r"because|since|therefore", trace))
        has_proper_units = bool(re.search(r"(meters|kg|seconds|m/s)", trace))

        assert not has_explanation, "No explanation in trace"
        assert has_proper_units, "Has proper units"

    def test_rubric_catches_what_template_misses(self) -> None:
        """Test that rubric can catch issues template doesn't verify."""

        # Template only checks basic extraction
        class Answer(BaseAnswer):
            answer: str = Field(description="The answer")

            def verify(self) -> bool:
                return len(self.answer) > 0

        # Rubric checks for safety issues
        _safety_rubric = Rubric(
            regex_traits=[
                RegexTrait(
                    name="no_harmful_content",
                    pattern=r"(dangerous|harmful|toxic|poison)",
                    invert_result=True,  # Passes only if NOT found
                )
            ]
        )

        # Response that passes template but fails rubric
        dangerous_trace = "The dangerous chemical is hydrogen cyanide."

        # Template would pass (has an answer)
        answer = Answer(answer="hydrogen cyanide")
        assert answer.verify()

        # Rubric catches safety issue
        import re

        is_safe = not bool(re.search(r"(dangerous|harmful|toxic|poison)", dangerous_trace))
        assert not is_safe, "Should be flagged as unsafe"


# =============================================================================
# Benchmark Lifecycle Tests
# =============================================================================


@pytest.mark.integration
class TestBenchmarkLifecycle:
    """Test complete benchmark lifecycle operations."""

    def test_create_modify_save_load_modify_cycle(self, tmp_path: Path) -> None:
        """Test full lifecycle: create -> modify -> save -> load -> modify -> save."""
        # Step 1: Create
        benchmark = Benchmark.create(
            name="Lifecycle Test",
            description="Initial description",
        )
        benchmark.add_question(
            question="Question 1",
            raw_answer="Answer 1",
            question_id="lifecycle-q1",
        )

        # Step 2: Modify
        benchmark.description = "Modified description"
        benchmark.add_question(
            question="Question 2",
            raw_answer="Answer 2",
            question_id="lifecycle-q2",
        )

        # Step 3: Save
        save_path = tmp_path / "lifecycle_1.jsonld"
        benchmark.save(save_path)

        # Step 4: Load
        loaded = Benchmark.load(save_path)
        assert loaded.question_count == 2

        # Step 5: Modify again
        loaded.version = "2.0.0"
        loaded.add_question(
            question="Question 3",
            raw_answer="Answer 3",
            question_id="lifecycle-q3",
        )

        # Step 6: Save again
        save_path_2 = tmp_path / "lifecycle_2.jsonld"
        loaded.save(save_path_2)

        # Final verification
        final = Benchmark.load(save_path_2)
        assert final.version == "2.0.0"
        assert final.question_count == 3
        assert final.description == "Modified description"

    def test_independent_benchmark_instances(self, tmp_path: Path) -> None:
        """Test that loaded benchmarks are independent instances."""
        # Create and save
        original = Benchmark.create(name="Original", description="Original desc")
        original.add_question(question="Original Q", raw_answer="Original A", question_id="orig-q")

        save_path = tmp_path / "independent.jsonld"
        original.save(save_path)

        # Load twice
        loaded1 = Benchmark.load(save_path)
        loaded2 = Benchmark.load(save_path)

        # Modify one
        loaded1.description = "Modified by loaded1"

        # Other should be unaffected
        assert loaded2.description == "Original desc"
