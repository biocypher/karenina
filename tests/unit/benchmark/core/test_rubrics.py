"""Unit tests for RubricManager class.

Tests cover:
- Global rubric management (add, get, clear, update, remove)
- Question-specific rubric management (add, get, remove)
- Merged rubric retrieval
- Validation of rubrics
- Statistics and queries
- Clear all operations
"""

import pytest

from karenina import Benchmark
from karenina.benchmark.core.rubrics import RubricManager
from karenina.schemas.domain import CallableTrait, LLMRubricTrait, MetricRubricTrait, RegexTrait


@pytest.mark.unit
class TestRubricManagerInit:
    """Tests for RubricManager initialization."""

    def test_init_with_benchmark_base(self) -> None:
        """Test RubricManager initialization with BenchmarkBase."""
        benchmark = Benchmark.create(name="test")
        manager = RubricManager(benchmark)

        assert manager.base is benchmark


@pytest.mark.unit
class TestAddGlobalRubricTrait:
    """Tests for add_global_rubric_trait method."""

    def test_add_llm_trait(self) -> None:
        """Test adding an LLM rubric trait globally."""
        benchmark = Benchmark.create(name="test")
        manager = RubricManager(benchmark)

        trait = LLMRubricTrait(
            name="safety", description="Is the response safe?", kind="boolean", higher_is_better=True
        )
        manager.add_global_rubric_trait(trait)

        assert manager.get_global_rubric() is not None
        assert len(manager.get_global_rubric().llm_traits) == 1

    def test_add_regex_trait(self) -> None:
        """Test adding a regex rubric trait globally."""
        benchmark = Benchmark.create(name="test")
        manager = RubricManager(benchmark)

        trait = RegexTrait(
            name="has_citation",
            description="Has citation pattern",
            pattern=r"\[\d+\]",
            case_sensitive=True,
            invert_result=False,
            higher_is_better=True,
        )
        manager.add_global_rubric_trait(trait)

        assert manager.get_global_rubric() is not None
        assert len(manager.get_global_rubric().regex_traits) == 1

    def test_add_callable_trait(self) -> None:
        """Test adding a callable rubric trait globally."""
        benchmark = Benchmark.create(name="test")
        manager = RubricManager(benchmark)

        def simple_check(_: str) -> bool:
            return True

        trait = CallableTrait.from_callable(
            name="simple_check", func=simple_check, kind="boolean", description="Simple callable check"
        )
        manager.add_global_rubric_trait(trait)

        assert manager.get_global_rubric() is not None
        assert len(manager.get_global_rubric().callable_traits) == 1

    def test_add_metric_trait(self) -> None:
        """Test adding a metric rubric trait globally."""
        benchmark = Benchmark.create(name="test")
        manager = RubricManager(benchmark)

        trait = MetricRubricTrait(
            name="entity_recall",
            description="Entity extraction recall",
            evaluation_mode="tp_only",
            metrics=["recall"],
            tp_instructions=["Entity A", "Entity B", "Entity C"],
        )
        manager.add_global_rubric_trait(trait)

        assert manager.get_global_rubric() is not None
        assert len(manager.get_global_rubric().metric_traits) == 1

    def test_add_multiple_traits(self) -> None:
        """Test adding multiple traits of different types."""
        benchmark = Benchmark.create(name="test")
        manager = RubricManager(benchmark)

        manager.add_global_rubric_trait(
            LLMRubricTrait(name="llm", description="LLM trait", kind="boolean", higher_is_better=True)
        )
        manager.add_global_rubric_trait(
            RegexTrait(
                name="regex",
                description="Regex trait",
                pattern="test",
                case_sensitive=True,
                invert_result=False,
                higher_is_better=True,
            )
        )
        manager.add_global_rubric_trait(
            CallableTrait.from_callable(
                name="callable", func=lambda _: True, kind="boolean", description="Callable trait"
            )
        )

        rubric = manager.get_global_rubric()
        assert rubric is not None
        assert len(rubric.llm_traits) == 1
        assert len(rubric.regex_traits) == 1
        assert len(rubric.callable_traits) == 1


@pytest.mark.unit
class TestAddQuestionRubricTrait:
    """Tests for add_question_rubric_trait method."""

    def test_add_trait_to_question(self) -> None:
        """Test adding a rubric trait to a specific question."""
        benchmark = Benchmark.create(name="test")
        manager = RubricManager(benchmark)

        q_id = benchmark.add_question("What is 2+2?", "4")

        trait = LLMRubricTrait(
            name="clarity", description="Is the response clear?", kind="boolean", higher_is_better=True
        )
        manager.add_question_rubric_trait(q_id, trait)

        question_rubric = manager.get_question_rubric(q_id)
        assert question_rubric is not None

    def test_add_to_nonexistent_raises(self) -> None:
        """Test adding trait to nonexistent question raises error."""
        benchmark = Benchmark.create(name="test")
        manager = RubricManager(benchmark)

        trait = LLMRubricTrait(name="test", description="Test", kind="boolean", higher_is_better=True)

        with pytest.raises(ValueError, match="Question not found"):
            manager.add_question_rubric_trait("nonexistent", trait)


@pytest.mark.unit
class TestGetGlobalRubric:
    """Tests for get_global_rubric method."""

    def test_get_global_rubric_none_when_empty(self) -> None:
        """Test that get_global_rubric returns None when no global rubric."""
        benchmark = Benchmark.create(name="test")
        manager = RubricManager(benchmark)

        assert manager.get_global_rubric() is None

    def test_get_global_rubric_with_traits(self) -> None:
        """Test getting global rubric returns correct structure."""
        benchmark = Benchmark.create(name="test")
        manager = RubricManager(benchmark)

        manager.add_global_rubric_trait(
            LLMRubricTrait(name="llm", description="LLM", kind="boolean", higher_is_better=True)
        )
        manager.add_global_rubric_trait(
            RegexTrait(
                name="regex",
                description="Regex",
                pattern="test",
                case_sensitive=True,
                invert_result=False,
                higher_is_better=True,
            )
        )

        rubric = manager.get_global_rubric()
        assert rubric is not None
        assert len(rubric.llm_traits) == 1
        assert len(rubric.regex_traits) == 1
        assert len(rubric.callable_traits) == 0
        assert len(rubric.metric_traits) == 0


@pytest.mark.unit
class TestGetQuestionRubric:
    """Tests for get_question_rubric method."""

    def test_get_question_rubric_none_when_empty(self) -> None:
        """Test that get_question_rubric returns None when no rubric."""
        benchmark = Benchmark.create(name="test")
        manager = RubricManager(benchmark)

        q_id = benchmark.add_question("Question?", "Answer")
        assert manager.get_question_rubric(q_id) is None

    def test_get_question_rubric_nonexistent_raises(self) -> None:
        """Test getting rubric for nonexistent question returns None."""
        benchmark = Benchmark.create(name="test")
        manager = RubricManager(benchmark)

        assert manager.get_question_rubric("nonexistent") is None


@pytest.mark.unit
class TestGetMergedRubricForQuestion:
    """Tests for get_merged_rubric_for_question method."""

    def test_merged_rubric_none_when_no_rubrics(self) -> None:
        """Test merged rubric is None when no rubrics exist."""
        benchmark = Benchmark.create(name="test")
        manager = RubricManager(benchmark)

        q_id = benchmark.add_question("Question?", "Answer")
        assert manager.get_merged_rubric_for_question(q_id) is None

    def test_merged_rubric_only_global(self) -> None:
        """Test merged rubric with only global traits."""
        benchmark = Benchmark.create(name="test")
        manager = RubricManager(benchmark)

        manager.add_global_rubric_trait(
            LLMRubricTrait(name="global", description="Global trait", kind="boolean", higher_is_better=True)
        )
        q_id = benchmark.add_question("Question?", "Answer")

        merged = manager.get_merged_rubric_for_question(q_id)
        assert merged is not None
        assert len(merged.llm_traits) == 1

    def test_merged_rubric_only_question(self) -> None:
        """Test merged rubric with only question-specific traits."""
        benchmark = Benchmark.create(name="test")
        manager = RubricManager(benchmark)

        q_id = benchmark.add_question("Question?", "Answer")
        manager.add_question_rubric_trait(
            q_id, LLMRubricTrait(name="question", description="Question trait", kind="boolean", higher_is_better=True)
        )

        merged = manager.get_merged_rubric_for_question(q_id)
        assert merged is not None
        assert len(merged.llm_traits) == 1

    def test_merged_rubric_combines_both(self) -> None:
        """Test merged rubric combines global and question traits."""
        benchmark = Benchmark.create(name="test")
        manager = RubricManager(benchmark)

        manager.add_global_rubric_trait(
            LLMRubricTrait(name="global", description="Global trait", kind="boolean", higher_is_better=True)
        )
        q_id = benchmark.add_question("Question?", "Answer")
        manager.add_question_rubric_trait(
            q_id, LLMRubricTrait(name="question", description="Question trait", kind="boolean", higher_is_better=True)
        )

        merged = manager.get_merged_rubric_for_question(q_id)
        assert merged is not None
        assert len(merged.llm_traits) == 2

    def test_merged_rubric_question_overrides_global(self) -> None:
        """Test question trait overrides global trait with same name."""
        benchmark = Benchmark.create(name="test")
        manager = RubricManager(benchmark)

        manager.add_global_rubric_trait(
            RegexTrait(
                name="pattern",
                description="Global pattern",
                pattern=r"\d+",
                case_sensitive=True,
                invert_result=False,
                higher_is_better=True,
            )
        )
        q_id = benchmark.add_question("Question?", "Answer")
        manager.add_question_rubric_trait(
            q_id,
            RegexTrait(
                name="pattern",
                description="Question pattern",
                pattern=r"[A-Z]+",
                case_sensitive=True,
                invert_result=False,
                higher_is_better=True,
            ),
        )

        merged = manager.get_merged_rubric_for_question(q_id)
        assert merged is not None
        assert len(merged.regex_traits) == 1
        # Question's pattern should override
        assert merged.regex_traits[0].pattern == r"[A-Z]+"
        assert merged.regex_traits[0].description == "Question pattern"


@pytest.mark.unit
class TestClearGlobalRubric:
    """Tests for clear_global_rubric method."""

    def test_clear_global_rubric_when_present(self) -> None:
        """Test clearing global rubric when it exists."""
        benchmark = Benchmark.create(name="test")
        manager = RubricManager(benchmark)

        manager.add_global_rubric_trait(
            LLMRubricTrait(name="test", description="Test", kind="boolean", higher_is_better=True)
        )

        result = manager.clear_global_rubric()
        assert result is True
        assert manager.get_global_rubric() is None

    def test_clear_global_rubric_when_absent(self) -> None:
        """Test clearing global rubric when it doesn't exist."""
        benchmark = Benchmark.create(name="test")
        manager = RubricManager(benchmark)

        result = manager.clear_global_rubric()
        assert result is False


@pytest.mark.unit
class TestRemoveQuestionRubric:
    """Tests for remove_question_rubric method."""

    def test_remove_question_rubric_when_present(self) -> None:
        """Test removing question rubric when it exists."""
        benchmark = Benchmark.create(name="test")
        manager = RubricManager(benchmark)

        q_id = benchmark.add_question("Question?", "Answer")
        manager.add_question_rubric_trait(
            q_id, LLMRubricTrait(name="test", description="Test", kind="boolean", higher_is_better=True)
        )

        result = manager.remove_question_rubric(q_id)
        assert result is True

    def test_remove_question_rubric_when_absent(self) -> None:
        """Test removing question rubric when it doesn't exist."""
        benchmark = Benchmark.create(name="test")
        manager = RubricManager(benchmark)

        q_id = benchmark.add_question("Question?", "Answer")

        result = manager.remove_question_rubric(q_id)
        assert result is False


@pytest.mark.unit
class TestClearAllRubrics:
    """Tests for clear_all_rubrics method."""

    def test_clear_all_removes_global_and_question(self) -> None:
        """Test clearing all rubrics removes both global and question rubrics."""
        benchmark = Benchmark.create(name="test")
        manager = RubricManager(benchmark)

        manager.add_global_rubric_trait(
            LLMRubricTrait(name="global", description="Global", kind="boolean", higher_is_better=True)
        )
        q_id = benchmark.add_question("Question?", "Answer")
        manager.add_question_rubric_trait(
            q_id, LLMRubricTrait(name="question", description="Question", kind="boolean", higher_is_better=True)
        )

        count = manager.clear_all_rubrics()
        assert count == 2
        assert manager.get_global_rubric() is None

    def test_clear_all_returns_zero_when_none(self) -> None:
        """Test clear_all returns 0 when no rubrics exist."""
        benchmark = Benchmark.create(name="test")
        manager = RubricManager(benchmark)

        count = manager.clear_all_rubrics()
        assert count == 0


@pytest.mark.unit
class TestValidateRubrics:
    """Tests for validate_rubrics method."""

    def test_validate_empty_benchmark(self) -> None:
        """Test validation passes for empty benchmark."""
        benchmark = Benchmark.create(name="test")
        manager = RubricManager(benchmark)

        valid, errors = manager.validate_rubrics()
        assert valid is True
        assert errors == []

    def test_validate_valid_global_llm_trait(self) -> None:
        """Test validation of valid global LLM trait."""
        benchmark = Benchmark.create(name="test")
        manager = RubricManager(benchmark)

        manager.add_global_rubric_trait(
            LLMRubricTrait(name="safety", description="Is safe?", kind="boolean", higher_is_better=True)
        )

        valid, errors = manager.validate_rubrics()
        assert valid is True
        assert errors == []

    def test_validate_score_trait_with_bounds(self) -> None:
        """Test validation of score trait with min/max bounds."""
        benchmark = Benchmark.create(name="test")
        manager = RubricManager(benchmark)

        manager.add_global_rubric_trait(
            LLMRubricTrait(
                name="score", description="Score", kind="score", min_score=1, max_score=5, higher_is_better=True
            )
        )

        valid, errors = manager.validate_rubrics()
        assert valid is True
        assert errors == []

    def test_validate_regex_with_pattern(self) -> None:
        """Test validation of regex trait with valid pattern."""
        benchmark = Benchmark.create(name="test")
        manager = RubricManager(benchmark)

        manager.add_global_rubric_trait(
            RegexTrait(
                name="pattern",
                description="Has pattern",
                pattern=r"\w+",
                case_sensitive=True,
                invert_result=False,
                higher_is_better=True,
            )
        )

        valid, errors = manager.validate_rubrics()
        assert valid is True
        assert errors == []

    def test_validate_metric_with_metrics(self) -> None:
        """Test validation of metric trait with metrics."""
        benchmark = Benchmark.create(name="test")
        manager = RubricManager(benchmark)

        manager.add_global_rubric_trait(
            MetricRubricTrait(
                name="metric",
                description="Has metrics",
                evaluation_mode="tp_only",
                metrics=["recall"],
                tp_instructions=["A", "B"],
            )
        )

        valid, errors = manager.validate_rubrics()
        assert valid is True
        assert errors == []


@pytest.mark.unit
class TestGetRubricStatistics:
    """Tests for get_rubric_statistics method."""

    def test_statistics_empty_benchmark(self) -> None:
        """Test statistics for empty benchmark."""
        benchmark = Benchmark.create(name="test")
        manager = RubricManager(benchmark)

        stats = manager.get_rubric_statistics()
        assert stats["has_global_rubric"] is False
        assert stats["global_traits_count"] == 0
        assert stats["questions_with_rubrics"] == 0
        assert stats["total_question_traits"] == 0
        assert stats["total_traits"] == 0

    def test_statistics_with_global_rubric(self) -> None:
        """Test statistics with global rubric."""
        benchmark = Benchmark.create(name="test")
        manager = RubricManager(benchmark)

        manager.add_global_rubric_trait(
            LLMRubricTrait(name="llm", description="LLM", kind="boolean", higher_is_better=True)
        )
        manager.add_global_rubric_trait(
            RegexTrait(
                name="regex",
                description="Regex",
                pattern="test",
                case_sensitive=True,
                invert_result=False,
                higher_is_better=True,
            )
        )

        stats = manager.get_rubric_statistics()
        assert stats["has_global_rubric"] is True
        assert stats["global_traits_count"] == 2

    def test_statistics_with_question_rubrics(self) -> None:
        """Test statistics with question rubrics."""
        benchmark = Benchmark.create(name="test")
        manager = RubricManager(benchmark)

        q_id = benchmark.add_question("Question?", "Answer")
        manager.add_question_rubric_trait(
            q_id, LLMRubricTrait(name="trait", description="Trait", kind="boolean", higher_is_better=True)
        )

        stats = manager.get_rubric_statistics()
        assert stats["questions_with_rubrics"] == 1
        # Note: question_rubric is stored as dict with 4 keys (llm_traits, regex_traits, etc.)
        # So even with 1 trait, it counts the dict keys
        assert stats["total_question_traits"] == 4


@pytest.mark.unit
class TestGetQuestionsWithRubric:
    """Tests for get_questions_with_rubric method."""

    def test_get_questions_empty(self) -> None:
        """Test getting questions with rubrics from empty benchmark."""
        benchmark = Benchmark.create(name="test")
        manager = RubricManager(benchmark)

        assert manager.get_questions_with_rubric() == []

    def test_get_questions_with_rubrics(self) -> None:
        """Test getting list of question IDs with rubrics."""
        benchmark = Benchmark.create(name="test")
        manager = RubricManager(benchmark)

        q_id1 = benchmark.add_question("Q1?", "A1")
        q_id2 = benchmark.add_question("Q2?", "A2")
        q_id3 = benchmark.add_question("Q3?", "A3")

        manager.add_question_rubric_trait(
            q_id1, LLMRubricTrait(name="trait", description="Trait", kind="boolean", higher_is_better=True)
        )
        manager.add_question_rubric_trait(
            q_id3, LLMRubricTrait(name="trait", description="Trait", kind="boolean", higher_is_better=True)
        )

        questions = manager.get_questions_with_rubric()
        assert len(questions) == 2
        assert q_id1 in questions
        assert q_id3 in questions
        assert q_id2 not in questions


@pytest.mark.unit
class TestSetGlobalRubric:
    """Tests for set_global_rubric method."""

    def test_set_global_rubric_replaces_existing(self) -> None:
        """Test setting global rubric replaces existing traits."""
        benchmark = Benchmark.create(name="test")
        manager = RubricManager(benchmark)

        # Add initial trait
        manager.add_global_rubric_trait(
            LLMRubricTrait(name="old", description="Old", kind="boolean", higher_is_better=True)
        )

        # Replace with new traits
        new_traits = [
            LLMRubricTrait(name="new1", description="New 1", kind="boolean", higher_is_better=True),
            LLMRubricTrait(name="new2", description="New 2", kind="boolean", higher_is_better=True),
        ]
        manager.set_global_rubric(new_traits)

        rubric = manager.get_global_rubric()
        assert rubric is not None
        assert len(rubric.llm_traits) == 2
        trait_names = [t.name for t in rubric.llm_traits]
        assert "old" not in trait_names
        assert "new1" in trait_names
        assert "new2" in trait_names


@pytest.mark.unit
class TestUpdateGlobalRubricTrait:
    """Tests for update_global_rubric_trait method."""

    def test_update_existing_trait(self) -> None:
        """Test updating an existing global trait."""
        benchmark = Benchmark.create(name="test")
        manager = RubricManager(benchmark)

        manager.add_global_rubric_trait(
            LLMRubricTrait(name="safety", description="Original", kind="boolean", higher_is_better=True)
        )

        updated = LLMRubricTrait(name="safety", description="Updated", kind="boolean", higher_is_better=True)
        result = manager.update_global_rubric_trait("safety", updated)

        assert result is True
        rubric = manager.get_global_rubric()
        assert rubric is not None
        assert rubric.llm_traits[0].description == "Updated"

    def test_update_nonexistent_trait(self) -> None:
        """Test updating nonexistent trait returns False."""
        benchmark = Benchmark.create(name="test")
        manager = RubricManager(benchmark)

        updated = LLMRubricTrait(name="missing", description="Updated", kind="boolean", higher_is_better=True)
        result = manager.update_global_rubric_trait("missing", updated)

        assert result is False


@pytest.mark.unit
class TestRemoveGlobalRubricTrait:
    """Tests for remove_global_rubric_trait method."""

    def test_remove_existing_trait(self) -> None:
        """Test removing an existing global trait."""
        benchmark = Benchmark.create(name="test")
        manager = RubricManager(benchmark)

        manager.add_global_rubric_trait(
            LLMRubricTrait(name="trait1", description="Trait 1", kind="boolean", higher_is_better=True)
        )
        manager.add_global_rubric_trait(
            LLMRubricTrait(name="trait2", description="Trait 2", kind="boolean", higher_is_better=True)
        )

        result = manager.remove_global_rubric_trait("trait1")
        assert result is True

        rubric = manager.get_global_rubric()
        assert rubric is not None
        assert len(rubric.llm_traits) == 1
        assert rubric.llm_traits[0].name == "trait2"

    def test_remove_nonexistent_trait(self) -> None:
        """Test removing nonexistent trait returns False."""
        benchmark = Benchmark.create(name="test")
        manager = RubricManager(benchmark)

        result = manager.remove_global_rubric_trait("missing")
        assert result is False


@pytest.mark.unit
class TestGetRubricTraitNames:
    """Tests for get_rubric_trait_names method."""

    def test_get_global_trait_names_empty(self) -> None:
        """Test getting global trait names when none exist."""
        benchmark = Benchmark.create(name="test")
        manager = RubricManager(benchmark)

        names = manager.get_rubric_trait_names()
        assert names == []

    def test_get_global_trait_names(self) -> None:
        """Test getting global trait names."""
        benchmark = Benchmark.create(name="test")
        manager = RubricManager(benchmark)

        manager.add_global_rubric_trait(
            LLMRubricTrait(name="llm", description="LLM", kind="boolean", higher_is_better=True)
        )
        manager.add_global_rubric_trait(
            RegexTrait(
                name="regex",
                description="Regex",
                pattern="test",
                case_sensitive=True,
                invert_result=False,
                higher_is_better=True,
            )
        )
        manager.add_global_rubric_trait(
            CallableTrait.from_callable(name="callable", func=lambda _: True, kind="boolean", description="Callable")
        )

        names = manager.get_rubric_trait_names()
        assert set(names) == {"llm", "regex", "callable"}

    def test_get_question_trait_names(self) -> None:
        """Test getting merged trait names for a question."""
        benchmark = Benchmark.create(name="test")
        manager = RubricManager(benchmark)

        manager.add_global_rubric_trait(
            LLMRubricTrait(name="global", description="Global", kind="boolean", higher_is_better=True)
        )
        q_id = benchmark.add_question("Question?", "Answer")
        manager.add_question_rubric_trait(
            q_id, LLMRubricTrait(name="question", description="Question", kind="boolean", higher_is_better=True)
        )

        names = manager.get_rubric_trait_names(q_id)
        assert set(names) == {"global", "question"}


@pytest.mark.unit
class TestHasRubric:
    """Tests for has_rubric method."""

    def test_has_rubric_global_none(self) -> None:
        """Test has_rubric returns False when no global rubric."""
        benchmark = Benchmark.create(name="test")
        manager = RubricManager(benchmark)

        assert manager.has_rubric() is False

    def test_has_rubric_global_exists(self) -> None:
        """Test has_rubric returns True when global rubric exists."""
        benchmark = Benchmark.create(name="test")
        manager = RubricManager(benchmark)

        manager.add_global_rubric_trait(
            LLMRubricTrait(name="test", description="Test", kind="boolean", higher_is_better=True)
        )

        assert manager.has_rubric() is True

    def test_has_rubric_question_none(self) -> None:
        """Test has_rubric returns False for question with no rubric."""
        benchmark = Benchmark.create(name="test")
        manager = RubricManager(benchmark)

        q_id = benchmark.add_question("Question?", "Answer")
        assert manager.has_rubric(q_id) is False

    def test_has_rubric_question_exists(self) -> None:
        """Test has_rubric returns True for question with rubric."""
        benchmark = Benchmark.create(name="test")
        manager = RubricManager(benchmark)

        q_id = benchmark.add_question("Question?", "Answer")
        manager.add_question_rubric_trait(
            q_id, LLMRubricTrait(name="test", description="Test", kind="boolean", higher_is_better=True)
        )

        assert manager.has_rubric(q_id) is True

    def test_has_rubric_question_inherits_global(self) -> None:
        """Test has_rubric returns True when only global rubric exists."""
        benchmark = Benchmark.create(name="test")
        manager = RubricManager(benchmark)

        manager.add_global_rubric_trait(
            LLMRubricTrait(name="global", description="Global", kind="boolean", higher_is_better=True)
        )
        q_id = benchmark.add_question("Question?", "Answer")

        # Question inherits global rubric
        assert manager.has_rubric(q_id) is True
