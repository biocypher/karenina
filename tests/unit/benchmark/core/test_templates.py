"""Unit tests for TemplateManager class.

Tests cover:
- Template initialization
- Adding, updating, and removing templates
- Template validation
- Template copying and global application
"""

import pytest

from karenina import Benchmark
from karenina.benchmark.core.templates import TemplateManager

# Simple valid template for testing
VALID_TEMPLATE = '''class Answer(BaseAnswer):
    """Simple answer template."""

    value: str = Field(description="The answer value")

    def verify(self) -> bool:
        return len(self.value) > 0
'''


@pytest.mark.unit
class TestTemplateManagerInit:
    """Tests for TemplateManager initialization."""

    def test_init_with_benchmark(self) -> None:
        """Test TemplateManager initialization with Benchmark."""
        benchmark = Benchmark.create(name="test")
        manager = TemplateManager(benchmark)

        assert manager.base is benchmark


@pytest.mark.unit
class TestIsDefaultTemplate:
    """Tests for _is_default_template private method."""

    def test_default_template_detection(self) -> None:
        """Test detection of auto-generated default template."""
        benchmark = Benchmark.create(name="test")
        manager = TemplateManager(benchmark)

        question = "What is 2+2?"
        default = manager._create_default_template(question)

        assert manager._is_default_template(default, question) is True

    def test_custom_template_not_default(self) -> None:
        """Test that custom templates are not detected as default."""
        benchmark = Benchmark.create(name="test")
        manager = TemplateManager(benchmark)

        assert manager._is_default_template(VALID_TEMPLATE, "test?") is False

    def test_empty_template_not_default(self) -> None:
        """Test that empty template is not considered default."""
        benchmark = Benchmark.create(name="test")
        manager = TemplateManager(benchmark)

        assert manager._is_default_template("", "test?") is False

    def test_whitespace_variations(self) -> None:
        """Test that stripped templates match correctly."""
        benchmark = Benchmark.create(name="test")
        manager = TemplateManager(benchmark)

        question = "Test question?"
        default = manager._create_default_template(question)

        # Add whitespace
        whitespace_version = f"   {default}   "

        assert manager._is_default_template(whitespace_version, question) is True


@pytest.mark.unit
class TestCreateDefaultTemplate:
    """Tests for _create_default_template private method."""

    def test_default_template_structure(self) -> None:
        """Test that default template has expected structure."""
        benchmark = Benchmark.create(name="test")
        manager = TemplateManager(benchmark)

        question = "What is the capital of France?"
        template = manager._create_default_template(question)

        assert "class Answer(BaseAnswer):" in template
        assert "response: str = Field" in template
        assert "def verify(self) -> bool:" in template
        assert question[:50] in template

    def test_default_template_truncates_long_question(self) -> None:
        """Test that long questions are truncated in default template."""
        benchmark = Benchmark.create(name="test")
        manager = TemplateManager(benchmark)

        long_question = "This is a very long question that exceeds fifty characters and should be truncated"
        template = manager._create_default_template(long_question)

        # Should truncate at ~50 chars
        assert "..." in template
        assert long_question[:50] in template


@pytest.mark.unit
class TestAddAnswerTemplate:
    """Tests for add_answer_template method."""

    def test_add_template_to_empty_benchmark(self) -> None:
        """Test adding template raises error for empty benchmark."""
        benchmark = Benchmark.create(name="test")
        manager = TemplateManager(benchmark)

        with pytest.raises(ValueError, match="Question not found"):
            manager.add_answer_template("nonexistent", VALID_TEMPLATE)

    def test_add_invalid_template_raises_error(self) -> None:
        """Test that invalid template raises ValueError."""
        benchmark = Benchmark.create(name="test")
        manager = TemplateManager(benchmark)

        # Add a question first
        q_id = benchmark.add_question("What is 2+2?", "4")

        # Invalid template - not valid Python
        invalid_template = "class Answer(BaseAnswer):\n    def verify(self"

        with pytest.raises(ValueError, match="Invalid template"):
            manager.add_answer_template(q_id, invalid_template)

    def test_add_valid_template(self) -> None:
        """Test adding a valid template."""
        benchmark = Benchmark.create(name="test")
        manager = TemplateManager(benchmark)

        q_id = benchmark.add_question("What is 2+2?", "4")

        manager.add_answer_template(q_id, VALID_TEMPLATE)

        assert manager.has_template(q_id) is True

    def test_add_template_updates_existing(self) -> None:
        """Test that adding template to same question updates it."""
        benchmark = Benchmark.create(name="test")
        manager = TemplateManager(benchmark)

        q_id = benchmark.add_question("What is 2+2?", "4")

        manager.add_answer_template(q_id, VALID_TEMPLATE)

        new_template = VALID_TEMPLATE.replace("Simple answer", "Updated answer")
        manager.add_answer_template(q_id, new_template)

        retrieved = manager.get_template(q_id)
        assert "Updated answer" in retrieved


@pytest.mark.unit
class TestHasTemplate:
    """Tests for has_template method."""

    def test_has_template_false_for_nonexistent(self) -> None:
        """Test has_template returns False for nonexistent question."""
        benchmark = Benchmark.create(name="test")
        manager = TemplateManager(benchmark)

        assert manager.has_template("nonexistent") is False

    def test_has_template_true_after_adding(self) -> None:
        """Test has_template returns True after adding template."""
        benchmark = Benchmark.create(name="test")
        manager = TemplateManager(benchmark)

        q_id = benchmark.add_question("Test?", "Answer")
        manager.add_answer_template(q_id, VALID_TEMPLATE)

        assert manager.has_template(q_id) is True

    def test_has_template_false_for_default(self) -> None:
        """Test has_template returns False for default template."""
        benchmark = Benchmark.create(name="test")
        manager = TemplateManager(benchmark)

        # Questions get default templates automatically
        q_id = benchmark.add_question("Test?", "Answer")

        # Default template should not count as having a template
        assert manager.has_template(q_id) is False


@pytest.mark.unit
class TestGetTemplate:
    """Tests for get_template method."""

    def test_get_template_nonexistent_raises(self) -> None:
        """Test getting template for nonexistent question raises error."""
        benchmark = Benchmark.create(name="test")
        manager = TemplateManager(benchmark)

        with pytest.raises(ValueError, match="Question not found"):
            manager.get_template("nonexistent")

    def test_get_template_no_template_raises(self) -> None:
        """Test getting template when none exists raises error."""
        benchmark = Benchmark.create(name="test")
        manager = TemplateManager(benchmark)

        q_id = benchmark.add_question("Test?", "Answer")

        with pytest.raises(ValueError, match="has no template"):
            manager.get_template(q_id)

    def test_get_template_returns_code(self) -> None:
        """Test getting template returns the template code."""
        benchmark = Benchmark.create(name="test")
        manager = TemplateManager(benchmark)

        q_id = benchmark.add_question("Test?", "Answer")
        manager.add_answer_template(q_id, VALID_TEMPLATE)

        retrieved = manager.get_template(q_id)

        assert "Simple answer template" in retrieved


@pytest.mark.unit
class TestUpdateTemplate:
    """Tests for update_template method."""

    def test_update_template_calls_add(self) -> None:
        """Test that update_template calls add_answer_template."""
        benchmark = Benchmark.create(name="test")
        manager = TemplateManager(benchmark)

        q_id = benchmark.add_question("Test?", "Answer")
        manager.add_answer_template(q_id, VALID_TEMPLATE)

        updated = VALID_TEMPLATE.replace("Simple", "Updated")
        manager.update_template(q_id, updated)

        assert "Updated" in manager.get_template(q_id)


@pytest.mark.unit
class TestCopyTemplate:
    """Tests for copy_template method."""

    def test_copy_template_success(self) -> None:
        """Test copying template from one question to another."""
        benchmark = Benchmark.create(name="test")
        manager = TemplateManager(benchmark)

        q_id1 = benchmark.add_question("Question 1?", "Answer 1")
        q_id2 = benchmark.add_question("Question 2?", "Answer 2")

        manager.add_answer_template(q_id1, VALID_TEMPLATE)
        manager.copy_template(q_id1, q_id2)

        assert manager.has_template(q_id2) is True
        assert "Simple answer template" in manager.get_template(q_id2)

    def test_copy_template_source_not_found_raises(self) -> None:
        """Test copying from nonexistent source raises error."""
        benchmark = Benchmark.create(name="test")
        manager = TemplateManager(benchmark)

        q_id = benchmark.add_question("Test?", "Answer")

        with pytest.raises(ValueError, match="Question not found"):
            manager.copy_template("nonexistent", q_id)

    def test_copy_template_source_no_template_raises(self) -> None:
        """Test copying from source without template raises error."""
        benchmark = Benchmark.create(name="test")
        manager = TemplateManager(benchmark)

        q_id1 = benchmark.add_question("Question 1?", "Answer 1")
        q_id2 = benchmark.add_question("Question 2?", "Answer 2")

        with pytest.raises(ValueError, match="has no template"):
            manager.copy_template(q_id1, q_id2)


@pytest.mark.unit
class TestGetFinishedTemplates:
    """Tests for get_finished_templates method."""

    def test_get_finished_empty(self) -> None:
        """Test getting finished templates from empty benchmark."""
        benchmark = Benchmark.create(name="test")
        manager = TemplateManager(benchmark)

        templates = manager.get_finished_templates()

        assert templates == []

    def test_get_finished_no_finished_questions(self) -> None:
        """Test getting finished templates when none are finished."""
        benchmark = Benchmark.create(name="test")
        manager = TemplateManager(benchmark)

        benchmark.add_question("Test?", "Answer")

        templates = manager.get_finished_templates()

        assert templates == []

    def test_get_finished_with_template(self) -> None:
        """Test getting finished templates includes template info."""
        benchmark = Benchmark.create(name="test")
        manager = TemplateManager(benchmark)

        q_id = benchmark.add_question("What is 2+2?", "4")
        manager.add_answer_template(q_id, VALID_TEMPLATE)

        # Mark as finished by setting finished flag in cache
        benchmark._questions_cache[q_id]["finished"] = True

        templates = manager.get_finished_templates()

        assert len(templates) == 1
        assert templates[0].question_id == q_id
        assert templates[0].template_code == VALID_TEMPLATE

    def test_get_finished_filtered_by_question_ids(self) -> None:
        """Test filtering finished templates by question IDs."""
        benchmark = Benchmark.create(name="test")
        manager = TemplateManager(benchmark)

        q1 = benchmark.add_question("Question 1?", "Answer 1")
        q2 = benchmark.add_question("Question 2?", "Answer 2")
        q3 = benchmark.add_question("Question 3?", "Answer 3")
        for q_id in [q1, q2, q3]:
            manager.add_answer_template(q_id, VALID_TEMPLATE)
        # Set finished after all template additions (add_answer_template triggers _rebuild_cache)
        for q_id in [q1, q2, q3]:
            benchmark._questions_cache[q_id]["finished"] = True

        templates = manager.get_finished_templates(question_ids={q1, q3})

        assert len(templates) == 2
        returned_ids = {t.question_id for t in templates}
        assert q1 in returned_ids
        assert q3 in returned_ids
        assert q2 not in returned_ids

    def test_get_finished_filtered_by_nonexistent_ids(self) -> None:
        """Test filtering with non-existent IDs returns empty list."""
        benchmark = Benchmark.create(name="test")
        manager = TemplateManager(benchmark)

        q_id = benchmark.add_question("Question?", "Answer")
        manager.add_answer_template(q_id, VALID_TEMPLATE)
        benchmark._questions_cache[q_id]["finished"] = True

        templates = manager.get_finished_templates(question_ids={"nonexistent"})

        assert templates == []

    def test_get_finished_none_question_ids_returns_all(self) -> None:
        """Test that None question_ids returns all finished templates."""
        benchmark = Benchmark.create(name="test")
        manager = TemplateManager(benchmark)

        q1 = benchmark.add_question("Question 1?", "Answer 1")
        q2 = benchmark.add_question("Question 2?", "Answer 2")
        for q_id in [q1, q2]:
            manager.add_answer_template(q_id, VALID_TEMPLATE)
        # Set finished after all template additions
        for q_id in [q1, q2]:
            benchmark._questions_cache[q_id]["finished"] = True

        templates = manager.get_finished_templates(question_ids=None)

        assert len(templates) == 2

    def test_get_finished_truncates_long_preview(self) -> None:
        """Test that long questions are truncated in preview."""
        benchmark = Benchmark.create(name="test")
        manager = TemplateManager(benchmark)

        long_question = "This is a very long question text that exceeds one hundred characters and should be truncated in the preview field"
        q_id = benchmark.add_question(long_question, "Answer")
        manager.add_answer_template(q_id, VALID_TEMPLATE)
        benchmark._questions_cache[q_id]["finished"] = True

        templates = manager.get_finished_templates()

        assert len(templates[0].question_preview) <= 103  # 100 + "..."
        assert "..." in templates[0].question_preview


@pytest.mark.unit
class TestGetMissingTemplates:
    """Tests for get_missing_templates method."""

    def test_get_missing_empty_benchmark(self) -> None:
        """Test getting missing templates from empty benchmark."""
        benchmark = Benchmark.create(name="test")
        manager = TemplateManager(benchmark)

        missing = manager.get_missing_templates(ids_only=True)

        assert missing == []

    def test_get_missing_ids_only(self) -> None:
        """Test getting missing templates with ids_only=True."""
        benchmark = Benchmark.create(name="test")
        manager = TemplateManager(benchmark)

        q_id1 = benchmark.add_question("Question 1?", "Answer 1")
        q_id2 = benchmark.add_question("Question 2?", "Answer 2")

        manager.add_answer_template(q_id1, VALID_TEMPLATE)

        missing = manager.get_missing_templates(ids_only=True)

        assert len(missing) == 1
        assert q_id2 in missing

    def test_get_missing_full_objects(self) -> None:
        """Test getting missing templates returns full objects."""
        benchmark = Benchmark.create(name="test")
        manager = TemplateManager(benchmark)

        q_id1 = benchmark.add_question("Question 1?", "Answer 1")
        benchmark.add_question("Question 2?", "Answer 2")  # Intentionally not captured

        manager.add_answer_template(q_id1, VALID_TEMPLATE)

        missing = manager.get_missing_templates(ids_only=False)

        assert len(missing) == 1
        assert missing[0]["question"] == "Question 2?"


@pytest.mark.unit
class TestApplyGlobalTemplate:
    """Tests for apply_global_template method."""

    def test_apply_global_to_all(self) -> None:
        """Test applying template to all questions without one."""
        benchmark = Benchmark.create(name="test")
        manager = TemplateManager(benchmark)

        q_id1 = benchmark.add_question("Q1?", "A1")
        q_id2 = benchmark.add_question("Q2?", "A2")

        updated = manager.apply_global_template(VALID_TEMPLATE)

        assert len(updated) == 2
        assert q_id1 in updated
        assert q_id2 in updated

    def test_apply_global_skips_existing(self) -> None:
        """Test that apply_global skips questions with templates."""
        benchmark = Benchmark.create(name="test")
        manager = TemplateManager(benchmark)

        q_id1 = benchmark.add_question("Q1?", "A1")
        q_id2 = benchmark.add_question("Q2?", "A2")

        manager.add_answer_template(q_id1, VALID_TEMPLATE)

        updated = manager.apply_global_template(VALID_TEMPLATE)

        assert len(updated) == 1
        assert q_id2 in updated
        assert q_id1 not in updated

    def test_apply_global_returns_empty_when_all_have_templates(self) -> None:
        """Test apply_global returns empty list when all have templates."""
        benchmark = Benchmark.create(name="test")
        manager = TemplateManager(benchmark)

        q_id = benchmark.add_question("Q?", "A")
        manager.add_answer_template(q_id, VALID_TEMPLATE)

        updated = manager.apply_global_template(VALID_TEMPLATE)

        assert updated == []


@pytest.mark.unit
class TestValidateTemplates:
    """Tests for validate_templates method."""

    def test_validate_empty_benchmark(self) -> None:
        """Test validating templates with empty benchmark."""
        benchmark = Benchmark.create(name="test")
        manager = TemplateManager(benchmark)

        valid, errors = manager.validate_templates()

        assert valid is True
        assert errors == []

    def test_validate_all_valid(self) -> None:
        """Test validating when all templates are valid."""
        benchmark = Benchmark.create(name="test")
        manager = TemplateManager(benchmark)

        q_id = benchmark.add_question("Q?", "A")
        manager.add_answer_template(q_id, VALID_TEMPLATE)

        valid, errors = manager.validate_templates()

        assert valid is True
        assert errors == []

    def test_validate_syntax_error(self) -> None:
        """Test validating template with syntax error."""
        benchmark = Benchmark.create(name="test")
        manager = TemplateManager(benchmark)

        q_id = benchmark.add_question("Q?", "A")

        # Manually set invalid template to bypass add_answer_template validation
        benchmark._questions_cache[q_id]["answer_template"] = "class Answer(BaseAnswer):\n    def verify("

        valid, errors = manager.validate_templates()

        assert valid is False
        assert len(errors) == 1
        assert "Syntax error" in errors[0]["error"]
        assert q_id in errors[0]["question_id"]
