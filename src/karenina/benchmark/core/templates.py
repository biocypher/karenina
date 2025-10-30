"""Template management functionality for benchmarks."""

import ast
from datetime import datetime
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .base import BenchmarkBase

from ..models import FinishedTemplate
from ..verification.utils.validation import validate_answer_template


class TemplateManager:
    """Manager for answer template operations and validation."""

    def __init__(self, base: "BenchmarkBase") -> None:
        """Initialize with reference to benchmark base."""
        self.base = base

    def add_answer_template(self, question_id: str, template_code: str) -> None:
        """
        Add or update an answer template for a question.

        Args:
            question_id: The question ID
            template_code: Python code defining the Answer class

        Raises:
            ValueError: If question_id not found or template is invalid
        """
        # Validate the template using the verification system
        is_valid, error_msg, _ = validate_answer_template(template_code)
        if not is_valid:
            raise ValueError(f"Invalid template: {error_msg}")

        # Find the question in the checkpoint
        found = False
        for item in self.base._checkpoint.dataFeedElement:
            if self.base._get_item_id(item) == question_id:
                item.item.hasPart.text = template_code
                item.dateModified = datetime.now().isoformat()
                found = True
                break

        if not found:
            raise ValueError(f"Question not found: {question_id}")

        # Update cache
        self.base._rebuild_cache()

    def has_template(self, question_id: str) -> bool:
        """
        Check if a question has a non-default template.

        Args:
            question_id: The question ID

        Returns:
            True if question has a meaningful template, False otherwise
        """
        if question_id not in self.base._questions_cache:
            return False

        template = self.base._questions_cache[question_id].get("answer_template")
        if not template:
            return False

        # Check if it's just the default template
        question_text = self.base._questions_cache[question_id].get("question", "")
        return not self._is_default_template(template, question_text)

    def get_template(self, question_id: str) -> str:
        """
        Get template code for a question.

        Args:
            question_id: The question ID

        Returns:
            Template code string

        Raises:
            ValueError: If question not found or has no template
        """
        if question_id not in self.base._questions_cache:
            raise ValueError(f"Question not found: {question_id}")

        template = self.base._questions_cache[question_id].get("answer_template")
        if not template:
            raise ValueError(f"Question {question_id} has no template")

        # Check if it's just the default template
        question_text = self.base._questions_cache[question_id].get("question", "")
        if self._is_default_template(template, question_text):
            raise ValueError(f"Question {question_id} has no template")

        return str(template)

    def update_template(self, question_id: str, template_code: str) -> None:
        """Update existing template (alias for add_answer_template)."""
        self.add_answer_template(question_id, template_code)

    def copy_template(self, from_id: str, to_id: str) -> None:
        """
        Copy template from one question to another.

        Args:
            from_id: Source question ID
            to_id: Destination question ID

        Raises:
            ValueError: If source question not found or has no template
        """
        template = self.get_template(from_id)
        self.add_answer_template(to_id, template)

    def get_finished_templates(self) -> list[FinishedTemplate]:
        """
        Get all finished templates for verification.

        Returns:
            List of FinishedTemplate objects ready for verification
        """
        templates = []
        for q_id, q_data in self.base._questions_cache.items():
            if q_data.get("finished", False):
                # Convert question rubric to dict format if present
                question_rubric = None
                if q_data.get("question_rubric"):
                    question_rubric = {"traits": [trait.model_dump() for trait in q_data["question_rubric"]]}

                template = FinishedTemplate(
                    question_id=q_id,
                    question_text=q_data["question"],
                    question_preview=q_data["question"][:100] + "..."
                    if len(q_data["question"]) > 100
                    else q_data["question"],
                    template_code=q_data["answer_template"],
                    last_modified=q_data.get("date_modified", datetime.now().isoformat()),
                    finished=True,
                    question_rubric=question_rubric,
                )
                templates.append(template)
        return templates

    def get_missing_templates(self) -> list[str]:
        """Get list of question IDs that don't have non-default templates."""
        return [q_id for q_id in self.base._questions_cache if not self.has_template(q_id)]

    def apply_global_template(self, template_code: str) -> list[str]:
        """
        Apply a template to all questions that don't have one.

        Args:
            template_code: The template code to apply

        Returns:
            List of question IDs that received the template
        """
        updated_ids = []
        for q_id in self.base._questions_cache:
            if not self.has_template(q_id):
                self.add_answer_template(q_id, template_code)
                updated_ids.append(q_id)
        return updated_ids

    def validate_templates(self) -> tuple[bool, list[dict[str, str]]]:
        """
        Validate all templates are valid Python code.

        Returns:
            Tuple of (all_valid, list_of_errors)
            Each error dict has 'question_id', 'error' keys
        """
        errors = []

        for q_id, q_data in self.base._questions_cache.items():
            template = q_data.get("answer_template")
            if not template:
                continue

            try:
                # Try to parse as valid Python syntax
                ast.parse(template)
            except SyntaxError as e:
                errors.append({"question_id": q_id, "error": f"Syntax error: {e.msg} at line {e.lineno}"})
            except Exception as e:
                errors.append({"question_id": q_id, "error": f"Parse error: {str(e)}"})

        return len(errors) == 0, errors

    def validate_template_with_verification_system(self, question_id: str) -> tuple[bool, str | None]:
        """
        Validate a specific template using the verification system.

        Args:
            question_id: The question ID

        Returns:
            Tuple of (is_valid, error_message)
        """
        if question_id not in self.base._questions_cache:
            return False, f"Question not found: {question_id}"

        template = self.base._questions_cache[question_id].get("answer_template")
        if not template:
            return False, f"Question {question_id} has no template"

        is_valid, error_msg, _ = validate_answer_template(template)
        return is_valid, error_msg

    def get_template_statistics(self) -> dict[str, Any]:
        """Get statistics about templates in the benchmark."""
        templates = [
            q.get("answer_template", "") for q_id, q in self.base._questions_cache.items() if self.has_template(q_id)
        ]

        if not templates:
            return {
                "total_templates": 0,
                "avg_template_length": 0,
                "min_template_length": 0,
                "max_template_length": 0,
                "templates_with_errors": 0,
            }

        avg_template_length = int(sum(len(t) for t in templates) / len(templates))

        # Count templates with syntax errors
        _, errors = self.validate_templates()
        templates_with_errors = len(errors)

        return {
            "total_templates": len(templates),
            "avg_template_length": avg_template_length,
            "min_template_length": min(len(t) for t in templates),
            "max_template_length": max(len(t) for t in templates),
            "templates_with_errors": templates_with_errors,
        }

    def _is_default_template(self, template: str, question: str) -> bool:
        """Check if a template is the auto-generated default."""
        if not template:
            return False
        # Check if it matches the default template pattern
        expected_default = self._create_default_template(question)
        return template.strip() == expected_default.strip()

    def _create_default_template(self, question: str) -> str:
        """Create a minimal default template for a question."""
        return f'''class Answer(BaseAnswer):
    """Answer template for: {question[:50]}..."""

    response: str = Field(description="The answer response")

    def verify(self) -> bool:
        # TODO: Implement verification logic
        return True
'''
