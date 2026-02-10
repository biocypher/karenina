"""Template management functionality for benchmarks."""

import ast
import inspect
import logging
import textwrap
from datetime import datetime
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .base import BenchmarkBase

from karenina.schemas.verification import FinishedTemplate

from ..verification.utils.template_validation import validate_answer_template

logger = logging.getLogger(__name__)


def _rename_answer_class_to_standard(source_code: str, original_class_name: str) -> str:
    """Rename a BaseAnswer subclass to 'Answer' in source code.

    This allows users to define classes with any name (e.g., VenetoclaxAnswer),
    but stores them with the standard 'Answer' name that the verification system expects.

    Args:
        source_code: The source code containing the class definition
        original_class_name: The original name of the class to rename

    Returns:
        Modified source code with the class renamed to 'Answer'
    """
    if original_class_name == "Answer":
        return source_code

    try:
        tree = ast.parse(source_code)
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef) and node.name == original_class_name:
                node.name = "Answer"
        return ast.unparse(tree)
    except Exception:
        logger.debug(
            "AST parsing failed for class rename %s -> Answer, falling back to string replacement",
            original_class_name,
            exc_info=True,
        )
        return source_code.replace(f"class {original_class_name}(", "class Answer(")


def resolve_template_code(template: str | type) -> str:
    """Convert a template argument to source code string.

    Accepts either a source code string or a BaseAnswer subclass. When given
    a class, extracts its source code and renames it to 'Answer' if needed.

    Extraction order:
    1. ``cls.get_source_code()`` — explicit ``_source_code`` attribute
    2. ``inspect.getsource()`` — file-based classes
    3. IPython cell history — classes defined in Jupyter notebooks

    Args:
        template: Template source code string or BaseAnswer subclass

    Returns:
        Template source code string ready for storage

    Raises:
        TypeError: If template is not a string or BaseAnswer subclass
        ValueError: If source code cannot be extracted from the class
    """
    if isinstance(template, str):
        return template

    if inspect.isclass(template):
        from karenina.schemas.entities import BaseAnswer

        if not issubclass(template, BaseAnswer):
            raise TypeError(f"Template class must inherit from BaseAnswer, got {template.__name__}")

        original_class_name = getattr(template, "__name__", "Answer")
        source_code = template.get_source_code()

        if source_code is None:
            try:
                source_code = inspect.getsource(template)
            except OSError:
                source_code = _get_source_from_notebook(original_class_name)

        if source_code is None:
            raise ValueError(
                f"Could not extract source code from class {original_class_name}. "
                "For dynamically created classes, set _source_code on the class "
                "or call cls.set_source_code_from_notebook() in Jupyter."
            )

        source_code = textwrap.dedent(source_code)
        return _rename_answer_class_to_standard(source_code, original_class_name)

    raise TypeError(f"template must be a string or BaseAnswer subclass, got {type(template).__name__}")


def _get_source_from_notebook(class_name: str) -> str | None:
    """Try to extract class source code from IPython/Jupyter cell history.

    Args:
        class_name: Name of the class to find

    Returns:
        Source code string if found, None otherwise
    """
    try:
        from IPython import get_ipython  # type: ignore[attr-defined]

        ip = get_ipython()  # type: ignore[no-untyped-call]
        if ip is None:
            return None

        history = list(ip.history_manager.get_range())
        for _, _, cell_content in reversed(history[-10:]):
            if f"class {class_name}(" in cell_content:
                lines = cell_content.strip().split("\n")
                class_lines: list[str] = []
                in_class = False
                base_indent: int | None = None

                for line in lines:
                    if f"class {class_name}(" in line:
                        in_class = True
                        base_indent = len(line) - len(line.lstrip())
                        class_lines.append(line)
                    elif in_class:
                        if line.strip() == "" or (
                            base_indent is not None and len(line) - len(line.lstrip()) > base_indent
                        ):
                            class_lines.append(line)
                        else:
                            break

                if class_lines:
                    return "\n".join(class_lines)
    except ImportError:
        pass
    except Exception:
        logger.debug("Failed to extract source from notebook history", exc_info=True)
    return None


class TemplateManager:
    """Manager for answer template operations and validation."""

    def __init__(self, base: "BenchmarkBase") -> None:
        """Initialize with reference to benchmark base."""
        self.base = base

    def add_answer_template(self, question_id: str, template_code: str | type) -> None:
        """
        Add or update an answer template for a question.

        Args:
            question_id: The question ID
            template_code: Python code defining the Answer class, or a BaseAnswer subclass

        Raises:
            TypeError: If template_code is not a string or BaseAnswer subclass
            ValueError: If question_id not found or template is invalid
        """
        template_code = resolve_template_code(template_code)
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

    def update_template(self, question_id: str, template_code: str | type) -> None:
        """Update existing template (alias for add_answer_template).

        Args:
            question_id: The question ID
            template_code: Python code defining the Answer class, or a BaseAnswer subclass
        """
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

    def get_finished_templates(self, question_ids: set[str] | None = None) -> list[FinishedTemplate]:
        """
        Get all finished templates for verification.

        Args:
            question_ids: Optional set of question IDs to filter by. If None, returns all finished templates.

        Returns:
            List of FinishedTemplate objects ready for verification
        """
        templates = []
        for q_id, q_data in self.base._questions_cache.items():
            if q_data.get("finished", False) and (question_ids is None or q_id in question_ids):
                # Convert question rubric to dict format if present
                question_rubric = None
                if q_data.get("question_rubric"):
                    # question_rubric is now a dict with llm_traits, regex_traits, callable_traits, metric_traits
                    rubric_dict = q_data["question_rubric"]
                    question_rubric = {
                        "llm_traits": [trait.model_dump() for trait in rubric_dict.get("llm_traits", [])],
                        "regex_traits": [trait.model_dump() for trait in rubric_dict.get("regex_traits", [])],
                        "callable_traits": [trait.model_dump() for trait in rubric_dict.get("callable_traits", [])],
                        "metric_traits": [trait.model_dump() for trait in rubric_dict.get("metric_traits", [])],
                    }

                template = FinishedTemplate(
                    question_id=q_id,
                    question_text=q_data["question"],
                    question_preview=q_data["question"][:100] + "..."
                    if len(q_data["question"]) > 100
                    else q_data["question"],
                    raw_answer=q_data.get("raw_answer"),
                    template_code=q_data["answer_template"],
                    last_modified=q_data.get("date_modified", datetime.now().isoformat()),
                    finished=True,
                    question_rubric=question_rubric,
                    keywords=q_data.get("keywords"),
                    few_shot_examples=q_data.get("few_shot_examples"),
                )
                templates.append(template)
        return templates

    def get_missing_templates(self, ids_only: bool = False) -> list[str] | list[dict[str, Any]]:
        """
        Get questions that don't have non-default templates.

        Args:
            ids_only: If True, return only question IDs. If False (default), return full question objects.

        Returns:
            List of question IDs (if ids_only=True) or list of question dictionaries (if ids_only=False)
        """
        if ids_only:
            return [q_id for q_id in self.base._questions_cache if not self.has_template(q_id)]
        else:
            return [q_data for q_id, q_data in self.base._questions_cache.items() if not self.has_template(q_id)]

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
        # Default template: returns False to indicate custom verification needed
        return False
'''
