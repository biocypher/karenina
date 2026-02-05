"""Question management functionality for benchmarks.

This module provides the QuestionManager class which handles all question
CRUD operations and metadata management. Filtering and search operations
are delegated to QuestionQueryBuilder for single responsibility.
"""

import ast
import inspect
import logging
from collections.abc import Iterator
from datetime import datetime
from typing import TYPE_CHECKING, Any, Union

if TYPE_CHECKING:
    from ...schemas.entities import Question
    from .base import BenchmarkBase

from ...utils.checkpoint import add_question_to_benchmark
from .question_query import QuestionQueryBuilder

logger = logging.getLogger(__name__)

# Sentinel value to detect if finished parameter was explicitly provided
_NOT_PROVIDED = object()


def _rename_answer_class_to_standard(source_code: str, original_class_name: str) -> str:
    """
    Rename a BaseAnswer subclass to 'Answer' in source code.

    This allows users to define classes with any name (e.g., VenetoclaxAnswer),
    but stores them with the standard 'Answer' name that the verification system expects.

    Args:
        source_code: The source code containing the class definition
        original_class_name: The original name of the class to rename

    Returns:
        Modified source code with the class renamed to 'Answer'
    """
    # If already named "Answer", no change needed
    if original_class_name == "Answer":
        return source_code

    try:
        tree = ast.parse(source_code)

        # Find and rename the class definition
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef) and node.name == original_class_name:
                node.name = "Answer"

        # Convert back to source code
        return ast.unparse(tree)
    except Exception:
        # If AST parsing fails, fall back to simple string replacement
        # This is a safety net but should rarely be needed
        logger.debug(
            "AST parsing failed for class rename %s -> Answer, falling back to string replacement",
            original_class_name,
            exc_info=True,
        )
        return source_code.replace(f"class {original_class_name}(", "class Answer(")


class QuestionManager:
    """Manager for question CRUD operations and metadata.

    This class handles all question-related operations including:
    - CRUD operations (add, get, update, remove questions)
    - Metadata management (author, sources, custom properties)
    - Finished status management

    Filtering, searching, and aggregation operations are delegated to
    QuestionQueryBuilder for single responsibility.
    """

    def __init__(self, base: "BenchmarkBase") -> None:
        """Initialize with reference to benchmark base."""
        self.base = base
        self._query_builder: QuestionQueryBuilder | None = None

    @property
    def query_builder(self) -> QuestionQueryBuilder:
        """Lazy-initialized query builder for filtering and searching."""
        if self._query_builder is None:
            self._query_builder = QuestionQueryBuilder(
                self.base,
                template_checker=self._is_default_template,
            )
        return self._query_builder

    def add_question(
        self,
        question: Union[str, "Question"],
        raw_answer: str | None = None,
        answer_template: str | type | None = None,
        question_id: str | None = None,
        finished: bool | object = _NOT_PROVIDED,
        author: dict[str, Any] | None = None,
        sources: list[dict[str, Any]] | None = None,
        custom_metadata: dict[str, Any] | None = None,
        few_shot_examples: list[dict[str, str]] | None = None,
    ) -> str:
        """
        Add a question to the benchmark.

        This method supports three usage patterns:
        1. Traditional kwargs: add_question("What is 2+2?", "4", ...)
        2. Question object: add_question(Question(...), ...)
        3. Answer class: add_question("What is 2+2?", "4", answer_template=AnswerClass)

        Args:
            question: Either the question text (str) or a Question object
            raw_answer: The expected answer text (required if question is str)
            answer_template: Optional Python code (str), Answer class (type), or None
            question_id: Optional question ID (will be generated if not provided)
            finished: Whether the template is finished (auto-set to True if answer_template provided)
            author: Optional author information
            sources: Optional source documents
            custom_metadata: Optional custom metadata
            few_shot_examples: Optional list of few-shot examples with 'question' and 'answer' keys

        Returns:
            The question ID

        Examples:
            # Traditional usage with kwargs
            q_id = manager.add_question("What is Python?", "A programming language")

            # New usage with Question object
            q_obj = Question(question="What is Python?", raw_answer="A programming language")
            q_id = manager.add_question(q_obj)

            # New usage with Answer class - automatically marked as finished
            class MyAnswer(BaseAnswer):
                value: int
                def verify(self): return self.value == 42
            q_id = manager.add_question("What is 6*7?", "42", answer_template=MyAnswer)
        """
        # Import Question class here to avoid circular imports
        from ...schemas.entities import Question

        # Track whether user provided an answer template (before we set default)
        user_provided_template = answer_template is not None

        # Handle Question object input
        if isinstance(question, Question):
            question_text = question.question
            raw_answer_text = question.raw_answer
            # Use the Question object's auto-generated ID if no ID provided
            if question_id is None:
                question_id = question.id
            # Extract tags/keywords if provided
            keywords = [tag for tag in question.tags if tag is not None] if question.tags else None
            # Use few-shot examples from Question object if not overridden
            if few_shot_examples is None:
                few_shot_examples = question.few_shot_examples
        elif isinstance(question, str):
            # Traditional string input
            question_text = question
            raw_answer_text = raw_answer or ""
            keywords = None

            # Validate required parameters for string input
            if raw_answer is None:
                raise ValueError("raw_answer is required when question is a string")
        else:
            # Invalid type
            raise TypeError(f"question must be either a string or Question object, got {type(question)}")

        # Handle Answer class input for answer_template
        if answer_template is not None:
            if inspect.isclass(answer_template):
                # Import BaseAnswer here to avoid circular imports
                from ...schemas.entities import BaseAnswer

                # Validate that it's an Answer class
                if not issubclass(answer_template, BaseAnswer):
                    raise TypeError(f"answer_template class must inherit from BaseAnswer, got {answer_template}")

                # Capture the original class name for renaming
                original_class_name = getattr(answer_template, "__name__", "Answer")

                # Convert Answer class to source code string
                try:
                    # First try to get source code using the BaseAnswer method
                    source_code = answer_template.get_source_code()
                    if source_code is None:
                        source_code = inspect.getsource(answer_template)

                    # Rename the class to "Answer" for verification system compatibility
                    # This allows users to use any class name (e.g., VenetoclaxAnswer)
                    answer_template = _rename_answer_class_to_standard(source_code, original_class_name)
                except (OSError, TypeError) as e:
                    class_name = getattr(answer_template, "__name__", "Unknown")
                    raise ValueError(
                        f"Could not extract source code from Answer class {class_name}. "
                        f"For dynamically created classes, ensure _source_code is set. Error: {e}"
                    ) from e
            elif not isinstance(answer_template, str):
                # If it's not None, not a class, and not a string, it's invalid
                raise TypeError(
                    f"answer_template must be either a string, a BaseAnswer subclass, or None. Got {type(answer_template)}"
                )

        # If no template provided, create a minimal one
        if answer_template is None:
            answer_template = self._create_default_template(question_text)

        # At this point, answer_template is guaranteed to be a string
        assert isinstance(answer_template, str), "answer_template should be a string at this point"

        # Auto-set finished flag if user provided a template but didn't explicitly set finished
        # This enables backend usage (like ManualTraces) to skip the manual mark_finished() call
        # Frontend behavior is preserved since it always explicitly passes finished=False
        if finished is _NOT_PROVIDED:
            # User didn't provide finished parameter - auto-set based on template presence
            finished = bool(user_provided_template)
        # else: User explicitly provided finished value, use it as-is

        # Type narrowing: finished is guaranteed to be bool at this point
        assert isinstance(finished, bool), "finished must be bool after sentinel check"

        q_id = add_question_to_benchmark(
            self.base._checkpoint,
            question_text,
            raw_answer_text,
            answer_template,
            question_id,
            None,  # question rubric traits added separately
            finished,
            author,
            sources,
            custom_metadata,
            keywords,  # Pass keywords from Question object if available
            few_shot_examples,
        )

        # Update cache
        self.base._rebuild_cache()
        return q_id

    def remove_question(self, question_id: str) -> bool:
        """
        Remove a specific question from the benchmark.

        Args:
            question_id: The question ID to remove

        Returns:
            True if question was removed, False if not found
        """
        # Remove from cache
        if question_id not in self.base._questions_cache:
            return False

        del self.base._questions_cache[question_id]

        # Remove from checkpoint data
        items_to_remove = []
        for i, item in enumerate(self.base._checkpoint.dataFeedElement):
            if self.base._get_item_id(item) == question_id:
                items_to_remove.append(i)

        # Remove in reverse order to maintain indices
        for i in reversed(items_to_remove):
            del self.base._checkpoint.dataFeedElement[i]

        self.base._checkpoint.dateModified = datetime.now().isoformat()
        return True

    def get_question_ids(self) -> list[str]:
        """
        Get all question IDs in the benchmark.

        This is a convenience method that calls get_all_questions(ids_only=True).

        Returns:
            List of question IDs
        """
        from typing import cast

        return cast(list[str], self.get_all_questions(ids_only=True))

    def get_question(self, question_id: str) -> dict[str, Any]:
        """
        Get a question by ID.

        Args:
            question_id: The question ID

        Returns:
            Question dictionary with all fields

        Raises:
            ValueError: If question not found
        """
        if question_id not in self.base._questions_cache:
            raise ValueError(f"Question not found: {question_id}")
        return self.base._questions_cache[question_id]

    def get_all_questions(self, ids_only: bool = False) -> list[str] | list[dict[str, Any]]:
        """
        Get all questions in the benchmark.

        Args:
            ids_only: If True, return only question IDs. If False (default), return full question objects.

        Returns:
            List of question IDs (if ids_only=True) or list of question dictionaries (if ids_only=False)
        """
        if ids_only:
            return list(self.base._questions_cache.keys())
        else:
            return list(self.base._questions_cache.values())

    def get_question_as_object(self, question_id: str) -> "Question":
        """
        Get a question as a Question object (for use with TaskEval).

        Args:
            question_id: The question ID

        Returns:
            Question object with the question data

        Raises:
            ValueError: If question not found
        """
        from ...schemas.entities import Question

        q_data = self.get_question(question_id)
        return Question(
            question=q_data["question"],
            raw_answer=q_data["raw_answer"],
            tags=q_data.get("keywords", []) or [],
            few_shot_examples=q_data.get("few_shot_examples"),
        )

    def get_all_questions_as_objects(self) -> list["Question"]:
        """
        Get all questions as Question objects.

        Returns:
            List of Question objects
        """
        from typing import cast

        from ...schemas.entities import Question

        objects = []
        for q_data in cast(list[dict[str, Any]], self.get_all_questions()):
            objects.append(
                Question(
                    question=q_data["question"],
                    raw_answer=q_data["raw_answer"],
                    tags=q_data.get("keywords", []) or [],
                    few_shot_examples=q_data.get("few_shot_examples"),
                )
            )
        return objects

    def add_question_from_object(self, question_obj: "Question", **metadata: Any) -> str:
        """
        Add a question to the benchmark from a Question object.

        Args:
            question_obj: Question object to add
            **metadata: Additional metadata (author, sources, etc.)

        Returns:
            The question ID that was assigned
        """
        from ...schemas.entities import Question

        if not isinstance(question_obj, Question):
            raise ValueError("question_obj must be a Question instance")

        # Use the Question object's auto-generated ID
        question_id = question_obj.id

        # Check if question already exists
        if question_id in self.base._questions_cache:
            raise ValueError(f"Question with ID {question_id} already exists")

        # Add to benchmark using existing add_question method with the Question object's ID
        self.add_question(
            question=question_obj.question,
            raw_answer=question_obj.raw_answer,
            question_id=question_id,  # Use the Question object's auto-generated ID
            few_shot_examples=question_obj.few_shot_examples,
            **metadata,
        )

        # Add keywords separately if provided
        if question_obj.tags:
            # Update the question with keywords
            for item in self.base._checkpoint.dataFeedElement:
                if self.base._get_item_id(item) == question_id:
                    # Convert tags to appropriate format, filtering out None values
                    item.keywords = [tag for tag in question_obj.tags if tag is not None]
                    break
            # Rebuild cache to reflect changes
            self.base._rebuild_cache()

        return question_id

    def update_question_metadata(self, question_id: str, **metadata: Any) -> None:
        """
        Update question metadata fields.

        Args:
            question_id: The question ID
            **metadata: Metadata fields to update

        Raises:
            ValueError: If question not found
        """
        if question_id not in self.base._questions_cache:
            raise ValueError(f"Question not found: {question_id}")

        # Update cache
        question_data = self.base._questions_cache[question_id]

        # Update basic fields
        if "question" in metadata:
            question_data["question"] = metadata["question"]
            # Update underlying JSON-LD structure
            for item in self.base._checkpoint.dataFeedElement:
                if self.base._get_item_id(item) == question_id:
                    item.item.text = metadata["question"]
                    break

        if "raw_answer" in metadata:
            question_data["raw_answer"] = metadata["raw_answer"]
            # Update underlying JSON-LD structure
            for item in self.base._checkpoint.dataFeedElement:
                if self.base._get_item_id(item) == question_id:
                    item.item.acceptedAnswer.text = metadata["raw_answer"]
                    break

        # Update additional properties
        for key in ["author", "sources"]:
            if key in metadata:
                question_data[key] = metadata[key]
                self.base._update_question_property(question_id, key, metadata[key])

        # Update custom metadata
        if "custom_metadata" in metadata:
            custom_meta = metadata["custom_metadata"] or {}
            question_data["custom_metadata"] = custom_meta
            # Update each custom field as a separate property
            for custom_key, custom_value in custom_meta.items():
                self.base._update_question_property(question_id, f"custom_{custom_key}", custom_value)

        # Update modification timestamp
        question_data["date_modified"] = datetime.now().isoformat()
        for item in self.base._checkpoint.dataFeedElement:
            if self.base._get_item_id(item) == question_id:
                item.dateModified = question_data["date_modified"]
                break

    def get_question_metadata(self, question_id: str) -> dict[str, Any]:
        """
        Get all metadata for a specific question.

        Args:
            question_id: The question ID

        Returns:
            Dictionary containing all question metadata

        Raises:
            ValueError: If question not found
        """
        question_data = self.get_question(question_id)
        return {
            "id": question_data["id"],
            "question": question_data["question"],
            "raw_answer": question_data["raw_answer"],
            "date_created": question_data["date_created"],
            "date_modified": question_data["date_modified"],
            "finished": question_data.get("finished", False),
            "author": question_data.get("author"),
            "sources": question_data.get("sources"),
            "custom_metadata": question_data.get("custom_metadata", {}),
            "has_template": self._has_non_default_template(question_id),
            "has_rubric": bool(question_data.get("question_rubric")),
        }

    def get_question_custom_property(self, question_id: str, name: str) -> Any:
        """
        Get a custom property from question metadata.

        Args:
            question_id: The question ID
            name: Property name

        Returns:
            Property value or None if not found

        Raises:
            ValueError: If question not found
        """
        question_data = self.get_question(question_id)
        custom_metadata = question_data.get("custom_metadata", {})
        return custom_metadata.get(name)

    def set_question_custom_property(self, question_id: str, name: str, value: Any) -> None:
        """
        Set a custom property on question metadata.

        Args:
            question_id: The question ID
            name: Property name
            value: Property value

        Raises:
            ValueError: If question not found
        """
        if question_id not in self.base._questions_cache:
            raise ValueError(f"Question not found: {question_id}")

        # Update cache
        question_data = self.base._questions_cache[question_id]
        if not question_data.get("custom_metadata"):
            question_data["custom_metadata"] = {}
        question_data["custom_metadata"][name] = value

        # Update underlying JSON-LD structure
        self.base._update_question_property(question_id, f"custom_{name}", value)

    def remove_question_custom_property(self, question_id: str, name: str) -> bool:
        """
        Remove a custom property from question metadata.

        Args:
            question_id: The question ID
            name: Property name

        Returns:
            True if property was found and removed, False otherwise

        Raises:
            ValueError: If question not found
        """
        if question_id not in self.base._questions_cache:
            raise ValueError(f"Question not found: {question_id}")

        # Update cache
        question_data = self.base._questions_cache[question_id]
        custom_metadata = question_data.get("custom_metadata", {})

        if name not in custom_metadata:
            return False

        del custom_metadata[name]
        question_data["custom_metadata"] = custom_metadata if custom_metadata else None

        # Remove from underlying JSON-LD structure
        for item in self.base._checkpoint.dataFeedElement:
            if self.base._get_item_id(item) == question_id:
                if item.item.additionalProperty:
                    for i, prop in enumerate(item.item.additionalProperty):
                        if prop.name == f"custom_{name}":
                            del item.item.additionalProperty[i]
                            item.dateModified = datetime.now().isoformat()
                            return True
                break

        return False

    def get_question_author(self, question_id: str) -> dict[str, Any] | None:
        """Get author information for a question."""
        question_data = self.get_question(question_id)
        return question_data.get("author")

    def set_question_author(self, question_id: str, author: dict[str, Any] | None) -> None:
        """Set author information for a question."""
        self.update_question_metadata(question_id, author=author)

    def get_question_sources(self, question_id: str) -> list[dict[str, Any]] | None:
        """Get source documents for a question."""
        question_data = self.get_question(question_id)
        return question_data.get("sources")

    def set_question_sources(self, question_id: str, sources: list[dict[str, Any]] | None) -> None:
        """Set source documents for a question."""
        self.update_question_metadata(question_id, sources=sources)

    def get_question_timestamps(self, question_id: str) -> dict[str, str]:
        """
        Get creation and modification timestamps for a question.

        Args:
            question_id: The question ID

        Returns:
            Dictionary with 'created' and 'modified' timestamp strings

        Raises:
            ValueError: If question not found
        """
        question_data = self.get_question(question_id)
        return {"created": question_data["date_created"], "modified": question_data["date_modified"]}

    def clear_questions(self) -> int:
        """
        Remove all questions from the benchmark.

        Returns:
            Number of questions that were removed
        """
        count = len(self.base._questions_cache)
        self.base._questions_cache.clear()
        self.base._checkpoint.dataFeedElement.clear()
        self.base._checkpoint.dateModified = datetime.now().isoformat()
        return count

    def add_questions_batch(self, questions_data: list[dict[str, Any]]) -> list[str]:
        """
        Add multiple questions at once.

        Args:
            questions_data: List of dictionaries with question data

        Returns:
            List of question IDs that were created
        """
        question_ids = []
        for data in questions_data:
            q_id = self.add_question(
                question=data["question"],
                raw_answer=data["raw_answer"],
                answer_template=data.get("answer_template"),
                question_id=data.get("question_id"),
                finished=data.get("finished", False),
                author=data.get("author"),
                sources=data.get("sources"),
                custom_metadata=data.get("custom_metadata"),
            )
            question_ids.append(q_id)
        return question_ids

    def mark_finished(self, question_id: str) -> None:
        """Mark a question as finished."""
        if question_id in self.base._questions_cache:
            # Update cache
            self.base._questions_cache[question_id]["finished"] = True
            # Update underlying JSON-LD structure
            self.base._update_question_property(question_id, "finished", True)

    def mark_unfinished(self, question_id: str) -> None:
        """Mark a question as unfinished."""
        if question_id in self.base._questions_cache:
            # Update cache
            self.base._questions_cache[question_id]["finished"] = False
            # Update underlying JSON-LD structure
            self.base._update_question_property(question_id, "finished", False)

    def mark_finished_batch(self, question_ids: list[str]) -> None:
        """Mark multiple questions as finished."""
        for q_id in question_ids:
            self.mark_finished(q_id)

    def mark_unfinished_batch(self, question_ids: list[str]) -> None:
        """Mark multiple questions as unfinished."""
        for q_id in question_ids:
            self.mark_unfinished(q_id)

    def toggle_finished(self, question_id: str) -> bool:
        """
        Toggle finished status of a question.

        Args:
            question_id: The question ID

        Returns:
            New finished status (True/False)

        Raises:
            ValueError: If question not found
        """
        if question_id not in self.base._questions_cache:
            raise ValueError(f"Question not found: {question_id}")

        current_status = self.base._questions_cache[question_id].get("finished", False)
        new_status = not current_status
        # Update cache
        self.base._questions_cache[question_id]["finished"] = new_status
        # Update underlying JSON-LD structure
        self.base._update_question_property(question_id, "finished", new_status)
        return new_status

    def get_unfinished_questions(self, ids_only: bool = False) -> list[str] | list[dict[str, Any]]:
        """
        Get questions that are not marked as finished.

        Args:
            ids_only: If True, return only question IDs. If False (default), return full question objects.

        Returns:
            List of question IDs (if ids_only=True) or list of question dictionaries (if ids_only=False)
        """
        if ids_only:
            return [q_id for q_id, q_data in self.base._questions_cache.items() if not q_data.get("finished", False)]
        else:
            return [q_data for q_data in self.base._questions_cache.values() if not q_data.get("finished", False)]

    def get_finished_questions(self, ids_only: bool = False) -> list[str] | list[dict[str, Any]]:
        """
        Get questions that are marked as finished.

        Args:
            ids_only: If True, return only question IDs. If False (default), return full question objects.

        Returns:
            List of question IDs (if ids_only=True) or list of question dictionaries (if ids_only=False)
        """
        if ids_only:
            return [q_id for q_id, q_data in self.base._questions_cache.items() if q_data.get("finished", False)]
        else:
            return [q_data for q_data in self.base._questions_cache.values() if q_data.get("finished", False)]

    def filter_questions(
        self,
        finished: bool | None = None,
        has_template: bool | None = None,
        has_rubric: bool | None = None,
        author: str | None = None,
        custom_filter: Any = None,  # Callable[[dict[str, Any]], bool] | None
    ) -> list[dict[str, Any]]:
        """
        Filter questions based on criteria.

        Args:
            finished: Filter by finished status (True/False/None for all)
            has_template: Filter by template existence (True/False/None for all)
            has_rubric: Filter by rubric existence (True/False/None for all)
            author: Filter by author name (None for all)
            custom_filter: Optional lambda/function to apply custom filtering logic

        Returns:
            List of question dictionaries matching criteria

        Examples:
            # Filter by system metadata
            finished_qs = benchmark.filter_questions(finished=True)

            # Filter with custom lambda
            high_priority = benchmark.filter_questions(
                custom_filter=lambda q: q.get("custom_metadata", {}).get("priority") == "high"
            )

            # Combine system and custom filters
            results = benchmark.filter_questions(
                finished=True,
                has_template=True,
                custom_filter=lambda q: q.get("custom_metadata", {}).get("category") == "math"
            )
        """
        return self.query_builder.filter_questions(
            finished=finished,
            has_template=has_template,
            has_rubric=has_rubric,
            author=author,
            custom_filter=custom_filter,
        )

    def filter_by_metadata(
        self,
        field_path: str,
        value: Any,
        match_mode: str = "exact",
    ) -> list[dict[str, Any]]:
        """Filter questions by a metadata field using dot notation."""
        return self.query_builder.filter_by_metadata(field_path, value, match_mode)

    def filter_by_custom_metadata(
        self,
        match_all: bool = True,
        **criteria: Any,
    ) -> list[dict[str, Any]]:
        """Filter questions by custom metadata fields with AND/OR logic."""
        return self.query_builder.filter_by_custom_metadata(match_all, **criteria)

    def search_questions(
        self,
        query: str | list[str],
        match_all: bool = True,
        fields: list[str] | None = None,
        case_sensitive: bool = False,
        regex: bool = False,
    ) -> list[dict[str, Any]]:
        """Search for questions containing the query text (unified search method)."""
        return self.query_builder.search_questions(query, match_all, fields, case_sensitive, regex)

    def get_questions_by_author(self, author: str) -> list[dict[str, Any]]:
        """Get questions created by a specific author."""
        return self.query_builder.get_questions_by_author(author)

    def get_questions_with_rubric(self) -> list[dict[str, Any]]:
        """Get questions that have question-specific rubrics."""
        return self.query_builder.get_questions_with_rubric()

    def count_by_field(
        self,
        field_path: str,
        questions: list[dict[str, Any]] | None = None,
    ) -> dict[Any, int]:
        """Count questions grouped by a field value using dot notation."""
        return self.query_builder.count_by_field(field_path, questions)

    def __iter__(self) -> Iterator[dict[str, Any]]:
        """Iterate over questions in the benchmark."""
        return iter(self.base._questions_cache.values())

    def _create_default_template(self, question: str) -> str:
        """Create a minimal default template for a question."""
        return f'''class Answer(BaseAnswer):
    """Answer template for: {question[:50]}..."""

    response: str = Field(description="The answer response")

    def verify(self) -> bool:
        # Default template: returns False to indicate custom verification needed
        return False
'''

    def _is_default_template(self, template: str, question: str) -> bool:
        """Check if a template is the auto-generated default."""
        if not template:
            return False
        # Check if it matches the default template pattern
        expected_default = self._create_default_template(question)
        return template.strip() == expected_default.strip()

    def _has_non_default_template(self, question_id: str) -> bool:
        """Check if a question has a non-default template."""
        if question_id not in self.base._questions_cache:
            return False

        template = self.base._questions_cache[question_id].get("answer_template")
        if not template:
            return False

        # Check if it's just the default template
        question_text = self.base._questions_cache[question_id].get("question", "")
        return not self._is_default_template(template, question_text)
