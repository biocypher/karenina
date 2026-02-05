"""Question query and filtering functionality for benchmarks.

This module provides the QuestionQueryBuilder class which handles all filtering,
searching, and aggregation operations on questions, extracted from QuestionManager
to follow single responsibility principle.
"""

import logging
import re
from collections import Counter
from typing import TYPE_CHECKING, Any

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from .base import BenchmarkBase


class QuestionQueryBuilder:
    """Builder class for querying and filtering questions in a benchmark.

    This class provides a fluent interface for building complex queries
    against benchmark questions. It handles filtering, searching, and
    aggregation operations.

    Example:
        # Direct usage
        query_builder = QuestionQueryBuilder(benchmark_base)
        results = query_builder.filter_questions(finished=True, has_template=True)

        # Search with multiple terms
        results = query_builder.search_questions(
            ["quantum", "mechanics"],
            match_all=True,
            fields=["question", "raw_answer"]
        )

        # Count by custom metadata field
        counts = query_builder.count_by_field("custom_metadata.category")
    """

    def __init__(
        self,
        base: "BenchmarkBase",
        template_checker: Any = None,
    ) -> None:
        """Initialize the query builder.

        Args:
            base: Reference to the benchmark base containing questions cache
            template_checker: Optional callable to check if a template is default
                             (takes template, question as args, returns bool)
        """
        self.base = base
        self._template_checker = template_checker

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
            finished_qs = query_builder.filter_questions(finished=True)

            # Filter with custom lambda
            high_priority = query_builder.filter_questions(
                custom_filter=lambda q: q.get("custom_metadata", {}).get("priority") == "high"
            )

            # Combine system and custom filters
            results = query_builder.filter_questions(
                finished=True,
                has_template=True,
                custom_filter=lambda q: q.get("custom_metadata", {}).get("category") == "math"
            )
        """
        results = []

        for _q_id, q_data in self.base._questions_cache.items():
            # Check finished status
            if finished is not None and q_data.get("finished", False) != finished:
                continue

            # Check template existence (non-default templates only)
            if has_template is not None:
                has_tmpl = self._check_has_template(q_data)
                if has_tmpl != has_template:
                    continue

            # Check rubric existence
            if has_rubric is not None:
                has_rub = bool(q_data.get("question_rubric"))
                if has_rub != has_rubric:
                    continue

            # Check author
            if author is not None:
                q_author = q_data.get("author", {}).get("name", "") if q_data.get("author") else ""
                if author.lower() not in q_author.lower():
                    continue

            # Apply custom filter function
            if custom_filter is not None:
                try:
                    if not custom_filter(q_data):
                        continue
                except Exception:
                    logger.debug("Custom filter raised exception for question, skipping", exc_info=True)
                    continue

            results.append(q_data)

        return results

    def filter_by_metadata(
        self,
        field_path: str,
        value: Any,
        match_mode: str = "exact",
    ) -> list[dict[str, Any]]:
        """
        Filter questions by a metadata field using dot notation for nested fields.

        Args:
            field_path: Dot-notation path to field (e.g., "custom_metadata.category", "finished")
            value: Value to match against
            match_mode: Matching mode - "exact", "contains", "in", or "regex"

        Returns:
            List of question dictionaries matching the criteria

        Examples:
            # Exact match on custom metadata
            math_qs = query_builder.filter_by_metadata("custom_metadata.category", "math")

            # Value in list (for tags/arrays)
            algebra_qs = query_builder.filter_by_metadata("custom_metadata.tags", "algebra", match_mode="in")

            # Substring match
            bio_qs = query_builder.filter_by_metadata("custom_metadata.domain", "bio", match_mode="contains")

            # Regex match
            calc_qs = query_builder.filter_by_metadata("question", r"calculate.*sum", match_mode="regex")
        """
        results = []
        field_parts = field_path.split(".")

        for q_data in self.base._questions_cache.values():
            # Navigate to the field using dot notation
            field_value = self._get_nested_field(q_data, field_parts)

            # Skip if field doesn't exist
            if field_value is None:
                continue

            # Apply matching logic based on match_mode
            if self._matches_value(field_value, value, match_mode):
                results.append(q_data)

        return results

    def filter_by_custom_metadata(
        self,
        match_all: bool = True,
        **criteria: Any,
    ) -> list[dict[str, Any]]:
        """
        Filter questions by custom metadata fields with AND/OR logic.

        Args:
            match_all: If True, all criteria must match (AND). If False, any criterion matches (OR)
            **criteria: Keyword arguments for custom metadata fields to match

        Returns:
            List of question dictionaries matching the criteria

        Examples:
            # AND logic: both category and difficulty must match
            math_hard = query_builder.filter_by_custom_metadata(category="math", difficulty="hard")

            # OR logic: match any of the categories
            stem = query_builder.filter_by_custom_metadata(
                match_all=False,
                category="math",
                subject="science"
            )
        """
        results = []

        for q_data in self.base._questions_cache.values():
            custom_meta = q_data.get("custom_metadata")

            # Skip questions without custom metadata
            if custom_meta is None:
                continue

            # Ensure custom_meta is a dict
            if not isinstance(custom_meta, dict):
                continue

            if match_all:
                # AND logic: all criteria must match
                if all(custom_meta.get(key) == val for key, val in criteria.items()):
                    results.append(q_data)
            else:
                # OR logic: any criterion matches
                if any(custom_meta.get(key) == val for key, val in criteria.items()):
                    results.append(q_data)

        return results

    def search_questions(
        self,
        query: str | list[str],
        match_all: bool = True,
        fields: list[str] | None = None,
        case_sensitive: bool = False,
        regex: bool = False,
    ) -> list[dict[str, Any]]:
        """
        Search for questions containing the query text (unified search method).

        Args:
            query: Single search term (str) or list of terms for multi-term search
            match_all: For multi-term search - True for AND logic, False for OR logic
            fields: Fields to search in. Default: ["question"]. Can include "raw_answer"
            case_sensitive: Whether to perform case-sensitive search
            regex: Whether to treat query as regex pattern

        Returns:
            List of question dictionaries matching the search criteria

        Examples:
            # Simple search (backward compatible)
            results = query_builder.search_questions("machine learning")

            # Multi-term AND search
            results = query_builder.search_questions(["quantum", "mechanics"], match_all=True)

            # Multi-term OR search
            results = query_builder.search_questions(["python", "java"], match_all=False)

            # Search in multiple fields
            results = query_builder.search_questions("algorithm", fields=["question", "raw_answer"])

            # Case-sensitive search
            results = query_builder.search_questions("Python", case_sensitive=True)

            # Regex search
            results = query_builder.search_questions(r"what (is|are)", regex=True)
        """
        # Default fields to search
        if fields is None:
            fields = ["question"]

        # Normalize query to list for uniform handling
        queries = [query] if isinstance(query, str) else query

        # Handle empty query list - return empty results
        if not queries:
            return []

        results = []

        for q_data in self.base._questions_cache.values():
            # Collect text from specified fields
            search_texts = []
            for field in fields:
                if field in q_data:
                    search_texts.append(q_data[field])

            # Combine all field texts
            combined_text = " ".join(str(t) for t in search_texts if t)

            # Check if question matches the search criteria
            if self._matches_search(combined_text, queries, match_all, case_sensitive, regex):
                results.append(q_data)

        return results

    def get_questions_by_author(self, author: str) -> list[dict[str, Any]]:
        """Get questions created by a specific author."""
        return self.filter_questions(author=author)

    def get_questions_with_rubric(self) -> list[dict[str, Any]]:
        """Get questions that have question-specific rubrics."""
        return self.filter_questions(has_rubric=True)

    def count_by_field(
        self,
        field_path: str,
        questions: list[dict[str, Any]] | None = None,
    ) -> dict[Any, int]:
        """
        Count questions grouped by a field value using dot notation.

        Args:
            field_path: Dot-notation path to field (e.g., "custom_metadata.category", "finished")
            questions: Optional list of questions to count from (defaults to all questions)

        Returns:
            Dictionary mapping field values to their counts

        Examples:
            # Count by custom metadata field
            category_counts = query_builder.count_by_field("custom_metadata.category")
            # Result: {"math": 45, "science": 32, "history": 18}

            # Count finished vs unfinished
            status_counts = query_builder.count_by_field("finished")
            # Result: {True: 67, False: 28}

            # Count on filtered subset
            math_qs = query_builder.filter_by_custom_metadata(category="math")
            difficulty_counts = query_builder.count_by_field("custom_metadata.difficulty", questions=math_qs)
        """
        # Use all questions if not specified
        if questions is None:
            questions = list(self.base._questions_cache.values())

        field_parts = field_path.split(".")
        values = []

        for q_data in questions:
            # Navigate to the field using dot notation
            field_value = self._get_nested_field(q_data, field_parts)
            # Add value to list (including None for missing fields)
            values.append(field_value)

        # Count occurrences
        return dict(Counter(values))

    # ========================================================================
    # Private Helper Methods
    # ========================================================================

    def _check_has_template(self, q_data: dict[str, Any]) -> bool:
        """Check if a question has a non-default template."""
        template = q_data.get("answer_template")
        if not template:
            return False

        # Use the provided template checker if available
        if self._template_checker is not None:
            question_text = q_data.get("question", "")
            return not self._template_checker(template, question_text)

        # Default: assume any template is non-default
        return True

    def _get_nested_field(self, data: dict[str, Any], field_parts: list[str]) -> Any:
        """Navigate to a nested field using dot notation parts.

        Args:
            data: The dictionary to navigate
            field_parts: List of field names representing the path

        Returns:
            The value at the specified path, or None if not found
        """
        field_value: Any = data
        try:
            for part in field_parts:
                if isinstance(field_value, dict):
                    field_value = field_value.get(part)
                else:
                    return None
        except (KeyError, TypeError, AttributeError):
            return None
        return field_value

    def _matches_value(self, field_value: Any, value: Any, match_mode: str) -> bool:
        """Check if a field value matches the expected value based on match mode.

        Args:
            field_value: The actual field value from the question
            value: The expected value to match against
            match_mode: The matching strategy ("exact", "contains", "in", "regex")

        Returns:
            True if the value matches according to the match mode

        Raises:
            ValueError: If match_mode is not recognized
        """
        if match_mode == "exact":
            return bool(field_value == value)
        elif match_mode == "contains":
            # Substring match (works for strings)
            if isinstance(field_value, str) and isinstance(value, str):
                return value.lower() in field_value.lower()
            return False
        elif match_mode == "in":
            # Check if value is in a list/array field
            if isinstance(field_value, list | tuple):
                return value in field_value
            return False
        elif match_mode == "regex":
            # Regex match (works for strings)
            if isinstance(field_value, str) and isinstance(value, str):
                try:
                    return bool(re.search(value, field_value, re.IGNORECASE))
                except re.error:
                    logger.debug("Invalid regex pattern in filter: %s", value)
                    return False
            return False
        else:
            raise ValueError(f"Invalid match_mode: {match_mode}. Must be 'exact', 'contains', 'in', or 'regex'")

    def _matches_search(
        self,
        combined_text: str,
        queries: list[str],
        match_all: bool,
        case_sensitive: bool,
        regex: bool,
    ) -> bool:
        """Check if combined text matches search queries.

        Args:
            combined_text: The text to search in
            queries: List of search terms
            match_all: If True, all terms must match (AND). If False, any term matches (OR)
            case_sensitive: Whether to perform case-sensitive search
            regex: Whether to treat queries as regex patterns

        Returns:
            True if the text matches according to the search criteria
        """
        # Prepare text for searching
        if not case_sensitive and not regex:
            combined_text = combined_text.lower()

        # Check each query
        matches = []
        for q in queries:
            # Prepare query
            if not case_sensitive and not regex:
                q = q.lower()

            # Perform search
            if regex:
                try:
                    flags = 0 if case_sensitive else re.IGNORECASE
                    match = bool(re.search(q, combined_text, flags))
                except re.error:
                    logger.debug("Invalid regex pattern in search: %s", q)
                    match = False
            else:
                match = q in combined_text

            matches.append(match)

        # Apply AND/OR logic for multi-term searches
        if match_all:
            return all(matches)
        else:
            return any(matches)
