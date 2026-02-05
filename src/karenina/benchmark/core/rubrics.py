"""Rubric management functionality for benchmarks."""

from datetime import datetime
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .base import BenchmarkBase

from karenina.schemas.entities import CallableTrait, LLMRubricTrait, MetricRubricTrait, RegexTrait, Rubric
from karenina.utils.checkpoint import (
    add_global_rubric_to_benchmark,
    extract_global_rubric_from_benchmark,
)


class RubricManager:
    """Manager for global and question-specific rubric operations."""

    def __init__(self, base: "BenchmarkBase") -> None:
        """Initialize with reference to benchmark base."""
        self.base = base

    def add_global_rubric_trait(self, trait: LLMRubricTrait | RegexTrait | CallableTrait | MetricRubricTrait) -> None:
        """
        Add a global rubric trait to the benchmark.

        Args:
            trait: The rubric trait to add (LLM, regex, callable, or metric)
        """
        current_traits = extract_global_rubric_from_benchmark(self.base._checkpoint) or []
        current_traits.append(trait)
        add_global_rubric_to_benchmark(self.base._checkpoint, current_traits)

    def add_question_rubric_trait(
        self, question_id: str, trait: LLMRubricTrait | RegexTrait | CallableTrait | MetricRubricTrait
    ) -> None:
        """
        Add a question-specific rubric trait.

        Args:
            question_id: The question ID
            trait: The rubric trait to add

        Raises:
            ValueError: If question not found
        """
        from karenina.utils.checkpoint import convert_rubric_trait_to_rating

        # Find the question
        found = False
        for item in self.base._checkpoint.dataFeedElement:
            if self.base._get_item_id(item) == question_id:
                rating = convert_rubric_trait_to_rating(trait, "question-specific")
                if item.item.rating is None:
                    item.item.rating = []
                item.item.rating.append(rating)
                item.dateModified = datetime.now().isoformat()
                found = True
                break

        if not found:
            raise ValueError(f"Question not found: {question_id}")

        # Update cache
        self.base._rebuild_cache()

    def get_global_rubric(self) -> Rubric | None:
        """
        Get the global rubric from the benchmark.

        Returns:
            Rubric object or None if no global rubric
        """
        traits = extract_global_rubric_from_benchmark(self.base._checkpoint)
        if traits:
            # Separate traits by type
            llm_traits = [t for t in traits if isinstance(t, LLMRubricTrait)]
            regex_traits = [t for t in traits if isinstance(t, RegexTrait)]
            callable_traits = [t for t in traits if isinstance(t, CallableTrait)]
            metric_traits = [t for t in traits if isinstance(t, MetricRubricTrait)]
            return Rubric(
                llm_traits=llm_traits,
                regex_traits=regex_traits,
                callable_traits=callable_traits,
                metric_traits=metric_traits,
            )
        return None

    def get_question_rubric(
        self, question_id: str
    ) -> list[LLMRubricTrait | RegexTrait | CallableTrait | MetricRubricTrait] | None:
        """
        Get question-specific rubric traits.

        Args:
            question_id: The question ID

        Returns:
            List of rubric traits or None if no question rubric
        """
        if question_id in self.base._questions_cache:
            return self.base._questions_cache[question_id].get("question_rubric")
        return None

    def get_merged_rubric_for_question(self, question_id: str) -> Rubric | None:
        """
        Get merged rubric for a question (global + question-specific traits).

        Args:
            question_id: The question ID

        Returns:
            Merged rubric with both global and question-specific traits, or None if no rubrics
        """
        # Get global rubric traits
        global_traits = extract_global_rubric_from_benchmark(self.base._checkpoint) or []

        # Get question-specific rubric traits
        question_traits: list[LLMRubricTrait | RegexTrait | CallableTrait | MetricRubricTrait] = []
        if question_id in self.base._questions_cache:
            q_data = self.base._questions_cache[question_id]
            question_rubric = q_data.get("question_rubric")
            if question_rubric:
                # question_rubric is stored as a dict with llm_traits, regex_traits, etc.
                if isinstance(question_rubric, dict):
                    question_traits.extend(question_rubric.get("llm_traits", []))
                    question_traits.extend(question_rubric.get("regex_traits", []))
                    question_traits.extend(question_rubric.get("callable_traits", []))
                    question_traits.extend(question_rubric.get("metric_traits", []))
                elif isinstance(question_rubric, list):
                    # Backwards compatibility if it's already a flat list
                    question_traits = question_rubric

        # Merge traits (question-specific traits override global ones with same name)
        merged_traits = list(global_traits)  # Start with global traits

        # Add question-specific traits, replacing any with the same name
        for q_trait in question_traits:
            # Remove any global trait with the same name
            merged_traits = [t for t in merged_traits if t.name != q_trait.name]
            # Add the question-specific trait
            merged_traits.append(q_trait)

        # Return merged rubric if we have any traits
        if merged_traits:
            # Separate traits by type
            llm_traits = [t for t in merged_traits if isinstance(t, LLMRubricTrait)]
            regex_traits = [t for t in merged_traits if isinstance(t, RegexTrait)]
            callable_traits = [t for t in merged_traits if isinstance(t, CallableTrait)]
            metric_traits = [t for t in merged_traits if isinstance(t, MetricRubricTrait)]
            return Rubric(
                llm_traits=llm_traits,
                regex_traits=regex_traits,
                callable_traits=callable_traits,
                metric_traits=metric_traits,
            )

        return None

    def clear_global_rubric(self) -> bool:
        """
        Remove the global rubric.

        Returns:
            True if global rubric was removed, False if none existed
        """
        if self.base._checkpoint.rating:
            self.base._checkpoint.rating = None
            self.base._checkpoint.dateModified = datetime.now().isoformat()
            return True
        return False

    def remove_question_rubric(self, question_id: str) -> bool:
        """
        Remove question-specific rubric.

        Args:
            question_id: The question ID

        Returns:
            True if rubric was removed, False if not found
        """
        for item in self.base._checkpoint.dataFeedElement:
            if self.base._get_item_id(item) == question_id and item.item.rating:
                item.item.rating = None
                item.dateModified = datetime.now().isoformat()
                self.base._checkpoint.dateModified = datetime.now().isoformat()
                self.base._rebuild_cache()
                return True
        return False

    def clear_all_rubrics(self) -> int:
        """
        Remove all rubrics (global and question-specific).

        Returns:
            Number of rubrics that were removed
        """
        count = 0

        # Clear global rubric
        if self.clear_global_rubric():
            count += 1

        # Clear question-specific rubrics
        for item in self.base._checkpoint.dataFeedElement:
            if item.item.rating:
                item.item.rating = None
                item.dateModified = datetime.now().isoformat()
                count += 1

        if count > 0:
            self.base._checkpoint.dateModified = datetime.now().isoformat()
            self.base._rebuild_cache()

        return count

    def validate_rubrics(self) -> tuple[bool, list[str]]:
        """
        Validate all rubrics are properly configured.

        Returns:
            Tuple of (all_valid, list_of_errors)
        """
        errors = []

        # Check global rubric
        global_rubric = self.get_global_rubric()
        if global_rubric:
            # Validate LLM traits
            for trait in global_rubric.llm_traits:
                if not trait.name or not trait.description:
                    errors.append("Global LLM rubric trait missing name or description")
                if trait.kind == "score" and (trait.min_score is None or trait.max_score is None):
                    errors.append(f"Score trait '{trait.name}' missing min/max scores")
            # Validate regex traits
            for regex_trait in global_rubric.regex_traits:
                if not regex_trait.name or not regex_trait.description:
                    errors.append("Global regex rubric trait missing name or description")
                if not regex_trait.pattern:
                    errors.append(f"Regex trait '{regex_trait.name}' missing pattern")
            # Validate callable traits
            for callable_trait in global_rubric.callable_traits:
                if not callable_trait.name or not callable_trait.description:
                    errors.append("Global callable rubric trait missing name or description")
            # Validate metric traits
            for metric_trait in global_rubric.metric_traits:
                if not metric_trait.name or not metric_trait.description:
                    errors.append("Global metric rubric trait missing name or description")
                if not metric_trait.metrics:
                    errors.append(f"Metric trait '{metric_trait.name}' has no metrics defined")

        # Check question-specific rubrics
        for q_id, q_data in self.base._questions_cache.items():
            if q_data.get("question_rubric"):
                for trait in q_data["question_rubric"]:
                    if not trait.name or not trait.description:
                        errors.append(f"Question {q_id} rubric trait missing name or description")
                    # Check score fields for LLM traits
                    if (
                        isinstance(trait, LLMRubricTrait)
                        and trait.kind == "score"
                        and (trait.min_score is None or trait.max_score is None)
                    ):
                        errors.append(f"Question {q_id} score trait '{trait.name}' missing min/max scores")
                    # Check regex traits
                    if isinstance(trait, RegexTrait) and not trait.pattern:
                        errors.append(f"Question {q_id} regex trait '{trait.name}' missing pattern")
                    # Check metric traits
                    if isinstance(trait, MetricRubricTrait) and not trait.metrics:
                        errors.append(f"Question {q_id} metric trait '{trait.name}' has no metrics defined")

        return len(errors) == 0, errors

    def get_rubric_statistics(self) -> dict[str, Any]:
        """Get statistics about rubrics in the benchmark."""
        global_rubric = self.get_global_rubric()
        global_traits_count = (
            (
                len(global_rubric.llm_traits)
                + len(global_rubric.regex_traits)
                + len(global_rubric.callable_traits)
                + len(global_rubric.metric_traits)
            )
            if global_rubric
            else 0
        )

        question_rubrics_count = sum(
            1 for q_data in self.base._questions_cache.values() if q_data.get("question_rubric")
        )

        total_question_traits = sum(
            len(q_data.get("question_rubric", []))
            for q_data in self.base._questions_cache.values()
            if q_data.get("question_rubric")
        )

        return {
            "has_global_rubric": global_rubric is not None,
            "global_traits_count": global_traits_count,
            "questions_with_rubrics": question_rubrics_count,
            "total_question_traits": total_question_traits,
            "total_traits": global_traits_count + total_question_traits,
        }

    def get_questions_with_rubric(self) -> list[str]:
        """Get list of question IDs that have question-specific rubrics."""
        return [q_id for q_id, q_data in self.base._questions_cache.items() if q_data.get("question_rubric")]

    def set_global_rubric(self, traits: list[LLMRubricTrait | RegexTrait | CallableTrait | MetricRubricTrait]) -> None:
        """
        Set the global rubric with a list of traits.

        Args:
            traits: List of rubric traits (LLM, regex, callable, or metric-based)
        """
        add_global_rubric_to_benchmark(self.base._checkpoint, traits)

    def update_global_rubric_trait(
        self, trait_name: str, updated_trait: LLMRubricTrait | RegexTrait | CallableTrait | MetricRubricTrait
    ) -> bool:
        """
        Update a specific trait in the global rubric.

        Args:
            trait_name: Name of the trait to update
            updated_trait: The updated trait

        Returns:
            True if trait was found and updated, False otherwise
        """
        current_traits = extract_global_rubric_from_benchmark(self.base._checkpoint) or []

        for i, trait in enumerate(current_traits):
            if trait.name == trait_name:
                current_traits[i] = updated_trait
                add_global_rubric_to_benchmark(self.base._checkpoint, current_traits)
                return True

        return False

    def remove_global_rubric_trait(self, trait_name: str) -> bool:
        """
        Remove a specific trait from the global rubric.

        Args:
            trait_name: Name of the trait to remove

        Returns:
            True if trait was found and removed, False otherwise
        """
        current_traits = extract_global_rubric_from_benchmark(self.base._checkpoint) or []

        original_count = len(current_traits)
        current_traits = [trait for trait in current_traits if trait.name != trait_name]

        if len(current_traits) < original_count:
            add_global_rubric_to_benchmark(self.base._checkpoint, current_traits)
            return True

        return False

    def get_rubric_trait_names(self, question_id: str | None = None) -> list[str]:
        """
        Get all rubric trait names for a question or globally.

        Args:
            question_id: Optional question ID. If None, returns global trait names.

        Returns:
            List of trait names
        """
        if question_id is None:
            # Get global trait names
            global_rubric = self.get_global_rubric()
            if global_rubric:
                trait_names = [trait.name for trait in global_rubric.llm_traits]
                trait_names.extend([trait.name for trait in global_rubric.regex_traits])
                trait_names.extend([trait.name for trait in global_rubric.callable_traits])
                trait_names.extend([trait.name for trait in global_rubric.metric_traits])
                return trait_names
            return []
        else:
            # Get merged trait names for the question
            merged_rubric = self.get_merged_rubric_for_question(question_id)
            if merged_rubric:
                trait_names = [trait.name for trait in merged_rubric.llm_traits]
                trait_names.extend([trait.name for trait in merged_rubric.regex_traits])
                trait_names.extend([trait.name for trait in merged_rubric.callable_traits])
                trait_names.extend([trait.name for trait in merged_rubric.metric_traits])
                return trait_names
            return []

    def has_rubric(self, question_id: str | None = None) -> bool:
        """
        Check if a question or the benchmark has rubrics.

        Args:
            question_id: Optional question ID. If None, checks for global rubric.

        Returns:
            True if rubrics exist, False otherwise
        """
        if question_id is None:
            return self.get_global_rubric() is not None
        else:
            merged_rubric = self.get_merged_rubric_for_question(question_id)
            return merged_rubric is not None
