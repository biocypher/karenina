"""Rubric management functionality for benchmarks."""

from datetime import datetime
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .base import BenchmarkBase

from ...schemas.domain import ManualRubricTrait, MetricRubricTrait, Rubric, RubricTrait
from ...utils.checkpoint import (
    add_global_rubric_to_benchmark,
    extract_global_rubric_from_benchmark,
)


class RubricManager:
    """Manager for global and question-specific rubric operations."""

    def __init__(self, base: "BenchmarkBase") -> None:
        """Initialize with reference to benchmark base."""
        self.base = base

    def add_global_rubric_trait(self, trait: RubricTrait | ManualRubricTrait) -> None:
        """
        Add a global rubric trait to the benchmark.

        Args:
            trait: The rubric trait to add
        """
        current_traits = extract_global_rubric_from_benchmark(self.base._checkpoint) or []
        current_traits.append(trait)
        add_global_rubric_to_benchmark(self.base._checkpoint, current_traits)

    def add_question_rubric_trait(self, question_id: str, trait: RubricTrait | ManualRubricTrait) -> None:
        """
        Add a question-specific rubric trait.

        Args:
            question_id: The question ID
            trait: The rubric trait to add

        Raises:
            ValueError: If question not found
        """
        from ...utils.checkpoint import convert_rubric_trait_to_rating

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
            rubric_traits = [t for t in traits if isinstance(t, RubricTrait)]
            manual_traits = [t for t in traits if isinstance(t, ManualRubricTrait)]
            return Rubric(traits=rubric_traits, manual_traits=manual_traits)
        return None

    def get_question_rubric(self, question_id: str) -> list[RubricTrait | ManualRubricTrait] | None:
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
        question_traits = []
        if question_id in self.base._questions_cache:
            q_data = self.base._questions_cache[question_id]
            question_rubric = q_data.get("question_rubric")
            if question_rubric:
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
            rubric_traits = [t for t in merged_traits if isinstance(t, RubricTrait)]
            manual_traits = [t for t in merged_traits if isinstance(t, ManualRubricTrait)]
            return Rubric(traits=rubric_traits, manual_traits=manual_traits)

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
            for trait in global_rubric.traits:
                if not trait.name or not trait.description:
                    errors.append("Global rubric trait missing name or description")
                if trait.kind == "score" and (trait.min_score is None or trait.max_score is None):
                    errors.append(f"Score trait '{trait.name}' missing min/max scores")
            for manual_trait in global_rubric.manual_traits:
                if not manual_trait.name or not manual_trait.description:
                    errors.append("Global manual rubric trait missing name or description")

        # Check question-specific rubrics
        for q_id, q_data in self.base._questions_cache.items():
            if q_data.get("question_rubric"):
                for trait in q_data["question_rubric"]:
                    if not trait.name or not trait.description:
                        errors.append(f"Question {q_id} rubric trait missing name or description")
                    # Only check score fields for RubricTrait (not ManualRubricTrait)
                    if (
                        isinstance(trait, RubricTrait)
                        and trait.kind == "score"
                        and (trait.min_score is None or trait.max_score is None)
                    ):
                        errors.append(f"Question {q_id} score trait '{trait.name}' missing min/max scores")

        return len(errors) == 0, errors

    def get_rubric_statistics(self) -> dict[str, Any]:
        """Get statistics about rubrics in the benchmark."""
        global_rubric = self.get_global_rubric()
        global_traits_count = (len(global_rubric.traits) + len(global_rubric.manual_traits)) if global_rubric else 0

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

    def set_global_rubric(self, traits: list[RubricTrait | ManualRubricTrait | MetricRubricTrait]) -> None:
        """
        Set the global rubric with a list of traits.

        Args:
            traits: List of rubric traits (LLM, manual, or metric-based)
        """
        add_global_rubric_to_benchmark(self.base._checkpoint, traits)

    def update_global_rubric_trait(self, trait_name: str, updated_trait: RubricTrait | ManualRubricTrait) -> bool:
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
                trait_names = [trait.name for trait in global_rubric.traits]
                trait_names.extend([trait.name for trait in global_rubric.manual_traits])
                return trait_names
            return []
        else:
            # Get merged trait names for the question
            merged_rubric = self.get_merged_rubric_for_question(question_id)
            if merged_rubric:
                trait_names = [trait.name for trait in merged_rubric.traits]
                trait_names.extend([trait.name for trait in merged_rubric.manual_traits])
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
