"""Base benchmark management functionality."""

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any

from karenina.schemas.checkpoint import JsonLdCheckpoint
from karenina.utils.checkpoint import (
    create_jsonld_benchmark,
    extract_questions_from_benchmark,
    validate_jsonld_benchmark,
)

logger = logging.getLogger(__name__)


class BenchmarkBase:
    """
    Base class for benchmark management providing core functionality.

    This class handles the fundamental operations like loading, saving,
    validation, and basic property management.
    """

    def __init__(
        self,
        name: str,
        description: str = "",
        version: str = "0.1.0",
        creator: str = "Karenina Benchmarking System",
    ):
        """
        Initialize a new benchmark.

        Args:
            name: Name of the benchmark
            description: Description of the benchmark
            version: Version of the benchmark content
            creator: Creator name or organization
        """
        self._checkpoint = create_jsonld_benchmark(name, description, version, creator)
        self._questions_cache: dict[str, dict[str, Any]] = {}
        self._rebuild_cache()

    @classmethod
    def load(cls, path: Path) -> "BenchmarkBase":
        """
        Load a benchmark from a JSON-LD file.

        Args:
            path: Path to the JSON-LD benchmark file

        Returns:
            A BenchmarkBase instance loaded from the file

        Raises:
            ValueError: If the file is not valid JSON-LD
            FileNotFoundError: If the file doesn't exist
        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Benchmark file not found: {path}")

        with open(path, encoding="utf-8") as f:
            data = json.load(f)

        # Parse into JsonLdCheckpoint model
        try:
            checkpoint_data = JsonLdCheckpoint(**data)
        except Exception as e:
            raise ValueError(f"Invalid JSON-LD benchmark format: {e}") from e

        # Validate structure
        is_valid, error_msg = validate_jsonld_benchmark(checkpoint_data)
        if not is_valid:
            raise ValueError(f"Invalid benchmark: {error_msg}")

        # Create instance and set data
        instance = cls.__new__(cls)
        instance._checkpoint = checkpoint_data
        instance._questions_cache = {}
        instance._rebuild_cache()

        return instance

    def save(self, path: Path, save_deep_judgment_config: bool = False) -> None:
        """
        Save the benchmark to a JSON-LD file.

        Args:
            path: Path where to save the benchmark
            save_deep_judgment_config: If True, include deep judgment configuration
                in LLM rubric traits. If False (default), deep judgment settings
                are stripped before saving. Default is False for backward compatibility.
        """
        from copy import deepcopy

        from karenina.utils.checkpoint import strip_deep_judgment_config_from_checkpoint

        path = Path(path)

        # Ensure .jsonld extension
        if path.suffix not in [".jsonld", ".json"]:
            path = path.with_suffix(".jsonld")

        # Update modified timestamp
        self._checkpoint.dateModified = datetime.now().isoformat()

        # Make a deep copy to avoid modifying in-memory checkpoint
        checkpoint_to_save = deepcopy(self._checkpoint)

        # Strip deep judgment config if not requested
        if not save_deep_judgment_config:
            strip_deep_judgment_config_from_checkpoint(checkpoint_to_save)

        # Convert to dict for JSON serialization
        benchmark_dict = checkpoint_to_save.model_dump(by_alias=True, exclude_none=True)

        # Write to file
        with open(path, "w", encoding="utf-8") as f:
            json.dump(benchmark_dict, f, indent=2, ensure_ascii=False)

    def validate(self) -> tuple[bool, str]:
        """
        Validate the benchmark structure.

        Returns:
            Tuple of (is_valid, error_message)
        """
        return validate_jsonld_benchmark(self._checkpoint)

    def _rebuild_cache(self) -> None:
        """Rebuild the internal questions cache from benchmark data."""
        self._questions_cache = {}
        questions = extract_questions_from_benchmark(self._checkpoint)
        for q in questions:
            self._questions_cache[q["id"]] = q

    def _get_item_id(self, item: Any) -> str:
        """Get the ID for a DataFeedItem."""
        if item.id:
            return str(item.id)
        # Generate from question text if no ID
        from karenina.utils.checkpoint import generate_question_id

        return generate_question_id(item.item.text)

    def _update_question_property(self, question_id: str, property_name: str, value: Any) -> None:
        """Update a property in the underlying JSON-LD structure."""
        # Find the DataFeedItem for this question
        for item in self._checkpoint.dataFeedElement:
            if self._get_item_id(item) == question_id:
                # Update dateModified
                item.dateModified = datetime.now().isoformat()

                # Find or create the property in additionalProperty
                if not hasattr(item.item, "additionalProperty") or item.item.additionalProperty is None:
                    from karenina.schemas.checkpoint import SchemaOrgPropertyValue

                    item.item.additionalProperty = []

                # Find existing property or create new one
                prop_found = False
                for prop in item.item.additionalProperty:
                    if prop.name == property_name:
                        prop.value = value
                        prop_found = True
                        break

                if not prop_found:
                    from karenina.schemas.checkpoint import SchemaOrgPropertyValue

                    new_prop = SchemaOrgPropertyValue(name=property_name, value=value)
                    item.item.additionalProperty.append(new_prop)

                # Update main benchmark dateModified
                self._checkpoint.dateModified = datetime.now().isoformat()
                break

    # Property accessors for common attributes
    @property
    def jsonld_data(self) -> JsonLdCheckpoint:
        """Get the raw JSON-LD benchmark data."""
        return self._checkpoint

    @property
    def name(self) -> str:
        """Get the benchmark name."""
        return self._checkpoint.name

    @name.setter
    def name(self, value: str) -> None:
        """Set the benchmark name."""
        self._checkpoint.name = value
        self._checkpoint.dateModified = datetime.now().isoformat()

    @property
    def description(self) -> str:
        """Get the benchmark description."""
        return self._checkpoint.description or ""

    @description.setter
    def description(self, value: str) -> None:
        """Set the benchmark description."""
        self._checkpoint.description = value
        self._checkpoint.dateModified = datetime.now().isoformat()

    @property
    def version(self) -> str:
        """Get the benchmark version."""
        return self._checkpoint.version or "0.1.0"

    @version.setter
    def version(self, value: str) -> None:
        """Set the benchmark version."""
        self._checkpoint.version = value
        self._checkpoint.dateModified = datetime.now().isoformat()

    @property
    def creator(self) -> str:
        """Get the benchmark creator."""
        creator = self._checkpoint.creator
        if creator is None:
            return "Unknown"
        # Handle both string and SchemaOrgPerson types
        if isinstance(creator, str):
            return creator
        # SchemaOrgPerson - extract name
        return creator.name

    @creator.setter
    def creator(self, value: str) -> None:
        """Set the benchmark creator."""
        self._checkpoint.creator = value
        self._checkpoint.dateModified = datetime.now().isoformat()

    @property
    def id(self) -> str | None:
        """Get the benchmark ID."""
        return self._checkpoint.id

    @id.setter
    def id(self, value: str | None) -> None:
        """Set the benchmark ID."""
        self._checkpoint.id = value
        self._checkpoint.dateModified = datetime.now().isoformat()

    @property
    def created_at(self) -> str:
        """Get the creation timestamp."""
        return self._checkpoint.dateCreated

    @created_at.setter
    def created_at(self, value: str) -> None:
        """Set the creation timestamp."""
        self._checkpoint.dateCreated = value

    @property
    def modified_at(self) -> str:
        """Get the last modification timestamp."""
        return self._checkpoint.dateModified

    @modified_at.setter
    def modified_at(self, value: str) -> None:
        """Set the last modification timestamp."""
        self._checkpoint.dateModified = value

    @property
    def question_count(self) -> int:
        """Get the total number of questions."""
        return len(self._questions_cache)

    @property
    def finished_count(self) -> int:
        """Get the number of finished questions."""
        return sum(1 for q in self._questions_cache.values() if q.get("finished", False))

    @property
    def is_empty(self) -> bool:
        """Check if the benchmark has no questions."""
        return len(self._questions_cache) == 0

    @property
    def is_complete(self) -> bool:
        """Check if all questions have templates and are finished."""
        if self.is_empty:
            return False
        return all(q.get("answer_template") and q.get("finished", False) for q in self._questions_cache.values())

    def set_metadata(self, **metadata: Any) -> None:
        """
        Set benchmark metadata.

        Args:
            **metadata: Metadata fields to update (name, description, version, creator)
        """
        for key, value in metadata.items():
            if hasattr(self._checkpoint, key):
                setattr(self._checkpoint, key, value)
        self._checkpoint.dateModified = datetime.now().isoformat()

    # Magic methods for better usability
    def __str__(self) -> str:
        """Human-readable string representation."""
        progress = self.get_progress()
        return f"Benchmark '{self.name}' ({self.question_count} questions, {progress:.1f}% complete)"

    def __repr__(self) -> str:
        """Developer-friendly representation."""
        return (
            f"Benchmark(name='{self.name}', "
            f"version='{self.version}', "
            f"questions={self.question_count}, "
            f"finished={self.finished_count})"
        )

    def __len__(self) -> int:
        """Return the number of questions in the benchmark."""
        return len(self._questions_cache)

    def __contains__(self, question_id: str) -> bool:
        """Check if a question ID exists in the benchmark."""
        return question_id in self._questions_cache

    def __getitem__(self, question_id: str) -> dict[str, Any]:
        """Get a question by ID using bracket notation."""
        if question_id not in self._questions_cache:
            raise ValueError(f"Question not found: {question_id}")
        return self._questions_cache[question_id]

    def __eq__(self, other: object) -> bool:
        """Compare two benchmarks for equality."""
        if not isinstance(other, BenchmarkBase):
            return NotImplemented

        # Compare basic metadata
        if self.name != other.name or self.version != other.version or self.question_count != other.question_count:
            return False

        # Compare questions
        return self._questions_cache == other._questions_cache

    def get_progress(self) -> float:
        """Get completion progress as percentage (0-100)."""
        if self.question_count == 0:
            return 0.0
        return (self.finished_count / self.question_count) * 100
