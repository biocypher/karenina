"""Unit tests for rubric serializer key consistency fix (Issue 042).

Verifies that serialize_rubric_to_dict() uses the "llm_traits" key (not "traits")
and that deserialization supports both the new key and the old key for backward
compatibility.
"""

import pytest

from karenina.schemas.entities.rubric import LLMRubricTrait
from karenina.storage.rubric_serializer import (
    deserialize_rubric_from_dict,
    serialize_rubric_to_dict,
)


def _make_llm_trait(name: str = "clarity") -> LLMRubricTrait:
    """Create a minimal LLMRubricTrait for testing."""
    return LLMRubricTrait(
        name=name,
        description="Evaluates response clarity",
        kind="boolean",
        higher_is_better=True,
    )


@pytest.mark.unit
class TestRubricSerializerLLMTraitKey:
    """Tests for consistent llm_traits key in rubric serialization."""

    def test_serialize_uses_llm_traits_key(self) -> None:
        """Serialized output must use 'llm_traits', not 'traits'."""
        trait = _make_llm_trait()
        result = serialize_rubric_to_dict(llm_traits=[trait])

        assert result is not None
        assert "llm_traits" in result, "Expected 'llm_traits' key in serialized output"
        assert "traits" not in result, "Legacy 'traits' key should not appear in new output"
        assert len(result["llm_traits"]) == 1
        assert result["llm_traits"][0]["name"] == "clarity"

    def test_deserialize_reads_llm_traits_key(self) -> None:
        """Deserialization must read traits from 'llm_traits' key."""
        rubric_data = {
            "llm_traits": [
                {
                    "name": "precision",
                    "description": "Evaluates precision",
                    "kind": "boolean",
                    "higher_is_better": True,
                }
            ],
        }
        rubric = deserialize_rubric_from_dict(rubric_data)

        assert rubric is not None
        assert len(rubric.llm_traits) == 1
        assert rubric.llm_traits[0].name == "precision"

    def test_deserialize_backward_compat_old_key(self) -> None:
        """Deserialization must still read from the legacy 'traits' key."""
        rubric_data = {
            "traits": [
                {
                    "name": "coherence",
                    "description": "Evaluates coherence",
                    "kind": "score",
                    "higher_is_better": True,
                    "min_score": 1,
                    "max_score": 5,
                }
            ],
        }
        rubric = deserialize_rubric_from_dict(rubric_data)

        assert rubric is not None
        assert len(rubric.llm_traits) == 1
        assert rubric.llm_traits[0].name == "coherence"
        assert rubric.llm_traits[0].kind == "score"

    def test_round_trip_with_new_key(self) -> None:
        """Serialize then deserialize: LLM traits must survive the round trip."""
        original_trait = _make_llm_trait("thoroughness")
        serialized = serialize_rubric_to_dict(llm_traits=[original_trait])

        assert serialized is not None
        rubric = deserialize_rubric_from_dict(serialized)

        assert rubric is not None
        assert len(rubric.llm_traits) == 1
        restored = rubric.llm_traits[0]
        assert restored.name == "thoroughness"
        assert restored.kind == "boolean"
        assert restored.higher_is_better is True
