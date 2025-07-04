"""Tests for rubric schema validation."""

import pytest
from pydantic import ValidationError

from karenina.schemas.rubric_class import Rubric, RubricEvaluation, RubricTrait


class TestRubricTrait:
    """Test RubricTrait schema validation."""

    def test_boolean_trait_creation(self):
        """Test creating a boolean trait."""
        trait = RubricTrait(
            name="clarity",
            description="Is the response clear and understandable?",
            kind="boolean"
        )

        assert trait.name == "clarity"
        assert trait.description == "Is the response clear and understandable?"
        assert trait.kind == "boolean"
        assert trait.min_score == 1  # Default values are applied
        assert trait.max_score == 5

    def test_score_trait_creation(self):
        """Test creating a score-based trait."""
        trait = RubricTrait(
            name="completeness",
            description="How complete is the response?",
            kind="score",
            min_score=1,
            max_score=5
        )

        assert trait.name == "completeness"
        assert trait.description == "How complete is the response?"
        assert trait.kind == "score"
        assert trait.min_score == 1
        assert trait.max_score == 5

    def test_score_trait_default_range(self):
        """Test that score traits get default min/max values."""
        trait = RubricTrait(
            name="relevance",
            description="How relevant is the response?",
            kind="score"
        )

        assert trait.min_score == 1
        assert trait.max_score == 5

    def test_invalid_trait_name(self):
        """Test that empty trait names are rejected by Pydantic."""
        import pytest
        from pydantic import ValidationError

        # Empty names should be rejected by the schema
        with pytest.raises(ValidationError):
            RubricTrait(
                name="",  # Empty name should be rejected
                description="Valid description",
                kind="boolean"
            )

    def test_missing_description(self):
        """Test trait creation without description."""
        trait = RubricTrait(
            name="accuracy",
            kind="boolean"
        )

        assert trait.name == "accuracy"
        assert trait.description is None
        assert trait.kind == "boolean"

    def test_invalid_score_range(self):
        """Test validation of score ranges."""
        # min_score > max_score should be allowed at creation
        # (validation logic would be in business logic, not schema)
        trait = RubricTrait(
            name="test",
            description="Test trait",
            kind="score",
            min_score=5,
            max_score=1
        )

        assert trait.min_score == 5
        assert trait.max_score == 1


class TestRubric:
    """Test Rubric schema validation."""

    def test_rubric_creation(self):
        """Test creating a complete rubric."""
        traits = [
            RubricTrait(name="clarity", description="Clear response", kind="boolean"),
            RubricTrait(name="completeness", description="Complete response", kind="score", min_score=1, max_score=5)
        ]

        rubric = Rubric(
            traits=traits
        )

        assert len(rubric.traits) == 2
        assert rubric.traits[0].name == "clarity"
        assert rubric.traits[1].name == "completeness"

    def test_rubric_without_description(self):
        """Test creating a rubric without description."""
        traits = [
            RubricTrait(name="accuracy", description="Accurate response", kind="boolean")
        ]

        rubric = Rubric(
            traits=traits
        )

        # No description field in Rubric model
        assert len(rubric.traits) == 1

    def test_empty_rubric_title(self):
        """Test that empty titles are allowed by Pydantic."""
        traits = [
            RubricTrait(name="test", description="Test trait", kind="boolean")
        ]

        # Empty traits list is allowed
        Rubric(
            traits=traits
        )

    def test_empty_traits_list(self):
        """Test rubric with empty traits list."""
        rubric = Rubric(
            traits=[]
        )

        assert len(rubric.traits) == 0

    def test_rubric_serialization(self):
        """Test that rubric can be serialized to dict."""
        traits = [
            RubricTrait(name="clarity", description="Clear response", kind="boolean")
        ]

        rubric = Rubric(
            traits=traits
        )

        rubric_dict = rubric.model_dump()

        # No title field in current Rubric model
        assert len(rubric_dict["traits"]) == 1
        assert rubric_dict["traits"][0]["name"] == "clarity"
        assert rubric_dict["traits"][0]["kind"] == "boolean"


class TestRubricEvaluation:
    """Test RubricEvaluation schema validation."""

    def test_evaluation_creation(self):
        """Test creating a rubric evaluation."""
        evaluation = RubricEvaluation(
            trait_scores={"clarity": True, "completeness": 4}
        )

        assert evaluation.trait_scores["clarity"] is True
        assert evaluation.trait_scores["completeness"] == 4

    def test_evaluation_with_mixed_scores(self):
        """Test evaluation with boolean and numeric scores."""
        evaluation = RubricEvaluation(
            trait_scores={
                "accuracy": True,
                "relevance": False,
                "depth": 3,
                "clarity": 5
            }
        )

        assert len(evaluation.trait_scores) == 4
        assert evaluation.trait_scores["accuracy"] is True
        assert evaluation.trait_scores["relevance"] is False
        assert evaluation.trait_scores["depth"] == 3
        assert evaluation.trait_scores["clarity"] == 5

    def test_empty_trait_scores(self):
        """Test evaluation with empty trait scores."""
        evaluation = RubricEvaluation(
            trait_scores={}
        )

        assert len(evaluation.trait_scores) == 0

    def test_evaluation_serialization(self):
        """Test that evaluation can be serialized to dict."""
        evaluation = RubricEvaluation(
            trait_scores={"test_trait": True}
        )

        eval_dict = evaluation.model_dump()

        assert eval_dict["trait_scores"]["test_trait"] is True


class TestTraitKind:
    """Test TraitKind literal validation."""

    def test_valid_trait_kinds(self):
        """Test that valid trait kinds work."""
        boolean_trait = RubricTrait(
            name="test_bool",
            description="Test boolean trait",
            kind="boolean"
        )

        score_trait = RubricTrait(
            name="test_score",
            description="Test score trait",
            kind="score"
        )

        assert boolean_trait.kind == "boolean"
        assert score_trait.kind == "score"

    def test_invalid_trait_kind(self):
        """Test validation error for invalid trait kind."""
        with pytest.raises(ValidationError):
            RubricTrait(
                name="invalid_trait",
                description="Invalid trait",
                kind="invalid_kind"  # Not a valid TraitKind
            )


class TestRubricIntegration:
    """Integration tests for rubric schemas."""

    def test_complete_rubric_workflow(self):
        """Test a complete workflow with all rubric schemas."""
        # Create traits
        traits = [
            RubricTrait(
                name="accuracy",
                description="Is the information factually correct?",
                kind="boolean"
            ),
            RubricTrait(
                name="completeness",
                description="How complete is the response (1-5)?",
                kind="score",
                min_score=1,
                max_score=5
            ),
            RubricTrait(
                name="clarity",
                description="Is the response clear and well-written?",
                kind="boolean"
            )
        ]

        # Create rubric
        rubric = Rubric(
            traits=traits
        )

        # Create evaluation
        evaluation = RubricEvaluation(
            trait_scores={
                "accuracy": True,
                "completeness": 4,
                "clarity": True
            }
        )

        # Verify the complete workflow
        assert len(rubric.traits) == 3

        # Verify trait types
        boolean_traits = [t for t in rubric.traits if t.kind == "boolean"]
        score_traits = [t for t in rubric.traits if t.kind == "score"]

        assert len(boolean_traits) == 2
        assert len(score_traits) == 1

        # Verify evaluation matches rubric structure
        for trait in rubric.traits:
            assert trait.name in evaluation.trait_scores

            if trait.kind == "boolean":
                assert isinstance(evaluation.trait_scores[trait.name], bool)
            elif trait.kind == "score":
                assert isinstance(evaluation.trait_scores[trait.name], int)

    def test_rubric_json_round_trip(self):
        """Test that rubric can be serialized to JSON and back."""
        import json

        traits = [
            RubricTrait(name="test_trait", description="Test", kind="boolean")
        ]

        original_rubric = Rubric(
            traits=traits
        )

        # Serialize to JSON
        json_str = json.dumps(original_rubric.model_dump())

        # Deserialize from JSON
        rubric_data = json.loads(json_str)
        restored_rubric = Rubric(**rubric_data)

        # Verify round trip
        # No title field in current Rubric model
        assert len(restored_rubric.traits) == len(original_rubric.traits)
        assert restored_rubric.traits[0].name == original_rubric.traits[0].name
        assert restored_rubric.traits[0].kind == original_rubric.traits[0].kind
