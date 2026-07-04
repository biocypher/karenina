"""Tests for dynamic rubric structured output schemas."""

import pytest

from karenina.schemas.outputs.rubric import ConceptPresenceItem, ConceptPresenceResult


@pytest.mark.unit
class TestConceptPresenceItem:
    """Tests for ConceptPresenceItem construction."""

    def test_construction_present_true(self):
        item = ConceptPresenceItem(trait_name="clarity", present=True)
        assert item.trait_name == "clarity"
        assert item.present is True

    def test_construction_present_false(self):
        item = ConceptPresenceItem(trait_name="specificity", present=False)
        assert item.trait_name == "specificity"
        assert item.present is False


@pytest.mark.unit
class TestConceptPresenceResult:
    """Tests for ConceptPresenceResult construction and to_dict."""

    def test_to_dict_multiple_items(self):
        result = ConceptPresenceResult(
            results=[
                ConceptPresenceItem(trait_name="clarity", present=True),
                ConceptPresenceItem(trait_name="specificity", present=False),
                ConceptPresenceItem(trait_name="completeness", present=True),
            ]
        )
        expected = {
            "clarity": True,
            "specificity": False,
            "completeness": True,
        }
        assert result.to_dict() == expected

    def test_to_dict_empty_results(self):
        result = ConceptPresenceResult(results=[])
        assert result.to_dict() == {}

    def test_model_json_schema_has_results_key(self):
        schema = ConceptPresenceResult.model_json_schema()
        assert "results" in schema["properties"]
