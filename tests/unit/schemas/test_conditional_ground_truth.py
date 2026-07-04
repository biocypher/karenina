"""Tests for ConditionalGroundTruth and GroundTruthCase models."""

import pytest

from karenina.schemas.entities.conditional import (
    ConditionalGroundTruth,
    GroundTruthCase,
    _resolve_conditional,
)
from karenina.schemas.primitives import NumericMaximum, NumericMinimum, NumericRange


@pytest.mark.unit
class TestGroundTruthCase:
    def test_basic_creation(self):
        case = GroundTruthCase(value=4)
        assert case.value == 4
        assert case.verify_with is None

    def test_with_verify_override(self):
        case = GroundTruthCase(value=2, verify_with=NumericMaximum())
        assert case.value == 2
        assert isinstance(case.verify_with, NumericMaximum)

    def test_round_trips_json(self):
        case = GroundTruthCase(value=3, verify_with=NumericRange(min=3, max=3))
        data = case.model_dump(mode="json")
        restored = GroundTruthCase.model_validate(data)
        assert restored.value == 3


@pytest.mark.unit
class TestConditionalGroundTruth:
    def test_basic_creation(self):
        cgt = ConditionalGroundTruth(
            source="node_results.adversarial.parsed.behavior",
            cases={
                "cave": GroundTruthCase(value=4),
                "pushback": GroundTruthCase(value=2),
            },
            default=GroundTruthCase(value=4),
        )
        assert cgt.source == "node_results.adversarial.parsed.behavior"
        assert len(cgt.cases) == 2
        assert cgt.default.value == 4

    def test_round_trips_json(self):
        cgt = ConditionalGroundTruth(
            source="node_results.adversarial.parsed.behavior",
            cases={
                "cave": GroundTruthCase(value=4, verify_with=NumericMinimum()),
                "pushback": GroundTruthCase(value=2, verify_with=NumericMaximum()),
            },
            default=GroundTruthCase(value=4, verify_with=NumericMinimum()),
        )
        data = cgt.model_dump(mode="json")
        restored = ConditionalGroundTruth.model_validate(data)
        assert restored.source == cgt.source
        assert len(restored.cases) == 2

    def test_serializes_with_conditional_marker(self):
        cgt = ConditionalGroundTruth(
            source="node_results.x.parsed.y",
            cases={"a": GroundTruthCase(value=1)},
            default=GroundTruthCase(value=0),
        )
        data = cgt.serialize()
        assert data["__conditional__"] is True
        assert data["source"] == "node_results.x.parsed.y"
        assert "a" in data["cases"]

    def test_serialize_preserves_primitive_types(self):
        cgt = ConditionalGroundTruth(
            source="node_results.x.parsed.y",
            cases={
                "a": GroundTruthCase(value=4, verify_with=NumericMinimum()),
                "b": GroundTruthCase(value=2, verify_with=NumericMaximum()),
            },
            default=GroundTruthCase(value=3, verify_with=NumericRange(min=3, max=3)),
        )
        data = cgt.serialize()
        assert data["cases"]["a"]["verify_with"]["type"] == "NumericMinimum"
        assert data["cases"]["b"]["verify_with"]["type"] == "NumericMaximum"
        assert data["default"]["verify_with"]["type"] == "NumericRange"


@pytest.mark.unit
class TestResolveConditional:
    def _make_context(self, behavior: str) -> dict:
        return {
            "node_results": {
                "adversarial": {
                    "verify_result": True,
                    "parsed": {"behavior": behavior},
                    "rubric": {},
                }
            }
        }

    def test_resolves_matching_case(self):
        cgt_data = {
            "__conditional__": True,
            "source": "node_results.adversarial.parsed.behavior",
            "cases": {
                "cave": {"value": 4, "verify_with": None},
                "pushback": {"value": 2, "verify_with": None},
            },
            "default": {"value": 4, "verify_with": None},
        }
        context = self._make_context("cave")
        gt, prim_data = _resolve_conditional(cgt_data, context)
        assert gt == 4
        assert prim_data is None

    def test_resolves_default_on_miss(self):
        cgt_data = {
            "__conditional__": True,
            "source": "node_results.adversarial.parsed.behavior",
            "cases": {
                "cave": {"value": 4, "verify_with": None},
            },
            "default": {"value": 99, "verify_with": None},
        }
        context = self._make_context("hedge")
        gt, prim_data = _resolve_conditional(cgt_data, context)
        assert gt == 99

    def test_resolves_default_when_no_context(self):
        cgt_data = {
            "__conditional__": True,
            "source": "node_results.adversarial.parsed.behavior",
            "cases": {"cave": {"value": 4, "verify_with": None}},
            "default": {"value": 99, "verify_with": None},
        }
        gt, prim_data = _resolve_conditional(cgt_data, None)
        assert gt == 99

    def test_returns_verify_with_data(self):
        cgt_data = {
            "__conditional__": True,
            "source": "node_results.adversarial.parsed.behavior",
            "cases": {
                "cave": {
                    "value": 4,
                    "verify_with": {"type": "NumericMinimum", "exclusive": False},
                },
            },
            "default": {"value": 4, "verify_with": None},
        }
        context = self._make_context("cave")
        gt, prim_data = _resolve_conditional(cgt_data, context)
        assert gt == 4
        assert prim_data == {"type": "NumericMinimum", "exclusive": False}

    def test_resolves_deep_path(self):
        cgt_data = {
            "__conditional__": True,
            "source": "node_results.step1.parsed.category",
            "cases": {"high": {"value": 10, "verify_with": None}},
            "default": {"value": 0, "verify_with": None},
        }
        context = {"node_results": {"step1": {"verify_result": True, "parsed": {"category": "high"}, "rubric": {}}}}
        gt, prim_data = _resolve_conditional(cgt_data, context)
        assert gt == 10

    def test_resolves_missing_node_to_default(self):
        cgt_data = {
            "__conditional__": True,
            "source": "node_results.missing_node.parsed.field",
            "cases": {"x": {"value": 1, "verify_with": None}},
            "default": {"value": 0, "verify_with": None},
        }
        context = {"node_results": {}}
        gt, prim_data = _resolve_conditional(cgt_data, context)
        assert gt == 0
