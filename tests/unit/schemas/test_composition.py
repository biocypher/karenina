"""Tests for composition strategy nodes."""

import pytest

from karenina.schemas.entities.composition import (
    AllOf,
    AnyOf,
    AtLeastN,
    FieldCheck,
    evaluate_strategy,
)


@pytest.mark.unit
class TestFieldCheck:
    """Test FieldCheck leaf node."""

    def test_passing_field(self):
        results = {"target": True, "dose": False}
        assert evaluate_strategy(FieldCheck(field="target"), results) is True

    def test_failing_field(self):
        results = {"target": True, "dose": False}
        assert evaluate_strategy(FieldCheck(field="dose"), results) is False

    def test_missing_field_raises(self):
        with pytest.raises(KeyError):
            evaluate_strategy(FieldCheck(field="missing"), {})


@pytest.mark.unit
class TestAllOf:
    """Test AllOf composition node."""

    def test_all_pass(self):
        results = {"a": True, "b": True}
        strategy = AllOf(conditions=[FieldCheck(field="a"), FieldCheck(field="b")])
        assert evaluate_strategy(strategy, results) is True

    def test_one_fails(self):
        results = {"a": True, "b": False}
        strategy = AllOf(conditions=[FieldCheck(field="a"), FieldCheck(field="b")])
        assert evaluate_strategy(strategy, results) is False

    def test_empty_conditions(self):
        assert evaluate_strategy(AllOf(conditions=[]), {}) is True


@pytest.mark.unit
class TestAnyOf:
    """Test AnyOf composition node."""

    def test_one_passes(self):
        results = {"a": False, "b": True}
        strategy = AnyOf(conditions=[FieldCheck(field="a"), FieldCheck(field="b")])
        assert evaluate_strategy(strategy, results) is True

    def test_none_pass(self):
        results = {"a": False, "b": False}
        strategy = AnyOf(conditions=[FieldCheck(field="a"), FieldCheck(field="b")])
        assert evaluate_strategy(strategy, results) is False

    def test_empty_conditions(self):
        assert evaluate_strategy(AnyOf(conditions=[]), {}) is False


@pytest.mark.unit
class TestAtLeastN:
    """Test AtLeastN composition node."""

    def test_exactly_n(self):
        results = {"a": True, "b": True, "c": False}
        strategy = AtLeastN(
            n=2,
            conditions=[FieldCheck(field="a"), FieldCheck(field="b"), FieldCheck(field="c")],
        )
        assert evaluate_strategy(strategy, results) is True

    def test_below_n(self):
        results = {"a": True, "b": False, "c": False}
        strategy = AtLeastN(
            n=2,
            conditions=[FieldCheck(field="a"), FieldCheck(field="b"), FieldCheck(field="c")],
        )
        assert evaluate_strategy(strategy, results) is False


@pytest.mark.unit
class TestNestedStrategies:
    """Test nested composition trees."""

    def test_allof_with_nested_anyof(self):
        results = {"target": True, "dose": False, "unit": True}
        strategy = AllOf(
            conditions=[
                FieldCheck(field="target"),
                AnyOf(conditions=[FieldCheck(field="dose"), FieldCheck(field="unit")]),
            ]
        )
        assert evaluate_strategy(strategy, results) is True

    def test_allof_with_nested_anyof_fails(self):
        results = {"target": True, "dose": False, "unit": False}
        strategy = AllOf(
            conditions=[
                FieldCheck(field="target"),
                AnyOf(conditions=[FieldCheck(field="dose"), FieldCheck(field="unit")]),
            ]
        )
        assert evaluate_strategy(strategy, results) is False


@pytest.mark.unit
class TestSerializationRoundTrip:
    """Test JSON serialization and deserialization."""

    def test_round_trip(self):
        strategy = AllOf(
            conditions=[
                FieldCheck(field="target"),
                AnyOf(conditions=[FieldCheck(field="dose"), FieldCheck(field="unit")]),
            ]
        )
        data = strategy.model_dump(mode="json")
        restored = AllOf.model_validate(data)
        results = {"target": True, "dose": False, "unit": True}
        assert evaluate_strategy(restored, results) is True
