"""Tests for generic evaluate_composition()."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from karenina.schemas.primitives.composition import (
    AllOf,
    AnyOf,
    AtLeastN,
    evaluate_composition,
)


class FakeCheck:
    """Minimal leaf node for testing."""

    def __init__(self, value: bool):
        self.value = value


@pytest.mark.unit
class TestEvaluateComposition:
    def test_leaf_delegates_to_evaluator(self):
        leaf = FakeCheck(True)
        result = evaluate_composition(leaf, lambda n: n.value)
        assert result is True

    def test_all_of_all_true(self):
        checks = [FakeCheck(True), FakeCheck(True)]
        node = AllOf(conditions=checks)
        assert evaluate_composition(node, lambda n: n.value) is True

    def test_all_of_one_false(self):
        checks = [FakeCheck(True), FakeCheck(False)]
        node = AllOf(conditions=checks)
        assert evaluate_composition(node, lambda n: n.value) is False

    def test_all_of_empty_returns_true(self):
        node = AllOf(conditions=[])
        assert evaluate_composition(node, lambda n: n.value) is True

    def test_any_of_one_true(self):
        checks = [FakeCheck(False), FakeCheck(True)]
        node = AnyOf(conditions=checks)
        assert evaluate_composition(node, lambda n: n.value) is True

    def test_any_of_all_false(self):
        checks = [FakeCheck(False), FakeCheck(False)]
        node = AnyOf(conditions=checks)
        assert evaluate_composition(node, lambda n: n.value) is False

    def test_any_of_empty_returns_false(self):
        node = AnyOf(conditions=[])
        assert evaluate_composition(node, lambda n: n.value) is False

    def test_at_least_n(self):
        checks = [FakeCheck(True), FakeCheck(False), FakeCheck(True)]
        node = AtLeastN(n=2, conditions=checks)
        assert evaluate_composition(node, lambda n: n.value) is True

    def test_at_least_n_not_enough(self):
        checks = [FakeCheck(True), FakeCheck(False), FakeCheck(False)]
        node = AtLeastN(n=2, conditions=checks)
        assert evaluate_composition(node, lambda n: n.value) is False

    def test_nested_composition(self):
        inner = AllOf(conditions=[FakeCheck(True), FakeCheck(True)])
        outer = AnyOf(conditions=[FakeCheck(False), inner])
        assert evaluate_composition(outer, lambda n: n.value) is True


@pytest.mark.unit
class TestAtLeastNValidation:
    """Test AtLeastN construction validation."""

    def test_negative_n_rejected(self):
        with pytest.raises(ValidationError):
            AtLeastN(n=-1)

    def test_zero_n_accepted(self):
        node = AtLeastN(n=0)
        assert node.n == 0

    def test_positive_n_accepted(self):
        node = AtLeastN(n=3)
        assert node.n == 3
