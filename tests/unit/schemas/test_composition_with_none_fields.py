"""Tests for composition strategy evaluation with tri-valued field results.

After Fix C, ``field_results`` may contain ``None`` for fields the agentic
parser returned as null. At composition time None behaves like a soft False
(not-satisfied), but it must never be conflated with a real False. The
distinction is preserved in the cached field_results dict for downstream
consumers (granular scoring, serialized result rows, dataframe export).
"""

import pytest

from karenina.schemas.entities.composition import (
    AllOf,
    AnyOf,
    AtLeastN,
    FieldCheck,
    evaluate_strategy,
)


@pytest.mark.unit
class TestFieldCheckWithNone:
    """A leaf FieldCheck must return False when the field is None."""

    def test_none_is_not_passing(self):
        results = {"target": None}
        assert evaluate_strategy(FieldCheck(field="target"), results) is False

    def test_true_still_passes(self):
        results = {"target": True}
        assert evaluate_strategy(FieldCheck(field="target"), results) is True

    def test_false_still_fails(self):
        results = {"target": False}
        assert evaluate_strategy(FieldCheck(field="target"), results) is False


@pytest.mark.unit
class TestAllOfWithNone:
    """AllOf must reject when any leaf is None (soft-False)."""

    def test_all_true_passes(self):
        results = {"a": True, "b": True, "c": True}
        strategy = AllOf(
            conditions=[
                FieldCheck(field="a"),
                FieldCheck(field="b"),
                FieldCheck(field="c"),
            ],
        )
        assert evaluate_strategy(strategy, results) is True

    def test_one_none_fails(self):
        results = {"a": True, "b": None, "c": True}
        strategy = AllOf(
            conditions=[
                FieldCheck(field="a"),
                FieldCheck(field="b"),
                FieldCheck(field="c"),
            ],
        )
        assert evaluate_strategy(strategy, results) is False

    def test_mix_true_false_none(self):
        results = {"a": True, "b": False, "c": None}
        strategy = AllOf(
            conditions=[
                FieldCheck(field="a"),
                FieldCheck(field="b"),
                FieldCheck(field="c"),
            ],
        )
        assert evaluate_strategy(strategy, results) is False


@pytest.mark.unit
class TestAnyOfWithNone:
    """AnyOf passes only when at least one leaf is exactly True."""

    def test_one_true_among_nones_passes(self):
        results = {"a": None, "b": True, "c": None}
        strategy = AnyOf(
            conditions=[
                FieldCheck(field="a"),
                FieldCheck(field="b"),
                FieldCheck(field="c"),
            ],
        )
        assert evaluate_strategy(strategy, results) is True

    def test_all_none_fails(self):
        results = {"a": None, "b": None, "c": None}
        strategy = AnyOf(
            conditions=[
                FieldCheck(field="a"),
                FieldCheck(field="b"),
                FieldCheck(field="c"),
            ],
        )
        assert evaluate_strategy(strategy, results) is False

    def test_none_and_false_no_true_fails(self):
        results = {"a": None, "b": False, "c": None}
        strategy = AnyOf(
            conditions=[
                FieldCheck(field="a"),
                FieldCheck(field="b"),
                FieldCheck(field="c"),
            ],
        )
        assert evaluate_strategy(strategy, results) is False


@pytest.mark.unit
class TestAtLeastNWithNone:
    """AtLeastN counts only exactly-True leaves toward N."""

    def test_two_trues_one_none_meets_n2(self):
        results = {"a": True, "b": True, "c": None}
        strategy = AtLeastN(
            n=2,
            conditions=[
                FieldCheck(field="a"),
                FieldCheck(field="b"),
                FieldCheck(field="c"),
            ],
        )
        assert evaluate_strategy(strategy, results) is True

    def test_one_true_two_nones_fails_n2(self):
        results = {"a": True, "b": None, "c": None}
        strategy = AtLeastN(
            n=2,
            conditions=[
                FieldCheck(field="a"),
                FieldCheck(field="b"),
                FieldCheck(field="c"),
            ],
        )
        assert evaluate_strategy(strategy, results) is False

    def test_n0_passes_even_with_all_none(self):
        results = {"a": None, "b": None}
        strategy = AtLeastN(
            n=0,
            conditions=[FieldCheck(field="a"), FieldCheck(field="b")],
        )
        assert evaluate_strategy(strategy, results) is True

    def test_n_equals_passing_trues(self):
        # 3 of 5 leaves True, the rest None and False: n=3 succeeds, n=4 fails.
        results = {"a": True, "b": True, "c": True, "d": None, "e": False}
        conditions = [
            FieldCheck(field="a"),
            FieldCheck(field="b"),
            FieldCheck(field="c"),
            FieldCheck(field="d"),
            FieldCheck(field="e"),
        ]
        assert evaluate_strategy(AtLeastN(n=3, conditions=conditions), results) is True
        assert evaluate_strategy(AtLeastN(n=4, conditions=conditions), results) is False


@pytest.mark.unit
class TestNestedCompositionWithNone:
    """Nested compositions handle None at every level."""

    def test_allof_of_anyof_with_none(self):
        # AllOf( AnyOf(a,b), AnyOf(c,d) ) with a=True, b=None, c=None, d=True.
        results = {"a": True, "b": None, "c": None, "d": True}
        strategy = AllOf(
            conditions=[
                AnyOf(conditions=[FieldCheck(field="a"), FieldCheck(field="b")]),
                AnyOf(conditions=[FieldCheck(field="c"), FieldCheck(field="d")]),
            ]
        )
        assert evaluate_strategy(strategy, results) is True

    def test_allof_of_anyof_all_none_in_branch_fails(self):
        # AllOf( AnyOf(a,b), AnyOf(c,d) ) with first branch all None.
        results = {"a": None, "b": None, "c": True, "d": False}
        strategy = AllOf(
            conditions=[
                AnyOf(conditions=[FieldCheck(field="a"), FieldCheck(field="b")]),
                AnyOf(conditions=[FieldCheck(field="c"), FieldCheck(field="d")]),
            ]
        )
        assert evaluate_strategy(strategy, results) is False
