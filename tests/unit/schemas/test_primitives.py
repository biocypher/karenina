"""Tests for verification primitives."""

import pytest
from pydantic import ValidationError

from karenina.schemas.primitives import (
    BooleanMatch,
    ContainsAll,
    ContainsAny,
    DateMatch,
    DateRange,
    DateTolerance,
    ExactMatch,
    LiteralMatch,
    NumericExact,
    NumericMaximum,
    NumericMinimum,
    NumericRange,
    NumericTolerance,
    OrderedMatch,
    RegexMatch,
    SetContainment,
    SynonymMap,
    TraceContains,
    TraceLength,
    TraceRegex,
)


@pytest.mark.unit
class TestExactMatch:
    """Test ExactMatch primitive."""

    def test_exact_match_default_normalization(self):
        p = ExactMatch()
        assert p.check("BCL2", "bcl2") is True

    def test_exact_match_with_whitespace(self):
        p = ExactMatch()
        assert p.check("  BCL2 ", "bcl2") is True

    def test_exact_match_no_normalization(self):
        p = ExactMatch(normalize=[])
        assert p.check("BCL2", "bcl2") is False

    def test_exact_match_mismatch(self):
        p = ExactMatch()
        assert p.check("TP53", "BCL2") is False

    def test_exact_match_empty_strings(self):
        p = ExactMatch()
        assert p.check("", "") is True

    def test_exact_match_none_coerced(self):
        p = ExactMatch()
        assert p.check(None, "None") is True


@pytest.mark.unit
class TestBooleanMatch:
    """Test BooleanMatch primitive."""

    def test_true_true(self):
        assert BooleanMatch().check(True, True) is True

    def test_false_false(self):
        assert BooleanMatch().check(False, False) is True

    def test_true_false(self):
        assert BooleanMatch().check(True, False) is False

    def test_truthy_coercion(self):
        assert BooleanMatch().check(1, True) is True

    def test_falsy_coercion(self):
        assert BooleanMatch().check(0, False) is True


@pytest.mark.unit
class TestContainsAny:
    """Test ContainsAny primitive."""

    def test_contains_one(self):
        p = ContainsAny(substrings=["BCL2", "TP53"])
        assert p.check("The target is BCL2", "ignored") is True

    def test_contains_none(self):
        p = ContainsAny(substrings=["BCL2", "TP53"])
        assert p.check("The target is BRCA1", "ignored") is False

    def test_case_insensitive(self):
        p = ContainsAny(substrings=["bcl2"], normalize=["lowercase"])
        assert p.check("The target is BCL2", "ignored") is True

    def test_empty_substrings(self):
        p = ContainsAny(substrings=[])
        assert p.check("anything", "ignored") is False


@pytest.mark.unit
class TestContainsAll:
    """Test ContainsAll primitive."""

    def test_contains_all(self):
        p = ContainsAll(substrings=["BCL2", "venetoclax"])
        assert p.check("BCL2 is the target of venetoclax", "ignored") is True

    def test_missing_one(self):
        p = ContainsAll(substrings=["BCL2", "venetoclax"])
        assert p.check("BCL2 is the target", "ignored") is False

    def test_empty_substrings_vacuous_truth(self):
        p = ContainsAll(substrings=[])
        assert p.check("anything", "ignored") is True


@pytest.mark.unit
class TestRegexMatch:
    """Test RegexMatch primitive."""

    def test_match(self):
        p = RegexMatch(pattern=r"BCL\d+")
        assert p.check("BCL2", "ignored") is True

    def test_no_match(self):
        p = RegexMatch(pattern=r"BCL\d+")
        assert p.check("BRCA1", "ignored") is False

    def test_case_insensitive_flag(self):
        p = RegexMatch(pattern=r"bcl\d+", flags=["IGNORECASE"])
        assert p.check("BCL2", "ignored") is True


@pytest.mark.unit
class TestNumericExact:
    """Test NumericExact primitive."""

    def test_equal_ints(self):
        assert NumericExact().check(5, 5) is True

    def test_unequal_ints(self):
        assert NumericExact().check(5, 6) is False

    def test_float_coercion(self):
        assert NumericExact().check(5.0, 5) is True

    def test_string_coercion(self):
        assert NumericExact().check("5", 5) is True


@pytest.mark.unit
class TestNumericTolerance:
    """Test NumericTolerance primitive."""

    def test_within_relative_tolerance(self):
        p = NumericTolerance(tolerance=0.1, mode="relative")
        assert p.check(5.2, 5.0) is True

    def test_outside_relative_tolerance(self):
        p = NumericTolerance(tolerance=0.01, mode="relative")
        assert p.check(5.2, 5.0) is False

    def test_within_absolute_tolerance(self):
        p = NumericTolerance(tolerance=0.5, mode="absolute")
        assert p.check(5.3, 5.0) is True

    def test_outside_absolute_tolerance(self):
        p = NumericTolerance(tolerance=0.1, mode="absolute")
        assert p.check(5.3, 5.0) is False

    def test_zero_expected_relative(self):
        p = NumericTolerance(tolerance=0.1, mode="relative")
        assert p.check(0, 0) is True

    def test_zero_expected_nonzero_extracted(self):
        p = NumericTolerance(tolerance=0.1, mode="relative")
        assert p.check(0.1, 0) is False

    def test_exact_boundary(self):
        p = NumericTolerance(tolerance=0.1, mode="relative")
        assert p.check(5.5, 5.0) is True

    def test_unknown_mode_rejected(self):
        with pytest.raises(ValidationError):
            NumericTolerance(tolerance=0.1, mode="bogus")

    def test_valid_modes_accepted(self):
        p_rel = NumericTolerance(tolerance=0.1, mode="relative")
        assert p_rel.mode == "relative"
        p_abs = NumericTolerance(tolerance=0.1, mode="absolute")
        assert p_abs.mode == "absolute"


@pytest.mark.unit
class TestNumericRange:
    """Test NumericRange primitive."""

    def test_within_range(self):
        p = NumericRange(min=1.0, max=10.0)
        assert p.check(5.0, "ignored") is True

    def test_below_min(self):
        p = NumericRange(min=1.0, max=10.0)
        assert p.check(0.5, "ignored") is False

    def test_above_max(self):
        p = NumericRange(min=1.0, max=10.0)
        assert p.check(10.5, "ignored") is False

    def test_no_min(self):
        p = NumericRange(min=None, max=10.0)
        assert p.check(-100.0, "ignored") is True

    def test_no_max(self):
        p = NumericRange(min=1.0, max=None)
        assert p.check(9999.0, "ignored") is True

    def test_boundary_inclusive(self):
        p = NumericRange(min=1.0, max=10.0)
        assert p.check(1.0, "ignored") is True
        assert p.check(10.0, "ignored") is True


@pytest.mark.unit
class TestNumericMinimum:
    """Test NumericMinimum primitive."""

    def test_at_minimum(self):
        p = NumericMinimum()
        assert p.check(10, 10) is True

    def test_above_minimum(self):
        p = NumericMinimum()
        assert p.check(15, 10) is True

    def test_below_minimum(self):
        p = NumericMinimum()
        assert p.check(5, 10) is False

    def test_exclusive_at_boundary(self):
        p = NumericMinimum(exclusive=True)
        assert p.check(10, 10) is False

    def test_exclusive_above(self):
        p = NumericMinimum(exclusive=True)
        assert p.check(10.001, 10) is True

    def test_string_coercion(self):
        p = NumericMinimum()
        assert p.check("10", 5) is True

    def test_negative_minimum(self):
        p = NumericMinimum()
        assert p.check(-5, -10) is True

    def test_zero_minimum(self):
        p = NumericMinimum()
        assert p.check(0, 0) is True


@pytest.mark.unit
class TestNumericMaximum:
    """Test NumericMaximum primitive."""

    def test_at_maximum(self):
        p = NumericMaximum()
        assert p.check(10, 10) is True

    def test_below_maximum(self):
        p = NumericMaximum()
        assert p.check(5, 10) is True

    def test_above_maximum(self):
        p = NumericMaximum()
        assert p.check(15, 10) is False

    def test_exclusive_at_boundary(self):
        p = NumericMaximum(exclusive=True)
        assert p.check(10, 10) is False

    def test_exclusive_below(self):
        p = NumericMaximum(exclusive=True)
        assert p.check(9.999, 10) is True

    def test_string_coercion(self):
        p = NumericMaximum()
        assert p.check("3", 5) is True

    def test_negative_maximum(self):
        p = NumericMaximum()
        assert p.check(-5, -1) is True

    def test_zero_maximum(self):
        p = NumericMaximum()
        assert p.check(0, 0) is True


@pytest.mark.unit
class TestSetContainment:
    """Test SetContainment primitive."""

    def test_exact_match(self):
        p = SetContainment(mode="exact")
        assert p.check(["a", "b", "c"], ["a", "b", "c"]) is True

    def test_exact_match_order_independent(self):
        p = SetContainment(mode="exact")
        assert p.check(["c", "a", "b"], ["a", "b", "c"]) is True

    def test_exact_match_missing(self):
        p = SetContainment(mode="exact")
        assert p.check(["a", "b"], ["a", "b", "c"]) is False

    def test_subset(self):
        p = SetContainment(mode="subset")
        assert p.check(["a", "b"], ["a", "b", "c"]) is True

    def test_superset(self):
        p = SetContainment(mode="superset")
        assert p.check(["a", "b", "c", "d"], ["a", "b", "c"]) is True

    def test_overlap_with_min(self):
        p = SetContainment(mode="overlap", min_overlap=2)
        assert p.check(["a", "b", "x"], ["a", "b", "c"]) is True

    def test_overlap_below_min(self):
        p = SetContainment(mode="overlap", min_overlap=3)
        assert p.check(["a", "x", "y"], ["a", "b", "c"]) is False

    def test_empty_sets_exact(self):
        p = SetContainment(mode="exact")
        assert p.check([], []) is True


@pytest.mark.unit
class TestOrderedMatch:
    """Test OrderedMatch primitive."""

    def test_matching_order(self):
        p = OrderedMatch()
        assert p.check(["BCL2", "TP53"], ["bcl2", "tp53"]) is True

    def test_wrong_order(self):
        p = OrderedMatch()
        assert p.check(["TP53", "BCL2"], ["bcl2", "tp53"]) is False

    def test_different_lengths(self):
        p = OrderedMatch()
        assert p.check(["BCL2"], ["bcl2", "tp53"]) is False


@pytest.mark.unit
class TestLiteralMatch:
    """Test LiteralMatch primitive."""

    def test_matching_value(self):
        assert LiteralMatch().check("approved", "approved") is True

    def test_non_matching(self):
        assert LiteralMatch().check("pending", "approved") is False


@pytest.mark.unit
class TestDateMatch:
    """Test DateMatch primitive."""

    def test_same_date_different_formats(self):
        p = DateMatch()
        assert p.check("2024-01-15", "January 15, 2024") is True

    def test_different_dates(self):
        p = DateMatch()
        assert p.check("2024-01-15", "2024-02-15") is False

    def test_with_format_hint(self):
        p = DateMatch(format="%Y-%m-%d")
        assert p.check("2024-01-15", "2024-01-15") is True


@pytest.mark.unit
class TestDateTolerance:
    """Test DateTolerance primitive."""

    def test_within_tolerance(self):
        p = DateTolerance(tolerance=3, unit="days")
        assert p.check("2024-01-15", "2024-01-17") is True

    def test_outside_tolerance(self):
        p = DateTolerance(tolerance=1, unit="days")
        assert p.check("2024-01-15", "2024-01-20") is False

    def test_unknown_unit_rejected(self):
        with pytest.raises(ValidationError):
            DateTolerance(tolerance=3, unit="weeks")

    def test_valid_units_accepted(self):
        for unit in ("days", "hours", "minutes"):
            p = DateTolerance(tolerance=1, unit=unit)
            assert p.unit == unit


@pytest.mark.unit
class TestDateRange:
    """Test DateRange primitive."""

    def test_within_range(self):
        p = DateRange(min="2024-01-01", max="2024-12-31")
        assert p.check("2024-06-15", "ignored") is True

    def test_before_min(self):
        p = DateRange(min="2024-01-01", max="2024-12-31")
        assert p.check("2023-12-31", "ignored") is False


@pytest.mark.unit
class TestTraceRegex:
    """Test TraceRegex trace primitive."""

    def test_pattern_found(self):
        p = TraceRegex(pattern=r"\[\d+\]")
        assert p.check_trace("See reference [1] and [2]") is True

    def test_pattern_not_found(self):
        p = TraceRegex(pattern=r"\[\d+\]")
        assert p.check_trace("No citations here") is False

    def test_count_min_met(self):
        p = TraceRegex(pattern=r"\[\d+\]", count_min=3)
        assert p.check_trace("[1] [2] [3]") is True

    def test_count_min_not_met(self):
        p = TraceRegex(pattern=r"\[\d+\]", count_min=3)
        assert p.check_trace("[1] [2]") is False

    def test_empty_trace(self):
        p = TraceRegex(pattern=r"\[\d+\]")
        assert p.check_trace("") is False


@pytest.mark.unit
class TestTraceContains:
    """Test TraceContains trace primitive."""

    def test_substring_found(self):
        p = TraceContains(substring="disclaimer")
        assert p.check_trace("This includes a disclaimer about safety") is True

    def test_substring_not_found(self):
        p = TraceContains(substring="disclaimer")
        assert p.check_trace("No warnings here") is False


@pytest.mark.unit
class TestTraceLength:
    """Test TraceLength trace primitive."""

    def test_within_char_range(self):
        p = TraceLength(min=10, max=100, unit="chars")
        assert p.check_trace("This is a medium-length response.") is True

    def test_below_min_chars(self):
        p = TraceLength(min=100, unit="chars")
        assert p.check_trace("Short") is False

    def test_above_max_chars(self):
        p = TraceLength(max=5, unit="chars")
        assert p.check_trace("Too long") is False

    def test_word_count(self):
        p = TraceLength(min=3, max=10, unit="words")
        assert p.check_trace("three words here") is True

    def test_no_limits(self):
        p = TraceLength(unit="chars")
        assert p.check_trace("anything") is True


@pytest.mark.unit
class TestPrimitiveSerializationRoundTrip:
    """Test JSON serialization and deserialization of primitives."""

    def test_exact_match_round_trip(self):
        p = ExactMatch(normalize=["lowercase", "strip"])
        data = p.model_dump(mode="json")
        restored = ExactMatch.model_validate(data)
        assert restored.check("BCL2", "bcl2") is True

    def test_exact_match_with_synonym_map_round_trip(self):
        syn = SynonymMap(mapping={"Bcl-2": "BCL2"})
        p = ExactMatch(normalize=[syn, "lowercase"])
        data = p.model_dump(mode="json")
        restored = ExactMatch.model_validate(data)
        assert restored.check("Bcl-2", "bcl2") is True

    def test_numeric_tolerance_round_trip(self):
        p = NumericTolerance(tolerance=0.1, mode="relative")
        data = p.model_dump(mode="json")
        restored = NumericTolerance.model_validate(data)
        assert restored.check(5.2, 5.0) is True
