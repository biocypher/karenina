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
    NumericGraded,
    NumericMaximum,
    NumericMinimum,
    NumericRange,
    NumericRangeGraded,
    NumericThresholdGraded,
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
class TestNumericGraded:
    """Test NumericGraded primitive (distance-graded gate + partial credit)."""

    # --- Single-band (full_credit unset) ---

    def test_single_band_exact_full_credit(self):
        p = NumericGraded(cutoff=0.25, mode="relative")
        assert p.check(42, 42) is True
        assert p.score(42, 42) == 1.0

    def test_single_band_relative_linear_decay(self):
        p = NumericGraded(cutoff=0.25, mode="relative", decay="linear")
        # d = 4/42 = 0.09524 ; r = d/0.25 ; score = 1 - r
        assert p.check(46, 42) is True
        assert p.score(46, 42) == pytest.approx(1.0 - (4 / 42) / 0.25)

    def test_single_band_quadratic_decay(self):
        p = NumericGraded(cutoff=0.25, mode="relative", decay="quadratic")
        r = (4 / 42) / 0.25
        assert p.score(46, 42) == pytest.approx(1.0 - r * r)

    def test_single_band_absolute_mode(self):
        p = NumericGraded(cutoff=2.0, mode="absolute")
        assert p.check(43, 42) is True
        assert p.score(43, 42) == pytest.approx(0.5)  # 1 - 1/2
        assert p.check(45, 42) is False
        assert p.score(45, 42) == 0.0

    def test_single_band_at_cutoff_scores_zero_but_checks_true(self):
        p = NumericGraded(cutoff=0.25, mode="relative")
        # d exactly at cutoff: score 0.0 (d >= cutoff), but within the gate
        assert p.score(52.5, 42) == 0.0
        assert p.check(52.5, 42) is True

    def test_single_band_beyond_cutoff(self):
        p = NumericGraded(cutoff=0.25, mode="relative")
        assert p.check(60, 42) is False
        assert p.score(60, 42) == 0.0

    def test_single_band_score_positive_implies_check(self):
        p = NumericGraded(cutoff=0.25, mode="relative")
        for extracted in (42, 43, 45, 48, 50, 51, 52, 55, 60):
            if p.score(extracted, 42) > 0.0:
                assert p.check(extracted, 42) is True

    def test_zero_expected_relative_exact_only(self):
        p = NumericGraded(cutoff=0.1, mode="relative")
        assert p.check(0, 0) is True
        assert p.score(0, 0) == 1.0
        assert p.check(0.05, 0) is False
        assert p.score(0.05, 0) == 0.0

    # --- Double-band (full_credit set) ---

    def test_double_band_plateau(self):
        p = NumericGraded(cutoff=0.25, full_credit=0.01, mode="relative")
        # d = 0.3/42 = 0.00714 < full_credit -> full credit, and within the gate
        assert p.score(42.3, 42) == 1.0
        assert p.check(42.3, 42) is True

    def test_double_band_gate_is_inner(self):
        p = NumericGraded(cutoff=0.25, full_credit=0.01, mode="relative")
        # d = 4/42 = 0.0952 > full_credit -> beyond the gate, so check() False
        assert p.check(46, 42) is False

    def test_double_band_decay_between_inner_and_cutoff(self):
        p = NumericGraded(cutoff=0.25, full_credit=0.01, mode="relative")
        d = 4 / 42
        r = (d - 0.01) / (0.25 - 0.01)
        assert p.score(46, 42) == pytest.approx(1.0 - r)

    def test_double_band_intended_divergence(self):
        # In the decay region a near-miss is check() False but score() > 0.
        p = NumericGraded(cutoff=0.25, full_credit=0.01, mode="relative")
        assert p.check(46, 42) is False
        assert p.score(46, 42) > 0.0

    def test_double_band_beyond_cutoff(self):
        p = NumericGraded(cutoff=0.25, full_credit=0.01, mode="relative")
        assert p.check(60, 42) is False
        assert p.score(60, 42) == 0.0

    # --- Validation ---

    def test_cutoff_must_be_positive(self):
        with pytest.raises(ValidationError):
            NumericGraded(cutoff=0.0)
        with pytest.raises(ValidationError):
            NumericGraded(cutoff=-0.1)

    def test_full_credit_must_be_less_than_cutoff(self):
        with pytest.raises(ValidationError):
            NumericGraded(cutoff=0.1, full_credit=0.1)
        with pytest.raises(ValidationError):
            NumericGraded(cutoff=0.1, full_credit=0.2)

    def test_full_credit_zero_is_valid(self):
        p = NumericGraded(cutoff=0.25, full_credit=0.0)
        assert p.full_credit == 0.0

    def test_unknown_mode_rejected(self):
        with pytest.raises(ValidationError):
            NumericGraded(cutoff=0.25, mode="bogus")

    def test_unknown_decay_rejected(self):
        with pytest.raises(ValidationError):
            NumericGraded(cutoff=0.25, decay="bogus")


@pytest.mark.unit
class TestNumericRangeGraded:
    """Test NumericRangeGraded (acceptance band + soft-shoulder partial credit)."""

    def test_inside_band_full_credit(self):
        p = NumericRangeGraded(min=0.001, max=0.09, margin=0.02, mode="absolute")
        for x in (0.001, 0.021, 0.09):
            assert p.check(x, None) is True
            assert p.score(x, None) == 1.0

    def test_gate_matches_numeric_range(self):
        # check() is the same hard gate as NumericRange: inside the band only.
        graded = NumericRangeGraded(min=1.0, max=10.0, margin=2.0, mode="absolute")
        plain = NumericRange(min=1.0, max=10.0)
        for x in (0.0, 1.0, 5.0, 10.0, 11.0, 13.0):
            assert graded.check(x, None) == plain.check(x, "ignored")

    def test_shoulder_linear_decay_absolute(self):
        p = NumericRangeGraded(min=0.001, max=0.09, margin=0.02, mode="absolute", decay="linear")
        # x above max by 0.01 of a 0.02 shoulder -> 0.5
        assert p.check(0.10, None) is False
        assert p.score(0.10, None) == pytest.approx(0.5)
        # below min by 0.005 of 0.02 -> 0.75
        assert p.score(-0.004, None) == pytest.approx(0.75)

    def test_shoulder_quadratic_decay_relative(self):
        # relative margin is a fraction of band width (max - min)
        p = NumericRangeGraded(min=0.0, max=0.1, margin=0.5, mode="relative", decay="quadratic")
        shoulder = 0.5 * 0.1  # 0.05
        gap = 0.0223
        r = gap / shoulder
        assert p.score(0.1 + gap, None) == pytest.approx(1.0 - r * r)

    def test_beyond_shoulder_zero(self):
        p = NumericRangeGraded(min=0.001, max=0.09, margin=0.02, mode="absolute")
        assert p.score(0.11, None) == 0.0
        assert p.score(-0.02, None) == 0.0

    def test_intended_divergence_in_shoulder(self):
        p = NumericRangeGraded(min=1.0, max=10.0, margin=2.0, mode="absolute")
        # a near-miss just outside the band: check() False but score() > 0
        assert p.check(11.0, None) is False
        assert p.score(11.0, None) > 0.0

    def test_exclusive_edges_affect_only_gate(self):
        p = NumericRangeGraded(min=1.0, max=10.0, margin=2.0, mode="absolute", exclusive_max=True)
        # exactly at max: gate rejects (exclusive), but the score plateau still holds
        assert p.check(10.0, None) is False
        assert p.score(10.0, None) == 1.0

    def test_min_must_be_less_than_max(self):
        with pytest.raises(ValidationError):
            NumericRangeGraded(min=5.0, max=5.0, margin=1.0)
        with pytest.raises(ValidationError):
            NumericRangeGraded(min=6.0, max=5.0, margin=1.0)

    def test_margin_must_be_positive(self):
        with pytest.raises(ValidationError):
            NumericRangeGraded(min=0.0, max=1.0, margin=0.0)
        with pytest.raises(ValidationError):
            NumericRangeGraded(min=0.0, max=1.0, margin=-0.1)


@pytest.mark.unit
class TestNumericThresholdGraded:
    """Test NumericThresholdGraded (one-sided bound + soft-shoulder partial credit)."""

    def test_max_correct_side_full_credit(self):
        p = NumericThresholdGraded(direction="max", margin=1.0, mode="relative")
        # any value at or below the threshold scores 1.0, however far below
        assert p.check(1e-40, 1e-30) is True
        assert p.score(1e-40, 1e-30) == 1.0
        assert p.check(1e-30, 1e-30) is True
        assert p.score(1e-30, 1e-30) == 1.0

    def test_max_gate_matches_numeric_maximum(self):
        graded = NumericThresholdGraded(direction="max", margin=1.0, mode="relative")
        plain = NumericMaximum()
        for x in (5.0, 9.999, 10.0, 10.001, 20.0):
            assert graded.check(x, 10.0) == plain.check(x, 10.0)

    def test_max_shoulder_relative_linear_decay(self):
        p = NumericThresholdGraded(direction="max", margin=1.0, mode="relative", decay="linear")
        thr = 1e-30
        # 1.5e-30 is gap 0.5e-30 of a shoulder 1e-30 -> 0.5
        assert p.check(1.5e-30, thr) is False
        assert p.score(1.5e-30, thr) == pytest.approx(0.5)
        assert p.score(2e-30, thr) == 0.0

    def test_min_direction(self):
        p = NumericThresholdGraded(direction="min", margin=0.2, mode="absolute", decay="linear")
        assert p.check(12.0, 10.0) is True
        assert p.score(12.0, 10.0) == 1.0
        assert p.check(9.9, 10.0) is False
        assert p.score(9.9, 10.0) == pytest.approx(0.5)
        assert p.score(9.7, 10.0) == 0.0  # gap 0.3 is past the 0.2 shoulder

    def test_min_gate_matches_numeric_minimum(self):
        graded = NumericThresholdGraded(direction="min", margin=0.5, mode="absolute")
        plain = NumericMinimum()
        for x in (8.0, 9.999, 10.0, 10.001, 12.0):
            assert graded.check(x, 10.0) == plain.check(x, 10.0)

    def test_exclusive_gate(self):
        p = NumericThresholdGraded(direction="max", margin=0.5, mode="absolute", exclusive=True)
        assert p.check(10.0, 10.0) is False  # strict <
        assert p.check(9.999, 10.0) is True

    def test_relative_threshold_zero_scores_zero_on_wrong_side(self):
        p = NumericThresholdGraded(direction="max", margin=1.0, mode="relative")
        # threshold 0 with relative margin is undefined off-side -> 0.0 beyond
        assert p.score(0.0, 0.0) == 1.0
        assert p.score(0.5, 0.0) == 0.0

    def test_margin_must_be_positive(self):
        with pytest.raises(ValidationError):
            NumericThresholdGraded(direction="max", margin=0.0)
        with pytest.raises(ValidationError):
            NumericThresholdGraded(direction="min", margin=-1.0)

    def test_unknown_direction_rejected(self):
        with pytest.raises(ValidationError):
            NumericThresholdGraded(direction="sideways", margin=1.0)


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

    def test_numeric_graded_single_band_round_trip(self):
        p = NumericGraded(cutoff=0.25, mode="relative", decay="quadratic")
        data = p.model_dump(mode="json")
        restored = NumericGraded.model_validate(data)
        assert restored.check(46, 42) is True
        assert restored.score(46, 42) == p.score(46, 42)

    def test_numeric_graded_double_band_round_trip(self):
        p = NumericGraded(cutoff=0.25, full_credit=0.01, mode="absolute")
        data = p.model_dump(mode="json")
        restored = NumericGraded.model_validate(data)
        assert restored.full_credit == 0.01
        assert restored.score(46, 42) == p.score(46, 42)
        assert restored.check(46, 42) == p.check(46, 42)

    def test_numeric_range_graded_round_trip(self):
        p = NumericRangeGraded(min=0.001, max=0.09, margin=0.02, mode="absolute", decay="quadratic")
        data = p.model_dump(mode="json")
        restored = NumericRangeGraded.model_validate(data)
        assert restored.check(0.05, None) is True
        assert restored.score(0.10, None) == p.score(0.10, None)

    def test_numeric_threshold_graded_round_trip(self):
        p = NumericThresholdGraded(direction="max", margin=1.0, mode="relative", decay="linear")
        data = p.model_dump(mode="json")
        restored = NumericThresholdGraded.model_validate(data)
        assert restored.check(1e-40, 1e-30) is True
        assert restored.score(1.5e-30, 1e-30) == p.score(1.5e-30, 1e-30)
