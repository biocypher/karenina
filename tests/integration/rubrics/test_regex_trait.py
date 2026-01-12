"""Integration tests for RegexTrait evaluation.

These tests verify that RegexTrait correctly evaluates patterns against
actual LLM response traces, using the integration test fixtures.

Test scenarios:
- Pattern found in trace (citation patterns)
- Pattern not found in trace
- Multiple matches in single trace
- Empty trace handling
- Case sensitivity options
- Invert result functionality
- Complex regex patterns
"""

import pytest

from karenina.schemas.domain import RegexTrait

# =============================================================================
# Basic Pattern Matching Tests
# =============================================================================


@pytest.mark.integration
class TestRegexTraitPatternMatching:
    """Test basic pattern matching functionality."""

    def test_pattern_found_in_trace_with_citations(self, trace_with_citations: str):
        """Verify RegexTrait finds citation patterns [1], [2], etc."""
        trait = RegexTrait(
            name="has_citations",
            pattern=r"\[\d+\]",
            description="Response includes numeric citations",
        )
        result = trait.evaluate(trace_with_citations)
        assert result is True

    def test_pattern_not_found_in_trace_without_citations(self, trace_without_citations: str):
        """Verify RegexTrait correctly reports no match for clean traces."""
        trait = RegexTrait(
            name="has_citations",
            pattern=r"\[\d+\]",
            description="Response includes numeric citations",
        )
        result = trait.evaluate(trace_without_citations)
        assert result is False

    def test_url_pattern_not_found(self, trace_with_citations: str):
        """Verify URL pattern returns False when no URLs present."""
        trait = RegexTrait(
            name="has_urls",
            pattern=r"https?://\S+",
            description="Response includes URLs",
        )
        result = trait.evaluate(trace_with_citations)
        assert result is False

    def test_bcl2_pattern_found(self, trace_with_citations: str):
        """Verify pattern matching for specific content."""
        trait = RegexTrait(
            name="mentions_bcl2",
            pattern=r"BCL2",
            description="Response mentions BCL2 gene",
        )
        result = trait.evaluate(trace_with_citations)
        assert result is True


# =============================================================================
# Multiple Match Tests
# =============================================================================


@pytest.mark.integration
class TestRegexTraitMultipleMatches:
    """Test behavior with multiple pattern matches."""

    def test_multiple_citations_still_returns_true(self, trace_with_citations: str):
        """Verify trait returns True when multiple matches exist."""
        trait = RegexTrait(
            name="has_citations",
            pattern=r"\[\d+\]",
            description="Response includes numeric citations",
        )
        # The trace has [1], [2], [3], [4], [5] - multiple matches
        result = trait.evaluate(trace_with_citations)
        assert result is True

    def test_captures_any_match_in_long_text(self, trace_with_citations: str):
        """Verify pattern is found even when scattered throughout text."""
        # Pattern for "cancer" which appears multiple times
        trait = RegexTrait(
            name="mentions_cancer",
            pattern=r"cancer",
            description="Response mentions cancer",
            case_sensitive=False,
        )
        result = trait.evaluate(trace_with_citations)
        assert result is True


# =============================================================================
# Edge Case Tests
# =============================================================================


@pytest.mark.integration
class TestRegexTraitEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_empty_trace_returns_false(self):
        """Verify empty string returns False."""
        trait = RegexTrait(
            name="has_citations",
            pattern=r"\[\d+\]",
            description="Response includes numeric citations",
        )
        result = trait.evaluate("")
        assert result is False

    def test_whitespace_only_trace_returns_false(self):
        """Verify whitespace-only string returns False."""
        trait = RegexTrait(
            name="has_citations",
            pattern=r"\[\d+\]",
            description="Response includes numeric citations",
        )
        result = trait.evaluate("   \n\t  ")
        assert result is False

    def test_pattern_at_start_of_text(self):
        """Verify pattern at beginning of text is found."""
        trait = RegexTrait(
            name="starts_with_number",
            pattern=r"^\d+",
            description="Response starts with a number",
        )
        assert trait.evaluate("42 is the answer") is True
        assert trait.evaluate("The answer is 42") is False

    def test_pattern_at_end_of_text(self):
        """Verify pattern at end of text is found."""
        trait = RegexTrait(
            name="ends_with_period",
            pattern=r"\.$",
            description="Response ends with a period",
        )
        assert trait.evaluate("This is a sentence.") is True
        assert trait.evaluate("No period here") is False


# =============================================================================
# Case Sensitivity Tests
# =============================================================================


@pytest.mark.integration
class TestRegexTraitCaseSensitivity:
    """Test case sensitivity behavior."""

    def test_case_sensitive_exact_match(self, trace_with_citations: str):
        """Verify case-sensitive matching requires exact case."""
        # BCL2 appears in uppercase in the trace
        trait_exact = RegexTrait(
            name="bcl2_exact",
            pattern=r"BCL2",
            description="BCL2 in uppercase",
            case_sensitive=True,
        )
        assert trait_exact.evaluate(trace_with_citations) is True

        # bcl2 lowercase should not match when case-sensitive
        trait_lower = RegexTrait(
            name="bcl2_lower",
            pattern=r"bcl2",
            description="bcl2 in lowercase",
            case_sensitive=True,
        )
        assert trait_lower.evaluate(trace_with_citations) is False

    def test_case_insensitive_matches_any_case(self, trace_with_citations: str):
        """Verify case-insensitive matching finds any case variation."""
        trait = RegexTrait(
            name="bcl2_any",
            pattern=r"bcl2",
            description="BCL2 any case",
            case_sensitive=False,
        )
        assert trait.evaluate(trace_with_citations) is True
        assert trait.evaluate("bcl2 is a gene") is True
        assert trait.evaluate("Bcl2 protein") is True

    def test_default_is_case_sensitive(self):
        """Verify default behavior is case-sensitive."""
        trait = RegexTrait(
            name="test",
            pattern=r"HELLO",
            description="Test pattern",
            # case_sensitive defaults to True
        )
        assert trait.evaluate("hello world") is False
        assert trait.evaluate("HELLO world") is True


# =============================================================================
# Invert Result Tests
# =============================================================================


@pytest.mark.integration
class TestRegexTraitInvertResult:
    """Test invert_result functionality."""

    def test_invert_result_flips_true_to_false(self, trace_with_citations: str):
        """Verify invert_result=True flips a match to False."""
        trait = RegexTrait(
            name="no_citations",
            pattern=r"\[\d+\]",
            description="Response should NOT have citations",
            invert_result=True,
        )
        # trace_with_citations has citations, so inverted = False
        assert trait.evaluate(trace_with_citations) is False

    def test_invert_result_flips_false_to_true(self, trace_without_citations: str):
        """Verify invert_result=True flips no-match to True."""
        trait = RegexTrait(
            name="no_citations",
            pattern=r"\[\d+\]",
            description="Response should NOT have citations",
            invert_result=True,
        )
        # trace_without_citations has no citations, so inverted = True
        assert trait.evaluate(trace_without_citations) is True

    def test_invert_useful_for_absence_checking(self):
        """Verify invert_result is useful for 'must not contain' checks."""
        trait = RegexTrait(
            name="no_profanity",
            pattern=r"bad_word",
            description="Response should not contain profanity",
            invert_result=True,
        )
        assert trait.evaluate("This is a clean response") is True
        assert trait.evaluate("This has bad_word in it") is False


# =============================================================================
# Complex Pattern Tests
# =============================================================================


@pytest.mark.integration
class TestRegexTraitComplexPatterns:
    """Test complex regex patterns."""

    def test_email_pattern(self):
        """Verify email pattern matching."""
        trait = RegexTrait(
            name="has_email",
            pattern=r"[\w.+-]+@[\w-]+\.[\w.-]+",
            description="Response includes an email address",
        )
        assert trait.evaluate("Contact: user@example.com") is True
        assert trait.evaluate("No email here") is False

    def test_date_pattern(self):
        """Verify date pattern matching."""
        trait = RegexTrait(
            name="has_date",
            pattern=r"\d{4}-\d{2}-\d{2}",
            description="Response includes ISO date",
        )
        assert trait.evaluate("Date: 2024-01-15") is True
        assert trait.evaluate("January 15, 2024") is False

    def test_word_boundary_pattern(self):
        """Verify word boundary patterns work correctly."""
        trait = RegexTrait(
            name="whole_word",
            pattern=r"\bcat\b",
            description="Contains the word 'cat' as whole word",
            case_sensitive=False,
        )
        assert trait.evaluate("The cat sat") is True
        assert trait.evaluate("Category is different") is False
        assert trait.evaluate("concatenate") is False

    def test_multiline_pattern(self):
        """Verify patterns work across multiline text."""
        # Note: ^ anchor matches start of string only (no MULTILINE flag)
        # Use a pattern that matches bullet points anywhere in text
        trait = RegexTrait(
            name="has_bullet",
            pattern=r"\n[-*]\s+",
            description="Has bullet point after newline",
        )
        text = """Here is a list:
- Item one
- Item two"""
        assert trait.evaluate(text) is True

        # Text without bullets
        assert trait.evaluate("No bullets here") is False


# =============================================================================
# Integration with Rubric Fixtures
# =============================================================================


@pytest.mark.integration
class TestRegexTraitWithRubricFixtures:
    """Test RegexTrait using rubric fixtures from conftest."""

    def test_citation_regex_rubric_on_cited_trace(self, citation_regex_rubric, trace_with_citations: str):
        """Verify citation_regex_rubric correctly evaluates cited trace."""
        # citation_regex_rubric has has_citations and has_urls traits
        citation_trait = next(t for t in citation_regex_rubric.regex_traits if t.name == "has_citations")
        url_trait = next(t for t in citation_regex_rubric.regex_traits if t.name == "has_urls")

        assert citation_trait.evaluate(trace_with_citations) is True
        assert url_trait.evaluate(trace_with_citations) is False

    def test_citation_regex_rubric_on_uncited_trace(self, citation_regex_rubric, trace_without_citations: str):
        """Verify citation_regex_rubric correctly evaluates uncited trace."""
        citation_trait = next(t for t in citation_regex_rubric.regex_traits if t.name == "has_citations")

        assert citation_trait.evaluate(trace_without_citations) is False

    def test_multi_trait_rubric_regex_component(self, multi_trait_rubric, trace_with_citations: str):
        """Verify regex trait in multi_trait_rubric works correctly."""
        # multi_trait_rubric has one regex trait: has_citations
        assert len(multi_trait_rubric.regex_traits) == 1
        citation_trait = multi_trait_rubric.regex_traits[0]

        assert citation_trait.name == "has_citations"
        assert citation_trait.evaluate(trace_with_citations) is True
