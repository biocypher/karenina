"""Integration tests for CallableTrait evaluation.

These tests verify that CallableTrait correctly evaluates custom Python
functions against actual LLM response traces, using the integration test fixtures.

Test scenarios:
- Boolean callable returning true/false on traces
- Score callable evaluating trace properties
- Exception handling (graceful error capture)
- Callable with different logic patterns
- Integration with rubric fixtures (multi_trait_rubric pattern)

Note: Unit tests in test_callable_trait.py cover creation, serialization,
and validation. These tests focus on integration with real trace fixtures.
"""

import pytest

from karenina.schemas.domain import CallableTrait, Rubric

# =============================================================================
# Boolean Callable Tests with Trace Fixtures
# =============================================================================


@pytest.mark.integration
class TestCallableTraitBooleanWithTraces:
    """Test boolean CallableTraits against trace fixtures."""

    def test_callable_finds_citations_in_trace(self, trace_with_citations: str):
        """Verify callable can detect citation patterns in trace."""
        import re

        trait = CallableTrait.from_callable(
            name="has_citations",
            func=lambda text: bool(re.search(r"\[\d+\]", text)),
            kind="boolean",
            description="Check for numeric citations",
        )

        assert trait.evaluate(trace_with_citations) is True

    def test_callable_no_citations_in_clean_trace(self, trace_without_citations: str):
        """Verify callable correctly reports no citations."""
        import re

        trait = CallableTrait.from_callable(
            name="has_citations",
            func=lambda text: bool(re.search(r"\[\d+\]", text)),
            kind="boolean",
        )

        assert trait.evaluate(trace_without_citations) is False

    def test_callable_detects_abstention_language(self, trace_with_abstention: str):
        """Verify callable can detect abstention/refusal patterns."""

        def is_abstention(text: str) -> bool:
            lower = text.lower()
            abstention_phrases = [
                "i cannot",
                "i apologize",
                "i'm not able",
                "consult with",
                "please consult",
            ]
            return any(phrase in lower for phrase in abstention_phrases)

        trait = CallableTrait.from_callable(
            name="is_abstention",
            func=is_abstention,
            kind="boolean",
        )

        assert trait.evaluate(trace_with_abstention) is True

    def test_callable_detects_hedging_language(self, trace_with_hedging: str):
        """Verify callable can detect hedging/uncertainty patterns."""

        def has_hedging(text: str) -> bool:
            lower = text.lower()
            hedging_phrases = [
                "cannot be completely certain",
                "suggests that",
                "would be needed",
                "might be",
            ]
            return any(phrase in lower for phrase in hedging_phrases)

        trait = CallableTrait.from_callable(
            name="has_hedging",
            func=has_hedging,
            kind="boolean",
        )

        assert trait.evaluate(trace_with_hedging) is True

    def test_callable_detects_gene_mention(self, trace_with_citations: str):
        """Verify callable can detect specific scientific content."""
        trait = CallableTrait.from_callable(
            name="mentions_bcl2",
            func=lambda text: "BCL2" in text or "bcl2" in text.lower(),
            kind="boolean",
        )

        assert trait.evaluate(trace_with_citations) is True

    def test_callable_with_invert_for_absence_checking(self, trace_without_citations: str):
        """Verify invert_result works for 'must not contain' patterns."""
        import re

        trait = CallableTrait.from_callable(
            name="citation_free",
            func=lambda text: bool(re.search(r"\[\d+\]", text)),
            kind="boolean",
            invert_result=True,
            description="Response should NOT have citations",
        )

        # trace_without_citations has no citations, so inverted = True
        assert trait.evaluate(trace_without_citations) is True


# =============================================================================
# Score Callable Tests with Trace Fixtures
# =============================================================================


@pytest.mark.integration
class TestCallableTraitScoreWithTraces:
    """Test score CallableTraits against trace fixtures."""

    def test_callable_counts_citations(self, trace_with_citations: str):
        """Verify callable can count citation occurrences."""
        import re

        def count_citations(text: str) -> int:
            return len(re.findall(r"\[\d+\]", text))

        trait = CallableTrait.from_callable(
            name="citation_count",
            func=count_citations,
            kind="score",
            min_score=0,
            max_score=100,
            description="Count numeric citations",
        )

        # trace_with_citations has [1], [2], [3], [4], [5] and some duplicates
        result = trait.evaluate(trace_with_citations)
        assert result > 0
        assert isinstance(result, int)

    def test_callable_counts_words(self, trace_with_citations: str):
        """Verify callable can count words in trace."""
        trait = CallableTrait.from_callable(
            name="word_count",
            func=lambda text: len(text.split()),
            kind="score",
            min_score=0,
            max_score=1000,
        )

        result = trait.evaluate(trace_with_citations)
        assert result > 50  # Trace has substantial content
        assert isinstance(result, int)

    def test_callable_counts_paragraphs(self, trace_with_citations: str):
        """Verify callable can count paragraphs."""

        def count_paragraphs(text: str) -> int:
            # Split by double newlines or single newlines
            paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
            return len(paragraphs)

        trait = CallableTrait.from_callable(
            name="paragraph_count",
            func=count_paragraphs,
            kind="score",
            min_score=0,
            max_score=50,
        )

        result = trait.evaluate(trace_with_citations)
        assert result >= 1

    def test_callable_average_sentence_length(self, trace_without_citations: str):
        """Verify callable can compute average sentence length."""
        import re

        def avg_sentence_length(text: str) -> int:
            sentences = re.split(r"[.!?]+", text)
            sentences = [s.strip() for s in sentences if s.strip()]
            if not sentences:
                return 0
            avg = sum(len(s.split()) for s in sentences) / len(sentences)
            return int(avg)

        trait = CallableTrait.from_callable(
            name="avg_sentence_words",
            func=avg_sentence_length,
            kind="score",
            min_score=0,
            max_score=100,
        )

        result = trait.evaluate(trace_without_citations)
        assert result > 0
        assert result < 50  # Reasonable sentence length

    def test_callable_higher_is_better_false(self, trace_with_citations: str):
        """Verify higher_is_better=False for error counting."""

        def count_errors(text: str) -> int:
            lower = text.lower()
            error_words = ["error", "mistake", "wrong", "incorrect"]
            return sum(lower.count(word) for word in error_words)

        trait = CallableTrait.from_callable(
            name="error_count",
            func=count_errors,
            kind="score",
            min_score=0,
            max_score=100,
            higher_is_better=False,  # Fewer errors is better
        )

        assert trait.higher_is_better is False
        result = trait.evaluate(trace_with_citations)
        assert result == 0  # No error words in the trace


# =============================================================================
# Edge Case and Error Handling Tests
# =============================================================================


@pytest.mark.integration
class TestCallableTraitEdgeCases:
    """Test edge cases and error handling."""

    def test_callable_with_empty_text(self):
        """Verify callable handles empty string input."""
        trait = CallableTrait.from_callable(
            name="word_count",
            func=lambda text: len(text.split()),
            kind="score",
            min_score=0,
            max_score=100,
        )

        assert trait.evaluate("") == 0

    def test_callable_with_whitespace_only(self):
        """Verify callable handles whitespace-only input."""
        trait = CallableTrait.from_callable(
            name="has_content",
            func=lambda text: len(text.strip()) > 0,
            kind="boolean",
        )

        assert trait.evaluate("   \n\t  ") is False

    def test_callable_exception_wrapped_in_runtime_error(self):
        """Verify callable exceptions are wrapped properly."""

        def buggy_func(text: str) -> bool:
            raise ValueError("Intentional test error")

        trait = CallableTrait.from_callable(
            name="buggy",
            func=buggy_func,
            kind="boolean",
        )

        with pytest.raises(RuntimeError) as exc_info:
            trait.evaluate("test")

        assert "Failed to evaluate" in str(exc_info.value)

    def test_callable_with_unicode_content(self):
        """Verify callable handles unicode correctly."""

        def has_emoji(text: str) -> bool:
            import re

            emoji_pattern = re.compile(
                "["
                "\U0001f600-\U0001f64f"  # emoticons
                "\U0001f300-\U0001f5ff"  # symbols & pictographs
                "]",
                flags=re.UNICODE,
            )
            return bool(emoji_pattern.search(text))

        trait = CallableTrait.from_callable(
            name="has_emoji",
            func=has_emoji,
            kind="boolean",
        )

        assert trait.evaluate("Hello 😀") is True
        assert trait.evaluate("Hello world") is False

    def test_callable_with_very_long_text(self):
        """Verify callable handles long text efficiently."""
        trait = CallableTrait.from_callable(
            name="length",
            func=lambda text: len(text),
            kind="score",
            min_score=0,
            max_score=1000000,
        )

        long_text = "word " * 10000  # 50000+ characters
        result = trait.evaluate(long_text)
        assert result > 49000


# =============================================================================
# Complex Callable Logic Tests
# =============================================================================


@pytest.mark.integration
class TestCallableTraitComplexLogic:
    """Test complex callable logic patterns."""

    def test_callable_with_closure(self, trace_with_citations: str):
        """Verify callable with closure variables works."""
        required_keyword = "BCL2"

        def check_keyword(text: str) -> bool:
            return required_keyword in text

        trait = CallableTrait.from_callable(
            name="has_keyword",
            func=check_keyword,
            kind="boolean",
        )

        assert trait.evaluate(trace_with_citations) is True

    def test_callable_multi_condition_check(self, trace_with_citations: str):
        """Verify callable with multiple conditions."""

        def quality_check(text: str) -> bool:
            has_content = len(text) > 100
            has_paragraphs = "\n\n" in text or text.count("\n") > 2
            has_structure = any(marker in text for marker in ["[1]", "-", "*", "References"])
            return has_content and has_paragraphs and has_structure

        trait = CallableTrait.from_callable(
            name="quality_check",
            func=quality_check,
            kind="boolean",
            description="Check multiple quality criteria",
        )

        assert trait.evaluate(trace_with_citations) is True

    def test_callable_score_with_multiple_factors(self, trace_with_citations: str):
        """Verify callable computing composite score."""
        import re

        def composite_score(text: str) -> int:
            # Weight different factors
            word_count = len(text.split())
            citation_count = len(re.findall(r"\[\d+\]", text))
            has_structure = 1 if "\n\n" in text else 0

            # Compute weighted score (0-100)
            score = min(word_count // 5, 50)  # Up to 50 for length
            score += min(citation_count * 5, 30)  # Up to 30 for citations
            score += has_structure * 20  # 20 for structure

            return min(score, 100)

        trait = CallableTrait.from_callable(
            name="composite_quality",
            func=composite_score,
            kind="score",
            min_score=0,
            max_score=100,
        )

        result = trait.evaluate(trace_with_citations)
        assert 50 <= result <= 100  # Should score well on all factors


# =============================================================================
# Integration with Rubric Pattern Tests
# =============================================================================


@pytest.mark.integration
class TestCallableTraitWithRubric:
    """Test CallableTrait integration with Rubric structures."""

    def test_rubric_with_callable_and_regex_traits(self, trace_with_citations: str):
        """Verify Rubric can contain both callable and regex traits."""
        from karenina.schemas.domain import RegexTrait

        callable_trait = CallableTrait.from_callable(
            name="word_count_ok",
            func=lambda text: len(text.split()) >= 50,
            kind="boolean",
            description="Has at least 50 words",
        )

        regex_trait = RegexTrait(
            name="has_citations",
            pattern=r"\[\d+\]",
            description="Has numeric citations",
        )

        rubric = Rubric(
            callable_traits=[callable_trait],
            regex_traits=[regex_trait],
        )

        # Both should evaluate correctly
        assert rubric.callable_traits[0].evaluate(trace_with_citations) is True
        assert rubric.regex_traits[0].evaluate(trace_with_citations) is True

    def test_rubric_with_multiple_callable_traits(self, trace_with_citations: str):
        """Verify Rubric can have multiple callable traits."""
        traits = [
            CallableTrait.from_callable(
                name="has_content",
                func=lambda text: len(text) > 100,
                kind="boolean",
            ),
            CallableTrait.from_callable(
                name="word_count",
                func=lambda text: len(text.split()),
                kind="score",
                min_score=0,
                max_score=500,
            ),
        ]

        rubric = Rubric(callable_traits=traits)

        assert len(rubric.callable_traits) == 2
        assert rubric.callable_traits[0].evaluate(trace_with_citations) is True
        assert rubric.callable_traits[1].evaluate(trace_with_citations) > 0

    def test_callable_trait_in_rubric_roundtrip(self, trace_with_citations: str):
        """Verify CallableTrait survives Rubric serialization."""
        trait = CallableTrait.from_callable(
            name="test_trait",
            func=lambda text: "BCL2" in text,
            kind="boolean",
        )

        rubric = Rubric(callable_traits=[trait])

        # Serialize and deserialize
        data = rubric.model_dump()
        restored = Rubric(**data)

        # Should still work
        assert restored.callable_traits[0].evaluate(trace_with_citations) is True
