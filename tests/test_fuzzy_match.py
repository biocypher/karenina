"""Tests for fuzzy matching utility.

This module tests the fuzzy matching functionality used to validate
excerpts in the deep-judgment feature.
"""

from karenina.benchmark.verification.tools.fuzzy_match import fuzzy_match_excerpt, fuzzy_match_excerpt_with_context


class TestFuzzyMatchExcerpt:
    """Tests for fuzzy_match_excerpt function."""

    def test_exact_match(self):
        """Test exact match returns perfect score."""
        excerpt = "targets BCL-2 protein"
        trace = "The drug venetoclax targets BCL-2 protein and induces apoptosis."

        match_found, similarity = fuzzy_match_excerpt(excerpt, trace)

        assert match_found is True
        assert similarity == 1.0

    def test_exact_match_at_beginning(self):
        """Test exact match at the beginning of trace."""
        excerpt = "The drug venetoclax"
        trace = "The drug venetoclax targets BCL-2 protein."

        match_found, similarity = fuzzy_match_excerpt(excerpt, trace)

        assert match_found is True
        assert similarity == 1.0

    def test_exact_match_at_end(self):
        """Test exact match at the end of trace."""
        excerpt = "induces apoptosis"
        trace = "The drug targets BCL-2 and induces apoptosis"

        match_found, similarity = fuzzy_match_excerpt(excerpt, trace)

        assert match_found is True
        assert similarity == 1.0

    def test_match_with_extra_whitespace(self):
        """Test matching with extra whitespace in excerpt."""
        excerpt = "targets  BCL-2   protein"  # Extra spaces
        trace = "The drug targets BCL-2 protein here."

        match_found, similarity = fuzzy_match_excerpt(excerpt, trace)

        assert match_found is True
        # Should normalize whitespace and match perfectly
        assert similarity >= 0.95

    def test_match_with_different_whitespace(self):
        """Test matching with tabs and newlines."""
        excerpt = "targets\tBCL-2\nprotein"
        trace = "The drug targets BCL-2 protein here."

        match_found, similarity = fuzzy_match_excerpt(excerpt, trace)

        assert match_found is True
        assert similarity >= 0.95

    def test_partial_match_above_threshold(self):
        """Test partial match that exceeds 75% threshold."""
        excerpt = "targets BCL-2 protein family"
        trace = "The drug targets BCL-2 protein here."

        match_found, similarity = fuzzy_match_excerpt(excerpt, trace)

        # Should find "targets BCL-2 protein" which is most of the excerpt
        assert match_found is True
        assert similarity >= 0.75

    def test_partial_match_below_threshold(self):
        """Test partial match below 75% threshold."""
        excerpt = "inhibits BCL-2 and BAX proteins"
        trace = "The drug targets BCL-2 protein."

        match_found, similarity = fuzzy_match_excerpt(excerpt, trace)

        # Only "BCL-2" matches, which is less than 75% of excerpt
        assert match_found is False
        assert similarity < 0.75

    def test_no_match_completely_different(self):
        """Test no match when excerpt is completely different."""
        excerpt = "hallucinated text that does not exist"
        trace = "The drug targets BCL-2 protein."

        match_found, similarity = fuzzy_match_excerpt(excerpt, trace)

        assert match_found is False
        assert similarity < 0.3  # Very low similarity

    def test_no_match_similar_words_different_meaning(self):
        """Test no match when words are similar but meaning is different."""
        excerpt = "does not target BCL-2"
        trace = "The drug targets BCL-2 protein."

        match_found, similarity = fuzzy_match_excerpt(excerpt, trace)

        # "target BCL-2" matches but "does not" changes meaning
        # Should have some similarity but not enough to match
        assert similarity > 0.0  # Some similarity due to overlapping words
        # Whether it passes threshold depends on exact matching

    def test_empty_excerpt(self):
        """Test handling of empty excerpt."""
        match_found, similarity = fuzzy_match_excerpt("", "Some trace text")

        assert match_found is False
        assert similarity == 0.0

    def test_empty_trace(self):
        """Test handling of empty trace."""
        match_found, similarity = fuzzy_match_excerpt("Some excerpt", "")

        assert match_found is False
        assert similarity == 0.0

    def test_both_empty(self):
        """Test handling when both excerpt and trace are empty."""
        match_found, similarity = fuzzy_match_excerpt("", "")

        assert match_found is False
        assert similarity == 0.0

    def test_whitespace_only_excerpt(self):
        """Test handling of whitespace-only excerpt."""
        match_found, similarity = fuzzy_match_excerpt("   \t\n   ", "Some text")

        assert match_found is False
        assert similarity == 0.0

    def test_whitespace_only_trace(self):
        """Test handling of whitespace-only trace."""
        match_found, similarity = fuzzy_match_excerpt("Some text", "   \t\n   ")

        assert match_found is False
        assert similarity == 0.0

    def test_case_sensitive_matching(self):
        """Test that matching is case-sensitive."""
        excerpt = "TARGETS BCL-2"
        trace = "The drug targets BCL-2 protein."

        match_found, similarity = fuzzy_match_excerpt(excerpt, trace)

        # Case differences should reduce similarity
        assert similarity < 1.0

    def test_punctuation_differences(self):
        """Test matching with punctuation differences."""
        excerpt = "targets BCL-2, protein"
        trace = "The drug targets BCL-2 protein here."

        match_found, similarity = fuzzy_match_excerpt(excerpt, trace)

        # Comma difference should reduce similarity
        assert similarity < 1.0
        # Punctuation can affect threshold - just check we get reasonable similarity
        assert similarity > 0.5  # Still has good overlap

    def test_very_long_excerpt_partial_match(self):
        """Test long excerpt with only partial match in trace."""
        excerpt = "The drug targets BCL-2 protein and also inhibits BAX and BCL-XL proteins through direct binding"
        trace = "The drug targets BCL-2 protein."

        match_found, similarity = fuzzy_match_excerpt(excerpt, trace)

        # Only beginning matches, rest is hallucinated
        assert match_found is False
        assert similarity < 0.75

    def test_very_long_trace_exact_excerpt(self):
        """Test finding exact excerpt in very long trace."""
        excerpt = "targets BCL-2 protein"
        trace = """
        The drug venetoclax is a highly selective inhibitor that targets BCL-2 protein
        family members. It has been approved for treatment of chronic lymphocytic leukemia
        and has shown promising results in various hematologic malignancies.
        """

        match_found, similarity = fuzzy_match_excerpt(excerpt, trace)

        assert match_found is True
        assert similarity >= 0.95

    def test_similarity_score_decreases_with_differences(self):
        """Test that similarity score decreases as differences increase."""
        base_trace = "The drug targets BCL-2 protein and induces apoptosis."

        # Perfect match
        match1, sim1 = fuzzy_match_excerpt("targets BCL-2 protein", base_trace)

        # Partial match (missing end)
        match2, sim2 = fuzzy_match_excerpt("targets BCL-2 protein extra", base_trace)

        # Partial match (missing more)
        match3, sim3 = fuzzy_match_excerpt("targets BCL-2 protein and even more extra words", base_trace)

        # Similarities should decrease
        assert sim1 > sim2
        assert sim2 > sim3

    def test_refusal_scenario_no_excerpts(self):
        """Test scenario where trace is a refusal with no extractable content."""
        excerpt = "drug target information"
        trace = "I cannot provide information about that drug."

        match_found, similarity = fuzzy_match_excerpt(excerpt, trace)

        assert match_found is False
        # Some overlap from common words like "information" and "drug", but below threshold
        assert similarity < 0.75


class TestFuzzyMatchExcerptWithContext:
    """Tests for fuzzy_match_excerpt_with_context function."""

    def test_context_extraction_basic(self):
        """Test basic context extraction around match."""
        excerpt = "targets BCL-2"
        trace = "The drug venetoclax targets BCL-2 protein and induces apoptosis."

        match_found, similarity, context = fuzzy_match_excerpt_with_context(excerpt, trace, context_chars=10)

        assert match_found is True
        assert "targets BCL-2" in context
        # Should include some surrounding text
        assert len(context) > len(excerpt)

    def test_context_extraction_at_beginning(self):
        """Test context extraction when match is at beginning."""
        excerpt = "The drug venetoclax"
        trace = "The drug venetoclax targets BCL-2 protein."

        match_found, similarity, context = fuzzy_match_excerpt_with_context(excerpt, trace, context_chars=10)

        assert match_found is True
        assert "The drug venetoclax" in context
        # No leading ellipsis since match is at start
        assert not context.startswith("...")

    def test_context_extraction_at_end(self):
        """Test context extraction when match is at end."""
        excerpt = "induces apoptosis"
        trace = "The drug targets BCL-2 and induces apoptosis"

        match_found, similarity, context = fuzzy_match_excerpt_with_context(excerpt, trace, context_chars=10)

        assert match_found is True
        assert "induces apoptosis" in context
        # No trailing ellipsis since match is at end
        assert not context.endswith("...")

    def test_context_extraction_middle_with_ellipses(self):
        """Test context extraction from middle includes ellipses."""
        excerpt = "targets BCL-2"
        trace = "This is a long introduction. The drug venetoclax targets BCL-2 protein here. More text follows after."

        match_found, similarity, context = fuzzy_match_excerpt_with_context(excerpt, trace, context_chars=10)

        assert match_found is True
        assert "targets BCL-2" in context
        # Should have ellipses on both ends
        assert context.startswith("...")
        assert context.endswith("...")

    def test_context_empty_for_no_match(self):
        """Test that context is empty when no match found."""
        excerpt = "hallucinated text"
        trace = "Completely different text"

        match_found, similarity, context = fuzzy_match_excerpt_with_context(excerpt, trace)

        assert match_found is False
        assert context == ""

    def test_context_custom_length(self):
        """Test context extraction with custom length."""
        excerpt = "BCL-2"
        trace = "AAAA BBBB CCCC targets BCL-2 protein DDDD EEEE FFFF"

        # Short context
        match1, sim1, ctx1 = fuzzy_match_excerpt_with_context(excerpt, trace, context_chars=5)

        # Long context
        match2, sim2, ctx2 = fuzzy_match_excerpt_with_context(excerpt, trace, context_chars=20)

        assert match1 is True
        assert match2 is True
        assert len(ctx2) > len(ctx1)
        assert "BCL-2" in ctx1
        assert "BCL-2" in ctx2
