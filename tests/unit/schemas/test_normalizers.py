"""Tests for text normalizers used by verification primitives."""

import pytest

from karenina.schemas.primitives import (
    SynonymMap,
    apply_normalizer,
    apply_normalizers,
)


@pytest.mark.unit
class TestApplyNormalizer:
    """Test individual normalizer functions."""

    def test_lowercase(self):
        assert apply_normalizer("lowercase", "BCL2") == "bcl2"

    def test_lowercase_empty(self):
        assert apply_normalizer("lowercase", "") == ""

    def test_strip(self):
        assert apply_normalizer("strip", "  BCL2  ") == "BCL2"

    def test_strip_tabs_newlines(self):
        assert apply_normalizer("strip", "\t BCL2 \n") == "BCL2"

    def test_remove_punctuation(self):
        assert apply_normalizer("remove_punctuation", "BCL-2.") == "BCL2"

    def test_remove_punctuation_preserves_spaces(self):
        assert apply_normalizer("remove_punctuation", "a, b; c") == "a b c"

    def test_remove_punctuation_leaves_digits(self):
        assert apply_normalizer("remove_punctuation", "IC50=3.5nM") == "IC5035nM"

    def test_collapse_whitespace(self):
        assert apply_normalizer("collapse_whitespace", "a  b\n\nc") == "a b c"

    def test_collapse_whitespace_strips_edges(self):
        assert apply_normalizer("collapse_whitespace", "  a  ") == "a"

    def test_unknown_normalizer_raises(self):
        with pytest.raises(ValueError, match="Unknown normalizer"):
            apply_normalizer("unknown", "test")


@pytest.mark.unit
class TestSynonymMap:
    """Test SynonymMap normalizer."""

    def test_basic_mapping(self):
        syn = SynonymMap(mapping={"Bcl-2": "BCL2", "bcl2": "BCL2"})
        assert apply_normalizer(syn, "Bcl-2") == "BCL2"

    def test_no_match_returns_original(self):
        syn = SynonymMap(mapping={"Bcl-2": "BCL2"})
        assert apply_normalizer(syn, "TP53") == "TP53"

    def test_empty_mapping(self):
        syn = SynonymMap(mapping={})
        assert apply_normalizer(syn, "anything") == "anything"


@pytest.mark.unit
class TestApplyNormalizers:
    """Test composed normalizer chains."""

    def test_lowercase_then_strip(self):
        assert apply_normalizers(["lowercase", "strip"], "  BCL2  ") == "bcl2"

    def test_synonym_then_lowercase(self):
        syn = SynonymMap(mapping={"Bcl-2": "BCL2"})
        assert apply_normalizers([syn, "lowercase"], "Bcl-2") == "bcl2"

    def test_empty_list_returns_original(self):
        assert apply_normalizers([], "BCL2") == "BCL2"
