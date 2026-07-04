"""Tests for the Caveat enum."""

import pytest

from karenina.schemas.results.caveat import Caveat


@pytest.mark.unit
class TestCaveat:
    def test_values(self):
        assert Caveat.PARTIAL_CONTENT.value == "partial_content"
        assert Caveat.EMBEDDING_OVERRIDE.value == "embedding_override"
        assert Caveat.RETRIES_USED.value == "retries_used"
        assert Caveat.PARSE_DECISION_MALFORMED.value == "parse_decision_malformed"

    def test_is_str_enum(self):
        assert isinstance(Caveat.PARTIAL_CONTENT, str)
