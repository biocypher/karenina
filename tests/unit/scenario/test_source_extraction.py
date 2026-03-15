"""Tests for callable source extraction."""

import pytest

from karenina.scenario.source_extraction import extract_callable_source


@pytest.mark.unit
class TestExtractCallableSource:
    def test_extract_from_named_function(self):
        def my_func(acc, parsed):
            return {**acc, "x": parsed.get("x")}

        source = extract_callable_source(my_func)
        assert source is not None
        assert "parsed.get" in source

    def test_string_passthrough(self):
        """When given a string, return it as-is."""
        src = "lambda acc, p: {**acc, 'x': p.get('x')}"
        result = extract_callable_source(src)
        assert result == src

    def test_none_returns_none(self):
        result = extract_callable_source(None)
        assert result is None

    def test_extraction_failure_raises(self):
        """Built-in functions cannot have source extracted."""
        with pytest.raises(ValueError, match="extract.*source"):
            extract_callable_source(len)
