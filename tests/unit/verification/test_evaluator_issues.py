"""Tests for template evaluator bug fixes.

Issue 052: verify() return type unchecked; truthy non-bool values pass.
"""

import logging

import pytest


@pytest.mark.unit
class TestVerifyFieldsReturnTypeCheck:
    """Issue 052: verify_fields should warn when verify() returns non-bool truthy value."""

    def test_verify_fields_warns_on_non_bool_truthy_return(self, caplog):
        """verify() returning a truthy non-bool (e.g. 'yes') should log a warning."""
        from karenina.benchmark.verification.evaluators.template.evaluator import TemplateEvaluator

        class FakeAnswer:
            def verify(self):
                return "yes"

        evaluator = TemplateEvaluator.__new__(TemplateEvaluator)

        with caplog.at_level(logging.WARNING):
            result = evaluator.verify_fields(FakeAnswer())

        # The result should still capture the truthy value
        assert result.success == "yes"
        # But a warning should have been logged about non-bool return
        assert any(
            "non-bool" in record.message.lower() or "not a bool" in record.message.lower() for record in caplog.records
        ), f"Expected warning about non-bool return, got: {[r.message for r in caplog.records]}"

    def test_verify_fields_no_warning_on_bool_true(self, caplog):
        """verify() returning True should not produce a warning."""
        from karenina.benchmark.verification.evaluators.template.evaluator import TemplateEvaluator

        class FakeAnswer:
            def verify(self):
                return True

        evaluator = TemplateEvaluator.__new__(TemplateEvaluator)

        with caplog.at_level(logging.WARNING):
            result = evaluator.verify_fields(FakeAnswer())

        assert result.success is True
        # No non-bool warning
        assert not any(
            "non-bool" in record.message.lower() or "not a bool" in record.message.lower() for record in caplog.records
        )

    def test_verify_fields_no_warning_on_bool_false(self, caplog):
        """verify() returning False should not produce a warning."""
        from karenina.benchmark.verification.evaluators.template.evaluator import TemplateEvaluator

        class FakeAnswer:
            def verify(self):
                return False

        evaluator = TemplateEvaluator.__new__(TemplateEvaluator)

        with caplog.at_level(logging.WARNING):
            result = evaluator.verify_fields(FakeAnswer())

        assert result.success is False
        assert not any(
            "non-bool" in record.message.lower() or "not a bool" in record.message.lower() for record in caplog.records
        )

    def test_verify_fields_warns_on_integer_return(self, caplog):
        """verify() returning 1 (int, truthy but not bool) should log a warning."""
        from karenina.benchmark.verification.evaluators.template.evaluator import TemplateEvaluator

        class FakeAnswer:
            def verify(self):
                return 1

        evaluator = TemplateEvaluator.__new__(TemplateEvaluator)

        with caplog.at_level(logging.WARNING):
            result = evaluator.verify_fields(FakeAnswer())

        assert result.success == 1
        assert any(
            "non-bool" in record.message.lower() or "not a bool" in record.message.lower() for record in caplog.records
        )
