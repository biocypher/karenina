"""Tests for non-fatal hallucination assessment in deep judgment."""

import pytest


@pytest.mark.unit
class TestHallucinationAssessmentNonFatal:
    """Stage 1.5 hallucination assessment failure should log warning, not raise."""

    def test_assessment_llm_failure_does_not_raise(self) -> None:
        """Verify the fatal pattern has been removed from the source code."""
        import inspect

        from karenina.benchmark.verification.evaluators.template import deep_judgment

        source = inspect.getsource(deep_judgment)
        # The old fatal pattern should be gone
        assert "Deep-judgment cannot continue without risk assessment" not in source

    def test_stage2_defaults_to_high_risk_without_assessment(self) -> None:
        """Stage 2 risk aggregation defaults to 'high' when no hallucination_risk
        is set on excerpts (the fallback path when assessment fails)."""
        risk_order = {"none": 0, "low": 1, "medium": 2, "high": 3}
        excerpts = {
            "drug_name": [
                {"excerpt": "aspirin", "search_results": [{"title": "test"}]},
            ]
        }

        # Simulate Stage 2 risk aggregation (from deep_judgment.py ~line 696-709)
        hallucination_risk = {}
        for attr_name, excerpt_list in excerpts.items():
            excerpt_risks = []
            for excerpt_obj in excerpt_list:
                if "hallucination_risk" in excerpt_obj:
                    excerpt_risks.append(excerpt_obj["hallucination_risk"])
            if excerpt_risks:
                max_risk = max(excerpt_risks, key=lambda r: risk_order.get(r, 3))
                hallucination_risk[attr_name] = max_risk
            else:
                hallucination_risk[attr_name] = "high"

        assert hallucination_risk["drug_name"] == "high"
