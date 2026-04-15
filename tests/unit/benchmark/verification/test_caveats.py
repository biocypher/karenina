"""Unit tests for the orthogonal caveat collector."""

import pytest

from karenina.benchmark.verification.failure_classifier import collect_caveats
from karenina.schemas.results.caveat import Caveat
from tests.unit.benchmark.verification.stages.core._context_factory import make_context


@pytest.mark.unit
class TestCollectCaveats:
    def test_empty_by_default(self):
        assert collect_caveats(make_context()) == []

    def test_partial_content(self):
        ctx = make_context(response_timeout_partial=True)
        assert Caveat.PARTIAL_CONTENT in collect_caveats(ctx)

    def test_embedding_override(self):
        ctx = make_context(embedding_override_applied=True)
        assert Caveat.EMBEDDING_OVERRIDE in collect_caveats(ctx)

    def test_retries_used_fires_on_any_used(self):
        ctx = make_context(retry_counts={"timeout": {"used": 1, "budget": 3}})
        assert Caveat.RETRIES_USED in collect_caveats(ctx)

    def test_retries_used_does_not_fire_when_all_zero(self):
        ctx = make_context(retry_counts={"timeout": {"used": 0, "budget": 3}})
        assert Caveat.RETRIES_USED not in collect_caveats(ctx)

    def test_caveats_independent_of_verdict(self):
        ctx = make_context(
            template_verification_performed=True,
            verify_result=True,
            retry_counts={"timeout": {"used": 1, "budget": 3}},
        )
        assert collect_caveats(ctx) == [Caveat.RETRIES_USED]

    def test_retries_used_tolerates_none(self):
        # Matches the None-safety fix in classify_failure (Task 5 follow-up).
        ctx = make_context(retry_counts={"timeout": {"used": None, "budget": 3}})
        assert Caveat.RETRIES_USED not in collect_caveats(ctx)
