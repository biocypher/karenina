"""Unit tests for the replay exception hierarchy."""

from __future__ import annotations

import pytest

from karenina.exceptions import KareninaError
from karenina.replay.exceptions import (
    ReplayError,
    ReplayHydrationError,
    ReplayMissError,
    ReplayPersistenceError,
)


@pytest.mark.unit
class TestReplayExceptionHierarchy:
    def test_replay_error_is_karenina_error(self):
        assert issubclass(ReplayError, KareninaError)

    def test_all_subclasses_inherit_from_replay_error(self):
        assert issubclass(ReplayMissError, ReplayError)
        assert issubclass(ReplayHydrationError, ReplayError)
        assert issubclass(ReplayPersistenceError, ReplayError)

    def test_miss_error_carries_key(self):
        class FakeKey:
            pass

        key = FakeKey()
        exc = ReplayMissError("missing entry", key=key)
        assert exc.key is key
        assert "missing entry" in str(exc)

    def test_miss_error_chain_from(self):
        inner = RuntimeError("boom")
        with pytest.raises(ReplayMissError) as info:
            raise ReplayMissError("miss", key=None) from inner
        assert info.value.__cause__ is inner

    def test_hydration_error_carries_fields_and_validation_error(self):
        inner = ValueError("bad field")
        fields = {"mechanism": "x"}
        exc = ReplayHydrationError("hydration failed", captured_fields=fields, inner=inner)
        assert exc.captured_fields is fields
        assert exc.inner is inner

    def test_persistence_error_simple(self):
        exc = ReplayPersistenceError("bad version")
        assert "bad version" in str(exc)

    def test_re_exported_from_karenina_exceptions(self):
        from karenina.exceptions import (
            ReplayError as ReExportedReplayError,
        )
        from karenina.exceptions import (
            ReplayHydrationError as ReExportedHydrationError,
        )
        from karenina.exceptions import (
            ReplayMissError as ReExportedMissError,
        )
        from karenina.exceptions import (
            ReplayPersistenceError as ReExportedPersistenceError,
        )

        assert ReExportedReplayError is ReplayError
        assert ReExportedMissError is ReplayMissError
        assert ReExportedHydrationError is ReplayHydrationError
        assert ReExportedPersistenceError is ReplayPersistenceError


@pytest.mark.unit
class TestProjectionError:
    def test_projection_error_is_replay_error(self):
        from karenina.exceptions import ProjectionError, ReplayError

        assert issubclass(ProjectionError, ReplayError)

    def test_projection_error_stores_report_attribute(self):
        from karenina.exceptions import ProjectionError

        err = ProjectionError("bad projection", report="sentinel")
        assert err.message == "bad projection"
        assert err.report == "sentinel"

    def test_projection_error_report_defaults_to_none(self):
        from karenina.exceptions import ProjectionError

        err = ProjectionError("bad projection")
        assert err.report is None

    def test_projection_error_available_from_replay_namespace(self):
        from karenina.exceptions import ProjectionError as C
        from karenina.replay import ProjectionError as A
        from karenina.replay.exceptions import ProjectionError as B

        assert A is B is C
