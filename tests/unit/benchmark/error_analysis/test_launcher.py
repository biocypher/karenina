"""Tests for the launcher Protocol + registry and the prepare-only launcher."""

from __future__ import annotations

import pytest

from karenina.benchmark.error_analysis.exceptions import (
    LauncherNoOutputError,
    LauncherNotFoundError,
)
from karenina.benchmark.error_analysis.launcher import (
    get_launcher,
    list_launchers,
    register_launcher,
)
from karenina.benchmark.error_analysis.launchers.prepare_only import PrepareOnlyLauncher


@pytest.mark.unit
class TestRegistry:
    def test_prepare_only_always_registered(self):
        assert "prepare-only" in list_launchers()

    def test_get_launcher_returns_class(self):
        cls = get_launcher("prepare-only")
        assert cls is PrepareOnlyLauncher

    def test_unknown_launcher_raises_with_registered_names(self):
        with pytest.raises(LauncherNotFoundError) as exc_info:
            get_launcher("no-such-launcher")
        assert "prepare-only" in str(exc_info.value)

    def test_register_launcher_persists(self):
        class DummyLauncher:
            def run(self, analysis_dir, **_):
                return analysis_dir / "REPORT.md"

        register_launcher("dummy-test", DummyLauncher)
        try:
            assert "dummy-test" in list_launchers()
            assert get_launcher("dummy-test") is DummyLauncher
        finally:
            # Avoid cross-test pollution.
            from karenina.benchmark.error_analysis.launcher import _REGISTRY

            _REGISTRY.pop("dummy-test", None)


@pytest.mark.unit
class TestPrepareOnlyLauncher:
    def test_returns_report_path_when_file_exists(self, tmp_path):
        (tmp_path / "REPORT.md").write_text("written by hand")
        path = PrepareOnlyLauncher().run(tmp_path)
        assert path == tmp_path / "REPORT.md"

    def test_raises_when_report_missing(self, tmp_path):
        with pytest.raises(LauncherNoOutputError):
            PrepareOnlyLauncher().run(tmp_path)
