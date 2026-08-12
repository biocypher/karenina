"""Tests for the paper experiments bootstrap."""

from pathlib import Path

import pytest

from paper.common.bootstrap import DATA_ROOT_ENV, data_root


@pytest.mark.unit
class TestDataRoot:
    def test_unset_variable_raises(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv(DATA_ROOT_ENV, raising=False)
        with pytest.raises(RuntimeError, match="is not set"):
            data_root()

    def test_missing_directory_raises(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        monkeypatch.setenv(DATA_ROOT_ENV, str(tmp_path / "absent"))
        with pytest.raises(RuntimeError, match="missing directory"):
            data_root()

    def test_resolves_existing_directory(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        monkeypatch.setenv(DATA_ROOT_ENV, str(tmp_path))
        assert data_root() == tmp_path.resolve()
