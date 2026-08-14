"""Tests for the paper experiments bootstrap."""

import importlib
from pathlib import Path

import pytest

from paper.common.bootstrap import DATA_ROOT_ENV, data_root

bootstrap_module = importlib.import_module("paper.common.bootstrap")


@pytest.mark.unit
class TestDataRoot:
    def test_unset_variable_and_missing_standard_folders_raise(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
    ) -> None:
        monkeypatch.delenv(DATA_ROOT_ENV, raising=False)
        monkeypatch.setattr(
            bootstrap_module,
            "DEFAULT_DATA_ROOTS",
            (tmp_path / "missing-sibling", tmp_path / "missing-child"),
        )
        with pytest.raises(RuntimeError, match="deposit not found"):
            data_root()

    def test_missing_directory_raises(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        monkeypatch.setenv(DATA_ROOT_ENV, str(tmp_path / "absent"))
        with pytest.raises(RuntimeError, match="missing directory"):
            data_root()

    def test_resolves_existing_directory(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        monkeypatch.setenv(DATA_ROOT_ENV, str(tmp_path))
        assert data_root() == tmp_path.resolve()

    def test_discovers_standard_download_without_local_configuration(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
    ) -> None:
        deposit = tmp_path / "karenina-paper-experiments-data"
        deposit.mkdir()
        monkeypatch.delenv(DATA_ROOT_ENV, raising=False)
        monkeypatch.setattr(
            bootstrap_module,
            "DEFAULT_DATA_ROOTS",
            (deposit, tmp_path / "missing-child"),
        )

        assert data_root() == deposit.resolve()

    def test_discovers_download_inside_repository_as_fallback(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
    ) -> None:
        deposit = tmp_path / "repository" / "karenina-paper-experiments-data"
        deposit.mkdir(parents=True)
        monkeypatch.delenv(DATA_ROOT_ENV, raising=False)
        monkeypatch.setattr(
            bootstrap_module,
            "DEFAULT_DATA_ROOTS",
            (tmp_path / "missing-sibling", deposit),
        )

        assert data_root() == deposit.resolve()

    def test_explicit_environment_override_wins_over_standard_download(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
    ) -> None:
        automatic = tmp_path / "automatic"
        explicit = tmp_path / "explicit"
        automatic.mkdir()
        explicit.mkdir()
        monkeypatch.setattr(bootstrap_module, "DEFAULT_DATA_ROOTS", (automatic,))
        monkeypatch.setenv(DATA_ROOT_ENV, str(explicit))

        assert data_root() == explicit.resolve()
