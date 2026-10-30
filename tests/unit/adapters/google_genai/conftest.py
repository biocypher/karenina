"""Fixtures for the google_genai cold tests."""

from __future__ import annotations

import pytest

pytest.importorskip("google.genai", reason="google-genai not installed")


@pytest.fixture(autouse=True)
def _clear_google_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Keep the developer's Google environment out of the tests."""
    for name in ("GOOGLE_CLOUD_PROJECT", "GOOGLE_CLOUD_LOCATION", "GEMINI_API_KEY", "GOOGLE_API_KEY"):
        monkeypatch.delenv(name, raising=False)
