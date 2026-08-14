"""Tests for OpenAI-compatible endpoint probing."""

from __future__ import annotations

import io
import urllib.error

import pytest

from karenina.utils import probe_openai_endpoint, select_openai_endpoint


class _Response:
    def __init__(self, payload: bytes) -> None:
        self._payload = payload

    def __enter__(self) -> _Response:
        return self

    def __exit__(self, *_args: object) -> None:
        return None

    def read(self) -> bytes:
        return self._payload


@pytest.mark.unit
class TestEndpointProbe:
    def test_reports_advertised_models_and_authorization_header(self, monkeypatch):
        captured: dict[str, object] = {}

        def fake_urlopen(request, timeout):
            captured["url"] = request.full_url
            captured["authorization"] = request.headers.get("Authorization")
            captured["timeout"] = timeout
            return _Response(b'{"data": [{"id": "glm-5.1"}]}')

        monkeypatch.setattr("karenina.utils.endpoint_probe.urllib.request.urlopen", fake_urlopen)

        result = probe_openai_endpoint(
            "https://example.test/v1",
            api_key="secret",
            timeout=2.5,
        )

        assert result.reachable is True
        assert result.exposes("glm-5.1") is True
        assert captured == {
            "url": "https://example.test/v1/models",
            "authorization": "Bearer secret",
            "timeout": 2.5,
        }

    def test_preserves_versioned_provider_base_path(self, monkeypatch):
        captured: list[str] = []

        def fake_urlopen(request, timeout):
            captured.append(request.full_url)
            return _Response(b'{"data": [{"id": "glm-5.1"}]}')

        monkeypatch.setattr("karenina.utils.endpoint_probe.urllib.request.urlopen", fake_urlopen)

        probe_openai_endpoint("https://api.z.ai/api/coding/paas/v4")

        assert captured == ["https://api.z.ai/api/coding/paas/v4/models"]

    def test_returns_failure_without_exposing_key(self, monkeypatch):
        def fail(_request, timeout):
            raise urllib.error.HTTPError(
                "https://example.test/v1/models",
                401,
                "unauthorized",
                {},
                io.BytesIO(),
            )

        monkeypatch.setattr("karenina.utils.endpoint_probe.urllib.request.urlopen", fail)

        result = probe_openai_endpoint("https://example.test", api_key="secret")

        assert result.reachable is False
        assert result.models == ()
        assert "secret" not in (result.error or "")

    def test_selects_first_candidate_exposing_model(self, monkeypatch):
        def fake_urlopen(request, timeout):
            model = "other" if "first" in request.full_url else "glm-5.1"
            return _Response(f'{{"data": [{{"id": "{model}"}}]}}'.encode())

        monkeypatch.setattr("karenina.utils.endpoint_probe.urllib.request.urlopen", fake_urlopen)

        selected = select_openai_endpoint(
            ["https://first.test", "https://second.test"],
            "glm-5.1",
        )

        assert selected == "https://second.test"
