"""Read-only probes for OpenAI-compatible model endpoints."""

from __future__ import annotations

import json
import re
import urllib.error
import urllib.request
from dataclasses import dataclass
from time import monotonic

from pydantic import SecretStr


@dataclass(frozen=True)
class EndpointProbeResult:
    """Outcome of querying one OpenAI-compatible models endpoint."""

    endpoint: str
    models_url: str
    reachable: bool
    models: tuple[str, ...]
    elapsed_seconds: float
    error: str | None = None

    def exposes(self, model_name: str) -> bool:
        """Return whether the endpoint advertises ``model_name``."""

        return self.reachable and model_name in self.models


def _models_url(endpoint: str) -> str:
    base = endpoint.rstrip("/")
    return f"{base}/models" if re.search(r"/v\d+$", base) else f"{base}/v1/models"


def probe_openai_endpoint(
    endpoint: str,
    *,
    api_key: str | SecretStr | None = None,
    timeout: float = 5.0,
) -> EndpointProbeResult:
    """Query an OpenAI-compatible ``models`` endpoint without calling a model.

    Network and response errors are represented in the returned result. API
    keys are sent only as an authorization header and never included in errors.
    """

    url = _models_url(endpoint)
    headers = {"Accept": "application/json"}
    if api_key is not None:
        value = api_key.get_secret_value() if isinstance(api_key, SecretStr) else api_key
        headers["Authorization"] = f"Bearer {value}"
    request = urllib.request.Request(url, headers=headers)
    started = monotonic()
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            payload = json.loads(response.read().decode("utf-8"))
        raw_models = payload.get("data") if isinstance(payload, dict) else None
        if not isinstance(raw_models, list):
            raise ValueError("response does not contain a model data list")
        models = tuple(str(item["id"]) for item in raw_models if isinstance(item, dict) and item.get("id") is not None)
        return EndpointProbeResult(
            endpoint=endpoint,
            models_url=url,
            reachable=True,
            models=models,
            elapsed_seconds=monotonic() - started,
        )
    except (
        TimeoutError,
        OSError,
        urllib.error.URLError,
        urllib.error.HTTPError,
        json.JSONDecodeError,
        ValueError,
    ) as e:
        return EndpointProbeResult(
            endpoint=endpoint,
            models_url=url,
            reachable=False,
            models=(),
            elapsed_seconds=monotonic() - started,
            error=f"{type(e).__name__}: {e}",
        )


def select_openai_endpoint(
    candidates: list[str] | tuple[str, ...],
    model_name: str,
    *,
    api_key: str | SecretStr | None = None,
    timeout: float = 5.0,
) -> str:
    """Return the first reachable candidate advertising ``model_name``."""

    attempted: list[str] = []
    for endpoint in candidates:
        attempted.append(endpoint)
        result = probe_openai_endpoint(endpoint, api_key=api_key, timeout=timeout)
        if result.exposes(model_name):
            return endpoint
    rendered = ", ".join(attempted) or "<none>"
    raise RuntimeError(f"No reachable endpoint exposes {model_name}. Tried: {rendered}")
