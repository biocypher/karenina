"""CLI verification must forward config.async_enabled to the batch runner.

Regression: ``karenina verify --no-async`` set ``config.async_enabled=False``
but ``run_verification_with_progress`` never passed it to
``run_verification_batch``, so the runner fell back to the
``KARENINA_ASYNC_ENABLED`` env var (or the True default) and the run stayed
parallel. The config field already absorbs the env var at construction, so
forwarding it is the single source of truth.
"""

from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import pytest

from karenina.benchmark.verification.sinks import ProgressiveFileSink
from karenina.cli import verify_output
from karenina.schemas import VerificationConfig
from karenina.schemas.config import ModelConfig


def _config(async_enabled: bool) -> VerificationConfig:
    return VerificationConfig(
        answering_models=[
            ModelConfig(
                id="a",
                model_provider="mock",
                model_name="mock",
                interface="langchain",
            )
        ],
        parsing_models=[
            ModelConfig(
                id="p",
                model_provider="mock",
                model_name="mock",
                interface="langchain",
            )
        ],
        async_enabled=async_enabled,
    )


@pytest.mark.unit
@pytest.mark.cli
class TestAsyncEnabledForwarding:
    """--no-async must sequentialize the CLI verification run."""

    @pytest.mark.parametrize("async_flag,expected", [(False, False), (True, True)])
    def test_config_async_enabled_forwarded_to_runner(
        self, monkeypatch: pytest.MonkeyPatch, async_flag: bool, expected: bool, tmp_path: Path
    ) -> None:
        captured: dict[str, Any] = {}

        def mock_run(**kwargs: Any) -> MagicMock:
            captured.update(kwargs)
            return MagicMock()

        monkeypatch.setattr(verify_output, "run_verification_batch", mock_run)
        monkeypatch.delenv("KARENINA_ASYNC_ENABLED", raising=False)
        sink = ProgressiveFileSink(
            output_path=tmp_path / "out.json", config=_config(async_flag), benchmark_path=Path("bench.jsonld")
        )

        verify_output.run_verification_with_progress(
            templates=[],
            config=sink.config,
            benchmark=MagicMock(),
            progressive_sink=sink,
            show_progress=False,
        )

        assert captured["async_enabled"] is expected
