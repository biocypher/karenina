"""Unit tests for verification run directories and manifests."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

import pytest

from karenina.benchmark import create_run_directory, managed_run_directory, mask_run_configuration
from karenina.schemas.config import ModelConfig
from karenina.schemas.verification import VerificationConfig


def _config() -> VerificationConfig:
    return VerificationConfig(
        answering_models=[
            ModelConfig(
                id="answerer",
                model_name="answerer-model",
                interface="openai_endpoint",
                endpoint_base_url="http://example.test/v1",
                endpoint_api_key="private-key",
            )
        ],
        parsing_models=[
            ModelConfig(
                id="parser",
                model_name="parser-model",
                interface="langchain",
                model_provider="anthropic",
                anthropic_api_key="another-private-key",
            )
        ],
        replicate_count=3,
    )


@pytest.mark.unit
class TestRunDirectory:
    def test_creates_standard_paths_and_masked_manifest(self, tmp_path: Path) -> None:
        created = datetime(2026, 8, 12, 10, 11, 12, tzinfo=UTC)

        run = create_run_directory(
            tmp_path,
            "otp_model_comparison",
            benchmark_path="benchmark.jsonld",
            config=_config(),
            selected_question_count=144,
            now=created,
        )

        assert run.path.name == "otp_model_comparison_20260812T101112Z"
        assert run.traces_path.is_dir()
        assert run.workspaces_path.is_dir()
        payload = json.loads(run.manifest_path.read_text(encoding="utf-8"))
        assert payload["status"] == "running"
        assert payload["selected_question_count"] == 144
        assert payload["replicate_count"] == 3
        assert payload["configuration"]["answering_models"][0]["endpoint_api_key"] == "**********"
        assert payload["configuration"]["parsing_models"][0]["anthropic_api_key"] == "**********"
        assert "private-key" not in run.manifest_path.read_text(encoding="utf-8")

    def test_rejects_unsafe_label_and_existing_directory(self, tmp_path: Path) -> None:
        created = datetime(2026, 8, 12, tzinfo=UTC)
        with pytest.raises(ValueError, match="label"):
            create_run_directory(
                tmp_path,
                "../unsafe",
                benchmark_path="benchmark.jsonld",
                config=_config(),
                now=created,
            )

        create_run_directory(
            tmp_path,
            "safe",
            benchmark_path="benchmark.jsonld",
            config=_config(),
            now=created,
        )
        with pytest.raises(FileExistsError):
            create_run_directory(
                tmp_path,
                "safe",
                benchmark_path="benchmark.jsonld",
                config=_config(),
                now=created,
            )

    def test_managed_directory_records_completion(self, tmp_path: Path) -> None:
        with managed_run_directory(
            tmp_path,
            "complete",
            benchmark_path="benchmark.jsonld",
            config=_config(),
        ) as run:
            run.results_path.write_text("{}", encoding="utf-8")

        payload = json.loads(run.manifest_path.read_text(encoding="utf-8"))
        assert payload["status"] == "completed"
        assert payload["completed_at"] is not None

    def test_explicit_resume_reopens_existing_timestamped_directory(self, tmp_path: Path) -> None:
        created = datetime(2026, 8, 12, 10, 11, 12, tzinfo=UTC)
        original = create_run_directory(
            tmp_path,
            "resume",
            benchmark_path="benchmark.jsonld",
            config=_config(),
            now=created,
        )
        original.update_status("interrupted", now=created)

        resumed = create_run_directory(
            tmp_path,
            "resume",
            benchmark_path="benchmark.jsonld",
            config=_config(),
            now=created,
            resume=True,
        )

        assert resumed.path == original.path
        assert resumed.manifest.status == "running"
        assert resumed.manifest.completed_at is None

    def test_managed_directory_records_failure_and_reraises(self, tmp_path: Path) -> None:
        with (
            pytest.raises(RuntimeError, match="boom"),
            managed_run_directory(
                tmp_path,
                "failed",
                benchmark_path="benchmark.jsonld",
                config=_config(),
            ) as run,
        ):
            raise RuntimeError("boom")

        payload = json.loads(run.manifest_path.read_text(encoding="utf-8"))
        assert payload["status"] == "failed"
        assert payload["failure_type"] == "RuntimeError"
        assert payload["failure_message"] == "boom"


@pytest.mark.unit
class TestMaskRunConfiguration:
    def test_masks_nested_secret_like_keys_without_mutating_input(self) -> None:
        source = {
            "models": [{"token": "secret", "name": "model"}],
            "password": None,
        }

        masked = mask_run_configuration(source)

        assert masked == {
            "models": [{"token": "**********", "name": "model"}],
            "password": None,
        }
        assert source["models"][0]["token"] == "secret"
