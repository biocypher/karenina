"""Tests for loading legacy (v2.0/v2.1) verification exports.

Fixture rows are a minimal subset copied from a real legacy export
(local_data/outputs/old/haiku-subsample.json, format_version 2.0): flat
``answering_model``/``parsing_model`` strings, ``answering_replicate``/
``parsing_replicate``, ``completed_without_errors``/``error``, and no
``answering``/``parsing``/``result_id`` metadata.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from karenina.benchmark import ResultsIOManager
from tests.schemas._metadata_factory import make_metadata
from tests.unit.benchmark.core._result_factory import make_result


def _legacy_row(timestamp: str, replicate: int) -> dict[str, object]:
    """One legacy-format result row copied from the real export (shortened prompts)."""
    return {
        "metadata": {
            "question_id": "urn:uuid:8a6fb8bfe479574b424385734cd50a6f",
            "template_id": "c18b2f46979874031b988f2b46a522fa",
            "completed_without_errors": True,
            "error": None,
            "question_text": "Are atopic eczema and Atopic Dermatitites synonyms?",
            "raw_answer": "Yes",
            "keywords": ["Annotation", "Disease"],
            "answering_model": "anthropic/claude-haiku-4-5",
            "parsing_model": "anthropic/claude-haiku-4-5",
            "answering_system_prompt": "You are a biomedical question-answering assistant.",
            "parsing_system_prompt": "Extract the answer fields.",
            "execution_time": 7.85,
            "timestamp": timestamp,
            "run_name": "haiku-subsample",
            "answering_replicate": replicate,
            "parsing_replicate": replicate,
        },
        "template": {
            "raw_llm_response": "Yes",
            "template_verification_performed": True,
            "verify_result": True,
            "verify_granular_result": None,
            "verify_granular_result_details": None,
            "abstention_check_performed": False,
        },
        "rubric": {
            "rubric_evaluation_performed": True,
            "rubric_evaluation_strategy": "batch",
            "llm_trait_scores": {"Consequentiality": True},
            "regex_trait_scores": None,
            "callable_trait_scores": None,
            "metric_trait_scores": None,
        },
        "evaluation_input": "Yes",
        "used_full_trace": False,
        "trace_extraction_error": None,
    }


def _legacy_export() -> dict[str, object]:
    """A minimal legacy (v2.0) export built from the real file's shape."""
    return {
        "format_version": "2.0",
        "metadata": {
            "export_timestamp": "2025-12-12 18:06:47 UTC",
            "karenina_version": "0.1.0",
            "job_id": "55fa4222-bbb5-43e0-9fef-bce316067a48",
        },
        "shared_data": {"rubric_definition": {"llm_traits": []}},
        "results": [
            _legacy_row("2025-12-12 18:06:17", 3),
            _legacy_row("2025-12-12 18:06:25", 1),
        ],
    }


def _write_export(tmp_path: Path, payload: dict[str, object]) -> Path:
    path = tmp_path / "legacy.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


@pytest.mark.unit
class TestLoadLegacyResultSet:
    """Validate migration of legacy (v2.0/v2.1) exports."""

    def test_migrates_rows_with_intact_content(self, tmp_path: Path) -> None:
        path = _write_export(tmp_path, _legacy_export())

        loaded = ResultsIOManager.load_legacy_result_set(path)

        assert len(loaded.results) == 2
        for row, expected_replicate in zip(loaded.results, (3, 1), strict=True):
            assert row.metadata.question_id == "urn:uuid:8a6fb8bfe479574b424385734cd50a6f"
            assert row.template is not None and row.template.verify_result is True
            assert row.rubric is not None and row.rubric.llm_trait_scores == {"Consequentiality": True}
            assert row.metadata.replicate == expected_replicate
        assert loaded.metadata["job_id"] == "55fa4222-bbb5-43e0-9fef-bce316067a48"

    def test_drops_forbidden_keys_and_builds_model_identities(self, tmp_path: Path) -> None:
        path = _write_export(tmp_path, _legacy_export())

        loaded = ResultsIOManager.load_legacy_result_set(path)

        first = loaded.results[0]
        # Validation under extra="forbid" proves the flat keys are gone; the
        # nested ModelIdentity objects replace them.
        assert first.metadata.answering.model_name == "claude-haiku-4-5"
        assert first.metadata.answering.interface == "anthropic"
        assert first.metadata.parsing.model_name == "claude-haiku-4-5"
        assert first.metadata.answering.display_string == "anthropic:claude-haiku-4-5"
        # result_id is synthesized deterministically from question + models +
        # timestamp + replicate: reloading yields identical IDs.
        ids = [row.metadata.result_id for row in loaded.results]
        assert len(set(ids)) == len(ids)
        reloaded = ResultsIOManager.load_legacy_result_set(path)
        assert [row.metadata.result_id for row in reloaded.results] == ids

    def test_corrupted_row_yields_error_naming_index(self, tmp_path: Path) -> None:
        payload = _legacy_export()
        payload["results"][1]["metadata"]["timestamp"] = 12345  # not a string
        path = _write_export(tmp_path, payload)

        with pytest.raises(ValueError, match=r"index 1"):
            ResultsIOManager.load_legacy_result_set(path)

    def test_delegates_non_legacy_export(self, tmp_path: Path) -> None:
        result = make_result(metadata=make_metadata())
        path = tmp_path / "current.json"
        path.write_text(
            json.dumps({"metadata": {}, "results": [result.model_dump(mode="json")]}),
            encoding="utf-8",
        )

        loaded = ResultsIOManager.load_legacy_result_set(path)

        assert len(loaded.results) == 1
        assert loaded.results[0].metadata.result_id == result.metadata.result_id
