"""Unit tests for completed verification run repair."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

import pytest

from karenina.benchmark import ResultsIOManager
from karenina.benchmark.verification.repair import (
    RepairSelection,
    repair_results_export,
    select_repair_rows,
    splice_repaired_rows,
)
from karenina.schemas.config import ModelConfig
from karenina.schemas.results import VerificationResultSet
from karenina.schemas.results.failure import Failure, FailureCategory
from karenina.schemas.verification import (
    ModelIdentity,
    VerificationConfig,
    VerificationResult,
    VerificationResultMetadata,
    VerificationResultTemplate,
)


def _models() -> tuple[ModelConfig, ModelConfig]:
    answerer = ModelConfig(id="answerer", model_name="answerer", interface="openai_endpoint")
    parser = ModelConfig(id="parser", model_name="parser", interface="openai_endpoint")
    return answerer, parser


def _config() -> VerificationConfig:
    answerer, parser = _models()
    return VerificationConfig(answering_models=[answerer], parsing_models=[parser], replicate_count=2)


def _row(question_id: str, replicate: int, *, failed: bool, trace: str = "old") -> VerificationResult:
    answerer, parser = _models()
    answering = ModelIdentity.from_model_config(answerer)
    parsing = ModelIdentity.from_model_config(parser, role="parsing")
    timestamp = datetime(2026, 8, 12, 10, replicate, tzinfo=UTC).isoformat()
    failure = Failure(category=FailureCategory.CONTENT, stage="VerifyTemplate", reason="wrong") if failed else None
    return VerificationResult(
        metadata=VerificationResultMetadata(
            question_id=question_id,
            template_id="template",
            question_text="Question",
            answering=answering,
            parsing=parsing,
            replicate=replicate,
            run_name="comparison",
            execution_time=0.1,
            timestamp=timestamp,
            failure=failure,
            result_id=VerificationResultMetadata.compute_result_id(
                question_id=question_id,
                answering=answering,
                parsing=parsing,
                replicate=replicate,
                timestamp=timestamp,
            ),
        ),
        template=VerificationResultTemplate(raw_llm_response=trace),
    )


def _write_results(path: Path, rows: list[VerificationResult]) -> None:
    payload = {
        "format_version": "2.2",
        "metadata": {},
        "shared_data": {},
        "results": [row.model_dump(mode="json") for row in rows],
    }
    path.write_text(json.dumps(payload), encoding="utf-8")


class _Benchmark:
    def __init__(self, replacements: list[VerificationResult]) -> None:
        self.replacements = replacements
        self.calls: list[dict[str, object]] = []

    def run_verification(self, **kwargs: object) -> VerificationResultSet:
        self.calls.append(kwargs)
        return VerificationResultSet(results=self.replacements)


@pytest.mark.unit
class TestRepairSelection:
    def test_combines_failure_and_replicate_filters(self) -> None:
        rows = [_row("q1", 1, failed=True), _row("q1", 2, failed=False)]
        selection = RepairSelection(failure_groups={"content"}, replicates={1})

        assert select_repair_rows(rows, selection) == [rows[0]]

    def test_requires_filter_or_explicit_all(self) -> None:
        with pytest.raises(ValueError, match="requires"):
            select_repair_rows([_row("q1", 1, failed=True)], RepairSelection())


@pytest.mark.unit
class TestSpliceRepairedRows:
    def test_replaces_exact_identity_and_preserves_order(self) -> None:
        original = [_row("q1", 1, failed=True), _row("q1", 2, failed=False)]
        replacement = _row("q1", 1, failed=False, trace="new")

        merged = splice_repaired_rows(original, [replacement])

        assert [row.template.raw_llm_response for row in merged if row.template] == ["new", "old"]

    def test_rejects_duplicate_source_identity(self) -> None:
        row = _row("q1", 1, failed=True)
        with pytest.raises(ValueError, match="Duplicate"):
            splice_repaired_rows([row, row], [])


@pytest.mark.unit
class TestRepairResultsExport:
    def test_dry_run_does_not_call_benchmark_or_write_files(self, tmp_path: Path) -> None:
        source = tmp_path / "results.json"
        rows = [_row("q1", 1, failed=True), _row("q1", 2, failed=False)]
        _write_results(source, rows)
        benchmark = _Benchmark([])

        outcome = repair_results_export(
            benchmark,
            source,
            _config(),
            RepairSelection(failure_groups={"content"}),
            dry_run=True,
        )

        assert outcome.selected_count == 1
        assert outcome.replaced_count == 0
        assert benchmark.calls == []
        assert list(tmp_path.iterdir()) == [source]

    def test_replay_repairs_in_place_with_backup_and_provenance(self, tmp_path: Path) -> None:
        source = tmp_path / "results.json"
        original = [_row("q1", 1, failed=True), _row("q1", 2, failed=False)]
        replacement = _row("q1", 1, failed=False, trace="repaired")
        _write_results(source, original)
        benchmark = _Benchmark([replacement])

        outcome = repair_results_export(
            benchmark,
            source,
            _config(),
            RepairSelection(failure_groups={"content"}),
            now=datetime(2026, 8, 12, 12, tzinfo=UTC),
        )

        assert outcome.backup_path is not None and outcome.backup_path.exists()
        assert outcome.provenance_path is not None and outcome.provenance_path.exists()
        repaired = list(ResultsIOManager.iter_from_json(source))
        assert repaired[0].template and repaired[0].template.raw_llm_response == "repaired"
        targeted = benchmark.calls[0]["config"]
        assert isinstance(targeted, VerificationConfig)
        assert targeted.replay_store is not None
        assert targeted.replay_store.miss_policy == "strict"
        assert len(targeted.skip_triples or ()) == 1

    def test_live_mode_does_not_attach_replay_store(self, tmp_path: Path) -> None:
        source = tmp_path / "results.json"
        selected = _row("q1", 1, failed=True)
        _write_results(source, [selected])
        benchmark = _Benchmark([_row("q1", 1, failed=False, trace="live")])

        repair_results_export(
            benchmark,
            source,
            _config(),
            RepairSelection(select_all=True),
            mode="live",
            output_path=tmp_path / "repaired.json",
        )

        targeted = benchmark.calls[0]["config"]
        assert isinstance(targeted, VerificationConfig)
        assert targeted.replay_store is None
