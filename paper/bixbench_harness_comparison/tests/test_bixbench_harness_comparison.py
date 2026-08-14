"""Synthetic tests for the BixBench harness comparison."""

from __future__ import annotations

import csv
import json
import sys
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace

import pytest

from karenina.benchmark import ResultsIOManager
from karenina.schemas.results.failure import Failure, FailureCategory
from karenina.schemas.verification import (
    ModelIdentity,
    VerificationResult,
    VerificationResultMetadata,
    VerificationResultTemplate,
)
from paper.bixbench_harness_comparison import smoke as bixbench_smoke
from paper.bixbench_harness_comparison.analysis import (
    ResultSource,
    build_comparison_summary,
    build_result_tables,
    normalize_archive_harness,
)
from paper.bixbench_harness_comparison.config import (
    ExperimentCondition,
    build_config,
    selected_conditions,
)
from paper.bixbench_harness_comparison.rubrics import (
    TraceFailureBurdens,
    build_trace_anchors,
    trace_text,
)
from paper.bixbench_harness_comparison.run import (
    discover_archived_sources,
)
from paper.bixbench_harness_comparison.run import (
    run as run_experiment,
)


def _result(question_id: str, *, failed: bool = False) -> VerificationResult:
    answering = ModelIdentity(interface="claude_agent_sdk", model_name="glm-5.1")
    parsing = ModelIdentity(interface="claude_agent_sdk", model_name="glm-5.1")
    timestamp = datetime.now(UTC).isoformat()
    failure = (
        Failure(category=FailureCategory.TIMEOUT, stage="GenerateAnswer", reason="timeout")
        if failed
        else None
    )
    return VerificationResult(
        metadata=VerificationResultMetadata(
            question_id=question_id,
            template_id="template",
            question_text="Analyze the workspace",
            answering=answering,
            parsing=parsing,
            replicate=1,
            run_name="friendly-run",
            execution_time=2.0,
            timestamp=timestamp,
            failure=failure,
            result_id=VerificationResultMetadata.compute_result_id(
                question_id=question_id,
                answering=answering,
                parsing=parsing,
                replicate=1,
                timestamp=timestamp,
            ),
        ),
        template=VerificationResultTemplate(
            raw_llm_response="answer",
            trace_messages=[{"role": "assistant", "content": "answer"}],
            parsed_gt_response={"numeric": 10.0, "boolean": True},
            field_results={} if failed else {"numeric": False, "boolean": True},
            field_scores={} if failed else {"numeric": 0.75, "boolean": 1.0},
            agent_metrics={"iterations": 2, "tool_calls": 1},
        ),
    )


@pytest.mark.unit
class TestBixBenchConfig:
    def test_glm_subscription_builds_both_harnesses_without_codon(self, monkeypatch, tmp_path):
        monkeypatch.setenv("ZAI_API_KEY", "secret")
        conditions = selected_conditions("glm", "both")

        configs = [
            build_config(
                condition,
                runtime="host",
                replicates=3,
                workers=2,
                timeout=900,
                workspace_output_dir=tmp_path / condition.slug,
            )
            for condition in conditions
        ]

        assert [condition.harness for condition in conditions] == [
            "Claude Code",
            "DeepAgents",
        ]
        assert [config.answering_models[0].interface for config in configs] == [
            "claude_agent_sdk",
            "langchain_deep_agents",
        ]
        assert all(config.answering_models[0].model_name == "glm-5.1" for config in configs)
        assert all(config.parsing_models[0].model_name == "glm-5.1" for config in configs)
        assert all(
            config.parsing_models[0].interface == "langchain_deep_agents"
            for config in configs
        )
        assert all(
            config.parsing_models[0].agent_runtime is not None
            and config.parsing_models[0].agent_runtime.read_max_bytes == 20_000
            for config in configs
        )
        assert all(config.replicate_count == 3 for config in configs)
        assert all(config.agentic_parsing_trigger == "dynamic" for config in configs)
        assert all(config.allow_partial_trace_scoring is True for config in configs)
        assert all(config.retry_policy.timeout.max_attempts == 0 for config in configs)
        assert all(config.retry_policy.connection.max_attempts == 3 for config in configs)

    def test_archive_discovery_requires_three_replicates(self, tmp_path: Path):
        condition = ExperimentCondition(model="GLM-5.1", harness="Claude Code")
        for replicate in range(1, 4):
            run_dir = tmp_path / f"timestamp__glm51-csdk__judge-glm51-csdk__rep{replicate}"
            run_dir.mkdir()
            (run_dir / "results.json").write_text("{}")

        sources = discover_archived_sources(tmp_path, [condition])

        assert [source.replicate_override for source in sources] == [1, 2, 3]
        assert all(source.condition == condition for source in sources)

    def test_explicit_stored_judgment_path_is_fully_offline(
        self,
        monkeypatch,
        tmp_path: Path,
    ) -> None:
        runs_dir = tmp_path / "runs"
        runs_dir.mkdir()
        for replicate in range(1, 4):
            run_dir = runs_dir / f"timestamp__glm51-csdk__judge-glm51-csdk__rep{replicate}"
            run_dir.mkdir()
            result = _result(f"bix-{replicate}")
            (run_dir / "results.json").write_text(
                ResultsIOManager.export_to_json({result.metadata.result_id: result})
            )
        burden_path = tmp_path / "burdens.csv"
        with burden_path.open("w", newline="") as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=[
                    "model",
                    "harness",
                    "rep",
                    "task_id",
                    "environment_failure_burden",
                    "data_ingestion_failure_burden",
                    "tool_api_failure_burden",
                    "analysis_code_failure_burden",
                    "repetition_churn_burden",
                    "incompletion_burden",
                    "evidence",
                ],
            )
            writer.writeheader()
            for replicate in range(1, 4):
                writer.writerow(
                    {
                        "model": "glm51",
                        "harness": "csdk",
                        "rep": f"rep{replicate}",
                        "task_id": f"bix-{replicate}",
                        "environment_failure_burden": 0,
                        "data_ingestion_failure_burden": 0,
                        "tool_api_failure_burden": 0,
                        "analysis_code_failure_burden": 0,
                        "repetition_churn_burden": 0,
                        "incompletion_burden": 0,
                        "evidence": "",
                    }
                )
        monkeypatch.setattr(
            "paper.bixbench_harness_comparison.run.input_path",
            lambda relative: runs_dir if relative.endswith("/runs") else burden_path,
        )
        output_root = tmp_path / "output"

        run_experiment(
            output_root,
            reuse_stored_results=False,
            reuse_stored_judgments=True,
            model="glm",
            harness="claude-code",
        )

        manifest = json.loads((output_root / "run_manifest.json").read_text())
        assert manifest["model_calls"] is False
        assert (output_root / "analysis" / "comparison_summary.tsv").is_file()

    def test_smoke_selects_archived_all_condition_pass_task(self, monkeypatch):
        from paper.bixbench_harness_comparison import smoke

        captured: dict[str, object] = {}
        monkeypatch.setattr(smoke, "bootstrap", lambda _verbose: None)
        monkeypatch.setattr(
            smoke,
            "run",
            lambda output_root, **kwargs: captured.update(
                {"output_root": output_root, **kwargs}
            ),
        )
        monkeypatch.setattr(
            smoke,
            "_require_operational_smoke_success",
            lambda output_root: captured.update({"validated_output": output_root}),
        )
        monkeypatch.setattr(sys, "argv", ["smoke"])

        smoke.main()

        assert captured["question_ids"] == ("bix-18",)
        assert captured["model"] == "glm"
        assert captured["harness"] == "both"
        assert captured["validated_output"] == captured["output_root"]

    def test_smoke_rejects_non_content_failures(self, monkeypatch, tmp_path: Path):
        result_path = tmp_path / "result.json"
        (tmp_path / "run_manifest.json").write_text(
            json.dumps({"result_inputs": [str(result_path)]})
        )
        monkeypatch.setattr(
            bixbench_smoke.ResultsIOManager,
            "load_result_set_from_json",
            lambda _path: SimpleNamespace(results=[_result("bix-18", failed=True)]),
        )

        with pytest.raises(RuntimeError, match="timeout at GenerateAnswer"):
            bixbench_smoke._require_operational_smoke_success(tmp_path)

    def test_simplified_runs_live_benchmark_and_persists_result(
        self,
        monkeypatch,
        tmp_path: Path,
    ) -> None:
        from paper.bixbench_harness_comparison import simplified

        result = _result("bix-18")
        benchmark_path = tmp_path / "bix_bench.jsonld"
        output_dir = tmp_path / "simplified-output"
        captured: dict[str, object] = {}

        class FakeBenchmark:
            def run_verification(self, config, **kwargs):
                captured.update({"config": config, **kwargs})
                return SimpleNamespace(results=[result])

        monkeypatch.setattr(
            simplified.Benchmark,
            "load",
            lambda path, **kwargs: captured.update({"path": path, **kwargs})
            or FakeBenchmark(),
        )
        simplified.prepare_output_dir(output_dir)
        rows = simplified.run_live_task(
            benchmark_path=benchmark_path,
            question_id="bix-18",
            config=SimpleNamespace(),  # type: ignore[arg-type]
            output_dir=output_dir,
        )

        stored = list(ResultsIOManager.iter_from_json(output_dir / "results.json"))
        assert rows == [result]
        assert [row.metadata.question_id for row in stored] == ["bix-18"]
        assert captured["question_ids"] == ["bix-18"]
        assert captured["run_name"] == "simplified-live-evaluation"

    def test_simplified_builds_public_karenina_configuration(
        self,
        monkeypatch,
        tmp_path: Path,
    ) -> None:
        from paper.bixbench_harness_comparison.simplified import (
            build_models_and_verification,
        )

        monkeypatch.setenv("ZAI_API_KEY", "secret")
        config, judge = build_models_and_verification(tmp_path, 900)

        assert config.answering_models[0].interface == "claude_agent_sdk"
        assert config.parsing_models[0].interface == "langchain_deep_agents"
        assert config.answering_models[0].agent_runtime.backend == "container"
        assert config.parsing_models[0].agent_runtime.backend == "container"
        assert config.workspace_output_dir == tmp_path / "workspaces"
        assert judge.interface == "langchain_deep_agents"
        assert judge.agent_runtime.backend == "filesystem"

    def test_simplified_refuses_to_overwrite_live_artifacts(self, tmp_path: Path) -> None:
        from paper.bixbench_harness_comparison.simplified import prepare_output_dir

        output_dir = tmp_path / "simplified-output"
        output_dir.mkdir()
        (output_dir / "results.json").write_text("[]")

        with pytest.raises(FileExistsError, match="earlier live artifacts are retained"):
            prepare_output_dir(output_dir)


@pytest.mark.unit
class TestBixBenchAnalysis:
    @pytest.mark.parametrize(
        ("archive_label", "final_label"),
        [
            ("csdk", "Claude Code"),
            ("csdk-high", "Claude Code"),
            ("da", "DeepAgents"),
            ("da-high", "DeepAgents"),
        ],
    )
    def test_archived_harness_aliases_use_configured_labels(self, archive_label, final_label):
        assert normalize_archive_harness(archive_label) == final_label

    def test_missing_timeout_fields_are_scored_as_zero(self, monkeypatch, tmp_path):
        condition = ExperimentCondition(model="GLM-5.1", harness="Claude Code")
        complete = ResultSource(condition=condition, path=tmp_path / "complete.json")
        timeout = ResultSource(condition=condition, path=tmp_path / "timeout.json")
        rows = {complete.path: [_result("bix-1")], timeout.path: [_result("bix-1", failed=True)]}
        monkeypatch.setattr(
            "paper.bixbench_harness_comparison.analysis.iter_source_results",
            lambda source: iter(rows[source.path]),
        )

        fields, tasks = build_result_tables([complete, timeout])
        summary = build_comparison_summary(fields, tasks)

        assert len(fields) == 4
        assert sum(float(row["field_score"]) for row in fields) == 1.75
        assert tasks[1]["field_count"] == 2
        assert tasks[1]["graded_accuracy"] == 0.0
        assert tasks[1]["timed_out"] is True
        assert summary[0]["fields"] == 4

    def test_trace_anchors_and_schema_enforce_ordinal_burdens(self):
        result = _result("bix-1")
        rendered = trace_text(result)
        anchors = build_trace_anchors("Bash command python run.py\n" * 3)

        assert "Raw BixBench answering trace" in rendered
        assert anchors["repeated_command_fingerprints"]
        with pytest.raises(ValueError):
            TraceFailureBurdens(
                environment_failure_burden=4,
                data_ingestion_failure_burden=0,
                tool_api_failure_burden=0,
                analysis_code_failure_burden=0,
                repetition_churn_burden=0,
                incompletion_burden=0,
            )
