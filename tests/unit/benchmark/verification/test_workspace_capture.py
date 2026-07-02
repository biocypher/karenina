import json
import os
from pathlib import Path

from karenina.benchmark.verification.stages.core.base import ArtifactKeys, VerificationContext
from karenina.benchmark.verification.stages.pipeline.generate_answer import GenerateAnswerStage
from karenina.benchmark.verification.workspace_capture import (
    DEFAULT_WORKSPACE_OUTPUT_EXCLUDES,
    capture_workspace,
    compact_manifest,
    snapshot_workspace,
)
from karenina.schemas.config import ModelConfig
from karenina.schemas.verification import (
    ModelIdentity,
    VerificationResult,
    VerificationResultMetadata,
    VerificationResultTemplate,
)


def _result(question_id: str = "q/1", result_id: str = "rid1") -> VerificationResult:
    identity = ModelIdentity(
        interface="test",
        model_name="model",
        config_id="answering",
    )
    return VerificationResult(
        metadata=VerificationResultMetadata(
            question_id=question_id,
            template_id="template",
            failure=None,
            caveats=[],
            question_text="question",
            answering=identity,
            parsing=identity.model_copy(update={"config_id": "parsing"}),
            execution_time=0.1,
            timestamp="2026-05-20 12:00:00",
            result_id=result_id,
        ),
        template=VerificationResultTemplate(),
    )


def _manifest_rows(output_dir: Path) -> list[dict]:
    return [json.loads(line) for line in (output_dir / "manifest.jsonl").read_text().splitlines() if line.strip()]


def test_produced_capture_copies_new_and_modified_files(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    (workspace / "input.csv").write_text("before")
    (workspace / "unchanged.csv").write_text("same")
    baseline = snapshot_workspace(workspace, DEFAULT_WORKSPACE_OUTPUT_EXCLUDES)

    (workspace / "input.csv").write_text("after")
    (workspace / "result.json").write_text("{}")
    output_dir = tmp_path / "out" / "workspaces"

    capture_workspace(
        mode="produced",
        output_dir=output_dir,
        workspace_path=workspace,
        baseline=baseline,
        result=_result(),
        exclude_patterns=DEFAULT_WORKSPACE_OUTPUT_EXCLUDES,
    )

    captured = output_dir / "q_1__rid1"
    assert (captured / "input.csv").read_text() == "after"
    assert (captured / "result.json").read_text() == "{}"
    assert not (captured / "unchanged.csv").exists()
    assert _manifest_rows(output_dir)[0]["copied_file_count"] == 2


def test_full_capture_honors_excludes_and_preserves_symlinks(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    (workspace / "result.txt").write_text("result")
    (workspace / "empty_dir").mkdir()
    (workspace / ".venv").mkdir()
    (workspace / ".venv" / "installed.txt").write_text("noise")
    target = workspace / "target.txt"
    target.write_text("target")
    os.symlink("target.txt", workspace / "link.txt")
    output_dir = tmp_path / "out" / "workspaces"

    capture_workspace(
        mode="full",
        output_dir=output_dir,
        workspace_path=workspace,
        baseline=None,
        result=_result(result_id="rid2"),
        exclude_patterns=DEFAULT_WORKSPACE_OUTPUT_EXCLUDES,
    )

    captured = output_dir / "q_1__rid2"
    assert (captured / "result.txt").read_text() == "result"
    assert (captured / "empty_dir").is_dir()
    assert not (captured / ".venv").exists()
    assert (captured / "link.txt").is_symlink()
    assert os.readlink(captured / "link.txt") == "target.txt"


def test_capture_replaces_existing_destination_and_compacts_manifest(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    (workspace / "result.txt").write_text("v1")
    output_dir = tmp_path / "out" / "workspaces"
    result = _result(question_id="q1", result_id="same")

    capture_workspace(
        mode="full",
        output_dir=output_dir,
        workspace_path=workspace,
        baseline=None,
        result=result,
        exclude_patterns=DEFAULT_WORKSPACE_OUTPUT_EXCLUDES,
    )
    (workspace / "result.txt").write_text("v2")
    capture_workspace(
        mode="full",
        output_dir=output_dir,
        workspace_path=workspace,
        baseline=None,
        result=result,
        exclude_patterns=DEFAULT_WORKSPACE_OUTPUT_EXCLUDES,
    )

    assert (output_dir / "q1__same" / "result.txt").read_text() == "v2"
    compact_manifest(output_dir)
    manifest = json.loads((output_dir / "manifest.json").read_text())
    assert manifest["format_version"] == "1.0"
    assert len(manifest["entries"]) == 2


def test_missing_workspace_records_skip(tmp_path: Path) -> None:
    output_dir = tmp_path / "out" / "workspaces"
    capture_workspace(
        mode="full",
        output_dir=output_dir,
        workspace_path=tmp_path / "missing",
        baseline=None,
        result=_result(question_id="q1"),
        exclude_patterns=DEFAULT_WORKSPACE_OUTPUT_EXCLUDES,
    )

    row = _manifest_rows(output_dir)[0]
    assert row["status"] == "skipped"
    assert row["skip_reason"] == "workspace_not_available"


def test_full_output_mode_resolves_live_workspace_in_output_dir(tmp_path: Path) -> None:
    workspace_root = tmp_path / "benchmark"
    source = workspace_root / "workspaces" / "q1"
    source.mkdir(parents=True)
    (source / "input.txt").write_text("input")
    output_dir = tmp_path / "run" / "workspaces"
    model = ModelConfig(model_name="model", model_provider="provider", interface="langchain", id="m1")
    context = VerificationContext(
        question_id="q1",
        template_id="template",
        question_text="question",
        template_code="class Answer: pass",
        answering_model=model,
        parsing_model=model,
        agentic_parsing=True,
        workspace_root=workspace_root,
        question_workspace_path="workspaces/q1",
        workspace_copy=True,
        workspace_cleanup=True,
        workspace_output_mode="full",
        workspace_output_dir=output_dir,
    )
    context.set_result_field(ArtifactKeys.TIMESTAMP, "2026-05-20 12:00:00")

    GenerateAnswerStage()._resolve_workspace(context)

    assert context.workspace_path is not None
    assert context.workspace_path.parent == output_dir
    assert context.workspace_path.name.startswith("q1__")
    assert (context.workspace_path / "input.txt").read_text() == "input"
    assert context.workspace_is_copy is False
    assert not list(workspace_root.glob("*_run_*"))
