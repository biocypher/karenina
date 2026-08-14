"""Unit tests for standalone adversarial generation and summaries."""

import json
from pathlib import Path

import pytest

from paper.otp_adversarial_generation.analysis import summarize_pairs
from paper.otp_adversarial_generation.claude_code import (
    build_claude_command,
    extract_sample_text,
    trace_metadata,
    validate_open_targets_mcp,
)
from paper.otp_adversarial_generation.config import GENERATOR_MODEL
from paper.otp_adversarial_generation.generation import (
    AdversarialPair,
    BenchmarkItem,
    binary_pair,
    flip_binary_answer,
    load_approved_archive,
    parse_sample_text,
)
from paper.otp_adversarial_generation.run import generation_prompt


@pytest.mark.unit
class TestBinaryGeneration:
    """Check deterministic binary normalization."""

    @pytest.mark.parametrize(("answer", "flipped"), [("YES", "No"), (" no ", "Yes"), ("True", "False")])
    def test_flip_binary_answer(self, answer: str, flipped: str) -> None:
        assert flip_binary_answer(answer) == flipped

    def test_non_binary_raises(self) -> None:
        with pytest.raises(ValueError, match="Not a binary"):
            flip_binary_answer("BCL2")

    def test_binary_pair_is_a_draft_without_model_metadata(self) -> None:
        pair = binary_pair(_item("Yes"))
        assert pair.hard_adversarial == "No"
        assert pair.generation_route == "binary_flip"
        assert pair.review_status == "draft_requires_curator"
        assert pair.model_name is None


@pytest.mark.unit
class TestClaudeCodeBoundary:
    """Check the standalone command, prompt, trace, and sample contract."""

    def test_command_includes_required_cli_flags(self) -> None:
        command = build_claude_command("research this", model=GENERATOR_MODEL)
        assert command == [
            "claude",
            "--model",
            "claude-opus-4-6[1m]",
            "--dangerously-skip-permissions",
            "--verbose",
            "--output-format",
            "stream-json",
            "-p",
            "research this",
        ]

    def test_prompt_preserves_mcp_workflow_and_exact_output_path(self, tmp_path: Path) -> None:
        output_path = tmp_path / "adversarial.txt"
        prompt = generation_prompt(_item("BCL2"), output_path)
        assert "ToolSearch(query=\"select:mcp__open-targets__search_entities" in prompt
        assert "mcp__open-targets__get_open_targets_graphql_schema" in prompt
        assert "mcp__open-targets__query_open_targets_graphql" in prompt
        assert "wait 15 seconds" in prompt
        assert "retry up to 3 times" in prompt
        assert str(output_path) in prompt
        assert "Do NOT output any text to the conversation" in prompt

    def test_trace_reports_model_session_and_connected_mcp(self) -> None:
        trace = _trace()
        model, session_id, servers = trace_metadata(trace)
        assert model == "claude-opus-4-6[1m]"
        assert session_id == "session-123"
        assert servers == [{"name": "open-targets", "status": "connected"}]
        validate_open_targets_mcp(trace)

    def test_missing_connected_mcp_raises(self) -> None:
        trace = _trace(status="failed")
        with pytest.raises(RuntimeError, match="connected Open Targets"):
            validate_open_targets_mcp(trace)

    def test_lazy_mcp_connection_is_proven_by_tool_use(self) -> None:
        trace = "\n".join(
            [
                _trace(status="pending"),
                json.dumps(
                    {
                        "type": "assistant",
                        "message": {
                            "content": [
                                {
                                    "type": "tool_use",
                                    "name": "mcp__plugin_open-targets_OpenTargets__search_entities",
                                    "input": {"query_string": "BCL2"},
                                }
                            ]
                        },
                    }
                ),
            ]
        )
        validate_open_targets_mcp(trace)

    def test_extracts_write_payload_and_parses_draft(self) -> None:
        sample = _sample_text()
        trace = "\n".join(
            [
                _trace(),
                json.dumps(
                    {
                        "type": "assistant",
                        "message": {
                            "content": [
                                {
                                    "type": "tool_use",
                                    "name": "Write",
                                    "input": {"content": sample},
                                }
                            ]
                        },
                    }
                ),
            ]
        )
        extracted = extract_sample_text(trace)
        assert extracted == sample
        pair = parse_sample_text(
            extracted or "",
            _item("BCL2"),
            model_name="claude-opus-4-6[1m]",
            trace_id="session-123",
        )
        assert pair.hard_adversarial == "BCL3"
        assert pair.easy_adversarial == "TP53"
        assert pair.generation_route == "standalone_claude_code_mcp"
        assert pair.review_status == "draft_requires_curator"


@pytest.mark.unit
class TestValidationAndAnalysis:
    """Check pair validation, archived labels, deduplication, and summaries."""

    def test_pair_rejects_ground_truth_as_alternative(self) -> None:
        with pytest.raises(ValueError, match="repeats the ground truth"):
            _pair(hard="BCL2")

    def test_summary_crosses_source_strata_and_route(self) -> None:
        rows = summarize_pairs([_pair(), _pair(item_id="002", route="binary_flip")])
        counts = [row["count"] for row in rows]
        assert all(isinstance(count, int) for count in counts)
        assert sum(count for count in counts if isinstance(count, int)) == 2
        assert {str(row["generation_route"]) for row in rows} == {
            "standalone_claude_code_mcp",
            "binary_flip",
        }

    def test_approved_archive_detects_duplicate_ids(self, tmp_path: Path) -> None:
        archive = tmp_path / "pairs.csv"
        archive.write_text(
            "id,difficulty,area,type,question,ground_truth,hard_adversarial,easy_adversarial\n"
            "001,1,A,T,Q,BCL2,BCL3,TP53\n"
            "001,1,A,T,Q,BCL2,BCL3,TP53\n",
            encoding="utf-8",
        )
        with pytest.raises(ValueError, match="duplicate"):
            load_approved_archive(archive)


def _item(answer: str, item_id: str = "001") -> BenchmarkItem:
    return BenchmarkItem(
        item_id=item_id,
        area="Target",
        question_type="Annotation",
        question="Which target?",
        ground_truth=answer,
        source_difficulty=1.0,
    )


def _pair(
    *,
    hard: str = "BCL3",
    item_id: str = "001",
    route: str = "standalone_claude_code_mcp",
) -> AdversarialPair:
    return AdversarialPair(
        **_item("BCL2", item_id).model_dump(),
        hard_adversarial=hard,
        easy_adversarial="TP53",
        hard_rationale="Same narrow domain.",
        easy_rationale="Clearly inconsistent.",
        evidence_summary="Open Targets query evidence.",
        generation_route=route,
        review_status="draft_requires_curator",
    )


def _trace(*, status: str = "connected") -> str:
    return json.dumps(
        {
            "type": "system",
            "subtype": "init",
            "model": "claude-opus-4-6[1m]",
            "session_id": "session-123",
            "permissionMode": "bypassPermissions",
            "mcp_servers": [{"name": "open-targets", "status": status}],
        }
    )


def _sample_text() -> str:
    return """--- ADVERSARIAL SAMPLE ---
ID: 001
Area: Target
Type: Annotation
Question: Which target?
Ground Truth: BCL2

--- HARD ADVERSARIAL ---
Answer: BCL3
Reasoning: A nearby target supported by platform research.

--- EASY ADVERSARIAL ---
Answer: TP53
Reasoning: A real but clearly inconsistent biomedical target.

--- MCP DATA USED ---
Searched BCL2 and queried related target records.
"""
