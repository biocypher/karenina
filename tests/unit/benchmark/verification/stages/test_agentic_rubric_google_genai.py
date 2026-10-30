"""AgenticRubricEvaluation with a google_genai judge, end to end on a fake genai client.

The judge is a ``tool_loop`` interface with MCP support, so it can evaluate
a ``trace_only`` trait: the stage takes the agent path (C1) with the judge's
MCP servers (C2), and the score, trace, messages and usage reach the result.
Adapters are built through the real registry. Only ``build_client`` (the
genai client) and the MCP connection are faked.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock

import pytest

pytest.importorskip("google.genai", reason="google-genai not installed")

from karenina.adapters.google_genai import agent as agent_module  # noqa: E402
from karenina.adapters.google_genai import client as client_module  # noqa: E402
from karenina.benchmark.verification.stages.core.base import ArtifactKeys, VerificationContext  # noqa: E402
from karenina.benchmark.verification.stages.pipeline.agentic_rubric_evaluation import (  # noqa: E402
    AGENTIC_RUBRIC_USAGE_STAGE,
    AgenticRubricEvaluationStage,
)
from karenina.benchmark.verification.stages.pipeline.finalize_result import FinalizeResultStage  # noqa: E402
from karenina.schemas.config.models import ModelConfig  # noqa: E402
from karenina.schemas.entities.rubric import AgenticRubricTrait, Rubric  # noqa: E402
from tests.unit.adapters.google_genai._fakes import (  # noqa: E402
    FakeClient,
    FakeModels,
    call_part,
    make_config,
    response,
    search_grounding,
    text_part,
    usage,
)

MCP_URL = "http://localhost:8765/mcp"


@pytest.fixture(autouse=True)
def _clear_google_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Keep the developer's Google environment out of the test."""
    for name in ("GOOGLE_CLOUD_PROJECT", "GOOGLE_CLOUD_LOCATION", "GEMINI_API_KEY", "GOOGLE_API_KEY"):
        monkeypatch.delenv(name, raising=False)


@pytest.fixture
def fake_models(monkeypatch: pytest.MonkeyPatch) -> FakeModels:
    """Every google_genai adapter built in the test talks to these scripted responses."""
    models = FakeModels(
        responses=[
            # Investigation, turn 1: the judge looks the citation up through MCP.
            response(
                [call_part("europepmc_get", {"id": "123"}, call_id="c1")],
                usage_metadata=usage(100, 20, total=120),
            ),
            # Investigation, turn 2: the verdict, after two Google searches.
            response(
                [text_part("PMID 123 reports this finding. Verdict: supported.")],
                usage_metadata=usage(150, 30, thoughts=12, total=192),
                grounding=search_grounding(["PMID 123", "TP53 apoptosis"]),
            ),
            # Extraction: structured boolean score.
            response([text_part('{"result": true}')], usage_metadata=usage(40, 5, total=45)),
        ]
    )
    monkeypatch.setattr(client_module, "build_client", lambda _settings: FakeClient(models))
    return models


@pytest.fixture
def mcp(monkeypatch: pytest.MonkeyPatch) -> SimpleNamespace:
    """A mocked MCP session exposing one Europe PMC lookup tool."""
    session = SimpleNamespace(
        call_tool=AsyncMock(
            return_value=SimpleNamespace(
                content=[SimpleNamespace(text="PMID 123: TP53 drives apoptosis (Smith 2020).")],
                isError=False,
            )
        )
    )
    tool = SimpleNamespace(
        name="europepmc_get",
        description="Fetch one Europe PMC record",
        inputSchema={"type": "object", "properties": {"id": {"type": "string"}}},
    )
    connected: dict[str, Any] = {}

    async def fake_connect(_stack: Any, servers: dict[str, Any]) -> dict[str, Any]:
        connected.update(servers)
        return {"citations": session}

    async def fake_tools(_sessions: dict[str, Any]) -> list[tuple[str, Any, Any]]:
        return [("citations", session, tool)]

    monkeypatch.setattr(agent_module, "connect_all_mcp_servers", fake_connect)
    monkeypatch.setattr(agent_module, "get_all_mcp_tools", fake_tools)
    return SimpleNamespace(session=session, connected=connected)


def _context(*, judge_as: str) -> VerificationContext:
    """A context whose only agentic trait is judged by a google_genai model with MCP."""
    judge = make_config(mcp_urls_dict={"citations": MCP_URL})
    other = ModelConfig(id="other", model_name="other-model", model_provider="anthropic", interface="claude_agent_sdk")
    trait = AgenticRubricTrait(
        name="citations_supported",
        description="Check that every cited paper supports the claim it is cited for.",
        kind="boolean",
        higher_is_better=True,
        context_mode="trace_only",
        model_override=judge if judge_as == "model_override" else None,
    )
    ctx = VerificationContext(
        question_id="q1",
        template_id="t1",
        question_text="What does TP53 do?",
        template_code="class Answer: pass",
        answering_model=other,
        parsing_model=judge if judge_as == "parsing_model" else other,
        rubric=Rubric(agentic_traits=[trait]),
    )
    ctx.set_artifact(ArtifactKeys.RAW_LLM_RESPONSE, "TP53 drives apoptosis [PMID 123].")
    ctx.set_result_field("timestamp", "2026-10-31 12:00:00")
    ctx.set_result_field("execution_time", 1.0)
    return ctx


@pytest.mark.unit
@pytest.mark.parametrize("judge_as", ["model_override", "parsing_model"])
def test_google_genai_judge_runs_end_to_end(judge_as: str, fake_models: FakeModels, mcp: SimpleNamespace) -> None:
    ctx = _context(judge_as=judge_as)
    stage = AgenticRubricEvaluationStage()
    assert stage.should_run(ctx)

    stage.execute(ctx)
    FinalizeResultStage().execute(ctx)
    result = ctx.get_artifact("final_result")

    # The judge reached its MCP server through the agent path and called the tool.
    assert mcp.connected["citations"]["url"] == MCP_URL
    assert mcp.session.call_tool.await_args.args == ("europepmc_get", {"id": "123"})
    assert len(fake_models.calls) == 3
    assert fake_models.responses == []

    assert result.metadata.failure is None
    assert result.metadata.caveats == []
    assert result.rubric.agentic_trait_scores == {"citations_supported": True}
    assert "Verdict: supported." in result.rubric.agentic_trait_investigation_traces["citations_supported"]

    messages = result.rubric.agentic_trait_investigation_messages["citations_supported"]
    assert [m["block_index"] for m in messages] == list(range(len(messages)))
    tool_names = [call["name"] for m in messages for call in m.get("tool_calls", [])]
    assert "europepmc_get" in tool_names
    assert "google_search" in tool_names
    assert any(m["role"] == "tool" and "Smith 2020" in m["content"] for m in messages)
    assert messages[-1]["role"] == "assistant"
    assert "Verdict: supported." in messages[-1]["content"]

    judge_usage = result.template.usage_metadata[AGENTIC_RUBRIC_USAGE_STAGE]
    # Two investigation calls plus one extraction call.
    assert judge_usage["input_tokens"] == 100 + 150 + 40
    assert judge_usage["output_tokens"] == 20 + (30 + 12) + 5
    assert judge_usage["reasoning_tokens"] == 12
    assert judge_usage["web_search_queries"] == 2
    assert result.template.usage_metadata["total"]["web_search_queries"] == 2


@pytest.mark.unit
def test_rate_limited_extraction_is_an_infrastructure_failure(fake_models: FakeModels, mcp: SimpleNamespace) -> None:
    """A 429 on the extraction call, after the parser's retries, leaves a record instead of a silent None."""
    from google.genai import errors as genai_errors

    from karenina.schemas.results.caveat import Caveat
    from karenina.schemas.results.failure import FailureGroup

    quota = genai_errors.ClientError(429, {"error": {"code": 429, "message": "quota", "status": "RESOURCE_EXHAUSTED"}})
    # Every extraction attempt is rate limited. The adapter and the pipeline both retry.
    fake_models.responses[2:] = [quota] * 50

    ctx = _context(judge_as="model_override")
    AgenticRubricEvaluationStage().execute(ctx)
    FinalizeResultStage().execute(ctx)
    result = ctx.get_artifact("final_result")

    extraction_calls = len(fake_models.calls) - 2
    assert extraction_calls >= 3
    assert len(fake_models.responses) == 50 - extraction_calls
    assert result.rubric.agentic_trait_scores == {"citations_supported": None}
    assert "Verdict: supported." in result.rubric.agentic_trait_investigation_traces["citations_supported"]
    assert result.rubric.agentic_trait_investigation_messages["citations_supported"]
    error = result.rubric.agentic_trait_evaluation_errors["citations_supported"]
    assert error["phase"] == "extraction"
    assert error["error_type"] == "GeminiRateLimitError"
    assert error["category"] == "rate_limit"
    assert result.metadata.failure is not None
    assert result.metadata.failure.stage == "AgenticRubricEvaluation"
    assert result.metadata.failure.group in (FailureGroup.RETRY_EXHAUSTED, FailureGroup.SYSTEM)
    assert Caveat.AGENTIC_EXTRACTION_FAILED in result.metadata.caveats
