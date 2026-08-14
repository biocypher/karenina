"""Unit tests for scenario construction, analyses, and rubrics."""

import pandas as pd
import pytest

from karenina.schemas.config import ModelConfig
from karenina.schemas.entities import Question
from karenina.schemas.results import ScenarioResultRecord, VerificationResultSet
from paper.otp_sycophancy_scenarios.analysis import (
    build_behavior_composition,
    build_correction_recovery,
    build_guardrail_confusion,
)
from paper.otp_sycophancy_scenarios.config import (
    ANSWERING_SYSTEM_PROMPT,
    ANSWERING_SYSTEM_PROMPT_MCP,
    answerer_model,
    default_parser_model,
    guardrail_model,
)
from paper.otp_sycophancy_scenarios.rubrics import (
    ABSTENTION_TRAIT,
    CAVE_REGEX_TRAIT,
    GROUNDED_CAVE_TRAIT,
    abstention_rubric,
    cave_regex_rubric,
    grounded_cave_rubric,
)
from paper.otp_sycophancy_scenarios.run import _require_completed_scenarios
from paper.otp_sycophancy_scenarios.scenarios import build_scenario


@pytest.mark.unit
class TestScenarioGraph:
    """Validate the four-node graph."""

    def test_graph_nodes_routes_and_outcomes(self) -> None:
        scenario = build_scenario(
            Question(question="Which target?", raw_answer="BCL2"),
            row={"id": "001", "easy_adversarial": "TP53", "hard_adversarial": "BCL3"},
            difficulty="hard",
            framing="authority",
            parser_model=_manual_model("parser"),
            guardrail_model=_manual_model("guardrail"),
        ).validate()
        assert set(scenario.nodes) == {"ask", "adversarial", "correction", "guardrail_check"}
        assert scenario.entry_node == "ask"
        assert {criterion.name for criterion in scenario.outcome_criteria} == {
            "initial_correct",
            "resists_sycophancy",
            "self_corrects",
            "clean_resist",
            "sycophancy_caught",
        }
        targets = {(edge.source, edge.target) for edge in scenario.edges}
        assert ("ask", "adversarial") in targets
        assert ("ask", "correction") in targets
        assert ("adversarial", "guardrail_check") in targets

    def test_unknown_strata_raise(self) -> None:
        with pytest.raises(ValueError, match="difficulty"):
            build_scenario(
                Question(question="Which target?", raw_answer="BCL2"),
                row={"id": "001", "easy_adversarial": "TP53", "hard_adversarial": "BCL3"},
                difficulty="medium",
                framing="casual",
                parser_model=_manual_model("parser"),
                guardrail_model=_manual_model("guardrail"),
            )


@pytest.mark.unit
class TestAsRunModels:
    """Validate the final experiment's answerer and guardrail roles."""

    def test_answerer_prompts_match_tool_regime(self) -> None:
        assert answerer_model("claude-haiku-4-5").system_prompt == ANSWERING_SYSTEM_PROMPT
        mcp = answerer_model("claude-haiku-4-5", mcp_url="http://localhost:8765/mcp")
        assert mcp.system_prompt == ANSWERING_SYSTEM_PROMPT_MCP
        assert mcp.mcp_http_timeout == 240.0
        assert mcp.mcp_sse_read_timeout == 600.0
        assert mcp.agent_timeout == 900

    def test_guardrail_uses_same_model_without_answerer_prompt(self) -> None:
        haiku = guardrail_model("claude-haiku-4-5")
        assert haiku.model_name == "claude-haiku-4-5"
        assert haiku.system_prompt is None
        assert haiku.mcp_urls_dict is None

    def test_base_parser_uses_reported_qwen_config(self) -> None:
        parser = default_parser_model()
        assert parser.model_name == "qwen3.5-122b-a10b"
        assert parser.temperature == 0.0
        assert parser.system_prompt is None


@pytest.mark.unit
class TestDirectAnalysis:
    """Check behavior, guardrail, and correction arithmetic."""

    def test_behavior_composition(self) -> None:
        rows = build_behavior_composition(_turns())
        assert rows["rows"].sum() == 2
        assert set(rows["behavior"]) == {"pushback", "cave"}

    def test_guardrail_score_bands(self) -> None:
        rows = build_guardrail_confusion(_turns())
        mapping = dict(zip(rows["parsed_behavior"], rows["guardrail_band"], strict=True))
        assert mapping == {"pushback": "pushback", "cave": "cave"}

    def test_correction_recovery(self) -> None:
        scenarios = pd.DataFrame(
            [
                {"answerer": "a", "regime": "mcp", "outcome_initial_correct": False, "outcome_self_corrects": True},
                {"answerer": "a", "regime": "mcp", "outcome_initial_correct": False, "outcome_self_corrects": False},
                {"answerer": "a", "regime": "mcp", "outcome_initial_correct": True, "outcome_self_corrects": False},
            ]
        )
        rows = build_correction_recovery(scenarios)
        assert rows.iloc[0]["rows"] == 2
        assert rows.iloc[0]["recovered"] == 1


@pytest.mark.unit
class TestLiveRunBoundary:
    """Require the live wrapper to surface terminal scenario failures."""

    def test_incomplete_scenario_raises(self) -> None:
        result_set = VerificationResultSet(
            results=[],
            scenario_results=[
                ScenarioResultRecord(
                    scenario_id="scenario-1",
                    status="error",
                    path=["ask", "correction"],
                    turn_count=2,
                )
            ]
        )
        with pytest.raises(RuntimeError, match=r"scenario-1 \(error\)"):
            _require_completed_scenarios(result_set)


@pytest.mark.unit
class TestSidecarRubrics:
    """Check rubric definitions and the ground-truth boundary."""

    def test_trait_names_and_types(self) -> None:
        assert abstention_rubric().llm_traits[0].name == ABSTENTION_TRAIT
        assert cave_regex_rubric().regex_traits[0].name == CAVE_REGEX_TRAIT
        grounded = grounded_cave_rubric().llm_traits[0]
        assert grounded.name == GROUNDED_CAVE_TRAIT
        assert grounded.include_ground_truth is True


def _manual_model(name: str) -> ModelConfig:
    return ModelConfig(id=name, model_name="manual", interface="manual", manual_traces={})


def _turns() -> pd.DataFrame:
    common = {
        "answerer": "a",
        "regime": "mcp",
        "difficulty": "easy",
        "framing": "casual",
        "source_file": "source.json",
        "status": "completed",
    }
    return pd.DataFrame(
        [
            {**common, "scenario_id": "s1", "node_id": "adversarial", "parsed_fields": {"behavior": "pushback"}},
            {**common, "scenario_id": "s1", "node_id": "guardrail_check", "parsed_fields": {"sycophancy_score": 2}},
            {**common, "scenario_id": "s2", "node_id": "adversarial", "parsed_fields": {"behavior": "cave"}},
            {**common, "scenario_id": "s2", "node_id": "guardrail_check", "parsed_fields": {"sycophancy_score": 5}},
        ]
    )
