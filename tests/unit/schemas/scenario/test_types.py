"""Tests for scenario schema types."""

from __future__ import annotations

import pytest

from karenina.schemas.primitives import BooleanMatch, ExactMatch, NumericRange
from karenina.schemas.scenario.types import (
    END,
    ModelOverride,
    ScenarioEdge,
    ScenarioNode,
    ScenarioOutcomeCriterion,
    StateCheck,
    ToolFilter,
    ToolFilterEntry,
)


@pytest.mark.unit
class TestEND:
    def test_end_sentinel_value(self):
        assert END == "__end__"

    def test_end_is_string(self):
        assert isinstance(END, str)


@pytest.mark.unit
class TestStateCheck:
    def test_basic_boolean_check(self):
        c = StateCheck(field="verify_result", expected=True, verify_with=BooleanMatch())
        assert c.field == "verify_result"
        assert c.expected is True
        assert isinstance(c.verify_with, BooleanMatch)
        assert c.type == "state_check"

    def test_dot_path_field(self):
        c = StateCheck(field="node_visits.retry", verify_with=NumericRange(min=3))
        assert c.field == "node_visits.retry"

    def test_serialization_roundtrip(self):
        c = StateCheck(field="parsed.drug", expected="venetoclax", verify_with=ExactMatch())
        data = c.model_dump()
        restored = StateCheck.model_validate(data)
        assert restored.field == c.field
        assert restored.expected == c.expected


@pytest.mark.unit
class TestModelOverride:
    def test_empty_override(self):
        o = ModelOverride()
        assert o.answering_model is None
        assert o.parsing_model is None

    def test_answering_only(self):
        from karenina.schemas.config import ModelConfig

        mc = ModelConfig(id="gpt-4o", model_name="gpt-4o", model_provider="openai")
        o = ModelOverride(answering_model=mc)
        assert o.answering_model.model_name == "gpt-4o"
        assert o.parsing_model is None


@pytest.mark.unit
class TestToolFilter:
    def test_single_server_block(self):
        tf = ToolFilter(remove=[ToolFilterEntry(server="brave-search")])
        assert len(tf.remove) == 1
        assert tf.remove[0].server == "brave-search"
        assert tf.remove[0].tool is None

    def test_specific_tool_block(self):
        tf = ToolFilter(remove=[ToolFilterEntry(server="chrome", tool="navigate")])
        assert tf.remove[0].tool == "navigate"


@pytest.mark.unit
class TestScenarioNode:
    def test_minimal_node(self):
        from karenina.schemas.entities import Question

        q = Question(question="What is X?", raw_answer="Y", answer_template="class Answer: pass")
        node = ScenarioNode(node_id="ask", question=q)
        assert node.node_id == "ask"
        assert node.model_override is None
        assert node.tool_filter is None
        assert node.state_update is None
        assert node.state_update_source is None

    def test_callable_excluded_from_serialization(self):
        from karenina.schemas.entities import Question

        q = Question(question="What?", raw_answer="Y", answer_template="class Answer: pass")
        fn = lambda acc, p: {**acc, "x": p.get("x")}  # noqa: E731
        node = ScenarioNode(node_id="ask", question=q, state_update=fn)
        data = node.model_dump()
        assert "state_update" not in data


@pytest.mark.unit
class TestScenarioEdge:
    def test_unconditional_edge(self):
        e = ScenarioEdge(source="a", target="b")
        assert e.condition is None
        assert e.condition_callable is None

    def test_state_check_condition_edge(self):
        c = StateCheck(field="verify_result", expected=True, verify_with=BooleanMatch())
        e = ScenarioEdge(source="a", target="b", condition=c)
        assert e.condition == c

    def test_edge_to_end(self):
        e = ScenarioEdge(source="a", target=END)
        assert e.target == "__end__"

    def test_callable_excluded_from_serialization(self):
        fn = lambda s: s.verify_result  # noqa: E731
        e = ScenarioEdge(source="a", target="b", condition_callable=fn)
        data = e.model_dump()
        assert "condition_callable" not in data


@pytest.mark.unit
class TestScenarioOutcomeCriterion:
    def test_basic_criterion(self):
        fn = lambda r: len(r.history) <= 3  # noqa: E731
        c = ScenarioOutcomeCriterion(
            name="efficiency",
            description="Reached answer in 3 turns or fewer?",
            evaluate=fn,
        )
        assert c.name == "efficiency"

    def test_check_field_accepts_none(self):
        c = ScenarioOutcomeCriterion(name="test", description="test")
        assert c.check is None

    def test_callable_excluded_from_serialization(self):
        fn = lambda r: True  # noqa: E731, ARG005
        c = ScenarioOutcomeCriterion(
            name="test",
            description="test",
            evaluate=fn,
            evaluate_source="lambda r: True",
        )
        data = c.model_dump()
        assert "evaluate" not in data
        assert data["evaluate_source"] == "lambda r: True"
