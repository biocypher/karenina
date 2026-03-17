"""Tests for Benchmark scenario integration."""

import pytest

from karenina.benchmark.benchmark import Benchmark
from karenina.scenario.builder import Scenario
from karenina.schemas.scenario.types import END


def _make_question(text="What?"):
    from karenina.schemas.entities import Question

    return Question(question=text, raw_answer="Y", answer_template="class Answer: pass")


def _build_scenario(name="test"):
    s = Scenario(name)
    s.add_node("ask", question=_make_question())
    s.add_edge("ask", END)
    s.set_entry("ask")
    return s


@pytest.mark.unit
class TestBenchmarkScenarioIntegration:
    def test_add_scenario_from_builder(self):
        bm = Benchmark("test_bm")
        s = _build_scenario()
        bm.add_scenario(s)
        assert bm.is_scenario_benchmark is True
        assert len(bm.get_scenarios()) == 1

    def test_add_scenario_from_definition(self):
        bm = Benchmark("test_bm")
        defn = _build_scenario().validate()
        bm.add_scenario(defn)
        assert bm.is_scenario_benchmark is True

    def test_get_scenario_by_name(self):
        bm = Benchmark("test_bm")
        bm.add_scenario(_build_scenario("alpha"))
        bm.add_scenario(_build_scenario("beta"))
        s = bm.get_scenario("alpha")
        assert s.name == "alpha"

    def test_get_scenario_not_found_raises(self):
        bm = Benchmark("test_bm")
        with pytest.raises(KeyError):
            bm.get_scenario("nonexistent")

    def test_remove_scenario(self):
        bm = Benchmark("test_bm")
        bm.add_scenario(_build_scenario("alpha"))
        bm.remove_scenario("alpha")
        assert len(bm.get_scenarios()) == 0

    def test_homogeneous_enforcement_scenario_then_question(self):
        bm = Benchmark("test_bm")
        bm.add_scenario(_build_scenario())
        with pytest.raises(ValueError, match="[Ss]cenario|[Hh]omogeneous"):
            bm.add_question("What?", raw_answer="Y", answer_template="class Answer: pass")

    def test_homogeneous_enforcement_question_then_scenario(self):
        bm = Benchmark("test_bm")
        bm.add_question("What?", raw_answer="Y", answer_template="class Answer: pass")
        with pytest.raises(ValueError, match="[Qq]uestion|[Hh]omogeneous"):
            bm.add_scenario(_build_scenario())

    def test_is_scenario_benchmark_false_by_default(self):
        bm = Benchmark("test_bm")
        assert bm.is_scenario_benchmark is False

    def test_duplicate_scenario_name_raises(self):
        bm = Benchmark("test_bm")
        bm.add_scenario(_build_scenario("alpha"))
        with pytest.raises(ValueError, match="already exists"):
            bm.add_scenario(_build_scenario("alpha"))
