"""Tests for issue 103: full Question serialization in scenario checkpoint."""

import pytest

from karenina.scenario.builder import Scenario
from karenina.scenario.checkpoint import scenario_to_schema_org, schema_org_to_scenario
from karenina.schemas.entities import Question
from karenina.schemas.scenario.types import END


def _make_rich_question() -> Question:
    """Create a Question with many optional fields set."""
    return Question(
        question="What drug targets BRCA1?",
        raw_answer="Olaparib",
        answer_template="class Answer: pass",
        keywords=["bio", "pharma", "oncology"],
        few_shot_examples=[{"q": "What targets EGFR?", "a": "Erlotinib"}],
        author={"name": "Test Author", "url": "https://example.com"},
        sources=[{"name": "PubMed", "url": "https://pubmed.ncbi.nlm.nih.gov"}],
        custom_metadata={"difficulty": "hard", "domain": "oncology"},
        workspace_path="task_01",
        answer_notes="Olaparib is a PARP inhibitor",
    )


def _build_scenario_with_rich_question():
    """Build a scenario with a Question that has all optional fields."""
    s = Scenario("rich_test")
    s.add_node("ask", question=_make_rich_question())
    s.add_edge("ask", END)
    s.set_entry("ask")
    return s.validate()


@pytest.mark.unit
class TestCheckpointQuestionRoundtrip:
    """Issue 103: all Question fields should survive checkpoint round-trip."""

    def test_rich_question_survives_roundtrip(self):
        """Keywords, few_shot_examples, author, sources, custom_metadata, workspace_path survive."""
        defn = _build_scenario_with_rich_question()
        original_q = defn.nodes["ask"].question

        # Round-trip through checkpoint
        schema_org = scenario_to_schema_org(defn)
        restored = schema_org_to_scenario(schema_org)
        restored_q = restored.nodes["ask"].question

        assert restored_q.question == original_q.question
        assert restored_q.raw_answer == original_q.raw_answer
        assert restored_q.answer_template == original_q.answer_template
        assert restored_q.keywords == ["bio", "pharma", "oncology"]
        assert restored_q.few_shot_examples == [{"q": "What targets EGFR?", "a": "Erlotinib"}]
        assert restored_q.author == {"name": "Test Author", "url": "https://example.com"}
        assert restored_q.sources == [{"name": "PubMed", "url": "https://pubmed.ncbi.nlm.nih.gov"}]
        assert restored_q.custom_metadata == {"difficulty": "hard", "domain": "oncology"}
        assert restored_q.workspace_path == "task_01"
        assert restored_q.answer_notes == "Olaparib is a PARP inhibitor"

    def test_backward_compat_without_question_data(self):
        """Old checkpoints without questionData still deserialize (3-field fallback)."""
        defn = _build_scenario_with_rich_question()
        schema_org = scenario_to_schema_org(defn)

        # Simulate old checkpoint: remove questionData from all nodes
        for node in schema_org.nodes.values():
            if hasattr(node, "questionData"):
                node.questionData = None

        restored = schema_org_to_scenario(schema_org)
        restored_q = restored.nodes["ask"].question

        # Basic fields should still work
        assert restored_q.question == "What drug targets BRCA1?"
        assert restored_q.raw_answer == "Olaparib"
        assert restored_q.answer_template == "class Answer: pass"
        # Rich fields lost in old format (this is expected)
        assert restored_q.keywords == []  # default
