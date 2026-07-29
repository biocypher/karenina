"""Regression coverage for literal callable checkpoint persistence."""

from pathlib import Path

import pytest

from karenina.benchmark import Benchmark
from karenina.schemas.entities.rubric import CallableRubricTrait, Rubric


@pytest.mark.unit
def test_literal_callable_trait_attaches_and_survives_checkpoint_roundtrip(tmp_path: Path) -> None:
    """Literal callable classes survive eager cache rebuild and save/load."""
    classes = {
        "refuted": "Database refutes it",
        "supported": "Database supports it",
    }
    trait = CallableRubricTrait.from_callable(
        name="kb_verdict",
        func=lambda _text: "supported",
        kind="literal",
        classes=classes,
    )
    benchmark = Benchmark.create(name="literal-callable", description="")
    benchmark.add_question(question="q", raw_answer="a", question_id="q1")

    benchmark.set_question_rubric("q1", Rubric(callable_traits=[trait]))

    checkpoint_path = tmp_path / "literal_callable.jsonld"
    benchmark.save(checkpoint_path)
    restored = Benchmark.load(checkpoint_path)
    rubric = restored.get_question_rubric("q1")

    assert rubric is not None
    restored_trait = rubric.callable_traits[0]
    assert restored_trait.classes == classes
    assert restored_trait.evaluate("anything") == 1
