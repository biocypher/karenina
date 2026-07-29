"""Tests for k-shot example selection reproducibility across processes.

``FewShotConfig.resolve_examples_for_question`` must select the same examples for
the same inputs regardless of which Python process runs it. ``hash()`` is
randomized per-process via ``PYTHONHASHSEED``, so the implementation seeds its
RNG from a stable digest of the inputs. The realistic regression signal is:

* identical inputs → identical selection (within one process and across two
  processes with different ``PYTHONHASHSEED`` values), and
* different inputs → different selections.

We do NOT inspect the implementation's source text for the hashing function;
instead we run the resolution in two child processes with different
``PYTHONHASHSEED`` values and assert their outputs agree.
"""

import subprocess
import sys

import pytest

from karenina.schemas.config.models import FewShotConfig

_RESOLVE_SCRIPT = """
import json, sys
from karenina.schemas.config.models import FewShotConfig

question_id = sys.argv[1]
config = FewShotConfig(source="question_pool", pool_mode="k-shot", pool_k=3)
examples = [{"question": "q{:02d}".format(i), "answer": "a{:02d}".format(i)} for i in range(10)]
result = config.resolve_examples_for_question(question_id, examples)
print(json.dumps(result))
"""


def _resolve_in_child(question_id: str, hashseed: str) -> list[dict[str, str]]:
    """Run the k-shot resolution in a fresh Python process with a fixed PYTHONHASHSEED."""
    import json

    proc = subprocess.run(
        [sys.executable, "-c", _RESOLVE_SCRIPT, question_id],
        env={"PYTHONHASHSEED": hashseed, "PATH": ""},
        capture_output=True,
        text=True,
        check=True,
    )
    return json.loads(proc.stdout.strip())


@pytest.mark.unit
class TestKShotReproducibility:
    """k-shot seeding must be stable across processes (not depend on PYTHONHASHSEED)."""

    def test_resolve_deterministic_across_calls(self) -> None:
        """Same question_id yields same k-shot selection on repeated calls."""
        config = FewShotConfig(source="question_pool", pool_mode="k-shot", pool_k=3)
        examples = [{"question": f"q{i}", "answer": f"a{i}"} for i in range(10)]

        first = config.resolve_examples_for_question("test-q-123", examples)
        second = config.resolve_examples_for_question("test-q-123", examples)
        assert first == second

    def test_different_question_ids_yield_different_selections(self) -> None:
        """Different question_ids should (usually) produce different selections."""
        config = FewShotConfig(source="question_pool", pool_mode="k-shot", pool_k=3)
        examples = [{"question": f"q{i}", "answer": f"a{i}"} for i in range(20)]

        sel_a = config.resolve_examples_for_question("question-a", examples)
        sel_b = config.resolve_examples_for_question("question-b", examples)
        assert sel_a != sel_b

    def test_seed_is_stable_across_pythonhashseed_values(self) -> None:
        """Two subprocesses with different PYTHONHASHSEED must agree.

        If the implementation regressed to ``hash(question_id)``, the two
        processes would diverge because string hashing is randomized per
        process. This is the real cross-process reproducibility contract.
        """
        examples_a = _resolve_in_child("test-q-123", hashseed="0")
        examples_b = _resolve_in_child("test-q-123", hashseed="13")
        assert examples_a == examples_b
        # And the selection is a real 3-element subset of the pool, not empty.
        assert len(examples_a) == 3
