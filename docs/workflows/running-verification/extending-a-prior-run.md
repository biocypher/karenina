---
jupyter:
  jupytext:
    formats: docs/workflows/running-verification//md,docs/notebooks/running-verification//ipynb
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.3'
      jupytext_version: 1.18.1
  kernelspec:
    display_name: Python 3
    language: python
    name: python3
---

# Extending a Prior Run

This tutorial walks through the full extension lifecycle end-to-end: run a benchmark, save it with a [`ProgressiveFileSink`](../../reference/api/sinks.md), load it back, attach a [rubric](../../core_concepts/rubrics/), and call [`extend_rubric`](../../core_concepts/extending-runs.md) to score the prior traces against the new rubric. The same pattern carries over to [`extend_template`](../../core_concepts/extending-runs.md) when you want to add new judges, new answerers, or new replicates instead.

For the conceptual surface (validation rules, merge semantics, the [ReplayStore](../../advanced-pipeline/replay-store.md) the engines build internally), see [Extending Runs](../../core_concepts/extending-runs.md). For the sink internals that make the extension resumable, see [Progressive Save and Resume](./progressive-save.md) and the [Sinks reference](../../reference/api/sinks.md).

**What you'll learn:**

- Phase A: run a benchmark and save with `ProgressiveFileSink`
- Phase B: load the saved results
- Phase C: call `extend_template` with new parsing models, persisting through a fresh sink so the extension itself is resumable
- Phase D: call `extend_rubric` on the prior result with a new rubric
- Inspect the merged result

The tutorial uses a tiny two-question benchmark and the [Replay Store](../../advanced-pipeline/replay-store.md) to short-circuit every LLM call, so the whole notebook executes without API keys.

```python tags=["hide-cell"]
# Setup: silence two unrelated warnings that fire during in-notebook
# pipeline runs (the langchain adapter cleanup timeout and the openai
# placeholder API key complaint). Neither is reached because the replay
# short-circuit returns before any adapter is built.
import logging
import os

os.environ.setdefault("OPENAI_API_KEY", "placeholder-not-used-replay-only")

logging.getLogger("karenina.benchmark.verification.utils.resource_helpers").setLevel(logging.ERROR)
```

---

## Phase A: Run and Save

Build a tiny benchmark with two questions and a regex template. Capture the answers and parsed fields into a [`ReplayStore`](../../advanced-pipeline/replay-store.md) so the run executes without LLM calls. Pass a [`ProgressiveFileSink`](../../reference/api/sinks.md) so the result is persisted to disk as it runs.

```python
import tempfile
from pathlib import Path

from karenina.benchmark import Benchmark
from karenina.benchmark.verification.sinks import ProgressiveFileSink
from karenina.replay import ReplayEntry, ReplayKey, ReplayStore
from karenina.schemas.config import ModelConfig
from karenina.schemas.verification import VerificationConfig

TEMPLATE = """
from karenina.schemas.entities.answer import BaseAnswer
from karenina.schemas.entities.verified_field import VerifiedField
from karenina.schemas.primitives.comparisons import RegexMatch


class Answer(BaseAnswer):
    city: str = VerifiedField(
        description="Capital city",
        verify_with=RegexMatch(pattern="(?i)paris|rome"),
        ground_truth="Paris",
        default="",
    )

    def verify(self) -> bool:
        return self.city.strip().lower() in {"paris", "rome"}
"""

bench = Benchmark.create(name="extend-tutorial", version="1.0.0")
qid_paris = bench.add_question(
    question="What is the capital of France?",
    raw_answer="Paris",
    answer_template=TEMPLATE,
)
qid_rome = bench.add_question(
    question="What is the capital of Italy?",
    raw_answer="Rome",
    answer_template=TEMPLATE,
)

# Pre-populate a replay store so Phase A executes without any LLM call.
seed_store = ReplayStore(miss_policy="strict")
seed_store.register(
    ReplayKey(question_id=qid_paris),
    ReplayEntry(
        raw_trace="The capital of France is Paris.",
        parsed_answer_fields={"city": "Paris"},
    ),
)
seed_store.register(
    ReplayKey(question_id=qid_rome),
    ReplayEntry(
        raw_trace="The capital of Italy is Rome.",
        parsed_answer_fields={"city": "Rome"},
    ),
)

phase_a_config = VerificationConfig(
    answering_models=[
        ModelConfig(id="ans", model_name="gpt-fake", interface="langchain", model_provider="openai"),
    ],
    parsing_models=[
        ModelConfig(id="par", model_name="gpt-fake", interface="langchain", model_provider="openai"),
    ],
    replay_store=seed_store,
    replicate_count=1,
)
print("Phase A configured.")
```

Save the benchmark checkpoint and the verification output paths to a temp directory, then run with a `ProgressiveFileSink`:

```python
work_dir = Path(tempfile.mkdtemp())
benchmark_path = work_dir / "benchmark.jsonld"
output_path = work_dir / "phase_a.json"

bench.save(str(benchmark_path))

phase_a_sink = ProgressiveFileSink(
    output_path=output_path,
    config=phase_a_config,
    benchmark_path=str(benchmark_path),
)

prior = bench.run_verification(
    config=phase_a_config,
    run_name="extend-tutorial",
    sink=phase_a_sink,
)

print("Phase A rows:", len(prior.results))
print("Final export exists:", output_path.exists())
print("Sidecars cleaned up:", not (output_path.with_suffix(".json.state").exists()))
```

`output_path` now holds the canonical export, and the `.state` / `.results.jsonl` sidecars have been removed because the run completed cleanly. If the run had been interrupted, the sidecars would have remained so [`Benchmark.resume_verification`](./progressive-save.md) could pick up where it stopped.

---

## Phase B: Load the Saved Results

Reload the prior run from the on-disk export. In a real workflow this is the boundary between two processes (or two terminal sessions): one process produces the prior, another extends it.

```python
from karenina.schemas.results import VerificationResultSet

reloaded_prior = VerificationResultSet.model_validate_json(output_path.read_text())
print("Reloaded rows:", len(reloaded_prior.results))
print("Run name:", reloaded_prior.results[0].metadata.run_name)
```

The reloaded set is what the extension facades expect as `prior_results`. Captured traces remain on each row's metadata, so [`capture_from_result_set`](../../advanced-pipeline/replay-store.md#capturing-live-runs) can rebuild a replay store internally.

> **Caveat.** If you reconstruct `prior_results` by hand or post-process it through a transform that drops the captured trace, the internal capture will return an empty store and every triple will run live during the extension, silently re-spending answering tokens. See [Extending Runs: Failure Mode](../../core_concepts/extending-runs.md#9-failure-mode-prior-results-without-a-captured-replay-store).

---

## Phase C: Extend with a New Judge

Add a second parsing model. The captured prior traces serve the answering stage from replay; only the new judge runs live. We persist Phase C through a *fresh* `ProgressiveFileSink` so the extension itself is resumable: if the run crashes after the new judge has finished some questions, `resume_verification` skips the completed triples on retry.

```python
phase_c_config = VerificationConfig(
    answering_models=[
        ModelConfig(id="ans", model_name="gpt-fake", interface="langchain", model_provider="openai"),
    ],
    parsing_models=[
        ModelConfig(id="par", model_name="gpt-fake", interface="langchain", model_provider="openai"),
        ModelConfig(id="par-2", model_name="gpt-other", interface="langchain", model_provider="openai"),
    ],
    replicate_count=1,
)
phase_c_output = work_dir / "phase_c.json"

# In production this is the call you make. We do not execute it here because
# extend_template runs the parsing stage live for the new judge, and that
# requires a real LLM endpoint. The pattern is identical to extend_rubric
# below: pass a sink to make the extension resumable.
#
# phase_c_sink = ProgressiveFileSink(
#     output_path=phase_c_output,
#     config=phase_c_config,
#     benchmark_path=str(benchmark_path),
# )
# merged_template = bench.extend_template(
#     prior_results=reloaded_prior,
#     config=phase_c_config,
#     run_name="extend-tutorial",
#     sink=phase_c_sink,
# )
print("Phase C pattern shown in comments above (requires a live parsing model).")
```

The contract for `extend_template`:

- `parsing_models` lists every judge in the merged output (old + new). Old judges' rows pass through verbatim; new judges' rows run answering from replay and parsing live.
- `answering_models` must cover every answering identity in the prior (superset allowed).
- `replicate_count` must be `>=` the observed prior fan-out.
- `replay_store` must be `None`; the engine builds the store internally from `prior_results`.

For the full validation list see [Extending Runs: Validation Reference](../../core_concepts/extending-runs.md#12-validation-reference).

---

## Phase D: Attach a Rubric

`extend_rubric` is the dual: it reuses the prior parses and templates and only runs the rubric stage. We can execute this end-to-end in the notebook because the rubric-only pipeline does not call any parsing LLM, and rubric traits in this example are pure regex (no LLM trait).

```python
from karenina.schemas.entities import RegexRubricTrait, Rubric

bench.set_global_rubric(
    Rubric(
        regex_traits=[
            RegexRubricTrait(
                name="mentions_capital",
                description="True if the response mentions the word 'capital'.",
                pattern=r"(?i)capital",
            ),
            RegexRubricTrait(
                name="cites_country",
                description="True if the response names a country (France or Italy).",
                pattern=r"(?i)\b(france|italy)\b",
            ),
        ],
    )
)

phase_d_config = VerificationConfig(
    answering_models=[
        ModelConfig(id="ans", model_name="gpt-fake", interface="langchain", model_provider="openai"),
    ],
    parsing_models=[
        ModelConfig(id="par", model_name="gpt-fake", interface="langchain", model_provider="openai"),
    ],
    replicate_count=1,
)

phase_d_output = work_dir / "phase_d.json"
phase_d_sink = ProgressiveFileSink(
    output_path=phase_d_output,
    config=phase_d_config,
    benchmark_path=str(benchmark_path),
)

enriched = bench.extend_rubric(
    prior_results=reloaded_prior,
    config=phase_d_config,
    run_name="extend-tutorial",
    sink=phase_d_sink,
    store=False,
)
print("Phase D rows:", len(enriched.results))
print("Sidecars cleaned up:", not phase_d_output.with_suffix(".json.state").exists())
```

The shape contract for `extend_rubric` is stricter than for `extend_template`: `answering_models`, `parsing_models`, and `replicate_count` must equal the prior shape exactly (`==`, not `>=`). Row count is preserved: the merged set has one enriched row per prior row.

---

## Inspect the Merged Result

Each enriched row carries a fresh `rubric.regex_trait_scores` dict; the prior `template.verify_result` field is preserved.

```python
for row in enriched.results:
    print("---")
    print("question:", row.metadata.question_text)
    print("verify  :", row.template.verify_result)
    print("rubric  :", row.rubric.regex_trait_scores if row.rubric else None)
```

If the prior rows already carried rubric scores (for example because the prior run had used `evaluation_mode="template_and_rubric"`), the new trait dicts would be unioned with the existing ones by trait name. Same-name collisions raise `ValueError`; see [Extending Runs: Merge Semantics](../../core_concepts/extending-runs.md#42-merge-semantics-trait-name-union).

---

## Composing in Sequence

To add new judges *and* new rubric traits in one workflow, chain the two facades. The order matters: `extend_template` first to settle the row shape, then `extend_rubric` on the result. The rubric phase requires `replicate_count` equal to the observed fan-out of its input, so use the same `replicate_count` in both phases.

```python
# Pattern only (Phase C requires a live parsing model in the example above):
#
# template_extended = bench.extend_template(
#     prior_results=reloaded_prior,
#     config=phase_c_config,
#     run_name="extend-tutorial",
# )
# fully_extended = bench.extend_rubric(
#     prior_results=template_extended,
#     config=phase_d_config,
#     run_name="extend-tutorial",
# )
print("Sequential composition pattern shown in comments above")
```

See [Extending Runs: Composing the two facades](../../core_concepts/extending-runs.md#8-composing-extend_template-with-extend_rubric) for the full chaining contract.

---

## Cleanup

```python tags=["hide-cell"]
import shutil
shutil.rmtree(work_dir, ignore_errors=True)
```

---

## Next Steps

- [Extending Runs](../../core_concepts/extending-runs.md): conceptual reference, validation rules, merge semantics
- [Progressive Save and Resume](./progressive-save.md): sink internals, resume mechanics, partial-failure handling
- [Replay Store](../../advanced-pipeline/replay-store.md): the cache layer the extensions build internally
- [Sinks reference](../../reference/api/sinks.md): `ResultSink` Protocol, `ProgressiveFileSink`, `CompositeSink`, `DBSink`
