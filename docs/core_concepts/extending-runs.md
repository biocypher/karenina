---
jupyter:
  jupytext:
    formats: docs/core_concepts//md,docs/notebooks/core_concepts//ipynb
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

# Extending Prior Runs: `extend_template` and `extend_rubric`

A verification run captures enough state on every result row that later runs can reuse parts of it instead of re-executing the whole pipeline. Two facades on `Benchmark` expose this reuse as a first-class operation:

- [`Benchmark.extend_template`](../verification-pipeline/) extends a prior run along the template-verification axes: new judges, new answerers, more replicates. Prior `(question, answerer, replicate)` traces are served from a [ReplayStore](../../reference/replay/) so no answering tokens are spent twice; parsing runs live against new judges.
- [`Benchmark.extend_rubric`](../rubrics/) attaches a new rubric to a prior run and scores every existing trace against it. Answering is replayed, template parsing and verification are skipped entirely, and the new trait scores are merged onto copies of the prior rows.

```python tags=["hide-cell"]
from karenina.benchmark import Benchmark
from karenina.schemas import ModelConfig, VerificationConfig
from karenina.schemas.entities import LLMRubricTrait, RegexRubricTrait, Rubric
```

## 1. When to Use Which

| You want to... | Use | Output shape |
|---|---|---|
| Score the same traces with a second judge LLM | `extend_template` | prior rows + one new row per (qid, answerer, new-judge, replicate) |
| Add a second answering model to an existing matrix | `extend_template` | prior rows + one new row per (qid, new-answerer, judge, replicate) |
| Add more replicates (`replicate_count=3` -> `=4`) | `extend_template` | prior rows + one new row per added replicate |
| Attach a rubric to a prior run that didn't have one | `extend_rubric` | prior rows enriched in-place; row count unchanged |
| Add a trait to a rubric you already ran | `extend_rubric` | prior rows enriched; `rubric.*_trait_scores` unioned by trait name |

Composing both: if you need new judges *and* new rubric traits on top of a single prior run, call `extend_template` first, then `extend_rubric` on the merged result. The two facades do not compose in a single call.

## 2. Shared Mechanics

Both facades build a [ReplayStore](../../reference/replay/) from the prior `VerificationResultSet` and pass it through `VerificationConfig.replay_store`. At the generate-answer stage, any `(question, answering_model, replicate)` triple that hits the store replays the captured trace instead of invoking the LLM. Anything that misses runs live.

| Field captured in the store | `extend_template` | `extend_rubric` |
|---|---|---|
| Raw trace / messages | Yes | Yes |
| Parsed template fields | No (parsing always runs live) | No (parsing is skipped entirely) |

The two facades differ in what the pipeline does downstream:

- `extend_template` runs the full template-verification path in `evaluation_mode="template_only"`. New triples run answering live, parsing runs live for everyone. Prior triples are filtered out of the task queue via `VerificationConfig.skip_triples` so they pass through the merge verbatim.
- `extend_rubric` runs the pipeline in `evaluation_mode="rubric_only"` (the orchestrator keeps `GenerateAnswer` + rubric evaluation + finalize, drops parse/verify). Every prior triple flows through; the rubric scores are then merged onto copies of the prior rows by matching `(qid, answerer, judge, replicate)`.

In both cases the merged `VerificationResultSet` carries a single `run_name` so downstream consumers see one logical run.

## 3. `extend_template`: Three Composable Axes

### 3.1 Minimum viable call: add a second judge

The caller expresses the **final** shape in the config as a full union of everything they want in the output. The helper diffs against `prior_results` to decide what is new.

```python
answerer = ModelConfig(id="answerer", model_provider="anthropic", model_name="claude-haiku-4-5")
judge_a = ModelConfig(id="judge-a", model_provider="anthropic", model_name="claude-haiku-4-5", temperature=0.0)
judge_b = ModelConfig(id="judge-b", model_provider="openai", model_name="gpt-4.1-mini", temperature=0.0)

phase_a = VerificationConfig(answering_models=[answerer], parsing_models=[judge_a])
# prior = bench.run_verification(config=phase_a, run_name="demo")

phase_b = VerificationConfig(
    answering_models=[answerer],              # same as prior
    parsing_models=[judge_a, judge_b],        # old + new, full union
    replicate_count=1,                        # same as prior
)
# merged = bench.extend_template(prior_results=prior, config=phase_b, run_name="demo")
```

After the call, `merged.results` has the shape of a joint run with `parsing_models=[judge_a, judge_b]`. The `judge_a` rows come verbatim from `prior`; the `judge_b` rows were produced live with answering served from replay.

### 3.2 Add an answering model

```python
other_answerer = ModelConfig(id="other", model_provider="openai", model_name="gpt-4.1-mini")

phase_b = VerificationConfig(
    answering_models=[answerer, other_answerer],   # union: old + new
    parsing_models=[judge_a],                       # same as prior
    replicate_count=1,
)
# merged = bench.extend_template(prior_results=prior, config=phase_b)
```

`(answerer, judge_a)` rows pass through; `(other_answerer, judge_a)` rows run answering and parsing live.

### 3.3 Add replicates

```python
phase_b = VerificationConfig(
    answering_models=[answerer],
    parsing_models=[judge_a],
    replicate_count=3,                              # prior had 2
)
# merged = bench.extend_template(prior_results=prior, config=phase_b)
```

Replicates 1 and 2 pass through; replicate 3 runs answering and parsing live for every question.

### 3.4 All three axes at once

```python
phase_b = VerificationConfig(
    answering_models=[answerer, other_answerer],
    parsing_models=[judge_a, judge_b],
    replicate_count=3,
)
# merged = bench.extend_template(prior_results=prior, config=phase_b)
```

If prior was `(answerer, judge_a)` at `replicate_count=2` over `N` questions, the merged output has the full symmetric matrix `2 answerers x 2 judges x N questions x 3 replicates` rows. The `2N` prior rows pass through verbatim; everything else is produced live (with answering replayed for `answerer` replicates 1 and 2).

### 3.5 Validation

`extend_template` raises `ValueError` when:

- `prior_results` is empty.
- `config.replay_store` is pre-populated (the helper owns that slot).
- `config.answering_models` does not cover every answering identity in `prior_results` (superset allowed, missing is rejected).
- `config.replicate_count` is lower than the fan-out observed in `prior_results`. Replicate reduction is out of scope.
- `run_name` is inconsistent across `prior_results` and no override is passed.

## 4. `extend_rubric`: Attach a Rubric to a Prior Run

### 4.1 Usage pattern

The rubric lives on the benchmark. Attach it (global and/or per-question) before the call:

```python
# prior = bench.run_verification(config=phase_a, run_name="demo")

bench.set_global_rubric(
    Rubric(
        llm_traits=[
            LLMRubricTrait(
                name="answer_confidence",
                description="Rate how confident the response sounds, 1 (hedged) to 5 (definitive).",
                kind="score",
                min_score=1,
                max_score=5,
            ),
        ],
        regex_traits=[
            RegexRubricTrait(
                name="cites_source",
                description="True if the response mentions a source or reference.",
                pattern=r"(?i)\b(source|reference|according to)\b",
            ),
        ],
    )
)

phase_b = VerificationConfig(
    answering_models=[answerer],        # must match prior
    parsing_models=[judge_a],           # must match prior
    replicate_count=1,                  # must match prior
)
# merged = bench.extend_rubric(prior_results=prior, config=phase_b)
```

`merged.results` has **exactly the same length** as `prior.results`. Each row is a deep copy of the prior row with `rubric.llm_trait_scores` and `rubric.regex_trait_scores` now populated.

### 4.2 Merge semantics: trait-name union

If a prior row already carried rubric scores (for example, the seed ran under `evaluation_mode="template_and_rubric"`), the new trait dictionaries are unioned with the old ones by trait name.

- New trait `clarity` added to a row that already had `coherence` -> merged row has both.
- Same-name trait on both sides -> `ValueError`, naming the colliding trait and the bucket (one of `llm_trait_scores`, `llm_trait_labels`, `regex_trait_scores`, `callable_trait_scores`, `agentic_trait_scores`, `agentic_trait_investigation_traces`).

This rules out silent overwrites: if you want to re-run an existing trait, clear it from the prior results first.

### 4.3 Shape-match requirements

Unlike `extend_template`, `extend_rubric` preserves the prior shape exactly. All three axes must equal what was observed in `prior_results`:

- `config.answering_models` identities match.
- `config.parsing_models` identities match.
- `config.replicate_count` equals the observed replicate fan-out (`==`, not `>=`).

This is intentional. Changing the shape while extending the rubric would require either dropping prior rows (inconsistent) or duplicating them (breaks the 1:1 mapping). If you need both, run `extend_template` first, then `extend_rubric` on the merged result.

### 4.4 Scope

Supported trait types: LLM, regex, callable, agentic, plus `DynamicRubric` presence-gated traits.

**Metric traits are rejected.** Metric evaluation consumes parsed template fields, and the rubric-only pipeline does not produce them. Attaching a metric trait to the benchmark before calling `extend_rubric` raises `ValueError`. Use `template_and_rubric` on a fresh run if you need metric evaluation.

### 4.5 Validation

`extend_rubric` raises `ValueError` when:

- `prior_results` is empty.
- `config.replay_store` is pre-populated.
- `config.evaluation_mode` is set to `"template_and_rubric"` (the caller should not preset a conflicting mode; the helper rewrites it to `"rubric_only"` internally).
- The benchmark has no rubric attached anywhere (global or per-question) for any question in `prior_results`.
- Any attached rubric contains metric traits.
- `config.answering_models`, `config.parsing_models`, or `config.replicate_count` disagree with the observed prior shape.

## 5. Reading the Merged Result

Both facades return a `VerificationResultSet` with a single `run_name`. They differ in what changes:

| Aspect | `extend_template` | `extend_rubric` |
|---|---|---|
| Row count vs prior | Can grow | Equal |
| Prior rows | Passed through verbatim | Copied and enriched |
| New rows | Appended | None |
| `rubric.*` fields on prior rows | Unchanged | Populated / unioned |
| `result_id` stability | Preserved on prior rows | Preserved (deep copy keeps the same id) |

The `run_name` on every row is stamped to the effective name: the `run_name=` override if passed, otherwise the run name inferred from `prior_results` (rejected when inconsistent).

## 6. Storage Flag

Both facades accept `store: bool = True`. When true, the merged set is written to the benchmark's results manager under the effective `run_name` so `bench.get_verification_results(run_name=...)` can retrieve it. Set `store=False` for exploratory use where you only want the returned object.

## 7. Further Reading

- [Verification Pipeline](../verification-pipeline/): the 13-stage engine both facades drive.
- [Evaluation Modes](../evaluation-modes/): how `rubric_only` differs from `template_only` and `template_and_rubric`.
- [Rubrics](../../../core_concepts/rubrics/): trait types and `DynamicRubric`.
- `karenina/src/karenina/benchmark/verification/extension.py`: the helper implementations (`extend_template_run`, `extend_rubric_run`) if you need to read the exact validation and merge code.
