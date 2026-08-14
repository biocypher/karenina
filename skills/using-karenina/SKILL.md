---
name: using-karenina
description: >
  Guide users through LLM evaluation with karenina: benchmark creation,
  template/rubric authoring, verification pipeline configuration, and
  results analysis. Use when evaluating LLMs, building benchmarks, writing
  answer templates, defining rubrics, running verification, analyzing results,
  or working with TaskEval/Scenarios. Activate even if the user doesn't name
  karenina directly but describes evaluating an LLM's outputs against criteria,
  testing model quality, or benchmarking AI responses. This skill provides
  shared context (cross-cutting gotchas, API references, imports) and then
  routes to a leaf skill: karenina-qa, karenina-task-eval, karenina-scenarios,
  karenina-template-authoring, karenina-rubric-authoring, karenina-verification,
  karenina-cli, karenina-manual, or karenina-results. It is crucial to read and follow through
  the routing instructions inside this skill to invoke the correct leaf skill
  before starting work.
---

# Using Karenina

Karenina is a Python framework for structured LLM evaluation. It has three modes:

- **Benchmark** (single-turn, closed-loop): define questions with ground truth, generate responses via an answering LLM, evaluate with a judge LLM using structured templates.
- **Scenarios** (multi-turn, closed-loop): define conversation graphs with branching and outcome criteria.
- **TaskEval** (open-loop): supply pre-recorded outputs, evaluate with the same template/rubric engine.

All three share the evaluation engine: answer templates (BaseAnswer + VerifiedField), rubrics (5 trait types), and a verification pipeline whose stage sequence is assembled from config (validate → generate → guards/autofails → abstention/sufficiency checks → parse → verify → embedding check → deep judgment → rubric evaluation → finalize).

## Decision Tree

Read the user's message. Match the first applicable route. If ambiguous, ask ONE clarifying question.

### User wants to evaluate an LLM, build a benchmark, or test model quality

**Has pre-recorded outputs they want to score (chatbot logs, API responses, agent traces, JSON files with model outputs)?**
Invoke `karenina-task-eval`.
Trigger phrases: "I have", "pre-recorded", "collected", "logs", "traces", "JSON file with responses", "score existing outputs", "evaluate pre-collected data without running an LLM", "existing model responses", "chatbot logs", "agent traces".
Context: the user already has LLM outputs collected and wants to assess them against structured criteria. No LLM generation happens; only parsing and evaluation via the TaskEval API with logging, template attachment, multi-step evaluation, and merge strategies. Secondary quality goals beyond the first step (e.g. "check tone") re-route to `karenina-rubric-authoring` per the multi-intent rule.

**Wants multi-turn conversation testing with branching?**
Invoke `karenina-scenarios`.
Trigger phrases: "multi-turn", "conversation", "branching", "dialogue paths", "sycophancy", "scenario graph", "nodes", "edges", "outcome criteria", "TurnCheck".
Context: the user wants multi-turn scenario evaluation with branching dialogue paths, covering scenario graphs, nodes, edges, outcome criteria, and TurnCheck.

Clarification for ambiguous cases: multi-turn with branching between conversation paths = scenarios. Multiple independent single-turn questions = QA benchmark. Evaluating existing conversation logs = task-eval (not scenarios). If unclear, ask: "Are your questions part of a single multi-turn conversation with branching, or are they independent questions?"

**Wants single-turn Q&A evaluation (default)?**
Invoke `karenina-qa`.
Trigger phrases: "benchmark", "QA", "factual questions", "extraction tasks", "single-turn", "question definition", "template attachment", "checkpoint", "benchmark creation", "loading", "iteration".
Context: the user wants to build or run single-turn QA benchmarks covering question definition, template attachment, checkpoint management, running verification, and result analysis.

### User wants to write, fix, or modify a template, BaseAnswer, or VerifiedField

Invoke `karenina-template-authoring`.
Trigger phrases: "template", "BaseAnswer", "VerifiedField", "field types", "verification primitives", "description writing", "ground truth", "extract from LLM responses", "verify correctness".
Context: describe what the user is trying to evaluate and what fields they need. Covers creating answer templates (BaseAnswer subclasses with VerifiedField), field types, verification primitives, description writing, and ground truth configuration.

### User wants to define quality criteria, a rubric, or traits

Invoke `karenina-rubric-authoring`.
Trigger phrases: "rubric", "trait", "LLM trait", "regex trait", "callable trait", "metric trait", "agentic trait", "DynamicRubric", "quality criteria", "safety", "conciseness", "citation quality", "format compliance", "conditional evaluation".
Context: describe the quality dimensions the user wants to assess. Covers all 5 trait types (LLM, regex, callable, metric, agentic) and DynamicRubric for conditional evaluation.

### User wants to configure the pipeline, run verification, or understand pipeline output

Invoke `karenina-verification`.
Trigger phrases: "VerificationConfig", "ModelConfig", "adapter", "guard", "replicate", "preset", "prompt assembly", "deep judgment", "MCP integration", "run_verification()", "auto-fail", "pipeline execution", "stage failure", "pipeline output".
Context: describe the configuration or execution task. Covers model selection, adapter choice, guard configuration, replicates, presets, prompt assembly, deep judgment, MCP integration, pipeline execution, and initial result inspection.

### User wants to extend a prior verification run (re-judge, add answerers/replicates, add a rubric)

Invoke `karenina-verification` (task 9 covers both `Benchmark.extend_template` and `Benchmark.extend_rubric`).
Trigger phrases: "extend_template", "extend_rubric", "re-judge", "rejudge", "add a new judge", "add a parsing model to existing results", "add more replicates", "add more answerers to my run", "score my traces against a new rubric", "attach a rubric to existing results", "replay", "ReplayStore", "without regenerating", "reuse prior answers".
Context: the user has a completed run and wants to extend it along one axis (new judge, more answerers, more replicates) or score the same traces against a new rubric. Prior traces are replayed; no answer regeneration. See `references/core_concepts/extending-runs.md` for the full treatment.

### User wants to use the full pipeline but provide answers manually (ManualAdapter)

Invoke `karenina-manual`.
Trigger phrases: "ManualAdapter", "manual", "provide answers myself", "type the answers", "full pipeline without LLM generation", "offline evaluation".
Context: same machinery as QA benchmarks (full verification pipeline with all stages), but the user produces answers instead of an LLM. ManualAdapter intercepts the generation stage. Differs from TaskEval: manual uses the full opinionated pipeline; TaskEval is lightweight, skips pipeline stages entirely.

### User wants to run evaluation from the command line (CLI, terminal, bash)

Invoke `karenina-cli`.
Trigger phrases: "CLI", "command line", "terminal", "bash", "karenina verify", "run from shell", "preset", "karenina command", "shell", "run verification from terminal".
Context: the user wants to run verification using terminal commands rather than Python scripts. Covers preset creation, `karenina verify` with flags, progressive save/resume, model comparison, and result export. If the user already has a checkpoint and just wants to run it, this is the fastest path.

### User asks about results AFTER verification has already run (DataFrames, export, comparison)

Invoke `karenina-results`.
Trigger phrases: "DataFrame", "export", "compare scores", "compare across models", "compare across runs", "VerificationResult structure", "scoring", "export formats", "template results", "rubric results", "scenario results".
Context: the user has already run verification and wants to navigate, analyze, compare, or export results. Note: "I have outputs and want to score them" is task-eval, not results. Results is for post-verification analysis only.

### User asks a conceptual question ("what is karenina", "how does X work")

Answer directly from `references/`. Do not delegate. Read the relevant reference file and answer the question.

### User wants to do multiple things (build + run + analyze)

Route to the FIRST step they haven't completed yet. After that skill finishes, re-evaluate and route to the next step. Do not try to route to multiple skills simultaneously.

### Ambiguous intent

Ask ONE question: "Are you evaluating an LLM in real-time, or do you have pre-recorded outputs you want to assess?"

If that still does not resolve, ask: "Are you looking for an explanation, or do you need to build or configure something?"

## Delegation Protocol

When delegating to a skill, follow these rules:

1. **Always include context.** The delegated skill starts fresh and does not inherit the current conversation. Provide enough context for the delegated skill to proceed without asking the user to repeat themselves.

2. **Caller resumes after delegation.** When the delegated skill finishes, continue the current procedure at the next step. Do not re-invoke the dispatcher.

3. **No circular delegation.** The delegation graph is a DAG. Leaf skills (qa, task-eval, scenarios, template-authoring, rubric-authoring, verification, cli, manual, results) never delegate back to modality skills.

4. **Delegation is a suggestion.** If the user already has a working template or config, skip the delegation step.

5. **GET INFORMATION = answer from references. DO SOMETHING = delegate.** Conceptual questions are answered directly; action-oriented requests route to the appropriate skill.

## Cross-Cutting Gotchas

These apply across all evaluation modes. Read them before doing any karenina work.

### Templates

- Template classes MUST inherit from `BaseAnswer`, not Pydantic `BaseModel`. Using `BaseModel` directly will silently break verification because the pipeline expects `BaseAnswer` methods.

- `VerifiedField.description` is what the **judge LLM** sees as its parsing instruction. Write it for the judge: be explicit about edge cases, specify what counts as True vs False, describe boundary conditions. Do not write it as documentation for humans.

- `ground_truth` must match the field's Python type. For `BooleanMatch`, ground_truth must be `bool` (True/False). For `ExactMatch`, ground_truth must be `str`. For numeric primitives (`NumericExact`, `NumericTolerance`, `NumericRange`), ground_truth must be `int` or `float`. A type mismatch causes a silent verification failure.

- Do not use `ExactMatch` for free-text fields. The judge LLM will almost never produce an exact string match with the ground truth. Use `BooleanMatch` instead: have the judge answer True/False about whether the response contains the expected information.

### Rubrics

- **`Rubric` uses typed trait lists, not a generic `traits` parameter.** Traits go into `llm_traits`, `regex_traits`, `callable_traits`, `metric_traits`, or `agentic_traits`. There is no unified `traits` list or `name` field on `Rubric`.

- **`LLMRubricTrait` requires a `kind` field** (`"boolean"`, `"score"`, or `"literal"`). This is a required parameter with no default. Omitting it raises a validation error.

- **`RegexRubricTrait` uses `invert_result` and `case_sensitive`, not `expected_match`.** For negative matching (pattern should be absent), use `invert_result=True`. For case-insensitive matching, use `case_sensitive=False`. `higher_is_better` is `bool | None` on all trait types, defaulting to `True` on LLM, regex, and callable traits. Set to `None` when directionality does not apply.

- **Attach rubrics with `benchmark.set_global_rubric(rubric)`**, not `add_rubric()`. For question-specific rubrics, use `benchmark.set_question_rubric(question_id, rubric)`. Both methods also accept a list of traits (mixed types) instead of a `Rubric` object; use `Rubric.from_traits()` if you need to construct a `Rubric` from a trait list explicitly.

- `LLMRubricTrait.description` IS the rubric evaluator's full prompt. The evaluator LLM receives this description as its instruction. Include explicit True/False semantics: "Answer True if the response does X. Answer False if Y." Vague descriptions produce unreliable evaluations.

- All five rubric trait types (LLM, regex, callable, metric, agentic) support an optional `summary` field. This is a short concept label used for dynamic rubric presence checking. Every trait in a `DynamicRubric` must have at least `summary` or `description`, or the presence check will skip it.

- `DynamicRubric` conditionally evaluates traits based on whether the relevant concept appears in the response. Before rubric evaluation, the pipeline runs a batch presence check. Traits whose concept is absent are skipped entirely.

### Configuration

- `VerificationConfig.evaluation_mode` defaults to `"template_only"` (`Literal["template_only", "template_and_rubric", "rubric_only"]`). There is no `"auto"` value and no auto-upgrade: attaching a rubric does NOT change the mode. To evaluate attached rubrics you MUST set `evaluation_mode="template_and_rubric"` (or `"rubric_only"`). `rubric_enabled` is a computed property, not a config field. `rubric_only` is ignored (with a UserWarning) on scenario runs, which auto-detect from rubric presence.

- **Do not hardcode model names with date suffixes unless the user explicitly asks you to do so** Use the base model name (e.g., `claude-haiku-4-5`, `claude-sonnet-4-5`). Always ask the user which models to use rather than assuming defaults.

- `parsing_only=True` is required for TaskEval. Without it, the pipeline tries to generate answers from an answering LLM, which fails because TaskEval has no answering model. In Benchmark mode, `parsing_only=True` skips generation, which is only useful for re-evaluating existing responses.

- Adapters use duck typing (no explicit Protocol inheritance). Create adapters only through factory functions: `get_llm()`, `get_agent()`, `get_parser()`. Never instantiate adapter classes directly.

### Questions and Checkpoints

- `Question.id` is auto-generated from the MD5 hash of the question text. If two questions have identical text, they get the same ID, and the second silently overwrites the first. Use distinct question text, or set `id` explicitly.

- Checkpoint files are JSON-LD. Do not manually edit them. Use `Benchmark.load()` to resume from a checkpoint. Manual edits can corrupt the file and cause silent data loss.

### Pipeline Guards

- Guards (abstention check, sufficiency check, recursion limit, trace validation) auto-fail questions BEFORE parsing. If a question unexpectedly fails with `verify_result=False` and no parsed output, check: `result.template.recursion_limit_reached`, `result.template.abstention_detected`, `result.template.abstention_reasoning`, and `result.metadata.failure`.

- To disable guards that are causing false positives:
  `VerificationConfig(abstention_enabled=False, sufficiency_enabled=False)`

- Autofail guards (recursion limit, trace validation, placeholder retry) run immediately after generation; abstention and sufficiency are opt-in checks that run before parsing (enabled via `abstention_enabled` / `sufficiency_enabled`). If a guard triggers, no parsing, verification, or rubric evaluation occurs for that question. The pipeline has no fixed stage numbering — its stage sequence is assembled from `VerificationConfig`.

### Imports

- `BaseAnswer` and `VerifiedField`: `from karenina.schemas.entities import BaseAnswer, VerifiedField`
- Verification primitives: `from karenina.schemas.primitives import BooleanMatch, ExactMatch, NumericExact, NumericTolerance, NumericRange`
- Rubric traits: `from karenina.schemas.entities.rubric import LLMRubricTrait, RegexRubricTrait, CallableRubricTrait, MetricRubricTrait, AgenticRubricTrait, Rubric, DynamicRubric`
- Configuration: `from karenina.schemas.verification.config import VerificationConfig`
- Models: `from karenina.schemas.config.models import ModelConfig`
- Failure taxonomy: `from karenina.schemas.results.failure import Failure, FailureCategory`
- Prompt config: `from karenina.schemas.verification.prompt_config import PromptConfig`
- Few-shot config: `from karenina.schemas.config import FewShotConfig`
- TaskEval: `from karenina.benchmark.task_eval import TaskEval`
- Single-question verification: `from karenina.benchmark import run_question_verification` (public alias of `run_single_model_verification`)
- Messages: `from karenina.ports.messages import Message`
- ManualTraces: `from karenina.adapters.manual import ManualTraces`

### Import Pitfalls

- **`VerifiedField` is NOT in `answer.py`**: use `from karenina.schemas.entities import VerifiedField`, not `from karenina.schemas.entities.answer import VerifiedField`.
- **Do NOT import from internal pipeline modules.** These are not part of the public API:
  - `karenina.benchmark.verification.runner` (its `run_single_model_verification` is public only via the `karenina.benchmark` alias `run_question_verification` — prefer that)
  - `karenina.benchmark.verification.stages.*` (internal stages)
  - `karenina.schemas.verification.result_components` (use `karenina.schemas.verification`)
- **`ManualTraces`** is in `karenina.adapters.manual`, not `karenina.schemas.config.models`.
- **There is no `VerificationContext`**; the correct name is `VerificationConfig`.
- **`MetricRubricTrait`** not `MetricTrait`: the class is called `MetricRubricTrait`.

<!-- AUTO:benchmark-api -->
### Benchmark API Quick Reference

> Auto-generated for karenina v0.1.0.

Do not guess method names. If a method is not listed here,
verify it exists with Grep before calling it.

> All karenina Pydantic models use `extra="forbid"`. Passing
> an undefined field raises `ValidationError`.

#### Create / Load / Save

| Method | Returns |
|--------|---------|
| `Benchmark(name, description=, version=, creator=, workspace_root=)` | `Benchmark` |
| `Benchmark.create(name, description=, version=, creator=, workspace_root=)` | `Benchmark` |
| `Benchmark.load(path, workspace_root=)` | `Benchmark` |
| `save(path, save_deep_judgment_config=)` | `None` |

#### Questions

| Method | Returns |
|--------|---------|
| `add_question(question, raw_answer=, answer_template=, question_id=, ...)` | `str` |
| `add_questions(questions_data, finished=)` | `list[str]` |
| `get_question(question_id)` | `dict[str, Any]` |
| `get_question_ids()` | `list[str]` |
| `get_all_questions(ids_only=)` | `list[str] | list[dict[str, Any]]` |
| `get_question_as_object(question_id)` | `Question` |
| `remove_question(question_id)` | `bool` |

#### Templates

| Method | Returns |
|--------|---------|
| `add_answer_template(question_id, template_code)` | `None` |
| `get_template(question_id)` | `str` |
| `has_template(question_id)` | `bool` |
| `update_template(question_id, template_code)` | `None` |
| `apply_global_template(template_code)` | `list[str]` |
| `get_missing_templates(ids_only=)` | `list[str] | list[dict[str, Any]]` |

#### Rubrics

| Method | Returns |
|--------|---------|
| `set_global_rubric(rubric)` | `None` |
| `get_global_rubric()` | `Rubric | None` |
| `set_question_rubric(question_id, rubric)` | `None` |
| `get_question_rubric(question_id)` | `Rubric | None` |
| `add_global_rubric_trait(trait)` | `None` |
| `add_question_rubric_trait(question_id, trait)` | `None` |
| `clear_global_rubric()` | `bool` |
| `remove_question_rubric(question_id)` | `bool` |

#### Verification

| Method | Returns |
|--------|---------|
| `run_verification(config, question_ids=, run_name=, async_enabled=, ...)` | `VerificationResultSet` |
| `resume_verification(state_path, config=, question_ids=, run_name=, ...)` | `VerificationResultSet` |
| `extend_template(prior_results, config, run_name=, question_ids=, ...)` | `VerificationResultSet` |
| `extend_rubric(prior_results, config, run_name=, question_ids=, ...)` | `VerificationResultSet` |

#### Results

| Method | Returns |
|--------|---------|
| `get_verification_results(question_ids=, run_name=)` | `dict[str, VerificationResult]` |
| `get_verification_history(question_id=)` | `dict[str, dict[str, VerificationResult]]` |
| `get_verification_summary(run_name=)` | `dict[str, Any]` |
| `store_verification_results(results, run_name=)` | `None` |
| `export_verification_results_to_file(file_path, question_ids=, run_name=, format=, global_rubric=)` | `None` |
| `clear_verification_results(question_ids=, run_name=)` | `int` |

#### Methods That Do NOT Exist on Benchmark

| Hallucinated Name | Use Instead |
|-------------------|-------------|
| `set_template()` | `add_answer_template() or update_template()` |
| `get_latest_results()` | `ResultsStore.get_latest() (get_verification_results() is deprecated)` |
| `add_rubric()` | `set_global_rubric() or set_question_rubric()` |
| `load_checkpoint()` | `Benchmark.load()` |
| `verify()` | `run_verification()` |

#### Deprecated Benchmark Methods (emit DeprecationWarning)

| Method | Use Instead |
|--------|-------------|
| `store_verification_results()` | `ResultsStore.add()` |
| `get_verification_results()` | `ResultsStore.get_by_run() / get_latest()` |
| `get_verification_history()` | `ResultsStore.get_by_question()` |
| `get_verification_summary()` | `ResultsStore.get_summary()` |
| `export_verification_results_to_file()` | `ResultsStore.export_to_file()` |
| `clear_verification_results()` | `ResultsStore.clear()` |

<!-- /AUTO:benchmark-api -->

<!-- AUTO:result-api -->
### VerificationResult Accessor Reference

> Auto-generated for karenina v0.1.0.

Always access through sub-objects (`result.metadata.*`,
`result.template.*`, `result.rubric.*`), never directly on result.

#### metadata (always present)

| Path | Type |
|------|------|
| `result.metadata.question_id` | `str` |
| `result.metadata.template_id` | `str` |
| `result.metadata.question_text` | `str` |
| `result.metadata.failure` | `Failure | None` |
| `result.metadata.caveats` | `list[Caveat]` |
| `result.metadata.warnings` | `list[str]` |
| `result.metadata.retry_counts` | `dict[str, dict[str, int]] | None` |
| `result.metadata.evaluation_mode` | `str | None` |
| `result.metadata.answering` | `ModelIdentity` |
| `result.metadata.parsing` | `ModelIdentity` |
| `result.metadata.execution_time` | `float` |
| `result.metadata.timestamp` | `str` |
| `result.metadata.result_id` | `str` |
| `result.metadata.run_name` | `str | None` |
| `result.metadata.replicate` | `int | None` |
| `result.metadata.scenario_id` | `str | None` |
| `result.metadata.scenario_node` | `str | None` |
| `result.metadata.scenario_turn` | `int | None` |
| `result.metadata.scenario_path` | `list[str] | None` |

#### template (when template evaluation ran)

| Path | Type |
|------|------|
| `result.template.verify_result` | `bool | None` |
| `result.template.raw_llm_response` | `str` |
| `result.template.parsed_llm_response` | `dict[str, Any] | None` |
| `result.template.parsed_gt_response` | `dict[str, Any] | None` |
| `result.template.abstention_detected` | `bool | None` |
| `result.template.abstention_reasoning` | `str | None` |
| `result.template.embedding_similarity_score` | `float | None` |
| `result.template.recursion_limit_reached` | `bool` |
| `result.template.field_results` | `dict[str, bool | None] | None` |
| `result.template.verify_granular_result` | `Any | None` |
| `result.template.response_timeout_partial` | `bool` |
| `result.template.usage_metadata` | `dict[str, dict[str, Any]] | None` |

#### rubric (when rubric evaluation ran)

| Path | Type |
|------|------|
| `result.rubric.rubric_evaluation_performed` | `bool` |
| `result.rubric.llm_trait_scores` | `dict[str, Any] | None` |
| `result.rubric.llm_trait_labels` | `dict[str, str] | None` |
| `result.rubric.regex_trait_scores` | `dict[str, bool] | None` |
| `result.rubric.callable_trait_scores` | `dict[str, bool | int | float] | None` |
| `result.rubric.metric_trait_scores` | `dict[str, dict[str, float]] | None` |
| `result.rubric.metric_trait_confusion_lists` | `dict[str, dict[str, list[str]]] | None` |
| `result.rubric.agentic_trait_scores` | `dict[str, int | bool | float | str | list[Any] | None] | None` |
| `result.rubric.dynamic_rubric_skipped_traits` | `dict[str, str] | None` |
| `result.rubric.dynamic_rubric_promoted_traits` | `list[str] | None` |
| `result.rubric.get_all_trait_scores()` | `dict[str, int | bool | float | str | list[Any] | dict[str, float] | None]` |
| `result.rubric.get_trait_by_name(name)` | `tuple[Any, str] | None` |
| `result.rubric.get_llm_trait_labels()` | `dict[str, str]` |

#### Paths That Do NOT Exist on VerificationResult

| Wrong | Correct |
|-------|---------|
| `result.template.parsed_output` | `result.template.parsed_llm_response` |
| `result.verify_result` | `result.template.verify_result` |
| `result.question_id` | `result.metadata.question_id` |
| `result.raw_response` | `result.template.raw_llm_response` |
| `result.error` | `result.metadata.failure (Failure.category / .stage / .reason)` |
| `result.metadata.error` | `result.metadata.failure (Failure.category / .stage / .reason)` |
| `result.metadata.completed_without_errors` | `result.metadata.failure is None` |

<!-- /AUTO:result-api -->


## Reference Index

When answering conceptual questions directly, read the relevant reference file.

| Topic | Reference Path |
|-------|---------------|
| Getting started, installation, quickstart | `references/getting-started/` |
| Answer templates, VerifiedField, BaseAnswer | `references/core_concepts/answer-templates.md` |
| Verification primitives (BooleanMatch, ExactMatch, etc.) | `references/core_concepts/verification-primitives.md` |
| Rubrics (all 5 trait types) | `references/core_concepts/rubrics/` |
| Questions, benchmarks, checkpoints | `references/core_concepts/questions-and-benchmarks/` |
| Verification pipeline (config-assembled stages) | `references/core_concepts/verification-pipeline.md` |
| Templates vs rubrics decision guide | `references/core_concepts/template-vs-rubric.md` |
| Evaluation modes (template_only, rubric_only, template_and_rubric) | `references/core_concepts/evaluation-modes.md` |
| Scenarios (multi-turn, branching, outcomes) | `references/core_concepts/scenarios/` |
| TaskEval (pre-recorded outputs) | `references/core_concepts/task-eval.md` |
| Manual interface, offline evaluation | `references/core_concepts/manual-interface.md` |
| Results and scoring | `references/core_concepts/results-and-scoring.md` |
| Adapters and ports | `references/advanced-adapters/` |
| Prompt assembly | `references/advanced-pipeline/prompt-assembly.md` |
| Deep judgment | `references/advanced-pipeline/deep-judgment-templates.md` |
| Agentic evaluation | `references/core_concepts/agentic-evaluation.md` |
| Configuration (VerificationConfig, ModelConfig) | `references/reference/configuration/` |
| CLI commands (verify, init, preset, serve) | `references/reference/cli/` |
| Results analysis workflows | `references/workflows/analyzing-results/` |
| Creating benchmarks workflows | `references/workflows/creating-benchmarks/` |
| Running verification workflows | `references/workflows/running-verification/` |
| TaskEval workflows | `references/workflows/task-eval/` |
| Scenario workflows | `references/workflows/scenarios/` |
| Philosophy, design principles | `references/home/philosophy.md` |
