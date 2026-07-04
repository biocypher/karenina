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

# Agentic Evaluation

This scenario walks through a complete agentic evaluation end-to-end, using a real BixBench task. The answering agent works in a filesystem workspace (reading data, writing code, executing analysis), and the judge agent independently verifies the results by re-examining the same workspace.

**What you'll learn:**

- Set up a workspace-based coding benchmark
- Define an answer template with `VerifiedField` primitives for a data analysis task
- Configure agentic answering (via `claude_agent_sdk`) and agentic parsing
- Run verification and inspect the investigation trace
- Compare agentic vs trace-only parsing on the same answer
- Combine agentic parsing with agentic rubric evaluation using `AgenticRubricTrait`

For the conceptual background on how agentic evaluation works, see [Agentic Evaluation (Concepts)](../../core_concepts/agentic-evaluation/). For pipeline internals (stage ordering, context modes, investigation prompts), see [Agentic Evaluation (Advanced)](../../advanced-pipeline/agentic-evaluation/).

```python tags=["hide-cell"]
# Setup cell: creates mock data and patches run_verification so that all code
# cells execute without live LLM calls.
# This cell is hidden in the rendered documentation.
import datetime
import tempfile
from pathlib import Path

from karenina import Benchmark
from karenina.schemas.config import ModelConfig
from karenina.schemas.entities import Question
from karenina.schemas.results import VerificationResultSet
from karenina.schemas.verification import (
    VerificationConfig,
    VerificationResult,
    VerificationResultMetadata,
    VerificationResultRubric,
    VerificationResultTemplate,
)
from karenina.schemas.verification.model_identity import ModelIdentity

# ---------------------------------------------------------------------------
# Mock identities and timestamp
# ---------------------------------------------------------------------------
_answering = ModelIdentity(model_name="claude-sonnet-4-6", interface="claude_agent_sdk")
_parsing = ModelIdentity(model_name="claude-sonnet-4-6", interface="claude_agent_sdk")
_ts = datetime.datetime.now(tz=datetime.UTC).isoformat()

# ---------------------------------------------------------------------------
# Template-level mock data (agentic parsing results)
# ---------------------------------------------------------------------------
_mock_template = VerificationResultTemplate(
    raw_llm_response=(
        "I'll analyze the patient data using logistic regression.\n"
        "import pandas as pd\nimport statsmodels.api as sm\n\n"
        "Age is significant (p=0.002). BMI is not significant (p=0.087).\n"
        "Predicted probability at age 65: 0.3948\n"
        "AIC for age-only model: 104.2"
    ),
    verify_result=True,
    template_verification_performed=True,
    agentic_parsing_performed=True,
    investigation_trace=(
        "Investigation: I examined the workspace and found analysis.py.\n"
        "The script imports statsmodels and runs sm.Logit() for logistic regression.\n"
        "Running the script confirms: age p-value=0.002 (significant), "
        "BMI p-value=0.087 (not significant).\n"
        "Predicted probability at age 65: 0.3948. AIC: 104.2."
    ),
    parsed_gt_response={
        "age_significant": True,
        "bmi_not_significant": True,
        "age_predicted_probability_65": 0.3953,
        "age_model_aic": 104.14,
    },
    parsed_llm_response={
        "age_significant": True,
        "bmi_not_significant": True,
        "age_predicted_probability_65": 0.3948,
        "age_model_aic": 104.2,
    },
)

# ---------------------------------------------------------------------------
# Rubric-level mock data (agentic rubric evaluation results)
# ---------------------------------------------------------------------------
_mock_rubric = VerificationResultRubric(
    rubric_evaluation_performed=True,
    rubric_evaluation_strategy="individual",
    agentic_trait_scores={"logistic_regression_library": 0},
    agentic_trait_investigation_traces={
        "logistic_regression_library": (
            "Investigation trace: Examined workspace files. Found analysis.py "
            "containing 'import statsmodels.api as sm'. The logistic regression "
            "was implemented using sm.Logit(endog, exog).fit(). No scikit-learn "
            "imports found. Conclusion: the library used is statsmodels."
        ),
    },
)

# ---------------------------------------------------------------------------
# Assemble mock result
# ---------------------------------------------------------------------------
_question_id = "mock-bix51-question-id"
_result_id = VerificationResultMetadata.compute_result_id(
    _question_id, _answering, _parsing, _ts,
)

_mock_result = VerificationResult(
    metadata=VerificationResultMetadata(
        question_id=_question_id,
        template_id="tmpl_bix51",
        failure=None,
        caveats=[],
        question_text=(
            "Using the patient data in data.xlsx, perform logistic regression analysis "
            "to determine which demographics (age, BMI, gender) predict treatment "
            "remission."
        ),
        raw_answer=(
            "Age is significant (p=0.002), BMI is not (p=0.087). "
            "P(remission|age=65)=0.395, AIC=104.1"
        ),
        answering=_answering,
        parsing=_parsing,
        execution_time=45.3,
        timestamp=_ts,
        result_id=_result_id,
    ),
    template=_mock_template,
    rubric=_mock_rubric,
)

_mock_result_set = VerificationResultSet(results=[_mock_result])

# ---------------------------------------------------------------------------
# Build a mock result for "template_only" mode (no rubric)
# ---------------------------------------------------------------------------
_mock_result_template_only = VerificationResult(
    metadata=VerificationResultMetadata(
        question_id=_question_id,
        template_id="tmpl_bix51",
        failure=None,
        caveats=[],
        question_text=_mock_result.metadata.question_text,
        raw_answer=_mock_result.metadata.raw_answer,
        answering=_answering,
        parsing=_parsing,
        execution_time=38.7,
        timestamp=_ts,
        result_id=_result_id,
    ),
    template=_mock_template,
)

_mock_result_set_template_only = VerificationResultSet(results=[_mock_result_template_only])

# ---------------------------------------------------------------------------
# Build a mock result for trace-only comparison
# ---------------------------------------------------------------------------
_mock_trace_only_template = VerificationResultTemplate(
    raw_llm_response=_mock_template.raw_llm_response,
    verify_result=True,
    template_verification_performed=True,
    agentic_parsing_performed=False,
    parsed_gt_response=_mock_template.parsed_gt_response,
    parsed_llm_response={
        "age_significant": True,
        "bmi_not_significant": True,
        "age_predicted_probability_65": 0.395,
        "age_model_aic": 104.1,
    },
)

_mock_trace_only_result = VerificationResult(
    metadata=VerificationResultMetadata(
        question_id=_question_id,
        template_id="tmpl_bix51",
        failure=None,
        caveats=[],
        question_text=_mock_result.metadata.question_text,
        raw_answer=_mock_result.metadata.raw_answer,
        answering=_answering,
        parsing=_parsing,
        execution_time=12.1,
        timestamp=_ts,
        result_id=_result_id,
    ),
    template=_mock_trace_only_template,
)

# ---------------------------------------------------------------------------
# Track which call we are on so different sections get appropriate results
# ---------------------------------------------------------------------------
_run_call_count = 0


def _patched_run(self, config, **kwargs):
    global _run_call_count
    _run_call_count += 1
    # First call: template-only (Steps 1-5)
    # Second call: template_and_rubric (agentic rubric section)
    if _run_call_count == 1:
        return _mock_result_set_template_only
    return _mock_result_set


Benchmark.run_verification = _patched_run

# Provide a temporary directory for EXPERIMENT_DIR so save/load calls work
EXPERIMENT_DIR = Path(tempfile.mkdtemp())
(EXPERIMENT_DIR / "workspace").mkdir(exist_ok=True)
(EXPERIMENT_DIR / "outputs").mkdir(exist_ok=True)
```

---

## The Task

BixBench task bix-51: a biostatistics coding task. Given an Excel dataset of 80 hepatocellular carcinoma patients, the agent must perform logistic regression analysis to determine which demographics (age, BMI, gender) predict treatment remission.

The workspace ships with a single file, `data.xlsx`, containing patient records with columns: Sequence number, Groups, Efficacy, Age, Family history, ECOG-PS, Child-Pugh, Chemotherapy experience, HBV infection, Gender, BMI, Toxic and side effect.

Key expected results:

| Quantity | Expected Value |
|----------|---------------|
| Age significant (p < 0.05) | `True` |
| BMI significant (p < 0.05) | `False` |
| P(remission at age 65), age-only model | approximately 0.39 |
| AIC, age-only model | approximately 104 |

---

## Step 1: Set Up the Workspace

The data file lives alongside the benchmark script:

```
my_benchmark/
    workspace/
        data.xlsx          # Pre-existing patient data (shipped with the benchmark)
    run_evaluation.py      # Script below
```

No generated files are needed beforehand. The answering agent will create its own analysis scripts and outputs inside the workspace during execution.

---

## Step 2: Define the Answer Template

Each field uses a `VerifiedField` with a verification primitive that checks the agent's output against ground truth. Boolean fields use `BooleanMatch`; numeric fields use `NumericTolerance`.

```python
from karenina.schemas.entities import BaseAnswer, VerifiedField
from karenina.schemas.primitives import BooleanMatch, NumericTolerance


class Answer(BaseAnswer):
    """Template for logistic regression analysis task."""

    age_significant: bool = VerifiedField(
        description=(
            "True if age is a statistically significant predictor of remission "
            "(p < 0.05) in the logistic regression model."
        ),
        ground_truth=True,
        verify_with=BooleanMatch(),
    )
    bmi_not_significant: bool = VerifiedField(
        description=(
            "True if BMI is NOT a statistically significant predictor "
            "(p >= 0.05) in the simple logistic regression model."
        ),
        ground_truth=True,
        verify_with=BooleanMatch(),
    )
    age_predicted_probability_65: float = VerifiedField(
        description=(
            "Predicted probability of remission (PR) for a 65-year-old patient "
            "using the age-only logistic regression model. Should be approximately 0.39."
        ),
        ground_truth=0.3953,
        verify_with=NumericTolerance(tolerance=0.02, mode="absolute"),
    )
    age_model_aic: float = VerifiedField(
        description=(
            "AIC (Akaike Information Criterion) for the logistic regression model "
            "using age as the sole predictor."
        ),
        ground_truth=104.14,
        verify_with=NumericTolerance(tolerance=1.0, mode="absolute"),
    )
```

The template class name (`Answer` by convention) is discovered by type inheritance from `BaseAnswer`. See [Answer Templates](../../core_concepts/answer-templates/) for the full `VerifiedField` API and available primitives.

---

## Step 3: Create the Benchmark

```python
from pathlib import Path
from karenina.benchmark import Benchmark
from karenina.schemas.entities import Question

# Create benchmark with workspace root
benchmark = Benchmark(
    name="BixBench bix-51: Logistic Regression",
    description="Logistic regression on patient demographics for treatment remission prediction.",
    version="1.0.0",
)
benchmark.set_workspace_root(EXPERIMENT_DIR)

# Create question with workspace path
question = Question(
    question=(
        "Using the patient data in data.xlsx, perform logistic regression analysis "
        "to determine which demographics (age, BMI, gender) predict treatment "
        "remission (Groups: 1=responder, 2=non-responder). "
        "Fit individual models for each predictor and a combined model. "
        "Report: is age significant? Is BMI significant? "
        "What is the predicted probability of remission at age 65 (age-only model)? "
        "What is the AIC of the age-only model?"
    ),
    raw_answer="Age is significant (p=0.002), BMI is not (p=0.087). P(remission|age=65)=0.395, AIC=104.1",
    workspace_path="workspace",
)

question_id = benchmark.add_question(question)
benchmark.update_template(question_id, Answer)
```

The `workspace_path` on the `Question` is relative to the benchmark's `workspace_root`. The pipeline resolves the full path as `workspace_root / workspace_path`, giving the agent access to `my_benchmark/workspace/data.xlsx`.

---

## Step 4: Configure Verification

Two key configuration groups control agentic evaluation: the **answering model** (which must use a deep agent interface) and the **agentic parsing** settings (which enable the investigation judge).

```python
from karenina.schemas.verification.config import VerificationConfig
from karenina.schemas.config.models import ModelConfig, AgentMiddlewareConfig, AgentLimitConfig

config = VerificationConfig(
    answering_models=[
        ModelConfig(
            id="answering",
            model_name="claude-sonnet-4-6",
            interface="claude_agent_sdk",
            system_prompt=(
                "You are a biostatistician. Work in the provided workspace directory. "
                "Load the Excel data, perform the requested analysis using Python, "
                "and report your findings. Save your analysis scripts and result "
                "files in the workspace so they can be inspected independently."
            ),
            agent_timeout=300,
            agent_middleware=AgentMiddlewareConfig(
                limits=AgentLimitConfig(model_call_limit=50),
            ),
        ),
    ],
    parsing_models=[
        ModelConfig(
            id="parsing",
            model_name="claude-sonnet-4-6",
            interface="claude_agent_sdk",
        ),
    ],
    evaluation_mode="template_only",
    agentic_parsing=True,
    agentic_judge_context="workspace_only",
    agentic_parsing_max_turns=50,
    agentic_parsing_timeout=300,
    workspace_copy=True,
    workspace_cleanup=False,
)
```

### Answering Model

| Field | Value | Why |
|-------|-------|-----|
| `interface` | `"claude_agent_sdk"` | Deep agent tier: the SDK manages tool use internally, producing a multi-turn trace. The pipeline detects this via the adapter's `agent_tier="deep_agent"` setting and routes through `AgentPort` instead of `LLMPort`. |
| `agent_timeout` | `300` | Data analysis tasks can take several minutes. The default is 180s. |
| `system_prompt` | *(custom)* | Tells the agent to work in the workspace, use Python, and save scripts and results to files. The judge inspects these workspace artifacts; if the agent only prints to stdout without saving files, the judge has nothing to evaluate in `workspace_only` mode. |
| `agent_middleware` | `AgentMiddlewareConfig(limits=AgentLimitConfig(model_call_limit=50))` | Data analysis tasks typically need more than the default 25 turns: reading data, writing scripts, installing packages, running analysis, and saving results. |

### Agentic Parsing

| Field | Default | Value Here | Description |
|-------|---------|------------|-------------|
| `agentic_parsing` | `False` | `True` | Activates Stage 7b (`AgenticParseTemplate`) instead of the classical Stage 7a (`ParseTemplate`). |
| `agentic_judge_context` | `"workspace_only"` | `"workspace_only"` | The investigation agent receives only the question and workspace path, not the answering agent's trace. This forces independent verification. |
| `agentic_parsing_max_turns` | `15` | `50` | The investigation agent may need many turns to read files and re-run scripts left by the answering agent. |
| `agentic_parsing_timeout` | `120.0` | `300` | Seconds. Matches the answering agent timeout for consistency. |

### Workspace

| Field | Default | Value Here | Description |
|-------|---------|------------|-------------|
| `workspace_copy` | `True` | `True` | Copies `workspace/` to `workspace_run_{timestamp}/` before execution, protecting the original data for re-runs. |
| `workspace_cleanup` | `True` | `False` | Keeps the working copy after the run so you can inspect the agent's generated files. |

The parsing model must also use an interface that supports `AgentPort` (required by `agentic_parsing=True`). The `claude_agent_sdk` interface satisfies this.

---

## Step 5: Run Verification

```python
results = benchmark.run_verification(config=config, run_name="agentic_eval")
result = results.results[0]

# Inspect results
print(f"Completed: {(result.metadata.failure is None)}")
print(f"Verify: {result.template.verify_result}")
print(f"Agentic: {result.template.agentic_parsing_performed}")

if result.template.parsed_llm_response:
    for field, value in result.template.parsed_llm_response.items():
        print(f"  {field}: {value}")
```

Expected output (values may vary slightly across runs):

```
Completed: True
Verify: True
Agentic: True
  age_significant: True
  bmi_not_significant: True
  age_predicted_probability_65: 0.3948
  age_model_aic: 104.2
```

### What the Pipeline Executed

Four stages do the substantive work in this configuration:

1. **GenerateAnswer** (Stage 2): Detects that `claude_agent_sdk` has `agent_tier="deep_agent"` and uses `AgentPort`. Copies `workspace/` to `workspace_run_{timestamp}/`. The agent reads `data.xlsx`, writes Python scripts, executes them, and produces analysis output.

2. **AgenticParseTemplate** (Stage 7b): The investigation agent independently examines the workspace. In `workspace_only` mode, it receives only the question text and workspace path (not the answering agent's trace). It reads the scripts and output files the answering agent left behind, optionally re-runs them to confirm their output, and reports structured findings. The judge does not re-implement the task from scratch. A separate extraction parser then converts these findings into the `Answer` schema.

3. **VerifyTemplate** (Stage 8): Runs `BooleanMatch` and `NumericTolerance` primitives against the ground truth values declared in each `VerifiedField`.

4. **FinalizeResult** (Stage 13): Stores the investigation trace in the result. Because `workspace_cleanup=False`, the working directory is preserved.

For full details on stage ordering, context modes, and the two-step investigation/extraction architecture, see [Agentic Evaluation (Advanced)](../../advanced-pipeline/agentic-evaluation/).

---

## Step 6: Save Results

```python
import time

output_dir = EXPERIMENT_DIR / "outputs" / time.strftime("%Y%m%dT%H%M%S")
output_dir.mkdir(parents=True, exist_ok=True)

# Save full result
(output_dir / "result.json").write_text(result.model_dump_json(indent=2))

# Save benchmark checkpoint
benchmark.save(output_dir / "benchmark.jsonld")

# Save traces for review
if result.template.raw_llm_response:
    (output_dir / "answering_trace.txt").write_text(result.template.raw_llm_response)
if result.template.investigation_trace:
    (output_dir / "investigation_trace.txt").write_text(result.template.investigation_trace)
```

The `investigation_trace` is the raw output from the agentic judge's workspace investigation. Comparing it against `raw_llm_response` (the answering agent's trace) reveals whether the judge independently arrived at the same conclusions.

---

## Comparing Agentic vs Trace-Only Parsing

To evaluate how much the agentic judge adds over classical trace-only parsing, run a second verification using the same answering agent output but with `agentic_parsing=False`. The `cached_answer_data` mechanism reuses the existing answer without re-invoking the answering model.

```python
trace_only_result = _mock_trace_only_result  # In production: use run_single_model_verification

print(f"Agentic verify:    {result.template.verify_result}")
print(f"Trace-only verify: {trace_only_result.template.verify_result}")

# Compare field-level results
if result.template.parsed_llm_response and trace_only_result.template.parsed_llm_response:
    for field in result.template.parsed_llm_response:
        agentic_val = result.template.parsed_llm_response[field]
        trace_val = trace_only_result.template.parsed_llm_response.get(field)
        match = "MATCH" if agentic_val == trace_val else "DIFFER"
        print(f"  {field}: agentic={agentic_val}, trace={trace_val} [{match}]")
```

In production, use `run_single_model_verification` with `cached_answer_data` to reuse the existing answer without re-invoking the answering model:

```python
# Production code (not executed in this notebook):
# from karenina.benchmark.verification.runner import run_single_model_verification
# from karenina.benchmark.verification.utils.cache_helpers import extract_answer_data_from_result
#
# cached = extract_answer_data_from_result(result)
# trace_only_result = run_single_model_verification(
#     question_id=question_id,
#     question_text=question.question,
#     template_code=benchmark.get_template(question_id),
#     answering_model=config.answering_models[0],
#     parsing_model=config.parsing_models[0],
#     agentic_parsing=False,
#     cached_answer_data=cached,
# )
```

For coding and data analysis tasks, the agentic judge should be more reliable on verification-requiring fields because it can execute code and inspect outputs directly, rather than relying solely on the answering agent's self-reported results in the trace text.

---

## Configuration Reference

### `agentic_judge_context` Modes

| Mode | Investigation Receives | Independence | Best For |
|------|----------------------|-------------|----------|
| `"workspace_only"` | Question + workspace path | Maximum: judge cannot see answering agent's reasoning | Coding tasks where the answering agent saves scripts and results as workspace artifacts |
| `"trace_and_workspace"` | Question + answering trace + workspace path | Moderate: judge can see what the agent did, then verify | Tasks where trace context helps guide investigation |
| `"trace_only"` | Question + answering trace | Equivalent to classical Stage 7a parsing | Not recommended with `agentic_parsing=True` (a warning is logged) |

### When to Use Agentic Evaluation

Agentic evaluation adds the most value when the answering agent produces verifiable artifacts (files, code outputs, database entries) rather than just text. Typical use cases:

- **Data analysis**: Agent reads data, runs statistical analysis, produces numeric results
- **Code generation**: Agent writes and executes code; judge can re-run or inspect outputs
- **File transformation**: Agent converts or processes files; judge checks the output files
- **Multi-step research**: Agent uses tools to gather information; judge independently verifies findings

For pure text QA (no workspace, no tools), classical parsing (`agentic_parsing=False`) is sufficient and faster.

---

## Combining Agentic Parsing with Agentic Rubric Evaluation

The examples above use `evaluation_mode="template_only"`, where the pipeline checks factual correctness via the answer template. To also assess qualitative properties that require workspace investigation, add an `AgenticRubricTrait` and switch to `evaluation_mode="template_and_rubric"`.

An `AgenticRubricTrait` is like an `LLMRubricTrait`, but instead of a single LLM call operating on the response text, it launches a separate agent session with tool access. The agent investigates the workspace (reading files, running scripts) before producing a score. This is Stage 11b (`AgenticRubricEvaluation`) in the pipeline.

### Define the Agentic Rubric Trait

This trait detects which Python library the answering agent used for logistic regression. A classical `LLMRubricTrait` could only inspect the response text, which may not mention the library explicitly. The `AgenticRubricTrait` examines the actual source files in the workspace.

```python
from karenina.schemas.entities.rubric import AgenticRubricTrait, Rubric

library_detection = AgenticRubricTrait(
    name="logistic_regression_library",
    description=(
        "Examine the Python scripts in the workspace to determine which library "
        "was used to implement logistic regression. Look at the actual import "
        "statements and function calls in the code files, not just what the "
        "agent's response claims."
    ),
    kind="literal",
    classes={
        "statsmodels": "Uses statsmodels (sm.Logit, smf.logit, GLM with Binomial)",
        "scikit-learn": "Uses scikit-learn (LogisticRegression from sklearn)",
        "other": "Uses a different library or manual implementation",
    },
    higher_is_better=False,
    context_mode="workspace_only",
    max_turns=15,
    timeout_seconds=120,
)
```

Key fields on `AgenticRubricTrait`:

| Field | Value | Purpose |
|-------|-------|---------|
| `kind` | `"literal"` | Classifies into ordered categories; returns the class index (0, 1, or 2) |
| `classes` | `dict` | Ordered mapping of class names to descriptions. Dict order determines indices. |
| `higher_is_better` | `False` | For this trait, the class index is a label, not a quality ranking |
| `context_mode` | `"workspace_only"` | The agent sees the question and workspace path, not the answering trace |
| `max_turns` | `15` | Maximum agent turns for investigation |
| `timeout_seconds` | `120` | Wall-clock timeout for the investigation |

### Attach the Rubric and Configure

```python
benchmark.set_global_rubric(Rubric(agentic_traits=[library_detection]))

rubric_config = VerificationConfig(
    answering_models=[
        ModelConfig(
            id="answering",
            model_name="claude-sonnet-4-6",
            interface="claude_agent_sdk",
            system_prompt=(
                "You are a biostatistician. Work in the provided workspace directory. "
                "Load the Excel data, perform the requested analysis using Python, "
                "and report your findings. Save your analysis scripts and result "
                "files in the workspace so they can be inspected independently."
            ),
            agent_timeout=300,
            agent_middleware=AgentMiddlewareConfig(
                limits=AgentLimitConfig(model_call_limit=50),
            ),
        ),
    ],
    parsing_models=[
        ModelConfig(
            id="parsing",
            model_name="claude-sonnet-4-6",
            interface="claude_agent_sdk",
        ),
    ],
    evaluation_mode="template_and_rubric",
    agentic_rubric_strategy="individual",
    agentic_parsing=True,
    agentic_judge_context="workspace_only",
    agentic_parsing_max_turns=50,
    agentic_parsing_timeout=300,
    workspace_copy=True,
    workspace_cleanup=False,
)
```

Three settings control agentic rubric evaluation:

| Field | Value | Description |
|-------|-------|-------------|
| `evaluation_mode` | `"template_and_rubric"` | Runs both template verification (Stages 7/8) and rubric evaluation (Stage 11) |
| `agentic_rubric_strategy` | `"individual"` | Each agentic trait gets its own agent session. The alternative, `"shared"`, evaluates all agentic traits in a single session. |
| `agentic_rubric_parallel` | `False` (default) | When `True` and strategy is `"individual"`, agentic trait sessions run concurrently. Ignored under the `"shared"` strategy. |

### Run Verification and Inspect Rubric Results

```python
rubric_results = benchmark.run_verification(config=rubric_config, run_name="agentic_rubric_eval")
rubric_result = rubric_results.results[0]

# Template results (same as before)
print(f"Template verify: {rubric_result.template.verify_result}")
print(f"Agentic parsing: {rubric_result.template.agentic_parsing_performed}")

# Agentic rubric results
print(f"\nAgentic trait scores: {rubric_result.rubric.agentic_trait_scores}")
```

The `agentic_trait_scores` dictionary maps trait names to their scores. For a `literal` trait, the score is the class index: `0` for `"statsmodels"`, `1` for `"scikit-learn"`, `2` for `"other"`.

### Inspect the Investigation Trace

Each agentic trait produces an investigation trace that records the agent's reasoning and findings:

```python
if rubric_result.rubric.agentic_trait_investigation_traces:
    for name, trace in rubric_result.rubric.agentic_trait_investigation_traces.items():
        print(f"Trait: {name}")
        print(f"Trace ({len(trace)} chars):")
        print(trace[:300])
```

The investigation trace is useful for debugging and auditing. It shows what files the agent examined, what patterns it found, and how it reached its classification. Unlike `LLMRubricTrait` (which receives only the response text), the agentic trait's trace reflects direct inspection of workspace artifacts.

### Pipeline Stages for Combined Evaluation

When `evaluation_mode="template_and_rubric"` with both `agentic_parsing=True` and agentic rubric traits, the pipeline executes:

| Stage | Name | What It Does |
|-------|------|-------------|
| 2 | GenerateAnswer | Answering agent runs in the workspace |
| 7b | AgenticParseTemplate | Investigation agent examines workspace, parser extracts into template schema |
| 8 | VerifyTemplate | Runs verification primitives against ground truth |
| 11b | AgenticRubricEvaluation | Separate agent session(s) investigate the workspace for each agentic trait |
| 13 | FinalizeResult | Stores all results, including agentic trait scores and traces |

Stage 7b and Stage 11b each launch independent agent sessions. The Stage 7b agent fills in the template schema; the Stage 11b agent(s) produce rubric scores. They do not share context or state with each other.

For details on agentic rubric internals (investigation prompts, extraction, error handling), see [Agentic Rubric Evaluation (Advanced)](../../advanced-pipeline/agentic-rubric-evaluation/). For the `AgenticRubricTrait` API, see [Agentic Traits (Concepts)](../../core_concepts/rubrics/agentic-traits/).

---

## Related Pages

- [Agentic Evaluation (Concepts)](../../core_concepts/agentic-evaluation/): Conceptual overview, context modes, independence guarantees
- [Agentic Evaluation (Advanced)](../../advanced-pipeline/agentic-evaluation/): Pipeline internals, stage architecture, investigation prompts
- [Agentic Traits (Concepts)](../../core_concepts/rubrics/agentic-traits/): `AgenticRubricTrait` API, kinds, context modes
- [Agentic Rubric Evaluation (Advanced)](../../advanced-pipeline/agentic-rubric-evaluation/): Stage 11b internals, investigation prompts, extraction
- [MCP Agent Evaluation](../mcp-agent-evaluation/): Evaluating tool-using agents with MCP servers
- [Basic Verification](../basic-verification/): Simplest verification path (template-only, no agents)
- [Answer Templates](../../core_concepts/answer-templates/): `VerifiedField` API and verification primitives
- [VerificationConfig Reference](../../reference/configuration/verification-config.md): All configuration fields
