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

For the conceptual background on how agentic evaluation works, see [Agentic Evaluation (Concepts)](../../core_concepts/agentic-evaluation.md). For pipeline internals (stage ordering, context modes, investigation prompts), see [Agentic Evaluation (Advanced)](../../advanced-pipeline/agentic-evaluation.md).

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
from karenina.schemas.entities.primitives import BooleanMatch, NumericTolerance

TEMPLATE_CODE = '''
class LogisticRegressionAnswer(BaseAnswer):
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
'''
```

The template class name (`LogisticRegressionAnswer`) is arbitrary; the pipeline discovers the class by type inheritance from `BaseAnswer`. See [Answer Templates](../../core_concepts/answer-templates.md) for the full `VerifiedField` API and available primitives.

---

## Step 3: Create the Benchmark

```python
from pathlib import Path
from karenina.benchmark import Benchmark
from karenina.schemas.entities import Question

EXPERIMENT_DIR = Path(__file__).parent

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

question_id = benchmark.add_question(question, answer_template=TEMPLATE_CODE)
```

The `workspace_path` on the `Question` is relative to the benchmark's `workspace_root`. The pipeline resolves the full path as `workspace_root / workspace_path`, giving the agent access to `my_benchmark/workspace/data.xlsx`.

---

## Step 4: Configure Verification

Two key configuration groups control agentic evaluation: the **answering model** (which must use a natively agentic interface) and the **agentic parsing** settings (which enable the investigation judge).

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
| `interface` | `"claude_agent_sdk"` | Natively agentic: the SDK manages tool use internally, producing a multi-turn trace. The pipeline detects this via the adapter's `natively_agentic=True` flag and routes through `AgentPort` instead of `LLMPort`. |
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
print(f"Completed: {result.metadata.completed_without_errors}")
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

1. **GenerateAnswer** (Stage 2): Detects that `claude_agent_sdk` is `natively_agentic` and uses `AgentPort`. Copies `workspace/` to `workspace_run_{timestamp}/`. The agent reads `data.xlsx`, writes Python scripts, executes them, and produces analysis output.

2. **AgenticParseTemplate** (Stage 7b): The investigation agent independently examines the workspace. In `workspace_only` mode, it receives only the question text and workspace path (not the answering agent's trace). It reads the scripts and output files the answering agent left behind, optionally re-runs them to confirm their output, and reports structured findings. The judge does not re-implement the task from scratch. A separate extraction parser then converts these findings into the `LogisticRegressionAnswer` schema.

3. **VerifyTemplate** (Stage 8): Runs `BooleanMatch` and `NumericTolerance` primitives against the ground truth values declared in each `VerifiedField`.

4. **FinalizeResult** (Stage 13): Stores the investigation trace in the result. Because `workspace_cleanup=False`, the working directory is preserved.

For full details on stage ordering, context modes, and the two-step investigation/extraction architecture, see [Agentic Evaluation (Advanced)](../../advanced-pipeline/agentic-evaluation.md).

---

## Step 6: Save Results

```python
import json
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
from karenina.benchmark.verification.runner import run_single_model_verification
from karenina.benchmark.verification.utils.cache_helpers import extract_answer_data_from_result

cached = extract_answer_data_from_result(result)

trace_only_result = run_single_model_verification(
    question_id=question_id,
    question_text=question.question,
    template_code=TEMPLATE_CODE,
    answering_model=config.answering_models[0],
    parsing_model=config.parsing_models[0],
    agentic_parsing=False,
    cached_answer_data=cached,
)

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

## Related Pages

- [Agentic Evaluation (Concepts)](../../core_concepts/agentic-evaluation.md): Conceptual overview, context modes, independence guarantees
- [Agentic Evaluation (Advanced)](../../advanced-pipeline/agentic-evaluation.md): Pipeline internals, stage architecture, investigation prompts
- [MCP Agent Evaluation](mcp-agent-evaluation.ipynb): Evaluating tool-using agents with MCP servers
- [Basic Verification](basic-verification.ipynb): Simplest verification path (template-only, no agents)
- [Answer Templates](../../core_concepts/answer-templates.md): `VerifiedField` API and verification primitives
- [VerificationConfig Reference](../../reference/configuration/verification-config.md): All configuration fields
