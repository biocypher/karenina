# TaskEval: Task-Centric Trace Evaluation

Evaluate pre-logged agent workflow outputs by attaching verification criteria to existing traces. TaskEval inverts the typical benchmark workflow: instead of starting with questions and generating answers, you start with agent execution traces and attach verification criteria (templates and rubrics) to evaluate them.

---

## Quick Navigation

- [What is TaskEval?](#what-is-taskeval)
- [Task-Centric vs Question-Centric](#task-centric-vs-question-centric)
- [When to Use TaskEval](#when-to-use-taskeval)
- [Quick Start](#quick-start)
- [Attaching Verification Criteria](#attaching-verification-criteria)
  - [Templates for Correctness](#templates-for-correctness)
  - [Rubrics for Quality](#rubrics-for-quality)
- [Dict Trace Logging](#dict-trace-logging)
- [Evaluation Modes](#evaluation-modes)
- [Step-Specific Evaluation](#step-specific-evaluation)
- [Results and Display](#results-and-display)
- [Replicate Aggregation](#replicate-aggregation)
- [Best Practices](#best-practices)
- [Complete Examples](#complete-examples)

---

## What is TaskEval?

**TaskEval** is a trace-centric evaluation framework that inverts the traditional benchmarking workflow. Instead of defining questions first and generating answers, TaskEval lets you start with **existing agent execution traces** and attach **verification criteria** (both templates and rubrics) to evaluate them.

### The Paradigm Shift

**Traditional Benchmark (Question-Centric)**:
```
Questions → Templates/Rubrics → Generate Answers → Verify
```

**TaskEval (Task/Trace-Centric)**:
```
Agent Traces → Attach Verification Criteria → Evaluate
```

### Why This Matters

- **Agent-first workflow**: Design for evaluating agent outputs, not just Q&A
- **No re-execution**: Evaluate expensive agent workflows without re-running them
- **Full verification**: Use both templates (correctness) AND rubrics (quality)
- **Multi-step support**: Evaluate complex workflows with multiple phases
- **Cost-effective**: Skip generation costs, only pay for verification

---

## Task-Centric vs Question-Centric

Understanding the conceptual difference between TaskEval and Benchmark is key to using them effectively.

### Benchmark (Question-Centric)

**Workflow**: Question → Answer → Verify

```python
# Start with questions
benchmark.add_question(
    question="What is the capital of France?",
    raw_answer="Paris"
)

# Attach verification criteria TO THE QUESTION
benchmark.generate_template(question_id="q1")
benchmark.add_rubric(Rubric(...))

# Generate answer and verify
results = benchmark.run_verification(config)
```

**Characteristics**:

- Questions drive the workflow
- LLM generates new answers
- Verification criteria attached to questions
- Best for: Benchmarking LLM capabilities, testing knowledge

### TaskEval (Task-Centric)

**Workflow**: Execute Task → Collect Traces → Attach Verification → Evaluate

```python
# Start with agent execution traces
task = TaskEval(task_id="my_agent_run")
task.log("Agent reasoning: Analyzed problem...")
task.log("Agent action: Implemented solution...")
task.log("Agent result: Success")

# Attach verification criteria TO THE TRACES
task.add_question({...})  # Optional: for template verification
task.add_rubric(Rubric(...))  # For quality assessment

# Evaluate existing traces
result = task.evaluate(config)
```

**Characteristics**:

- Agent traces drive the workflow
- No LLM generation (traces already exist)
- Verification criteria attached to traces/tasks
- Best for: Evaluating agent workflows, analyzing production outputs

### When to Use Each

| Use Case | Use Benchmark | Use TaskEval |
|----------|--------------|-------------|
| Test LLM knowledge | ✅ | ❌ |
| Evaluate agent workflows | ❌ | ✅ |
| Generate new answers | ✅ | ❌ |
| Analyze existing traces | ❌ | ✅ |
| Question-driven evaluation | ✅ | ❌ |
| Task-driven evaluation | ❌ | ✅ |
| Re-running is expensive | ❌ | ✅ |
| Need fresh responses | ✅ | ❌ |

---

## What Can TaskEval Verify?

TaskEval supports the **full verification pipeline** from standard benchmarking:

### ✅ Templates (Correctness Verification)

Just like Benchmark, you can attach answer templates to verify factual correctness:

```python
task.add_question({
    "id": "action_check",
    "question": "Did the agent take the correct action?",
    "raw_answer": "solution_a",
    "answer_template": template_code  # Pydantic template
})
```

### ✅ Rubrics (Quality Assessment)

Evaluate qualitative aspects using all three rubric types:

```python
# LLM-based traits
RubricTrait(name="clarity", description="...", kind="score", ...)

# Regex-based traits
RegexTrait(name="has_code", pattern=r"```", ...)

# Metric-based traits
MetricRubricTrait(name="coverage", tp_instructions=[...], ...)
```

### ✅ Combined Verification

Use templates AND rubrics together for comprehensive evaluation:

```python
# Attach both
task.add_question({...})  # Template verification
task.add_rubric(Rubric(...))  # Quality rubrics

# Evaluate - gets both verify_result (template) and verify_rubric (quality)
result = task.evaluate(config)
```

### Why TaskEval Is Not "Rubric-Only"

While TaskEval *can* operate in rubric-only mode (convenient for pure quality assessment), it's **not limited to rubrics**. The core innovation is the **task-centric workflow** that lets you start from traces and attach any verification criteria you need.

---

## When to Use TaskEval

### Ideal Use Cases

**✅ Agent Workflow Evaluation**:

- Multi-step agent executions (planning → execution → reflection)
- Complex reasoning chains with intermediate outputs
- Agentic systems where re-execution is expensive or non-deterministic

**✅ Post-Hoc Analysis**:

- Evaluate production agent logs collected over time
- Debug agent failures by attaching verification criteria
- Compare agent versions using historical traces

**✅ Trace-First Evaluation**:

- You already have agent outputs and want to evaluate them
- Verification criteria come after data collection
- Task execution is separate from evaluation

### When to Use Standard Benchmarking Instead

**❌ Testing LLM Knowledge**: Use Benchmark to test raw model capabilities

**❌ Generating New Responses**: Use Benchmark when you need fresh LLM generations

**❌ Question-Driven Evaluation**: Use Benchmark when questions drive the workflow

**❌ Simple Q&A Testing**: Use Benchmark for straightforward question-answer pairs

---

## Quick Start

Here's a minimal example showing the task-centric workflow with both templates and rubrics:

```python
from karenina.benchmark.task_eval import TaskEval
from karenina.schemas.domain import Rubric, RubricTrait, RegexTrait
from karenina.schemas.workflow import ModelConfig, VerificationConfig

# 1. Create TaskEval instance (task-centric)
task = TaskEval(task_id="agent_code_generation")

# 2. Log agent execution traces
task.log("Reasoning: Need to implement a REST API endpoint")
task.log("Plan: Use FastAPI with proper error handling")
task.log("Implementation: Created /api/users endpoint with validation")
task.log("Testing: Verified with 5 test cases, all passed")

# 3. Attach verification criteria to traces
# 3a. Template for correctness verification
answer_template = '''
from karenina.schemas.domain import BaseAnswer
from pydantic import Field

class Answer(BaseAnswer):
    endpoint_created: bool = Field(description="Was endpoint created?")
    has_error_handling: bool = Field(description="Has error handling?")

    def model_post_init(self, __context):
        self.correct = {"endpoint": True, "errors": True}

    def verify(self) -> bool:
        return self.endpoint_created and self.has_error_handling
'''

task.add_question({
    "id": "implementation_check",
    "question": "Did the agent implement the endpoint correctly?",
    "raw_answer": "endpoint with error handling",
    "answer_template": answer_template
})

# 3b. Rubrics for quality assessment
task.add_rubric(Rubric(
    traits=[
        RubricTrait(
            name="clarity",
            description="How clear is the implementation plan?",
            kind="score",
            min_score=1,
            max_score=5
        )
    ],
    manual_traits=[
        RegexTrait(
            name="mentions_testing",
            description="Agent mentioned testing",
            pattern=r"(?i)(test|testing|verified)"
        )
    ]
))

# 4. Configure evaluation
config = VerificationConfig(
    parsing_models=[
        ModelConfig(
            id="parser",
            model_provider="openai",
            model_name="gpt-4o-mini"
        )
    ],
    parsing_only=True
)

# 5. Evaluate traces against verification criteria
result = task.evaluate(config)

# 6. Display results
print(result.display())
# Shows:
# - Template verification result (correct/incorrect)
# - Rubric scores (clarity score, testing check)
```

**That's it!** TaskEval automatically:

- Concatenates logged traces for evaluation
- Runs template verification (factual correctness)
- Evaluates rubrics (quality assessment)
- Returns comprehensive verification results

---

## Attaching Verification Criteria

The key innovation of TaskEval is that you attach verification criteria **to traces** rather than to questions. This section shows how to attach both templates and rubrics.

### Templates for Correctness

Attach answer templates to verify factual correctness, just like in Benchmark:

```python
# Define template
answer_template = '''
from karenina.schemas.domain import BaseAnswer
from pydantic import Field

class Answer(BaseAnswer):
    action_taken: str = Field(description="What action did the agent take?")
    result: str = Field(description="What was the result?")

    def model_post_init(self, __context):
        self.correct = {"action": "implement_api", "result": "success"}

    def verify(self) -> bool:
        return (self.action_taken.lower() == self.correct["action"] and
                self.result.lower() == self.correct["result"])
'''

# Attach to traces
task.add_question({
    "id": "correctness_check",
    "question": "Did the agent complete the task correctly?",
    "raw_answer": "implement_api success",
    "answer_template": answer_template
})
```

**Result**: `verify_result` (bool) indicating if traces match expected correctness

### Rubrics for Quality

Attach rubrics to assess qualitative aspects. All three rubric types are supported:

#### LLM-Based Rubrics

```python
from karenina.schemas.domain import RubricTrait

task.add_rubric(Rubric(
    traits=[
        RubricTrait(
            name="reasoning_quality",
            description="How clear and logical is the agent's reasoning?",
            kind="score",
            min_score=1,
            max_score=5
        ),
        RubricTrait(
            name="is_thorough",
            description="Did the agent thoroughly address all aspects?",
            kind="boolean"
        )
    ]
))
```

**Result**: `verify_rubric` dict with scores/booleans for each trait

#### Regex-Based Rubrics

```python
from karenina.schemas.domain import RegexTrait

task.add_rubric(Rubric(
    manual_traits=[
        RegexTrait(
            name="has_code_block",
            description="Contains code implementation",
            pattern=r"```[\s\S]*?```"
        ),
        RegexTrait(
            name="no_errors",
            description="No error mentions",
            pattern=r"(?i)(error|exception|failed)",
            invert_result=True
        )
    ]
))
```

**Result**: Boolean pass/fail for each pattern

#### Metric-Based Rubrics

```python
from karenina.schemas.domain import MetricRubricTrait

task.add_rubric(Rubric(
    metric_traits=[
        MetricRubricTrait(
            name="requirement_coverage",
            description="Coverage of required steps",
            metrics=["precision", "recall", "f1"],
            tp_instructions=[
                "Analyze requirements",
                "Design solution",
                "Implement code",
                "Write tests",
                "Deploy solution"
            ]
        )
    ]
))
```

**Result**: Confusion matrix and computed metrics (precision, recall, F1)

### Combining Templates and Rubrics

Use both for comprehensive evaluation:

```python
# Template: Verify correctness
task.add_question({
    "id": "correctness",
    "question": "Is the implementation correct?",
    "raw_answer": "correct implementation",
    "answer_template": template_code
})

# Rubrics: Assess quality
task.add_rubric(Rubric(
    traits=[RubricTrait(...)],        # LLM evaluation
    manual_traits=[RegexTrait(...)],  # Regex checks
    metric_traits=[MetricRubricTrait(...)]   # Metrics
))

# Get both results
result = task.evaluate(config)
vr = result.global_eval.verification_results["correctness"][0]
print(f"Correct: {vr.verify_result}")  # Template verification
print(f"Quality: {vr.verify_rubric}")  # Rubric scores
```

---

## Dict Trace Logging

**Dict traces** are the recommended way to log structured agent outputs. Each key becomes a separate evaluation point.

### Basic Dict Logging

```python
task = TaskEval(task_id="structured_eval")

# Log a dict trace
task.log({
    "analysis": "Examined requirements and constraints",
    "plan": "Generated 3-step implementation plan",
    "execution": "Implemented all steps successfully",
    "testing": "Verified with 5 test cases"
})
```

### What Happens During Evaluation

In rubric-only mode, each dict key becomes a **synthetic question**:

```python
# Logged dict:
{"reasoning": "Analyzed carefully", "action": "Implemented solution"}

# Becomes these synthetic questions:
# - "dict_key_reasoning" → evaluates "Analyzed carefully"
# - "dict_key_action" → evaluates "Implemented solution"
```

### Benefits of Dict Traces

✅ **Structured evaluation**: Each aspect evaluated separately
✅ **Granular insights**: Per-key rubric scores
✅ **No template writing**: Rubric-only mode handles everything
✅ **Clear organization**: Dict keys define evaluation dimensions

### String Logs (Alternative)

You can also log plain strings:

```python
# String logs concatenated together
task.log("Step 1: Analyzed problem")
task.log("Step 2: Generated solution")
task.log("Step 3: Validated output")

# Evaluates concatenated: "Step 1: Analyzed problem\n\nStep 2: Generated solution\n\nStep 3: Validated output"
```

**Recommendation**: Use dict traces for better structure and per-key evaluation.

---


## Evaluation Modes

TaskEval automatically detects the evaluation mode based on what verification criteria you attach to traces.

### Automatic Mode Detection

```python
# Mode is determined by what you attach:
# - Has templates only → template_only
# - Has rubrics only → rubric_only
# - Has both → template_and_rubric
# - Has neither → ValueError
```

### Mode 1: Template-Only

**When**: Only questions with templates are attached

**Use for**: Verifying factual correctness without quality assessment

**Example**:
```python
task = TaskEval(task_id="correctness_only")

# Log traces
task.log("Agent implemented the API endpoint successfully")

# Attach template (no rubrics)
task.add_question({
    "id": "check",
    "question": "Did implementation succeed?",
    "raw_answer": "success",
    "answer_template": template_code
})

# Evaluate - template_only mode
result = task.evaluate(config)

# Get verification result
vr = result.global_eval.verification_results["check"][0]
print(f"Correct: {vr.verify_result}")  # True/False
```

### Mode 2: Rubric-Only

**When**: Only rubrics are attached, no templates

**Use for**: Quality assessment without correctness verification

**Example**:
```python
task = TaskEval(task_id="quality_only")

# Log dict traces (auto-generates synthetic questions)
task.log({
    "reasoning": "Analyzed problem carefully",
    "implementation": "Built robust solution"
})

# Attach rubrics (no templates)
task.add_rubric(Rubric(
    traits=[
        RubricTrait(name="clarity", description="...", kind="score", min_score=1, max_score=5)
    ]
))

# Evaluate - rubric_only mode
result = task.evaluate(config)

# Get quality scores
for question_id, vr_list in result.global_eval.verification_results.items():
    vr = vr_list[0]
    print(f"{question_id}: {vr.verify_rubric}")
```

**Note**: In rubric_only mode with dict traces, TaskEval creates synthetic questions for each dict key (e.g., "dict_key_reasoning", "dict_key_implementation").

### Mode 3: Template and Rubric

**When**: Both templates and rubrics are attached

**Use for**: Comprehensive evaluation (correctness + quality)

**Example**:
```python
task = TaskEval(task_id="comprehensive")

# Log traces
task.log("Successfully implemented REST API with proper error handling")

# Attach template
task.add_question({
    "id": "correctness",
    "question": "Implementation correct?",
    "raw_answer": "REST API",
    "answer_template": template_code
})

# Attach rubrics
task.add_rubric(Rubric(
    traits=[RubricTrait(name="clarity", ...)],
    manual_traits=[RegexTrait(name="has_errors", ...)]
))

# Evaluate - template_and_rubric mode
result = task.evaluate(config)

# Get both results
vr = result.global_eval.verification_results["correctness"][0]
print(f"Correct: {vr.verify_result}")  # Template verification
print(f"Quality: {vr.verify_rubric}")  # Rubric scores
```

### Choosing the Right Mode

| Mode | Use When | Gets You |
|------|----------|----------|
| **template_only** | Need to verify correctness only | `verify_result` (bool) |
| **rubric_only** | Need quality assessment only | `verify_rubric` (dict) |
| **template_and_rubric** | Need both correctness and quality | Both `verify_result` and `verify_rubric` |

---

## Step-Specific Evaluation

Evaluate different workflow phases with phase-specific rubrics.

### Global vs Step-Specific

**Global evaluation**: Evaluates all logs across the entire workflow

**Step-specific evaluation**: Evaluates logs from specific workflow steps

### Setting Up Steps

```python
task = TaskEval(task_id="multi_step_agent")

# Add step-specific rubrics
planning_rubric = Rubric(
    manual_traits=[
        RegexTrait(
            name="has_plan",
            description="Contains planning keywords",
            pattern=r"(?i)(plan|strategy|approach)"
        )
    ]
)
task.add_rubric(planning_rubric, step_id="planning")

execution_rubric = Rubric(
    manual_traits=[
        RegexTrait(
            name="has_implementation",
            description="Contains implementation details",
            pattern=r"(?i)(implement|execute|build)"
        )
    ]
)
task.add_rubric(execution_rubric, step_id="execution")

# Log to specific steps
task.log(
    {"analysis": "Analyzed problem", "plan": "Created plan"},
    step_id="planning"
)

task.log(
    {"code": "Implemented solution", "tests": "Verified output"},
    step_id="execution"
)
```

### Evaluating Steps

```python
# Option 1: Evaluate globally (automatically evaluates all steps)
result = task.evaluate(config)

print(f"Global results: {len(result.global_eval.verification_results)} questions")
print(f"Step evaluations: {list(result.per_step.keys())}")

for step_id, step_result in result.per_step.items():
    print(f"\nStep '{step_id}':")
    print(f"  Questions: {len(step_result.verification_results)}")
    for question_id, vr_list in step_result.verification_results.items():
        print(f"  {question_id}: {vr_list[0].verify_rubric}")

# Option 2: Evaluate specific step only
planning_result = task.evaluate(config, step_id="planning")
```

### Use Cases for Step-Specific Evaluation

✅ **Phase-specific quality**: Different quality criteria for planning vs execution
✅ **Failure localization**: Identify which step caused issues
✅ **Granular analysis**: Understand per-phase performance
✅ **Progressive evaluation**: Evaluate as workflow progresses

---

## Results and Display

TaskEval provides multiple ways to access and display evaluation results.

### Display Methods

```python
result = task.evaluate(config)

# 1. Full formatted display
print(result.display())
# Output:
# ════════════════════════════════════════════════════════════════════════════════
#                         TASK EVALUATION RESULTS
# ════════════════════════════════════════════════════════════════════════════════
# Task ID: agent_eval_001
# Timestamp: 2025-11-11 14:30:00
#
# ────────────────────────────────────────────────────────────
# GLOBAL EVALUATION
# ────────────────────────────────────────────────────────────
# Verification Results:
#   Question: dict_key_reasoning
#     Status: ✓ PASSED
#     Rubric: clarity=4, completeness=✓
#     Metric [step_coverage]: precision=0.85, recall=0.92, f1=0.88

# 2. Compact summary
print(result.summary())
# Output: "3/3 questions passed | 5/5 rubric traits passed"

# 3. One-line summary
print(result.summary_compact())
# Output: "TaskEval [agent_eval_001]: 3/3 questions, 5/5 traits (100%)"
```

### Accessing Verification Results

```python
# Get results for a specific question
for question_id, vr_list in result.global_eval.verification_results.items():
    # vr_list contains all replicates (usually just [0] if replicate_count=1)
    vr = vr_list[0]

    # Basic verification
    print(f"Question: {question_id}")
    print(f"Passed: {vr.verify_result}")
    print(f"Completed: {vr.completed_without_errors}")

    # Rubric scores (LLM + Manual traits)
    if vr.verify_rubric:
        print(f"Rubric scores: {vr.verify_rubric}")
        # Example: {"clarity": 4, "has_code": True, "completeness": 5}

    # Metric trait results
    if vr.metric_trait_metrics:
        for trait_name, metrics in vr.metric_trait_metrics.items():
            print(f"Metrics [{trait_name}]: {metrics}")
            # Example: {"precision": 0.85, "recall": 0.92, "f1": 0.88}

    if vr.metric_trait_confusion_lists:
        for trait_name, confusion in vr.metric_trait_confusion_lists.items():
            print(f"Confusion [{trait_name}]:")
            print(f"  TP: {confusion['tp']}")
            print(f"  FP: {confusion['fp']}")
            print(f"  FN: {confusion['fn']}")
```

### Unified Rubric Results Interface

Access all rubric results in a structured format:

```python
vr = vr_list[0]

# Get organized rubric results
rubric_data = vr.rubric_results
# Structure:
# {
#     "llm": {"clarity": 4, "completeness": 5},
#     "manual": {"has_code": True, "mentions_testing": True},
#     "metric": {
#         "step_coverage": {
#             "metrics": {"precision": 0.85, "recall": 0.92},
#             "confusion": {"tp": [...], "fp": [...], "fn": [...], "tn": [...]}
#         }
#     }
# }

# Look up specific trait by name
trait_value, trait_type = vr.get_trait_by_name("clarity")
# Returns: (4, "llm")

# Get all scores flattened (useful for export)
all_scores = vr.get_all_scores()
# Returns: {"clarity": 4, "completeness": 5, "has_code": True, ...}
```

### Export Formats

```python
# JSON export
json_str = result.export_json(include_logs=True, indent=2)
with open("evaluation_results.json", "w") as f:
    f.write(json_str)

# Markdown export
md_str = result.export_markdown()
with open("evaluation_results.md", "w") as f:
    f.write(md_str)

# Simplified dict
simple_dict = result.to_dict_clean()
```

---

## Replicate Aggregation

Run multiple evaluation replicates for statistical analysis.

### Running Multiple Replicates

```python
config = VerificationConfig(
    parsing_models=[model_config],
    replicate_count=3  # Run each evaluation 3 times
)

result = task.evaluate(config)

# Access individual replicates
for question_id, vr_list in result.global_eval.verification_results.items():
    print(f"Question {question_id}: {len(vr_list)} replicates")
    for i, vr in enumerate(vr_list):
        print(f"  Replicate {i+1}: {vr.verify_rubric}")
```

### Aggregating Results

```python
# Aggregate rubric scores across replicates
aggregated = result.global_eval.aggregate_rubric_results()

# Structure:
# {
#     "question_id": {
#         "llm": {"clarity": 4.33, "completeness": 4.67},  # Averaged
#         "manual": {"has_code": 0.67},  # Pass rate (2/3 = 0.67)
#         "metric": {
#             "step_coverage": {
#                 "metrics": {"precision": 0.85, "recall": 0.92}  # Averaged
#             }
#         },
#         "failed_replicate_count": 0  # Number of failed replicates
#     }
# }
```

### Aggregation Behavior

**LLM Traits (Scores)**: Averaged across successful replicates
```python
# Replicate 1: clarity=4, Replicate 2: clarity=5, Replicate 3: clarity=4
# Aggregated: {"clarity": 4.33}
```

**Manual Traits (Booleans)**: Converted to pass rate (0.0 to 1.0)
```python
# Replicate 1: has_code=True, Replicate 2: has_code=True, Replicate 3: has_code=False
# Aggregated: {"has_code": 0.67}  # 2/3 passed
```

**Metric Traits**: Metrics averaged, confusion matrices omitted
```python
# Replicate 1: precision=0.8, recall=0.9
# Replicate 2: precision=0.9, recall=0.95
# Aggregated: {"metrics": {"precision": 0.85, "recall": 0.925}}
```

**Failed Replicates**: Excluded from aggregation, count tracked
```python
{
    "llm": {"clarity": 4.5},  # Only successful replicates
    "failed_replicate_count": 1
}
```

### Use Cases for Replicates

✅ **Statistical reliability**: Average scores across multiple runs
✅ **Variance detection**: Identify inconsistent rubric evaluations
✅ **Pass rate analysis**: Measure reliability of boolean checks
✅ **Quality confidence**: Higher replicate agreement = higher confidence

---

## Best Practices

### 1. Use Dict Traces for Structure

```python
# ✅ Good: Structured dict traces
task.log({
    "analysis": "Examined requirements",
    "implementation": "Built solution with error handling",
    "testing": "Verified with 10 test cases"
})

# ❌ Less ideal: Unstructured string
task.log("I looked at the requirements, built something, and tested it")
```

### 2. Choose Appropriate Rubric Types

**LLM Traits**: Qualitative assessment (clarity, completeness)
**Manual Traits**: Deterministic checks (has keywords, no errors)
**Metric Traits**: Quantitative coverage (requirement completion)

```python
# ✅ Good: Mix of all three for comprehensive evaluation
rubric = Rubric(
    traits=[RubricTrait(name="clarity", ...)],  # Subjective quality
    manual_traits=[RegexTrait(name="has_code", ...)],  # Objective check
    metric_traits=[MetricRubricTrait(name="coverage", ...)]  # Quantitative
)
```

### 3. Use Step IDs for Complex Workflows

```python
# ✅ Good: Track workflow phases
task.log({"plan": "..."}, step_id="planning")
task.log({"code": "..."}, step_id="execution")
task.log({"analysis": "..."}, step_id="reflection")

# Evaluate each phase separately
for step in ["planning", "execution", "reflection"]:
    result = task.evaluate(config, step_id=step)
    print(f"{step}: {result.summary()}")
```

### 4. Monitor Token Usage

```python
result = task.evaluate(config)

for question_id, vr_list in result.global_eval.verification_results.items():
    vr = vr_list[0]
    if vr.usage_metadata:
        tokens = vr.usage_metadata.get("total", {}).get("total_tokens", 0)
        print(f"{question_id}: {tokens} tokens")
```

### 5. Use Replicates for Critical Evaluations

```python
# For important evaluations, use multiple replicates
config = VerificationConfig(
    parsing_models=[model_config],
    replicate_count=3  # Run 3 times for reliability
)

result = task.evaluate(config)
aggregated = result.global_eval.aggregate_rubric_results()

# Check for consistency
for question_id, scores in aggregated.items():
    if scores.get("failed_replicate_count", 0) > 0:
        print(f"⚠️  {question_id}: Some replicates failed")
```

---

## Complete Examples

### Example 1: Agent Workflow Quality Assessment

```python
from karenina.benchmark.task_eval import TaskEval
from karenina.schemas.domain import Rubric, RubricTrait, RegexTrait, MetricRubricTrait
from karenina.schemas.workflow import ModelConfig, VerificationConfig

# Create TaskEval for a code generation agent
task = TaskEval(
    task_id="codegen_agent_v1.2",
    metadata={
        "agent_version": "1.2.0",
        "task_type": "code_generation",
        "timestamp": "2025-11-11T14:00:00"
    }
)

# Log agent outputs (multi-step workflow)
task.log(
    {
        "requirements_analysis": "Analyzed requirements: build REST API with authentication",
        "architecture": "Designed 3-tier architecture with FastAPI, PostgreSQL, JWT auth",
        "implementation": "Implemented all endpoints with proper error handling and validation",
        "testing": "Created unit tests achieving 85% coverage"
    }
)

# Define comprehensive quality rubrics
rubric = Rubric(
    # LLM qualitative assessment
    traits=[
        RubricTrait(
            name="clarity",
            description="How clear and well-explained is the agent's output?",
            kind="score",
            min_score=1,
            max_score=5
        ),
        RubricTrait(
            name="technical_soundness",
            description="Is the technical approach sound and appropriate?",
            kind="boolean"
        )
    ],

    # Deterministic checks
    manual_traits=[
        RegexTrait(
            name="mentions_testing",
            description="Agent mentioned testing or verification",
            pattern=r"(?i)(test|testing|verification|validate)"
        ),
        RegexTrait(
            name="no_errors",
            description="No error indicators in output",
            pattern=r"(?i)(error|exception|failed|broken)",
            invert_result=True
        )
    ],

    # Quantitative coverage
    metric_traits=[
        MetricRubricTrait(
            name="requirement_coverage",
            description="Coverage of required implementation steps",
            metrics=["precision", "recall", "f1"],
            tp_instructions=[
                "Analyze requirements",
                "Design architecture",
                "Implement solution",
                "Write tests",
                "Handle errors"
            ]
        )
    ]
)
task.add_rubric(rubric)

# Configure evaluation
config = VerificationConfig(
    parsing_models=[
        ModelConfig(
            id="evaluator",
            model_provider="openai",
            model_name="gpt-4o-mini",
            temperature=0.0
        )
    ],
    parsing_only=True,
    replicate_count=3  # Run 3 times for reliability
)

# Evaluate
result = task.evaluate(config)

# Display results
print(result.display())

# Get aggregated scores
aggregated = result.global_eval.aggregate_rubric_results()
print("\nAggregated Rubric Scores:")
for question_id, scores in aggregated.items():
    print(f"\n{question_id}:")
    for category, traits in scores.items():
        if category != "failed_replicate_count":
            print(f"  {category}: {traits}")

# Export for further analysis
with open("agent_evaluation.json", "w") as f:
    f.write(result.export_json(include_logs=True, indent=2))
```

### Example 2: Multi-Step Agent Evaluation

```python
task = TaskEval(task_id="multi_step_agent")

# Planning phase
planning_rubric = Rubric(
    traits=[
        RubricTrait(
            name="plan_quality",
            description="Quality of the planning and strategy",
            kind="score",
            min_score=1,
            max_score=5
        )
    ],
    manual_traits=[
        RegexTrait(
            name="has_plan",
            description="Contains planning keywords",
            pattern=r"(?i)(plan|strategy|approach|steps)"
        )
    ]
)
task.add_rubric(planning_rubric, step_id="planning")

# Execution phase
execution_rubric = Rubric(
    traits=[
        RubricTrait(
            name="execution_quality",
            description="Quality of execution and implementation",
            kind="score",
            min_score=1,
            max_score=5
        )
    ],
    manual_traits=[
        RegexTrait(
            name="has_implementation",
            description="Contains implementation details",
            pattern=r"(?i)(implement|execute|build|create)"
        )
    ]
)
task.add_rubric(execution_rubric, step_id="execution")

# Log outputs to respective steps
task.log(
    {
        "analysis": "Analyzed the problem: need to build data pipeline",
        "strategy": "Plan: 1) Extract data 2) Transform 3) Load to warehouse",
        "considerations": "Considered error handling and retry logic"
    },
    step_id="planning"
)

task.log(
    {
        "code": "Implemented ETL pipeline with proper error handling",
        "testing": "Tested with sample data, validated output format",
        "deployment": "Deployed to staging environment"
    },
    step_id="execution"
)

# Configure
config = VerificationConfig(
    parsing_models=[ModelConfig(model_provider="openai", model_name="gpt-4o-mini")],
    parsing_only=True
)

# Evaluate globally (auto-evaluates all steps)
result = task.evaluate(config)

# Analyze per-step results
for step_id, step_result in result.per_step.items():
    print(f"\n=== Step: {step_id} ===")
    print(step_result.summary())

    for question_id, vr_list in step_result.verification_results.items():
        vr = vr_list[0]
        print(f"{question_id}: {vr.verify_rubric}")
```

### Example 3: Comparative Agent Evaluation

```python
# Evaluate two agent versions on the same task

# Agent V1
task_v1 = TaskEval(task_id="agent_v1")
task_v1.add_rubric(quality_rubric)
task_v1.log({"response": "Agent V1 output here"})
result_v1 = task_v1.evaluate(config)

# Agent V2
task_v2 = TaskEval(task_id="agent_v2")
task_v2.add_rubric(quality_rubric)
task_v2.log({"response": "Agent V2 output here"})
result_v2 = task_v2.evaluate(config)

# Compare results
print("Agent V1:", result_v1.summary())
print("Agent V2:", result_v2.summary())

# Detailed comparison
agg_v1 = result_v1.global_eval.aggregate_rubric_results()
agg_v2 = result_v2.global_eval.aggregate_rubric_results()

for question_id in agg_v1.keys():
    scores_v1 = agg_v1[question_id].get("llm", {})
    scores_v2 = agg_v2[question_id].get("llm", {})

    print(f"\n{question_id}:")
    for trait in scores_v1.keys():
        diff = scores_v2.get(trait, 0) - scores_v1.get(trait, 0)
        print(f"  {trait}: V1={scores_v1[trait]}, V2={scores_v2[trait]}, Δ={diff:+.2f}")
```

---

## Related Documentation

- **[Rubrics](../using-karenina/rubrics.md)** - Detailed rubric trait documentation
- **[Verification](../using-karenina/verification.md)** - Core verification pipeline
- **[Manual Traces](manual-traces.md)** - Replaying pre-recorded LLM responses (related concept)
- **[API Reference](../api-reference.md)** - Full API documentation for TaskEval classes

---

## Learn More

- Explore the [examples directory](../../examples/taskeval/) for complete working examples
- Review the [simplified interface example](../../examples/task_eval_simplified_interface.py) for quick start
- Check the [API Reference](../api-reference.md#taskeval) for detailed class documentation

---

*For questions or issues with TaskEval, please refer to the [troubleshooting guide](../troubleshooting.md) or consult the API reference.*
