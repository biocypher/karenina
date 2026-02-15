# Iterating on Your Benchmark

Verification results often reveal that some templates or rubric traits need refinement. This page covers the workflow for identifying failures, making targeted improvements, and re-running verification to measure progress.

---

## The Iteration Cycle

```
Run verification
    │
    ▼
Identify failures (failing templates, low rubric scores)
    │
    ▼
Diagnose root causes (inspect parsed responses, field mismatches)
    │
    ▼
Make targeted fixes (update templates, adjust rubrics)
    │
    ▼
Re-run verification (on affected questions only)
    │
    ▼
Measure improvement (compare pass rates, trait scores)
```

Each step uses specific APIs covered below.

---

## Identifying Failing Templates

### Filter to Failures

The `TemplateResults` accessor provides direct filtering for failed verifications:

```python
results = benchmark.run_verification(config)

# Get template results and filter to failures
template_results = results.get_template_results()
failed = template_results.filter(failed_only=True)

print(f"Failed: {len(failed.results)} / {len(template_results.results)}")
```

### Find Failing Questions

Use a DataFrame to identify which questions are failing and why:

```python
df = template_results.to_dataframe()

# Questions where verify_result is False
failing_rows = df[df["verify_result"] == False]
failing_question_ids = failing_rows["question_id"].unique().tolist()
print(f"Questions failing: {len(failing_question_ids)}")
```

### Pass Rate by Question

Aggregate pass rates to find the weakest questions:

```python
pass_rates = template_results.aggregate_pass_rate(by="question_id")
for q_id, rate in sorted(pass_rates.items(), key=lambda x: x[1]):
    if rate < 1.0:
        print(f"  {q_id}: {rate:.0%}")
```

### Get a Quick Summary

```python
summary = template_results.get_template_summary()
print(f"Overall pass rate: {summary['pass_rate']:.0%}")
print(f"Passed: {summary['num_passed']} / {summary['num_results']}")
```

---

## Diagnosing Template Failures

Once you know which questions are failing, inspect the parsed responses to understand why.

### Inspect Field-Level Mismatches

The template DataFrame includes field-by-field comparison columns:

```python
df = failed.to_dataframe()

# Each row represents one field comparison
for _, row in df.iterrows():
    if not row["field_match"]:
        print(f"Question: {row['question_id']}")
        print(f"  Field: {row['field_name']}")
        print(f"  Expected (GT): {row['gt_value']}")
        print(f"  Parsed (LLM):  {row['llm_value']}")
        print()
```

This reveals common issues:
- **Case differences**: `"BCL2"` vs `"bcl2"` — add `.lower()` normalization in `verify()`
- **Format differences**: `"42.0"` vs `"42"` — use numeric comparison with tolerance
- **Extra text**: `"The answer is Paris"` vs `"Paris"` — improve field description to ask for the value only
- **Missing fields**: `None` vs expected value — the Judge LLM couldn't extract the field

### Inspect Raw Responses

For deeper diagnosis, look at the full verification result:

```python
for result in results:
    if result.template and result.template.verify_result is False:
        meta = result.metadata
        print(f"Question: {meta.question_text[:80]}")
        print(f"Model: {meta.answering.model_name}")

        # What the Judge LLM parsed
        if result.template.parsed_llm_response:
            print(f"Parsed LLM response: {result.template.parsed_llm_response}")
        if result.template.parsed_gt_response:
            print(f"Parsed GT response:  {result.template.parsed_gt_response}")
        print()
```

---

## Fixing Templates

### Update a Template

Once you've diagnosed the issue, update the template code in-memory — no need to save and reload:

```python
new_template = '''
from pydantic import Field
from karenina.schemas.entities import BaseAnswer

class Answer(BaseAnswer):
    gene_symbol: str = Field(
        description="The official HGNC gene symbol mentioned in the response"
    )

    def model_post_init(self, __context):
        self.correct = {"gene_symbol": "BCL2"}

    def verify(self) -> bool:
        return self.gene_symbol.strip().upper() == self.correct["gene_symbol"].upper()
'''

benchmark.update_template(question_id, new_template)
```

`update_template()` replaces the template for that question immediately. The change is in the `Benchmark` object's state — you can re-run verification without calling `save()` first.

### Validate After Editing

After modifying templates, validate that the code is syntactically correct:

```python
is_valid, errors = benchmark.validate_templates()
if not is_valid:
    for err in errors:
        print(f"  {err['question_id']}: {err['error']}")
```

---

## Improving Rubric Traits

### Identify Low-Scoring Traits

Use rubric DataFrames to find traits with consistently low scores:

```python
rubric_results = results.get_rubrics_results()
df_rubric = rubric_results.to_dataframe()

# Average score per trait
trait_scores = df_rubric.groupby("trait_name")["trait_score"].mean()
for name, score in trait_scores.sort_values().items():
    print(f"  {name}: {score:.2f}")
```

### Update Global Rubric Traits

Remove a poorly-performing trait and replace it with an improved version:

```python
from karenina.schemas import LLMRubricTrait

# Remove the old trait
benchmark.remove_global_rubric_trait("clarity")

# Add an improved version with a better description
benchmark.add_global_rubric_trait(
    LLMRubricTrait(
        name="clarity",
        kind="score",
        description="Rate how clearly the response communicates its answer. "
                    "A clear response states the answer directly without unnecessary "
                    "preamble, hedging, or tangential information. Score 1 for "
                    "unclear, 5 for excellent clarity.",
        min_score=1,
        max_score=5,
        higher_is_better=True,
    )
)
```

### Update Question-Specific Traits

For traits that only apply to certain questions:

```python
from karenina.schemas import RegexTrait

# Replace a question-specific rubric entirely
from karenina.schemas import Rubric

benchmark.set_question_rubric(
    question_id,
    Rubric(regex_traits=[
        RegexTrait(
            name="citation_format",
            pattern=r"\[\d+\]",
            description="Response includes numbered citations",
            higher_is_better=True,
        )
    ])
)
```

---

## Re-Running Verification

### Re-Run Only Failing Questions

The key to efficient iteration — re-run verification only on questions that failed:

```python
# Collect failing question IDs from earlier analysis
failing_question_ids = [...]

# Re-run only those questions
results_v2 = benchmark.run_verification(
    config,
    question_ids=failing_question_ids,
)
```

This avoids re-running questions that already pass, saving time and API costs.

### Compare Before and After

```python
# Before
summary_v1 = results.get_template_results().get_template_summary()

# After (on the subset that was re-run)
summary_v2 = results_v2.get_template_results().get_template_summary()

print(f"Before: {summary_v1['pass_rate']:.0%} ({summary_v1['num_passed']}/{summary_v1['num_results']})")
print(f"After:  {summary_v2['pass_rate']:.0%} ({summary_v2['num_passed']}/{summary_v2['num_results']})")
```

### Tag Runs for Tracking

Use `run_name` to label iteration runs so you can distinguish them later:

```python
results_v1 = benchmark.run_verification(config, run_name="v1-initial")

# ... make fixes ...

results_v2 = benchmark.run_verification(
    config,
    question_ids=failing_question_ids,
    run_name="v2-fixed-templates",
)
```

Run names appear in `result.metadata.run_name` and in exported results.

---

## Common Iteration Patterns

### Pattern 1: Fix-and-Verify Loop

For systematic template improvement:

```python
for q_id in failing_question_ids:
    # Inspect the current template
    current = benchmark.get_template(q_id)
    print(f"\n--- {q_id} ---")
    print(current)

    # Write and apply a fix
    fixed_template = """..."""  # Your improved template
    benchmark.update_template(q_id, fixed_template)

    # Verify just this question
    single_result = benchmark.run_verification(config, question_ids=[q_id])
    summary = single_result.get_template_results().get_template_summary()
    print(f"Result: {'PASS' if summary['pass_rate'] == 1.0 else 'FAIL'}")
```

### Pattern 2: Multi-Model Failure Analysis

Identify questions that fail on specific models:

```python
template_results = results.get_template_results()
pass_rates = template_results.aggregate_pass_rate(by="question_id")

by_model = template_results.group_by_model()
for model_name, model_results in by_model.items():
    model_pass = model_results.aggregate_pass_rate(by="question_id")
    for q_id, rate in model_pass.items():
        if rate < 1.0 and pass_rates.get(q_id, 0) < 1.0:
            print(f"  {q_id} fails on {model_name} (rate: {rate:.0%})")
```

### Pattern 3: Save After Iterating

Once your templates and rubrics are refined, save the benchmark to preserve changes:

```python
# Save updated benchmark (templates + rubrics persisted)
benchmark.save("benchmark_v2.jsonld")

# Or save to database
benchmark.save_to_db("sqlite:///dbs/benchmarks.db")
```

---

## Tips for Effective Iteration

1. **Start with templates** — Template failures are the most actionable. Fix `verify()` logic before tuning rubric trait descriptions.

2. **Use `question_ids` for targeted re-runs** — Avoid re-running the entire benchmark when only a few questions need attention.

3. **Improve field descriptions first** — Many parsing failures come from ambiguous field descriptions. A clearer `description` in the template often resolves extraction issues without changing `verify()`.

4. **Normalize in `verify()`** — Add `.strip().lower()` or numeric tolerance to handle format differences the Judge LLM introduces.

5. **Check abstention and sufficiency** — If many questions fail, check whether the answering model is abstaining or giving insufficient responses. These show up in `result.template.abstention_detected` and `result.template.sufficiency_detected`.

6. **Use `run_name` to track iterations** — This makes it easy to compare results across refinement cycles.

---

## Next Steps

- [VerificationResult Structure](verification-result.md) — Understand all available result fields
- [DataFrame Analysis](dataframe-analysis.md) — Detailed DataFrame analysis patterns
- [Exporting Results](exporting.md) — Save results for sharing or archival
- [Writing Custom Templates](../05-creating-benchmarks/writing-templates.md) — Template patterns for complex verify logic
- [Defining Rubrics](../05-creating-benchmarks/defining-rubrics.md) — Creating and modifying rubric traits
