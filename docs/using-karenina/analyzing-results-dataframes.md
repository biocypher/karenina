# Analyzing Verification Results with DataFrames

This guide covers how to analyze verification results using the DataFrame-first approach, providing flexible and powerful data analysis with pandas.

**Quick Navigation:**

- [Overview](#overview) - Introduction to DataFrame-first analysis
- [Quick Start](#quick-start) - Get started in 5 minutes
- [DataFrame Methods](#dataframe-methods-reference) - Complete API reference
- [Common Patterns](#common-patterns) - Frequently used analysis patterns
- [Helper Methods](#helper-methods) - Convenience aggregation methods
- [Advanced Usage](#advanced-usage) - Custom aggregators and complex workflows
- [Performance Tips](#performance-tips) - Optimize your analysis

---

## Overview

### What This Guide Covers

This guide focuses on **analyzing verification results** after you've run verification. The DataFrame-first approach provides a modern, flexible way to wrangle and analyze verification output using pandas DataFrames.

**Important**: This guide is about **analyzing results**, not running verification. To learn how to run verification, see the [Verification Guide](verification.md).

**Typical Workflow**:
```python
# 1. Run verification (see verification.md for details)
result_set = benchmark.run_verification(config)

# 2. Extract results (this guide starts here!)
template_results = result_set.get_templates()

# 3. Convert to DataFrame for analysis
df = template_results.to_dataframe()

# 4. Analyze with pandas
pass_rates = df.groupby('question_id')['field_match'].mean()
```

### Why Use DataFrames for Result Analysis?

The DataFrame-first approach offers several advantages over working with raw VerificationResult objects:

- **Familiar API**: Use standard pandas operations for filtering, grouping, and aggregation
- **Flexibility**: Combine multiple verification aspects in custom ways
- **Performance**: Leverage pandas' optimized operations for large result sets
- **Interoperability**: Easy integration with visualization libraries and data science tools
- **Exploratory Analysis**: Quick iteration on analysis without writing custom code

### Design Philosophy

1. **One DataFrame per verification aspect**: TemplateResults, RubricResults, JudgmentResults
2. **Exploded rows**: Each row represents the smallest meaningful unit (field, trait, excerpt)
3. **Consistent schema**: All DataFrames share common Status, Identification, and Metadata columns
4. **Helper methods**: Convenience methods for common operations, implemented using pandas

### When to Use This Approach

Use DataFrames when you need to:
- ✅ Analyze verification results across multiple questions
- ✅ Compare model performance
- ✅ Calculate pass rates, trait scores, or other metrics
- ✅ Find patterns or anomalies in verification outcomes
- ✅ Generate reports or visualizations from results
- ✅ Export results to CSV, Excel, or other formats for external analysis

**Not covered in this guide**:
- ❌ Running verification (see [Verification Guide](verification.md))
- ❌ Defining benchmarks and questions (see [Defining Benchmarks](defining-benchmark.md))
- ❌ Creating templates and rubrics (see [Templates](templates.md) and [Rubrics](rubrics.md))

---

## Quick Start

### Prerequisites

Before using this guide, you should have:
1. **Run verification** and obtained a `VerificationResultSet` object
2. Basic familiarity with pandas DataFrames

**If you haven't run verification yet**, see the [Verification Guide](verification.md) first.

### Basic Workflow

```python
# STEP 0: Run verification (see verification.md for details)
# This guide assumes you already have results from verification:
result_set = benchmark.run_verification(config)

# ============================================================
# This guide starts here - analyzing verification results
# ============================================================

# STEP 1: Extract result type wrappers from verification output
template_results = result_set.get_templates()
rubric_results = result_set.get_rubrics()
judgment_results = result_set.get_judgments()

# STEP 2: Convert verification results to DataFrames
template_df = template_results.to_dataframe()
rubric_df = rubric_results.to_dataframe(trait_type="llm")
judgment_df = judgment_results.to_dataframe()

# STEP 3: Analyze with pandas
pass_rate = template_df.groupby('question_id')['field_match'].mean()
print(pass_rate)
```

### Basic Example: Template Verification Analysis

```python
# Get template results DataFrame
df = template_results.to_dataframe()

# Filter to successful verifications only
successful = df[df['completed_without_errors'] == True]

# Calculate pass rate by question
pass_rates = successful.groupby('question_id')['field_match'].mean()

# Find questions with low pass rates
low_performers = pass_rates[pass_rates < 0.8]
print(f"Questions with <80% pass rate: {len(low_performers)}")

# Analyze by field
field_performance = successful.groupby('field_name')['field_match'].agg(['mean', 'count'])
print(field_performance)
```

---

## DataFrame Methods Reference

### TemplateResults

#### `to_dataframe()`

Convert template verification results to pandas DataFrame with field-level explosion.

**Returns**: `pd.DataFrame` with one row per parsed field comparison

**Key Columns**:
- **Status**: `completed_without_errors`, `error`, `recursion_limit_reached`
- **Field Comparison**: `field_name`, `gt_value`, `llm_value`, `field_match`, `field_type`
- **Verification Checks**: `embedding_check_enabled`, `embedding_score`, `abstention_detected`, `regex_validated`
- **Identification**: `question_id`, `template_id`, `answering_model`, `parsing_model`

**Example**:
```python
df = template_results.to_dataframe()

# Each field gets its own row
# question_1 with 3 fields → 3 rows
# question_2 with 5 fields → 5 rows

# Analyze specific fields
drug_fields = df[df['field_name'] == 'drug_target']
match_rate = drug_fields['field_match'].mean()
```

#### `to_regex_dataframe()`

Convert regex validation results to DataFrame with pattern explosion.

**Returns**: `pd.DataFrame` with one row per regex pattern evaluation

**Key Columns**:
- `pattern_name`: Name of the regex pattern
- `pattern_regex`: The actual regex expression
- `pattern_matched`: Whether pattern matched
- `extracted_value`: Value extracted by the pattern
- `match_position`: Position of match in text

**Example**:
```python
regex_df = template_results.to_regex_dataframe()

# Analyze pattern success rates
pattern_stats = regex_df.groupby('pattern_name').agg({
    'pattern_matched': 'mean',
    'extracted_value': 'count'
})
```

#### `to_usage_dataframe(totals_only=False)`

Convert token usage data to DataFrame.

**Parameters**:
- `totals_only` (bool): If True, return only total usage per verification. If False, explode by stage.

**Returns**: `pd.DataFrame` with token usage metrics

**Key Columns** (when `totals_only=False`):
- `usage_stage`: Stage name (answering, parsing, embedding_check, etc.)
- `input_tokens`, `output_tokens`, `total_tokens`: Token counts
- `agent_iterations`, `agent_tool_calls`: Agent-specific metrics

**Example**:
```python
# Detailed usage by stage
usage_df = template_results.to_usage_dataframe(totals_only=False)
stage_costs = usage_df.groupby('usage_stage')['total_tokens'].sum()

# Total usage only
totals_df = template_results.to_usage_dataframe(totals_only=True)
total_cost = totals_df['total_tokens'].sum()
```

---

### RubricResults

#### `to_dataframe(trait_type="all")`

Convert rubric evaluation results to DataFrame with trait explosion.

**Parameters**:
- `trait_type` (str): Filter trait type - "llm", "llm_score", "llm_binary", "manual", "metric", or "all"

**Returns**: `pd.DataFrame` with one row per trait evaluation

**Key Columns**:
- `trait_name`: Name of the rubric trait
- `trait_type`: Type (llm_score, llm_binary, manual, metric)
- `trait_score`: Score value (1-5 for llm_score, True/False for binary)
- `trait_reasoning`: LLM reasoning (for LLM traits)
- `rubric_evaluation_performed`: Whether rubric was evaluated

**Example**:
```python
# Get all LLM-scored traits
llm_df = rubric_results.to_dataframe(trait_type="llm_score")

# Analyze trait performance
trait_scores = llm_df.groupby('trait_name')['trait_score'].agg(['mean', 'std', 'count'])

# Find low-scoring traits
low_scores = trait_scores[trait_scores['mean'] < 3.0]

# Compare models
model_comparison = llm_df.pivot_table(
    values='trait_score',
    index='question_id',
    columns='answering_model',
    aggfunc='mean'
)
```

---

### JudgmentResults

#### `to_dataframe()`

Convert deep judgment results to DataFrame with attribute and excerpt explosion.

**Returns**: `pd.DataFrame` with one row per attribute-excerpt pair

**Key Columns**:
- `attribute_name`: Name of the answer attribute
- `gt_attribute_value`, `llm_attribute_value`: Ground truth and LLM values
- `attribute_match`: Whether attribute matched
- `excerpt_index`: Index of excerpt (0, 1, 2, ...) or None if no excerpts
- `excerpt_text`: Text of the excerpt
- `excerpt_confidence`: Confidence level (none/low/medium/high)
- `excerpt_similarity_score`: Fuzzy match score (0.0-1.0)
- `excerpt_hallucination_risk`: Risk level (low/medium/high)
- `attribute_reasoning`: LLM reasoning about the attribute

**Explosion Logic**:
- Attributes with 3 excerpts → 3 rows (one per excerpt)
- Attributes with 0 excerpts → 1 row (excerpt fields are None)

**Example**:
```python
judgment_df = judgment_results.to_dataframe()

# Analyze excerpt quality
excerpt_stats = judgment_df.groupby('attribute_name').agg({
    'excerpt_confidence': lambda x: (x == 'high').mean(),
    'excerpt_similarity_score': 'mean',
    'excerpt_hallucination_risk': lambda x: (x == 'high').sum()
})

# Find attributes with hallucination risk
risky = judgment_df[judgment_df['excerpt_hallucination_risk'] == 'high']
print(f"High-risk excerpts: {len(risky)}")

# Analyze by question
question_excerpts = judgment_df.groupby('question_id')['excerpt_index'].max() + 1
avg_excerpts = question_excerpts.mean()
```

---

## Common Patterns

### Pattern 1: Calculate Pass Rates

**Template verification pass rate by question:**
```python
df = template_results.to_dataframe()

# Filter to successful verifications
successful = df[df['completed_without_errors'] == True]

# Calculate pass rate by question
pass_rates = successful.groupby('question_id')['field_match'].mean()

# Get questions below threshold
failing = pass_rates[pass_rates < 0.9]
print(f"Questions with <90% pass rate: {list(failing.index)}")
```

**Rubric trait scores by model:**
```python
rubric_df = rubric_results.to_dataframe(trait_type="llm_score")

# Average score by model
model_scores = rubric_df.groupby('answering_model')['trait_score'].mean()

# Detailed breakdown
model_trait_scores = rubric_df.pivot_table(
    values='trait_score',
    index='answering_model',
    columns='trait_name',
    aggfunc='mean'
)
```

### Pattern 2: Multi-Dimensional Analysis

**Template + Rubric combined analysis:**
```python
# Get DataFrames
template_df = template_results.to_dataframe()
rubric_df = rubric_results.to_dataframe(trait_type="llm")

# Aggregate to question level
template_agg = template_df.groupby('question_id')['field_match'].mean()
rubric_agg = rubric_df.groupby('question_id')['trait_score'].mean()

# Merge
combined = pd.DataFrame({
    'template_pass_rate': template_agg,
    'rubric_avg_score': rubric_agg
})

# Find discrepancies
discrepancies = combined[
    (combined['template_pass_rate'] > 0.9) &
    (combined['rubric_avg_score'] < 3.0)
]
print(f"Questions with high template pass but low rubric: {len(discrepancies)}")
```

### Pattern 3: Field-Level Analysis

**Identify problematic fields:**
```python
df = template_results.to_dataframe()
successful = df[df['completed_without_errors'] == True]

# Calculate match rate by field
field_performance = successful.groupby('field_name').agg({
    'field_match': ['mean', 'count', 'sum'],
    'field_type': 'first'
})

# Sort by match rate
field_performance = field_performance.sort_values(('field_match', 'mean'))

# Show worst performing fields
print("Worst performing fields:")
print(field_performance.head(10))
```

### Pattern 4: Model Comparison

**Compare multiple models:**
```python
df = template_results.to_dataframe()

# Pivot: questions × models
model_comparison = df.pivot_table(
    values='field_match',
    index='question_id',
    columns='answering_model',
    aggfunc='mean'
)

# Calculate relative performance
model_comparison['best_model'] = model_comparison.idxmax(axis=1)
model_comparison['performance_spread'] = model_comparison.max(axis=1) - model_comparison.min(axis=1)

# Find questions with high model variance
high_variance = model_comparison[model_comparison['performance_spread'] > 0.3]
```

### Pattern 5: Temporal Analysis

**Track performance over replicates:**
```python
df = template_results.to_dataframe()

# Group by question and replicate
replicate_performance = df.groupby(['question_id', 'replicate']).agg({
    'field_match': 'mean',
    'execution_time': 'mean'
})

# Calculate stability (std across replicates)
stability = replicate_performance.groupby('question_id')['field_match'].std()

# Find unstable questions
unstable = stability[stability > 0.1]
print(f"Unstable questions (high replicate variance): {list(unstable.index)}")
```

### Pattern 6: Error Analysis

**Analyze verification failures:**
```python
df = template_results.to_dataframe()

# Get failed verifications
failed = df[df['completed_without_errors'] == False]

# Group by error type
error_summary = failed.groupby('error').agg({
    'question_id': 'count',
    'answering_model': lambda x: list(x.unique())
})
error_summary.columns = ['count', 'affected_models']

print("Error summary:")
print(error_summary.sort_values('count', ascending=False))
```

### Pattern 7: Cost Analysis

**Calculate token usage and costs:**
```python
usage_df = template_results.to_usage_dataframe(totals_only=True)

# Assuming cost per 1K tokens
INPUT_COST_PER_1K = 0.0001
OUTPUT_COST_PER_1K = 0.0003

usage_df['input_cost'] = usage_df['input_tokens'] / 1000 * INPUT_COST_PER_1K
usage_df['output_cost'] = usage_df['output_tokens'] / 1000 * OUTPUT_COST_PER_1K
usage_df['total_cost'] = usage_df['input_cost'] + usage_df['output_cost']

# Cost by model
model_costs = usage_df.groupby('answering_model')['total_cost'].sum()

# Cost by question
question_costs = usage_df.groupby('question_id')['total_cost'].sum()
expensive_questions = question_costs.nlargest(10)
```

---

## Helper Methods

Helper methods provide convenient aggregations for common operations. They are implemented using the DataFrame API internally, so you can always replicate their behavior with custom pandas code.

### TemplateResults Helper Methods

#### `aggregate_pass_rate(by="question_id", strategy="mean")`

Calculate template verification pass rates.

**Parameters**:
- `by` (str): Grouping column ("question_id", "answering_model", etc.)
- `strategy` (str): Aggregation strategy ("mean", "median", "min", "max")

**Returns**: `dict[str, float]` mapping group values to pass rates

**Example**:
```python
# Pass rate by question
question_rates = template_results.aggregate_pass_rate(by="question_id")

# Pass rate by model
model_rates = template_results.aggregate_pass_rate(by="answering_model")

# Equivalent pandas code:
df = template_results.to_dataframe()
successful = df[df['completed_without_errors'] == True]
question_rates_manual = successful.groupby('question_id')['field_match'].mean().to_dict()
```

#### `aggregate_embedding_scores(by="question_id", strategy="mean")`

Aggregate embedding similarity scores for questions with embedding checks.

**Returns**: `dict[str, float]` mapping to average embedding scores

**Example**:
```python
# Average embedding score by question
embedding_scores = template_results.aggregate_embedding_scores(by="question_id")

# Equivalent pandas:
df = template_results.to_dataframe()
with_embedding = df[df['embedding_check_enabled'] == True]
scores = with_embedding.groupby('question_id')['embedding_score'].mean().to_dict()
```

#### `aggregate_regex_success_rate(by="question_id", strategy="mean")`

Calculate regex validation success rates.

**Returns**: `dict[str, float]` mapping to regex success rates

#### `aggregate_abstention_rate(by="question_id", strategy="mean")`

Calculate abstention detection rates.

**Returns**: `dict[str, float]` mapping to abstention rates

---

### RubricResults Helper Methods

#### `aggregate_llm_traits(by="question_id", strategy="mean")`

Aggregate LLM trait scores.

**Returns**: `dict[str, dict[str, float]]` - nested dict with trait scores per group

**Example**:
```python
# Get average scores by question
question_traits = rubric_results.aggregate_llm_traits(by="question_id")

# Access specific question's traits
question_1_scores = question_traits["question-1"]
print(f"Accuracy: {question_1_scores['accuracy']}")
print(f"Completeness: {question_1_scores['completeness']}")

# Equivalent pandas:
df = rubric_results.to_dataframe(trait_type="llm")
for question_id, group in df.groupby('question_id'):
    trait_scores = group.groupby('trait_name')['trait_score'].mean().to_dict()
```

#### `aggregate_manual_traits(by="question_id", strategy="mean")`

Aggregate manual trait values.

**Returns**: `dict[str, dict[str, Any]]` - nested dict with manual trait values

#### `aggregate_metric_traits(by="question_id", strategy="mean")`

Aggregate computed metric trait values.

**Returns**: `dict[str, dict[str, float]]` - nested dict with metric values

---

### JudgmentResults Helper Methods

#### `aggregate_excerpt_counts(by="question_id", strategy="mean")`

Aggregate excerpt counts per attribute.

**Returns**: `dict[str, dict[str, float]]` - nested dict mapping to attribute-level counts

**Example**:
```python
# Get average excerpt counts by question
question_excerpts = judgment_results.aggregate_excerpt_counts(by="question_id")

# Access specific question
q1_excerpts = question_excerpts["question-1"]
print(f"drug_target: {q1_excerpts['drug_target']} excerpts")
print(f"mechanism: {q1_excerpts['mechanism']} excerpts")

# Equivalent pandas:
df = judgment_results.to_dataframe()
df_with_excerpts = df[df['excerpt_index'].notna()]
counts = df_with_excerpts.groupby(['question_id', 'attribute_name'])['excerpt_index'].max() + 1
```

#### `aggregate_hallucination_risk_distribution(by="question_id")`

Get distribution of hallucination risk levels.

**Returns**: `dict[str, dict[str, int]]` - counts of risk levels per group

#### `aggregate_model_calls(by="question_id", strategy="sum")`

Aggregate deep judgment model call counts.

**Returns**: `dict[str, int]` - total model calls per group

---

## Advanced Usage

### Custom Aggregators

You can register custom aggregation functions:

```python
from karenina.schemas.workflow import register_aggregator

# Define custom aggregator
def weighted_score(df, by, weights):
    """Calculate weighted average of trait scores."""
    def weighted_mean(group):
        return (group['trait_score'] * group['trait_name'].map(weights)).sum() / sum(weights.values())

    return df.groupby(by).apply(weighted_mean).to_dict()

# Register it
register_aggregator("weighted_score", weighted_score)

# Use it
weights = {'accuracy': 0.4, 'completeness': 0.3, 'relevance': 0.3}
weighted_scores = rubric_results.aggregate(
    aggregator="weighted_score",
    by="question_id",
    weights=weights
)
```

### Merging Multiple Verification Runs

```python
# Run multiple verifications
results_1 = benchmark.run_verification(config_1)
results_2 = benchmark.run_verification(config_2)

# Get DataFrames
df1 = results_1.get_templates().to_dataframe()
df2 = results_2.get_templates().to_dataframe()

# Add run identifier
df1['run'] = 'config_1'
df2['run'] = 'config_2'

# Concatenate
combined = pd.concat([df1, df2], ignore_index=True)

# Compare configs
config_comparison = combined.pivot_table(
    values='field_match',
    index='question_id',
    columns='run',
    aggfunc='mean'
)
```

### Exporting for Visualization

```python
import matplotlib.pyplot as plt
import seaborn as sns

# Get rubric scores
df = rubric_results.to_dataframe(trait_type="llm_score")

# Create heatmap: questions × traits
pivot = df.pivot_table(
    values='trait_score',
    index='question_id',
    columns='trait_name',
    aggfunc='mean'
)

# Plot
plt.figure(figsize=(12, 8))
sns.heatmap(pivot, annot=True, cmap='RdYlGn', center=3, vmin=1, vmax=5)
plt.title('Rubric Trait Scores by Question')
plt.tight_layout()
plt.savefig('rubric_heatmap.png')
```

---

## Performance Tips

### 1. Filter Early

Filter DataFrames before aggregation to reduce computational load:

```python
# Good: Filter first
df = template_results.to_dataframe()
successful = df[df['completed_without_errors'] == True]
pass_rates = successful.groupby('question_id')['field_match'].mean()

# Less efficient: Aggregate then filter
all_rates = df.groupby('question_id')['field_match'].mean()
```

### 2. Use Helper Methods for Simple Cases

Helper methods are optimized for common operations:

```python
# Prefer helper for simple aggregation
pass_rates = template_results.aggregate_pass_rate(by="question_id")

# Use DataFrame for complex custom logic
df = template_results.to_dataframe()
custom_metric = df.groupby('question_id').apply(my_complex_function)
```

### 3. Avoid Repeated Conversions

Cache DataFrames if you need them multiple times:

```python
# Good: Convert once
template_df = template_results.to_dataframe()
pass_rates = template_df.groupby('question_id')['field_match'].mean()
field_stats = template_df.groupby('field_name')['field_match'].agg(['mean', 'std'])

# Less efficient: Convert multiple times
pass_rates = template_results.to_dataframe().groupby('question_id')['field_match'].mean()
field_stats = template_results.to_dataframe().groupby('field_name')['field_match'].agg(['mean', 'std'])
```

### 4. Use Appropriate Trait Filters

When working with rubrics, filter to specific trait types to reduce DataFrame size:

```python
# Efficient: Filter at conversion
llm_df = rubric_results.to_dataframe(trait_type="llm_score")

# Less efficient: Filter after conversion
all_df = rubric_results.to_dataframe(trait_type="all")
llm_df = all_df[all_df['trait_type'] == 'llm_score']
```

### 5. Use totals_only for Cost Analysis

For simple cost calculations, use `totals_only=True`:

```python
# Efficient for total costs
totals = template_results.to_usage_dataframe(totals_only=True)
total_cost = calculate_cost(totals['total_tokens'].sum())

# Use detailed only when analyzing by stage
detailed = template_results.to_usage_dataframe(totals_only=False)
stage_costs = detailed.groupby('usage_stage')['total_tokens'].sum()
```

---

---

## Related Documentation

- **[DataFrame Quick Reference](dataframe-quick-reference.md)** - Cheat sheet for common operations
- **[Verification Guide](verification.md)** - Running verification to generate results
- **[Accessing and Filtering Questions](accessing-filtering.md)** - Working with benchmark questions
- **[Rubrics](rubrics.md)** - Understanding rubric evaluation
- **[Saving and Loading](saving-loading.md)** - Exporting results to files

## Additional Resources

- **API Reference**: See individual method docstrings for detailed parameter descriptions
- **Integration Tests**: Check `tests/test_dataframe_integration*.py` for real-world usage examples
- **Code Examples**: Explore the test files for working code patterns

For questions or feedback, please consult the main Karenina documentation or open an issue on GitHub.
