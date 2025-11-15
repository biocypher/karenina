# DataFrame Quick Reference

Quick reference cheat sheet for common DataFrame operations in Karenina verification result analysis.

**See the full guide**: [Analyzing Results with DataFrames](analyzing-results-dataframes.md) for detailed explanations and examples.

---

## Getting DataFrames

```python
# From verification results
result_set = benchmark.run_verification(config)

# Get result type wrappers
templates = result_set.get_templates()
rubrics = result_set.get_rubrics()
judgments = result_set.get_judgments()

# Convert to DataFrames
template_df = templates.to_dataframe()
rubric_df = rubrics.to_dataframe(trait_type="llm")
judgment_df = judgments.to_dataframe()

# Usage DataFrames
usage_detailed = templates.to_usage_dataframe(totals_only=False)
usage_totals = templates.to_usage_dataframe(totals_only=True)

# Regex DataFrame
regex_df = templates.to_regex_dataframe()
```

---

## Common Column Groups

### All DataFrames Have

```python
# Status columns (always first)
'completed_without_errors', 'error', 'recursion_limit_reached'

# Identification
'question_id', 'template_id', 'answering_model', 'parsing_model', 'replicate'

# Execution Metadata
'execution_time', 'timestamp', 'run_name', 'job_id'
```

### TemplateResults Columns

```python
# Field-level data
'field_name', 'gt_value', 'llm_value', 'field_match', 'field_type'

# Verification checks
'embedding_check_enabled', 'embedding_score'
'abstention_enabled', 'abstention_detected', 'abstention_explanation'
'regex_validated', 'regex_patterns_matched', 'regex_patterns_failed'
```

### RubricResults Columns

```python
# Trait data
'trait_name', 'trait_type', 'trait_score', 'trait_reasoning'

# Rubric metadata
'rubric_evaluation_performed'
```

### JudgmentResults Columns

```python
# Attribute-level
'attribute_name', 'gt_attribute_value', 'llm_attribute_value', 'attribute_match'

# Excerpt-level (exploded)
'excerpt_index', 'excerpt_text', 'excerpt_confidence', 'excerpt_similarity_score'
'excerpt_hallucination_risk', 'excerpt_hallucination_justification'

# Deep judgment metadata
'deep_judgment_performed', 'attribute_reasoning', 'attribute_overall_risk'
```

---

## Quick Calculations

### Pass Rates

```python
# Template pass rate
df = template_df[template_df['completed_without_errors'] == True]
pass_rate = df.groupby('question_id')['field_match'].mean()

# Helper method
pass_rate = templates.aggregate_pass_rate(by="question_id")
```

### Trait Scores

```python
# Average trait score by question
trait_scores = rubric_df.groupby(['question_id', 'trait_name'])['trait_score'].mean()

# Helper method
trait_scores = rubrics.aggregate_llm_traits(by="question_id")
```

### Excerpt Counts

```python
# Excerpts per attribute
excerpts = judgment_df[judgment_df['excerpt_index'].notna()]
counts = excerpts.groupby(['question_id', 'attribute_name'])['excerpt_index'].max() + 1

# Helper method
counts = judgments.aggregate_excerpt_counts(by="question_id")
```

### Token Usage

```python
# Total tokens
total_tokens = usage_totals['total_tokens'].sum()

# Tokens by stage
stage_tokens = usage_detailed.groupby('usage_stage')['total_tokens'].sum()

# Tokens by model
model_tokens = usage_totals.groupby('answering_model')['total_tokens'].sum()
```

---

## Common Filters

### Successful Verifications Only

```python
successful = df[df['completed_without_errors'] == True]
```

### Specific Question

```python
question_data = df[df['question_id'] == 'my-question-id']
```

### Specific Model

```python
model_data = df[df['answering_model'] == 'gpt-4']
```

### Specific Field

```python
drug_fields = template_df[template_df['field_name'] == 'drug_target']
```

### Specific Trait

```python
accuracy = rubric_df[rubric_df['trait_name'] == 'accuracy']
```

### High-Risk Excerpts

```python
risky = judgment_df[judgment_df['excerpt_hallucination_risk'] == 'high']
```

---

## Common Aggregations

### Mean

```python
avg_pass_rate = df.groupby('question_id')['field_match'].mean()
```

### Count

```python
field_counts = df.groupby('field_name').size()
```

### Multiple Statistics

```python
stats = df.groupby('question_id')['field_match'].agg(['mean', 'std', 'count', 'min', 'max'])
```

### Pivot Table

```python
# Questions × Models
pivot = df.pivot_table(
    values='field_match',
    index='question_id',
    columns='answering_model',
    aggfunc='mean'
)

# Questions × Traits
trait_pivot = rubric_df.pivot_table(
    values='trait_score',
    index='question_id',
    columns='trait_name',
    aggfunc='mean'
)
```

---

## Model Comparison

### Basic Comparison

```python
# Average performance by model
model_perf = df.groupby('answering_model')['field_match'].mean()

# Sort models by performance
model_perf = model_perf.sort_values(ascending=False)
```

### Side-by-Side Comparison

```python
# Create pivot table
comparison = df.pivot_table(
    values='field_match',
    index='question_id',
    columns='answering_model',
    aggfunc='mean'
)

# Find best model per question
comparison['best_model'] = comparison.idxmax(axis=1)

# Calculate performance spread
comparison['spread'] = comparison.max(axis=1) - comparison.min(axis=1)
```

---

## Error Analysis

### Error Summary

```python
failed = df[df['completed_without_errors'] == False]
error_counts = failed.groupby('error').size().sort_values(ascending=False)
```

### Errors by Model

```python
error_by_model = failed.groupby(['error', 'answering_model']).size()
```

### Questions with Errors

```python
error_questions = failed['question_id'].unique()
```

---

## Field-Level Analysis

### Field Performance

```python
successful = template_df[template_df['completed_without_errors'] == True]

field_stats = successful.groupby('field_name').agg({
    'field_match': ['mean', 'count'],
    'field_type': 'first'
})

# Sort by match rate
field_stats = field_stats.sort_values(('field_match', 'mean'))
```

### Problematic Fields

```python
# Fields with <80% match rate
low_performing = field_stats[field_stats[('field_match', 'mean')] < 0.8]
```

---

## Trait Analysis

### Trait Performance

```python
trait_stats = rubric_df.groupby('trait_name').agg({
    'trait_score': ['mean', 'std', 'min', 'max', 'count']
})

# Sort by average score
trait_stats = trait_stats.sort_values(('trait_score', 'mean'))
```

### Binary Traits

```python
binary_traits = rubric_df[rubric_df['trait_type'] == 'llm_binary']
binary_pass_rate = binary_traits.groupby('trait_name')['trait_score'].mean()
```

---

## Deep Judgment Analysis

### Excerpt Quality

```python
# Average confidence by attribute
confidence_map = {'none': 0, 'low': 1, 'medium': 2, 'high': 3}
judgment_df['confidence_score'] = judgment_df['excerpt_confidence'].map(confidence_map)

attr_confidence = judgment_df.groupby('attribute_name')['confidence_score'].mean()
```

### Hallucination Risk

```python
# Count high-risk excerpts by attribute
risky_excerpts = judgment_df[judgment_df['excerpt_hallucination_risk'] == 'high']
risk_counts = risky_excerpts.groupby('attribute_name').size()
```

### Attribute Match Rates

```python
# Overall attribute match rate
attr_match = judgment_df.groupby('attribute_name')['attribute_match'].mean()

# By question
attr_by_question = judgment_df.pivot_table(
    values='attribute_match',
    index='question_id',
    columns='attribute_name',
    aggfunc='mean'
)
```

---

## Cost Analysis

### Total Cost

```python
# Assuming pricing per 1K tokens
INPUT_RATE = 0.0001   # $ per 1K input tokens
OUTPUT_RATE = 0.0003  # $ per 1K output tokens

usage = templates.to_usage_dataframe(totals_only=True)

usage['input_cost'] = usage['input_tokens'] / 1000 * INPUT_RATE
usage['output_cost'] = usage['output_tokens'] / 1000 * OUTPUT_RATE
usage['total_cost'] = usage['input_cost'] + usage['output_cost']

total_cost = usage['total_cost'].sum()
```

### Cost by Model

```python
model_costs = usage.groupby('answering_model')['total_cost'].sum()
```

### Cost by Question

```python
question_costs = usage.groupby('question_id')['total_cost'].sum()
expensive = question_costs.nlargest(10)
```

### Cost by Stage

```python
detailed = templates.to_usage_dataframe(totals_only=False)

detailed['stage_cost'] = (
    detailed['input_tokens'] / 1000 * INPUT_RATE +
    detailed['output_tokens'] / 1000 * OUTPUT_RATE
)

stage_costs = detailed.groupby('usage_stage')['stage_cost'].sum()
```

---

## Exporting

### To CSV

```python
df.to_csv('results.csv', index=False)
```

### To Excel

```python
with pd.ExcelWriter('results.xlsx') as writer:
    template_df.to_excel(writer, sheet_name='Templates', index=False)
    rubric_df.to_excel(writer, sheet_name='Rubrics', index=False)
    judgment_df.to_excel(writer, sheet_name='Judgments', index=False)
```

### To JSON

```python
df.to_json('results.json', orient='records', indent=2)
```

---

## Visualization

### Simple Plot

```python
import matplotlib.pyplot as plt

pass_rates = template_df.groupby('question_id')['field_match'].mean()
pass_rates.plot(kind='bar')
plt.title('Pass Rate by Question')
plt.ylabel('Pass Rate')
plt.tight_layout()
plt.savefig('pass_rates.png')
```

### Heatmap

```python
import seaborn as sns

pivot = rubric_df.pivot_table(
    values='trait_score',
    index='question_id',
    columns='trait_name',
    aggfunc='mean'
)

plt.figure(figsize=(12, 8))
sns.heatmap(pivot, annot=True, cmap='RdYlGn', center=3, vmin=1, vmax=5)
plt.title('Trait Scores Heatmap')
plt.tight_layout()
plt.savefig('trait_heatmap.png')
```

---

## Tips

1. **Filter early**: `df[df['completed_without_errors'] == True]` before aggregations
2. **Use helpers**: For simple cases, helper methods are cleaner than pandas code
3. **Cache DataFrames**: If reusing, convert once and store
4. **Filter trait types**: Use `trait_type` parameter to reduce DataFrame size
5. **Check column existence**: Some columns only exist when features are enabled

---

## Related Documentation

- **[Analyzing Results with DataFrames](analyzing-results-dataframes.md)** - Comprehensive guide with detailed examples
- **[Verification Guide](verification.md)** - Running verification to generate results
- **[Saving and Loading](saving-loading.md)** - Exporting results to CSV, Excel, JSON

## Additional Resources

- **Integration Tests**: `tests/test_dataframe_integration*.py` - Real-world usage examples
- **API Reference**: Method docstrings for detailed parameter descriptions
