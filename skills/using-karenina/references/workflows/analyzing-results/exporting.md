# Exporting Results

After running verification and analyzing results, you'll often want to export data for sharing, archival, or external analysis. Karenina provides two complementary export approaches:

- **Benchmark export methods** — Export complete verification results as JSON or CSV strings/files
- **DataFrame export** — Export pandas DataFrames to any format pandas supports

---

## Benchmark Export Methods

> **Deprecated methods:**
> `export_verification_results()`, `export_verification_results_to_file()`, and `load_verification_results_from_file()` on `Benchmark` are deprecated and emit a `DeprecationWarning`. They still work, so the export examples on this page remain accurate. Use `ResultsIOManager` to load standard verification-result JSON or CSV exports. Use `ResultsStore.from_file()` only for the store-specific JSON format written by `ResultsStore.export_to_file()`.
> 
The `Benchmark` class provides two methods for exporting verification results:

| Method | Returns | Use When |
|--------|---------|----------|
| `export_verification_results()` | `str` | You need the data as a string (e.g., for an API response or further processing) |
| `export_verification_results_to_file()` | `None` | You want to write directly to a file |

### Export as JSON

JSON export produces a structured document with metadata, configuration, and per-result details:

```python
# Export all results as JSON string
json_str = benchmark.export_verification_results(format="json")

# Export to a JSON file
from pathlib import Path
benchmark.export_verification_results_to_file(Path("results.json"))
```

The JSON output uses a v2.0 format with optimizations:

- Rubric definitions stored once in `shared_data` (not repeated per-result)
- Trace filtering fields at result root level (shared by template and rubric evaluation)
- 50-70% smaller than the legacy format

### Export as CSV

CSV export produces a flat table with one row per verification result:

```python
# Export all results as CSV string
csv_str = benchmark.export_verification_results(format="csv")

# Export to a CSV file
benchmark.export_verification_results_to_file(Path("results.csv"))
```

CSV export handles rubric traits by:

- **Global rubric traits** appear as dedicated columns (e.g., `rubric_safety`, `rubric_clarity`)
- **Question-specific traits** are consolidated into a `question_specific_rubrics` column as JSON

To separate global and question-specific traits correctly, pass the global rubric:

```python
global_rubric = benchmark.get_global_rubric()
csv_str = benchmark.export_verification_results(
    format="csv",
    global_rubric=global_rubric,
)
```

### Filtering Exports

Both methods support filtering by question IDs and run name:

```python
# Export results for specific questions
json_str = benchmark.export_verification_results(
    question_ids=["urn:uuid:question-abc123", "urn:uuid:question-def456"],
    format="json",
)

# Export results from a specific run
csv_str = benchmark.export_verification_results(
    run_name="experiment-2026-02-06",
    format="csv",
)
```

### Export to File with Auto-Detection

When using `export_verification_results_to_file()`, the format is auto-detected from the file extension:

```python
# Auto-detected as JSON
benchmark.export_verification_results_to_file(Path("results.json"))

# Auto-detected as CSV
benchmark.export_verification_results_to_file(Path("results.csv"))

# Explicit format (overrides extension)
benchmark.export_verification_results_to_file(
    Path("results.txt"),
    format="json",
)
```

Supported extensions: `.json` (JSON format), `.csv` (CSV format). Other extensions require an explicit `format` parameter.

---

## DataFrame Export

DataFrames built from the [DataFrame analysis](dataframe-analysis.md) workflow can be exported using standard pandas methods:

### Export to CSV

```python
template_results = results.get_template_results()
df = template_results.to_dataframe()

# Export to CSV
df.to_csv("template_analysis.csv", index=False)
```

### Export to JSON

```python
# Export as JSON records
df.to_json("template_analysis.json", orient="records", indent=2)
```

### Export to Excel

```python
# Requires openpyxl: pip install openpyxl
df.to_excel("template_analysis.xlsx", index=False)
```

### Export Multiple DataFrames

Export template, rubric, and judgment results to separate files:

```python
# Template results
template_df = results.get_template_results().to_dataframe()
template_df.to_csv("template_results.csv", index=False)

# Rubric results
rubric_df = results.get_rubrics_results().to_dataframe()
rubric_df.to_csv("rubric_results.csv", index=False)

# Deep judgment results (if any results have judgment data)
judgment_results = results.get_judgment_results()
if judgment_results.get_results_with_judgment():
    judgment_df = judgment_results.to_dataframe()
    judgment_df.to_csv("judgment_results.csv", index=False)
```

---

## Loading Exported Results

`ResultsIOManager` is the default loader for verification-result exports. It
returns validated `VerificationResult` objects:

```python
from pathlib import Path
from karenina.benchmark import ResultsIOManager

results_dict = ResultsIOManager.load_from_json(Path("results.json"))
print(f"Loaded {len(results_dict)} results")
```

This returns a `dict[str, VerificationResult]` mapping result IDs to result
objects. For multi-gigabyte files, stream validated rows instead of loading the
whole export:

```python
for result in ResultsIOManager.iter_from_json(Path("results.json")):
    print(result.metadata.question_id)
```

Pass `raw=True` only for low-level migration or recovery work that cannot
validate rows against the current `VerificationResult` schema.

For an export that also contains multi-turn scenario executions, load the
complete validated result set. This preserves run metadata and the compact
scenario records in addition to the flat turn results:

```python
result_set = ResultsIOManager.load_result_set_from_json(Path("scenario_results.json"))

scenario_results = result_set.get_scenario_results()
scenario_df = scenario_results.to_dataframe()
turn_df = scenario_results.to_turn_dataframe()
outcome_df = scenario_results.to_outcome_dataframe()
```

`scenario_df` has one row per scenario execution, including status, path,
terminal failure, and dynamic `outcome_*` columns. `turn_df` has one row per
visited node and links each turn to its flat verification result ID.
`outcome_df` has one typed scalar row per outcome criterion. The same accessors
work on the live `VerificationResultSet` returned by
`Benchmark.run_verification()`, so live and stored analyses use one interface.

When an analysis needs a slice of stored structured messages, format it with
Karenina's canonical trace formatter. This retains assistant text, tool calls,
and tool results for a post-hoc rubric:

```python
from karenina.benchmark import format_trace_messages

messages = result.template.trace_messages if result.template else []
post_hoc_text = format_trace_messages(messages)
```

> **Note:**
> Only JSON format can be loaded back. CSV exports are one-way (export only) since CSV cannot represent the full nested result structure.
> 
---

## Choosing an Export Approach

| Scenario | Approach |
|----------|----------|
| Share complete results with a colleague | `export_verification_results_to_file("results.json")` |
| Archive a run for reproducibility | `export_verification_results_to_file("results.json")` |
| Quick spreadsheet analysis | `export_verification_results_to_file("results.csv")` |
| Custom pandas analysis workflow | `df.to_csv(...)` or `df.to_excel(...)` |
| Feed results into another tool | `export_verification_results(format="json")` as string |
| Re-analyze previous results | `ResultsIOManager.load_from_json(Path("results.json"))` |
| Stream a large results export | `ResultsIOManager.iter_from_json(Path("results.json"))` |
| Re-analyze multi-turn scenarios | `ResultsIOManager.load_result_set_from_json(Path("results.json"))` |
| Reload a `ResultsStore` archive | `ResultsStore.from_file("store.json")` |

---

## Next Steps

- [Understand result structure](verification-result.md) — Fields available in each result
- [Analyze with DataFrames](dataframe-analysis.md) — Build and explore DataFrames before exporting
- [Iterate on your benchmark](iterating.md) — Use exports to identify and fix failures
- [Run verification](../running-verification/basic-verification.md) — Generate results to export
- [CLI export](../running-verification/basic-verification.md) — Export directly from the command line with `--output`
