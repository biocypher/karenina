# Exporting Results

After running verification and analyzing results, you'll often want to export data for sharing, archival, or external analysis. Karenina provides two complementary export approaches:

- **Benchmark export methods** — Export complete verification results as JSON or CSV strings/files
- **DataFrame export** — Export pandas DataFrames to any format pandas supports

---

## Benchmark Export Methods

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

# Deep judgment results (if available)
judgment_results = results.get_judgment_results()
if judgment_results is not None:
    judgment_df = judgment_results.to_dataframe()
    judgment_df.to_csv("judgment_results.csv", index=False)
```

---

## Loading Exported Results

Previously exported JSON results can be loaded back into a benchmark:

```python
from pathlib import Path

results_dict = benchmark.load_verification_results_from_file(
    Path("results.json"),
    run_name="imported-run",  # optional: assign a run name
)
print(f"Loaded {len(results_dict)} results")
```

This returns a `dict[str, VerificationResult]` mapping result IDs to result objects. The loaded results are stored in the benchmark's in-memory results store and can be analyzed using the same DataFrame builders and filtering methods.

!!! note
    Only JSON format can be loaded back. CSV exports are one-way (export only) since CSV cannot represent the full nested result structure.

---

## Choosing an Export Approach

| Scenario | Approach |
|----------|----------|
| Share complete results with a colleague | `export_verification_results_to_file("results.json")` |
| Archive a run for reproducibility | `export_verification_results_to_file("results.json")` |
| Quick spreadsheet analysis | `export_verification_results_to_file("results.csv")` |
| Custom pandas analysis workflow | `df.to_csv(...)` or `df.to_excel(...)` |
| Feed results into another tool | `export_verification_results(format="json")` as string |
| Re-analyze previous results | `load_verification_results_from_file("results.json")` |

---

## Next Steps

- [Understand result structure](verification-result.md) — Fields available in each result
- [Analyze with DataFrames](dataframe-analysis.md) — Build and explore DataFrames before exporting
- [Iterate on your benchmark](iterating.md) — Use exports to identify and fix failures
- [Run verification](../06-running-verification/python-api.md) — Generate results to export
- [CLI export](../06-running-verification/cli.md) — Export directly from the command line with `--output`
