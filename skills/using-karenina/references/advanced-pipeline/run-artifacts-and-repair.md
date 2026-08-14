# Stored Results, Run Directories, and Repair

Karenina provides public workflows for loading completed result exports,
organizing new runs, evaluating saved responses, and repairing selected rows.
These APIs keep analysis code on the same validated pipeline used during normal
verification.

## Load stored results

Use `ResultsIOManager` as the default loader for Karenina result exports:

```python
from pathlib import Path

from karenina.benchmark import ResultsIOManager

rows = ResultsIOManager.iter_from_json(Path("results.json"))
for result in rows:
    print(result.metadata.question_id, result.metadata.failure)
```

`iter_from_json()` validates each row as a `VerificationResult` and streams
large exports without loading the complete file. Its `raw=True` option is a
low-level migration and recovery facility. Normal analysis code should use the
validated default.

Scenario exports should be loaded as a complete result set so their compact
execution records and run metadata remain available:

```python
result_set = ResultsIOManager.load_result_set_from_json(Path("scenario_results.json"))
scenarios = result_set.get_scenario_results()

scenario_df = scenarios.to_dataframe()
turn_df = scenarios.to_turn_dataframe()
outcome_df = scenarios.to_outcome_dataframe()
```

This is the standard stored-scenario path. It validates flat results and
scenario records together instead of requiring application code to parse the
JSON wrapper.

Use `karenina.benchmark.format_trace_messages` when a post-hoc analysis needs
to format all or part of a stored `template.trace_messages` list. The public
helper retains assistant text, tool calls, and tool results in Karenina's
canonical trace format.

To evaluate a new rubric over saved responses, use the public TaskEval-backed
facade:

```python
from karenina.benchmark import evaluate_rubric_on_results

judgments = evaluate_rubric_on_results(
    ResultsIOManager.iter_from_json(Path("results.json")),
    rubric,
    parsing_model,
)
```

This path does not regenerate answers. Model-backed rubric traits still call
their configured judge.

## Timestamped run directories

The CLI can create a standard run directory automatically:

```bash
karenina verify benchmark.jsonld \
  --preset comparison.json \
  --run-root runs \
  --run-label otp_parametric
```

The directory name contains a UTC timestamp. It contains `results.json`,
`run_manifest.json`, `traces/`, and `workspaces/`. The manifest records model
identities, the benchmark source, selected question count, configuration,
paths, timestamps, and lifecycle status. Credential fields are masked.

Python callers can use `create_run_directory()` or
`managed_run_directory()` from `karenina.benchmark`. The managed context marks
the manifest completed, interrupted, or failed when the context exits. Pass
`resume=True` with the same label and creation timestamp to reopen an existing
directory explicitly. Accidental reuse raises `FileExistsError`.

## Repair selected rows

`karenina repair` loads a completed JSON export with `ResultsIOManager`, selects
exact row identities, reruns them through the ordinary benchmark pipeline, and
splices only those replacements into the export.

Start with a dry run:

```bash
karenina repair results.json \
  --benchmark benchmark.jsonld \
  --preset comparison.json \
  --failure-group retry \
  --dry-run
```

Then run the repair:

```bash
karenina repair results.json \
  --benchmark benchmark.jsonld \
  --preset comparison.json \
  --failure-group retry \
  --mode replay
```

Replay mode reuses the selected answer traces and reruns downstream parsing and
verification. It uses strict replay, so a missing trace raises instead of
silently calling an answerer. Live mode regenerates the selected answers.

An in-place repair writes a timestamped backup before replacing the export and
writes a JSON provenance sidecar containing the source checksum and replaced
row identities. At least one filter is required unless `--all` is explicit.
