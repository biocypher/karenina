# Error Analysis

## What it does

Takes a finished verification run (`VerificationResultSet` plus the corresponding `Benchmark` checkpoint) and materializes it into a navigable directory. Passes and failures are split into per-category buckets; every case gets a markdown file with YAML frontmatter, a rendered trace, parsed fields, rubric scores, and the full failure record when applicable. A `PROMPT.md` is written alongside to orient an analyst agent. Running the agent is optional: the default `prepare-only` launcher just verifies `REPORT.md` exists afterwards, and an opt-in `claude-code` launcher shells out to the Claude Code CLI. Custom launchers can be registered against a small Protocol.

This is a post-run diagnostic, not part of the [verification pipeline](../../core_concepts/verification-pipeline.md).

## When to use it

- Debugging a new benchmark where the pass rate is lower than expected.
- Comparing two runs qualitatively (pattern detection over individual cases).
- Iterating on a rubric or template and looking for recurring failure modes.

When aggregate statistics are enough, use [DataFrame analysis](dataframe-analysis.md) instead.

## Quick start

Prepare the directory without running any agent:

```bash
karenina analyze-errors \
    --results runs/2026-04-16_claude.json \
    --checkpoint benchmarks/legal_qa.json \
    --out-dir ./analysis/
```

Python equivalent:

```python
from pathlib import Path

from karenina.benchmark.error_analysis import analyze_errors

analyze_errors(
    results=Path("runs/2026-04-16_claude.json"),
    checkpoint=Path("benchmarks/legal_qa.json"),
    out_dir=Path("./analysis/"),
)
```

By default the command exits with status 3 because no agent has produced `REPORT.md` yet. Open the directory in Claude Code, Codex, or any other coding agent; point it at `PROMPT.md`; write `REPORT.md`; re-run with `--force` if desired.

## Directory layout

```
analysis/
├── INDEX.md             # stats, breakdown tables, directory map
├── PROMPT.md            # analyst prompt (default or user-provided)
├── REPORT.md            # agent output
├── benchmark/
│   ├── metadata.md
│   ├── questions.jsonl
│   ├── rubric.json
│   └── templates/
├── passes/              # one file per passing case
├── failures/
│   └── <category>/      # one file per failure, bucketed by FailureCategory
└── case_assets/
    └── <case_id>/
        └── traces/artifacts/
```

Links for further reading: [answer templates](../../core_concepts/answer-templates.md), [rubrics](../../core_concepts/rubrics/index.md), [failure categories](../../advanced-pipeline/error-handling.md), [scenarios](../../core_concepts/scenarios/index.md).

## How bucketing works

Each `VerificationResult.failure.category` leaf becomes a folder name (`content`, `parsing`, `recursion_limit`, `trace_validation`, `deep_judgment`, `deep_judgment_rubric`, `abstention`, `sufficiency`, `template_validation`, `timeout`, `connection`, `rate_limit`, `server_error`, `unexpected_error`). Passes go into `passes/`. Scenarios land in the bucket of their first failing turn; the `failed_turn` frontmatter field pinpoints the turn. The `FailureGroup` is surfaced in the INDEX tables, never as a folder.

Everything the agent needs is inside the analysis directory. The agent does not reach the SQLAlchemy database, other runs, or anything outside `out_dir`.

## Running an agent against the directory

Two supported paths.

**Prepare only (default).** `karenina analyze-errors ...` writes the directory and exits. Open it in your agent of choice, feed it `PROMPT.md`, and let it write `REPORT.md`. Good for agents without a CLI entry point.

**Built-in `claude-code` launcher (opt-in).** Requires `claude` on `PATH`.

```bash
karenina analyze-errors ... --launcher claude-code --timeout 1800
```

The launcher runs `claude -p @PROMPT.md --permission-mode acceptEdits` with `cwd=<out-dir>`. A missing `claude` binary raises `LauncherUnavailableError`.

## Customizing the prompt

The default prompt ships inside the package. Substituted placeholders:

- `$BENCHMARK_NAME`
- `$ANSWERING_MODEL` (comma-joined when multiple)
- `$TOTAL`, `$PASSED`, `$FAILED`
- `$FAILURE_CATEGORIES`
- `$RUN_TIMESTAMP` (latest verification timestamp in the result set)

Override by passing `--prompt path/to/custom.md` on the CLI (or `prompt_path=...` in Python). A user prompt that contains no placeholders is copied unchanged.

## Writing a custom launcher

```python
from pathlib import Path
import subprocess

from karenina.benchmark.error_analysis import ErrorAnalystLauncher, register_launcher


class CodexLauncher:
    def run(self, analysis_dir: Path, **kwargs) -> Path:
        subprocess.run(
            ["codex", "run", "--file", "PROMPT.md"],
            cwd=analysis_dir,
            check=True,
        )
        return analysis_dir / "REPORT.md"


register_launcher("codex", CodexLauncher)
```

Contract the launcher must honor: read the materialized files under `analysis_dir`, write `analysis_dir / REPORT.md`, and do not mutate anything outside the directory.

## Trace size handling

Long trace content is offloaded to per-case artifact files via the same mechanism the scenario handover uses. Offloaded chunks land in `case_assets/<case_id>/traces/artifacts/`. The inline trace carries a `[Content offloaded: N chars]` marker followed by a `File: <absolute path>` line (a newline separates the two, and the character count is comma-grouped). Threshold: `KARENINA_TRACE_TRUNCATION_THRESHOLD` environment variable (default 2000 characters). Override per analysis with `--max-trace-chars N`.

## Limits

- No live re-execution of failing cases. The directory is a snapshot.
- One run per analysis directory. Cross-run comparison is out of scope.
- Report quality is the agent's responsibility. karenina only checks that `REPORT.md` exists.
