# Open Targets Adversarial Alternative Generation

This package generates contradictory alternatives one item at a time with the
Claude Code CLI. Each non-binary item starts an independent Claude Opus 4.6
session with a one-million-token context and Open Targets MCP access. The
runner captures stream-JSON traces, validates the required sample format, and
writes per-item artifacts and composition summaries.

Alternative generation is a standalone preprocessing step. Karenina consumes
curator-approved alternatives in the downstream sycophancy scenario
experiment. Every fresh model-generated pair remains a draft until a domain
curator approves it.

## Fresh rerun

From the Karenina repository root:

```bash
uv run python -m paper.otp_adversarial_generation.run
```

The command processes all 144 source items. The 43 binary questions are
deterministic flips and make no model call. Each of the 101 non-binary
questions starts an independent Claude Code session using
`claude-opus-4-6[1m]` and the `open-targets` MCP server configured in Claude
Code. No second parser model is called.

## Archive-based reproduction

```bash
uv run python -m paper.otp_adversarial_generation.run --reuse-stored-samples
```

This command is fully offline. It validates the manually approved archive and
writes composition summaries without calling a model.

## Smoke check and minimal example

```bash
uv run python -m paper.otp_adversarial_generation.smoke
uv run python -m paper.otp_adversarial_generation.simplified
```

`smoke.py` is a one-item live Claude Code check. `simplified.py` is a minimal,
commented example of the standalone generation workflow.

## Inputs

The standard sibling data deposit is discovered automatically. Set
`KARENINA_PAPER_DATA` for another location. The workflows consume:

- `adversarial/data/OT_MCP_benchmark.csv`: 144 source questions and generation routes;
- `adversarial/data/adversarial_samples.csv`: manually approved hard and easy alternatives.

Per-item traces from the reported generation run are not runtime inputs and are
therefore not part of the downloadable experiment data.

## Outputs

Fresh output contains per-item `adversarial.txt`, `trace_raw.jsonl`, and
`trace.md` files, plus a validated draft CSV, composition table, and run
manifest. Archive rows are labeled `curator_approved_archive`; fresh rows are
always labeled `draft_requires_curator`.

## Environment and execution policy

- `KARENINA_PAPER_DATA`: optional override for the data-deposit location;
- Claude Code authentication and a `claude` executable on `PATH`;
- a connected Claude Code MCP server named `open-targets`.

The runner validates the MCP connection from each session's initialization
trace. Fresh generation uses Claude Code 2.1.85, `bypassPermissions`, a
600-second per-session timeout, two retries after the first attempt, a
30-second retry delay, and a 5-second inter-item delay. The command includes
`--verbose`, `--output-format stream-json`, `--dangerously-skip-permissions`,
and `-p`. Run it only in the dedicated output directory documented here.

The approved CSV is analysis input only. Fresh generation never loads it
implicitly, and no generated draft is treated as curator-approved.
