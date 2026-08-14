# Open Targets Model Comparison

This package implements the multi-model Open Targets question-answering
experiment. It compares parametric models with the same models using Open
Targets MCP tools. The design contains seven answerers, seven parsers, 144
questions, and three replicates in each regime.

## Fresh rerun

From the Karenina repository root:

```bash
uv run python -m paper.otp_model_comparison.run
```

This command executes both arms and calls every configured answerer and parser.
For the MCP arm, it starts one managed Open Targets MCP server per answerer.

For operational recovery, `--arm parametric` or `--arm mcp` runs and analyzes
only one arm. The default remains `--arm both`.

## Archive-based reproduction

To regenerate the deterministic analysis tables from the reported-run exports:

```bash
uv run python -m paper.otp_model_comparison.run --reuse-stored-results
```

This command is offline. It loads both exports through `ResultsIOManager`.

## Smoke check and minimal example

```bash
uv run python -m paper.otp_model_comparison.smoke --limit 2 --model gpt-oss-120b
uv run python -m paper.otp_model_comparison.simplified \
  --mcp-url http://127.0.0.1:8765/mcp
```

The minimal example uses one already-running MCP server and a small question
slice to demonstrate both experiment arms. The caller supplies the MCP URL so
the example focuses on model and verification configuration rather than server
process management.

## Inputs and configuration

The standard sibling data deposit is discovered automatically. A nonstandard
location can be selected with `KARENINA_PAPER_DATA`; it must contain:

- `paper_examples/QA/qa_benchmark.jsonld`
- `paper_examples/QA/megarun/definitive/qa_megarun_nomcp.json`
- `paper_examples/QA/megarun/definitive/qa_megarun_mcp.json`

The endpoint defaults identify the Codon deployment used for the reported
runs. Override them with `KARENINA_PAPER_<MODEL>_URL`. Set
`KARENINA_PAPER_OTP_MCP_SOURCE` to a local Open Targets MCP checkout, or leave
it unset to launch the released server through `uvx`. Credentials are read
from `.env` and are never stored in run manifests.

## Outputs

Each run writes its result exports below `out/runs/`, deterministic TSV tables
below `out/analysis/`, and configuration metadata in `out/run_manifest.json`.
Fresh outputs are stochastic and are reported as a new run rather than compared
row by row with the manuscript values.
