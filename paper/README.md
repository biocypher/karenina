# Karenina Paper Experiments

Reproduction packages for the experiments reported in the Karenina paper.
This folder is repo-only: it ships in neither wheels nor sdists.

## Experiments

| Package | Paper anchor | Status |
|---|---|---|
| `otp_benchmark_curation` | Building the 144-question Open Targets benchmark | planned (Phase 5) |
| `otp_model_comparison` | Multi-model QA evaluation, MCP and parametric arms | planned (Phase 2) |
| `otp_citation_audit` | Citation-integrity panel | planned (Phase 2) |
| `otp_response_characterization` | Failure tree, grounding, trace analyses | planned (Phase 1) |
| `otp_adversarial_generation` | Hard and easy wrong answers via Claude with OTP MCP | planned (Phase 3) |
| `otp_sycophancy_scenarios` | Multi-turn challenge experiments and rechecks | planned (Phase 3) |
| `bixbench_harness_comparison` | Agentic BixBench across harnesses and scoring levels | planned (Phase 4) |

## Conventions

Every experiment package has the same shape:

- `README.md`: which paper claims it backs, prerequisites, exact commands.
- `config.py`: experiment configuration (models, judges, arms, endpoints).
- `run.py`: fresh execution entry point with a `--smoke` slice mode.
- `rederive.py`: replay path from raw traces to the paper's derived numbers.
- `expected/`: small committed copies of the published numbers.
- `tests/`: offline test asserting `rederive` output matches `expected/`.

Run everything from the karenina repo root:

```bash
uv run python -m paper.<experiment>.run --smoke
uv run python -m paper.<experiment>.rederive
uv run pytest paper/ -m paper_replay
```

## Data

Raw archives are never committed. Set `KARENINA_PAPER_DATA` to the data
root. Each experiment README lists the archive members it needs. Replay
tests skip cleanly when the variable is unset.

The replay path consumes stored LLM judgments and never re-runs judges.
Fresh judging is exercised only by `run.py --smoke`, which needs live
endpoints and credentials from `.env` (never committed).

## Provenance

The original, as-run experiment code lives in the paper monorepo under
`paper_examples/`, `adversarial/`, and `paper_artifacts/` and is not
modified by this folder. Each experiment README records the karenina
version the original runs used.
