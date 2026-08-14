# Karenina Paper Reproducibility Suite

This directory contains the executable methods, analysis code, and validation
checks for the experiments reported in the Karenina paper. It is maintained as
part of the repository and is not included in wheels or source distributions.

## Experiments

| Package | Experiment | Available workflows |
|---|---|---|
| `otp_benchmark_curation` | Construction of the 144-question Open Targets benchmark | Fresh drafting and review export; requires the expert-authored source table |
| `otp_model_comparison` | Multi-model QA evaluation with parametric and Open Targets MCP arms | Archived-result reproduction and fresh rerun |
| `otp_citation_audit` | Citation-integrity screening and web audit | Archived-judgment reproduction and fresh rerun |
| `otp_response_characterization` | Failure tree, grounding, and trace analyses | Archived-judgment reproduction and fresh rerun |
| `otp_adversarial_generation` | Hard and easy contradictory-answer generation | Approved-archive analysis and fresh draft generation |
| `otp_sycophancy_scenarios` | Multi-turn challenge experiments and sidecar audits | Archived-judgment reproduction and fresh rerun |
| `bixbench_harness_comparison` | BixBench evaluation across models and agent harnesses | Archived-judgment reproduction and fresh rerun |

Each package README defines its experimental design, required inputs, commands,
and outputs.

## Reproducibility model

The suite distinguishes two operations:

- **Archive-based reproduction** regenerates deterministic analyses from the
  outputs and, where applicable, stored stochastic judgments from the reported
  runs. These commands do not call models when the package's complete reuse
  option is selected.
- **Fresh rerun** executes the experimental protocol with configured model and
  tool services. Model outputs and model-based judgments are stochastic, so a
  rerun follows the experiment design and produces the documented output
  schemas rather than the exact reported values.

Smoke commands exercise a small slice of a fresh rerun. Minimal examples in
`simplified.py` or `analyses/simplified.py` demonstrate the central workflow
without the complete experiment matrix or run-management machinery.

Most experiment packages contain:

- `README.md`: method, prerequisites, commands, and outputs;
- `config.py`: model rosters, conditions, thresholds, and constructors;
- `run.py`: complete experiment entry point;
- `smoke.py`: small live-dependency check;
- `simplified.py`: minimal workflow example; and
- `tests/`: schema, invariant, error-path, and synthetic-data tests.

Shared archive paths, endpoint defaults, and output paths are defined in
`paper/config.py`. Stored Karenina exports are loaded with
`karenina.benchmark.ResultsIOManager`, and post-hoc rubrics are evaluated with
the TaskEval-backed `evaluate_rubric_on_results` interface.

Adversarial alternative generation has a different execution boundary. It
uses independent Claude Code sessions with Open Targets MCP and produces draft
alternatives. Karenina consumes curator-approved alternatives in the downstream
sycophancy scenario experiment.

Run commands from the Karenina repository root:

```bash
uv run python -m paper.<experiment>.run
uv run python -m paper.<experiment>.smoke
uv run pytest paper/ -m paper_archive
```

See [REGENERATION.md](REGENERATION.md) for the complete command matrix and
[DEPENDENCIES.md](DEPENDENCIES.md) for environment requirements.

## Data deposit

The experiment archive is distributed separately from the source repository;
its versioned record is identified in the paper's data-availability statement.
Extract it as `karenina-paper-experiments-data` beside the cloned repository:

```text
workspace/
├── karenina/
└── karenina-paper-experiments-data/
```

All entry points discover this layout automatically. Extraction inside the
repository is also supported. Set
`KARENINA_PAPER_DATA=/path/to/karenina-paper-experiments-data` only for another
location. Each experiment README lists the archive members it consumes.
Archive-backed tests skip when neither a standard location nor the environment
override is available.

The archive includes `MANIFEST.sha256`, which records a SHA-256 checksum for
each payload and documentation file. Verify an extracted copy from the archive
root with:

```bash
shasum -a 256 -c MANIFEST.sha256
```

Live commands read credentials from the repository-root `.env` file or the
process environment. Credentials are never committed or written to manifests.

## Docker

The controller and two BixBench agent sandboxes are defined under
[`paper/docker`](docker). Build the controller and run an archive-based
reproduction with:

```bash
paper/docker/compose.sh build
paper/docker/compose.sh run \
  paper.otp_model_comparison.run \
  --reuse-stored-results
```

Build all three images for amd64, including from an arm64 host, with:

```bash
paper/docker/compose.sh build-amd64
paper/docker/compose.sh build-agent-images-amd64
```

The wrapper mounts the portable data layout and assigns new output files to the
invoking user. Fresh BixBench runs and standalone Claude Code generation use
separate opt-in commands because they require Docker-daemon access or a mounted
Claude configuration. Review the dependency contract before enabling either
boundary.

## Benchmark curation source

Benchmark curation requires an authorized copy of the expert-authored source
table, which is not included in the repository or data archive. The curation
package documents and tests the source-to-draft workflow, but it is not part of
the archive-based reproduction path. The finalized benchmark checkpoint used
by all downstream experiments is included in the data archive.
