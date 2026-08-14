# BixBench Harness Comparison

This package implements the comparison of three scientific agents under the
Claude Code and DeepAgents harnesses. Each of 53 grouped BixBench tasks has a
data workspace and one or more verified answer fields, for 199 fields in total.
The experiment uses three replicates per model and harness condition.

`run.py` is the complete experiment entry point. Without a reuse option it
starts from the benchmark workspaces, calls the selected answerers and the
GLM-5.1 parser, and calls GLM-5.1 again for the six-axis trace
failure-burden rubric. LLM outputs are stochastic, so each fresh execution is
reported as a new run.

## Fresh rerun

From the Karenina repository root:

```bash
# Complete three-model, two-harness, three-replicate experiment.
uv run python -m paper.bixbench_harness_comparison.run

# GLM-5.1 comparison through a Z.ai subscription, without Codon HPC.
uv run python -m paper.bixbench_harness_comparison.run --model glm

# One-task GLM subscription check through both harnesses.
uv run python -m paper.bixbench_harness_comparison.smoke

# One-task Claude Opus subscription check through Claude Code.
uv run python -m paper.bixbench_harness_comparison.smoke \
  --model opus --harness claude-code
```

The GLM-only command requires `ZAI_API_KEY` in `.env`. It uses Z.ai's
Anthropic-compatible route for Claude Code and its OpenAI-compatible coding
route for DeepAgents and failure-burden judging. It does not use a Codon
endpoint. The complete matrix additionally requires a reachable Qwen endpoint,
Claude Code subscription authentication for the Opus Claude Code cell, and
`ANTHROPIC_API_KEY` for the Opus DeepAgents cell.

Docker is the default local runtime. Set the two image variables when the image
names differ from the defaults:

```bash
export KARENINA_BIX_CLAUDE_CONTAINER_IMAGE=karenina-bixbench-claude:latest
export KARENINA_BIX_DEEPAGENTS_CONTAINER_IMAGE=karenina-bixbench:latest
```

Build the default images with:

```bash
paper/docker/compose.sh build-agent-images
```

The fully containerized controller smoke is:

```bash
paper/docker/compose.sh run-agentic \
  paper.bixbench_harness_comparison.smoke
```

The controller mounts the host Docker socket only for `run-agentic`. Read the
[dependency contract](../DEPENDENCIES.md) for the trust boundary, same-path
workspace requirement, image pins, and storage estimate.

`--runtime singularity` and `--runtime apptainer` accept local `.sif` paths
through the same image variables. `--runtime host` is available for trusted
workspaces, but it does not isolate agent-written code from the host.

## Archive-based reproduction and partial reruns

The standard sibling data deposit is discovered automatically. For another
location, set `KARENINA_PAPER_DATA`, then run:

```bash
# Reuse answerer/parser exports and rerun the GLM burden rubric.
uv run python -m paper.bixbench_harness_comparison.run \
  --reuse-stored-results

# Reproduce the analysis offline from stored results and judgments.
uv run python -m paper.bixbench_harness_comparison.run \
  --reuse-stored-judgments
```

The required members are:

- `paper_examples/bix_bench/benchmark/bix_bench.jsonld`
- `paper_examples/bix_bench/benchmark/workspaces/`
- `paper_examples/bix_bench/outputs/runs/`
- `paper_examples/bix_bench/outputs/trace_failure_burdens_v5/trace_failure_burdens_summary.csv`

Stored exports are loaded through `ResultsIOManager`. Fresh post-hoc judgments
use `evaluate_rubric_on_results`, backed by TaskEval. The package does not read
trace sidecar files directly.

## Minimal example

The minimal example demonstrates one complete task without the full experiment
matrix or run-management machinery. By default it runs bix-18 with GLM-5.1
through Claude Code, parses and scores the answer, and sends the trace to
GLM-5.1 for failure-burden evaluation through TaskEval. It writes
`results.json`, captured workspace files, and `failure_burdens.json` under
`out/simplified/`:

```bash
uv run python -m paper.bixbench_harness_comparison.simplified
```

The script refuses to overwrite a non-empty output directory. Select another
`--output-dir` to retain multiple live runs.

The example can also load an existing export. It still reruns the burden judge
unless `--summary-only` is supplied:

```bash
uv run python -m paper.bixbench_harness_comparison.simplified \
  --results /path/to/results.json --limit 5
```

## Outputs

`out/analysis/` contains per-field scores, per-task outcomes, cell summaries,
per-trace failure burdens, and burden summaries as TSV files. The run root also
contains result exports, retained workspaces, and a manifest.

## Experimental design

The design is a 3 model by 2 harness by 3 replicate matrix over 53 grouped tasks
and 199 graded or Boolean fields. Opus uses high effort. The GLM-5.1
DeepAgents parser and judge have read-only access and a 20,000-byte file-read
cap. Runs use dynamic agentic parsing, full workspace retention, 900-second
Qwen and GLM timeouts, a 2,400-second Opus timeout, partial timeout scoring, a
connection-only retry policy, and the six-axis v5 burden rubric with a
600-second judge timeout.

The reported runs used Singularity on EBI Codon. The runtime setting also
supports Apptainer, local Docker, and an explicit trusted-host mode.
