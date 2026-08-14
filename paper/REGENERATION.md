# Reproducing and Rerunning the Paper Experiments

Archive-based commands regenerate analyses from the artifacts of the reported
runs. Fresh commands rerun the experimental protocol and make model calls.
Because LLM outputs are stochastic, fresh reruns produce new result values with
the documented experiment design and output schemas.

Extract `karenina-paper-experiments-data` beside the cloned `karenina`
repository. The scripts discover it automatically; set `KARENINA_PAPER_DATA`
only when using another location. Each package README lists its additional
endpoints, credentials, tools, and source files.

Shared software, service, credential, storage, and container requirements are
defined in [`DEPENDENCIES.md`](DEPENDENCIES.md). Experiment modules can run
natively with `uv run python -m` or through the Docker controller.

## Command matrix

| Experiment | Archive-based reproduction | Fresh rerun |
|---|---|---|
| Response characterization | `python -m paper.otp_response_characterization.run --reuse-stored-judgments` | Reruns both model-backed judgments and all deterministic analyses from the archived QA exports |
| Model comparison | `python -m paper.otp_model_comparison.run --reuse-stored-results` | Reruns both benchmark arms, all answerer/parser cells and replicates, and the analysis tables |
| Citation audit | `python -m paper.otp_citation_audit.run --reuse-stored-judgments` | Reruns citation screening, balanced selection, web audit, and analysis tables from the archived QA exports |
| Adversarial generation | `python -m paper.otp_adversarial_generation.run --reuse-stored-samples` | Regenerates binary alternatives and Claude Code/MCP drafts for every non-binary item |
| Sycophancy scenarios | `python -m paper.otp_sycophancy_scenarios.run --reuse-stored-judgments` | Reruns scenario checkpoints, all crossed cells, sidecar judgments, and analysis tables |
| BixBench comparison | `python -m paper.bixbench_harness_comparison.run --reuse-stored-judgments` | Reruns every selected task, model/harness/replicate cell, parser, burden judgment, and analysis table |
| Benchmark curation | Not available without the expert-authored source table | `python -m paper.otp_benchmark_curation.simplified --source /path/to/questions.xlsx` drafts every row, validates templates, and exports review files |

Prefix the commands above with `uv run`, as shown in the complete fresh command
block below.

`--reuse-stored-results` is also available for the sycophancy and BixBench
packages as a partial rerun mode. It skips upstream answer generation but still
calls the configured downstream judges. Use `--reuse-stored-judgments` for the
fully offline path in those packages.

## Complete fresh commands

Run from the Karenina repository root:

```bash
uv run python -m paper.otp_response_characterization.run
uv run python -m paper.otp_model_comparison.run
uv run python -m paper.otp_citation_audit.run
uv run python -m paper.otp_adversarial_generation.run
uv run python -m paper.otp_sycophancy_scenarios.run
uv run python -m paper.bixbench_harness_comparison.run
uv run python -m paper.otp_benchmark_curation.simplified \
  --source /path/to/open_targets_questions.xlsx
```

Use a new `--output-dir` or `--output-root` to retain multiple runs. The
response-characterization command accepts `--output-dir`; selecting no analysis
names runs all analyses. Managed benchmark runs save progressive results and
manifests below their output roots.

## Experimental boundaries

- Response characterization and citation auditing are post-hoc experiments.
  Their fresh commands rerun rubrics over archived QA answers.
- The first sycophancy answer is replayed from the first eligible replicate of
  the source QA run. Challenge turns, parsing, guardrail evaluation, and
  sidecars are executed during the scenario run.
- Adversarial generation produces drafts that require human curation. A fresh
  draft is never labeled as curator-approved automatically.
- Benchmark curation produces unfinished items and a review workbook. Human
  approval remains outside the generation command.

Smoke commands and minimal examples are operational checks, not substitutes
for the full experiment commands. Benchmark curation is the only package whose
complete workflow is implemented in `simplified.py`.
