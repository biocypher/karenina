# Open Targets Benchmark Curation

This package implements the benchmark-construction experiment. It imports an
expert-authored question table, calls Claude Opus 4.6 to draft answer templates,
validates their structure, and exports an unfinished benchmark for domain-expert
review.

## Fresh rerun

The required source spreadsheet is not distributed with the repository or data
deposit. Supply an authorized Excel or CSV copy explicitly:

```bash
uv run python -m paper.otp_benchmark_curation.simplified \
  --source /path/to/open_targets_questions.xlsx
```

The complete table is drafted by default. A single-row live check uses the
same code path:

```bash
uv run python -m paper.otp_benchmark_curation.simplified \
  --source /path/to/open_targets_questions.xlsx \
  --limit 1 \
  --output-dir paper/otp_benchmark_curation/out/one-item-check
```

Both commands call Claude Opus 4.6 through Anthropic at temperature zero and
draft templates sequentially. Set `ANTHROPIC_API_KEY` in `.env` or the
environment.

## Inputs

The script expects these columns:

- `id`
- `Area`
- `Subcategories`
- `Question`
- `Answer`
- `Complexity`

## Outputs and review

It writes `draft_benchmark.jsonld`, `template_backup.json`, and
`curation_review.xlsx`. The script refuses to overwrite a non-empty output
directory containing completed or unrelated artifacts so stochastic drafts
remain available for inspection. If an interrupted run left only
`template_backup.json`, rerun the same command and output directory to resume.

Generated templates remain drafts even after structural validation. A domain
expert must open the checkpoint in Karenina's graphical interface, inspect the
fields, judge instructions, and reference values, then mark accepted items
finished. Run `karenina serve --port 8080` to start the interface.

`ResultsIOManager` and TaskEval are not used here because this phase creates a
benchmark. It does not load verification results or evaluate stored responses.

## Experimental method

The reported benchmark contains 144 source rows. For each row, Karenina drafts
a field plan, reference-value specification, and judge-facing instructions.
Progressive backup supports interrupted runs. Drafting is stochastic, and a
generated template remains unfinished until a domain expert approves it.

## Reproducibility limitation

The expert-authored source table is not part of the repository or data archive,
so this workflow can be rerun only with an authorized copy. The package
documents and tests the source-to-draft workflow without substituting a
synthetic or model-generated table for the scientific input. The finalized
benchmark checkpoint consumed by downstream experiments is included in the
data archive.
