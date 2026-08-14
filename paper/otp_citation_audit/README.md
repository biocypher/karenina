# Open Targets Citation Integrity Audit

This package screens Claude answers for explicit published-paper citations,
selects a balanced 72-trace panel across model, regime, and outcome, and asks a
web-enabled Claude agent to verify every citation.

The operational taxonomy and search contract are documented in
[`citation_integrity.md`](citation_integrity.md).

## Fresh rerun

```bash
uv run python -m paper.otp_citation_audit.run
```

This command calls the configured GPT-OSS screener and the Claude Opus
web-search judge. Fresh judgments are stochastic and may differ from the
classifications reported in the paper.

## Archive-based reproduction

The offline command is:

```bash
uv run python -m paper.otp_citation_audit.run --reuse-stored-judgments
```

It loads the selected cohort and judgments from the data deposit and makes no
model calls.

The cohort includes at most 10 citations per answer, then samples six pass and
six fail traces per model and regime when the source pool supports that
balance.

## Smoke check and minimal example

```bash
uv run python -m paper.otp_citation_audit.smoke --limit 5
uv run python -m paper.otp_citation_audit.smoke --limit 5 --screen-only
uv run python -m paper.otp_citation_audit.simplified --limit 5
```

The smoke command checks both live stages. `--screen-only` isolates the
GPT-OSS screen when the separate Claude web-audit dependency is unavailable.
It still writes the screened and selected records for inspection.

The minimal example demonstrates stored-result loading and TaskEval-backed
rubric evaluation. It omits balanced sampling, manifests, stored judgments,
and complete analysis tables.

## Inputs

The standard sibling data deposit is discovered automatically. A nonstandard
location can be selected with `KARENINA_PAPER_DATA`; it must contain:

- `paper_examples/QA/megarun/definitive/qa_megarun_nomcp.json`
- `paper_examples/QA/megarun/definitive/qa_megarun_mcp.json`
- `paper_examples/rubrics/agentic/citation_integrity/out/selected.jsonl`
- `paper_examples/rubrics/agentic/citation_integrity/out/opus_agentic.jsonl`

Fresh execution uses `KARENINA_PAPER_GPT_OSS_URL` for screening and Anthropic
credentials from `.env` for the web-enabled audit agent.

## Outputs

The run writes screening, selection, and citation-judgment JSONL files. The
`analysis/` directory contains condition summaries, aggregate metrics, and a
curation table.
