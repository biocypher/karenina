# Open Targets Response Characterization

This package implements the response-characterization analyses for the Open
Targets Platform QA experiment: response failures, tool use, evidence
grounding, trace length, and answer-generation tokens.

## Inputs

The standard sibling data deposit is discovered automatically. A nonstandard
location can be selected with `KARENINA_PAPER_DATA`; it must contain:

- `paper_examples/QA/megarun/definitive/qa_megarun_mcp.json`
- `paper_examples/QA/megarun/definitive/qa_megarun_nomcp.json`
- `paper_examples/QA/qa_benchmark.jsonld`
- `paper_examples/rubrics/LLM/empty_trailing_ai_characterization/out/empty_trailing_ai_judgments.jsonl`, only for `--reuse-stored-judgments`
- `paper_examples/rubrics/LLM/evidence_grounded_answer/out/mcp_evidence_grounded_deep_judgment_scores.jsonl`, only for `--reuse-stored-judgments`
- `paper_examples/rubrics/LLM/evidence_grounded_answer/out/mcp_evidence_grounded_deep_judgment_traces.jsonl`, only for `--reuse-stored-judgments`

The package reads QA exports and optional stored judgments from the data
deposit and writes generated tables to `out/`.

## Minimal example

`analyses/simplified.py` demonstrates the central Karenina flow: load
archived results with `ResultsIOManager`, score response shapes with a regex
rubric (offline), classify blank-final traces with a literal LLM rubric, and
judge evidence grounding with a boolean LLM rubric. Run it with:

```bash
uv run python -m paper.otp_response_characterization.analyses.simplified [N]
```

The optional N limits how many traces each LLM step judges (default 5).

## Fresh rerun

Run all analyses from the karenina repository root:

```bash
uv run python -m paper.otp_response_characterization.run
```

To retain a separate reconstruction instead of writing to the default output
tree, pass `--output-dir /path/to/new/output`. With no analysis names, the
command runs all five analyses.

Pass one or more analysis names to run a subset:

```bash
uv run python -m paper.otp_response_characterization.run failure_tree trace_tokens
```

The default full analysis reruns the missing-final-message and
evidence-grounding rubrics from the archived QA results, so it calls the
configured GPT-OSS endpoint. Evidence grounding uses a two-call
deep-judgment flow, schema masking, 120,000-token input gate, and 16,384-token
output cap per call. The input gate counts the complete rendered first-stage
prompt through the judge endpoint's chat-template-aware tokenizer.
Verification results load through `ResultsIOManager`. All rubrics use the
TaskEval-backed post-hoc evaluation path.

Fresh LLM outputs vary across runs and are recorded as new judgments.

## Archive-based reproduction

For an offline analysis of the stored stochastic judgments:

```bash
uv run python -m paper.otp_response_characterization.run --reuse-stored-judgments
```

## Experimental method

The analyses use the following cohort and evaluation rules:

- Missing final message: select `EmptyTrailingAI`, deduplicate by question,
  answerer identity, replicate, and trace hash, inject the benchmark question
  and reference answer, then run the three-class GPT-OSS 120B literal rubric
  at temperature 0 with one worker.
- Evidence grounding: select rows whose saved correctness verdict is true and
  whose answerer had tools configured, deduplicate by question, answerer, and
  replicate, mask GraphQL schema payloads, count the complete first-stage
  prompt with the judge endpoint tokenizer, skip prompts over 120,000 tokens,
  and run reasoning plus Boolean extraction with GPT-OSS 120B at temperature 0.
  Each call has a 16,384-token output cap and the full run uses 32 workers.
- Evidence panel construction keeps null verdicts unscored, applies response
  shape and tool-less exclusions before the scored residual cohort, and uses
  the specified peer-family and Maraviroc evidence rules.

Both model-backed rubrics run through `evaluate_rubric_on_results`, backed by
TaskEval.

## Smoke checks

The smoke checks run the model-backed rubric workflows on a small slice:

```bash
uv run python -m paper.otp_response_characterization.smoke missing_final_message
uv run python -m paper.otp_response_characterization.smoke evidence_grounding
```

Configure the endpoint with `KARENINA_PAPER_GPT_OSS_URL` and optionally
`KARENINA_PAPER_GPT_OSS_KEY`. Fresh outcomes are illustrative because LLM
judgments are stochastic.

## Tests

Offline tests use synthetic inputs and validate schemas, joins, partitions,
and arithmetic invariants:

```bash
uv run pytest paper/otp_response_characterization/tests/ paper/tests/ -q
```

Archive-backed consistency tests carry the `paper_archive` marker and skip
only when no standard deposit or explicit override is available.
