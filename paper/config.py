"""Shared paths and endpoint defaults for the paper experiments.

Experiment semantics such as model rosters, condition sets, rubric prompts,
and analysis thresholds stay in their owning packages.
"""

from __future__ import annotations

from pathlib import Path

PAPER_DIR = Path(__file__).resolve().parent
REPOSITORY_ROOT = PAPER_DIR.parent
DATA_ROOT_ENV = "KARENINA_PAPER_DATA"
DATA_DEPOSIT_DIRNAME = "karenina-paper-experiments-data"
DEFAULT_DATA_ROOTS = (
    REPOSITORY_ROOT.parent / DATA_DEPOSIT_DIRNAME,
    REPOSITORY_ROOT / DATA_DEPOSIT_DIRNAME,
)

# Open Targets inputs shared across experiment packages.
OTP_BENCHMARK_JSONLD = "paper_examples/QA/qa_benchmark.jsonld"
OTP_PARAMETRIC_RESULTS = "paper_examples/QA/megarun/definitive/qa_megarun_nomcp.json"
OTP_MCP_RESULTS = "paper_examples/QA/megarun/definitive/qa_megarun_mcp.json"
OTP_ADVERSARIAL_BENCHMARK_CSV = "adversarial/data/OT_MCP_benchmark.csv"
OTP_ADVERSARIAL_CSV = "adversarial/data/adversarial_samples.csv"

# Response-characterization archive paths.
RESPONSE_EMPTY_TRAILING_JUDGMENTS = (
    "paper_examples/rubrics/LLM/empty_trailing_ai_characterization/out/"
    "empty_trailing_ai_judgments.jsonl"
)
RESPONSE_GROUNDING_SCORES = (
    "paper_examples/rubrics/LLM/evidence_grounded_answer/out/"
    "mcp_evidence_grounded_deep_judgment_scores.jsonl"
)
RESPONSE_GROUNDING_TRACES = (
    "paper_examples/rubrics/LLM/evidence_grounded_answer/out/"
    "mcp_evidence_grounded_deep_judgment_traces.jsonl"
)
RESPONSE_GROUNDING_PROMPT = (
    "paper_examples/rubrics/LLM/evidence_grounded_answer/evidence_grounded_answer.md"
)

# Citation-audit archive paths.
CITATION_ARCHIVED_SELECTED = (
    "paper_examples/rubrics/agentic/citation_integrity/out/selected.jsonl"
)
CITATION_ARCHIVED_JUDGMENTS = (
    "paper_examples/rubrics/agentic/citation_integrity/out/opus_agentic.jsonl"
)

# Sycophancy archive paths.
SYCOPHANCY_DEFINITIVE_RESULTS = "paper_examples/scenarios/sycophancy/out/definitive"
SYCOPHANCY_ABSTENTION_JUDGMENTS = (
    "paper_examples/scenarios/scenarios_abstention/derived/"
    "autocorrection_abstention_recheck.jsonl"
)
SYCOPHANCY_CAVE_REGEX_JUDGMENTS = (
    "paper_examples/scenarios/sycophancy_checks/derived/"
    "haiku_caved_recheck_longform.jsonl"
)
SYCOPHANCY_CAVE_GROUNDING_JUDGMENTS = (
    "paper_examples/scenarios/sycophancy_checks/derived/"
    "haiku_rechecked_caves_deep_judgment.jsonl"
)

# BixBench archive paths.
BIXBENCH_BENCHMARK_JSONLD = "paper_examples/bix_bench/benchmark/bix_bench.jsonld"
BIXBENCH_ARCHIVED_RUNS = "paper_examples/bix_bench/outputs/runs"
BIXBENCH_ARCHIVED_BURDENS = (
    "paper_examples/bix_bench/outputs/trace_failure_burdens_v5/"
    "trace_failure_burdens_summary.csv"
)

# Default output locations.
RESPONSE_OUTPUT_DIR = PAPER_DIR / "otp_response_characterization" / "out"
MODEL_COMPARISON_OUTPUT_ROOT = PAPER_DIR / "otp_model_comparison" / "out"
MODEL_COMPARISON_SMOKE_OUTPUT_ROOT = MODEL_COMPARISON_OUTPUT_ROOT / "smoke"
CITATION_OUTPUT_DIR = PAPER_DIR / "otp_citation_audit" / "out"
CITATION_SMOKE_OUTPUT_DIR = CITATION_OUTPUT_DIR / "smoke"
ADVERSARIAL_OUTPUT_DIR = PAPER_DIR / "otp_adversarial_generation" / "out"
SYCOPHANCY_OUTPUT_DIR = PAPER_DIR / "otp_sycophancy_scenarios" / "out"
BIXBENCH_OUTPUT_ROOT = PAPER_DIR / "bixbench_harness_comparison" / "out"
BIXBENCH_SIMPLIFIED_OUTPUT_DIR = BIXBENCH_OUTPUT_ROOT / "simplified"
BENCHMARK_CURATION_OUTPUT_DIR = PAPER_DIR / "otp_benchmark_curation" / "out" / "draft"

# Service endpoint defaults. Environment-variable overrides remain documented
# and are applied by the package-specific model builders.
RESPONSE_GPT_OSS_ENDPOINT = "http://codon-gpu-001.ebi.ac.uk:8101"
CITATION_GPT_OSS_ENDPOINT = "http://codon-gpu-001:8101"
SYCOPHANCY_QWEN_ENDPOINT = "http://codon-gpu-001:8000"
SYCOPHANCY_GPT_OSS_ENDPOINT = "http://codon-gpu-001:8101/v1"

MODEL_COMPARISON_ENDPOINTS = {
    "gpt-oss-120b": "http://codon-gpu-001:8101",
    "qwen3.5-a3b": "http://codon-gpu-003:8002",
    "qwen3.6-a3b": "http://codon-gpu-003:8103",
    "qwen3.5-122b-a10b": "http://codon-gpu-001:8000",
}
MODEL_COMPARISON_SIMPLIFIED_ENDPOINT = "http://localhost:8000"

ZAI_ANTHROPIC_ENDPOINT = "https://api.z.ai/api/anthropic"
ZAI_OPENAI_ENDPOINT = "https://api.z.ai/api/coding/paas/v4"
BIXBENCH_QWEN_ENDPOINTS = (
    "http://codon-gpu-001:8000",
    "http://codon-gpu-003:8000",
    "http://hl-codon-gpu-020:8000",
)
