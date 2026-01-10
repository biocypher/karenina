"""
Verbose Real GEPA Optimization on AIME

This script demonstrates a real GEPA optimization loop with:
- Verbose logging for tracking optimization progress
- LLM-generated feedback for richer diagnostic information
- Differential analysis comparing successful vs failed responses
- Multi-model evaluation using both Haiku and Sonnet

Extracted from notebook: 11_verbose_real_optimization.ipynb (up to Step 9)
"""

import sys
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

# Ensure src is in path
sys.path.insert(0, str(Path.cwd().parent.parent.parent / "src"))

# Core Karenina imports
from karenina import Benchmark

# GEPA integration imports
from karenina.integrations.gepa import (
    GEPA_AVAILABLE,
    KareninaAdapter,
    ObjectiveConfig,
    OptimizationTarget,
    SimpleLogger,
    split_benchmark,
)
from karenina.schemas import ModelConfig, VerificationConfig

print(f"GEPA available: {GEPA_AVAILABLE}")

if not GEPA_AVAILABLE:
    raise ImportError("GEPA is required for this script. Install with: pip install gepa")

# =============================================================================
# Step 2: Load the AIME Benchmark
# =============================================================================

benchmark_path = Path.home() / "Projects/karenina-monorepo/local_data/data/checkpoints/aime_2025.jsonld"
benchmark = Benchmark.load(benchmark_path)

print(f"Benchmark: {benchmark.name}")
print(f"Description: {benchmark.description}")
print(f"Total questions: {len(benchmark.get_question_ids())}")

# Explore a sample question
question_ids = benchmark.get_question_ids()
sample_q = benchmark.get_question(question_ids[0])

print("Sample AIME problem:")
print(f"  ID: {question_ids[0]}")
print(f"  Question: {sample_q['question'][:150]}...")
print(f"  Answer: {sample_q['raw_answer']}")

# =============================================================================
# Step 3: Configure the Optimization
# =============================================================================

# Define the seed prompt (starting point for optimization)
SEED_PROMPT = """You are a helpful math assistant.
Solve the problem step by step and provide the final answer."""

# Optimization parameters
MAX_METRIC_CALLS = 100  # Total evaluation budget; production would use 500-1000
TRAIN_RATIO = 0.8
VAL_RATIO = 0.2
SPLIT_SEED = 42

# Model configuration
REFLECTION_MODEL = "anthropic/claude-sonnet-4-5"  # Sonnet for reflection
ANSWERING_MODELS = [
    ("claude-haiku", "claude-haiku-4-5"),
    ("claude-sonnet", "claude-sonnet-4-5"),
]
PARSING_MODEL = "claude-haiku-4-5"  # Haiku for parsing (fast)

print("Optimization Configuration:")
print(f"  Seed prompt: {SEED_PROMPT[:50]}...")
print(f"  Max metric calls: {MAX_METRIC_CALLS}")
print(f"  Reflection model: {REFLECTION_MODEL}")
print(f"  Answering models: {[m[1] for m in ANSWERING_MODELS]}")
print(f"  Split: {TRAIN_RATIO:.0%} train, {VAL_RATIO:.0%} val")

# =============================================================================
# Step 4: Split the Benchmark
# =============================================================================

split = split_benchmark(
    benchmark,
    train_ratio=TRAIN_RATIO,
    val_ratio=VAL_RATIO,
    seed=SPLIT_SEED,
)

print("Data Split:")
print(f"  {split.summary()}")
print(f"\n  Train: {len(split.train)} questions (for optimization feedback)")
print(f"  Val: {len(split.val)} questions (for candidate selection)")

# =============================================================================
# Step 5: Create Verification Config
# =============================================================================

verification_config = VerificationConfig(
    answering_models=[
        ModelConfig(
            id=model_id,
            model_provider="anthropic",
            model_name=model_name,
            temperature=0.0,
            interface="langchain",
            system_prompt=SEED_PROMPT,
        )
        for model_id, model_name in ANSWERING_MODELS
    ],
    parsing_models=[
        ModelConfig(
            id="parser",
            model_provider="anthropic",
            model_name=PARSING_MODEL,
            temperature=0.0,
            interface="langchain",
        )
    ],
    evaluation_mode="template_only",  # AIME uses template-based correctness
    replicate_count=1,
)

print("Verification Config:")
print(f"  Answering models:")
for model in verification_config.answering_models:
    print(f"    - {model.id}: {model.model_name}")
print(f"  Parsing model: {PARSING_MODEL}")
print(f"  Evaluation mode: {verification_config.evaluation_mode}")

# =============================================================================
# Step 6: Create the Karenina Adapter with LLM Feedback
# =============================================================================

# Configure feedback model for LLM-generated feedback (using Sonnet for quality)
feedback_model_config = ModelConfig(
    id="feedback-model",
    model_provider="anthropic",
    model_name="claude-sonnet-4-5",  # Sonnet for quality feedback
    temperature=0.0,
    interface="langchain",
)

print("Feedback Model Config:")
print(f"  Model: {feedback_model_config.model_name}")
print(f"  Provider: {feedback_model_config.model_provider}")

# Create adapter for GEPA optimization with LLM feedback
adapter = KareninaAdapter(
    benchmark=benchmark,
    base_config=verification_config,
    targets=[OptimizationTarget.ANSWERING_SYSTEM_PROMPT],
    objective_config=ObjectiveConfig(
        include_template=True,  # Optimize for template correctness
        include_rubric=False,   # AIME doesn't use rubrics
    ),
    # Enable LLM-generated feedback for richer diagnostics
    feedback_model_config=feedback_model_config,
    # Enable differential analysis: compare successful vs failed responses
    enable_differential_analysis=True,
)

print("KareninaAdapter created with LLM feedback enabled")
print(f"  Targets: {[t.value for t in adapter.targets]}")
print(f"  LLM Feedback: {adapter.feedback_generator is not None}")
print(f"  Differential Analysis: {adapter.enable_differential_analysis}")

# =============================================================================
# Step 7: Run GEPA Optimization
# =============================================================================

import gepa

# Prepare seed candidate
seed_candidate = {
    "answering_system_prompt": SEED_PROMPT,
}

print("Starting GEPA Optimization...")
print("=" * 60)
print(f"Seed prompt: {SEED_PROMPT[:60]}...")
print(f"Max metric calls: {MAX_METRIC_CALLS}")
print(f"Train set: {len(split.train)} questions")
print(f"Val set: {len(split.val)} questions")
print("=" * 60)
print()

# Create simple logger for progress tracking
logger = SimpleLogger(show_all=False)

# Run GEPA optimization
result = gepa.optimize(
    seed_candidate=seed_candidate,
    trainset=split.train,
    valset=split.val,
    adapter=adapter,
    reflection_lm=REFLECTION_MODEL,
    max_metric_calls=MAX_METRIC_CALLS,
    frontier_type="objective",
    logger=logger,
    display_progress_bar=False,
)

# =============================================================================
# Step 8: Analyze the Results
# =============================================================================

print("\n" + "=" * 60)
print("Results")
print("=" * 60)

# Extract results
best_candidate = result.best_candidate
val_scores = result.val_aggregate_scores

print(f"Candidates evaluated: {len(val_scores)}")
print(f"Best candidate: {result.best_idx}")

if val_scores:
    baseline_score = val_scores[0]
    best_score = val_scores[result.best_idx]
    improvement = (best_score - baseline_score) / baseline_score if baseline_score > 0 else 0
    print(f"Baseline: {baseline_score:.2%} → Best: {best_score:.2%} ({improvement:+.1%})")

# Show the optimized prompt
optimized_prompt = best_candidate.get("answering_system_prompt", "")

print("\n" + "=" * 60)
print("Optimized Prompt:")
print("=" * 60)
print(optimized_prompt)

# Save to file
output_file = Path.cwd() / "optimized_prompt.txt"
output_file.write_text(optimized_prompt)
print(f"\nSaved to: {output_file}")
