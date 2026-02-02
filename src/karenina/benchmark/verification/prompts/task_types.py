"""Enum identifying every distinct LLM call in the verification pipeline.

Each value maps to a specific prompt construction site. Used by PromptAssembler
to look up adapter instructions and user instructions for a given call.
"""

from enum import Enum


class PromptTask(str, Enum):
    """Identifies a distinct LLM call type in the verification pipeline.

    Values are grouped by pipeline stage:
    - Generation and parsing of answers
    - Trace-level quality checks (abstention, sufficiency)
    - Rubric trait evaluation (LLM, literal, metric)
    - Deep judgment stages for both template and rubric flows
    """

    # --- Answer generation and parsing ---

    GENERATION = "generation"
    """LLM generates a response to the benchmark question."""

    PARSING = "parsing"
    """Judge LLM parses the raw response into a structured answer template."""

    # --- Trace-level quality checks ---

    ABSTENTION_DETECTION = "abstention_detection"
    """Detects whether the answering model refused to answer."""

    SUFFICIENCY_DETECTION = "sufficiency_detection"
    """Checks whether the response contains enough information to populate the template."""

    # --- Rubric trait evaluation ---

    RUBRIC_LLM_TRAIT_BATCH = "rubric_llm_trait_batch"
    """Evaluates all boolean/score LLM rubric traits in a single batched call."""

    RUBRIC_LLM_TRAIT_SINGLE = "rubric_llm_trait_single"
    """Evaluates a single boolean/score LLM rubric trait sequentially."""

    RUBRIC_LITERAL_TRAIT_BATCH = "rubric_literal_trait_batch"
    """Evaluates all literal (categorical) rubric traits in a single batched call."""

    RUBRIC_LITERAL_TRAIT_SINGLE = "rubric_literal_trait_single"
    """Evaluates a single literal (categorical) rubric trait sequentially."""

    RUBRIC_METRIC_TRAIT = "rubric_metric_trait"
    """Evaluates a metric rubric trait via confusion matrix extraction."""

    # --- Deep judgment: template parsing flow ---

    DJ_TEMPLATE_EXCERPT_EXTRACTION = "dj_template_excerpt_extraction"
    """Stage 1: Extracts verbatim excerpts from the response per template attribute."""

    DJ_TEMPLATE_HALLUCINATION = "dj_template_hallucination"
    """Stage 1.5: Assesses hallucination risk for extracted excerpts via search."""

    DJ_TEMPLATE_REASONING = "dj_template_reasoning"
    """Stage 2: Generates reasoning mapping excerpts to template attributes."""

    # --- Deep judgment: rubric trait evaluation flow ---

    DJ_RUBRIC_EXCERPT_EXTRACTION = "dj_rubric_excerpt_extraction"
    """Stage 1: Extracts excerpts supporting deep-judgment-enabled rubric traits."""

    DJ_RUBRIC_HALLUCINATION = "dj_rubric_hallucination"
    """Stage 1.5: Assesses per-excerpt hallucination risk using search results."""

    DJ_RUBRIC_REASONING = "dj_rubric_reasoning"
    """Stage 2: Generates reasoning explaining trait evaluation based on excerpts."""

    DJ_RUBRIC_SCORE_EXTRACTION = "dj_rubric_score_extraction"
    """Stage 3: Extracts the final score from deep judgment reasoning."""
