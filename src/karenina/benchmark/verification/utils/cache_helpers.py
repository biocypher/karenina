"""Cache-related helper functions for verification.

This module provides cache key generation and data extraction utilities
for the AnswerTraceCache used in batch verification.
"""

import logging
from typing import Any

from ....schemas.verification import VerificationResult
from ....utils.answer_cache import AnswerTraceCache

logger = logging.getLogger(__name__)


def generate_answer_cache_key(task: dict[str, Any]) -> str:
    """Generate cache key for answer traces.

    Cache key format: {question_id}_{answering_model_id}_{replicate}

    Args:
        task: Task dictionary

    Returns:
        Cache key string
    """
    question_id = task["question_id"]
    answering_model_id = task["answering_model"].id
    replicate = task.get("replicate")

    if replicate is None:
        return f"{question_id}_{answering_model_id}"
    else:
        return f"{question_id}_{answering_model_id}_rep{replicate}"


def extract_answer_data_from_result(result: VerificationResult) -> dict[str, Any]:
    """Extract answer data from verification result for caching.

    Args:
        result: Verification result

    Returns:
        Dictionary with answer data to cache
    """
    # Extract usage metadata for answer generation stage
    # Convert from stage summary format to callback metadata format
    # Stage summary: {"input_tokens": 123, "model": "gpt-4", ...}
    # Callback format: {"gpt-4": {"input_tokens": 123, ...}}
    template = result.template
    usage_metadata = None
    raw_usage = template.usage_metadata if template else None
    if raw_usage and "answer_generation" in raw_usage:
        stage_summary = raw_usage["answer_generation"]
        if stage_summary and isinstance(stage_summary, dict):
            # Extract model name and create callback-style nested dict
            model_name = stage_summary.get("model", "unknown")
            usage_metadata = {
                model_name: {
                    "input_tokens": stage_summary.get("input_tokens", 0),
                    "output_tokens": stage_summary.get("output_tokens", 0),
                    "total_tokens": stage_summary.get("total_tokens", 0),
                    "input_token_details": stage_summary.get("input_token_details", {}),
                    "output_token_details": stage_summary.get("output_token_details", {}),
                }
            }

    return {
        "raw_llm_response": template.raw_llm_response if template else None,
        "usage_metadata": usage_metadata,
        "agent_metrics": template.agent_metrics if template else None,
        "recursion_limit_reached": template.recursion_limit_reached if template else None,
        "answering_mcp_servers": template.answering_mcp_servers if template else None,
    }


def log_cache_stats(answer_cache: AnswerTraceCache, mode: str = "sequential") -> None:
    """Log answer cache statistics if there were cache interactions.

    Args:
        answer_cache: The answer trace cache instance
        mode: Execution mode for logging ("sequential" or "parallel mode")
    """
    stats = answer_cache.get_stats()
    if stats["hits"] > 0 or stats["waits"] > 0:
        logger.info(
            f"Answer cache statistics ({mode}): {stats['hits']} hits, {stats['misses']} misses, "
            f"{stats['waits']} {'IN_PROGRESS encounters' if mode == 'parallel mode' else 'waits'}, "
            f"{stats['timeouts']} timeouts"
        )
