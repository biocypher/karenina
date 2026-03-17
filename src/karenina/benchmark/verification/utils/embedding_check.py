"""Embedding similarity utilities for verification fallback."""

import json
import logging
import os
import threading
from collections import OrderedDict
from typing import Any

from karenina.adapters.factory import get_llm
from karenina.ports.messages import Message
from karenina.schemas.config import ModelConfig
from karenina.schemas.verification.config import DEFAULT_EMBEDDING_MODEL, DEFAULT_EMBEDDING_THRESHOLD

logger = logging.getLogger(__name__)

# Global cache for embedding models (thread-safe) with LRU eviction
# Using OrderedDict to track access order for LRU eviction policy
_embedding_model_cache: OrderedDict[str, Any] = OrderedDict()
_cache_lock = threading.Lock()

# Maximum number of embedding models to keep in cache
# SentenceTransformer models are 100MB-1GB each, so limit to prevent memory exhaustion
_MAX_CACHED_MODELS = 3


def preload_embedding_model() -> str:
    """
    Preload the embedding model for the current job.

    Uses LRU (Least Recently Used) caching to limit memory usage.
    When the cache exceeds _MAX_CACHED_MODELS, the oldest models are evicted.

    Returns:
        Model name that was loaded

    Raises:
        ImportError: If sentence-transformers is not available
        RuntimeError: If model loading fails
    """
    model_name = _get_embedding_model_name()

    with _cache_lock:
        if model_name in _embedding_model_cache:
            # Move to end to mark as most recently used
            _embedding_model_cache.move_to_end(model_name)
            return model_name

        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as e:
            raise ImportError(
                "sentence-transformers is required for embedding check. Install it with: pip install karenina[embeddings]"
            ) from e

        try:
            model = SentenceTransformer(model_name)
            _embedding_model_cache[model_name] = model

            # Evict oldest models if cache exceeds limit
            while len(_embedding_model_cache) > _MAX_CACHED_MODELS:
                # Remove oldest (first) item - the least recently used
                evicted_name, evicted_model = _embedding_model_cache.popitem(last=False)
                # Explicitly delete reference to help garbage collection
                del evicted_model

            return model_name
        except Exception as e:
            raise RuntimeError(f"Failed to preload embedding model {model_name}: {e}") from e


def clear_embedding_model_cache() -> None:
    """Clear the embedding model cache to free memory."""
    with _cache_lock:
        _embedding_model_cache.clear()


def _get_cached_embedding_model(model_name: str) -> Any:
    """
    Get a cached embedding model or load it if not cached.

    Args:
        model_name: Name of the embedding model

    Returns:
        The SentenceTransformer model instance

    Raises:
        ImportError: If sentence-transformers is not available
        RuntimeError: If model loading fails
    """
    with _cache_lock:
        if model_name in _embedding_model_cache:
            # Move to end to mark as most recently used
            _embedding_model_cache.move_to_end(model_name)
            return _embedding_model_cache[model_name]

    # Model not cached, preload it
    preload_embedding_model()

    with _cache_lock:
        return _embedding_model_cache[model_name]


def _should_use_embedding_check() -> bool:
    """
    Check if embedding check should be performed.

    Returns:
        True if embedding check is enabled, False otherwise
    """
    return os.getenv("EMBEDDING_CHECK", "false").lower() in ("true", "1", "yes", "on")


def _get_embedding_model_name() -> str:
    """
    Get the embedding model name from environment variable.

    Returns:
        The embedding model name to use
    """
    return os.getenv("EMBEDDING_CHECK_MODEL", DEFAULT_EMBEDDING_MODEL)


def _get_embedding_threshold() -> float:
    """
    Get the embedding similarity threshold from environment variable.

    Returns:
        The similarity threshold (0.0 to 1.0)
    """
    try:
        threshold = float(os.getenv("EMBEDDING_CHECK_THRESHOLD", str(DEFAULT_EMBEDDING_THRESHOLD)))
        return max(0.0, min(1.0, threshold))  # Clamp between 0 and 1
    except ValueError:
        return DEFAULT_EMBEDDING_THRESHOLD  # Default fallback


def _convert_to_comparable_string(data: dict[str, Any] | None) -> str:
    """
    Convert parsed response data to a comparable string.

    Args:
        data: The parsed response data (ground truth or LLM response)

    Returns:
        A string representation suitable for embedding comparison
    """
    if data is None:
        return ""

    try:
        # Convert to JSON string with sorted keys for consistency
        return json.dumps(data, sort_keys=True, default=str)
    except (TypeError, ValueError):
        # Fallback to string representation
        return str(data)


def compute_embedding_similarity(
    ground_truth_data: dict[str, Any] | None,
    llm_response_data: dict[str, Any] | None,
) -> tuple[float, str]:
    """
    Compute embedding similarity between ground truth and LLM response.

    Args:
        ground_truth_data: The ground truth parsed data
        llm_response_data: The LLM response parsed data

    Returns:
        Tuple of (similarity_score, model_name_used)

    Raises:
        ImportError: If sentence-transformers is not available
        RuntimeError: If embedding computation fails
    """
    if not ground_truth_data or not llm_response_data:
        return 0.0, _get_embedding_model_name()

    # Convert data to comparable strings
    gt_text = _convert_to_comparable_string(ground_truth_data)
    llm_text = _convert_to_comparable_string(llm_response_data)

    if not gt_text or not llm_text:
        return 0.0, _get_embedding_model_name()

    model_name = _get_embedding_model_name()

    try:
        # Get the cached embedding model (loads if not cached)
        model = _get_cached_embedding_model(model_name)

        # Compute embeddings
        embeddings = model.encode([gt_text, llm_text])

        # Compute cosine similarity
        from sklearn.metrics.pairwise import cosine_similarity

        similarity = cosine_similarity([embeddings[0]], [embeddings[1]])[0][0]

        # Ensure similarity is between 0 and 1
        similarity = max(0.0, min(1.0, float(similarity)))

        return similarity, model_name

    except Exception as e:
        raise RuntimeError(f"Failed to compute embedding similarity with model {model_name}: {e}") from e


def check_semantic_equivalence(
    ground_truth_data: dict[str, Any] | None,
    llm_response_data: dict[str, Any] | None,
    parsing_model: ModelConfig,
    question_text: str | None = None,
) -> bool:
    """
    Check semantic equivalence between ground truth and LLM response using parsing LLM.

    Args:
        ground_truth_data: The ground truth parsed data
        llm_response_data: The LLM response parsed data
        parsing_model: The parsing model configuration to use for semantic check
        question_text: The original question text for context (optional but recommended)

    Returns:
        True if responses are semantically equivalent, False otherwise

    Raises:
        RuntimeError: If semantic equivalence check fails
    """
    if not ground_truth_data or not llm_response_data:
        return False

    try:
        # Get LLM via the port/adapter factory (respects interface routing)
        parsing_llm = get_llm(parsing_model)

        # Convert data to readable strings
        gt_text = _convert_to_comparable_string(ground_truth_data)
        llm_text = _convert_to_comparable_string(llm_response_data)

        # Create semantic equivalence prompt with question context
        system_prompt = """You are a semantic equivalence evaluator. Your task is to determine if two parsed responses are semantically equivalent in the context of the original question, even if they differ in exact structure or wording.

Consider two responses semantically equivalent if:
1. They answer the original question with the same core meaning or information
2. They would lead to the same conclusion or decision in the context of the question
3. Minor differences in formatting, wording, or structure don't affect the substance of the answer
4. Both responses demonstrate the same level of correctness for the given question

Respond with exactly "YES" if they are semantically equivalent, or "NO" if they are not."""

        # Build user prompt with optional question context
        if question_text:
            user_prompt = f"""Original Question:
{question_text}

Ground Truth Response:
{gt_text}

Model Response:
{llm_text}

Given the original question above, are these two responses semantically equivalent in answering the question?"""
        else:
            user_prompt = f"""Ground Truth Response:
{gt_text}

Model Response:
{llm_text}

Are these two responses semantically equivalent?"""

        messages = [Message.system(system_prompt), Message.user(user_prompt)]

        # Get LLM response via port interface
        response = parsing_llm.invoke(messages)
        response_text = response.content

        # Parse response
        response_clean = response_text.strip().upper()
        is_equivalent = response_clean.startswith("YES")

        return is_equivalent

    except Exception as e:
        raise RuntimeError(f"Failed to perform semantic equivalence check: {e}") from e


def perform_embedding_check(
    ground_truth_data: dict[str, Any] | None,
    llm_response_data: dict[str, Any] | None,
    parsing_model: ModelConfig,
    question_text: str | None = None,
) -> tuple[bool, float | None, str | None, bool]:
    """
    Perform complete embedding check with fallback to semantic equivalence.

    Args:
        ground_truth_data: The ground truth parsed data
        llm_response_data: The LLM response parsed data
        parsing_model: The parsing model configuration for semantic check
        question_text: The original question text for context (optional but recommended)

    Returns:
        Tuple of (
            should_override_result,
            similarity_score,
            embedding_model_used,
            embedding_check_performed
        )
    """
    if not _should_use_embedding_check():
        return False, None, None, False

    try:
        # Compute embedding similarity
        similarity_score, model_name = compute_embedding_similarity(ground_truth_data, llm_response_data)

        threshold = _get_embedding_threshold()

        # Check if similarity exceeds threshold
        if similarity_score >= threshold:
            # Perform semantic equivalence check
            try:
                is_equivalent = check_semantic_equivalence(
                    ground_truth_data, llm_response_data, parsing_model, question_text
                )

                return (
                    is_equivalent,  # should_override_result
                    similarity_score,
                    model_name,
                    True,  # embedding_check_performed
                )

            except RuntimeError:
                # If semantic check fails, don't override but still record embedding info
                return (
                    False,  # should_override_result
                    similarity_score,
                    model_name,
                    True,  # embedding_check_performed
                )
        else:
            # Similarity below threshold - don't override
            return (
                False,  # should_override_result
                similarity_score,
                model_name,
                True,  # embedding_check_performed
            )

    except (ImportError, RuntimeError):
        # If embedding check fails entirely, return no override but log the attempt
        return (
            False,  # should_override_result
            None,
            None,
            True,  # embedding_check_performed (we tried)
        )
