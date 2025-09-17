"""Embedding similarity utilities for verification fallback."""

import json
import os
import threading
from typing import Any

from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage

from ...llm.interface import init_chat_model_unified
from ..models import ModelConfig

# Global cache for embedding models (thread-safe)
_embedding_model_cache: dict[str, Any] = {}
_cache_lock = threading.Lock()


def preload_embedding_model() -> str:
    """
    Preload the embedding model for the current job.

    Returns:
        Model name that was loaded

    Raises:
        ImportError: If sentence-transformers is not available
        RuntimeError: If model loading fails
    """
    model_name = _get_embedding_model_name()

    with _cache_lock:
        if model_name in _embedding_model_cache:
            return model_name

        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as e:
            raise ImportError(
                "sentence-transformers is required for embedding check. Install it with: uv add sentence-transformers"
            ) from e

        try:
            model = SentenceTransformer(model_name)
            _embedding_model_cache[model_name] = model
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
    return os.getenv("EMBEDDING_CHECK_MODEL", "all-MiniLM-L6-v2")


def _get_embedding_threshold() -> float:
    """
    Get the embedding similarity threshold from environment variable.

    Returns:
        The similarity threshold (0.0 to 1.0)
    """
    try:
        threshold = float(os.getenv("EMBEDDING_CHECK_THRESHOLD", "0.85"))
        return max(0.0, min(1.0, threshold))  # Clamp between 0 and 1
    except ValueError:
        return 0.85  # Default fallback


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
) -> tuple[bool, str]:
    """
    Check semantic equivalence between ground truth and LLM response using parsing LLM.

    Args:
        ground_truth_data: The ground truth parsed data
        llm_response_data: The LLM response parsed data
        parsing_model: The parsing model configuration to use for semantic check
        question_text: The original question text for context (optional but recommended)

    Returns:
        Tuple of (is_semantically_equivalent, check_details)

    Raises:
        RuntimeError: If semantic equivalence check fails
    """
    if not ground_truth_data or not llm_response_data:
        return False, "Missing data for semantic equivalence check"

    try:
        # Initialize parsing LLM
        parsing_llm = init_chat_model_unified(
            model=parsing_model.model_name,
            provider=parsing_model.model_provider,
            temperature=parsing_model.temperature,
            interface=parsing_model.interface,
        )

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

        messages: list[BaseMessage] = [SystemMessage(content=system_prompt), HumanMessage(content=user_prompt)]

        # Get LLM response
        response = parsing_llm.invoke(messages)
        response_text = response.content if hasattr(response, "content") else str(response)

        # Parse response
        response_clean = response_text.strip().upper()
        is_equivalent = response_clean.startswith("YES")

        details = f"LLM semantic check result: {response_text.strip()}"

        return is_equivalent, details

    except Exception as e:
        raise RuntimeError(f"Failed to perform semantic equivalence check: {e}") from e


def perform_embedding_check(
    ground_truth_data: dict[str, Any] | None,
    llm_response_data: dict[str, Any] | None,
    parsing_model: ModelConfig,
    question_text: str | None = None,
) -> tuple[bool, float | None, str | None, bool, str | None]:
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
            embedding_check_performed,
            semantic_check_details
        )
    """
    if not _should_use_embedding_check():
        return False, None, None, False, None

    try:
        # Compute embedding similarity
        similarity_score, model_name = compute_embedding_similarity(ground_truth_data, llm_response_data)

        threshold = _get_embedding_threshold()

        # Check if similarity exceeds threshold
        if similarity_score >= threshold:
            # Perform semantic equivalence check
            try:
                is_equivalent, check_details = check_semantic_equivalence(
                    ground_truth_data, llm_response_data, parsing_model, question_text
                )

                return (
                    is_equivalent,  # should_override_result
                    similarity_score,
                    model_name,
                    True,  # embedding_check_performed
                    check_details,
                )

            except RuntimeError:
                # If semantic check fails, don't override but still record embedding info
                return (
                    False,  # should_override_result
                    similarity_score,
                    model_name,
                    True,  # embedding_check_performed
                    "Semantic check failed - falling back to original result",
                )
        else:
            # Similarity below threshold - don't override
            return (
                False,  # should_override_result
                similarity_score,
                model_name,
                True,  # embedding_check_performed
                f"Similarity {similarity_score:.3f} below threshold {threshold:.3f}",
            )

    except (ImportError, RuntimeError) as e:
        # If embedding check fails entirely, return no override but log the attempt
        return (
            False,  # should_override_result
            None,
            None,
            True,  # embedding_check_performed (we tried)
            f"Embedding check failed: {e}",
        )
