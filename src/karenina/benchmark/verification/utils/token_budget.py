"""Approximate token counting and truncation for judge inputs.

Counts use tiktoken's ``o200k_base`` encoding by default. This is a stable,
offline approximation for non-OpenAI models. Callers needing exact provider
counts should count externally and use these helpers only for truncation.
"""

from __future__ import annotations

import functools
import json
import urllib.error
import urllib.request
from typing import TYPE_CHECKING

import tiktoken

from karenina.benchmark.verification.prompts.deep_judgment.rubric.deep_judgment import (
    DeepJudgmentPromptBuilder,
)

if TYPE_CHECKING:
    from karenina.schemas.config.models import ModelConfig
    from karenina.schemas.entities.rubric import LLMRubricTrait


@functools.lru_cache(maxsize=4)
def _encoding(encoding_name: str) -> tiktoken.Encoding:
    """Return a cached tiktoken encoding."""
    return tiktoken.get_encoding(encoding_name)


def count_tokens(text: str, encoding_name: str = "o200k_base") -> int:
    """Count tokens in text under the requested tiktoken encoding."""
    return len(_encoding(encoding_name).encode(text))


def count_deep_judgment_reasoning_tokens(
    answer: str,
    trait: LLMRubricTrait,
    *,
    question: str = "",
    task_eval_mode: bool = True,
    encoding_name: str = "o200k_base",
    parsing_model: ModelConfig | None = None,
    timeout: float = 10.0,
) -> int:
    """Count the rendered first-stage deep-judgment reasoning prompt.

    The count includes both system and user messages. It uses the stable
    offline encoding named by ``encoding_name`` and does not contact a model
    provider.

    Args:
        answer: Response or trace passed to the reasoning stage.
        trait: Deep-judgment rubric trait.
        question: Question rendered when TaskEval mode is disabled.
        task_eval_mode: Whether to use TaskEval prompt rendering.
        encoding_name: Tiktoken encoding used for an offline count.
        parsing_model: Optional OpenAI-compatible model whose ``/tokenize``
            endpoint provides an exact chat-template-aware count.
        timeout: Endpoint timeout in seconds.

    Returns:
        Token count for the complete reasoning prompt.

    Raises:
        ValueError: If ``parsing_model`` is not an OpenAI endpoint model.
        RuntimeError: If an exact endpoint count fails.
    """
    builder = DeepJudgmentPromptBuilder()
    system_prompt = builder.build_reasoning_system_prompt()
    user_prompt = builder.build_reasoning_user_prompt_without_excerpts(
        question=question,
        answer=answer,
        trait=trait,
        task_eval_mode=task_eval_mode,
    )
    if parsing_model is None:
        return count_tokens(f"{system_prompt}\n{user_prompt}", encoding_name)
    if parsing_model.interface != "openai_endpoint" or parsing_model.endpoint_base_url is None:
        raise ValueError("Exact token counting requires an openai_endpoint parsing model")

    base_url = str(parsing_model.endpoint_base_url).rstrip("/")
    if base_url.endswith("/v1"):
        base_url = base_url[: -len("/v1")]
    payload = json.dumps(
        {
            "model": parsing_model.model_name,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        }
    ).encode("utf-8")
    request = urllib.request.Request(
        f"{base_url}/tokenize",
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    api_key = parsing_model.endpoint_api_key
    if api_key is not None:
        secret = api_key.get_secret_value()
        if secret and secret != "EMPTY":
            request.add_header("Authorization", f"Bearer {secret}")
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            body = json.loads(response.read().decode("utf-8"))
    except (
        urllib.error.HTTPError,
        urllib.error.URLError,
        TimeoutError,
        json.JSONDecodeError,
    ) as error:
        raise RuntimeError("The model endpoint could not count the reasoning prompt") from error
    count = body.get("count")
    if not isinstance(count, int):
        raise RuntimeError("The model endpoint returned no integer token count")
    return count


def truncate_to_token_budget(
    text: str,
    max_tokens: int,
    encoding_name: str = "o200k_base",
) -> tuple[str, bool]:
    """Middle-truncate text so its encoded form fits a token budget.

    The head and tail are retained around an omission marker so both the
    beginning of a trace and its final messages remain available to a judge.

    Args:
        text: Text to constrain.
        max_tokens: Positive maximum number of encoded tokens.
        encoding_name: Tiktoken encoding used to count and truncate.

    Returns:
        The possibly truncated text and whether truncation occurred.

    Raises:
        ValueError: If ``max_tokens`` is not positive.
    """
    if max_tokens <= 0:
        raise ValueError("max_tokens must be positive")

    encoding = _encoding(encoding_name)
    tokens = encoding.encode(text)
    if len(tokens) <= max_tokens:
        return text, False

    omitted = len(tokens)
    marker_tokens: list[int] = []
    keep_count = 0
    for _ in range(4):
        marker = f"\n[truncated: {omitted} tokens omitted]\n"
        marker_tokens = encoding.encode(marker)
        keep_count = max_tokens - len(marker_tokens)
        if keep_count <= 0:
            return encoding.decode(marker_tokens[:max_tokens]), True
        updated_omitted = len(tokens) - keep_count
        if updated_omitted == omitted:
            break
        omitted = updated_omitted

    head_count = (keep_count + 1) // 2
    tail_count = keep_count - head_count
    result_tokens = tokens[:head_count] + marker_tokens
    if tail_count:
        result_tokens += tokens[-tail_count:]
    return encoding.decode(result_tokens), True
