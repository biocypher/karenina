"""Settings for the google_genai adapter.

Parses ``ModelConfig`` fields and ``extra_kwargs`` into one validated,
frozen settings object. Every adapter class builds its settings in its
constructor, so a configuration mistake (an unknown ``extra_kwargs`` key, a
bad built-in tool name, a conflicting setting) fails when the adapter is
built rather than on the first model call.
"""

from __future__ import annotations

import logging
import os
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Literal

if TYPE_CHECKING:
    from karenina.schemas.config import ModelConfig

logger = logging.getLogger(__name__)

Backend = Literal["vertex", "gemini_api"]

BUILTIN_GOOGLE_SEARCH = "google_search"
BUILTIN_URL_CONTEXT = "url_context"
BUILTIN_TOOL_NAMES: frozenset[str] = frozenset({BUILTIN_GOOGLE_SEARCH, BUILTIN_URL_CONTEXT})

DEFAULT_LOCATION = "global"
# Per-call guard used when the pipeline sets no request_timeout, so one
# stuck call cannot hold a whole run (design open question Q7).
DEFAULT_REQUEST_TIMEOUT = 300.0
DEFAULT_MAX_TOOL_RESULT_CHARS = 50_000

EXTRA_KWARGS_KEYS: frozenset[str] = frozenset(
    {
        "backend",
        "builtin_tools",
        "generate_content_config",
        "include_thoughts",
        "location",
        "max_search_queries",
        "project",
        "resolve_grounding_urls",
        "stream",
        "thinking_budget",
        "thinking_level",
    }
)

# AgentConfig.extra keys this adapter reads. Other keys are ignored, because
# callers pass the same AgentConfig to every adapter.
RUN_EXTRA_KEYS: frozenset[str] = frozenset({"builtin_tools", "max_search_queries", "max_tool_result_chars"})

# GenerateContentConfig fields the adapter sets itself. Passing them through
# generate_content_config would be ambiguous, so they are rejected.
_RESERVED_CONFIG_KEYS: frozenset[str] = frozenset(
    {
        "automatic_function_calling",
        "http_options",
        "max_output_tokens",
        "response_json_schema",
        "response_mime_type",
        "response_schema",
        "system_instruction",
        "temperature",
        "thinking_config",
        "tools",
    }
)


@dataclass(frozen=True)
class RunOptions:
    """Per-run options of the agent loop.

    Built from the settings and the run's ``AgentConfig.extra`` overrides,
    so a caller (for example the citation judge) can set a search budget
    per run.

    Attributes:
        builtin_tools: Gemini server-side tools enabled for the run.
        max_search_queries: Google Search budget for the run, or None for
            no budget.
        max_tool_result_chars: Tool results longer than this are cut with a
            marker before they reach the model.
    """

    builtin_tools: tuple[str, ...]
    max_search_queries: int | None
    max_tool_result_chars: int


@dataclass(frozen=True)
class GoogleGenAISettings:
    """Validated configuration of one google_genai adapter instance.

    Attributes:
        model_name: Gemini model id, for example ``gemini-4-argon``.
        backend: ``vertex`` (application default credentials) or
            ``gemini_api`` (API key plus optional base URL).
        project: Vertex project, or None to let the SDK resolve it.
        location: Vertex location.
        base_url: Custom endpoint base URL (``HttpOptions.base_url``).
        api_key: API key from ``ModelConfig.endpoint_api_key``. The
            environment fallback is resolved when the client is built.
        temperature: Sampling temperature, or None for the model default.
        max_output_tokens: Shared budget for thinking and answer tokens.
        system_prompt: Fallback system instruction.
        request_timeout: Per-call wall-clock guard in seconds.
        retry_policy: The ModelConfig retry policy (None means the default).
        builtin_tools: Server-side tools enabled by default.
        max_search_queries: Default Google Search budget per agent run.
        thinking_level: ``ThinkingConfig.thinking_level``, or None.
        thinking_budget: ``ThinkingConfig.thinking_budget``, or None.
        include_thoughts: Whether thought summaries are requested.
        stream: Whether calls use ``generate_content_stream``.
        resolve_grounding_urls: Whether grounding redirect links are
            followed to the real source URL.
        generate_content_config: Extra ``GenerateContentConfig`` fields.
        mcp_tool_filter: MCP tool names to expose, or None for all.
        mcp_tool_description_overrides: Tool descriptions to replace.
        mcp_http_timeout: Default MCP HTTP timeout in seconds.
        mcp_sse_read_timeout: Default MCP SSE read timeout in seconds.
    """

    model_name: str
    backend: Backend
    project: str | None
    location: str
    base_url: str | None
    api_key: str | None = field(repr=False)
    temperature: float | None
    max_output_tokens: int | None
    system_prompt: str | None
    request_timeout: float | None
    retry_policy: Any
    builtin_tools: tuple[str, ...]
    max_search_queries: int | None
    thinking_level: str | None
    thinking_budget: int | None
    include_thoughts: bool
    stream: bool
    resolve_grounding_urls: bool
    generate_content_config: Mapping[str, Any]
    mcp_tool_filter: tuple[str, ...] | None
    mcp_tool_description_overrides: Mapping[str, str]
    mcp_http_timeout: float | None
    mcp_sse_read_timeout: float | None

    @classmethod
    def from_model_config(cls, config: ModelConfig) -> GoogleGenAISettings:
        """Build validated settings from a ModelConfig.

        Args:
            config: The model configuration (``interface="google_genai"``).

        Returns:
            The validated settings.

        Raises:
            ValueError: If a required field is missing, an ``extra_kwargs``
                key is unknown, or a value is invalid or conflicting.
        """
        if not config.model_name:
            raise ValueError("model_name is required for the google_genai interface")

        extra = dict(config.extra_kwargs or {})
        unknown = sorted(set(extra) - EXTRA_KWARGS_KEYS)
        if unknown:
            raise ValueError(
                f"Unknown extra_kwargs for the google_genai interface: {unknown}. "
                f"Supported keys: {sorted(EXTRA_KWARGS_KEYS)}"
            )

        backend = _parse_backend(extra.get("backend"), config.endpoint_base_url)
        api_key = config.endpoint_api_key.get_secret_value() if config.endpoint_api_key is not None else None
        if backend == "vertex" and api_key:
            raise ValueError("endpoint_api_key is only used with backend='gemini_api'. Vertex uses Google login (ADC)")
        if backend == "gemini_api" and ("project" in extra or "location" in extra):
            logger.warning("google_genai: 'project' and 'location' are ignored with backend='gemini_api'")

        builtin_tools = _parse_builtin_tools(extra.get("builtin_tools", []), "extra_kwargs['builtin_tools']")
        max_search_queries = _parse_optional_int(
            extra.get("max_search_queries"), "extra_kwargs['max_search_queries']", minimum=0
        )
        if max_search_queries is not None and BUILTIN_GOOGLE_SEARCH not in builtin_tools:
            logger.warning("google_genai: max_search_queries is set but google_search is not in builtin_tools")

        thinking_level = _parse_thinking_level(extra.get("thinking_level"))
        thinking_budget = _parse_optional_int(extra.get("thinking_budget"), "extra_kwargs['thinking_budget']")
        if thinking_level is not None and thinking_budget is not None:
            raise ValueError(
                "Set either thinking_level or thinking_budget, not both (Gemini rejects requests with both)"
            )

        default_flag = backend == "gemini_api"
        include_thoughts = _parse_bool(extra.get("include_thoughts", default_flag), "include_thoughts")
        stream = _parse_bool(extra.get("stream", default_flag), "stream")
        resolve_grounding_urls = _parse_bool(extra.get("resolve_grounding_urls", False), "resolve_grounding_urls")

        location = extra.get("location") or os.environ.get("GOOGLE_CLOUD_LOCATION") or DEFAULT_LOCATION
        project = extra.get("project") or os.environ.get("GOOGLE_CLOUD_PROJECT")

        return cls(
            model_name=config.model_name,
            backend=backend,
            project=str(project) if project else None,
            location=str(location),
            base_url=config.endpoint_base_url,
            api_key=api_key,
            temperature=config.temperature,
            max_output_tokens=config.max_tokens or None,
            system_prompt=config.system_prompt,
            request_timeout=(config.request_timeout if config.request_timeout is not None else DEFAULT_REQUEST_TIMEOUT),
            retry_policy=config.retry_policy,
            builtin_tools=builtin_tools,
            max_search_queries=max_search_queries,
            thinking_level=thinking_level,
            thinking_budget=thinking_budget,
            include_thoughts=include_thoughts,
            stream=stream,
            resolve_grounding_urls=resolve_grounding_urls,
            generate_content_config=_parse_generate_content_config(extra.get("generate_content_config", {})),
            mcp_tool_filter=tuple(config.mcp_tool_filter) if config.mcp_tool_filter else None,
            mcp_tool_description_overrides=dict(config.mcp_tool_description_overrides or {}),
            mcp_http_timeout=config.mcp_http_timeout,
            mcp_sse_read_timeout=config.mcp_sse_read_timeout,
        )

    def run_options(self, extra: Mapping[str, Any] | None) -> RunOptions:
        """Resolve the agent run options, applying ``AgentConfig.extra`` overrides.

        Args:
            extra: The run's ``AgentConfig.extra``. Only the keys in
                ``RUN_EXTRA_KEYS`` are read.

        Returns:
            The run options.

        Raises:
            ValueError: If an override has an invalid value.
        """
        extra = extra or {}
        builtin_tools = self.builtin_tools
        if "builtin_tools" in extra:
            builtin_tools = _parse_builtin_tools(extra["builtin_tools"], "AgentConfig.extra['builtin_tools']")
        max_search_queries = self.max_search_queries
        if "max_search_queries" in extra:
            max_search_queries = _parse_optional_int(
                extra["max_search_queries"], "AgentConfig.extra['max_search_queries']", minimum=0
            )
        max_chars = _parse_optional_int(
            extra.get("max_tool_result_chars", DEFAULT_MAX_TOOL_RESULT_CHARS),
            "AgentConfig.extra['max_tool_result_chars']",
            minimum=1,
        )
        return RunOptions(
            builtin_tools=builtin_tools,
            max_search_queries=max_search_queries,
            max_tool_result_chars=max_chars if max_chars is not None else DEFAULT_MAX_TOOL_RESULT_CHARS,
        )

    def thinking_config_kwargs(self) -> dict[str, Any] | None:
        """Return ``ThinkingConfig`` keyword arguments, or None to omit it.

        The thinking config is omitted when nothing is requested, so models
        without thinking support never receive one.
        """
        if not self.include_thoughts and self.thinking_level is None and self.thinking_budget is None:
            return None
        kwargs: dict[str, Any] = {"include_thoughts": self.include_thoughts}
        if self.thinking_level is not None:
            kwargs["thinking_level"] = self.thinking_level
        if self.thinking_budget is not None:
            kwargs["thinking_budget"] = self.thinking_budget
        return kwargs


def _parse_backend(value: Any, endpoint_base_url: str | None) -> Backend:
    """Resolve the backend from extra_kwargs, defaulting on endpoint_base_url."""
    if value is None:
        return "gemini_api" if endpoint_base_url else "vertex"
    if value == "vertex":
        return "vertex"
    if value == "gemini_api":
        return "gemini_api"
    raise ValueError(f"extra_kwargs['backend'] must be 'vertex' or 'gemini_api', got {value!r}")


def _parse_builtin_tools(value: Any, source: str) -> tuple[str, ...]:
    """Validate a list of built-in tool names, keeping order and dropping repeats."""
    if isinstance(value, str) or not isinstance(value, list | tuple):
        raise ValueError(f"{source} must be a list of tool names, got {value!r}")
    names: list[str] = []
    for item in value:
        if item not in BUILTIN_TOOL_NAMES:
            raise ValueError(f"{source}: unknown built-in tool {item!r}. Supported: {sorted(BUILTIN_TOOL_NAMES)}")
        if item not in names:
            names.append(item)
    return tuple(names)


def _parse_optional_int(value: Any, source: str, *, minimum: int | None = None) -> int | None:
    """Validate an optional integer with an optional lower bound."""
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{source} must be an integer or None, got {value!r}")
    if minimum is not None and value < minimum:
        raise ValueError(f"{source} must be >= {minimum}, got {value}")
    return int(value)


def _parse_bool(value: Any, key: str) -> bool:
    """Validate a boolean extra_kwargs value."""
    if not isinstance(value, bool):
        raise ValueError(f"extra_kwargs['{key}'] must be a bool, got {value!r}")
    return value


def _parse_thinking_level(value: Any) -> str | None:
    """Validate thinking_level against the installed SDK's ThinkingLevel values."""
    if value is None:
        return None
    if not isinstance(value, str) or not value:
        raise ValueError(f"extra_kwargs['thinking_level'] must be a non-empty string, got {value!r}")
    level = value.upper()
    try:
        from google.genai import types
    except ImportError:
        return level
    level_enum = getattr(types, "ThinkingLevel", None)
    if level_enum is None:
        return level
    known = {name for name in level_enum.__members__ if name != "THINKING_LEVEL_UNSPECIFIED"}
    if level not in known:
        raise ValueError(f"extra_kwargs['thinking_level'] must be one of {sorted(known)}, got {value!r}")
    return level


def _parse_generate_content_config(value: Any) -> dict[str, Any]:
    """Validate the GenerateContentConfig passthrough dict."""
    if not isinstance(value, Mapping):
        raise ValueError(f"extra_kwargs['generate_content_config'] must be a dict, got {value!r}")
    passthrough = dict(value)
    reserved = sorted(set(passthrough) & _RESERVED_CONFIG_KEYS)
    if reserved:
        raise ValueError(
            f"extra_kwargs['generate_content_config'] cannot set {reserved}: the adapter sets these from "
            "ModelConfig fields and extra_kwargs"
        )
    try:
        from google.genai import types
    except ImportError:
        return passthrough
    unknown = sorted(set(passthrough) - set(types.GenerateContentConfig.model_fields))
    if unknown:
        raise ValueError(f"extra_kwargs['generate_content_config'] has unknown GenerateContentConfig fields: {unknown}")
    try:
        types.GenerateContentConfig(**passthrough)
    except Exception as e:
        raise ValueError(f"extra_kwargs['generate_content_config'] is invalid: {e}") from e
    return passthrough
