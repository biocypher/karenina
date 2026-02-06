# Writing Custom Adapters

This guide walks through creating a custom adapter that integrates a new LLM backend into karenina's verification pipeline. By the end, you'll have a fully registered adapter with its own prompt instructions, available through the standard factory functions.

---

## What You Need to Implement

A complete adapter provides up to three port implementations and a registration module:

| Component | Required | Purpose |
|-----------|----------|---------|
| `AgentPort` adapter | Optional | Multi-turn agent execution with tools/MCP |
| `ParserPort` adapter | Optional | LLM-based structured output parsing |
| `LLMPort` adapter | Optional | Simple LLM invocation |
| `registration.py` | Required | Register adapter with the factory system |
| Prompt instructions | Recommended | Adapter-specific prompt tuning |

You can implement any subset of the three ports. If your adapter only supports simple invocation, implement just `LLMPort` and set the other factory functions to `None`.

---

## Step 1: Create the Adapter Directory

Create a new directory under `karenina/src/karenina/adapters/`:

```
karenina/src/karenina/adapters/my_provider/
├── __init__.py          # Package marker
├── registration.py      # Adapter registration (required)
├── llm.py               # LLMPort implementation
├── parser.py            # ParserPort implementation
├── agent.py             # AgentPort implementation
└── prompts/             # Adapter-specific prompt instructions
    ├── __init__.py
    ├── parsing.py
    ├── rubric.py
    └── deep_judgment.py
```

---

## Step 2: Implement Port Protocols

Adapter classes use **duck typing** — they implement the port method signatures without inheriting from the protocol class. Any object with the right methods satisfies the protocol.

### Implementing LLMPort

The simplest port. Implement `ainvoke`, `invoke`, `with_structured_output`, and a `capabilities` property:

```python
from __future__ import annotations

from pydantic import BaseModel

from karenina.ports.capabilities import PortCapabilities
from karenina.ports.messages import Message
from karenina.ports.llm import LLMResponse
from karenina.ports.usage import UsageMetadata
from karenina.schemas.config import ModelConfig


class MyProviderLLMAdapter:
    """LLM adapter for MyProvider.

    Implements LLMPort protocol via duck typing (no explicit inheritance).
    """

    def __init__(self, model_config: ModelConfig) -> None:
        self._config = model_config
        # Initialize your provider's client here
        self._client = self._initialize_client()

    def _initialize_client(self):
        """Create the provider-specific client."""
        # Import your provider's SDK here (lazy import)
        from my_provider_sdk import Client
        return Client(
            model=self._config.model_name,
            api_key=...,
        )

    @property
    def capabilities(self) -> PortCapabilities:
        return PortCapabilities(
            supports_system_prompt=True,
            supports_structured_output=False,
        )

    async def ainvoke(self, messages: list[Message]) -> LLMResponse:
        """Async invocation — the primary API."""
        # Convert karenina Messages to your provider's format
        provider_messages = self._convert_messages(messages)

        # Call your provider
        response = await self._client.chat(provider_messages)

        # Return standardized response
        return LLMResponse(
            content=response.text,
            usage=UsageMetadata(
                input_tokens=response.usage.prompt_tokens,
                output_tokens=response.usage.completion_tokens,
            ),
            raw=response,
        )

    def invoke(self, messages: list[Message]) -> LLMResponse:
        """Sync wrapper — use asyncio.run()."""
        import asyncio
        return asyncio.run(self.ainvoke(messages))

    def with_structured_output(
        self, schema: type[BaseModel], *, max_retries: int | None = None
    ) -> "MyProviderLLMAdapter":
        """Return a new adapter configured for structured output.

        If your provider doesn't support native structured output,
        return self unchanged — the pipeline will handle JSON parsing.
        """
        return self

    def _convert_messages(self, messages: list[Message]) -> list[dict]:
        """Convert karenina Messages to provider format."""
        result = []
        for msg in messages:
            result.append({
                "role": msg.role.value,
                "content": msg.text,
            })
        return result
```

### Implementing ParserPort

The parser is a **pure executor** — it receives pre-assembled prompt messages from `PromptAssembler` and doesn't build prompts internally:

```python
from typing import TypeVar

from pydantic import BaseModel

from karenina.ports.capabilities import PortCapabilities
from karenina.ports.messages import Message
from karenina.ports.parser import ParsePortResult
from karenina.ports.usage import UsageMetadata
from karenina.schemas.config import ModelConfig

T = TypeVar("T", bound=BaseModel)


class MyProviderParserAdapter:
    """Parser adapter for MyProvider.

    Implements ParserPort protocol via duck typing.
    """

    def __init__(self, model_config: ModelConfig) -> None:
        self._config = model_config

    @property
    def capabilities(self) -> PortCapabilities:
        return PortCapabilities(
            supports_system_prompt=True,
            supports_structured_output=False,
        )

    async def aparse_to_pydantic(
        self, messages: list[Message], schema: type[T]
    ) -> ParsePortResult[T]:
        """Parse LLM response into a Pydantic model.

        The messages are pre-assembled by PromptAssembler with task
        instructions, adapter instructions, and user instructions.
        """
        # Call your provider with the pre-assembled messages
        response = await self._call_llm(messages)

        # Parse the JSON response into the schema
        import json
        data = json.loads(response.text)
        parsed = schema.model_validate(data)

        return ParsePortResult(
            parsed=parsed,
            usage=UsageMetadata(
                input_tokens=response.usage.prompt_tokens,
                output_tokens=response.usage.completion_tokens,
            ),
        )

    def parse_to_pydantic(
        self, messages: list[Message], schema: type[T]
    ) -> ParsePortResult[T]:
        """Sync wrapper."""
        import asyncio
        return asyncio.run(self.aparse_to_pydantic(messages, schema))
```

### Implementing AgentPort

The most complex port — handles multi-turn execution with optional tools and MCP servers:

```python
from karenina.ports.agent import AgentConfig, AgentResult
from karenina.ports.messages import Message
from karenina.ports.usage import UsageMetadata
from karenina.schemas.config import ModelConfig


class MyProviderAgentAdapter:
    """Agent adapter for MyProvider.

    Implements AgentPort protocol via duck typing.
    """

    def __init__(self, model_config: ModelConfig) -> None:
        self._config = model_config

    async def arun(
        self,
        messages: list[Message],
        tools: list | None = None,
        mcp_servers: dict | None = None,
        config: AgentConfig | None = None,
    ) -> AgentResult:
        """Execute an agent loop."""
        config = config or AgentConfig()

        # Run your agent with the given messages and tools
        # ... your implementation here ...

        return AgentResult(
            final_response="The agent's final response",
            raw_trace="--- AI Message ---\nThe response...",
            trace_messages=[Message.assistant("The response...")],
            usage=UsageMetadata(),
            turns=1,
            limit_reached=False,
        )

    def run(
        self,
        messages: list[Message],
        tools: list | None = None,
        mcp_servers: dict | None = None,
        config: AgentConfig | None = None,
    ) -> AgentResult:
        """Sync wrapper."""
        import asyncio
        return asyncio.run(self.arun(messages, tools, mcp_servers, config))
```

---

## Step 3: Write the Registration Module

The registration module connects your adapter to the factory system. This is the only required file.

### Availability Checker

Check whether your provider's SDK is installed:

```python
# karenina/src/karenina/adapters/my_provider/registration.py
import logging

from karenina.adapters.registry import (
    AdapterAvailability,
    AdapterRegistry,
    AdapterSpec,
)

logger = logging.getLogger(__name__)


def _check_availability() -> AdapterAvailability:
    """Check if MyProvider SDK is installed."""
    try:
        import my_provider_sdk  # noqa: F401
        return AdapterAvailability(
            available=True,
            reason="my_provider_sdk is installed and available",
        )
    except ImportError:
        return AdapterAvailability(
            available=False,
            reason=(
                "my_provider_sdk not installed. "
                "Install with: pip install my-provider-sdk"
            ),
            fallback_interface="langchain",  # Fall back to langchain
        )
```

### Factory Functions

Each factory takes a `ModelConfig` and returns a port implementation. Use **lazy imports** to avoid importing heavy SDKs at module level:

```python
from karenina.ports.agent import AgentPort
from karenina.ports.llm import LLMPort
from karenina.ports.parser import ParserPort
from karenina.schemas.config import ModelConfig


def _create_agent(config: ModelConfig) -> AgentPort:
    from karenina.adapters.my_provider.agent import MyProviderAgentAdapter
    return MyProviderAgentAdapter(config)


def _create_llm(config: ModelConfig) -> LLMPort:
    from karenina.adapters.my_provider.llm import MyProviderLLMAdapter
    return MyProviderLLMAdapter(config)


def _create_parser(config: ModelConfig) -> ParserPort:
    from karenina.adapters.my_provider.parser import MyProviderParserAdapter
    return MyProviderParserAdapter(config)
```

### Register the AdapterSpec

```python
_my_provider_spec = AdapterSpec(
    interface="my_provider",
    description="MyProvider adapter for custom LLM backend",
    agent_factory=_create_agent,
    llm_factory=_create_llm,
    parser_factory=_create_parser,
    availability_checker=_check_availability,
    fallback_interface="langchain",
    routes_to=None,
    supports_mcp=False,      # Set True if your adapter handles MCP
    supports_tools=True,
)

AdapterRegistry.register(_my_provider_spec)

logger.debug("Registered my_provider adapter with AdapterRegistry")
```

### Trigger Prompt Instruction Registration

At the end of `registration.py`, import your prompt modules so their instruction registrations execute:

```python
# Import prompt modules to trigger adapter instruction registration
import karenina.adapters.my_provider.prompts.parsing  # noqa: E402, F401
import karenina.adapters.my_provider.prompts.rubric  # noqa: E402, F401
import karenina.adapters.my_provider.prompts.deep_judgment  # noqa: E402, F401
```

### Enable Lazy Discovery

Add your adapter to the registry's initialization list in `karenina/src/karenina/adapters/registry.py` so it gets discovered automatically:

```python
# In AdapterRegistry._ensure_initialized():
import karenina.adapters.my_provider.registration  # noqa: F401
```

---

## Step 4: Register Prompt Instructions

Adapter instructions customize how prompts are assembled for your adapter. The `PromptAssembler` applies them after task instructions and before user instructions.

### How Prompt Assembly Works

```
Final prompt = Task instructions + Adapter instructions + User instructions
                                    ^^^^^^^^^^^^^^^^^^^^
                                    Your additions go here
```

Adapter instructions **append** text — they never replace the base prompt. This means your additions refine the instructions for your provider's specific capabilities.

### Create an Instruction Class

An instruction class provides `system_addition` and `user_addition` properties:

```python
# karenina/src/karenina/adapters/my_provider/prompts/parsing.py
from dataclasses import dataclass
from typing import Any

from karenina.ports.adapter_instruction import AdapterInstructionRegistry


@dataclass
class _MyProviderParsingInstruction:
    """Parsing instructions for MyProvider.

    Adds format guidance appropriate for this provider's capabilities.
    """

    json_schema: dict[str, Any] | None = None

    @property
    def system_addition(self) -> str:
        """Text appended to the system prompt."""
        return (
            "Return your response as a valid JSON object. "
            "Do not include markdown fences or surrounding text."
        )

    @property
    def user_addition(self) -> str:
        """Text appended to the user prompt."""
        if self.json_schema is None:
            return ""
        import json
        schema_json = json.dumps(self.json_schema, indent=2)
        return f"Your response must conform to this JSON schema:\n```json\n{schema_json}\n```"
```

### Write the Factory Function

The factory receives keyword arguments from the `instruction_context` dict. Common keys include `json_schema`, `format_instructions`, and `model_capabilities`:

```python
def _my_provider_parsing_factory(**kwargs: object) -> _MyProviderParsingInstruction:
    """Factory producing MyProvider parsing instructions."""
    return _MyProviderParsingInstruction(
        json_schema=kwargs.get("json_schema"),
    )
```

### Register with AdapterInstructionRegistry

```python
AdapterInstructionRegistry.register(
    "my_provider", "parsing", _my_provider_parsing_factory
)
```

The first argument is the interface name, the second is the `PromptTask` value string. Common task values:

| Task | Pipeline Stage | When to Register |
|------|---------------|-----------------|
| `parsing` | Stage 7 (parse_template) | Always — controls how your adapter receives parsing instructions |
| `rubric_llm_trait_batch` | Stage 11 | If rubric evaluation needs adapter-specific tuning |
| `rubric_llm_trait_single` | Stage 11 | Same, for single-trait evaluation |
| `rubric_literal_trait_batch` | Stage 11 | For literal trait evaluation |
| `rubric_literal_trait_single` | Stage 11 | Same, for single-trait evaluation |
| `rubric_metric_trait` | Stage 11 | For metric trait evaluation |
| `dj_template_excerpt` | Deep judgment | For excerpt extraction |
| `dj_template_reasoning` | Deep judgment | For reasoning generation |
| `dj_template_extraction` | Deep judgment | For parameter extraction |
| `dj_rubric_excerpt` | Deep judgment | For rubric excerpt extraction |
| `dj_rubric_reasoning` | Deep judgment | For rubric reasoning |
| `dj_rubric_scoring` | Deep judgment | For rubric scoring |
| `dj_rubric_aggregation` | Deep judgment | For rubric aggregation |

For most adapters, registering `parsing` instructions is sufficient. Rubric and deep judgment instructions are optional refinements.

### When Your Provider Has Native Structured Output

If your provider supports structured output natively (like Anthropic's `messages.parse`), your parsing instructions can be minimal — strip the JSON schema since the provider handles it:

```python
@dataclass
class _MyProviderParsingInstruction:
    @property
    def system_addition(self) -> str:
        return "Extract only what's stated — don't infer."

    @property
    def user_addition(self) -> str:
        return ""  # No schema needed — provider handles it natively
```

---

## Step 5: Use Your Adapter

Once registered, your adapter is available through the standard factory functions:

```python
from karenina.adapters.factory import get_agent, get_llm, get_parser
from karenina.schemas.config import ModelConfig

config = ModelConfig(
    id="my-model",
    model_name="my-model-v1",
    interface="my_provider",
)

llm = get_llm(config)           # Returns your MyProviderLLMAdapter
parser = get_parser(config)     # Returns your MyProviderParserAdapter
agent = get_agent(config)       # Returns your MyProviderAgentAdapter
```

It also works in `VerificationConfig`:

```python
from karenina.schemas.verification import VerificationConfig

config = VerificationConfig(
    answering_models=[ModelConfig(
        id="my-model",
        model_name="my-model-v1",
        interface="my_provider",
    )],
    parsing_models=[ModelConfig(
        id="my-parser",
        model_name="my-model-v1",
        interface="my_provider",
    )],
)
```

---

## AdapterSpec Fields Reference

| Field | Type | Description |
|-------|------|-------------|
| `interface` | `str` | Interface name used in `ModelConfig.interface` |
| `description` | `str` | Human-readable description |
| `agent_factory` | `Callable | None` | Factory for `AgentPort` — `None` if unsupported |
| `llm_factory` | `Callable | None` | Factory for `LLMPort` — `None` if unsupported |
| `parser_factory` | `Callable | None` | Factory for `ParserPort` — `None` if unsupported |
| `availability_checker` | `Callable | None` | Returns `AdapterAvailability` — `None` means always available |
| `fallback_interface` | `str | None` | Interface to fall back to when unavailable |
| `routes_to` | `str | None` | Interface this one delegates to (for routing interfaces) |
| `supports_mcp` | `bool` | Whether the adapter can connect to MCP servers |
| `supports_tools` | `bool` | Whether the adapter supports tool use |

---

## Key Design Patterns

**Lazy imports** — Import your provider's SDK inside factory functions and adapter methods, not at module level. This prevents import errors when the SDK isn't installed and keeps startup fast.

**Duck typing** — Don't inherit from port protocols. Implement the method signatures and Python's `@runtime_checkable` Protocol system handles the rest. This keeps your adapter independent of karenina's internal types.

**Async-first** — Implement `ainvoke`/`aparse_to_pydantic`/`arun` as the primary API. Sync wrappers (`invoke`/`parse_to_pydantic`/`run`) should call the async version via `asyncio.run()`.

**Message conversion** — Convert between karenina's `Message` type and your provider's message format in a dedicated method. The `Message` class provides `role`, `content` (list of content blocks), and `text` (convenience string property).

**Usage tracking** — Always populate `UsageMetadata` with at least `input_tokens` and `output_tokens`. The pipeline uses these for cost tracking and reporting.

**Adapter cleanup** — If your adapter holds resources (connections, sessions), implement an `aclose()` method. The registry calls `cleanup_all_adapters()` at shutdown to close tracked instances.

---

## Related

- [Adapter Architecture](index.md) — Hexagonal architecture overview
- [Port Types](ports.md) — Complete protocol signatures for all three ports
- [Available Adapters](available-adapters.md) — Existing adapter implementations for reference
- [Prompt Assembly](../11-advanced-pipeline/prompt-assembly.md) — How adapter instructions integrate into the prompt pipeline
- [Custom Stages](../11-advanced-pipeline/custom-stages.md) — Extending the verification pipeline with custom stages
