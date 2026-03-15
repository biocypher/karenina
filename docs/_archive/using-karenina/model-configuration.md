# Model Configuration

This guide covers how to configure LLM models in Karenina using `ModelConfig`. You'll learn about model providers, interfaces, advanced parameters, and how to pass vendor-specific options.

**Quick Navigation:**

- [What is ModelConfig?](#what-is-modelconfig) - Core concepts and use cases
- [Basic ModelConfig](#basic-modelconfig) - Minimal configuration example
- [ModelConfig Parameters](#modelconfig-parameters) - Required and optional parameters
- [Interfaces](#interfaces) - LangChain, OpenAI endpoint, OpenRouter, manual
- [Model Providers](#model-providers) - OpenAI, Google, Anthropic configuration
- [Temperature Parameter](#temperature-parameter) - Controlling randomness and determinism
- [Extra Keyword Arguments](#extra-keyword-arguments) - Vendor-specific options and API keys
- [System Prompts](#system-prompts) - Custom system prompt configuration
- [MCP Tool Integration](#mcp-tool-integration) - Enable tool use during answer generation
- [Common Configuration Patterns](#common-configuration-patterns) - Typical setup examples
- [Best Practices](#best-practices) - Recommendations for benchmarking and API keys
- [Troubleshooting](#troubleshooting) - Common errors and solutions

---

## What is ModelConfig?

`ModelConfig` is the configuration object that defines which LLM to use and how to interact with it. It's used in three key places:

1. **Template generation**: LLMs that generate answer templates for questions
2. **Answering models**: LLMs that generate responses to benchmark questions
3. **Parsing models** (judges): LLMs that extract structured data from responses using templates

A single `ModelConfig` can be used for all three roles, or you can use different models for each role.

---

## Basic ModelConfig

The simplest model configuration:

```python
from karenina.schemas import ModelConfig

model_config = ModelConfig(
    id="my-model",                 # Unique identifier
    model_name="gpt-4.1-mini",     # Model name
    model_provider="openai",       # Provider: openai, google_genai, anthropic, etc.
    temperature=0.0,               # Temperature (0.0 = deterministic)
    interface="langchain"          # Interface: langchain, openai_endpoint, openrouter, manual
)
```

---

## ModelConfig Parameters

### Required Parameters

| Parameter | Type | Description | Example |
|-----------|------|-------------|---------|
| `id` | `str` | Unique identifier for this model configuration | `"gpt-4.1-mini"`, `"my-custom-model"` |
| `model_name` | `str` | Full model name as recognized by the provider | `"gpt-4.1-mini"`, `"claude-sonnet-4.5"`, `"gemini-2.5-flash"` |
| `interface` | `str` | Interface type (see [Interfaces](#interfaces)) | `"langchain"`, `"openai_endpoint"`, `"openrouter"`, `"manual"` |

### Optional Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `model_provider` | `str` | Required for `langchain` | Provider name (see [Providers](#model-providers)) |
| `temperature` | `float` | `0.1` | Sampling temperature (0.0-1.0). Use 0.0 for deterministic benchmarking |
| `system_prompt` | `str` | `None` | Optional system prompt override |
| `max_retries` | `int` | `2` | Maximum retry attempts for API calls |
| `endpoint_base_url` | `str` | `None` | Custom endpoint URL (for `openai_endpoint` interface) |
| `endpoint_api_key` | `SecretStr` | `None` | API key for custom endpoint (for `openai_endpoint` interface) |
| `mcp_urls_dict` | `dict[str, str]` | `None` | MCP server URLs for tool use |
| `mcp_tool_filter` | `list[str]` | `None` | Filter specific MCP tools |
| `extra_kwargs` | `dict[str, Any]` | `None` | Additional keyword arguments (see [Extra Keyword Arguments](#extra-keyword-arguments)) |
| `manual_traces` | `ManualTraces` | `None` | Pre-computed traces (for `manual` interface) |

---

## Interfaces

Karenina supports four interfaces for connecting to LLMs. Choose based on your use case:

### 1. LangChain Interface (`langchain`)

**Default and recommended interface** for most use cases. Uses LangChain's model integrations.

```python
model_config = ModelConfig(
    id="gpt-4.1-mini",
    model_name="gpt-4.1-mini",
    model_provider="openai",
    interface="langchain",
    temperature=0.0
)
```

**When to use**:

- ✅ Working with OpenAI, Google, Anthropic, or other LangChain-supported providers
- ✅ Need standardized interface across multiple providers
- ✅ Want built-in retry logic and error handling

**Requirements**:

- API key must be set in environment variables (see [Configuration](../configuration.md#api-keys))
- Or API key can be passed via `extra_kwargs` (see [Extra Keyword Arguments](#extra-keyword-arguments))

**Supported providers**: `openai`, `google_genai`, `anthropic`, and others (see [langchain documentation](https://reference.langchain.com/python/langchain/models/?h=init_chat#langchain.chat_models.init_chat_model(model)))

---

### 2. OpenAI Endpoint Interface (`openai_endpoint`)

Use this interface for **custom endpoints** that implement the OpenAI-compatible API (e.g., vLLM, Ollama, local models).

```python
model_config = ModelConfig(
    id="local-model",
    model_name="glm-4.6",
    interface="openai_endpoint",
    endpoint_base_url="http://localhost:8000/v1",
    endpoint_api_key="dummy-key",  # Some servers require a key
    temperature=0.0
)
```

**When to use**:

- ✅ Running local models (vLLM, Ollama, etc.)
- ✅ Using custom inference servers
- ✅ OpenAI-compatible APIs

**Requirements**:

- `endpoint_base_url` must point to your server
- Some servers require `endpoint_api_key` (even if just a dummy value)

---

### 3. OpenRouter Interface (`openrouter`)

Use OpenRouter for unified access to multiple LLM providers through a single API.

```python
model_config = ModelConfig(
    id="claude-via-openrouter",
    model_name="anthropic/claude-3.5-sonnet",
    interface="openrouter",
    temperature=0.0
)
```

**When to use**:

- ✅ Want unified billing across multiple providers
- ✅ Want to switch between providers easily

**Requirements**:
- `OPENROUTER_API_KEY` must be set in environment variables
- pass the api key via `extra_kwargs`

**Note**: `model_provider` is not required for OpenRouter interface since the provider is specified in the `model_name`

---

### 4. Manual Interface (`manual`)

For testing and debugging with pre-computed responses (no LLM API calls). This interface allows you to provide pre-generated answer traces directly to the verification engine.

```python
from karenina.adapters.manual import ManualTraces

# Initialize manual traces
manual_traces = ManualTraces(benchmark)

# Register traces (see Manual Traces guide for details)
manual_traces.register_trace(
    "What is 2+2?",
    "The answer is 4",
    map_to_id=True
)

model_config = ModelConfig(
    interface="manual",
    manual_traces=manual_traces
)
```

**When to use**:

- ✅ Testing workflows without API costs
- ✅ Debugging specific scenarios
- ✅ Evaluating pre-recorded LLM responses
- ✅ Comparing different answer generation approaches

**For comprehensive documentation**, including:

- LangChain message list support
- Tool call metrics extraction
- Batch registration
- Session-based storage

See the **[Manual Traces Guide](../advanced/manual-traces.md)**

---

## Model Providers

Model providers are specified with the `model_provider` parameter (required for `langchain` interface).

### Supported Providers

| Provider | Value | Example Models | API Key Required |
|----------|-------|----------------|------------------|
| OpenAI | `"openai"` | `gpt-4.1-mini`, `gpt-4.1-mini`, `gpt-4-turbo` | `OPENAI_API_KEY` |
| Google | `"google_genai"` | `gemini-2.5-flash`, `gemini-2.5-pro` | `GOOGLE_API_KEY` |
| Anthropic | `"anthropic"` | `claude-4-5-sonnet`, `claude-4-5-opus` | `ANTHROPIC_API_KEY` |

### Example Configurations

**OpenAI (GPT-4.1-mini)**
```python
model_config = ModelConfig(
    id="gpt-4.1-mini",
    model_name="gpt-4.1-mini",
    model_provider="openai",
    interface="langchain",
    temperature=0.0
)
```

**Google (Gemini 2.5 Flash)**
```python
model_config = ModelConfig(
    id="gemini-flash",
    model_name="gemini-2.5-flash",
    model_provider="google_genai",
    interface="langchain",
    temperature=0.0
)
```

**Anthropic (Claude Sonnet 4.5)**
```python
model_config = ModelConfig(
    id="claude-sonnet",
    model_name="claude-sonnet-4.5",
    model_provider="anthropic",
    interface="langchain",
    temperature=0.0
)
```

---

## Temperature Parameter

The `temperature` parameter controls randomness in model outputs:

- **`0.0`** - Fully deterministic (recommended for benchmarking)
- **`0.1-0.3`** - Low randomness (slight variation)
- **`0.7-0.9`** - High randomness (creative responses)
- **`1.0+`** - Maximum randomness

**For benchmarking**: Always use `temperature=0.0` to ensure reproducible results. If you instead want to asess wheter your results are consistent in different runs, you can use higher temperatures and check if the results are consistent.

```python
# Deterministic benchmarking (recommended)
answering_model = ModelConfig(
    id="answering",
    model_name="gpt-4.1-mini",
    model_provider="openai",
    interface="langchain",
    temperature=0.3  # Low randomness
)

# Parsing/judging should always be deterministic
parsing_model = ModelConfig(
    id="parsing",
    model_name="gpt-4.1-mini",
    model_provider="openai",
    interface="langchain",
    temperature=0.0  # Always 0.0 for parsing
)
```

---

## Extra Keyword Arguments

The `extra_kwargs` field allows you to pass additional keyword arguments to the underlying model interface. This is useful for:

- **Passing API keys directly** (alternative to environment variables)
- **Vendor-specific parameters** (custom headers, special options)
- **Advanced model configuration** (thinking mode, chat templates, generation parameters)

### Example 1: Passing API Key Directly

If you don't want to use environment variables for API keys, you can pass them directly:

```python
model_config = ModelConfig(
    id="gemini-with-key",
    model_name="gemini-2.5-flash",
    model_provider="google_genai",
    interface="langchain",
    temperature=0.0,
    extra_kwargs={
        "google_api_key": "your_key"
    }
)
```

**Note**: While this works, storing API keys in environment variables (`.env` files) is still the recommended approach for security. This approach is useful for:

- Testing with multiple API keys
- Temporary key usage
- Programmatic key management

### Example 2: Disabling Thinking Mode

When using local models or custom endpoints that support thinking modes, you can control their behavior:

```python
model_config = ModelConfig(
    id="local-model",
    model_name="glm-4.5-air",
    interface="openai_endpoint",
    endpoint_base_url="http://localhost:8000/v1",
    temperature=0.0,
    extra_kwargs={
        "extra_body": {
            "chat_template_kwargs": {
                "enable_thinking": False
            },
            "separate_reasoning": False
        }
    }
)
```

### Example 3: Passing Generation Parameters

You can pass additional generation parameters that may not be exposed as top-level ModelConfig fields:

```python
model_config = ModelConfig(
    id="custom-params",
    model_name="gpt-4.1-mini",
    model_provider="openai",
    interface="langchain",
    temperature=0.0,
    extra_kwargs={
        "max_tokens": 500,
        "top_p": 0.9,
        "frequency_penalty": 0.1,
        "presence_penalty": 0.1
    }
)
```

### How extra_kwargs Works

The arguments are passed to different places depending on the interface:

| Interface | Where Arguments Go |
|-----------|-------------------|
| `langchain` | Passed to LangChain model constructor |
| `openai_endpoint` | Passed to OpenAI client's chat completion call |
| `openrouter` | Passed to OpenRouter API call |

**Common use cases**:

- Passing API keys without environment variables
- Enabling/disabling vendor-specific features (thinking, streaming, etc.)
- Passing generation parameters (max_tokens, top_p, frequency_penalty, etc.)
- Custom headers or metadata
- Timeout configuration

---

## System Prompts

You can override the default system prompt for template generation, answering, or parsing:

```python
answering_model = ModelConfig(
    id="biology-expert",
    model_name="gpt-4.1-mini",
    model_provider="openai",
    interface="langchain",
    temperature=0.0,
    system_prompt="You are a knowledgeable genomics expert. Provide detailed, accurate answers."
)
```

**When to use custom system prompts**:

- ✅ Domain-specific expertise needed (e.g., "You are a genomics expert")
- ✅ Specific tone or style required
- ✅ Additional context or constraints

**Default system prompts** (if not specified):

- **Template generation**: Instruction to create structured output templates
- **Answering models**: Basic instruction to answer the question
- **Parsing models**: Instruction to extract structured data using the template

---

## MCP Tool Integration

Karenina supports Model Context Protocol (MCP) for tool access during answer generation. LLMs can invoke external tools (web search, database queries, calculations, etc.) through MCP servers.

**Quick configuration example**:

```python
model_config = ModelConfig(
    id="tool-using-model",
    model_name="gpt-4.1-mini",
    model_provider="openai",
    interface="langchain",
    temperature=0.0,
    mcp_urls_dict={
        "calculator": "http://localhost:8001/mcp",
        "search": "http://localhost:8002/mcp"
    },
    mcp_tool_filter=["add", "multiply", "web_search"]  # Optional: whitelist tools
)
```

**Parameters**:

- `mcp_urls_dict`: Maps tool categories to MCP server URLs
- `mcp_tool_filter`: Optional list of allowed tool names

**Supported interfaces**: `langchain`, `openai_endpoint`, `openrouter` (not supported with `manual` interface)

**For comprehensive documentation**, including:

- MCP server setup and structure
- Tool discovery and invocation
- Multiple MCP server configuration
- Health checks and monitoring
- Security and access control

See the **[MCP Integration Guide](../advanced/mcp-integration.md)**

---

## Common Configuration Patterns

### Same Model for All Roles

Use a single model configuration for template generation, answering, and parsing (simplest approach):

```python
model_config = ModelConfig(
    id="gpt-4.1-mini",
    model_name="gpt-4.1-mini",
    model_provider="openai",
    interface="langchain",
    temperature=0.0
)

# Use for template generation
benchmark.generate_templates(model_config=model_config)

# Reuse in verification (see Running Verification guide)
```

### Different Models for Different Roles

Configure different models for specific tasks (optimal for cost/quality):

```python
# High-quality model for answering
answering_model = ModelConfig(
    id="sonnet-4.5",
    model_name="claude-4.5-sonnet",
    model_provider="anthropic",
    interface="langchain",
    temperature=0.0
)

# Fast, cheap model for parsing and template generation
utility_model = ModelConfig(
    id="gpt-4.1-mini",
    model_name="gpt-4.1-mini",
    model_provider="openai",
    interface="langchain",
    temperature=0.0
)

# Use utility model for templates
benchmark.generate_templates(model_config=utility_model)

# Use different models in verification (see Running Verification guide)
```

### Configuring Multiple Models

Create multiple model configurations for comparison testing:

```python
models_to_test = [
    ModelConfig(
        id="gpt-4.1-mini",
        model_name="gpt-4.1-mini",
        model_provider="openai",
        interface="langchain",
        temperature=0.0
    ),
    ModelConfig(
        id="claude-sonnet",
        model_name="claude-sonnet-4.5",
        model_provider="anthropic",
        interface="langchain",
        temperature=0.0
    ),
    ModelConfig(
        id="gemini-flash",
        model_name="gemini-2.5-flash",
        model_provider="google_genai",
        interface="langchain",
        temperature=0.0
    )
]

# Pass to VerificationConfig (see Running Verification guide)
```

### Local Model with Custom Endpoint

Configure a locally-hosted model:

```python
local_model = ModelConfig(
    id="local-qwen",
    model_name="qwen-3-235",
    interface="openai_endpoint",
    endpoint_base_url="http://localhost:8000/v1",
    endpoint_api_key="dummy",
    temperature=0.0
)
```

### Model with Domain Expertise

Configure a model with specialized system prompt:

```python
genomics_model = ModelConfig(
    id="genomics-expert",
    model_name="gpt-4.1-mini",
    model_provider="openai",
    interface="langchain",
    temperature=0.0,
    system_prompt="You are a genomics expert with deep knowledge of molecular biology. Answer concisely with precise scientific terminology."
)
```

**For complete verification workflows**, see the **[Running Verification Guide](verification.md)**

---

## Best Practices

### For Benchmarking

- ✅ Always use `temperature=0.0` for reproducible results
- ✅ Use the same parsing model across different answering models for fair comparison
- ✅ Document your model configurations in your project README
- ✅ Use descriptive `id` values (e.g., "gpt-4.1-mini-biology-expert")

### For API Keys

- ✅ Store API keys in `.env` files (see [Configuration](../configuration.md#api-keys))
- ✅ Use different keys for development and production
- ✅ Rotate keys regularly
- ❌ Never commit API keys to version control
- ⚠️ Only pass keys via `extra_kwargs` when necessary (testing, temporary use)

### For Model Selection

- ✅ Use `gpt-4.1-mini` as the default (fast, cost-effective)
- ✅ Use `gpt-5` or `claude-4-5-sonnet` for higher quality (more expensive)
- ✅ Use same model for all roles initially (simpler)
- ✅ Optimize later: cheaper model for parsing/templates, expensive for answering

### For System Prompts

- ✅ Keep system prompts concise and focused
- ✅ Test different prompts systematically
- ✅ Document any custom prompts in your benchmark metadata
- ❌ Avoid overly long or complex system prompts

---

## Troubleshooting

### API Key Not Found

```
Error: API key not found for provider 'openai'
```

**Solution**: Set the API key in environment variables:
```bash
export OPENAI_API_KEY="sk-..."
```

Or pass directly via `extra_kwargs`:
```python
extra_kwargs={"api_key": "sk-..."}
```

See [Configuration Guide](../configuration.md#api-keys) for more options.

---

### Invalid Model Name

```
Error: Model 'gpt-4-mini' not found
```

**Solution**: Check the correct model name for your provider:

- OpenAI: `gpt-4.1-mini`, `gpt-4.1-mini`, `gpt-4-turbo`
- Google: `gemini-2.5-flash`, `gemini-1.5-pro`
- Anthropic: `claude-sonnet-4.5`, `claude-3-5-sonnet-20241022`

---

### Custom Endpoint Connection Failed

```
Error: Connection refused to http://localhost:8000/v1
```

**Solution**:

1. Verify your inference server is running
2. Check the `endpoint_base_url` is correct
3. Ensure the endpoint implements OpenAI-compatible API
4. Test the endpoint with curl:
```bash
curl http://localhost:8000/v1/models
```

---

### MCP Tools Not Working

```
Error: MCP tools not supported with manual interface
```

**Solution**: MCP tools require `langchain`, `openai_endpoint`, or `openrouter` interface. The `manual` interface does not support dynamic tool use.

---

## Next Steps

- **[Running Verification](verification.md)** - Learn about `VerificationConfig` and running benchmarks
- **[Configuration Guide](../configuration.md)** - Environment variables and API key setup
- **[Configuration Presets](../advanced/presets.md)** - Save and load model configurations
- **[Manual Traces](../advanced/manual-traces.md)** - Detailed guide to pre-computed responses
- **[MCP Integration](../advanced/mcp-integration.md)** - Comprehensive tool integration guide
- **[Templates](templates.md)** - Generate and manage answer templates
