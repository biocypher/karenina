# Install

This page covers how to install Karenina and set up your environment for LLM benchmarking.

## Installation with uv

Karenina is currently available through `uv`, Python's fast package manager. To install Karenina:

```bash
# Install uv if you haven't already
curl -LsSf https://astral.sh/uv/install.sh | sh

# Install karenina
uv add karenina
```

## Setting up Environment Variables

Karenina requires API keys for various LLM providers to be available as environment variables. Set up the providers you plan to use:

### OpenAI
```bash
export OPENAI_API_KEY="your-openai-api-key-here"
```

### Google (Gemini)
```bash
export GOOGLE_API_KEY="your-google-api-key-here"
```

### Anthropic (Claude)
```bash
export ANTHROPIC_API_KEY="your-anthropic-api-key-here"
```

### OpenRouter (Optional)
```bash
export OPENROUTER_API_KEY="your-openrouter-api-key-here"
```

## Verify Installation

To verify that Karenina is installed correctly and can access your LLM providers:

```python
from karenina import Benchmark

# Create a simple benchmark to test
benchmark = Benchmark(name="test-benchmark")
print(f"Benchmark created: {benchmark.name}")
```

You're now ready to start creating benchmarks with Karenina!
