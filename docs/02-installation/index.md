# Installation

This guide covers installing Karenina and verifying your setup.

---

## Requirements

- **Python 3.11+** (3.11 and 3.12 are tested)
- **pip** or [**uv**](https://docs.astral.sh/uv/) (recommended)
- Supported platforms: **Linux**, **macOS**, **Windows**

---

## Basic Installation

Karenina is not yet published to PyPI. Install from the Git repository:

=== "uv (recommended)"

    ```bash
    uv pip install "karenina @ git+https://github.com/biocypher/karenina.git"
    ```

=== "pip"

    ```bash
    pip install "karenina @ git+https://github.com/biocypher/karenina.git"
    ```

This installs the core package with all required dependencies, including MCP client
support (`mcp`), LangChain integrations, Anthropic SDK, and the CLI.

---

## Optional Dependencies

Karenina offers optional extras for specific use cases. Install them by appending
the extra name in brackets:

```bash
# Example: install with embedding support
uv pip install "karenina[embeddings] @ git+https://github.com/biocypher/karenina.git"

# Multiple extras
uv pip install "karenina[embeddings,examples] @ git+https://github.com/biocypher/karenina.git"
```

| Extra | Packages | Purpose |
|-------|----------|---------|
| `dev` | pytest, ruff, mypy, vulture, pre-commit, mkdocs, mkdocs-material, mkdocstrings, ipython | Development and testing tools |
| `search` | langchain-community, tavily-python | Web search integration for agentic verification |
| `examples` | jupyter, ipykernel | Running example notebooks |
| `embeddings` | sentence-transformers | Embedding similarity checks (pipeline stage 9) |
| `gepa` | gepa | GEPA prompt optimization integration |

!!! note "MCP is a core dependency"
    The MCP client library (`mcp>=1.25.0`) is included in the base installation.
    You do **not** need a separate extra to use MCP transport.

---

## Verifying Installation

After installing, verify that everything works:

```bash
# Check CLI is available
karenina --version

# Check Python import
python -c "import karenina; print(karenina.__version__)"
```

Both commands should print the installed version (currently `0.1.0`).

---

## Development Setup

For contributors working on the Karenina codebase:

```bash
# Clone the repository
git clone https://github.com/biocypher/karenina.git
cd karenina

# Install in editable mode with dev dependencies
uv pip install -e ".[dev]"

# Run the test suite
uv run pytest tests/ -x -q

# Run linters
uv run ruff check src/ tests/
```

The `-e` flag installs in editable mode — changes to source files take effect
immediately without reinstalling.

---

## API Key Configuration

Karenina calls LLM providers during verification. Set the API keys for the
providers you plan to use:

```bash
# In your shell or a .env file in your working directory
export OPENAI_API_KEY="sk-..."
export ANTHROPIC_API_KEY="sk-ant-..."
export GOOGLE_API_KEY="AI..."
```

Alternatively, use `karenina init` to generate a `.env` template with all
supported variables. See [Configuration](../03-configuration/index.md) in the
Workflows section for details on the full configuration hierarchy.

---

## Troubleshooting

### `karenina` command not found

The CLI entry point may not be on your `PATH`. Try:

```bash
# Run via uv
uv run karenina --version

# Or via python module
python -m karenina --version
```

If using `pip install`, ensure your Python scripts directory is on `PATH`.

### Python version too old

Karenina requires Python 3.11+. Check your version:

```bash
python --version
```

If needed, install a newer Python with uv:

```bash
uv python install 3.11
```

### Import errors after installation

Reinstall to ensure all dependencies are resolved:

```bash
uv pip install -e "." --reinstall
```

### LLM provider errors

If verification fails with authentication errors, check that the correct API key
is set for your chosen provider. Karenina reads keys from environment variables
and `.env` files via `python-dotenv`.

```bash
# Verify your key is set
echo $OPENAI_API_KEY
echo $ANTHROPIC_API_KEY
```

---

## Next Steps

- **[Quick Start](../quickstart.md)** — End-to-end walkthrough: create a benchmark, run verification, inspect results
- **[Configuration](../03-configuration/index.md)** — Set up presets, environment variables, and workspace
- **[Core Concepts](../04-core-concepts/index.md)** — Understand checkpoints, templates, and rubrics
- **[Creating Benchmarks](../05-creating-benchmarks/index.md)** — Build your first benchmark
