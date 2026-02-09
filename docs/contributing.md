# Contributing

This page covers how to set up a development environment, run tests, and contribute to karenina.

## Development Setup

Clone the repository and install with development dependencies:

```bash
git clone https://github.com/your-org/karenina.git
cd karenina
uv pip install -e '.[dev]'
```

This installs the package in editable mode along with all testing and development tools (pytest, ruff, mypy, etc.).

## Running Tests

```bash
uv run pytest tests/ -x -q
```

Key flags:

| Flag | Purpose |
|------|---------|
| `-x` | Stop on first failure |
| `-q` | Quiet output |
| `-k "pattern"` | Run tests matching a pattern |
| `--co` | List tests without running them |

Tests use captured LLM response fixtures to avoid live API calls. See `tests/fixtures/llm_responses/README.md` for fixture documentation.

## Linting and Type Checking

```bash
# Lint with ruff
uv run ruff check src/ tests/

# Auto-fix lint issues
uv run ruff check src/ tests/ --fix

# Type checking with mypy
uv run mypy src/
```

## Extending Karenina

Karenina is designed for extension in two main areas:

### Adding Pipeline Stages

The verification pipeline uses a stage-based architecture where each stage implements the `VerificationStage` protocol. You can add custom stages for new checks, validations, or evaluation steps.

See [Advanced Pipeline](11-advanced-pipeline/index.md) for the full guide, including:

- [13 Stages in Detail](11-advanced-pipeline/stages.md) --- what each stage does
- [Writing Custom Stages](11-advanced-pipeline/custom-stages.md) --- the `VerificationStage` protocol, `VerificationContext`, artifact keys, and a complete example
- [Prompt Assembly](11-advanced-pipeline/prompt-assembly.md) --- the tri-section prompt system

### Creating Adapters

Adapters connect the pipeline to LLM backends through three port protocols: `LLMPort`, `ParserPort`, and `AgentPort`. You can add adapters for new LLM providers or custom backends.

See [Advanced Adapters](12-advanced-adapters/index.md) for the full guide, including:

- [Port Types](12-advanced-adapters/ports.md) --- protocol signatures for all three ports
- [Available Adapters](12-advanced-adapters/available-adapters.md) --- existing adapter implementations and capabilities
- [Writing Custom Adapters](12-advanced-adapters/writing-adapters.md) --- step-by-step guide with registration, factory functions, and prompt instructions

## Pull Request Process

1. Create a feature branch from `main`
2. Make your changes with tests
3. Run `uv run pytest tests/ -x -q` and `uv run ruff check src/ tests/` to verify
4. Submit a pull request with a clear description of the change

## Project Structure

```
src/karenina/
├── benchmark/          # Core benchmarking (verification pipeline, authoring)
├── ports/              # Protocol interfaces (LLMPort, AgentPort, ParserPort)
├── adapters/           # Backend implementations (LangChain, Claude SDK, etc.)
├── schemas/            # Pydantic models (config, entities, results)
├── storage/            # Database layer (SQLAlchemy)
├── cli/                # Typer CLI commands
└── utils/              # Shared utilities
```

## Related

- [Advanced Pipeline](11-advanced-pipeline/index.md) --- pipeline architecture and extension
- [Advanced Adapters](12-advanced-adapters/index.md) --- adapter architecture and extension
- [Installation](getting-started/installation.md) --- basic setup and dependencies
