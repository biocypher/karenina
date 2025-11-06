# Installation

This guide covers installing Karenina on your system.

---

## Prerequisites

- Python 3.10 or higher
- Git
- `uv` (Python's fast package manager)

---

## Install uv

If you don't have `uv` installed:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

For other installation methods, see [uv's documentation](https://docs.astral.sh/uv/getting-started/installation/).

---

## Install Karenina

Karenina is not yet published to PyPI. Install from the GitHub repository:

```bash
# Clone the repository
git clone https://github.com/biocypher/karenina.git
cd karenina

# Install with uv
uv pip install -e .
```

The `-e` flag installs in editable mode, allowing you to pull updates and see changes immediately without reinstalling.

---

## Verify Installation

Test that Karenina is installed correctly:

```python
from karenina import Benchmark

# Create a simple benchmark
benchmark = Benchmark.create(
    name="test-benchmark",
    description="Installation verification",
    version="1.0.0"
)

print(f"✓ Karenina installed successfully!")
print(f"✓ Benchmark created: {benchmark.name}")
```

---

## Updating Karenina

Pull the latest changes from GitHub:

```bash
cd karenina
git pull origin main
```

The editable installation (`-e`) means changes take effect immediately without reinstalling.

---

## Development Installation

For contributing to Karenina:

```bash
# Clone repository
git clone https://github.com/biocypher/karenina.git
cd karenina

# Install with dev dependencies
uv pip install -e ".[dev]"

# Run tests
uv run pytest

# Run linters
uv run ruff check .
```

See the [Contributing Guide](contributing.md) for development workflow details.

---

## Troubleshooting

### `uv` command not found

```bash
# Restart shell or reload config
source ~/.bashrc  # or ~/.zshrc

# Or reinstall uv
curl -LsSf https://astral.sh/uv/install.sh | sh
```

### Python version too old

```bash
# Check version
python --version

# Install Python 3.11 via uv
uv python install 3.11
```

### Import errors

```bash
# Reinstall in editable mode
cd karenina
uv pip install -e .
```

For configuration and API key setup, see [Configuration](configuration.md).

---

## Next Steps

After installation, you need to configure Karenina:

1. **[Configuration](configuration.md)** - **Start here!** Set up API keys and configure Karenina
2. **[Quick Start](quickstart.md)** - Create your first benchmark
3. **[User Guides](using-karenina/defining-benchmark.md)** - Learn how to use Karenina
4. **[Advanced Features](advanced/)** - Explore advanced capabilities
