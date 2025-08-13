# Installation

## Requirements

- Python 3.11+

## Install from PyPI

```bash
pip install karenina
```

## Dev install (contributors)

```bash
git clone https://github.com/biocypher/karenina.git
cd karenina
uv sync
uv pip install -e "[dev]"
pre-commit install
```

## Environment

Set provider keys as needed:

```bash
export OPENAI_API_KEY=...
export GOOGLE_API_KEY=...
export ANTHROPIC_API_KEY=...
export OPENROUTER_API_KEY=...
```

## Verify install

```python
import karenina
from karenina.benchmark import Benchmark

print(karenina.__version__)
print(Benchmark.create("_test_"))
```
