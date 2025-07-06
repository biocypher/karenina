# Installation

## Requirements

- Python 3.8+
- pip or uv package manager

## Install from Source

```bash
git clone https://github.com/example/karenina.git
cd karenina
uv install -e .
```

## Development Installation

```bash
cd karenina
make dev
```

## Dependencies

Core dependencies:
- `pydantic` - Data validation and settings management
- `pandas` - Data manipulation and analysis
- `langchain` - LLM framework and provider abstraction
- `langchain-openai` - OpenAI integration
- `langchain-google-genai` - Google AI integration
- `langchain-anthropic` - Anthropic integration
- `python-dotenv` - Environment variable management
- `tqdm` - Progress bars for long-running operations

## Environment Variables

Configure API keys for LLM providers:

```bash
export OPENAI_API_KEY="your-openai-key"
export GOOGLE_API_KEY="your-google-key"
export ANTHROPIC_API_KEY="your-anthropic-key"
export OPENROUTER_API_KEY="your-openrouter-key"
```

## Verification

```python
import karenina
print(karenina.__version__)
```
