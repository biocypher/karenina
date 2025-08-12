# Karenina Development Guide

This guide provides comprehensive information for developers who want to contribute to or extend Karenina.

## Table of Contents

- [Development Environment Setup](#development-environment-setup)
- [Project Structure](#project-structure)
- [Development Workflow](#development-workflow)
- [Code Standards](#code-standards)
- [Testing](#testing)
- [Documentation](#documentation)
- [Debugging](#debugging)
- [Performance Optimization](#performance-optimization)
- [Contributing Guidelines](#contributing-guidelines)
- [Release Process](#release-process)

## Development Environment Setup

### Prerequisites

- Python 3.11 or higher
- [uv](https://github.com/astral-sh/uv) package manager (recommended)
- Git for version control

### Initial Setup

```bash
# Clone the repository
git clone https://github.com/biocypher/karenina.git
cd karenina

# Install uv if you haven't already
curl -LsSf https://astral.sh/uv/install.sh | sh

# Create virtual environment and install dependencies
uv sync

# Install development dependencies
uv add --dev pytest pytest-cov pytest-asyncio pytest-mock ruff mypy pre-commit

# Install pre-commit hooks
pre-commit install

# Verify installation
make check
```

### IDE Configuration

#### VS Code Setup

Create `.vscode/settings.json`:

```json
{
    "python.defaultInterpreterPath": "./.venv/bin/python",
    "python.linting.enabled": true,
    "python.linting.ruffEnabled": true,
    "python.formatting.provider": "ruff",
    "python.linting.mypyEnabled": true,
    "editor.formatOnSave": true,
    "python.testing.pytestEnabled": true,
    "python.testing.pytestArgs": [
        "tests"
    ],
    "files.exclude": {
        "**/__pycache__": true,
        "**/*.pyc": true,
        ".mypy_cache": true,
        ".pytest_cache": true,
        "htmlcov": true
    }
}
```

#### PyCharm Setup

1. Open the project in PyCharm
2. Configure the interpreter to use `.venv/bin/python`
3. Enable Ruff for linting: Settings → Tools → External Tools
4. Configure pytest as the test runner
5. Enable mypy type checking

### Environment Variables for Development

Create a `.env` file in the project root:

```bash
# LLM Provider API Keys (for testing)
OPENAI_API_KEY=your-openai-api-key
GOOGLE_API_KEY=your-google-api-key
ANTHROPIC_API_KEY=your-anthropic-api-key
OPENROUTER_API_KEY=your-openrouter-api-key

# Development settings
PYTHONPATH=./src
KARENINA_DEBUG=true
KARENINA_LOG_LEVEL=DEBUG
```

## Project Structure

### Directory Layout

```
karenina/
├── src/karenina/           # Main package source
│   ├── __init__.py
│   ├── llm/               # LLM interface module
│   ├── questions/         # Question processing
│   ├── answers/           # Answer template generation
│   ├── benchmark/         # Verification system
│   ├── schemas/           # Data models
│   ├── prompts/           # LLM prompts
│   └── utils/             # Utility functions
├── tests/                 # Test suite
│   ├── __init__.py
│   ├── unit/              # Unit tests
│   ├── integration/       # Integration tests
│   ├── fixtures/          # Test fixtures
│   └── conftest.py        # Test configuration
├── docs/                  # Documentation
│   ├── api/               # API documentation
│   ├── guides/            # User guides
│   └── examples/          # Code examples
├── pyproject.toml         # Project configuration
├── Makefile               # Development commands
├── README.md              # Project overview
└── CHANGELOG.md           # Version history
```

### Module Organization

Each module follows a consistent structure:

```
module_name/
├── __init__.py           # Public API exports
├── core_functionality.py # Main implementation
├── models.py            # Data models (if needed)
├── exceptions.py        # Module-specific exceptions
└── utils.py             # Module utilities
```

## Development Workflow

### Branch Strategy

We use a simplified Git flow:

- `main`: Stable release branch
- `develop`: Integration branch for new features
- `feature/feature-name`: Feature development branches
- `bugfix/bug-description`: Bug fix branches
- `hotfix/critical-issue`: Critical production fixes

### Development Process

1. **Create a feature branch**:
   ```bash
   git checkout develop
   git pull origin develop
   git checkout -b feature/your-feature-name
   ```

2. **Make your changes** following code standards

3. **Run the test suite**:
   ```bash
   make check  # Runs all quality checks
   ```

4. **Commit your changes**:
   ```bash
   git add .
   git commit -m "feat: add new feature description"
   ```

5. **Push and create PR**:
   ```bash
   git push origin feature/your-feature-name
   # Create PR through GitHub interface
   ```

### Commit Message Convention

We follow [Conventional Commits](https://www.conventionalcommits.org/):

```
type(scope): description

[optional body]

[optional footer]
```

Types:
- `feat`: New features
- `fix`: Bug fixes
- `docs`: Documentation changes
- `style`: Code style changes
- `refactor`: Code refactoring
- `test`: Test additions/modifications
- `chore`: Build process or auxiliary tool changes

Examples:
```bash
feat(llm): add support for new OpenAI models
fix(questions): handle empty CSV files gracefully
docs(api): update function documentation
test(benchmark): add integration tests for verification
```

## Code Standards

### Python Style Guidelines

We follow PEP 8 with some modifications enforced by Ruff:

- Line length: 120 characters
- Use double quotes for strings
- Use trailing commas in multi-line constructs
- Import organization: standard library, third-party, local imports

### Type Annotations

All public functions must have type annotations:

```python
from typing import List, Dict, Optional, Union

def process_questions(
    questions: List[Question],
    config: VerificationConfig,
    batch_size: int = 10
) -> Dict[str, VerificationResult]:
    """Process questions with proper type hints."""
    pass
```

### Error Handling

Use specific exceptions and provide meaningful error messages:

```python
class KareninaError(Exception):
    """Base exception for Karenina."""
    pass

class QuestionExtractionError(KareninaError):
    """Raised when question extraction fails."""
    pass

def extract_questions(file_path: str) -> List[Question]:
    """Extract questions with proper error handling."""
    if not Path(file_path).exists():
        raise QuestionExtractionError(
            f"Question file not found: {file_path}"
        )

    try:
        # Processing logic
        pass
    except Exception as e:
        raise QuestionExtractionError(
            f"Failed to extract questions from {file_path}: {e}"
        ) from e
```

### Documentation Standards

All public functions need docstrings following Google style:

```python
def generate_answer_template(
    question: str,
    raw_answer: str,
    model: str = "gpt-4",
    temperature: float = 0.0
) -> str:
    """Generate Pydantic answer template from question and answer.

    Args:
        question: The question text to generate template for
        raw_answer: The expected answer text
        model: LLM model name to use for generation
        temperature: Sampling temperature for LLM

    Returns:
        Python code string containing the Pydantic answer class

    Raises:
        LLMError: When LLM fails to generate template
        ValidationError: When template validation fails

    Example:
        >>> template = generate_answer_template(
        ...     question="What is the capital of France?",
        ...     raw_answer="Paris"
        ... )
        >>> print(template)
        class Answer(BaseAnswer):
            capital: str = Field(description="Capital city name")
            ...
    """
    pass
```

## Testing

### Test Structure

Tests are organized by type and module:

```
tests/
├── unit/                  # Unit tests (fast, isolated)
│   ├── test_llm/
│   ├── test_questions/
│   ├── test_answers/
│   └── test_benchmark/
├── integration/           # Integration tests (slower, real APIs)
│   ├── test_workflows/
│   └── test_providers/
├── fixtures/              # Test data and fixtures
│   ├── sample_data/
│   ├── mock_responses/
│   └── test_configs/
└── conftest.py           # Shared test configuration
```

### Writing Tests

#### Unit Tests

```python
# tests/unit/test_questions/test_extractor.py
import pytest
from unittest.mock import Mock, patch
import pandas as pd

from karenina.questions.extractor import extract_questions_from_excel
from karenina.schemas.question_class import Question

class TestQuestionExtractor:
    """Test question extraction functionality."""

    def test_extract_questions_success(self):
        """Test successful question extraction."""
        # Arrange
        mock_df = pd.DataFrame({
            'Question': ['What is 2+2?', 'What is the capital of France?'],
            'Answer': ['4', 'Paris']
        })

        with patch('karenina.questions.extractor.pd.read_excel', return_value=mock_df):
            # Act
            questions = extract_questions_from_excel('fake_file.xlsx')

            # Assert
            assert len(questions) == 2
            assert questions[0].question == 'What is 2+2?'
            assert questions[0].raw_answer == '4'
            assert isinstance(questions[0], Question)

    def test_extract_questions_missing_columns(self):
        """Test extraction with missing required columns."""
        mock_df = pd.DataFrame({'Wrong': ['data']})

        with patch('karenina.questions.extractor.pd.read_excel', return_value=mock_df):
            with pytest.raises(ValueError, match="Required column 'Question' not found"):
                extract_questions_from_excel('fake_file.xlsx')

    @pytest.mark.parametrize("file_extension,expected_reader", [
        ('.xlsx', 'pd.read_excel'),
        ('.csv', 'pd.read_csv'),
        ('.tsv', 'pd.read_csv')
    ])
    def test_file_format_handling(self, file_extension, expected_reader):
        """Test different file format handling."""
        # Test implementation here
        pass
```

#### Integration Tests

```python
# tests/integration/test_workflows/test_complete_pipeline.py
import pytest
import tempfile
import os
from pathlib import Path

from karenina.questions.extractor import extract_and_generate_questions
from karenina.answers.generator import generate_answer_templates_from_questions_file

@pytest.mark.integration
@pytest.mark.skipif(not os.getenv('OPENAI_API_KEY'), reason="API key required")
class TestCompletePipeline:
    """Test complete workflow integration."""

    def test_end_to_end_workflow(self, sample_excel_file):
        """Test complete pipeline from file to templates."""
        with tempfile.TemporaryDirectory() as tmpdir:
            questions_file = Path(tmpdir) / "questions.py"

            # Step 1: Extract questions
            questions = extract_and_generate_questions(
                file_path=sample_excel_file,
                output_path=str(questions_file),
                question_column="Question",
                answer_column="Answer"
            )

            assert len(questions) > 0
            assert questions_file.exists()

            # Step 2: Generate templates
            templates = generate_answer_templates_from_questions_file(
                questions_py_path=str(questions_file),
                model="gpt-3.5-turbo",  # Use cheaper model for tests
                model_provider="openai"
            )

            assert len(templates) == len(questions)

            # Validate template structure
            for template in templates.values():
                assert "class Answer(BaseAnswer)" in template
                assert "def verify(self)" in template
```

### Test Fixtures

```python
# tests/conftest.py
import pytest
import pandas as pd
import tempfile
from pathlib import Path

@pytest.fixture
def sample_excel_file():
    """Create a temporary Excel file for testing."""
    df = pd.DataFrame({
        'Question': [
            'What is the capital of France?',
            'What is 2 + 2?',
            'Who wrote Romeo and Juliet?'
        ],
        'Answer': [
            'Paris',
            '4',
            'William Shakespeare'
        ]
    })

    with tempfile.NamedTemporaryFile(suffix='.xlsx', delete=False) as f:
        df.to_excel(f.name, index=False)
        yield f.name

    # Cleanup
    Path(f.name).unlink()

@pytest.fixture
def mock_llm_response():
    """Mock LLM response for testing."""
    return """
    class Answer(BaseAnswer):
        capital: str = Field(description="Capital city name")

        def model_post_init(self, __context):
            self.correct = {"capital": "Paris"}

        def verify(self) -> bool:
            return self.capital.lower() == self.correct["capital"].lower()
    """

@pytest.fixture
def verification_config():
    """Standard verification config for testing."""
    from karenina.benchmark.models import VerificationConfig, ModelConfiguration

    return VerificationConfig(
        answering_models=[
            ModelConfiguration(
                id="test-model",
                model_provider="openai",
                model_name="gpt-3.5-turbo",
                temperature=0.0,
                interface="langchain",
                system_prompt="Test prompt"
            )
        ],
        parsing_models=[
            ModelConfiguration(
                id="test-parser",
                model_provider="openai",
                model_name="gpt-3.5-turbo",
                temperature=0.0,
                interface="langchain",
                system_prompt="Parse responses"
            )
        ]
    )
```

### Running Tests

```bash
# Run all tests
make test

# Run with coverage
make test-cov

# Run only fast tests
pytest tests/unit/ -v

# Run integration tests (requires API keys)
pytest tests/integration/ -v -m integration

# Run specific test file
pytest tests/unit/test_questions/test_extractor.py -v

# Run with specific markers
pytest -m "not integration" -v

# Run tests matching pattern
pytest -k "test_extract" -v
```

### Test Configuration

pytest configuration in `pyproject.toml`:

```toml
[tool.pytest.ini_options]
testpaths = ["tests"]
python_files = ["test_*.py", "*_test.py"]
python_classes = ["Test*"]
python_functions = ["test_*"]
addopts = "-v --tb=short --strict-markers"
markers = [
    "integration: marks tests as integration tests (may be slow)",
    "unit: marks tests as unit tests (should be fast)",
    "llm: marks tests that require LLM API access",
    "slow: marks tests as slow (deselect with '-m \"not slow\"')"
]
filterwarnings = [
    "ignore::DeprecationWarning",
    "ignore::PendingDeprecationWarning"
]
```

## Documentation

### API Documentation

We use mkdocs with mkdocstrings for API documentation:

```bash
# Build documentation
make docs

# Serve documentation locally
make docs-serve

# Clean documentation
rm -rf site/
```

### Writing Documentation

#### Module Documentation

Each module should have a comprehensive docstring:

```python
# karenina/questions/__init__.py
"""Question extraction and processing module.

This module provides functionality for extracting questions from various file
formats (Excel, CSV, TSV) and processing them into structured Question objects
suitable for benchmarking.

Key Features:
- Multi-format file support (Excel, CSV, TSV)
- Automatic MD5 hash ID generation
- Custom column mapping
- Batch processing capabilities

Example:
    >>> from karenina.questions import extract_and_generate_questions
    >>> questions = extract_and_generate_questions(
    ...     file_path="data/questions.xlsx",
    ...     output_path="questions.py",
    ...     question_column="Question",
    ...     answer_column="Answer"
    ... )
"""
```

#### Adding Examples

Include practical examples in docstrings:

```python
def generate_answer_template(question: str, raw_answer: str) -> str:
    """Generate answer template with comprehensive example.

    Examples:
        Basic usage:

        >>> template = generate_answer_template(
        ...     question="What is the capital of France?",
        ...     raw_answer="Paris"
        ... )
        >>> print(template)
        class Answer(BaseAnswer):
            capital: str = Field(description="Capital city name")
            ...

        With custom model:

        >>> template = generate_answer_template(
        ...     question="Calculate 15% of 200",
        ...     raw_answer="30",
        ...     model="gpt-4",
        ...     temperature=0.0
        ... )
    """
```

## Debugging

### Logging Setup

Configure logging for development:

```python
# karenina/utils/logging.py
import logging
import os
from typing import Optional

def setup_logging(level: Optional[str] = None) -> logging.Logger:
    """Setup logging configuration for Karenina."""

    level = level or os.getenv("KARENINA_LOG_LEVEL", "INFO")

    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

    logger = logging.getLogger("karenina")

    # Add file handler for debug mode
    if os.getenv("KARENINA_DEBUG"):
        file_handler = logging.FileHandler("karenina_debug.log")
        file_handler.setLevel(logging.DEBUG)
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s'
        )
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger

# Usage in modules
logger = setup_logging()

def some_function():
    logger.debug("Starting function execution")
    logger.info("Processing data")
    logger.warning("Potential issue detected")
    logger.error("Error occurred", exc_info=True)
```

### Debugging LLM Interactions

```python
# Enable detailed LLM debugging
import os
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_ENDPOINT"] = "https://api.smith.langchain.com"
os.environ["LANGCHAIN_API_KEY"] = "your-api-key"  # Optional

# Or use simple logging
def debug_llm_call(llm, message, **kwargs):
    """Debug wrapper for LLM calls."""
    logger = logging.getLogger("karenina.llm")

    logger.debug(f"LLM Call - Model: {llm.__class__.__name__}")
    logger.debug(f"LLM Call - Message: {message[:200]}...")
    logger.debug(f"LLM Call - Kwargs: {kwargs}")

    start_time = time.time()
    try:
        response = llm.invoke(message, **kwargs)
        elapsed = time.time() - start_time

        logger.debug(f"LLM Response - Time: {elapsed:.2f}s")
        logger.debug(f"LLM Response - Content: {response.content[:200]}...")

        return response
    except Exception as e:
        elapsed = time.time() - start_time
        logger.error(f"LLM Error - Time: {elapsed:.2f}s - Error: {e}")
        raise
```

### Common Debugging Scenarios

#### Template Generation Issues

```python
def debug_template_generation(question: str, raw_answer: str):
    """Debug template generation step by step."""

    print(f"Question: {question}")
    print(f"Raw Answer: {raw_answer}")

    # Check prompt construction
    from karenina.prompts.answer_generation import ANSWER_GENERATION_SYS, ANSWER_GENERATION_USER

    system_prompt = ANSWER_GENERATION_SYS
    user_prompt = ANSWER_GENERATION_USER.format(
        question=question,
        raw_answer=raw_answer
    )

    print(f"System Prompt Length: {len(system_prompt)}")
    print(f"User Prompt: {user_prompt}")

    # Try generation with debug logging
    try:
        template = generate_answer_template(
            question=question,
            raw_answer=raw_answer,
            model="gpt-3.5-turbo",
            temperature=0.0
        )
        print(f"Generated Template:\n{template}")

        # Validate generated template
        from karenina.benchmark.verification.validation import validate_answer_template
        is_valid, error = validate_answer_template(template)
        print(f"Template Valid: {is_valid}")
        if not is_valid:
            print(f"Validation Error: {error}")

    except Exception as e:
        print(f"Template Generation Failed: {e}")
        import traceback
        traceback.print_exc()
```

#### Verification Debugging

```python
def debug_verification(question_id: str, question_text: str, template_code: str):
    """Debug verification process step by step."""

    print(f"Debugging verification for question: {question_id}")

    # Test template compilation
    try:
        exec(template_code)
        print("✓ Template compiles successfully")
    except SyntaxError as e:
        print(f"✗ Template syntax error: {e}")
        return

    # Test LLM response
    from karenina.llm.interface import init_chat_model_unified

    llm = init_chat_model_unified(
        model="gpt-3.5-turbo",
        provider="openai",
        temperature=0.0
    )

    response = llm.invoke([{"role": "user", "content": question_text}])
    print(f"Raw LLM Response: {response.content}")

    # Test parsing
    try:
        # Would test parsing logic here
        print("✓ Response parsing successful")
    except Exception as e:
        print(f"✗ Response parsing failed: {e}")
```

## Performance Optimization

### Profiling Code

```python
# Profile function performance
import cProfile
import pstats
from functools import wraps

def profile_function(func):
    """Decorator to profile function performance."""
    @wraps(func)
    def wrapper(*args, **kwargs):
        profiler = cProfile.Profile()
        profiler.enable()

        result = func(*args, **kwargs)

        profiler.disable()
        stats = pstats.Stats(profiler)
        stats.sort_stats('cumulative')
        stats.print_stats(10)  # Top 10 functions

        return result
    return wrapper

@profile_function
def slow_function():
    """Function to profile."""
    pass
```

### Memory Usage Monitoring

```python
import psutil
import os
from functools import wraps

def monitor_memory(func):
    """Monitor memory usage of a function."""
    @wraps(func)
    def wrapper(*args, **kwargs):
        process = psutil.Process(os.getpid())
        memory_before = process.memory_info().rss / 1024 / 1024  # MB

        result = func(*args, **kwargs)

        memory_after = process.memory_info().rss / 1024 / 1024  # MB
        memory_diff = memory_after - memory_before

        print(f"Memory usage - Before: {memory_before:.1f}MB, "
              f"After: {memory_after:.1f}MB, "
              f"Diff: {memory_diff:+.1f}MB")

        return result
    return wrapper
```

### Optimization Guidelines

1. **Batch API Calls**: Group multiple LLM requests when possible
2. **Cache Results**: Cache expensive computations and API responses
3. **Lazy Loading**: Load resources only when needed
4. **Async Operations**: Use async/await for I/O bound operations
5. **Connection Pooling**: Reuse HTTP connections for API calls

```python
# Example: Batch processing optimization
def batch_generate_templates(questions: List[Question], batch_size: int = 5):
    """Generate templates in batches for better performance."""

    templates = {}

    for i in range(0, len(questions), batch_size):
        batch = questions[i:i + batch_size]

        # Process batch
        batch_templates = process_question_batch(batch)
        templates.update(batch_templates)

        # Small delay to avoid rate limits
        time.sleep(0.1)

    return templates
```

## Contributing Guidelines

### Code Review Checklist

Before submitting a PR, ensure:

- [ ] Code follows style guidelines (Ruff passes)
- [ ] Type hints are complete (mypy passes)
- [ ] Tests are added/updated and pass
- [ ] Documentation is updated
- [ ] Commit messages follow conventional format
- [ ] No secrets or API keys in code
- [ ] Performance impact considered
- [ ] Backwards compatibility maintained

### PR Template

```markdown
## Description
Brief description of changes and motivation.

## Type of Change
- [ ] Bug fix (non-breaking change fixing an issue)
- [ ] New feature (non-breaking change adding functionality)
- [ ] Breaking change (fix or feature causing existing functionality to change)
- [ ] Documentation update

## Testing
- [ ] Unit tests added/updated
- [ ] Integration tests added/updated
- [ ] Manual testing performed
- [ ] All tests pass

## Documentation
- [ ] Code comments updated
- [ ] API documentation updated
- [ ] User guides updated (if applicable)

## Checklist
- [ ] Code follows style guidelines
- [ ] Self-review performed
- [ ] Tests added and pass
- [ ] Documentation updated
- [ ] No breaking changes (or marked as such)
```

## Release Process

### Semantic Versioning

We follow [Semantic Versioning](https://semver.org/):

- MAJOR: Breaking changes
- MINOR: New features (backwards compatible)
- PATCH: Bug fixes (backwards compatible)

### Release Steps

1. **Prepare Release**:
   ```bash
   # Update version and changelog
   git checkout develop
   git pull origin develop

   # Run semantic-release dry run
   make release-dry
   ```

2. **Create Release**:
   ```bash
   # Create and merge release PR to main
   git checkout main
   git merge develop

   # Create release
   make release
   ```

3. **Post-Release**:
   ```bash
   # Merge back to develop
   git checkout develop
   git merge main
   git push origin develop
   ```

### Changelog Generation

We use conventional commits to auto-generate changelogs:

```markdown
# Changelog

## [1.2.0] - 2024-01-15

### Added
- New LLM provider support for Anthropic Claude
- Rubric-based evaluation system
- Batch processing capabilities

### Changed
- Improved error handling for file operations
- Updated dependencies to latest versions

### Fixed
- Fixed template generation for complex nested structures
- Resolved memory leak in session management

### Deprecated
- Legacy single-model configuration (use multi-model instead)

### Removed
- Removed deprecated chat endpoints

### Security
- Updated dependencies to address security vulnerabilities
```

This development guide should provide everything needed to contribute effectively to Karenina. For questions not covered here, please open an issue on GitHub.
