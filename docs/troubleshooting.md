# Troubleshooting

This guide provides solutions to common issues when using Karenina.

## Quick Solutions

| Issue | Quick Fix |
|-------|-----------|
| API key not found | Set `OPENAI_API_KEY` environment variable |
| Template generation fails | Check API key and model name spelling |
| Verification always fails | Review template Pydantic class syntax |
| Database locked | Close other connections or use different database |
| Import errors | Install with `pip install karenina` or `uv pip install karenina` |
| Slow verification | Reduce `replicate_count` or enable async |

## API Key Issues

### Issue: Missing API Key

**Error Message**:
```
ValueError: OPENAI_API_KEY not set
KeyError: 'OPENAI_API_KEY'
```

**Cause**: Environment variable not configured for LLM provider.

**Solutions**:

**Option 1: Set environment variable**
```bash
export OPENAI_API_KEY="sk-..."
```

**Option 2: Use .env file**
```bash
# Create .env file
echo 'OPENAI_API_KEY="sk-..."' > .env
```

**Option 3: Set in Python**
```python
import os
os.environ["OPENAI_API_KEY"] = "sk-..."

# Then import karenina
from karenina import Benchmark
```

**Verification**:
```bash
# Check if key is set
echo $OPENAI_API_KEY

# Or in Python
import os
print("Key set:", bool(os.getenv("OPENAI_API_KEY")))
```

### Issue: Invalid API Key

**Error Message**:
```
AuthenticationError: Incorrect API key provided
401 Unauthorized
```

**Cause**: API key is invalid, expired, or has wrong format.

**Solutions**:

1. **Verify key format**:
   ```python
   import os
   key = os.getenv("OPENAI_API_KEY")

   # OpenAI keys start with "sk-"
   assert key.startswith("sk-"), "Invalid OpenAI key format"

   # Anthropic keys start with "sk-ant-"
   # Google keys start with "AIza"
   # OpenRouter keys start with "sk-or-"
   ```

2. **Generate new key**:
   - OpenAI: https://platform.openai.com/api-keys
   - Anthropic: https://console.anthropic.com/settings/keys
   - Google: https://aistudio.google.com/app/apikey

3. **Check key permissions**:
   - Ensure key has access to the model you're using
   - Verify account has credits/quota remaining

### Issue: Rate Limit Exceeded

**Error Message**:
```
RateLimitError: Rate limit exceeded
429 Too Many Requests
```

**Cause**: Too many API calls in short period.

**Solutions**:

**Option 1: Add delay between calls**
```python
import os

# Slow down async processing
os.environ["KARENINA_ASYNC_DELAY_BETWEEN_BATCHES"] = "2.0"  # 2 seconds
```

**Option 2: Reduce parallelism**
```python
import os

# Reduce concurrent workers
os.environ["KARENINA_ASYNC_MAX_WORKERS"] = "1"
```

**Option 3: Upgrade API tier**
- Increase rate limits with higher-tier API plan
- Contact provider for quota increase

## Model Configuration Issues

### Issue: Model Not Found

**Error Message**:
```
InvalidRequestError: The model `gpt-4.1-min` does not exist
NotFoundError: Model not found
```

**Cause**: Typo in model name or model not available.

**Solutions**:

**Check model name spelling**:
```python
from karenina.schemas import ModelConfig

# ❌ Wrong - typo
model_config = ModelConfig(
    model_name="gpt-4.1-min",  # Missing 'i'
    model_provider="openai"
)

# ✅ Correct
model_config = ModelConfig(
    model_name="gpt-4.1-mini",  # Correct spelling
    model_provider="openai",
    temperature=0.0
)
```

**Common model names**:
```python
# OpenAI
"gpt-4.1-mini"  # Default, recommended
"gpt-4o"
"gpt-4-turbo"

# Anthropic
"claude-3-5-sonnet-20241022"
"claude-3-opus-20240229"

# Google
"gemini-2.0-flash-exp"
"gemini-1.5-pro"
```

### Issue: Wrong Provider for Model

**Error Message**:
```
ValueError: Model gpt-4.1-mini not available for provider anthropic
```

**Cause**: Model name doesn't match provider.

**Solution**:
```python
from karenina.schemas import ModelConfig

# ❌ Wrong - GPT model with Anthropic provider
model_config = ModelConfig(
    model_name="gpt-4.1-mini",
    model_provider="anthropic"  # Wrong!
)

# ✅ Correct - Match model to provider
model_config = ModelConfig(
    model_name="gpt-4.1-mini",
    model_provider="openai",  # Correct
    temperature=0.0
)
```

## Template Generation Issues

### Issue: Template Generation Fails

**Error Message**:
```
TemplateGenerationError: Failed to generate template
ValidationError: Pydantic validation failed
```

**Cause**: LLM generated invalid Python code or Pydantic class.

**Solutions**:

**Option 1: Check question format**
```python
from karenina import Benchmark

benchmark = Benchmark.create(
    name="Genomics Benchmark",
    description="Testing genomics knowledge",
    version="1.0.0"
)

# ❌ Wrong - unclear question
benchmark.add_question(
    question="Tell me about BCL2",
    raw_answer="It's a protein"
)

# ✅ Better - specific question
benchmark.add_question(
    question="What is the approved drug target of Venetoclax?",
    raw_answer="BCL2"
)
```

**Option 2: Retry with better model**
```python
from karenina.schemas import ModelConfig

# Try more capable model
model_config = ModelConfig(
    model_name="gpt-4o",  # More capable than gpt-4.1-mini
    model_provider="openai",
    temperature=0.0
)

benchmark.generate_all_templates(model_config=model_config)
```

**Option 3: Generate manually**
```python
# If automatic generation fails, write template manually
from karenina.schemas import BaseAnswer

class DrugTargetAnswer(BaseAnswer):
    \"\"\"Answer template for drug target questions.\"\"\"
    target_protein: str

# Register manually
benchmark.questions["q1"].template_code = '''
class DrugTargetAnswer(BaseAnswer):
    target_protein: str
'''
```

### Issue: Template Has Syntax Errors

**Error Message**:
```
SyntaxError: invalid syntax
IndentationError: unexpected indent
```

**Cause**: Generated template has Python syntax errors.

**Solution**:
```python
from karenina import Benchmark

benchmark = Benchmark.create(
    name="Genomics Benchmark",
    description="Testing genomics knowledge",
    version="1.0.0"
)

benchmark.add_question(
    question="What is the approved drug target of Venetoclax?",
    raw_answer="BCL2"
)

# Regenerate template
from karenina.schemas import ModelConfig

model_config = ModelConfig(
    model_name="gpt-4o",  # More reliable
    model_provider="openai",
    temperature=0.0
)

# Regenerate specific question
question_id = list(benchmark.questions.keys())[0]
benchmark.generate_template(
    question_id=question_id,
    model_config=model_config
)
```

## Verification Issues

### Issue: All Verifications Fail

**Error Message**:
```
VerificationResult(verify_result=False, ...)
All questions failed verification
```

**Cause**: Template too strict, answer format mismatch, or parsing issues.

**Solutions**:

**Option 1: Review template definition**
```python
# Check generated template
question = benchmark.questions["q1"]
print(question.template_code)

# Look for overly strict types or required fields
```

**Option 2: Enable embedding check fallback**
```python
import os

# Enable semantic similarity fallback
os.environ["EMBEDDING_CHECK"] = "true"
os.environ["EMBEDDING_CHECK_THRESHOLD"] = "0.80"  # Lower = more permissive

# Retry verification
results = benchmark.run_verification(config)

# Check if embedding check helped
rescued = sum(1 for r in results.values() if r.embedding_override_applied)
print(f"Embedding check rescued {rescued} failures")
```

**Option 3: Use deep-judgment parsing**
```python
from karenina.schemas import VerificationConfig, ModelConfig

model_config = ModelConfig(
    model_name="gpt-4.1-mini",
    model_provider="openai",
    temperature=0.0
)

config = VerificationConfig(
    answering_models=[model_config],
    parsing_models=[model_config],
    replicate_count=3,
    deep_judgment_enabled=True  # Multi-stage parsing
)

results = benchmark.run_verification(config)
```

### Issue: Verification Inconsistent

**Symptom**: Same question passes sometimes, fails other times.

**Cause**: Temperature > 0 or insufficient replicates.

**Solution**:
```python
from karenina.schemas import VerificationConfig, ModelConfig

# ❌ Wrong - non-deterministic
model_config = ModelConfig(
    model_name="gpt-4.1-mini",
    model_provider="openai",
    temperature=0.7  # Too high for benchmarking
)

config = VerificationConfig(
    answering_models=[model_config],
    parsing_models=[model_config],
    replicate_count=1  # Not enough samples
)

# ✅ Correct - deterministic and robust
model_config = ModelConfig(
    model_name="gpt-4.1-mini",
    model_provider="openai",
    temperature=0.0  # Deterministic
)

config = VerificationConfig(
    answering_models=[model_config],
    parsing_models=[model_config],
    replicate_count=5  # More replicates = more robust
)

results = benchmark.run_verification(config)
```

### Issue: Rubric Evaluation Fails

**Error Message**:
```
RubricError: No rubric defined for question
ValidationError: Invalid rubric configuration
```

**Cause**: Rubric not created or improperly configured.

**Solution**:
```python
from karenina import Benchmark
from karenina.schemas import Rubric, RubricTrait, ModelConfig, VerificationConfig

benchmark = Benchmark.create(
    name="Genomics Benchmark",
    description="Testing genomics knowledge",
    version="1.0.0"
)

benchmark.add_question(
    question="What is the approved drug target of Venetoclax?",
    raw_answer="BCL2"
)

# Create rubric
rubric = Rubric(
    traits=[
        RubricTrait(
            name="accuracy",
            description="Is the protein name correct?",
            kind="binary"
        ),
        RubricTrait(
            name="completeness",
            description="Is the answer complete?",
            kind="score",
            scale=5
        )
    ]
)

# Assign to question
question_id = list(benchmark.questions.keys())[0]
benchmark.questions[question_id].rubric = rubric

# Generate templates first
model_config = ModelConfig(
    model_name="gpt-4.1-mini",
    model_provider="openai",
    temperature=0.0
)
benchmark.generate_all_templates(model_config=model_config)

# Run verification with rubric enabled
config = VerificationConfig(
    answering_models=[model_config],
    parsing_models=[model_config],
    replicate_count=3,
    rubric_enabled=True  # Must enable
)

results = benchmark.run_verification(config)

# Check rubric results
for result in results.values():
    print(f"Rubric scores: {result.verify_rubric}")
```

## Database Issues

### Issue: Database Locked

**Error Message**:
```
sqlite3.OperationalError: database is locked
DatabaseError: Cannot acquire lock
```

**Cause**: Multiple processes accessing same database file.

**Solutions**:

**Option 1: Close other connections**
```python
from pathlib import Path

# Ensure no other processes using database
db_path = Path("dbs/genomics.db")

# Close any open connections
# (SQLite only allows one writer at a time)
```

**Option 2: Use separate database files**
```python
import os
from pathlib import Path

# Use different database per workflow
os.environ["DB_PATH"] = "dbs/workflow_1.db"

# Or programmatically
db_path = Path("dbs/genomics_workflow_1.db")
benchmark.save_to_db(db_path)
```

**Option 3: Add timeout**
```python
import sqlite3

# Increase timeout (default is 5.0 seconds)
# This is handled internally by Karenina
# If issue persists, ensure no long-running transactions
```

### Issue: Database Not Found

**Error Message**:
```
FileNotFoundError: Database file not found
IOError: No such file or directory
```

**Cause**: Database directory doesn't exist.

**Solution**:
```python
from pathlib import Path

# Create database directory
db_path = Path("dbs/genomics.db")
db_path.parent.mkdir(parents=True, exist_ok=True)

# Save benchmark
benchmark.save_to_db(db_path)
```

### Issue: Loading Benchmark Fails

**Error Message**:
```
ValueError: Benchmark 'Genomics Benchmark' not found
DatabaseError: No such table
```

**Cause**: Benchmark doesn't exist or database is corrupt.

**Solutions**:

**Option 1: List available benchmarks**
```python
from pathlib import Path
from karenina.storage import list_benchmarks

db_path = Path("dbs/genomics.db")
benchmarks = list_benchmarks(db_path)
print("Available benchmarks:", benchmarks)

# Load with exact name
benchmark = Benchmark.load_from_db(db_path, benchmarks[0])
```

**Option 2: Check database integrity**
```bash
# Verify database is not corrupt
sqlite3 dbs/genomics.db "PRAGMA integrity_check;"
```

**Option 3: Recreate database**
```python
from pathlib import Path

# If database is corrupt, delete and recreate
db_path = Path("dbs/genomics.db")
db_path.unlink(missing_ok=True)

# Save benchmark fresh
benchmark.save_to_db(db_path)
```

## Import and Checkpoint Issues

### Issue: Checkpoint Load Fails

**Error Message**:
```
ValidationError: Invalid checkpoint format
JSONDecodeError: Expecting value
```

**Cause**: Corrupted or incompatible checkpoint file.

**Solutions**:

**Option 1: Validate JSON format**
```bash
# Check JSON is valid
python -m json.tool checkpoints/genomics.json
```

**Option 2: Check file permissions**
```bash
# Ensure file is readable
ls -l checkpoints/genomics.json
chmod 644 checkpoints/genomics.json
```

**Option 3: Regenerate checkpoint**
```python
from pathlib import Path

# Save new checkpoint
checkpoint_path = Path("checkpoints/genomics_new.json")
benchmark.save_checkpoint(checkpoint_path)

# Load fresh checkpoint
from karenina.utils.checkpoint import load_checkpoint_as_benchmark
benchmark = load_checkpoint_as_benchmark(checkpoint_path)
```

### Issue: Excel Import Fails

**Error Message**:
```
ImportError: No module named 'openpyxl'
ValueError: Excel file format not supported
```

**Cause**: Missing dependencies or incorrect file format.

**Solutions**:

**Option 1: Install Excel dependencies**
```bash
pip install openpyxl pandas
```

**Option 2: Use CSV instead**
```python
import pandas as pd

# Convert Excel to CSV first
df = pd.read_excel("questions.xlsx")
df.to_csv("questions.csv", index=False)

# Import CSV
from karenina.domain.questions.extractor import extract_questions_from_file

questions = extract_questions_from_file("questions.csv")
```

**Option 3: Verify file format**
```python
import pandas as pd

# Check file has required columns
df = pd.read_excel("questions.xlsx")
required_columns = ["question", "answer"]

for col in required_columns:
    assert col in df.columns, f"Missing column: {col}"
```

## Performance Issues

### Issue: Verification Too Slow

**Symptom**: Verification takes hours for small benchmarks.

**Causes and Solutions**:

**Cause 1: Too many replicates**
```python
# ❌ Slow - too many replicates
config = VerificationConfig(
    answering_models=[model_config],
    parsing_models=[model_config],
    replicate_count=20  # Excessive
)

# ✅ Fast - reasonable replicates
config = VerificationConfig(
    answering_models=[model_config],
    parsing_models=[model_config],
    replicate_count=3  # Sufficient
)
```

**Cause 2: Sequential processing**
```python
import os

# ✅ Enable parallel processing
os.environ["KARENINA_ASYNC_ENABLED"] = "true"
os.environ["KARENINA_ASYNC_MAX_WORKERS"] = "4"  # Adjust based on CPU
```

**Cause 3: Deep-judgment enabled unnecessarily**
```python
# ❌ Slow - deep-judgment adds 3-5x overhead
config = VerificationConfig(
    answering_models=[model_config],
    parsing_models=[model_config],
    deep_judgment_enabled=True  # Only use when needed
)

# ✅ Fast - standard parsing
config = VerificationConfig(
    answering_models=[model_config],
    parsing_models=[model_config],
    deep_judgment_enabled=False  # Default
)
```

**Cause 4: Expensive model**
```python
# ❌ Slow - expensive model
model_config = ModelConfig(
    model_name="gpt-4o",  # Slower, more expensive
    model_provider="openai"
)

# ✅ Fast - cheaper model
model_config = ModelConfig(
    model_name="gpt-4.1-mini",  # Faster, cheaper
    model_provider="openai",
    temperature=0.0
)
```

### Issue: High API Costs

**Symptom**: Unexpectedly high billing from LLM provider.

**Solutions**:

**Monitor token usage**:
```python
# After verification
for result in results.values():
    print(f"Tokens used: {result.tokens_used}")
    print(f"Cost estimate: {result.estimated_cost}")
```

**Reduce costs**:
```python
from karenina.schemas import ModelConfig, VerificationConfig

# 1. Use cheaper model
model_config = ModelConfig(
    model_name="gpt-4.1-mini",  # 60x cheaper than GPT-4o
    model_provider="openai",
    temperature=0.0
)

# 2. Reduce replicates
config = VerificationConfig(
    answering_models=[model_config],
    parsing_models=[model_config],
    replicate_count=1,  # Minimum for testing
    rubric_enabled=False  # Disable for cost savings
)

# 3. Test on subset first
test_questions = list(benchmark.questions.items())[:5]
# Run on small sample before full benchmark
```

## Export Issues

### Issue: Export Missing Data

**Symptom**: CSV or JSON export doesn't include all fields.

**Cause**: Some fields are optional and may be empty.

**Solution**:
```python
from pathlib import Path

# Export results
export_path = Path("exports/genomics_results.csv")
benchmark.export_results(
    results=results,
    output_path=export_path,
    format="csv"
)

# Check what's exported
import pandas as pd
df = pd.read_csv(export_path)
print("Exported columns:", df.columns.tolist())
print("Sample data:", df.head())
```

### Issue: Export File Permission Denied

**Error Message**:
```
PermissionError: Permission denied
OSError: Cannot write to file
```

**Cause**: Insufficient file permissions or directory doesn't exist.

**Solution**:
```python
from pathlib import Path

# Create export directory
export_path = Path("exports/genomics_results.csv")
export_path.parent.mkdir(parents=True, exist_ok=True)

# Ensure write permissions
import os
os.chmod(export_path.parent, 0o755)

# Export
benchmark.export_results(results, export_path, format="csv")
```

## Installation Issues

### Issue: Import Error

**Error Message**:
```
ModuleNotFoundError: No module named 'karenina'
ImportError: cannot import name 'Benchmark'
```

**Cause**: Package not installed or wrong Python environment.

**Solutions**:

**Option 1: Install package**
```bash
pip install karenina

# Or with uv
uv pip install karenina
```

**Option 2: Check Python environment**
```bash
# Verify correct environment
which python
pip list | grep karenina

# Activate virtual environment if needed
source venv/bin/activate
```

**Option 3: Install in development mode**
```bash
# If working from source
cd karenina
pip install -e .
```

### Issue: Version Mismatch

**Error Message**:
```
AttributeError: 'Benchmark' object has no attribute 'new_method'
ImportError: cannot import name 'NewClass'
```

**Cause**: Using old version of Karenina.

**Solution**:
```bash
# Update to latest version
pip install --upgrade karenina

# Or with uv
uv pip install --upgrade karenina

# Verify version
python -c "import karenina; print(karenina.__version__)"
```

## Getting Help

If you encounter an issue not covered here:

1. **Check Configuration**: Review [configuration.md](configuration.md)
2. **Review Examples**: See [quickstart.md](quickstart.md) for working examples
3. **Check API Reference**: See [api-reference.md](api-reference.md) for method signatures
4. **Enable Debug Logging**:
   ```python
   import logging
   logging.basicConfig(level=logging.DEBUG)
   ```
5. **Report Issue**: Open GitHub issue with:
   - Error message (full traceback)
   - Minimal reproducible example
   - Python version and OS
   - Karenina version

## Related Documentation

- **Configuration**: Environment variables and model setup
- **Quick Start**: Basic usage examples
- **API Reference**: Complete method documentation
- **Advanced Features**: Deep-judgment, embedding check, abstention detection

## Common Error Patterns

### API-Related Errors

| Error | Cause | Fix |
|-------|-------|-----|
| `401 Unauthorized` | Invalid API key | Check key format and validity |
| `429 Too Many Requests` | Rate limit | Add delays or reduce parallelism |
| `404 Model Not Found` | Wrong model name | Check spelling and provider |
| `500 Server Error` | Provider issue | Retry or check provider status |

### Data Errors

| Error | Cause | Fix |
|-------|-------|-----|
| `ValidationError` | Invalid data format | Check Pydantic model |
| `JSONDecodeError` | Corrupted checkpoint | Regenerate checkpoint |
| `FileNotFoundError` | Missing file/directory | Create directories first |
| `PermissionError` | File permissions | Check read/write permissions |

### Configuration Errors

| Error | Cause | Fix |
|-------|-------|-----|
| `KeyError` | Missing env variable | Set required variables |
| `TypeError` | Wrong type for parameter | Check API reference |
| `ValueError` | Invalid parameter value | Validate parameter ranges |
| `AttributeError` | Wrong object/method | Check version and imports |

## Summary

**Most Common Issues**:
1. Missing or invalid API keys → Set environment variables
2. Model name typos → Check spelling carefully
3. Template generation failures → Use better models or write manually
4. Database locks → Use separate databases for parallel workflows
5. Slow verification → Reduce replicates, use cheaper models, enable async

**Prevention Tips**:
- Always use `temperature=0.0` for deterministic results
- Start with `replicate_count=1` for testing
- Enable `EMBEDDING_CHECK` for better recall
- Use `gpt-4.1-mini` for cost-effective benchmarking
- Test on small subsets before full benchmarks
