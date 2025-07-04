# Answers Module

The `karenina.answers.generator` module creates structured Pydantic answer templates using LLM-powered code generation.

## Core Functions

### generate_answer_template

Generate a single answer template for a question using an LLM.

::: karenina.answers.generator.generate_answer_template

**Parameters:**
- `question` (str): Question text to generate template for
- `question_json` (str): JSON representation of Question object
- `model` (str): LLM model name (default: "gemini-2.0-flash")
- `model_provider` (str): Provider identifier (default: "google_genai")
- `temperature` (float): Sampling temperature (default: 0)
- `custom_system_prompt` (Optional[str]): Override default system prompt
- `interface` (str): LLM interface ("langchain" or "openrouter")

**Returns:**
- `str`: Generated Python code containing Pydantic Answer class

**Usage Examples:**

```python
# Basic template generation
question_text = "What is the capital of France?"
question_obj = Question(id="hash", question=question_text, raw_answer="Paris", tags=[])
template_code = generate_answer_template(
    question=question_text,
    question_json=question_obj.model_dump_json()
)

# Custom model and provider
template_code = generate_answer_template(
    question="Analyze this code performance",
    question_json=question_json,
    model="gpt-4",
    model_provider="openai",
    temperature=0.2
)

# Custom system prompt
custom_prompt = "Generate a detailed answer template with specific validation rules"
template_code = generate_answer_template(
    question=question_text,
    question_json=question_json,
    custom_system_prompt=custom_prompt
)
```

**Generated Template Example:**

```python
from pydantic import BaseModel, Field
from karenina.schemas.answer_class import BaseAnswer

class Answer(BaseAnswer):
    city_name: str = Field(description="The capital city name")
    country: str = Field(description="The country name")
    confidence: float = Field(ge=0.0, le=1.0, description="Confidence score")
    additional_info: Optional[str] = Field(default=None, description="Extra context")
```

### generate_answer_templates_from_questions_file

Batch generate templates from a questions Python file.

::: karenina.answers.generator.generate_answer_templates_from_questions_file

**Parameters:**
- `questions_py_path` (str): Path to generated questions.py file
- `model` (str): LLM model name (default: "gemini-2.0-flash")
- `model_provider` (str): Provider identifier (default: "google_genai")
- `interface` (str): LLM interface (default: "langchain")
- `return_blocks` (bool): Also return raw code blocks (default: False)

**Returns:**
- If `return_blocks=False`: `Dict[str, type]` - Mapping question IDs to Answer classes
- If `return_blocks=True`: `Tuple[Dict[str, type], Dict[str, str]]` - Classes and code blocks

**Usage Examples:**

```python
# Generate templates for all questions
templates = generate_answer_templates_from_questions_file("questions.py")

# Access specific template
question_id = "5f4dcc3b5aa765d61d8327deb882cf99"
AnswerClass = templates[question_id]

# Validate a response
response_data = {"city_name": "Paris", "country": "France", "confidence": 0.95}
validated_answer = AnswerClass(**response_data)

# Custom model configuration
templates = generate_answer_templates_from_questions_file(
    "complex_questions.py",
    model="claude-3-sonnet",
    model_provider="anthropic"
)

# Get both templates and code blocks
templates, code_blocks = generate_answer_templates_from_questions_file(
    "questions.py",
    return_blocks=True
)

# Save code blocks for later use
import json
with open("answer_templates.json", "w") as f:
    json.dump(code_blocks, f, indent=2)
```

**Processing Workflow:**

1. Import `all_questions` from the Python file
2. For each question:
   - Generate template using LLM
   - Extract Python code blocks from response
   - Execute code to create Answer class
   - Store in templates dictionary
3. Return mapping of question IDs to Answer classes

### load_answer_templates_from_json

Load pre-generated templates from JSON file containing code blocks.

::: karenina.answers.generator.load_answer_templates_from_json

**Parameters:**
- `json_file_path` (str): Path to JSON file with code blocks
- `return_blocks` (bool): Also return code blocks dictionary

**Returns:**
- If `return_blocks=False`: `Dict[str, type]` - Answer classes
- If `return_blocks=True`: `Tuple[Dict[str, type], Dict[str, str]]` - Classes and blocks

**JSON File Format:**
```json
{
  "question_hash_1": "from pydantic import BaseModel, Field\n\nclass Answer(BaseAnswer):\n    field1: str = Field(...)",
  "question_hash_2": "from pydantic import BaseModel, Field\n\nclass Answer(BaseAnswer):\n    field2: int = Field(...)"
}
```

**Usage Examples:**

```python
# Load pre-generated templates
templates = load_answer_templates_from_json("saved_templates.json")

# Use loaded templates
question_id = "abc123..."
AnswerClass = templates[question_id]
validated = AnswerClass(field1="value")

# Load with code blocks for inspection
templates, blocks = load_answer_templates_from_json("templates.json", return_blocks=True)

# Inspect generated code
print(blocks[question_id])
```

## Template Execution

The module uses safe code execution to create Answer classes:

### Execution Environment

- Clean namespace with controlled imports
- Access to Pydantic and BaseAnswer
- Isolated from global scope
- Exception handling for malformed code

### Security Considerations

```python
# Code execution is isolated
local_ns = {}
exec(code_blocks, globals(), local_ns)
Answer = local_ns["Answer"]
```

**Note:** Only execute code from trusted LLM sources. The system does not sandbox arbitrary code execution.

## Integration with Prompts

The module uses structured prompts from `karenina.prompts.answer_generation`:

### ANSWER_GENERATION_SYS

System prompt that instructs the LLM to:
- Generate Pydantic classes inheriting from BaseAnswer
- Include proper field validation
- Add descriptive Field() specifications
- Follow Python coding standards

### ANSWER_GENERATION_USER

User prompt template that provides:
- Question text and context
- JSON representation of Question object
- Formatting requirements
- Expected output structure

## Error Handling

Common errors and solutions:

```python
try:
    templates = generate_answer_templates_from_questions_file("questions.py")
except ImportError:
    print("Questions file not found or invalid format")
except SyntaxError as e:
    print(f"Generated code has syntax errors: {e}")
except Exception as e:
    print(f"Template generation failed: {e}")

# Validate individual templates
for q_id, AnswerClass in templates.items():
    try:
        # Test instantiation
        test_instance = AnswerClass()
    except Exception as e:
        print(f"Template {q_id} validation failed: {e}")
```

## Advanced Usage

### Custom Template Validation

```python
def validate_template_quality(templates: Dict[str, type]) -> Dict[str, bool]:
    """Validate generated templates meet quality standards."""
    results = {}

    for q_id, AnswerClass in templates.items():
        try:
            # Check if class has required fields
            fields = AnswerClass.model_fields
            has_description = all(f.description for f in fields.values())
            has_validation = any(f.constraints for f in fields.values())

            results[q_id] = has_description and len(fields) > 0
        except Exception:
            results[q_id] = False

    return results

# Use validation
templates = generate_answer_templates_from_questions_file("questions.py")
quality_check = validate_template_quality(templates)
print(f"Valid templates: {sum(quality_check.values())}/{len(quality_check)}")
```

### Batch Processing with Progress

```python
from tqdm import tqdm

def generate_with_progress(questions_file: str) -> Dict[str, type]:
    """Generate templates with progress tracking."""
    # The function already includes tqdm progress bars
    return generate_answer_templates_from_questions_file(questions_file)

# Monitor progress
templates = generate_with_progress("large_questions.py")
# Output: 100%|██████████| 500/500 [05:30<00:00, 1.51it/s]
```

### Template Caching

```python
import json
import os
from pathlib import Path

def cached_template_generation(questions_file: str, cache_dir: str = "cache") -> Dict[str, type]:
    """Generate templates with file-based caching."""

    cache_path = Path(cache_dir)
    cache_path.mkdir(exist_ok=True)

    # Generate cache key from questions file
    questions_hash = hash_question(Path(questions_file).read_text())
    cache_file = cache_path / f"templates_{questions_hash}.json"

    if cache_file.exists():
        print("Loading templates from cache")
        return load_answer_templates_from_json(str(cache_file))

    print("Generating new templates")
    templates, code_blocks = generate_answer_templates_from_questions_file(
        questions_file, return_blocks=True
    )

    # Save to cache
    with open(cache_file, "w") as f:
        json.dump(code_blocks, f, indent=2)

    return templates
```

## Integration Examples

### With Benchmark Runner

```python
# Generate templates
templates = generate_answer_templates_from_questions_file("questions.py")

# Use in benchmark
from karenina.benchmark.runner import run_benchmark
results = run_benchmark(questions_dict, responses_dict, templates)
```

### With Web API

```python
# API endpoint for template generation
@app.post("/generate_templates")
async def generate_templates_endpoint(questions_file: str):
    try:
        templates, code_blocks = generate_answer_templates_from_questions_file(
            questions_file, return_blocks=True
        )

        return {
            "success": True,
            "template_count": len(templates),
            "code_blocks": code_blocks
        }
    except Exception as e:
        return {"success": False, "error": str(e)}
```
