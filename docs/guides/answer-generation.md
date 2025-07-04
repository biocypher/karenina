# Answer Generation Guide

This guide covers the creation of structured Pydantic answer templates using LLM-powered code generation.

## Overview

Answer generation transforms questions into structured validation schemas, enabling consistent evaluation of LLM responses through Pydantic models.

## Core Workflow

### Step 1: Single Template Generation

```python
from karenina.answers.generator import generate_answer_template
from karenina.schemas.question_class import Question

# Create question object
question = Question(
    id="5f4dcc3b5aa765d61d8327deb882cf99",
    question="What is the capital of France?",
    raw_answer="Paris",
    tags=["geography"]
)

# Generate template
template_code = generate_answer_template(
    question=question.question,
    question_json=question.model_dump_json(),
    model="gemini-2.0-flash",
    model_provider="google_genai"
)

print("Generated template code:")
print(template_code)
```

**Generated Output Example:**
```python
from pydantic import BaseModel, Field
from karenina.schemas.answer_class import BaseAnswer

class Answer(BaseAnswer):
    capital_city: str = Field(description="Name of the capital city")
    country: str = Field(description="Country name")
    confidence: float = Field(ge=0.0, le=1.0, description="Confidence in the answer")
    additional_info: Optional[str] = Field(default=None, description="Any additional context")
```

### Step 2: Code Extraction and Execution

```python
from karenina.utils.code_parser import extract_and_combine_codeblocks

# Extract executable code from LLM response
executable_code = extract_and_combine_codeblocks(template_code)

# Execute to create Answer class
local_ns = {}
exec(executable_code, globals(), local_ns)
AnswerClass = local_ns["Answer"]

# Test the generated template
test_data = {
    "capital_city": "Paris",
    "country": "France",
    "confidence": 0.95
}

validated_answer = AnswerClass(**test_data)
print("Validated answer:", validated_answer)
```

### Step 3: Batch Template Generation

```python
from karenina.answers.generator import generate_answer_templates_from_questions_file

# Generate templates for all questions in a file
templates = generate_answer_templates_from_questions_file(
    questions_py_path="extracted_questions.py",
    model="gpt-4",
    model_provider="openai"
)

print(f"Generated {len(templates)} answer templates")

# Access specific template
question_id = "5f4dcc3b5aa765d61d8327deb882cf99"
AnswerTemplate = templates[question_id]

# Use template for validation
response_data = {"capital_city": "Paris", "country": "France", "confidence": 0.9}
validated = AnswerTemplate(**response_data)
```

## LLM Provider Configuration

### OpenAI Configuration

```python
# GPT-4 for complex reasoning
templates = generate_answer_templates_from_questions_file(
    "questions.py",
    model="gpt-4",
    model_provider="openai",
    interface="langchain"
)

# GPT-3.5 for faster generation
templates = generate_answer_templates_from_questions_file(
    "questions.py",
    model="gpt-3.5-turbo",
    model_provider="openai"
)
```

### Google AI Configuration

```python
# Gemini 2.0 Flash (default)
templates = generate_answer_templates_from_questions_file(
    "questions.py",
    model="gemini-2.0-flash",
    model_provider="google_genai"
)

# Gemini Pro for complex schemas
templates = generate_answer_templates_from_questions_file(
    "questions.py",
    model="gemini-pro",
    model_provider="google_genai"
)
```

### Anthropic Configuration

```python
# Claude 3 Sonnet
templates = generate_answer_templates_from_questions_file(
    "questions.py",
    model="claude-3-sonnet",
    model_provider="anthropic"
)
```

### OpenRouter Configuration

```python
# Access multiple models through OpenRouter
templates = generate_answer_templates_from_questions_file(
    "questions.py",
    model="anthropic/claude-3-sonnet",
    interface="openrouter"
)
```

## Advanced Template Customization

### Custom System Prompts

```python
custom_prompt = """
You are a expert in creating detailed Pydantic validation schemas.
Generate comprehensive answer templates with:
1. Specific field types and constraints
2. Detailed field descriptions
3. Appropriate validation rules
4. Optional fields for uncertainty

Focus on creating templates that can capture nuanced responses.
"""

template_code = generate_answer_template(
    question="Analyze the economic impact of climate change",
    question_json=question_json,
    custom_system_prompt=custom_prompt
)
```

### Temperature Control

```python
# Low temperature for consistent schemas
consistent_template = generate_answer_template(
    question=question_text,
    question_json=question_json,
    temperature=0.0  # Most deterministic
)

# Higher temperature for creative schemas
creative_template = generate_answer_template(
    question=question_text,
    question_json=question_json,
    temperature=0.7  # More varied output
)
```

## Template Quality Assurance

### Validation Testing

```python
def test_template_quality(templates: dict) -> dict:
    """Test generated templates for quality and functionality."""

    results = {}

    for question_id, AnswerClass in templates.items():
        test_result = {
            "has_fields": len(AnswerClass.model_fields) > 0,
            "has_descriptions": True,
            "instantiable": False,
            "validation_errors": []
        }

        # Check field descriptions
        for field_name, field_info in AnswerClass.model_fields.items():
            if not field_info.description:
                test_result["has_descriptions"] = False
                break

        # Test instantiation
        try:
            # Try with minimal data
            test_instance = AnswerClass()
            test_result["instantiable"] = True
        except Exception as e:
            test_result["validation_errors"].append(f"Instantiation failed: {e}")

        # Test with sample data
        try:
            sample_data = {}
            for field_name, field_info in AnswerClass.model_fields.items():
                if field_info.annotation == str:
                    sample_data[field_name] = "test_value"
                elif field_info.annotation == int:
                    sample_data[field_name] = 42
                elif field_info.annotation == float:
                    sample_data[field_name] = 0.5
                # Add more type handling as needed

            test_instance = AnswerClass(**sample_data)

        except Exception as e:
            test_result["validation_errors"].append(f"Sample data failed: {e}")

        results[question_id] = test_result

    return results

# Usage
templates = generate_answer_templates_from_questions_file("questions.py")
quality_results = test_template_quality(templates)

# Report results
passed = sum(1 for r in quality_results.values() if r["instantiable"] and r["has_descriptions"])
total = len(quality_results)
print(f"Template quality: {passed}/{total} passed all tests")
```

### Template Inspection

```python
def inspect_template(AnswerClass: type) -> dict:
    """Inspect a generated answer template."""

    inspection = {
        "class_name": AnswerClass.__name__,
        "fields": {},
        "inherits_from_base": issubclass(AnswerClass, BaseAnswer),
        "total_fields": len(AnswerClass.model_fields)
    }

    for field_name, field_info in AnswerClass.model_fields.items():
        inspection["fields"][field_name] = {
            "type": str(field_info.annotation),
            "description": field_info.description,
            "required": field_info.is_required(),
            "default": field_info.default if hasattr(field_info, 'default') else None
        }

    return inspection

# Usage
question_id = list(templates.keys())[0]
AnswerClass = templates[question_id]
inspection = inspect_template(AnswerClass)

print(f"Template inspection for {question_id}:")
for field_name, field_info in inspection["fields"].items():
    print(f"  {field_name}: {field_info['type']} - {field_info['description']}")
```

## Caching and Persistence

### Save Templates to JSON

```python
import json

def save_templates_to_json(templates: dict, code_blocks: dict, output_file: str):
    """Save generated templates as JSON for later use."""

    with open(output_file, 'w') as f:
        json.dump(code_blocks, f, indent=2)

    print(f"Saved {len(templates)} templates to {output_file}")

# Generate with code blocks
templates, code_blocks = generate_answer_templates_from_questions_file(
    "questions.py",
    return_blocks=True
)

# Save for later use
save_templates_to_json(templates, code_blocks, "answer_templates.json")
```

### Load Templates from JSON

```python
from karenina.answers.generator import load_answer_templates_from_json

# Load previously generated templates
templates = load_answer_templates_from_json("answer_templates.json")

print(f"Loaded {len(templates)} templates from cache")

# Use loaded templates immediately
for question_id, AnswerClass in templates.items():
    print(f"Template {question_id}: {len(AnswerClass.model_fields)} fields")
```

### Incremental Generation

```python
def incremental_template_generation(questions_file: str, cache_file: str):
    """Generate templates incrementally, using cache when available."""

    from karenina.questions.reader import read_questions_from_file
    import json
    from pathlib import Path

    # Load existing cache
    existing_templates = {}
    if Path(cache_file).exists():
        with open(cache_file) as f:
            existing_code_blocks = json.load(f)
        existing_templates = load_answer_templates_from_json(cache_file)
    else:
        existing_code_blocks = {}

    # Read all questions
    all_questions = read_questions_from_file(questions_file)

    # Find missing templates
    all_question_ids = {q.id for q in all_questions}
    cached_ids = set(existing_templates.keys())
    missing_ids = all_question_ids - cached_ids

    print(f"Found {len(cached_ids)} cached templates")
    print(f"Need to generate {len(missing_ids)} new templates")

    if missing_ids:
        # Generate only missing templates
        missing_questions = [q for q in all_questions if q.id in missing_ids]

        new_code_blocks = {}
        for question in missing_questions:
            print(f"Generating template for: {question.question[:50]}...")

            template_code = generate_answer_template(
                question.question,
                question.model_dump_json()
            )

            code_blocks = extract_and_combine_codeblocks(template_code)
            new_code_blocks[question.id] = code_blocks

        # Merge with existing cache
        existing_code_blocks.update(new_code_blocks)

        # Save updated cache
        with open(cache_file, 'w') as f:
            json.dump(existing_code_blocks, f, indent=2)

    # Load complete set
    return load_answer_templates_from_json(cache_file)

# Usage
templates = incremental_template_generation("questions.py", "template_cache.json")
```

## Error Handling and Recovery

### Robust Generation Pipeline

```python
def robust_template_generation(questions_file: str):
    """Generate templates with comprehensive error handling."""

    from karenina.questions.reader import read_questions_from_file

    all_questions = read_questions_from_file(questions_file)

    successful_templates = {}
    failed_questions = {}

    for question in all_questions:
        try:
            # Generate template
            template_code = generate_answer_template(
                question.question,
                question.model_dump_json()
            )

            # Extract code
            code_blocks = extract_and_combine_codeblocks(template_code)

            if not code_blocks.strip():
                failed_questions[question.id] = "No code blocks found"
                continue

            # Validate syntax
            import ast
            ast.parse(code_blocks)

            # Execute safely
            local_ns = {}
            exec(code_blocks, globals(), local_ns)

            if "Answer" not in local_ns:
                failed_questions[question.id] = "No Answer class found"
                continue

            AnswerClass = local_ns["Answer"]

            # Test instantiation
            test_instance = AnswerClass()

            successful_templates[question.id] = AnswerClass

        except SyntaxError as e:
            failed_questions[question.id] = f"Syntax error: {e}"
        except Exception as e:
            failed_questions[question.id] = f"Generation error: {e}"

    print(f"Successfully generated: {len(successful_templates)}")
    print(f"Failed: {len(failed_questions)}")

    if failed_questions:
        print("\nFailed questions:")
        for q_id, error in list(failed_questions.items())[:5]:
            print(f"  {q_id}: {error}")

    return successful_templates, failed_questions

# Usage
templates, failures = robust_template_generation("questions.py")
```

### Retry Logic

```python
import time
import random

def generate_with_retry(question: str, question_json: str, max_retries: int = 3):
    """Generate template with retry logic for transient failures."""

    for attempt in range(max_retries):
        try:
            template_code = generate_answer_template(question, question_json)

            # Validate the result
            code_blocks = extract_and_combine_codeblocks(template_code)
            if code_blocks.strip():
                return template_code
            else:
                raise ValueError("No code blocks generated")

        except Exception as e:
            if attempt < max_retries - 1:
                wait_time = (2 ** attempt) + random.uniform(0, 1)
                print(f"Attempt {attempt + 1} failed: {e}. Retrying in {wait_time:.1f}s...")
                time.sleep(wait_time)
            else:
                print(f"All {max_retries} attempts failed for question: {question[:50]}...")
                raise

    return None
```

## Integration Examples

### With Benchmark Pipeline

```python
def complete_benchmark_pipeline(input_file: str):
    """Complete pipeline from file to benchmark results."""

    from karenina.questions.extractor import extract_and_generate_questions
    from karenina.benchmark.runner import run_benchmark

    # Step 1: Extract questions
    extract_and_generate_questions(input_file, "questions.py")

    # Step 2: Generate answer templates
    templates = generate_answer_templates_from_questions_file("questions.py")

    # Step 3: Collect model responses (implementation depends on your needs)
    questions_dict = {}  # Load questions as dict
    responses_dict = {}  # Get responses from your model

    # Step 4: Run benchmark
    results = run_benchmark(questions_dict, responses_dict, templates)

    return results
```

### With Web API

```python
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

app = FastAPI()

class TemplateGenerationRequest(BaseModel):
    questions_file: str
    model: str = "gemini-2.0-flash"
    provider: str = "google_genai"

@app.post("/generate_templates")
async def generate_templates_endpoint(request: TemplateGenerationRequest):
    """API endpoint for template generation."""

    try:
        templates, code_blocks = generate_answer_templates_from_questions_file(
            request.questions_file,
            model=request.model,
            model_provider=request.provider,
            return_blocks=True
        )

        return {
            "success": True,
            "template_count": len(templates),
            "model": request.model,
            "provider": request.provider,
            "code_blocks": code_blocks
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/validate_template/{question_id}")
async def validate_template(question_id: str, response_data: dict):
    """Validate response data against template."""

    # Load templates (implement caching in production)
    templates = load_answer_templates_from_json("templates.json")

    if question_id not in templates:
        raise HTTPException(status_code=404, detail="Template not found")

    AnswerClass = templates[question_id]

    try:
        validated = AnswerClass(**response_data)
        return {
            "valid": True,
            "validated_data": validated.model_dump()
        }
    except Exception as e:
        return {
            "valid": False,
            "error": str(e)
        }
```

## Performance Optimization

### Parallel Generation

```python
import concurrent.futures
from functools import partial

def parallel_template_generation(questions_file: str, max_workers: int = 4):
    """Generate templates in parallel for better performance."""

    from karenina.questions.reader import read_questions_from_file

    all_questions = read_questions_from_file(questions_file)

    def generate_single_template(question):
        return question.id, generate_answer_template(
            question.question,
            question.model_dump_json()
        )

    templates = {}

    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_question = {
            executor.submit(generate_single_template, q): q
            for q in all_questions
        }

        for future in concurrent.futures.as_completed(future_to_question):
            question = future_to_question[future]
            try:
                question_id, template_code = future.result()

                # Process template code
                code_blocks = extract_and_combine_codeblocks(template_code)
                local_ns = {}
                exec(code_blocks, globals(), local_ns)

                templates[question_id] = local_ns["Answer"]

            except Exception as e:
                print(f"Error generating template for {question.id}: {e}")

    return templates
```
