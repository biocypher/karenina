# Benchmark Module

The `karenina.benchmark.runner` module executes structured evaluations of LLM responses against generated answer templates.

## Core Functions

### run_benchmark

Execute benchmark evaluation for questions, responses, and answer templates.

::: karenina.benchmark.runner.run_benchmark

**Parameters:**
- `question_dict` (dict): Questions mapped by ID `{question_id: question_text}`
- `response_dict` (dict): Model responses mapped by ID `{question_id: response_text}`
- `answer_templates` (dict): Answer template classes `{question_id: AnswerClass}`

**Returns:**
- `dict`: Structured evaluation results `{question_id: validated_answer_object}`

**Usage Examples:**

```python
from karenina.benchmark.runner import run_benchmark

# Prepare data
questions = {
    "q1_hash": "What is the capital of France?",
    "q2_hash": "What is 2 + 2?"
}

responses = {
    "q1_hash": "The capital of France is Paris, a major European city.",
    "q2_hash": "The sum of 2 and 2 equals 4."
}

# Answer templates (generated previously)
templates = {
    "q1_hash": ParisAnswerTemplate,  # Pydantic class
    "q2_hash": MathAnswerTemplate    # Pydantic class
}

# Run benchmark
results = run_benchmark(questions, responses, templates)

# Access structured results
for q_id, result in results.items():
    print(f"Question {q_id}: {result}")
    # result is validated Pydantic object with structured fields
```

**Processing Workflow:**

1. **Key Intersection**: Find common keys across all three input dictionaries
2. **Data Validation**: Warn if dictionaries have mismatched keys
3. **LLM Initialization**: Create evaluation LLM (gemini-2.5-flash-preview-05-20)
4. **Structured Evaluation**: For each question:
   - Format evaluation prompt with question and response
   - Use LLM with structured output (answer template)
   - Return validated Pydantic object

## Evaluation Process

### LLM Evaluator Configuration

The benchmark uses a fixed configuration:
- **Model**: `gemini-2.5-flash-preview-05-20`
- **Provider**: `google_genai`
- **Interface**: `langchain`
- **Structured Output**: Enforced via answer templates

### Prompt Structure

Uses prompts from `karenina.prompts.answer_evaluation`:

```python
# System prompt (ANSWER_EVALUATION_SYS)
# - Instructions for evaluation
# - Structured output requirements
# - Quality assessment criteria

# User prompt (ANSWER_EVALUATION_USER)
# - Question context: {question}
# - Response to evaluate: {response}
# - Template-specific validation
```

### Structured Output Enforcement

```python
# LLM generates response matching template structure
response = llm.with_structured_output(answer_template).invoke(messages)
```

## Data Handling

### Key Intersection Logic

```python
# Find common keys across all inputs
common_keys = set(question_dict.keys()) & set(response_dict.keys()) & set(answer_templates.keys())

# Warning for mismatched keys
if len(common_keys) != max(len(question_dict), len(response_dict), len(answer_templates)):
    warnings.warn("Dictionaries have different keys. Using intersection.")
```

### Filtered Processing

Only processes questions present in all three input dictionaries:

```python
# Filter to common keys only
filtered_questions = {k: v for k, v in question_dict.items() if k in common_keys}
filtered_responses = {k: v for k, v in response_dict.items() if k in common_keys}
filtered_templates = {k: v for k, v in answer_templates.items() if k in common_keys}
```

## Integration Examples

### Complete Evaluation Pipeline

```python
from karenina.questions.extractor import extract_and_generate_questions
from karenina.answers.generator import generate_answer_templates_from_questions_file
from karenina.benchmark.runner import run_benchmark

# 1. Extract questions
extract_and_generate_questions("data/benchmark.xlsx", "questions.py")

# 2. Generate answer templates
templates = generate_answer_templates_from_questions_file("questions.py")

# 3. Collect model responses (your implementation)
responses = collect_model_responses(questions, target_model="gpt-4")

# 4. Run benchmark
results = run_benchmark(questions, responses, templates)

# 5. Analyze results
analyze_benchmark_results(results)
```

### Multi-Model Evaluation

```python
models_to_test = ["gpt-4", "claude-3-sonnet", "gemini-2.0-flash"]
all_results = {}

for model in models_to_test:
    # Get responses from each model
    responses = get_model_responses(questions, model)
    
    # Run benchmark
    results = run_benchmark(questions, responses, templates)
    all_results[model] = results

# Compare model performance
compare_models(all_results)
```

### Result Analysis

```python
def analyze_results(results: dict):
    """Analyze benchmark results for insights."""
    
    for question_id, result in results.items():
        print(f"\nQuestion ID: {question_id}")
        
        # Access structured fields
        if hasattr(result, 'confidence'):
            print(f"Confidence: {result.confidence}")
        
        if hasattr(result, 'accuracy'):
            print(f"Accuracy: {result.accuracy}")
            
        # Convert to dict for full analysis
        result_dict = result.model_dump()
        print(f"Full result: {result_dict}")

# Usage
results = run_benchmark(questions, responses, templates)
analyze_results(results)
```

## Error Handling

### Common Issues and Solutions

```python
try:
    results = run_benchmark(questions, responses, templates)
except KeyError as e:
    print(f"Missing key in input dictionaries: {e}")
except Exception as e:
    print(f"Benchmark execution error: {e}")

# Validate inputs before running
def validate_benchmark_inputs(questions, responses, templates):
    """Validate input dictionaries for benchmark."""
    
    q_keys = set(questions.keys())
    r_keys = set(responses.keys())
    t_keys = set(templates.keys())
    
    if not q_keys:
        raise ValueError("Questions dictionary is empty")
    
    if not r_keys:
        raise ValueError("Responses dictionary is empty")
        
    if not t_keys:
        raise ValueError("Templates dictionary is empty")
    
    common = q_keys & r_keys & t_keys
    if not common:
        raise ValueError("No common keys found across input dictionaries")
    
    missing_in_responses = q_keys - r_keys
    if missing_in_responses:
        print(f"Warning: Missing responses for questions: {missing_in_responses}")
    
    missing_templates = q_keys - t_keys
    if missing_templates:
        print(f"Warning: Missing templates for questions: {missing_templates}")
    
    return common

# Use validation
common_keys = validate_benchmark_inputs(questions, responses, templates)
print(f"Will evaluate {len(common_keys)} questions")
```

### Template Validation Errors

```python
# Handle template validation failures
def safe_benchmark_run(questions, responses, templates):
    """Run benchmark with error handling for individual questions."""
    
    results = {}
    errors = {}
    
    for q_id in questions.keys():
        if q_id not in responses or q_id not in templates:
            continue
            
        try:
            # Single question benchmark
            single_result = run_benchmark(
                {q_id: questions[q_id]},
                {q_id: responses[q_id]},
                {q_id: templates[q_id]}
            )
            results[q_id] = single_result[q_id]
            
        except Exception as e:
            errors[q_id] = str(e)
            print(f"Error processing question {q_id}: {e}")
    
    return results, errors

# Usage with error tracking
results, errors = safe_benchmark_run(questions, responses, templates)
print(f"Successfully evaluated: {len(results)}")
print(f"Errors encountered: {len(errors)}")
```

## Performance Considerations

### Batch Processing

The current implementation processes questions sequentially. For large datasets:

```python
import concurrent.futures
from functools import partial

def parallel_benchmark(questions, responses, templates, max_workers=4):
    """Run benchmark with parallel processing."""
    
    # Split into chunks
    question_items = list(questions.items())
    chunk_size = len(question_items) // max_workers
    
    def process_chunk(chunk_items):
        chunk_questions = dict(chunk_items)
        chunk_responses = {k: responses[k] for k in chunk_questions.keys() if k in responses}
        chunk_templates = {k: templates[k] for k in chunk_questions.keys() if k in templates}
        
        return run_benchmark(chunk_questions, chunk_responses, chunk_templates)
    
    # Process chunks in parallel
    chunks = [question_items[i:i+chunk_size] for i in range(0, len(question_items), chunk_size)]
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        chunk_results = executor.map(process_chunk, chunks)
    
    # Combine results
    combined_results = {}
    for chunk_result in chunk_results:
        combined_results.update(chunk_result)
    
    return combined_results
```

### Memory Management

For large datasets, consider streaming processing:

```python
def streaming_benchmark(questions, responses, templates, batch_size=100):
    """Process benchmark in batches to manage memory usage."""
    
    all_results = {}
    question_ids = list(questions.keys())
    
    for i in range(0, len(question_ids), batch_size):
        batch_ids = question_ids[i:i+batch_size]
        
        batch_questions = {k: questions[k] for k in batch_ids}
        batch_responses = {k: responses[k] for k in batch_ids if k in responses}
        batch_templates = {k: templates[k] for k in batch_ids if k in templates}
        
        batch_results = run_benchmark(batch_questions, batch_responses, batch_templates)
        all_results.update(batch_results)
        
        print(f"Processed batch {i//batch_size + 1}/{(len(question_ids)-1)//batch_size + 1}")
    
    return all_results
```

## Advanced Usage

### Custom Evaluation Metrics

```python
def enhanced_benchmark_analysis(results: dict):
    """Analyze benchmark results with custom metrics."""
    
    metrics = {
        'total_questions': len(results),
        'field_completion_rates': {},
        'confidence_distribution': [],
        'validation_errors': 0
    }
    
    for q_id, result in results.items():
        # Analyze field completion
        result_dict = result.model_dump()
        for field, value in result_dict.items():
            if field not in metrics['field_completion_rates']:
                metrics['field_completion_rates'][field] = {'completed': 0, 'total': 0}
            
            metrics['field_completion_rates'][field]['total'] += 1
            if value is not None and value != "":
                metrics['field_completion_rates'][field]['completed'] += 1
        
        # Track confidence if available
        if hasattr(result, 'confidence') and result.confidence is not None:
            metrics['confidence_distribution'].append(result.confidence)
    
    # Calculate completion rates
    for field in metrics['field_completion_rates']:
        rate_data = metrics['field_completion_rates'][field]
        rate_data['rate'] = rate_data['completed'] / rate_data['total']
    
    return metrics
```