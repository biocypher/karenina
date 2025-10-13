# Running a Verification Job

This guide covers how to configure and execute verification jobs to evaluate LLM responses using your benchmark questions and templates.

## Understanding Verification

Verification in Karenina involves:

1. **Generating responses** from target LLMs for each question
2. **Parsing responses** using judge LLMs and templates
3. **Evaluating correctness** through template verification methods
4. **Scoring responses** using rubric criteria (if configured)
5. **Aggregating results** for analysis and reporting

## Setting Model Configuration

### Basic Model Configuration

```python
from karenina.schemas import ModelConfiguration

# Configure the model for verification
model_config = ModelConfiguration(
    provider="openai",
    model="gpt-4",
    temperature=0.1,
    max_tokens=1000
)
```

### Provider-Specific Configurations

```python
# OpenAI configuration
openai_config = ModelConfiguration(
    provider="openai",
    model="gpt-4-turbo",
    temperature=0.0,
    max_tokens=2000,
    top_p=1.0,
    frequency_penalty=0.0,
    presence_penalty=0.0
)

# Anthropic configuration
anthropic_config = ModelConfiguration(
    provider="anthropic",
    model="claude-3-sonnet",
    temperature=0.1,
    max_tokens=1500
)

# Google configuration
google_config = ModelConfiguration(
    provider="google",
    model="gemini-pro",
    temperature=0.2,
    max_tokens=1000
)
```

### Multi-Model Verification

```python
# Test multiple models in a single verification run
model_configs = [
    ModelConfiguration(provider="openai", model="gpt-4"),
    ModelConfiguration(provider="openai", model="gpt-4.1-mini"),
    ModelConfiguration(provider="anthropic", model="claude-3-sonnet"),
    ModelConfiguration(provider="google", model="gemini-pro")
]
```

## Running Verification

### Basic Verification

```python
# Ensure templates exist before verification
if not benchmark.all_questions_have_templates():
    benchmark.generate_answer_templates(
        model_config=model_config,
        system_prompt="Create evaluation template for this question"
    )

# Run verification
results = benchmark.run_verification(model_config=model_config)

print(f"Verification completed: {len(results)} results generated")
```

### Advanced Verification Options

```python
# Verification with custom options
results = benchmark.run_verification(
    model_config=model_config,
    judge_model_config=judge_config,  # Separate config for judge LLM
    include_rubric_evaluation=True,   # Enable rubric scoring
    parallel_requests=5,              # Number of concurrent requests
    retry_attempts=3,                 # Retries for failed requests
    timeout_seconds=30,               # Request timeout
    save_raw_responses=True           # Keep original LLM responses
)
```

### Verification with Filtering

```python
# Verify only specific questions
math_questions = benchmark.filter_questions(category="mathematics")
results = benchmark.run_verification(
    model_config=model_config,
    questions=math_questions  # Verify subset of questions
)

# Verify only unfinished questions
unverified = benchmark.get_questions_without_verification()
results = benchmark.run_verification(
    model_config=model_config,
    questions=unverified
)
```

## Verification Results

### Understanding Result Structure

```python
for result in results:
    print(f"Question ID: {result.question_id}")
    print(f"Model: {result.model_config.model}")
    print(f"Template Verification: {result.template_passed}")
    print(f"Raw Response: {result.raw_response}")
    print(f"Parsed Response: {result.parsed_response}")

    if result.rubric_evaluation:
        print(f"Rubric Scores: {result.rubric_evaluation}")

    if result.error:
        print(f"Error: {result.error}")
    print("-" * 50)
```

### Result Analysis

```python
def analyze_verification_results(results):
    """Analyze verification results for insights"""

    total_questions = len(results)
    passed_template = sum(1 for r in results if r.template_passed)
    failed_template = total_questions - passed_template

    print(f"=== Verification Summary ===")
    print(f"Total Questions: {total_questions}")
    print(f"Template Verification Passed: {passed_template} ({passed_template/total_questions*100:.1f}%)")
    print(f"Template Verification Failed: {failed_template} ({failed_template/total_questions*100:.1f}%)")

    # Analyze by category
    category_stats = {}
    for result in results:
        category = result.question.metadata.get("category", "unknown")
        if category not in category_stats:
            category_stats[category] = {"total": 0, "passed": 0}

        category_stats[category]["total"] += 1
        if result.template_passed:
            category_stats[category]["passed"] += 1

    print(f"\n=== Results by Category ===")
    for category, stats in category_stats.items():
        pass_rate = stats["passed"] / stats["total"] * 100
        print(f"{category}: {stats['passed']}/{stats['total']} ({pass_rate:.1f}%)")

# Analyze results
analyze_verification_results(results)
```

## Error Handling and Debugging

### Handling Verification Errors

```python
def handle_verification_errors(results):
    """Process and report verification errors"""

    errors_by_type = {}

    for result in results:
        if result.error:
            error_type = type(result.error).__name__
            if error_type not in errors_by_type:
                errors_by_type[error_type] = []

            errors_by_type[error_type].append({
                "question_id": result.question_id,
                "error": str(result.error),
                "model": result.model_config.model
            })

    # Report errors
    for error_type, errors in errors_by_type.items():
        print(f"\n{error_type} ({len(errors)} occurrences):")
        for error in errors[:3]:  # Show first 3 examples
            print(f"  Question {error['question_id']}: {error['error']}")
        if len(errors) > 3:
            print(f"  ... and {len(errors) - 3} more")

# Handle errors
handle_verification_errors(results)
```

### Retry Failed Verifications

```python
def retry_failed_verifications(benchmark, results, model_config):
    """Retry verification for failed questions"""

    failed_question_ids = [r.question_id for r in results if r.error or not r.template_passed]

    if not failed_question_ids:
        print("No failed verifications to retry")
        return []

    print(f"Retrying {len(failed_question_ids)} failed verifications...")

    failed_questions = [q for q in benchmark.questions if q.id in failed_question_ids]

    retry_results = benchmark.run_verification(
        model_config=model_config,
        questions=failed_questions,
        retry_attempts=5,  # More aggressive retry
        timeout_seconds=60  # Longer timeout
    )

    return retry_results
```

## Comparative Verification

### Multi-Model Comparison

```python
def compare_models(benchmark, model_configs):
    """Compare performance across different models"""

    comparison_results = {}

    for config in model_configs:
        model_name = f"{config.provider}-{config.model}"
        print(f"Running verification with {model_name}...")

        results = benchmark.run_verification(model_config=config)
        comparison_results[model_name] = results

    # Analyze comparative performance
    print(f"\n=== Model Comparison ===")
    for model_name, results in comparison_results.items():
        passed = sum(1 for r in results if r.template_passed)
        total = len(results)
        print(f"{model_name}: {passed}/{total} ({passed/total*100:.1f}%)")

    return comparison_results

# Compare multiple models
model_configs = [
    ModelConfiguration(provider="openai", model="gpt-4"),
    ModelConfiguration(provider="openai", model="gpt-4.1-mini"),
    ModelConfiguration(provider="anthropic", model="claude-3-sonnet")
]

comparison = compare_models(benchmark, model_configs)
```

### A/B Testing Templates

```python
def ab_test_templates(benchmark, template_a, template_b, model_config):
    """Compare two different templates on the same questions"""

    # Split questions randomly
    import random
    questions = list(benchmark.questions)
    random.shuffle(questions)

    mid_point = len(questions) // 2
    group_a = questions[:mid_point]
    group_b = questions[mid_point:]

    # Apply different templates
    for question in group_a:
        question.answer_template = template_a

    for question in group_b:
        question.answer_template = template_b

    # Run verification
    results_a = benchmark.run_verification(model_config=model_config, questions=group_a)
    results_b = benchmark.run_verification(model_config=model_config, questions=group_b)

    # Compare results
    pass_rate_a = sum(1 for r in results_a if r.template_passed) / len(results_a)
    pass_rate_b = sum(1 for r in results_b if r.template_passed) / len(results_b)

    print(f"Template A pass rate: {pass_rate_a:.2%}")
    print(f"Template B pass rate: {pass_rate_b:.2%}")

    return results_a, results_b
```

## Performance Optimization

### Parallel Processing

```python
# Configure for high-throughput verification
high_throughput_config = ModelConfiguration(
    provider="openai",
    model="gpt-4.1-mini",  # Faster model
    temperature=0.0,        # Deterministic
    max_tokens=500         # Shorter responses
)

# Run with high parallelism
results = benchmark.run_verification(
    model_config=high_throughput_config,
    parallel_requests=20,   # High concurrency
    batch_size=50,         # Process in batches
    timeout_seconds=15     # Quick timeout
)
```

### Incremental Verification

```python
def incremental_verification(benchmark, model_config, batch_size=10):
    """Process verification in increments to handle large benchmarks"""

    all_results = []
    questions = list(benchmark.questions)

    for i in range(0, len(questions), batch_size):
        batch = questions[i:i + batch_size]
        print(f"Processing batch {i//batch_size + 1}/{(len(questions) + batch_size - 1)//batch_size}")

        batch_results = benchmark.run_verification(
            model_config=model_config,
            questions=batch
        )

        all_results.extend(batch_results)

        # Optional: save intermediate results
        benchmark.save_checkpoint(f"checkpoint_batch_{i//batch_size + 1}.jsonld")

    return all_results
```

## Next Steps

After running verification:

- [Analyze results](../api-reference.md#verification-results) to understand model performance
- [Save results](saving-loading.md) to preserve verification outcomes
- Export results for further analysis or reporting
