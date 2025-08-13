# Benchmark Execution Guide

This guide covers the complete process of executing benchmarks to evaluate LLM responses against structured answer templates.

## Overview

Benchmark execution validates LLM responses using generated Pydantic templates, providing structured evaluation results with consistent scoring and analysis.

## Configuration with ModelConfiguration and VerificationConfig

Use these models to describe answering/parsing models and overall run settings.

```python
from karenina.benchmark.models import ModelConfiguration, VerificationConfig

answering = [
    ModelConfiguration(
        id="gpt4",
        model_provider="openai",
        model_name="gpt-4",
        temperature=0.1,
        interface="langchain",
        system_prompt="You are an expert assistant."
    ),
]

parsing = [
    ModelConfiguration(
        id="gpt35",
        model_provider="openai",
        model_name="gpt-3.5-turbo",
        temperature=0.0,
        interface="langchain",
        system_prompt="Parse and validate the response."
    ),
]

config = VerificationConfig(
    answering_models=answering,
    parsing_models=parsing,
    replicate_count=1,
    rubric_enabled=False,
)
```

### Multiple answering/parsing models

```python
answering = [
    ModelConfiguration(
        id="gpt4",
        model_provider="openai",
        model_name="gpt-4",
        temperature=0.1,
        interface="langchain",
        system_prompt="Be precise."
    ),
    ModelConfiguration(
        id="sonnet",
        model_provider="anthropic",
        model_name="claude-3-sonnet",
        temperature=0.1,
        interface="langchain",
        system_prompt="Be precise."
    ),
]

parsing = [
    ModelConfiguration(
        id="gpt35",
        model_provider="openai",
        model_name="gpt-3.5-turbo",
        temperature=0.0,
        interface="langchain",
        system_prompt="Parse strictly."
    ),
]

config = VerificationConfig(answering_models=answering, parsing_models=parsing, replicate_count=2)
```

### OpenRouter and manual interfaces

For `interface="openrouter"` or `interface="manual"`, `model_provider` may be empty.

```python
openrouter_cfg = ModelConfiguration(
    id="o1-mini",
    model_provider="",  # allowed
    model_name="openrouter/o1-mini",
    temperature=0.2,
    interface="openrouter",
    system_prompt="Be concise."
)

manual_cfg = ModelConfiguration(
    id="human",
    model_provider="",
    model_name="human-expert",
    temperature=0.0,
    interface="manual",
    system_prompt="N/A"
)

config = VerificationConfig(answering_models=[openrouter_cfg, manual_cfg], parsing_models=[parsing[0]])
```

### Rubric options

```python
from karenina.schemas.rubric_class import RubricTrait

config = VerificationConfig(
    answering_models=answering,
    parsing_models=parsing,
    replicate_count=3,
    rubric_enabled=True,
    rubric_trait_names=["clarity", "completeness"],  # optional filter
)
```

Validation rules include: at least one answering and parsing model; required fields set; `model_provider` optional only for `openrouter`/`manual` interfaces.

## Basic Benchmark Execution

### Single Model Evaluation (with VerificationConfig)

```python
from karenina.benchmark import Benchmark
from karenina.benchmark.models import ModelConfiguration, VerificationConfig

# Minimal benchmark with one question
bm = Benchmark.create("Exec Demo")
qid = bm.add_question("What is the capital of France?", "Paris")

# Configure models
cfg = VerificationConfig(
    answering_models=[
        ModelConfiguration(
            id="gpt4",
            model_provider="openai",
            model_name="gpt-4",
            temperature=0.1,
            interface="langchain",
            system_prompt="Answer accurately."
        )
    ],
    parsing_models=[
        ModelConfiguration(
            id="gpt35",
            model_provider="openai",
            model_name="gpt-3.5-turbo",
            temperature=0.0,
            interface="langchain",
            system_prompt="Parse strictly."
        )
    ]
)

# Run verification on finished templates/questions as appropriate
# Example: run for specific IDs
results = bm.verify_questions([qid], cfg)
print(results[qid].model_dump())
```

### Complete Pipeline Example

```python
from karenina.questions.extractor import extract_and_generate_questions
from karenina.answers.generator import generate_answer_templates_from_questions_file
from karenina.benchmark.runner import run_benchmark

def complete_benchmark_pipeline(data_file: str, model_name: str):
    """Execute complete benchmark pipeline."""

    # Step 1: Extract questions
    print("Extracting questions...")
    extract_and_generate_questions(
        file_path=data_file,
        output_path="benchmark_questions.py",
        question_column="Question",
        answer_column="Answer"
    )

    # Step 2: Generate answer templates
    print("Generating answer templates...")
    templates = generate_answer_templates_from_questions_file("benchmark_questions.py")

    # Step 3: Collect model responses
    print(f"Collecting responses from {model_name}...")
    questions_dict, responses_dict = collect_model_responses(
        "benchmark_questions.py",
        model_name
    )

    # Step 4: Run benchmark
    print("Running benchmark evaluation...")
    results = run_benchmark(questions_dict, responses_dict, templates)

    # Step 5: Analyze and report
    print("Analyzing results...")
    analysis = analyze_benchmark_results(results)

    return results, analysis

# Usage
results, analysis = complete_benchmark_pipeline("data/benchmark.xlsx", "gpt-4")
```

## Model Response Collection

### LLM Interface Integration

```python
from karenina.llm.interface import call_model
from karenina.questions.reader import read_questions_from_file

def collect_model_responses(questions_file: str, model: str, provider: str):
    """Collect responses from a specific model."""

    # Load questions
    questions = read_questions_from_file(questions_file)

    questions_dict = {}
    responses_dict = {}

    for question in questions:
        questions_dict[question.id] = question.question

        # Get model response
        try:
            response = call_model(
                model=model,
                provider=provider,
                message=question.question,
                temperature=0.3
            )
            responses_dict[question.id] = response.message

        except Exception as e:
            print(f"Error getting response for {question.id}: {e}")
            responses_dict[question.id] = ""

    return questions_dict, responses_dict

# Usage
questions_dict, responses_dict = collect_model_responses(
    "benchmark_questions.py",
    model="gpt-4",
    provider="openai"
)
```

### Batch Response Collection

```python
def collect_batch_responses(questions_file: str, models_config: list):
    """Collect responses from multiple models."""

    from karenina.questions.reader import read_questions_from_file

    questions = read_questions_from_file(questions_file)
    questions_dict = {q.id: q.question for q in questions}

    all_responses = {}

    for config in models_config:
        model_key = f"{config['provider']}_{config['model']}"
        print(f"Collecting responses from {model_key}...")

        responses = {}

        for question in questions:
            try:
                response = call_model(
                    model=config['model'],
                    provider=config['provider'],
                    message=question.question,
                    temperature=config.get('temperature', 0.3)
                )
                responses[question.id] = response.message

            except Exception as e:
                print(f"Error with {model_key} for {question.id}: {e}")
                responses[question.id] = ""

        all_responses[model_key] = responses

    return questions_dict, all_responses

# Usage
models_to_test = [
    {"model": "gpt-4", "provider": "openai", "temperature": 0.3},
    {"model": "claude-3-sonnet", "provider": "anthropic", "temperature": 0.3},
    {"model": "gemini-2.0-flash", "provider": "google_genai", "temperature": 0.3}
]

questions_dict, all_responses = collect_batch_responses("questions.py", models_to_test)
```

## Multi-Model Benchmarking

### Comparative Evaluation

```python
def multi_model_benchmark(questions_file: str, models_config: list):
    """Run benchmark across multiple models for comparison."""

    # Generate templates once
    templates = generate_answer_templates_from_questions_file(questions_file)

    # Collect responses from all models
    questions_dict, all_responses = collect_batch_responses(questions_file, models_config)

    # Run benchmark for each model
    all_results = {}

    for model_key, responses in all_responses.items():
        print(f"Evaluating {model_key}...")

        results = run_benchmark(questions_dict, responses, templates)
        all_results[model_key] = results

    return all_results

# Usage
models = [
    {"model": "gpt-4", "provider": "openai"},
    {"model": "claude-3-sonnet", "provider": "anthropic"},
    {"model": "gemini-2.0-flash", "provider": "google_genai"}
]

comparative_results = multi_model_benchmark("questions.py", models)
```

### Results Comparison

```python
def compare_model_results(all_results: dict):
    """Compare results across multiple models."""

    comparison = {}

    # Get all question IDs
    all_question_ids = set()
    for results in all_results.values():
        all_question_ids.update(results.keys())

    for question_id in all_question_ids:
        comparison[question_id] = {}

        for model_key, results in all_results.items():
            if question_id in results:
                result = results[question_id]

                # Extract structured data
                result_dict = result.model_dump()
                comparison[question_id][model_key] = result_dict
            else:
                comparison[question_id][model_key] = None

    return comparison

# Usage
comparison = compare_model_results(comparative_results)

# Analyze specific question across models
question_id = list(comparison.keys())[0]
print(f"Comparison for question {question_id}:")
for model, result in comparison[question_id].items():
    print(f"  {model}: {result}")
```

## Result Analysis and Reporting

### Structured Result Analysis

```python
def analyze_benchmark_results(results: dict) -> dict:
    """Comprehensive analysis of benchmark results."""

    analysis = {
        'total_questions': len(results),
        'field_analysis': {},
        'score_distribution': {},
        'completion_rates': {},
        'common_patterns': []
    }

    all_fields = set()
    field_values = {}

    # Collect all field data
    for question_id, result in results.items():
        result_dict = result.model_dump()

        for field_name, value in result_dict.items():
            all_fields.add(field_name)

            if field_name not in field_values:
                field_values[field_name] = []

            field_values[field_name].append(value)

    # Analyze each field
    for field_name in all_fields:
        values = field_values[field_name]
        non_null_values = [v for v in values if v is not None and v != ""]

        analysis['field_analysis'][field_name] = {
            'total_responses': len(values),
            'non_null_responses': len(non_null_values),
            'completion_rate': len(non_null_values) / len(values) if values else 0,
            'unique_values': len(set(str(v) for v in non_null_values)),
        }

        # Type-specific analysis
        if non_null_values:
            if all(isinstance(v, (int, float)) for v in non_null_values):
                import statistics
                analysis['field_analysis'][field_name].update({
                    'mean': statistics.mean(non_null_values),
                    'median': statistics.median(non_null_values),
                    'min': min(non_null_values),
                    'max': max(non_null_values)
                })

    return analysis

# Usage
analysis = analyze_benchmark_results(results)

print("Benchmark Analysis:")
print(f"Total questions: {analysis['total_questions']}")
print("\nField Analysis:")
for field, stats in analysis['field_analysis'].items():
    print(f"  {field}:")
    print(f"    Completion rate: {stats['completion_rate']:.2%}")
    print(f"    Unique values: {stats['unique_values']}")
    if 'mean' in stats:
        print(f"    Mean: {stats['mean']:.2f}")
```

### Performance Metrics

```python
def calculate_performance_metrics(results: dict, ground_truth: dict = None):
    """Calculate performance metrics from benchmark results."""

    metrics = {
        'response_quality': {},
        'confidence_scores': [],
        'accuracy_measures': {},
        'field_coverage': {}
    }

    for question_id, result in results.items():
        result_dict = result.model_dump()

        # Collect confidence scores if available
        if 'confidence' in result_dict and result_dict['confidence'] is not None:
            metrics['confidence_scores'].append(result_dict['confidence'])

        # Field coverage analysis
        total_fields = len(result_dict)
        populated_fields = sum(1 for v in result_dict.values() if v is not None and v != "")

        metrics['field_coverage'][question_id] = {
            'total_fields': total_fields,
            'populated_fields': populated_fields,
            'coverage_rate': populated_fields / total_fields if total_fields > 0 else 0
        }

        # Compare with ground truth if available
        if ground_truth and question_id in ground_truth:
            # Implement comparison logic based on your needs
            pass

    # Calculate aggregate metrics
    if metrics['confidence_scores']:
        import statistics
        metrics['avg_confidence'] = statistics.mean(metrics['confidence_scores'])
        metrics['confidence_std'] = statistics.stdev(metrics['confidence_scores']) if len(metrics['confidence_scores']) > 1 else 0

    coverage_rates = [info['coverage_rate'] for info in metrics['field_coverage'].values()]
    if coverage_rates:
        metrics['avg_field_coverage'] = statistics.mean(coverage_rates)

    return metrics

# Usage
performance = calculate_performance_metrics(results)
print(f"Average confidence: {performance.get('avg_confidence', 'N/A')}")
print(f"Average field coverage: {performance.get('avg_field_coverage', 'N/A'):.2%}")
```

## Advanced Benchmarking Patterns

### Custom Evaluation Criteria

```python
def custom_benchmark_evaluation(questions_dict, responses_dict, templates, evaluator_config):
    """Run benchmark with custom evaluation criteria."""

    # Standard benchmark
    standard_results = run_benchmark(questions_dict, responses_dict, templates)

    # Apply custom evaluation
    enhanced_results = {}

    for question_id, standard_result in standard_results.items():
        enhanced_result = {
            'standard_evaluation': standard_result.model_dump(),
            'custom_metrics': {}
        }

        # Custom evaluation logic
        question = questions_dict[question_id]
        response = responses_dict[question_id]

        # Length analysis
        enhanced_result['custom_metrics']['response_length'] = len(response)
        enhanced_result['custom_metrics']['question_length'] = len(question)
        enhanced_result['custom_metrics']['length_ratio'] = len(response) / len(question) if len(question) > 0 else 0

        # Keyword matching
        question_keywords = set(question.lower().split())
        response_keywords = set(response.lower().split())
        keyword_overlap = len(question_keywords & response_keywords)
        enhanced_result['custom_metrics']['keyword_overlap'] = keyword_overlap

        # Apply evaluator-specific criteria
        if evaluator_config.get('check_specificity'):
            # Check for specific terms or patterns
            specificity_score = calculate_specificity(response)
            enhanced_result['custom_metrics']['specificity'] = specificity_score

        enhanced_results[question_id] = enhanced_result

    return enhanced_results

def calculate_specificity(response: str) -> float:
    """Calculate specificity score for a response."""

    specific_indicators = ['exactly', 'precisely', 'specifically', 'namely', 'in particular']
    vague_indicators = ['maybe', 'possibly', 'probably', 'generally', 'usually']

    specific_count = sum(1 for indicator in specific_indicators if indicator in response.lower())
    vague_count = sum(1 for indicator in vague_indicators if indicator in response.lower())

    if specific_count + vague_count == 0:
        return 0.5  # Neutral

    return specific_count / (specific_count + vague_count)

# Usage
evaluator_config = {
    'check_specificity': True,
    'analyze_length': True
}

enhanced_results = custom_benchmark_evaluation(
    questions_dict, responses_dict, templates, evaluator_config
)
```

### Batch Processing for Large Datasets

```python
def large_scale_benchmark(questions_file: str, responses_file: str, batch_size: int = 100):
    """Process large datasets in batches."""

    import json
    from karenina.questions.reader import read_questions_from_file

    # Load templates
    templates = generate_answer_templates_from_questions_file(questions_file)

    # Load responses
    with open(responses_file) as f:
        all_responses = json.load(f)

    # Load questions
    questions = read_questions_from_file(questions_file)
    questions_dict = {q.id: q.question for q in questions}

    # Process in batches
    all_results = {}
    question_ids = list(questions_dict.keys())

    for i in range(0, len(question_ids), batch_size):
        batch_ids = question_ids[i:i+batch_size]

        print(f"Processing batch {i//batch_size + 1}/{(len(question_ids)-1)//batch_size + 1}")

        # Create batch dictionaries
        batch_questions = {qid: questions_dict[qid] for qid in batch_ids}
        batch_responses = {qid: all_responses[qid] for qid in batch_ids if qid in all_responses}
        batch_templates = {qid: templates[qid] for qid in batch_ids if qid in templates}

        # Run benchmark on batch
        batch_results = run_benchmark(batch_questions, batch_responses, batch_templates)
        all_results.update(batch_results)

    return all_results

# Usage
large_results = large_scale_benchmark(
    "large_questions.py",
    "large_responses.json",
    batch_size=50
)
```

## Error Handling and Recovery

### Robust Benchmark Execution

```python
def robust_benchmark_execution(questions_dict, responses_dict, templates):
    """Execute benchmark with comprehensive error handling."""

    successful_results = {}
    failed_evaluations = {}

    for question_id in questions_dict.keys():
        if question_id not in responses_dict:
            failed_evaluations[question_id] = "No response available"
            continue

        if question_id not in templates:
            failed_evaluations[question_id] = "No template available"
            continue

        try:
            # Single question benchmark
            single_result = run_benchmark(
                {question_id: questions_dict[question_id]},
                {question_id: responses_dict[question_id]},
                {question_id: templates[question_id]}
            )

            successful_results[question_id] = single_result[question_id]

        except Exception as e:
            failed_evaluations[question_id] = f"Evaluation failed: {str(e)}"

    return successful_results, failed_evaluations

# Usage
successful, failed = robust_benchmark_execution(questions_dict, responses_dict, templates)

print(f"Successful evaluations: {len(successful)}")
print(f"Failed evaluations: {len(failed)}")

if failed:
    print("\nFirst 5 failures:")
    for qid, error in list(failed.items())[:5]:
        print(f"  {qid}: {error}")
```

### Partial Result Recovery

```python
def recover_partial_results(results_dir: str):
    """Recover partial results from interrupted benchmark runs."""

    import os
    import pickle
    from pathlib import Path

    results_path = Path(results_dir)
    partial_files = list(results_path.glob("partial_*.pkl"))

    if not partial_files:
        print("No partial results found")
        return {}

    combined_results = {}

    for partial_file in partial_files:
        try:
            with open(partial_file, 'rb') as f:
                partial_results = pickle.load(f)

            combined_results.update(partial_results)
            print(f"Recovered {len(partial_results)} results from {partial_file.name}")

        except Exception as e:
            print(f"Error loading {partial_file.name}: {e}")

    return combined_results

# Save partial results during processing
def save_partial_results(results: dict, batch_num: int, results_dir: str):
    """Save partial results during long-running benchmarks."""

    import pickle
    from pathlib import Path

    Path(results_dir).mkdir(exist_ok=True)
    partial_file = Path(results_dir) / f"partial_{batch_num:04d}.pkl"

    with open(partial_file, 'wb') as f:
        pickle.dump(results, f)
```

## Reporting and Visualization

### Generate Benchmark Reports

```python
def generate_benchmark_report(results: dict, output_file: str):
    """Generate comprehensive benchmark report."""

    analysis = analyze_benchmark_results(results)
    performance = calculate_performance_metrics(results)

    report = f"""
# Benchmark Evaluation Report

## Summary
- Total Questions: {analysis['total_questions']}
- Average Field Coverage: {performance.get('avg_field_coverage', 0):.2%}
- Average Confidence: {performance.get('avg_confidence', 'N/A')}

## Field Analysis
"""

    for field_name, stats in analysis['field_analysis'].items():
        report += f"""
### {field_name}
- Completion Rate: {stats['completion_rate']:.2%}
- Unique Values: {stats['unique_values']}
"""
        if 'mean' in stats:
            report += f"- Mean Value: {stats['mean']:.2f}\n"
            report += f"- Range: {stats['min']:.2f} - {stats['max']:.2f}\n"

    report += """
## Individual Results
"""

    for question_id, result in list(results.items())[:10]:  # First 10 results
        report += f"""
### Question {question_id}
```json
{result.model_dump_json(indent=2)}
```
"""

    # Save report
    with open(output_file, 'w') as f:
        f.write(report)

    print(f"Report saved to {output_file}")

# Usage
generate_benchmark_report(results, "benchmark_report.md")
```
