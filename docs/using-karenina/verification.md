# Running Verification

This guide covers how to configure and execute verification to evaluate LLM responses using your benchmark questions, templates, and rubrics.

## Understanding Verification

Verification in Karenina evaluates LLM responses through a structured workflow:

1. **Question Selection**: Choose which questions to evaluate
2. **Answer Generation**: LLMs generate responses to questions
3. **Answer Parsing**: Judge LLMs extract structured data using templates
4. **Template Verification**: Check if extracted data matches expected answers
5. **Rubric Evaluation**: Assess qualitative traits (if enabled)
6. **Result Aggregation**: Collect metrics and scores for analysis

This two-model approach (answering model + judge model) allows natural language responses while maintaining structured evaluation.

## Verification Workflow

```
Questions → Answering Models → Raw Responses → Judge Models → Parsed Data → Verification
                                                                                    ↓
                                                                              Results with
                                                                              Scores & Metrics
```

**Key Concepts:**

- **Answering Models**: Generate responses to questions (can be any LLM)
- **Parsing Models** (Judges): Extract structured data from responses using templates
- **Templates**: Pydantic classes defining expected answer structure
- **Rubrics**: Qualitative evaluation criteria (optional)
- **Replication**: Run same question multiple times for statistical significance

---

## Basic Verification Configuration

### Configure Verification

Use `VerificationConfig` to specify how verification runs:

```python
from karenina.schemas import VerificationConfig, ModelConfig

# Configure verification
config = VerificationConfig(
    # Models that generate answers
    answering_models=[
        ModelConfig(
            id="gpt-4.1-mini",
            model_provider="openai",
            model_name="gpt-4.1-mini",
            temperature=0.7,
            interface="langchain"
        )
    ],

    # Models that parse/judge answers
    parsing_models=[
        ModelConfig(
            id="gpt-4.1-mini-judge",
            model_provider="openai",
            model_name="gpt-4.1-mini",
            temperature=0.0,  # Deterministic parsing
            interface="langchain"
        )
    ],

    # Evaluation settings
    evaluation_mode="template_only",  # or "template_and_rubric", "rubric_only"
    rubric_enabled=False,
    replicate_count=1
)
```

### ModelConfig Parameters

For a comprehensive guide to `ModelConfig` including all parameters, interfaces, providers, and the `extra_kwargs` feature, see the **[Model Configuration Guide](model-configuration.md)**.

**Quick reference**:

| Parameter | Type | Description | Example |
|-----------|------|-------------|---------|
| `id` | `str` | Unique identifier for this model | `"gpt-4.1-mini"` |
| `model_provider` | `str` | Provider name | `"openai"`, `"anthropic"`, `"google"` |
| `model_name` | `str` | Full model name | `"gpt-4.1-mini"`, `"claude-sonnet-4.5"` |
| `temperature` | `float` | Sampling temperature (0.0-1.0) | `0.7` for creativity, `0.0` for determinism |
| `interface` | `str` | Interface type | `"langchain"`, `"openrouter"`, `"openai_endpoint"`, `"manual"` |
| `system_prompt` | `str` | Optional system prompt | `"You are a knowledgeable genomics expert"` |
| `extra_kwargs` | `dict` | Additional keyword arguments | See [Model Configuration](model-configuration.md#extra-keyword-arguments) |

---

## Running Verification

### Basic Verification

Once you have templates and optionally rubrics, run verification:

```python
# Ensure all questions have templates
# (see templates.md for template generation)

# Run verification
results = benchmark.run_verification(config)

print(f"Verification complete: {len(results)} results generated")
```

### Verify Specific Questions

Verify only a subset of questions by providing question IDs:

```python
# Get question IDs for specific category
genomics_question_ids = [
    qid for qid in benchmark.questions.keys()
    if "chromosome" in benchmark.get_question(qid).question.lower()
]

# Run verification on subset
results = benchmark.run_verification(
    config=config,
    question_ids=genomics_question_ids
)

print(f"Verified {len(results)} genomics questions")
```

---

## Multi-Model Support

Karenina supports testing multiple models simultaneously and using different models for answering vs judging.

### Test Multiple Answering Models

Compare performance across different LLMs:

```python
from karenina.schemas import VerificationConfig, ModelConfig

config = VerificationConfig(
    # Multiple answering models to compare
    answering_models=[
        ModelConfig(
            id="gpt-4.1-mini",
            model_provider="openai",
            model_name="gpt-4.1-mini",
            temperature=0.7,
            interface="langchain"
        ),
        ModelConfig(
            id="claude-sonnet",
            model_provider="anthropic",
            model_name="claude-sonnet-4.5",
            temperature=0.7,
            interface="langchain"
        ),
        ModelConfig(
            id="gemini-pro",
            model_provider="google",
            model_name="gemini-2.5-flash",
            temperature=0.7,
            interface="langchain"
        )
    ],

    # Single judge model for consistent evaluation
    parsing_models=[
        ModelConfig(
            id="gpt-4.1-mini-judge",
            model_provider="openai",
            model_name="gpt-4.1-mini",
            temperature=0.0,
            interface="langchain"
        )
    ]
)

# This will generate 3 results per question (one per answering model)
results = benchmark.run_verification(config)
```

### Different Answering and Judge Models

Use a more capable model for judging:

```python
config = VerificationConfig(
    # Fast model for generating answers
    answering_models=[
        ModelConfig(
            id="gpt-4.1-mini",
            model_provider="openai",
            model_name="gpt-4.1-mini",
            temperature=0.7,
            interface="langchain"
        )
    ],

    # More capable model for judging
    parsing_models=[
        ModelConfig(
            id="gpt-4.1-large-judge",
            model_provider="openai",
            model_name="gpt-4.1",
            temperature=0.0,
            interface="langchain"
        )
    ]
)
```

### Multiple Judge Models

Compare how different judges evaluate the same answers:

```python
config = VerificationConfig(
    # Single answering model
    answering_models=[
        ModelConfig(
            id="gpt-4.1-mini",
            model_provider="openai",
            model_name="gpt-4.1-mini",
            temperature=0.7,
            interface="langchain"
        )
    ],

    # Multiple judges for comparison
    parsing_models=[
        ModelConfig(
            id="gpt-judge",
            model_provider="openai",
            model_name="gpt-4.1-mini",
            temperature=0.0,
            interface="langchain"
        ),
        ModelConfig(
            id="claude-judge",
            model_provider="anthropic",
            model_name="claude-sonnet-4.5",
            temperature=0.0,
            interface="langchain"
        )
    ]
)

# This will generate 2 results per question (one per judge)
# with automatic answer caching (see "Answer Caching" section)
results = benchmark.run_verification(config)
```

---

## Replication for Statistical Analysis

Run the same question multiple times to assess model consistency and compute statistical metrics.

### Configure Replication

```python
config = VerificationConfig(
    answering_models=[model_config],
    parsing_models=[judge_config],
    replicate_count=5  # Run each question 5 times
)

results = benchmark.run_verification(config)

# Results will include 5 independent runs per question
# Each run gets a unique replicate number (0, 1, 2, 3, 4)
```

### Analyze Replication Results

```python
from collections import defaultdict

# Group results by question
results_by_question = defaultdict(list)
for result_id, result in results.items():
    results_by_question[result.question_id].append(result)

# Compute pass rate for each question
for question_id, question_results in results_by_question.items():
    question = benchmark.get_question(question_id)
    pass_count = sum(1 for r in question_results if r.verify_result)
    total = len(question_results)
    pass_rate = pass_count / total

    print(f"{question.question[:50]}...")
    print(f"  Pass Rate: {pass_rate:.1%} ({pass_count}/{total})")

    # Check consistency
    if pass_rate == 1.0:
        print("  ✓ Consistent: Always correct")
    elif pass_rate == 0.0:
        print("  ✗ Consistent: Always incorrect")
    else:
        print(f"  ⚠ Inconsistent: {pass_rate:.1%} accuracy")
    print()
```

**Use Cases:**
- **Model Reliability**: Assess how consistently a model answers correctly
- **Statistical Significance**: Run 10+ replicates for robust metrics
- **Temperature Effects**: Compare variance at different temperature settings

---

## Evaluation Modes

Karenina supports three evaluation modes that control what gets evaluated during verification.

### Mode 1: template_only (Default)

Evaluate responses against templates only. Fast and focused on factual correctness.

```python
config = VerificationConfig(
    answering_models=[model_config],
    parsing_models=[judge_config],
    evaluation_mode="template_only",
    rubric_enabled=False  # Must be False
)

results = benchmark.run_verification(config)

# Results include:
# - template_verification_performed = True
# - verify_result = True/False (template pass/fail)
# - rubric_evaluation_performed = False
# - verify_rubric = None
```

**When to use:**
- Testing template parsing accuracy
- Fastest verification (no rubric overhead)
- Focus on structured output correctness

### Mode 2: template_and_rubric

Evaluate both template correctness AND rubric criteria. Comprehensive evaluation.

```python
config = VerificationConfig(
    answering_models=[model_config],
    parsing_models=[judge_config],
    evaluation_mode="template_and_rubric",
    rubric_enabled=True  # Must be True
)

results = benchmark.run_verification(config)

# Results include:
# - template_verification_performed = True
# - verify_result = True/False (template pass/fail)
# - rubric_evaluation_performed = True
# - verify_rubric = {"Clarity": 4, "Conciseness": 5, ...}
```

**When to use:**
- Production benchmarking with full metrics
- Evaluate both correctness (template) and quality (rubric)
- Comprehensive model assessment

### Mode 3: rubric_only

Evaluate rubric criteria only, skip template verification. Useful for qualitative assessment.

```python
config = VerificationConfig(
    answering_models=[model_config],
    parsing_models=[judge_config],
    evaluation_mode="rubric_only",
    rubric_enabled=True  # Must be True
)

results = benchmark.run_verification(config)

# Results include:
# - template_verification_performed = False
# - verify_result = None
# - rubric_evaluation_performed = True
# - verify_rubric = {"Clarity": 3, "Conciseness": 4, ...}
```

**When to use:**
- Qualitative evaluation without structured output requirements
- Rubric development and tuning
- Open-ended response evaluation
- Focus on content quality over format

### Choosing an Evaluation Mode

| Need | Recommended Mode | Why |
|------|-----------------|-----|
| Test template parsing | `template_only` | Fastest, focuses on structure |
| Production benchmarking | `template_and_rubric` | Complete evaluation |
| Rubric development | `rubric_only` | Skip template overhead |
| Open-ended responses | `rubric_only` | No structure requirements |
| Quality + Correctness | `template_and_rubric` | Both metrics |

---

## Advanced Configuration Options

### Enable Abstention Detection

Detect when models refuse to answer questions:

```python
config = VerificationConfig(
    answering_models=[model_config],
    parsing_models=[judge_config],
    abstention_enabled=True  # Detect refusals
)

results = benchmark.run_verification(config)

# Check abstention status
for result_id, result in results.items():
    if result.abstention_detected:
        print(f"Question {result.question_id}: Model refused to answer")
        print(f"  Reasoning: {result.abstention_reasoning}")
```

### Enable Deep Judgment

Extract detailed feedback with verbatim excerpts and reasoning:

```python
config = VerificationConfig(
    answering_models=[model_config],
    parsing_models=[judge_config],
    deep_judgment_enabled=True,
    deep_judgment_max_excerpts_per_attribute=3,
    deep_judgment_fuzzy_match_threshold=0.80
)

results = benchmark.run_verification(config)

# Access deep judgment data
for result_id, result in results.items():
    if result.parsed_response:
        print(f"Question: {result.question_id}")
        print(f"Excerpts: {result.parsed_response.get('excerpts', [])}")
        print(f"Reasoning: {result.parsed_response.get('reasoning', '')}")
```

See [Deep Judgment documentation](../advanced/deep-judgment.md) for comprehensive guide.

### Add System Prompts

Customize model behavior with system prompts:

```python
# Answering model with domain expertise
answering_model = ModelConfig(
    id="gpt-genomics",
    model_provider="openai",
    model_name="gpt-4.1-mini",
    temperature=0.7,
    interface="langchain",
    system_prompt="You are an expert in genomics and molecular biology. Answer concisely with precise scientific terminology."
)

# Judge model with strict evaluation
judge_model = ModelConfig(
    id="gpt-judge-strict",
    model_provider="openai",
    model_name="gpt-4.1-mini",
    temperature=0.0,
    interface="langchain",
    system_prompt="You are a strict evaluator. Parse responses carefully and extract only explicitly stated information."
)

config = VerificationConfig(
    answering_models=[answering_model],
    parsing_models=[judge_model]
)
```

### Configure Temperature

Control randomness and creativity:

```python
# High temperature: More creative, less consistent
creative_model = ModelConfig(
    id="gpt-creative",
    model_provider="openai",
    model_name="gpt-4.1-mini",
    temperature=0.9,  # More randomness
    interface="langchain"
)

# Zero temperature: Deterministic, consistent
deterministic_model = ModelConfig(
    id="gpt-deterministic",
    model_provider="openai",
    model_name="gpt-4.1-mini",
    temperature=0.0,  # No randomness
    interface="langchain"
)

# Compare creativity vs consistency
config = VerificationConfig(
    answering_models=[creative_model, deterministic_model],
    parsing_models=[judge_model],
    replicate_count=5  # See variance with replication
)
```

**Temperature Guidelines:**
- **0.0**: Deterministic, always returns same answer (best for factual questions)
- **0.3-0.5**: Slight variation, mostly consistent (good balance)
- **0.7-0.9**: Creative, more diverse responses (good for open-ended questions)
- **1.0+**: Very random, unpredictable (rarely useful for benchmarking)

---

## Using Different LLM Interfaces

Karenina supports four interface types for connecting to different LLM providers.

### Interface 1: langchain (OpenAI, Google, Anthropic)

Use LangChain for major cloud providers:

```python
# OpenAI via LangChain
openai_model = ModelConfig(
    id="gpt-4.1-mini",
    model_provider="openai",
    model_name="gpt-4.1-mini",
    interface="langchain"
)

# Anthropic via LangChain
anthropic_model = ModelConfig(
    id="claude-sonnet",
    model_provider="anthropic",
    model_name="claude-sonnet-4.5",
    interface="langchain"
)

# Google via LangChain
google_model = ModelConfig(
    id="gemini-pro",
    model_provider="google",
    model_name="gemini-2.5-flash",
    interface="langchain"
)
```

**Environment Variables Required:**
```bash
export OPENAI_API_KEY="your-openai-key"
export ANTHROPIC_API_KEY="your-anthropic-key"
export GOOGLE_API_KEY="your-google-key"
```

### Interface 2: openrouter

Use OpenRouter to access 200+ models through a single API:

```python
openrouter_model = ModelConfig(
    id="sonnet-via-openrouter",
    model_provider="openrouter",
    model_name="anthropic/claude-sonnet-4.5",
    interface="openrouter"
)
```

**Environment Variable Required:**
```bash
export OPENROUTER_API_KEY="your-openrouter-key"
```

### Interface 3: openai_endpoint (Local Models, Ollama)

Connect to OpenAI-compatible endpoints like Ollama, vLLM, or local servers:

```python
local_model = ModelConfig(
    id="llama-local",
    model_name="llama3.1:70b",
    interface="openai_endpoint",
    endpoint_base_url="http://localhost:11434/v1",
    endpoint_api_key="ollama"  # Ollama doesn't require real key
)
```

**Use Cases:**
- Local models via Ollama
- Self-hosted LLMs
- Custom API endpoints
- Private deployments

### Interface 4: manual (Testing without API Calls)

Use pre-recorded traces for testing and debugging:

```python
manual_model = ModelConfig(
    id="manual-traces",
    interface="manual"
)

# Upload manual traces via GUI or API
# Useful for testing evaluation logic without API costs
```

---

## Accessing Verification Results

### Result Structure

Each verification result contains comprehensive data:

```python
results = benchmark.run_verification(config)

# Results is a dict: {result_id: VerificationResult}
for result_id, result in results.items():
    # Identification
    print(f"Result ID: {result_id}")
    print(f"Question ID: {result.question_id}")
    print(f"Answering Model: {result.answering_model_id}")
    print(f"Parsing Model: {result.parsing_model_id}")
    print(f"Replicate: {result.replicate}")

    # Raw response
    print(f"Raw Answer: {result.raw_response}")

    # Template verification
    print(f"Template Passed: {result.verify_result}")
    print(f"Parsed Response: {result.parsed_response}")

    # Rubric evaluation (if enabled)
    if result.rubric_evaluation_performed:
        print(f"Rubric Scores: {result.verify_rubric}")

    # Abstention (if enabled)
    if result.abstention_detected:
        print(f"Abstention: Model refused to answer")
        print(f"Reasoning: {result.abstention_reasoning}")

    # Timestamps
    print(f"Timestamp: {result.timestamp}")
    print("-" * 50)
```

### Filter Results

```python
# Get only passing results
passing_results = {
    rid: r for rid, r in results.items()
    if r.verify_result
}

# Get results for specific model
gpt_results = {
    rid: r for rid, r in results.items()
    if r.answering_model_id == "gpt-4.1-mini"
}

# Get results for specific question
question_results = {
    rid: r for rid, r in results.items()
    if r.question_id == specific_question_id
}
```

### Compute Aggregate Metrics

```python
def compute_metrics(results):
    """Compute aggregate metrics from verification results."""
    total = len(results)
    if total == 0:
        return None

    # Template metrics
    template_passed = sum(1 for r in results.values() if r.verify_result)
    template_accuracy = template_passed / total

    # Rubric metrics (if available)
    rubric_scores = {}
    for result in results.values():
        if result.rubric_evaluation_performed and result.verify_rubric:
            for trait_name, score in result.verify_rubric.items():
                if trait_name not in rubric_scores:
                    rubric_scores[trait_name] = []
                rubric_scores[trait_name].append(score)

    rubric_averages = {
        trait: sum(scores) / len(scores)
        for trait, scores in rubric_scores.items()
    }

    # Abstention rate (if enabled)
    abstentions = sum(1 for r in results.values() if r.abstention_detected)
    abstention_rate = abstentions / total

    return {
        "total_questions": total,
        "template_accuracy": template_accuracy,
        "rubric_averages": rubric_averages,
        "abstention_rate": abstention_rate
    }

# Compute metrics
metrics = compute_metrics(results)
print(f"Template Accuracy: {metrics['template_accuracy']:.1%}")
print(f"Rubric Averages: {metrics['rubric_averages']}")
print(f"Abstention Rate: {metrics['abstention_rate']:.1%}")
```

---

## Progress Tracking

### Real-Time Progress Callback

Monitor verification progress with a callback function:

```python
def progress_callback(progress: float, message: str):
    """Called periodically during verification."""
    print(f"Progress: {progress:.1%} - {message}")

results = benchmark.run_verification(
    config=config,
    progress_callback=progress_callback
)
```

**Output:**
```
Progress: 0.0% - Starting verification...
Progress: 10.0% - Completed question 1/10
Progress: 20.0% - Completed question 2/10
...
Progress: 100.0% - Verification complete
```

### Batch Progress

For large benchmarks, track batch execution:

```python
from karenina.benchmark.verification import run_verification_batch

# Convert questions to templates format
templates = [
    benchmark.get_finished_template(qid)
    for qid in benchmark.questions.keys()
]

# Progress tracking
total_tasks = len(templates) * len(config.answering_models) * len(config.parsing_models) * config.replicate_count
completed_tasks = 0

def batch_progress_callback(current: int, total: int, result):
    global completed_tasks
    completed_tasks = current
    progress = current / total
    print(f"Progress: {progress:.1%} ({current}/{total} tasks)")

# Run with progress tracking
results = run_verification_batch(
    templates=templates,
    config=config,
    progress_callback=batch_progress_callback
)

print(f"Completed {completed_tasks}/{total_tasks} tasks")
```

---

## Answer Caching

Karenina automatically caches answer generation to improve efficiency when multiple judge models evaluate the same answering model response.

### How Answer Caching Works

**Without Caching:**
```
1 question × 1 answering model × 3 judge models = 3 answer generations
- Generate answer with Judge 1 → Parse with Judge 1
- Generate answer with Judge 2 → Parse with Judge 2
- Generate answer with Judge 3 → Parse with Judge 3
Result: Same answer generated 3 times (wasteful, potentially inconsistent)
```

**With Caching (Automatic):**
```
1 question × 1 answering model × 3 judge models = 1 answer generation
- Generate answer ONCE
- Parse with Judge 1 (using cached answer)
- Parse with Judge 2 (using cached answer)
- Parse with Judge 3 (using cached answer)
Result: Same answer reused 3 times (efficient, guaranteed consistent)
```

### Benefits

1. **Efficiency**: Reduces LLM API calls and costs (generate once, evaluate many times)
2. **Correctness**: Ensures all judges evaluate the exact same answer (important for fair comparison)
3. **Speed**: Faster verification by avoiding redundant answer generation

### Cache Behavior

The answer cache is:
- **Automatic**: No configuration required, works transparently
- **Thread-Safe**: Safe for parallel execution
- **Per-Question**: Cache key includes question ID, answering model ID, and replicate number
- **Replicate-Aware**: Each replicate gets independent answer generation

**Cache Key Format:** `{question_id}_{answering_model_id}_{replicate}`

### Caching with Replication

Each replicate run generates its own answer independently:

```python
config = VerificationConfig(
    answering_models=[model_config],
    parsing_models=[judge1, judge2, judge3],  # 3 judges
    replicate_count=2  # 2 replicates
)

# Total combinations: 1 question × 1 answering model × 3 judges × 2 replicates = 6 results
# Answer generations: 1 question × 1 answering model × 2 replicates = 2 generations
# Cache hits: 6 results - 2 generations = 4 cache reuses
```

**Result:**
- Replicate 0: Answer generated once, reused by all 3 judges
- Replicate 1: New answer generated once, reused by all 3 judges

---

## Complete Example

Here's a complete end-to-end example demonstrating multi-model verification with replication:

```python
from karenina import Benchmark
from karenina.schemas import VerificationConfig, ModelConfig

# Load benchmark with templates and rubrics already configured
benchmark = Benchmark.load("genomics_benchmark.jsonld")

# Configure three answering models for comparison
answering_models = [
    ModelConfig(
        id="gpt-4.1-mini",
        model_provider="openai",
        model_name="gpt-4.1-mini",
        temperature=0.7,
        interface="langchain",
        system_prompt="You are a genomics expert. Answer concisely."
    ),
    ModelConfig(
        id="claude-sonnet",
        model_provider="anthropic",
        model_name="claude-sonnet-4.5",
        temperature=0.7,
        interface="langchain",
        system_prompt="You are a genomics expert. Answer concisely."
    ),
    ModelConfig(
        id="gemini-flash",
        model_provider="google",
        model_name="gemini-2.5-flash",
        temperature=0.7,
        interface="langchain",
        system_prompt="You are a genomics expert. Answer concisely."
    )
]

# Configure single judge model for consistent evaluation
judge_model = ModelConfig(
    id="gpt-judge",
    model_provider="openai",
    model_name="gpt-4.1-mini",
    temperature=0.0,  # Deterministic parsing
    interface="langchain",
    system_prompt="You are a strict evaluator. Parse carefully."
)

# Configure verification with replication
config = VerificationConfig(
    answering_models=answering_models,
    parsing_models=[judge_model],
    evaluation_mode="template_and_rubric",
    rubric_enabled=True,
    replicate_count=3,  # Run each combination 3 times
    abstention_enabled=True
)

# Progress callback
def show_progress(progress: float, message: str):
    print(f"[{progress:.0%}] {message}")

# Run verification
print("Starting verification...")
results = benchmark.run_verification(
    config=config,
    progress_callback=show_progress
)

print(f"\nVerification complete: {len(results)} results generated")

# Analyze results by model
from collections import defaultdict

results_by_model = defaultdict(list)
for result in results.values():
    results_by_model[result.answering_model_id].append(result)

print("\n=== Results by Answering Model ===")
for model_id, model_results in results_by_model.items():
    passed = sum(1 for r in model_results if r.verify_result)
    total = len(model_results)
    accuracy = passed / total

    print(f"\n{model_id}:")
    print(f"  Template Accuracy: {accuracy:.1%} ({passed}/{total})")

    # Rubric averages
    rubric_scores = defaultdict(list)
    for r in model_results:
        if r.rubric_evaluation_performed and r.verify_rubric:
            for trait, score in r.verify_rubric.items():
                rubric_scores[trait].append(score)

    print(f"  Rubric Averages:")
    for trait, scores in rubric_scores.items():
        avg = sum(scores) / len(scores)
        print(f"    {trait}: {avg:.2f}")

    # Abstention rate
    abstentions = sum(1 for r in model_results if r.abstention_detected)
    abstention_rate = abstentions / total
    print(f"  Abstention Rate: {abstention_rate:.1%}")

# Save benchmark with results
benchmark.save("genomics_benchmark_verified.jsonld")
print("\n✓ Benchmark saved with verification results")
```

---

## Next Steps

After running verification:

- [Analyze Results](saving-loading.md#exporting-verification-results) - Export to CSV/JSON for deeper analysis
- [Save Benchmark](saving-loading.md) - Persist results to database or checkpoint
- [Advanced Features](../advanced/deep-judgment.md) - Use deep-judgment for detailed feedback
- [Few-Shot Prompting](../advanced/few-shot.md) - Guide responses with examples

---

## Related Documentation

- [Model Configuration](model-configuration.md) - Comprehensive guide to ModelConfig parameters and extra_kwargs
- [Defining Benchmarks](defining-benchmark.md) - Creating and configuring benchmarks
- [Templates](templates.md) - Structured answer evaluation
- [Rubrics](rubrics.md) - Qualitative assessment criteria
- [Saving & Loading](saving-loading.md) - Checkpoints, database, and export
- [Deep Judgment](../advanced/deep-judgment.md) - Extract detailed feedback with excerpts
- [Abstention Detection](../advanced/abstention-detection.md) - Handle model refusals
- [Few-Shot Prompting](../advanced/few-shot.md) - Guide responses with examples
