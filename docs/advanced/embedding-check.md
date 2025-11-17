# Embedding Check (Semantic Fallback)

Embedding check provides a semantic fallback mechanism that can rescue verification failures when answers are semantically correct despite structural differences.

## What is Embedding Check?

**Embedding check** is an optional feature that uses sentence embeddings to detect semantically equivalent answers that fail strict template matching. When verification fails, this feature computes the semantic similarity between the expected answer and the model's response. If similarity exceeds a configurable threshold, an LLM validates semantic equivalence and can override the initial failure.

**Key benefits:**

- **Reduces false negatives**: Catches paraphrased but correct answers
- **Flexible evaluation**: Handles structural variations without changing templates
- **Semantic awareness**: Uses deep learning embeddings for meaning comparison
- **LLM validation**: Confirms equivalence with parsing model judgment
- **Zero overhead when disabled**: Only runs on failed verifications

## How Embedding Check Works

Embedding check activates **only when initial verification returns `False`**:

**1. Initial Verification Fails**

Template-based verification returns `False` due to structural mismatch.

**2. Compute Embedding Similarity**

Uses SentenceTransformer models to generate embeddings for both the expected answer and the model's response, then computes cosine similarity (0.0-1.0).

**3. Check Threshold**

If similarity score exceeds the configured threshold (default: 0.85), proceed to LLM validation.

**4. LLM Semantic Validation**

The parsing model evaluates whether the two answers are semantically equivalent, providing a yes/no judgment with reasoning.

**5. Override Result**

If the LLM confirms semantic equivalence, the verification result is overridden from `False` to `True`.

**Workflow Diagram:**

```
Initial Verification → False (template mismatch)
    ↓
Embedding Check Enabled?
    ↓ Yes
Compute Embedding Similarity
    ↓
Similarity: 0.87 (> threshold 0.85)
    ↓
Ask Parsing Model: "Are these semantically equivalent?"
    ↓
LLM Response: Yes
    ↓
Override Result: False → True ✓
```

## Common Use Cases

### Use Case 1: Paraphrased Answers

**Scenario**: LLM provides the correct answer with different wording.

**Example**:

- Expected Answer: "BCL2"
- Model Response: "The BCL-2 protein"

**Result**:

- Initial verification: `False` (different structure)
- Embedding similarity: `0.91`
- Semantic check: `True` (same protein mentioned)
- Final result: `True` (overridden) ✓

### Use Case 2: Numerical Format Differences

**Scenario**: Same number in different representations.

**Example**:

- Expected Answer: "46"
- Model Response: "Forty-six chromosomes"

**Result**:

- Initial verification: `False` (string "46" ≠ "Forty-six chromosomes")
- Embedding similarity: `0.88`
- Semantic check: `True` (same numerical value)
- Final result: `True` (overridden) ✓

### Use Case 3: Structural Variations

**Scenario**: Correct information in different structure.

**Example**:

- Expected Answer: "4"
- Model Response: "Hemoglobin A consists of four protein subunits"

**Result**:

- Initial verification: `False` (template expects just number)
- Embedding similarity: `0.86`
- Semantic check: `True` (correct count mentioned)
- Final result: `True` (overridden) ✓

## Enabling Embedding Check

Embedding check is **disabled by default**. Enable it using environment variables:

### Configuration

```python
import os

# Enable embedding check
os.environ["EMBEDDING_CHECK"] = "true"

# Specify embedding model (default: all-MiniLM-L6-v2)
os.environ["EMBEDDING_CHECK_MODEL"] = "all-MiniLM-L6-v2"

# Set similarity threshold 0.0-1.0 (default: 0.85)
os.environ["EMBEDDING_CHECK_THRESHOLD"] = "0.85"
```

### Supported Embedding Models

Any SentenceTransformer model is supported. Popular choices:

| Model | Speed | Accuracy | Use Case |
|-------|-------|----------|----------|
| `all-MiniLM-L6-v2` (default) | Fast | Good | General purpose, balanced |
| `all-mpnet-base-v2` | Slower | Better | Higher accuracy needed |
| `multi-qa-MiniLM-L6-cos-v1` | Fast | Good | Question-answering tasks |
| `paraphrase-multilingual-MiniLM-L12-v2` | Medium | Good | Multilingual support |
| `all-distilroberta-v1` | Fast | Medium | Fast inference |

## Complete Example

Here's an end-to-end workflow using embedding check with a genomics benchmark:

```python
import os
from karenina import Benchmark
from karenina.schemas import ModelConfig, VerificationConfig

# 1. Enable embedding check
os.environ["EMBEDDING_CHECK"] = "true"
os.environ["EMBEDDING_CHECK_MODEL"] = "all-MiniLM-L6-v2"
os.environ["EMBEDDING_CHECK_THRESHOLD"] = "0.85"

# 2. Create benchmark with genomics questions
benchmark = Benchmark.create(
    name="Genomics Knowledge Benchmark",
    description="Testing LLM knowledge of genomics and molecular biology",
    version="1.0.0"
)

# Add questions
benchmark.add_question(
    question="What is the approved drug target of Venetoclax?",
    raw_answer="BCL2",
    author={"name": "Pharma Curator"}
)

benchmark.add_question(
    question="How many chromosomes are in a human somatic cell?",
    raw_answer="46",
    author={"name": "Genetics Curator"}
)

benchmark.add_question(
    question="How many protein subunits does hemoglobin A have?",
    raw_answer="4",
    author={"name": "Biochemistry Curator"}
)

# 3. Generate templates
model_config = ModelConfig(
    id="gpt-4.1-mini",
    model_provider="openai",
    model_name="gpt-4.1-mini",
    temperature=0.0,
    interface="langchain"
)

print("Generating templates...")
benchmark.generate_all_templates(model_config=model_config)

# 4. Run verification with embedding check
print("Running verification...")
config = VerificationConfig(
    answering_models=[model_config],
    parsing_models=[model_config]
)

results = benchmark.run_verification(config)

# 5. Analyze embedding check results
print("\n=== Embedding Check Results ===")
override_count = 0

for result in results.results:
    if result.embedding_check_performed:
        print(f"\nQuestion: {result.question_text}")
        print(f"  Expected: {benchmark.get_question(question_id).raw_answer}")
        print(f"  Model Response: {result.raw_llm_response[:100]}...")
        print(f"  Similarity Score: {result.embedding_similarity_score:.4f}")
        print(f"  Model Used: {result.embedding_model_used}")
        print(f"  Override Applied: {result.embedding_override_applied}")

        if result.embedding_override_applied:
            override_count += 1
            print(f"  ✓ Verification overridden: False → True")

print(f"\nTotal overrides: {override_count}")
```

**Example Output:**

```
Generating templates...
✓ Generated 3 templates

Running verification...

=== Embedding Check Results ===

Question: What is the approved drug target of Venetoclax?
  Expected: BCL2
  Model Response: The BCL-2 protein is the approved target of Venetoclax...
  Similarity Score: 0.9123
  Model Used: all-MiniLM-L6-v2
  Override Applied: True
  ✓ Verification overridden: False → True

Question: How many chromosomes are in a human somatic cell?
  Expected: 46
  Model Response: Human somatic cells contain forty-six chromosomes...
  Similarity Score: 0.8801
  Model Used: all-MiniLM-L6-v2
  Override Applied: True
  ✓ Verification overridden: False → True

Total overrides: 2
```

## Understanding Results

### Result Metadata

When embedding check runs, results include additional metadata:

```python
result = results[question_id]

# Embedding check metadata
result.embedding_check_performed        # Was embedding check run?
result.embedding_similarity_score       # Cosine similarity (0.0-1.0)
result.embedding_override_applied       # Was result overridden?
result.embedding_model_used             # Model name
result.embedding_check_details          # Additional details
```

### Filtering for Overrides

Find all cases where embedding check rescued a failed verification:

```python
# Get all overridden results
overridden = [
    (qid, result)
    for result in results.results
    if result.embedding_override_applied
]

print(f"Found {len(overridden)} overridden verifications")

for qid, result in overridden:
    print(f"{qid}: similarity={result.embedding_similarity_score:.4f}")
```

### Computing Override Statistics

```python
# Calculate embedding check statistics
total_questions = len(results)
embedding_checks_performed = sum(
    1 for r in results.results if r.embedding_check_performed
)
overrides_applied = sum(
    1 for r in results.results if r.embedding_override_applied
)

print(f"Total questions: {total_questions}")
print(f"Embedding checks performed: {embedding_checks_performed}")
print(f"Overrides applied: {overrides_applied}")
print(f"Override rate: {overrides_applied / embedding_checks_performed * 100:.1f}%")
```

## Performance Considerations

### When Disabled

- **Zero overhead**: Feature not loaded or executed
- **No dependencies required**: sentence-transformers not needed

### When Enabled

Embedding check only runs on **failed verifications**, so the impact depends on your failure rate.

**Cost impact:**

Embedding check adds one additional LLM call (semantic validation) for each failed verification where similarity exceeds the threshold. This uses your configured parsing model.

## Tuning the Similarity Threshold

The similarity threshold (default: 0.85) controls when LLM validation is triggered.

### Threshold Guidelines

| Threshold | Behavior | Use Case |
|-----------|----------|----------|
| **0.80-0.85** | Moderate selectivity | General purpose, balanced |
| **0.85-0.90** (default) | Higher selectivity | Reduce false overrides |
| **0.90-0.95** | Very selective | Only very similar answers |
| **0.75-0.80** | Lower selectivity | Catch more paraphrases |

### Finding the Right Threshold

**Start with default (0.85):**

```python
os.environ["EMBEDDING_CHECK_THRESHOLD"] = "0.85"
```

**Too many false overrides?** → Increase threshold:

```python
os.environ["EMBEDDING_CHECK_THRESHOLD"] = "0.90"
```

**Missing valid paraphrases?** → Decrease threshold:

```python
os.environ["EMBEDDING_CHECK_THRESHOLD"] = "0.80"
```

### Threshold Experimentation

Test different thresholds on a sample:

```python
import os
from karenina.benchmark.verification.tools.embedding_check import (
    compute_embedding_similarity
)

# Test cases
test_cases = [
    ("BCL2", "The BCL-2 protein"),
    ("46", "Forty-six chromosomes"),
    ("4", "Four protein subunits"),
    ("hemoglobin", "haemoglobin"),  # Spelling variant
]

# Try different thresholds
thresholds = [0.75, 0.80, 0.85, 0.90, 0.95]

for expected, response in test_cases:
    similarity = compute_embedding_similarity(
        expected,
        response,
        model_name="all-MiniLM-L6-v2"
    )

    print(f"\nExpected: '{expected}'")
    print(f"Response: '{response}'")
    print(f"Similarity: {similarity:.4f}")

    for threshold in thresholds:
        would_trigger = "✓" if similarity >= threshold else "✗"
        print(f"  Threshold {threshold}: {would_trigger}")
```

## When to Use Embedding Check

### ✅ Use Embedding Check When:

- **Paraphrased answers are common**: Models often rephrase correct answers
- **Format flexibility needed**: Accept "46" and "forty-six" as equivalent
- **Reducing false negatives**: Minimize cases where correct answers are marked wrong
- **Testing creative models**: Models that elaborate or rephrase more frequently
- **Multi-language evaluation**: Detecting equivalent meanings across languages

### ❌ Don't Use Embedding Check When:

- **Strict format required**: Exact format is critical (e.g., gene symbols, IDs)
- **High precision needed**: False positives are more costly than false negatives
- **Templates handle variations**: Templates already account for expected variations
- **Performance is critical**: Cannot afford extra 500-2000ms per failed verification
- **Deterministic evaluation required**: Need reproducible results without LLM judgment

## Best Practices

### 1. Enable Selectively

Don't enable embedding check for all benchmarks. Use it when you know paraphrasing is common:

```python
# Good: Enable for natural language questions
os.environ["EMBEDDING_CHECK"] = "true"
benchmark_nl = Benchmark.load("natural_language_qa.jsonld")

# Good: Disable for strict format questions
os.environ["EMBEDDING_CHECK"] = "false"
benchmark_ids = Benchmark.load("gene_id_extraction.jsonld")
```

### 2. Monitor Override Rates

Track how often embedding check overrides results:

```python
override_rate = sum(
    1 for r in results.results if r.embedding_override_applied
) / len(results)

if override_rate > 0.20:  # More than 20% overrides
    print("Warning: High override rate. Consider:")
    print("  - Reviewing template definitions")
    print("  - Adjusting similarity threshold")
    print("  - Examining overridden cases manually")
```

### 3. Review Overridden Cases

Manually inspect overridden verifications to ensure quality:

```python
# Review all overridden cases
for result in results.results:
    if result.embedding_override_applied:
        question = benchmark.get_question(result.question_id)
        print(f"\n=== Overridden: {qid} ===")
        print(f"Question: {question.question}")
        print(f"Expected: {question.raw_answer}")
        print(f"Got: {result.raw_llm_response}")
        print(f"Similarity: {result.embedding_similarity_score:.4f}")

        # Manual validation
        is_correct = input("Is this override correct? (y/n): ")
        if is_correct.lower() != 'y':
            print("⚠ False override detected - consider higher threshold")
```

### 4. Choose the Right Model

**For most use cases:** Use default `all-MiniLM-L6-v2` (fast, good accuracy)

**For higher accuracy:** Use `all-mpnet-base-v2` (slower, better)

**For question-answering:** Use `multi-qa-MiniLM-L6-cos-v1` (optimized for Q&A)

```python
# High-accuracy configuration
os.environ["EMBEDDING_CHECK_MODEL"] = "all-mpnet-base-v2"
os.environ["EMBEDDING_CHECK_THRESHOLD"] = "0.90"  # Higher threshold with better model
```

### 5. Combine with Deep-Judgment

Embedding check works well with deep-judgment parsing for maximum transparency:

```python
# Enable both features
os.environ["EMBEDDING_CHECK"] = "true"

config = VerificationConfig(
    answering_models=[model_config],
    parsing_models=[model_config],
    deep_judgment_enabled=True  # Also enable deep-judgment
)

results = benchmark.run_verification(config)

# Review cases where both features triggered
for result in results.results:
    if result.embedding_override_applied and result.deep_judgment_used:
        print(f"\n{qid}: Both embedding check and deep-judgment used")
        print(f"  Deep-judgment excerpts: {result.deep_judgment_excerpts}")
        print(f"  Embedding similarity: {result.embedding_similarity_score:.4f}")
```

## Integration with Other Features

### Embedding Check + Deep-Judgment

Use embedding check to catch paraphrases, deep-judgment for transparency:

```python
os.environ["EMBEDDING_CHECK"] = "true"

config = VerificationConfig(
    answering_models=[model_config],
    parsing_models=[model_config],
    deep_judgment_enabled=True,
    deep_judgment_max_excerpts_per_attribute=3
)
```

### Embedding Check + Abstention Detection

Both features can run together. Abstention detection identifies refusals; embedding check handles paraphrases:

```python
os.environ["EMBEDDING_CHECK"] = "true"

config = VerificationConfig(
    answering_models=[model_config],
    parsing_models=[model_config],
    abstention_enabled=True
)

# Results may have both abstention and embedding metadata
for result in results.results:
    if result.abstention_detected:
        print(f"{qid}: Model abstained")
    elif result.embedding_override_applied:
        print(f"{qid}: Embedding check overrode failure")
```

## Troubleshooting

### Issue 1: Embedding Check Not Running

**Symptom**: `embedding_check_performed` is always `False`

**Solutions**:

1. Verify environment variable is set: `os.getenv("EMBEDDING_CHECK")`
2. Check that initial verification is failing (embedding check only runs on failures)
3. Ensure sentence-transformers is installed: `pip install sentence-transformers`

### Issue 2: No Overrides Applied

**Symptom**: Embedding checks run but never override results

**Solutions**:

1. Lower the similarity threshold: `os.environ["EMBEDDING_CHECK_THRESHOLD"] = "0.80"`
2. Review similarity scores to see if they're below threshold
3. Try a more accurate embedding model: `all-mpnet-base-v2`

### Issue 3: Too Many Overrides

**Symptom**: High override rate (>20%) suggesting false positives

**Solutions**:

1. Raise the similarity threshold: `os.environ["EMBEDDING_CHECK_THRESHOLD"] = "0.90"`
2. Review templates to ensure they're capturing expected variations
3. Manually inspect overridden cases to identify patterns

### Issue 4: Slow Performance

**Symptom**: Verification takes too long with embedding check enabled

**Solutions**:

1. Use faster embedding model: `all-MiniLM-L6-v2` or `all-distilroberta-v1`
2. Increase threshold to reduce LLM validation calls
3. Improve templates to reduce initial verification failures
4. Disable embedding check for benchmarks where it's not needed

## Next Steps

Once you have embedding check configured, you can:

- [Deep-Judgment Parsing](deep-judgment.md) - Multi-stage parsing with evidence extraction
- [Abstention Detection](abstention-detection.md) - Identify model refusals
- [Verification](../using-karenina/verification.md) - Complete verification workflow
- [Saving and Loading](../using-karenina/saving-loading.md) - Persist benchmarks

## Related Documentation

- [Verification](../using-karenina/verification.md) - Core verification workflow
- [Templates](../using-karenina/templates.md) - Answer template creation
- [Deep-Judgment](deep-judgment.md) - Multi-stage parsing
- [Abstention Detection](abstention-detection.md) - Refusal detection
