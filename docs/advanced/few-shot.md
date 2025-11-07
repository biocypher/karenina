# Few-Shot Prompting

Few-shot prompting is a technique where example question-answer pairs are provided to the LLM before asking the main question, helping guide responses toward expected formats, styles, and content.

## What is Few-Shot Prompting?

**Few-shot prompting** provides the LLM with examples of the task before asking it to perform the same task. For example:

```
Question: What is the approved drug target of Venetoclax?
Answer: BCL2

Question: How many chromosomes are in a human somatic cell?
Answer: 46

Question: How many protein subunits does hemoglobin A have?
Answer: [Model will answer here]
```

This technique can significantly improve:

- **Response quality**: Models learn from good examples
- **Consistency**: Responses follow demonstrated patterns
- **Format adherence**: Models match example structure
- **Accuracy**: Examples clarify expectations

## Why Use Few-Shot Prompting?

### 1. Improve Answer Quality

Models perform better when shown examples:

```python
# Without few-shot: Verbose answer
"Hemoglobin A is a tetrameric protein consisting of two alpha and two beta subunits..."

# With few-shot: Concise answer (like examples)
"4"
```

### 2. Enforce Formatting

Guide models to specific answer formats:

```python
# Examples show concise numerical answers
few_shot_examples = [
    {"question": "How many chromosomes...", "answer": "46"},
    {"question": "How many subunits...", "answer": "4"},
]

# Model learns to give brief numerical answers
```

### 3. Demonstrate Style

Show models the desired response style:

```python
# Examples show technical nomenclature
few_shot_examples = [
    {"question": "What is the target of Venetoclax?", "answer": "BCL2"},
    {"question": "What does TP53 encode?", "answer": "tumor protein p53"},
]

# Model learns to use standard nomenclature
```

## Basic Configuration

### Simple K-Shot Mode

Use the same number of examples for all questions:

```python
from karenina import Benchmark
from karenina.schemas import VerificationConfig, ModelConfig, FewShotConfig

# Create few-shot config with k=3 (use 3 examples per question)
few_shot_config = FewShotConfig(
    enabled=True,
    global_mode="k-shot",
    global_k=3
)

# Create verification config with few-shot
model_config = ModelConfig(
    id="gpt-4.1-mini",
    model_provider="openai",
    model_name="gpt-4.1-mini",
    temperature=0.0,
    interface="langchain"
)

config = VerificationConfig(
    answering_models=[model_config],
    parsing_models=[model_config],
    few_shot_config=few_shot_config
)

# Run verification - will use 3 examples per question
results = benchmark.run_verification(config)
```

### Use All Available Examples

Use every available example for each question:

```python
# Configure to use all examples
few_shot_config = FewShotConfig(
    enabled=True,
    global_mode="all"
)

config = VerificationConfig(
    answering_models=[model_config],
    parsing_models=[model_config],
    few_shot_config=few_shot_config
)
```

**When to use:** Maximum context, small number of high-quality examples.

## Adding Examples to Questions

### When Creating Questions

Add few-shot examples when creating questions:

```python
from karenina import Benchmark

benchmark = Benchmark.create(name="Genomics Benchmark")

# Add question with few-shot examples
benchmark.add_question(
    question="What is the approved drug target of Venetoclax?",
    raw_answer="BCL2",
    few_shot_examples=[
        {
            "question": "What is the approved drug target of Imatinib?",
            "answer": "BCR-ABL tyrosine kinase"
        },
        {
            "question": "What is the approved drug target of Trastuzumab?",
            "answer": "HER2"
        },
        {
            "question": "What is the approved drug target of Rituximab?",
            "answer": "CD20"
        }
    ]
)
```

### Adding Examples Later

Add examples to existing questions:

```python
# Load benchmark
benchmark = Benchmark.load("genomics_benchmark.jsonld")

# Get question
question_id = list(benchmark.questions.keys())[0]
question = benchmark.get_question(question_id)

# Add few-shot examples
question.few_shot_examples = [
    {"question": "How many autosomal chromosome pairs...", "answer": "22"},
    {"question": "How many sex chromosomes...", "answer": "2"},
]

# Save updated benchmark
benchmark.save("genomics_benchmark.jsonld")
```

## Complete Example

Here's an end-to-end workflow using few-shot prompting with a genomics benchmark:

```python
from karenina import Benchmark
from karenina.schemas import VerificationConfig, ModelConfig, FewShotConfig
from pathlib import Path

# ============================================================
# STEP 1: Create benchmark with genomics questions
# ============================================================

benchmark = Benchmark.create(
    name="Genomics Knowledge Benchmark",
    description="Testing LLM knowledge of genomics with few-shot prompting",
    version="1.0.0"
)

# ============================================================
# STEP 2: Add questions with few-shot examples
# ============================================================

# Question 1: Drug target with similar drug examples
benchmark.add_question(
    question="What is the approved drug target of Venetoclax?",
    raw_answer="BCL2",
    author={"name": "Pharma Curator"},
    few_shot_examples=[
        {"question": "What is the approved drug target of Imatinib?",
         "answer": "BCR-ABL tyrosine kinase"},
        {"question": "What is the approved drug target of Trastuzumab?",
         "answer": "HER2"},
    ]
)

# Question 2: Numerical answer with similar numerical examples
benchmark.add_question(
    question="How many chromosomes are in a human somatic cell?",
    raw_answer="46",
    author={"name": "Genetics Curator"},
    few_shot_examples=[
        {"question": "How many autosomal chromosome pairs are in humans?",
         "answer": "22"},
        {"question": "How many sex chromosomes are in humans?",
         "answer": "2"},
    ]
)

# Question 3: Protein structure with similar structure examples
benchmark.add_question(
    question="How many protein subunits does hemoglobin A have?",
    raw_answer="4",
    author={"name": "Biochemistry Curator"},
    few_shot_examples=[
        {"question": "How many subunits does RNA polymerase have?",
         "answer": "5"},
        {"question": "How many catalytic subunits does DNA polymerase III have?",
         "answer": "3"},
    ]
)

# ============================================================
# STEP 3: Generate templates
# ============================================================

model_config = ModelConfig(
    id="gpt-4.1-mini",
    model_provider="openai",
    model_name="gpt-4.1-mini",
    temperature=0.0,
    interface="langchain"
)

print("Generating templates...")
benchmark.generate_all_templates(model_config=model_config)
print("✓ Templates generated")

# ============================================================
# STEP 4: Configure few-shot prompting
# ============================================================

# Option A: Use k-shot mode (same number of examples per question)
few_shot_config = FewShotConfig(
    enabled=True,
    global_mode="k-shot",
    global_k=2  # Use 2 examples per question
)

# Option B: Use all available examples
# few_shot_config = FewShotConfig(
#     enabled=True,
#     global_mode="all"
# )

# ============================================================
# STEP 5: Run verification with few-shot
# ============================================================

print("\nRunning verification with few-shot prompting...")
config = VerificationConfig(
    answering_models=[model_config],
    parsing_models=[model_config],
    few_shot_config=few_shot_config
)

results = benchmark.run_verification(config)
print(f"✓ Verification complete: {len(results)} questions")

# ============================================================
# STEP 6: Analyze results
# ============================================================

passed = sum(1 for r in results.values() if r.verify_result)
print(f"Pass rate: {passed}/{len(results)} ({passed/len(results)*100:.1f}%)")

# Show individual results
for question_id, result in results.items():
    question = benchmark.get_question(question_id)
    print(f"\nQuestion: {question.question}")
    print(f"  Expected: {question.raw_answer}")
    print(f"  Model answer: {result.raw_llm_response[:100]}...")
    print(f"  Correct: {'✓' if result.verify_result else '✗'}")

# Save benchmark with results
benchmark.save("genomics_benchmark_few_shot.jsonld")
print("\n✓ Benchmark saved")
```

**Example Output:**

```
Generating templates...
✓ Templates generated

Running verification with few-shot prompting...
✓ Verification complete: 3 questions
Pass rate: 3/3 (100.0%)

Question: What is the approved drug target of Venetoclax?
  Expected: BCL2
  Model answer: BCL2
  Correct: ✓

Question: How many chromosomes are in a human somatic cell?
  Expected: 46
  Model answer: 46
  Correct: ✓

Question: How many protein subunits does hemoglobin A have?
  Expected: 4
  Model answer: 4
  Correct: ✓

✓ Benchmark saved
```

## Advanced Configurations

### Different K Values Per Question

Use different numbers of examples for different questions:

```python
# Configure different k values per question
few_shot_config = FewShotConfig.k_shot_for_questions(
    question_k_mapping={
        "question_id_1": 5,  # Use 5 examples for complex question
        "question_id_2": 2,  # Use 2 examples for simple question
        "question_id_3": 3,  # Use 3 examples
    },
    global_k=3  # Fallback for questions not in mapping
)

config = VerificationConfig(
    answering_models=[model_config],
    parsing_models=[model_config],
    few_shot_config=few_shot_config
)
```

**When to use:** Questions have varying complexity levels.

### Custom Example Selection by Index

Manually select specific examples by their position:

```python
# Get question IDs
question_ids = list(benchmark.questions.keys())

# Select specific examples by index (0-based)
few_shot_config = FewShotConfig.from_index_selections({
    question_ids[0]: [0, 1],     # Use first 2 examples
    question_ids[1]: [0, 2],     # Use 1st and 3rd examples
    question_ids[2]: [1, 2, 3],  # Use 2nd, 3rd, and 4th examples
})

config = VerificationConfig(
    answering_models=[model_config],
    parsing_models=[model_config],
    few_shot_config=few_shot_config
)
```

**When to use:** Fine-grained control over which examples are used.

### Adding External Examples

Add examples that aren't from the question's available pool:

```python
# Create config with global external examples
few_shot_config = FewShotConfig(
    enabled=True,
    global_mode="k-shot",
    global_k=2,
    global_external_examples=[
        {
            "question": "What is the molecular weight of glucose?",
            "answer": "180.16 g/mol"
        },
        {
            "question": "What is the pH of neutral water?",
            "answer": "7.0"
        }
    ]
)

config = VerificationConfig(
    answering_models=[model_config],
    parsing_models=[model_config],
    few_shot_config=few_shot_config
)
```

**When to use:** Want to include domain-specific high-quality examples for all questions.

## Modes Overview

### "k-shot" Mode

Use the first k examples for each question:

```python
few_shot_config = FewShotConfig(
    global_mode="k-shot",
    global_k=3  # Use 3 examples
)
```

**Best for:** Consistent number of examples across all questions.

### "all" Mode

Use all available examples for each question:

```python
few_shot_config = FewShotConfig(
    global_mode="all"
)
```

**Best for:** Small number of high-quality examples, maximum context.

### "custom" Mode

Manually select specific examples:

```python
few_shot_config = FewShotConfig.from_index_selections({
    "question_1": [0, 2, 4],  # Select by index
})
```

**Best for:** Fine-grained control, curated example selection.

### "none" Mode

Disable few-shot for specific questions:

```python
few_shot_config = FewShotConfig(
    global_mode="k-shot",
    global_k=3,
    question_configs={
        "special_question_id": QuestionFewShotConfig(mode="none")  # No examples
    }
)
```

**Best for:** Testing impact of few-shot on specific questions.

## Prompt Format

Few-shot prompts are constructed in a simple Q&A format:

```
Question: What is the approved drug target of Imatinib?
Answer: BCR-ABL tyrosine kinase

Question: What is the approved drug target of Trastuzumab?
Answer: HER2

Question: What is the approved drug target of Venetoclax?
Answer: [Model generates answer here]
```

The LLM sees the examples before answering, learning from their format and content.

## When to Use Few-Shot

### ✅ Use Few-Shot When:

- **Enforcing formats**: Need specific answer structure (numerical, gene symbols, etc.)
- **Improving conciseness**: Models tend to be verbose, examples show brevity
- **Demonstrating style**: Want technical nomenclature or specific terminology
- **Complex tasks**: Task benefits from seeing examples
- **Consistency matters**: Need similar answers across similar questions

### ❌ Don't Use Few-Shot When:

- **Simple tasks**: Model already performs well without examples
- **Token limits**: Using large models with limited context windows
- **No good examples**: Don't have high-quality representative examples
- **Testing baselines**: Measuring model performance without assistance
- **Fast iteration**: Adding complexity during initial testing

## Best Practices

### 1. Start with K-Shot Mode

Begin with k-shot before moving to custom selection:

```python
# Start simple
few_shot_config = FewShotConfig(global_mode="k-shot", global_k=3)

# Iterate to custom if needed
```

### 2. Use 2-3 Examples

More examples aren't always better. Start small:

```python
# Good starting point
few_shot_config = FewShotConfig(global_mode="k-shot", global_k=2)

# Can increase if needed
few_shot_config = FewShotConfig(global_mode="k-shot", global_k=5)
```

**Why:** Diminishing returns after 3-5 examples, increased token costs.

### 3. Choose Representative Examples

Select examples that represent the task well:

```python
# Good: Similar domain, clear answers
few_shot_examples = [
    {"question": "What is the target of Venetoclax?", "answer": "BCL2"},
    {"question": "What is the target of Imatinib?", "answer": "BCR-ABL"},
]

# Bad: Unrelated domain
few_shot_examples = [
    {"question": "What is 2+2?", "answer": "4"},
    {"question": "What color is the sky?", "answer": "Blue"},
]
```

### 4. Match Example Format to Expected Answers

Examples should match the format you want:

```python
# For concise numerical answers
few_shot_examples = [
    {"question": "How many chromosomes...", "answer": "46"},
    {"question": "How many subunits...", "answer": "4"},
]

# For detailed technical answers
few_shot_examples = [
    {"question": "Describe the structure...",
     "answer": "Hemoglobin is a tetrameric protein consisting of..."},
]
```

### 5. Test With and Without Few-Shot

Measure the impact of few-shot prompting:

```python
# Baseline (no few-shot)
config_baseline = VerificationConfig(
    answering_models=[model_config],
    parsing_models=[model_config]
)
results_baseline = benchmark.run_verification(config_baseline)

# With few-shot
config_few_shot = VerificationConfig(
    answering_models=[model_config],
    parsing_models=[model_config],
    few_shot_config=FewShotConfig(global_mode="k-shot", global_k=3)
)
results_few_shot = benchmark.run_verification(config_few_shot)

# Compare pass rates
baseline_pass = sum(1 for r in results_baseline.values() if r.verify_result)
few_shot_pass = sum(1 for r in results_few_shot.values() if r.verify_result)

print(f"Baseline: {baseline_pass}/{len(results_baseline)}")
print(f"Few-shot: {few_shot_pass}/{len(results_few_shot)}")
```

### 6. Monitor Token Usage

More examples consume more tokens:

- Each example: ~50-200 tokens (depending on length)
- 3 examples: ~150-600 tokens
- 10 examples: ~500-2000 tokens

**Watch for:** Context window limits, increased API costs.

### 7. Use External Examples Sparingly

Only add external examples when necessary:

```python
# Good: Add domain-specific high-quality examples
few_shot_config = FewShotConfig(
    global_external_examples=[
        {"question": "High-quality domain example", "answer": "Perfect answer"}
    ]
)

# Bad: Too many unrelated examples
few_shot_config = FewShotConfig(
    global_external_examples=[...50 examples...]  # Too many!
)
```

## Troubleshooting

### Issue 1: Examples Not Being Used

**Symptom**: Few-shot enabled but no improvement in results.

**Solutions:**

1. **Verify few-shot is enabled**:
   ```python
   assert config.is_few_shot_enabled()
   ```

2. **Check questions have examples**:
   ```python
   for qid in benchmark.questions:
       question = benchmark.get_question(qid)
       if not question.few_shot_examples:
           print(f"Question {qid} has no examples")
   ```

3. **Check mode isn't "none"**:
   ```python
   assert few_shot_config.global_mode != "none"
   ```

### Issue 2: Too Many Examples

**Symptom**: LLM context limit exceeded, slow responses.

**Solutions:**

1. **Reduce k value**:
   ```python
   few_shot_config = FewShotConfig(global_mode="k-shot", global_k=2)
   ```

2. **Switch to custom selection**:
   ```python
   few_shot_config = FewShotConfig.from_index_selections({
       "question_1": [0, 1],  # Only 2 examples
   })
   ```

### Issue 3: Poor Example Quality

**Symptom**: Few-shot makes results worse.

**Solutions:**

1. **Review example quality**:
   ```python
   for example in question.few_shot_examples:
       print(f"Q: {example['question']}")
       print(f"A: {example['answer']}\n")
   ```

2. **Use custom selection to pick better examples**:
   ```python
   # Select only the best examples
   few_shot_config = FewShotConfig.from_index_selections({
       "question_1": [0, 2],  # Skip poor example at index 1
   })
   ```

3. **Add external high-quality examples**:
   ```python
   few_shot_config = FewShotConfig(
       global_external_examples=[
           {"question": "High-quality example", "answer": "Perfect answer"}
       ]
   )
   ```

### Issue 4: Inconsistent Results

**Symptom**: Results vary between runs.

**Solutions:**

1. **Set temperature to 0**:
   ```python
   model_config = ModelConfig(
       id="gpt-4.1-mini",
       model_provider="openai",
       model_name="gpt-4.1-mini",
       temperature=0.0,  # Deterministic
       interface="langchain"
   )
   ```

2. **Use deterministic example selection**:

   - K-shot mode automatically uses question ID as seed for reproducibility
   - Results should be consistent across runs

## Performance Considerations

### Token Usage

Few-shot prompting increases token consumption:

| Examples | Estimated Tokens (Input) | Cost Impact |
|----------|-------------------------|-------------|
| 0 (no few-shot) | Baseline | Baseline |
| 2 examples | +100-400 tokens | +5-10% |
| 5 examples | +250-1000 tokens | +10-20% |
| 10 examples | +500-2000 tokens | +20-40% |

### Latency

More examples slightly increase latency:

- Token generation time: ~50-100ms per 100 tokens
- 3 examples: +50-200ms additional latency

**Recommendation:** Start with k=2-3 to balance quality and cost.

## Next Steps

Once you have few-shot prompting configured, you can:

- [Verification](../using-karenina/verification.md) - Run verifications with few-shot
- [Presets](presets.md) - Save few-shot configurations in presets
- [Deep-Judgment](deep-judgment.md) - Combine with deep-judgment parsing
- [Templates](../using-karenina/templates.md) - Design templates that work with few-shot

## Related Documentation

- [Verification](../using-karenina/verification.md) - Core verification workflow
- [Adding Questions](../using-karenina/adding-questions.md) - How to add questions with examples
- [Presets](presets.md) - Save few-shot configurations
- [Templates](../using-karenina/templates.md) - Template creation
