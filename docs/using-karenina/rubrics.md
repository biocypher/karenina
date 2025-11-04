# Rubrics

Rubrics provide qualitative evaluation criteria beyond the basic template verification. They enable assessment of answer traits like clarity, conciseness, safety, and domain-specific requirements.

## What Are Rubrics?

**Rubrics** are collections of evaluation traits that assess qualitative aspects of LLM responses:

- **Qualitative assessment** - Evaluate traits like clarity, completeness, and style
- **Supplement template verification** - Templates check factual correctness, rubrics assess quality
- **Multiple trait types** - LLM-based, regex-based, and metric-based evaluation
- **Flexible scope** - Apply globally to all questions or to specific questions only

Unlike templates which focus on extracting and verifying structured data, rubrics evaluate broader characteristics of responses that require judgment or pattern matching.

## Why Use Rubrics?

Rubrics are essential for comprehensive evaluation:

1. **Quality Beyond Correctness**: Assess traits like clarity and conciseness that aren't captured by factual verification
2. **Domain Validation**: Check for required terminology or concepts in specialized domains
3. **Safety and Compliance**: Ensure responses meet safety standards or avoid prohibited content
4. **Quantitative Metrics**: Measure classification accuracy with precision, recall, and F1 scores
5. **Consistent Standards**: Apply uniform evaluation criteria across question sets

## Rubric Scope: Global vs Question-Specific

Rubrics can be applied at two different scopes:

### Global Rubrics - Apply to ALL Questions

**Global rubrics** are evaluated for **every question** in your benchmark. Use global rubrics for traits that should be assessed universally.

**Best for:**
- General quality traits (clarity, conciseness, completeness)
- Safety requirements that apply to all responses
- Style guidelines that should be consistent throughout

**Example use case:** You want to ensure **all** answers in your genomics benchmark are clear and concise, regardless of the specific question.

### Question-Specific Rubrics - Apply to ONE Question

**Question-specific rubrics** are evaluated for **a single question only**. Use question-specific rubrics for domain validation or specialized requirements.

**Best for:**
- Domain-specific terminology checks
- Question-specific validation requirements
- Classification or categorization metrics

**Example use case:** You want to check that the answer to "What is the approved drug target of Venetoclax?" mentions BH3 proteins, but this check only makes sense for that particular question.

---

## Three Types of Rubric Traits

Karenina supports three types of evaluation traits:

### 1. LLM-Based Traits (`RubricTrait`)

AI-evaluated traits where the parsing model assesses answer quality.

**Evaluation Kinds:**
- **`score`**: 1-5 scale for nuanced assessment (1=Poor, 5=Excellent)
- **`binary`**: Pass/fail evaluation (true/false)

**Structure:**
```python
from karenina.schemas import RubricTrait

RubricTrait(
    name="Conciseness",
    description="Rate the conciseness of the answer on a scale from 1 (very verbose) to 5 (extremely concise).",
    kind="score"  # or "binary"
)
```

### 2. Regex-Based Traits (`ManualRubricTrait`)

Deterministic pattern matching for format validation and keyword checks.

**Structure:**
```python
from karenina.schemas import ManualRubricTrait

ManualRubricTrait(
    name="Contains BH3",
    description="Answer must mention BH3 proteins",
    pattern=r"\bBH3\b",  # Regex pattern
    case_sensitive=False,
    invert=False  # Set to True to fail if pattern matches
)
```

### 3. Metric-Based Traits (`MetricRubricTrait`)

Confusion matrix-based traits for quantitative classification evaluation.

**Available Metrics:**
- **Precision**: TP / (TP + FP)
- **Recall**: TP / (TP + FN)
- **F1 Score**: 2 × (Precision × Recall) / (Precision + Recall)
- **Accuracy**: (TP + TN) / (TP + TN + FP + FN)
- **Specificity**: TN / (TN + FP)

**Structure:**
```python
from karenina.schemas import MetricRubricTrait

MetricRubricTrait(
    name="Disease Classification",
    description="Evaluate accuracy of identifying inflammatory diseases",
    metrics=["precision", "recall", "f1"],
    tp_instructions=[
        "asthma",      # Should be identified as inflammatory
        "bronchitis",
        "pneumonia"
    ],
    fp_instructions=[
        "emphysema",            # Should NOT be identified as inflammatory
        "pulmonary fibrosis"
    ]
)
```

---

## Creating a Global Rubric

Global rubrics apply to **all questions** in your benchmark. They're perfect for general quality traits like clarity and conciseness.

### Example: General Quality Assessment

```python
from karenina import Benchmark
from karenina.schemas import Rubric, RubricTrait

# Create benchmark
benchmark = Benchmark.create(name="Genomics Knowledge Benchmark")

# Add questions
benchmark.add_question(
    question="How many chromosomes are in a human somatic cell?",
    raw_answer="46"
)

benchmark.add_question(
    question="What is the approved drug target of Venetoclax?",
    raw_answer="BCL2"
)

benchmark.add_question(
    question="How many protein subunits does hemoglobin A have?",
    raw_answer="4"
)

# Create global rubric with LLM-based traits
# These traits will be evaluated for EVERY question
global_rubric = Rubric(
    name="Answer Quality Assessment",
    traits=[
        RubricTrait(
            name="Conciseness",
            description="Rate the conciseness of the answer on a scale from 1 (very verbose) to 5 (extremely concise).",
            kind="score"
        ),
        RubricTrait(
            name="Clarity",
            description="Rate how clear and understandable the answer is, from 1 (confusing) to 5 (crystal clear).",
            kind="score"
        )
    ]
)

# Set as global rubric - applies to ALL questions
benchmark.set_global_rubric(global_rubric)
```

**What happens during verification:**
- The parsing model evaluates **both** traits (Conciseness and Clarity) for **all three questions**
- Each question receives scores from 1-5 for each trait
- Results show how responses perform on general quality metrics

---

## Creating Question-Specific Rubrics

Question-specific rubrics apply to **a single question only**. They're perfect for domain validation and specialized requirements.

### Example 1: Regex-Based Domain Validation

Check that a specific answer mentions required terminology:

```python
from karenina.schemas import Rubric, ManualRubricTrait

# This rubric is ONLY for the Venetoclax question
venetoclax_rubric = Rubric(
    name="Drug Mechanism Validation",
    traits=[
        ManualRubricTrait(
            name="Mentions BH3 Proteins",
            description="Answer must mention BH3 proteins (the mechanism of BCL2 inhibition)",
            pattern=r"\bBH3\b",
            case_sensitive=False
        )
    ]
)

# Add question with specific rubric
# This rubric ONLY applies to THIS question, not the others
benchmark.add_question(
    question="What is the approved drug target of Venetoclax?",
    raw_answer="BCL2",
    rubric=venetoclax_rubric
)
```

**What happens during verification:**
- The regex pattern checks **only** the Venetoclax answer for "BH3"
- Other questions are NOT checked for this pattern
- Returns `True` if the pattern is found, `False` otherwise

### Example 2: Metric-Based Classification

Evaluate classification accuracy for a question that requires identifying items in categories:

```python
from karenina.schemas import Rubric, MetricRubricTrait

# This rubric is ONLY for the disease classification question
disease_rubric = Rubric(
    name="Inflammatory Disease Classification",
    traits=[
        MetricRubricTrait(
            name="Inflammatory Disease Accuracy",
            description="Evaluate accuracy of identifying inflammatory lung diseases",
            metrics=["precision", "recall", "f1"],
            tp_instructions=[
                "asthma",       # Inflammatory diseases (should be identified)
                "bronchitis",
                "pneumonia",
                "pleurisy"
            ],
            fp_instructions=[
                "emphysema",            # NOT inflammatory (should be excluded)
                "pulmonary fibrosis",
                "sarcoidosis"
            ]
        )
    ]
)

# Add question with classification metric rubric
# This rubric ONLY applies to THIS question
benchmark.add_question(
    question="Which of the following are inflammatory lung diseases: asthma, bronchitis, pneumonia, emphysema, pulmonary fibrosis, sarcoidosis, pleurisy?",
    raw_answer="asthma, bronchitis, pneumonia, pleurisy",
    rubric=disease_rubric
)
```

**What happens during verification:**
- The parsing model extracts disease names from the answer
- Each disease is categorized as TP (correct inflammatory), FP (incorrect inflammatory), or FN (missed inflammatory)
- Precision, recall, and F1 score are calculated
- Other questions do NOT use this metric evaluation

---

## Combining Global and Question-Specific Rubrics

You can use both global and question-specific rubrics in the same benchmark:

```python
from karenina import Benchmark
from karenina.schemas import Rubric, RubricTrait, ManualRubricTrait

# Create benchmark
benchmark = Benchmark.create(name="Genomics Knowledge Benchmark")

# Global rubric: applies to ALL questions
global_rubric = Rubric(
    name="General Quality",
    traits=[
        RubricTrait(
            name="Conciseness",
            description="Rate the conciseness of the answer on a scale from 1 (very verbose) to 5 (extremely concise).",
            kind="score"
        ),
        RubricTrait(
            name="Clarity",
            description="Rate how clear and understandable the answer is, from 1 (confusing) to 5 (crystal clear).",
            kind="score"
        )
    ]
)

benchmark.set_global_rubric(global_rubric)

# Question 1: Uses only global rubric
benchmark.add_question(
    question="How many chromosomes are in a human somatic cell?",
    raw_answer="46"
)

# Question 2: Uses global rubric + question-specific rubric
venetoclax_rubric = Rubric(
    name="Drug Mechanism Validation",
    traits=[
        ManualRubricTrait(
            name="Mentions BH3 Proteins",
            description="Answer must mention BH3 proteins",
            pattern=r"\bBH3\b",
            case_sensitive=False
        )
    ]
)

benchmark.add_question(
    question="What is the approved drug target of Venetoclax?",
    raw_answer="BCL2",
    rubric=venetoclax_rubric  # Question-specific rubric
)

# Question 3: Uses only global rubric
benchmark.add_question(
    question="How many protein subunits does hemoglobin A have?",
    raw_answer="4"
)
```

**Result:**
- **Question 1** (chromosomes): Evaluated for Conciseness and Clarity (global rubric)
- **Question 2** (Venetoclax): Evaluated for Conciseness, Clarity (global rubric) + BH3 mention check (question-specific rubric)
- **Question 3** (hemoglobin): Evaluated for Conciseness and Clarity (global rubric)

---

## Working with Rubric Results

After running verification with rubrics, you can access the results:

```python
from karenina.schemas import ModelConfig, VerificationConfig

# Configure verification
model_config = ModelConfig(
    id="gpt-4.1-mini",
    model_provider="openai",
    model_name="gpt-4.1-mini",
    temperature=0.0,
    interface="langchain"
)

config = VerificationConfig(
    answering_models=[model_config],
    parsing_models=[model_config]
)

# Run verification
results = benchmark.run_verification(config)

# Access rubric scores for each question
for result in results:
    print(f"\nQuestion: {result.question.question}")

    # LLM-based trait scores (global rubric)
    if result.rubric_scores:
        print("  Global Rubric Scores:")
        for trait_name, score in result.rubric_scores.items():
            print(f"    {trait_name}: {score}/5")

    # Regex trait results (question-specific)
    if result.manual_rubric_results:
        print("  Regex Trait Results:")
        for trait_name, passed in result.manual_rubric_results.items():
            status = "✓ Pass" if passed else "✗ Fail"
            print(f"    {trait_name}: {status}")

    # Metric trait results (question-specific)
    if result.metric_rubric_results:
        print("  Metric Trait Results:")
        for trait_name, metrics in result.metric_rubric_results.items():
            print(f"    {trait_name}:")
            for metric_name, value in metrics.items():
                print(f"      {metric_name}: {value:.2f}")
```

**Example output:**
```
Question: How many chromosomes are in a human somatic cell?
  Global Rubric Scores:
    Conciseness: 5/5
    Clarity: 5/5

Question: What is the approved drug target of Venetoclax?
  Global Rubric Scores:
    Conciseness: 4/5
    Clarity: 5/5
  Regex Trait Results:
    Mentions BH3 Proteins: ✓ Pass

Question: Which of the following are inflammatory lung diseases...
  Global Rubric Scores:
    Conciseness: 4/5
    Clarity: 4/5
  Metric Trait Results:
    Inflammatory Disease Accuracy:
      precision: 1.00
      recall: 0.75
      f1: 0.86
```

---

## Rubric Best Practices

### Use Global Rubrics For

✅ **Universal quality traits**:
- Clarity, conciseness, completeness
- General style requirements
- Safety checks that apply to all responses

✅ **Benchmark-wide standards**:
- Evaluation criteria that should be consistent across all questions

### Use Question-Specific Rubrics For

✅ **Domain-specific validation**:
- Required terminology checks (e.g., "must mention BH3")
- Question-specific format requirements

✅ **Classification metrics**:
- Precision/recall for categorization questions
- Accuracy for multi-label classification

✅ **Specialized requirements**:
- Checks that only make sense for specific questions

### Choose the Right Trait Type

**LLM-Based Traits (`RubricTrait`)**:
- Qualitative assessment requiring judgment
- Scoring nuanced qualities (clarity, completeness)
- When you need 1-5 scale or binary pass/fail

**Regex-Based Traits (`ManualRubricTrait`)**:
- Deterministic format validation
- Keyword or terminology checks
- When you need exact pattern matching

**Metric-Based Traits (`MetricRubricTrait`)**:
- Classification or categorization questions
- When you need quantitative metrics (precision, recall, F1)
- Multi-label evaluation with confusion matrix

---

## Complete Example

Here's a complete workflow showing both global and question-specific rubrics with all three trait types:

```python
from karenina import Benchmark
from karenina.schemas import (
    Rubric, RubricTrait, ManualRubricTrait, MetricRubricTrait,
    ModelConfig, VerificationConfig
)

# 1. Create benchmark
benchmark = Benchmark.create(
    name="Genomics Knowledge Benchmark",
    description="Testing LLM knowledge of genomics and molecular biology",
    version="1.0.0"
)

# 2. Create global rubric (applies to ALL questions)
global_rubric = Rubric(
    name="General Quality Assessment",
    traits=[
        RubricTrait(
            name="Conciseness",
            description="Rate the conciseness of the answer on a scale from 1 (very verbose) to 5 (extremely concise).",
            kind="score"
        ),
        RubricTrait(
            name="Clarity",
            description="Rate how clear and understandable the answer is, from 1 (confusing) to 5 (crystal clear).",
            kind="score"
        )
    ]
)

benchmark.set_global_rubric(global_rubric)

# 3. Add question with only global rubric
benchmark.add_question(
    question="How many chromosomes are in a human somatic cell?",
    raw_answer="46"
)

# 4. Add question with regex-based question-specific rubric
venetoclax_rubric = Rubric(
    name="Drug Mechanism Validation",
    traits=[
        ManualRubricTrait(
            name="Mentions BH3 Proteins",
            description="Answer must mention BH3 proteins",
            pattern=r"\bBH3\b",
            case_sensitive=False
        )
    ]
)

benchmark.add_question(
    question="What is the approved drug target of Venetoclax?",
    raw_answer="BCL2",
    rubric=venetoclax_rubric
)

# 5. Add question with metric-based question-specific rubric
disease_rubric = Rubric(
    name="Inflammatory Disease Classification",
    traits=[
        MetricRubricTrait(
            name="Inflammatory Disease Accuracy",
            description="Evaluate accuracy of identifying inflammatory lung diseases",
            metrics=["precision", "recall", "f1"],
            tp_instructions=["asthma", "bronchitis", "pneumonia", "pleurisy"],
            fp_instructions=["emphysema", "pulmonary fibrosis", "sarcoidosis"]
        )
    ]
)

benchmark.add_question(
    question="Which of the following are inflammatory lung diseases: asthma, bronchitis, pneumonia, emphysema, pulmonary fibrosis, sarcoidosis, pleurisy?",
    raw_answer="asthma, bronchitis, pneumonia, pleurisy",
    rubric=disease_rubric
)

# 6. Generate templates
model_config = ModelConfig(
    id="gpt-4.1-mini",
    model_provider="openai",
    model_name="gpt-4.1-mini",
    temperature=0.1,
    interface="langchain"
)

benchmark.generate_all_templates(model_config=model_config)

# 7. Run verification with rubrics
config = VerificationConfig(
    answering_models=[model_config],
    parsing_models=[model_config]
)

results = benchmark.run_verification(config)

# 8. Analyze results
print(f"✓ Verification complete: {len(results)} questions evaluated")
print(f"✓ All questions assessed for: Conciseness, Clarity")
print(f"✓ Venetoclax question checked for: BH3 mention")
print(f"✓ Disease question evaluated with: Precision, Recall, F1")

# 9. Save benchmark
benchmark.save("genomics_benchmark.jsonld")
```

---

## Next Steps

Once you have rubrics configured:

- [Run verification](verification.md) to apply both template and rubric evaluation
- [Analyze results](verification.md#analyzing-results) to understand performance across different criteria
- [Save and load benchmarks](saving-loading.md) to preserve your rubric configurations
- [Export results](saving-loading.md#exporting-results) to CSV or JSON for further analysis

---

## Related Documentation

- [Adding Questions](adding-questions.md) - Populate your benchmark with questions
- [Templates](templates.md) - Structured answer evaluation for factual correctness
- [Verification](verification.md) - Run evaluations with multiple models
- [Quick Start](../quickstart.md) - End-to-end workflow example
