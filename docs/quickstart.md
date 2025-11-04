# Quick Start

Get started with Karenina in just a few minutes! This guide walks you through creating your first benchmark, adding questions, generating templates, and running verification.

---

## Prerequisites

Before you begin, make sure you have:

1. **Installed Karenina**:
   ```bash
   pip install karenina
   ```

2. **Set up API keys** (for LLM providers):
   ```bash
   # For OpenAI
   export OPENAI_API_KEY="your-api-key-here"

   # For Google Gemini
   export GOOGLE_API_KEY="your-api-key-here"

   # For Anthropic Claude
   export ANTHROPIC_API_KEY="your-api-key-here"
   ```

3. **Python 3.9+** installed

---

## Complete Workflow Example

This example demonstrates the full Karenina workflow: creating a benchmark, adding questions, generating templates, creating a rubric, running verification, and exporting results.

### Step 1: Create a Benchmark

```python
from karenina import Benchmark

# Create a new benchmark
benchmark = Benchmark.create(
    name="Genomics Knowledge Benchmark",
    description="Testing LLM knowledge of genomics and molecular biology",
    version="1.0.0",
    creator="Your Name"
)

print(f"Created benchmark: {benchmark.name}")
```

---

### Step 2: Add Questions

You can add questions manually or extract them from files (Excel, CSV, TSV).

**Option A: Add questions manually**

```python
# Add a few questions with answers
questions = [
    {
        "question": "How many chromosomes are in a human somatic cell?",
        "answer": "46",
        "author": {"name": "Bio Curator", "email": "curator@example.com"}
    },
    {
        "question": "What is the approved drug target of Venetoclax?",
        "answer": "BCL2",
        "author": {"name": "Bio Curator", "email": "curator@example.com"}
    },
    {
        "question": "How many protein subunits does hemoglobin A have?",
        "answer": "4",
        "author": {"name": "Bio Curator", "email": "curator@example.com"}
    }
]

question_ids = []
for q in questions:
    qid = benchmark.add_question(
        question=q["question"],
        raw_answer=q["answer"],
        author=q["author"]
    )
    question_ids.append(qid)

print(f"Added {len(question_ids)} questions")
```

**Option B: Extract from a file**

```python
from karenina.questions.extractor import extract_questions_from_file

# Extract questions from Excel/CSV/TSV
questions = extract_questions_from_file(
    file_path="/path/to/questions.xlsx",
    question_column="Question",
    answer_column="Answer",
    author_name_column="Author",  # Optional
    keywords_column="Keywords"    # Optional
)

# Add extracted questions to benchmark
for q in questions:
    benchmark.add_question(**q)
```

---

### Step 3: Generate Answer Templates

Answer templates define how to extract and verify information from LLM responses. Karenina can generate these automatically using an LLM.

**Automatic template generation (recommended):**

```python
from karenina.schemas import ModelConfig

# Configure the LLM for template generation
model_config = ModelConfig(
    id="gpt-4.1-mini",
    model_provider="openai",
    model_name="gpt-4.1-mini",
    temperature=0.1,
    interface="langchain"
)

# Generate templates for all questions
print("Generating templates...")
results = benchmark.generate_all_templates(
    model_config=model_config,
    force_regenerate=False  # Skip questions that already have templates
)

print(f"Generated {len(results)} templates")
```

**Manual template creation (for advanced users):**

If you prefer full control, you can write templates manually:

```python
# Manual template example
template_code = '''class Answer(BaseAnswer):
    target: str = Field(description="The protein target mentioned in the response")

    def model_post_init(self, __context):
        self.correct = {"target": "BCL2"}

    def verify(self) -> bool:
        return self.target.strip().upper() == self.correct["target"].upper()
'''

# Add template to a specific question
benchmark.add_question(
    question="What is the approved drug target of Venetoclax?",
    raw_answer="BCL2",
    answer_template=template_code,
    finished=True  # Mark as ready for verification
)
```

---

### Step 4: Create a Rubric (Optional)

Rubrics assess qualitative aspects of answers. Karenina supports two types of rubrics:

**Global Rubrics**: Applied to ALL questions (great for general quality assessment)
**Question-Specific Rubrics**: Applied to ONE specific question (great for domain-specific validation)

#### Global Rubric (LLM-based traits)

These traits evaluate general answer quality across all questions:

```python
from karenina.schemas import RubricTrait

# Create a global rubric with LLM-based traits
# These will be evaluated for EVERY question in the benchmark
global_rubric = benchmark.create_global_rubric(
    name="Answer Quality",
    traits=[
        RubricTrait(
            name="Conciseness",
            description="Rate how concise the answer is on a scale of 1-5, where 1 is very verbose and 5 is extremely concise.",
            kind="score"  # Returns a score from 1-5
        ),
        RubricTrait(
            name="Clarity",
            description="Is the answer clear and easy to understand?",
            kind="binary"  # Returns pass/fail
        )
    ]
)

print(f"Created global rubric: {global_rubric.name}")
print("This rubric will be evaluated for ALL questions")
```

#### Question-Specific Rubric (Regex-based trait)

This trait validates that the answer contains the exact correct gene symbol:

```python
from karenina.schemas import ManualRubricTrait

# Find the drug target question ID
drug_target_qid = [qid for qid in question_ids
                   if "Venetoclax" in benchmark.get_question(qid).question][0]

# Create a regex trait specific to the drug target question
# The answer must contain "BCL2" (case-sensitive)
regex_trait = ManualRubricTrait(
    name="BH3 Mention",
    description="Answer must contain a mention to BH3 proteins",
    pattern=r"\bBH3\b",  # Matches the exact word "BH3" with word boundaries
    case_sensitive=True,  # "BH3" is correct, "bcl2" or "Bcl2" would fail
    invert=False  # Match = pass (answer contains BH3)
)

# Add ONLY to the drug target question (not global!)
benchmark.add_question_rubric(
    question_id=drug_target_qid,
    traits=[regex_trait]
)

print(f"Created question-specific rubric for question: {drug_target_qid}")
print("This rubric will ONLY be evaluated for the Venetoclax question")
print("It checks that the answer contains 'BH3' (case-sensitive)")
```

#### Question-Specific Rubric (Metric-based trait)

For questions requiring classification accuracy (e.g., identifying disease types):

```python
from karenina.schemas import MetricRubricTrait

# Example: If you had a question like "List inflammatory lung diseases"
# Add this question to demonstrate metric traits
disease_qid = benchmark.add_question(
    question="Which of the following are inflammatory lung diseases: asthma, bronchitis, pneumonia, emphysema, pulmonary fibrosis?",
    raw_answer="asthma, bronchitis, pneumonia",
    author={"name": "Bio Curator", "email": "curator@example.com"}
)

# Create a metric trait to evaluate classification accuracy
metric_trait = MetricRubricTrait(
    name="Inflammatory Disease Identification",
    description="Evaluate accuracy of identifying inflammatory lung diseases",
    metrics=["precision", "recall", "f1"],
    tp_instructions=[
        "asthma",       # Should be identified (inflammatory)
        "bronchitis",   # Should be identified (inflammatory)
        "pneumonia"     # Should be identified (inflammatory)
    ],
    fp_instructions=[
        "emphysema",            # Should NOT be identified (obstructive, not inflammatory)
        "pulmonary fibrosis"    # Should NOT be identified (restrictive, not inflammatory)
    ],
    repeated_extraction=True  # Remove duplicate mentions
)

# Add ONLY to the disease classification question
benchmark.add_question_rubric(
    question_id=disease_qid,
    traits=[metric_trait]
)

print(f"Created metric-based rubric for question: {disease_qid}")
print("This will compute precision, recall, and F1 score for this specific question")
```

**Key Distinction:**
- **Global rubrics** (clarity, conciseness): Assessed for every question → generic quality metrics
- **Question-specific rubrics** (gene format, disease classification): Assessed for one question → domain-specific validation

---

### Step 5: Run Verification

Configure models and run verification to evaluate LLM responses against your templates and rubrics.

```python
from karenina.schemas import VerificationConfig, ModelConfig

# Configure verification
config = VerificationConfig(
    # Models that generate answers (can use multiple for comparison)
    answering_models=[
        ModelConfig(
            id="gpt-4.1-mini",
            model_provider="openai",
            model_name="gpt-4.1-mini",
            temperature=0.7,
            interface="langchain",
            system_prompt="You are a knowledgeable assistant. Answer accurately and concisely."
        )
    ],
    # Models that parse/judge answers (usually more capable models)
    parsing_models=[
        ModelConfig(
            id="gpt-4.1-mini",
            model_provider="openai",
            model_name="gpt-4.1-mini",
            temperature=0.0,
            interface="langchain",
            system_prompt="You are an expert judge. Parse and evaluate responses carefully."
        )
    ],
    rubric_enabled=True,  # Enable rubric evaluation
    replicate_count=1,    # Number of times to run each question (use >1 for statistical analysis)
    deep_judgment_enabled=False,  # Enable for detailed feedback with excerpts
    abstention_check_enabled=True  # Detect when models refuse to answer
)

# Run verification
print("Running verification...")
results = benchmark.run_verification(config)

print(f"Verification complete! Processed {len(results)} questions")
```

**Using different interfaces:**

```python
# OpenRouter
ModelConfig(
    id="sonnet-4.5",
    model_provider="openrouter",
    model_name="anthropic/claude-sonnet-4.5",
    interface="openrouter"
)

# OpenAI-compatible endpoint (e.g., Ollama)
ModelConfig(
    id="glm46",
    model_name="glm-4.6",
    interface="openai_endpoint",
    endpoint_api_key="your-api-key",
    endpoint_base_url="http://localhost:11434/v1"
)

# Manual traces (for testing/debugging without API calls)
ModelConfig(
    id="manual",
    model_provider="manual",
    model_name="manual",
    interface="manual"
)
```

---

### Step 6: Access and Analyze Results

After verification, you can access detailed results for each question.

```python
# Iterate through results
for question_id, result in results.items():
    print(f"\nQuestion: {result.question_text}")
    print(f"Verification: {'✓ PASS' if result.verify_result else '✗ FAIL'}")
    print(f"Model Answer: {result.answering_response[:100]}...")  # First 100 chars

    # Access rubric scores (if rubric enabled)
    if result.rubric_scores:
        print("Rubric Scores:")
        for trait_name, score in result.rubric_scores.items():
            print(f"  - {trait_name}: {score}")

    # Check for abstention (if enabled)
    if result.abstention_detected:
        print(f"⚠ Model abstained: {result.abstention_reasoning}")
```

**Calculate aggregate metrics:**

```python
# Calculate pass rate
total = len(results)
passed = sum(1 for r in results.values() if r.verify_result)
pass_rate = (passed / total) * 100

print(f"\n{'='*50}")
print(f"Overall Pass Rate: {pass_rate:.1f}% ({passed}/{total})")
print(f"{'='*50}")
```

---

### Step 7: Save and Export

Save your benchmark as a checkpoint or export results for analysis.

**Save checkpoint (preserves full benchmark state):**

```python
from pathlib import Path

# Save benchmark with all questions, templates, and results
checkpoint_path = Path("genomics_benchmark.jsonld")
benchmark.save(checkpoint_path)
print(f"Saved checkpoint to {checkpoint_path}")

# Load later
loaded_benchmark = Benchmark.load(checkpoint_path)
```

**Export verification results to CSV/JSON:**

```python
# Export to CSV for spreadsheet analysis
benchmark.export_verification_results_to_file(
    file_path=Path("results.csv"),
    format="csv"
)

# Export to JSON for programmatic analysis
benchmark.export_verification_results_to_file(
    file_path=Path("results.json"),
    format="json"
)

print("Exported verification results to results.csv and results.json")
```

**Save to database:**

```python
# Save benchmark to SQLite database (with checkpoint file)
benchmark.save_to_db(
    storage="sqlite:///benchmarks.db",
    checkpoint_path=checkpoint_path
)

# Load from database later
loaded = Benchmark.load_from_db(
    benchmark_name="Genomics Knowledge Benchmark",
    storage="sqlite:///benchmarks.db"
)
```

---

## Complete Example Script

Here's the entire workflow in one script with both global and question-specific rubrics:

```python
from karenina import Benchmark
from karenina.schemas import (
    VerificationConfig, ModelConfig, RubricTrait,
    ManualRubricTrait, MetricRubricTrait
)
from pathlib import Path

# 1. Create benchmark
benchmark = Benchmark.create(
    name="Genomics Quiz",
    description="Basic genomics knowledge test",
    version="1.0.0",
    creator="Your Name"
)

# 2. Add questions
questions = [
    ("How many chromosomes are in a human somatic cell?", "46"),
    ("What is the approved drug target of Venetoclax?", "BCL2"),
    ("How many protein subunits does hemoglobin A have?", "4")
]

question_ids = []
for q, a in questions:
    qid = benchmark.add_question(question=q, raw_answer=a, author={"name": "Bio Curator"})
    question_ids.append(qid)

# Add a classification question for metric trait demonstration
disease_qid = benchmark.add_question(
    question="Which of the following are inflammatory lung diseases: asthma, bronchitis, pneumonia, emphysema, pulmonary fibrosis?",
    raw_answer="asthma, bronchitis, pneumonia",
    author={"name": "Bio Curator"}
)

# 3. Generate templates
model_config = ModelConfig(
    id="gpt-4.1-mini",
    model_provider="openai",
    model_name="gpt-4.1-mini",
    temperature=0.1,
    interface="langchain"
)
benchmark.generate_all_templates(model_config=model_config)

# 4. Create global rubric (applies to ALL questions)
benchmark.create_global_rubric(
    name="Answer Quality",
    traits=[
        RubricTrait(
            name="Conciseness",
            description="Rate conciseness 1-5",
            kind="score"
        ),
        RubricTrait(
            name="Clarity",
            description="Is the answer clear?",
            kind="binary"
        )
    ]
)

# 5. Add question-specific rubrics
# Regex trait for Venetoclax question
drug_target_qid = [qid for qid in question_ids if "Venetoclax" in benchmark.get_question(qid).question][0]
benchmark.add_question_rubric(
    question_id=drug_target_qid,
    traits=[ManualRubricTrait(
        name="BH3 Mention",
        description="Answer must mention BH3 proteins",
        pattern=r"\bBH3\b",
        case_sensitive=True,
        invert=False
    )]
)

# Metric trait for disease classification question
benchmark.add_question_rubric(
    question_id=disease_qid,
    traits=[MetricRubricTrait(
        name="Inflammatory Disease ID",
        description="Evaluate disease classification accuracy",
        metrics=["precision", "recall", "f1"],
        tp_instructions=["asthma", "bronchitis", "pneumonia"],
        fp_instructions=["emphysema", "pulmonary fibrosis"],
        repeated_extraction=True
    )]
)

# 6. Run verification
config = VerificationConfig(
    answering_models=[model_config],
    parsing_models=[model_config],
    rubric_enabled=True
)
results = benchmark.run_verification(config)

# 7. Analyze results
passed = sum(1 for r in results.values() if r.verify_result)
print(f"Pass Rate: {(passed/len(results)*100):.1f}%")

# 8. Save and export
checkpoint_path = Path("genomics_quiz.jsonld")
benchmark.save(checkpoint_path)
benchmark.export_verification_results_to_file(
    file_path=Path("results.csv"),
    format="csv"
)

print("Done! Check results.csv for detailed results.")
```

---

## Next Steps

Now that you've completed your first benchmark, explore these guides:

### Core Usage
- **[Defining Benchmarks](using-karenina/defining-benchmark.md)** - Benchmark creation, metadata, and organization
- **[Adding Questions](using-karenina/adding-questions.md)** - File extraction, metadata mapping, and management
- **[Templates](using-karenina/templates.md)** - Creating and customizing answer templates
- **[Rubrics](using-karenina/rubrics.md)** - Evaluation criteria and trait types
- **[Verification](using-karenina/verification.md)** - Configuration, replication, and result analysis
- **[Saving & Loading](using-karenina/saving-loading.md)** - Checkpoints, database persistence, and export

### Advanced Features
- **[Deep-Judgment](advanced/deep-judgment.md)** - Extract detailed feedback with excerpts and reasoning
- **[Few-Shot Prompting](advanced/few-shot.md)** - Guide responses with examples
- **[Abstention Detection](advanced/abstention-detection.md)** - Handle model refusals
- **[Embedding Check](advanced/embedding-check.md)** - Semantic similarity fallback
- **[Presets](advanced/presets.md)** - Save and reuse verification configurations
- **[System Integration](advanced/integration.md)** - Server and GUI integration

### Reference
- **[Features Overview](features.md)** - Complete feature catalog
- **[Configuration](configuration.md)** - Environment variables and defaults
- **[API Reference](api-reference.md)** - Complete API documentation
- **[Troubleshooting](troubleshooting.md)** - Common issues and solutions

---

## Tips for Success

1. **Start simple**: Begin with a few questions and manual templates to understand the workflow
2. **Use template generation**: Let Karenina generate templates automatically to save time
3. **Iterate on templates**: Review and refine generated templates for better accuracy
4. **Leverage rubrics**: Add rubrics to assess answer quality beyond correctness
5. **Run replications**: Use `replicate_count > 1` for statistical analysis of model consistency
6. **Save checkpoints**: Regularly save your benchmark to avoid losing work
7. **Export results**: Use CSV export for easy analysis in spreadsheet tools

---

## Common Questions

**Q: Do I need to write templates manually?**
A: No! Karenina can generate templates automatically using LLMs. Manual creation is only needed for complex custom logic.

**Q: Can I use local models?**
A: Yes! Use the `openai_endpoint` interface with Ollama, vLLM, or any OpenAI-compatible server.

**Q: How do I compare multiple models?**
A: Add multiple models to `answering_models` in your verification config. Karenina will test all of them.

**Q: What's the difference between templates and rubrics?**
A: Templates verify **factual correctness** (e.g., "Is the answer 'BCL2'?"), while rubrics assess **qualitative traits** (e.g., "Is the answer concise?").

**Q: Can I test without making API calls?**
A: Yes! Use the `manual` interface with pre-recorded traces for testing and debugging without costs.

---

[← Back to Documentation Index](index.md) | [View All Features](features.md)
