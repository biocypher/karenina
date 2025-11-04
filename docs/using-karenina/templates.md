# Answer Templates

Templates define how to evaluate LLM responses programmatically. This guide covers what templates are, why they're useful, and how to create them automatically or manually.

## What Are Templates?

**Answer templates** are Pydantic classes that specify:

- **What information to extract** from free-text LLM responses
- **How to verify correctness** by comparing extracted data against expected answers
- **The structure of expected answers** (e.g., a drug name, a number, a list of items)

Templates enable **LLM-as-a-judge evaluation**: The answering model generates free text, and the judge model extracts structured data from that text using the template schema. The template then programmatically verifies correctness.

## Why Use Templates?

Templates provide several key benefits:

1. **Flexible Input**: Answering models can respond naturally without strict formatting constraints
2. **Structured Evaluation**: Judge models extract specific fields, making evaluation deterministic
3. **Programmatic Verification**: The `verify()` method implements custom logic for checking correctness
4. **Reusable Patterns**: Templates can be generated automatically for common question types
5. **Transparent Logic**: Evaluation criteria are explicit and inspectable

## Automatic Template Generation (Recommended)

**The recommended approach** is to let Karenina automatically generate templates using an LLM. This is fast, consistent, and works well for most question types.

### Basic Generation

```python
from karenina import Benchmark
from karenina.schemas import ModelConfig

# Create benchmark and add questions
benchmark = Benchmark.create(name="Genomics Knowledge Benchmark")

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
results = benchmark.generate_all_templates(model_config=model_config)

print(f"Generated {len(results)} templates successfully")
```

**What happens:**
1. Karenina sends each question + answer to the LLM
2. The LLM generates a Pydantic class tailored to that specific question
3. The template is automatically validated and associated with the question
4. Questions are marked as "finished" and ready for verification

### Generated Template Example

For the question "What is the approved drug target of Venetoclax?" with answer "BCL2", the LLM might generate:

```python
class Answer(BaseAnswer):
    target: str = Field(description="The protein target mentioned in the response")

    def model_post_init(self, __context):
        self.correct = {"target": "BCL2"}

    def verify(self) -> bool:
        return self.target.strip().upper() == self.correct["target"].upper()
```

This template:
- Extracts the `target` field from free-text responses
- Compares it case-insensitively against "BCL2"
- Returns `True` if they match, `False` otherwise

---

## Manual Template Creation (Advanced)

For full control over evaluation logic, you can write templates manually. This is useful for complex verification requirements or custom validation rules.

### Basic Template Structure

Templates inherit from `BaseAnswer` and define:

1. **Fields**: What data to extract (with descriptions for the judge LLM)
2. **`model_post_init`**: Where to set the correct answer
3. **`verify()`**: Logic to check correctness

```python
from karenina.domain.answers import BaseAnswer
from pydantic import Field

class Answer(BaseAnswer):
    count: int = Field(description="The number of chromosomes mentioned in the response")

    def model_post_init(self, __context):
        self.correct = {"count": 46}

    def verify(self) -> bool:
        return self.count == self.correct["count"]
```

### Adding Manual Templates to Questions

```python
# Option 1: Provide template code when adding the question
template_code = '''class Answer(BaseAnswer):
    target: str = Field(description="The protein target mentioned")

    def model_post_init(self, __context):
        self.correct = {"target": "BCL2"}

    def verify(self) -> bool:
        return self.target.strip().upper() == self.correct["target"].upper()
'''

benchmark.add_question(
    question="What is the approved drug target of Venetoclax?",
    raw_answer="BCL2",
    answer_template=template_code,
    finished=True  # Mark as ready for verification
)

# Option 2: Add template to existing question
question_id = benchmark.add_question(
    question="How many protein subunits does hemoglobin A have?",
    raw_answer="4"
)

template_code = '''class Answer(BaseAnswer):
    count: int = Field(description="The number of subunits mentioned")

    def model_post_init(self, __context):
        self.correct = {"count": 4}

    def verify(self) -> bool:
        return self.count == self.correct["count"]
'''

```

### Complex Template Example

For more sophisticated evaluation, you can include multiple fields and custom logic:

```python
from typing import List

template_code = '''class Answer(BaseAnswer):
    diseases: List[str] = Field(description="List of diseases mentioned in the response")
    inflammatory_count: int = Field(description="Number of inflammatory diseases identified")

    def model_post_init(self, __context):
        self.correct = {
            "inflammatory_diseases": ["asthma", "bronchitis", "pneumonia"],
            "non_inflammatory": ["emphysema", "pulmonary fibrosis"]
        }

    def verify(self) -> bool:
        # Check if the correct inflammatory diseases are identified
        identified = [d.lower().strip() for d in self.diseases]
        correct_identified = sum(1 for d in self.correct["inflammatory_diseases"]
                                if d in identified)

        # At least 2 out of 3 correct inflammatory diseases
        return correct_identified >= 2
'''

benchmark.add_question(
    question="Which of the following are inflammatory lung diseases: asthma, bronchitis, pneumonia, emphysema, pulmonary fibrosis?",
    raw_answer="asthma, bronchitis, pneumonia",
    answer_template=template_code,
    finished=True
)
```

---

## Template Structure Deep Dive

### Required Components

Every template must include these three components:

**1. Field Definitions**

Fields specify what data to extract. Each field should have a clear description that guides the judge LLM:

```python
target: str = Field(description="The protein target mentioned in the response")
count: int = Field(description="The number of items mentioned")
is_correct: bool = Field(description="Whether the response is factually accurate")
```

**2. `model_post_init` Method**

This method sets the correct answer(s) that will be compared during verification:

```python
def model_post_init(self, __context):
    self.correct = {"target": "BCL2"}
```

**3. `verify()` Method**

This method implements the comparison logic and returns `True` if the answer is correct:

```python
def verify(self) -> bool:
    return self.target.strip().upper() == self.correct["target"].upper()
```

### Field Types

Templates can use various field types depending on the expected answer:

```python
# String fields
gene_name: str = Field(description="The gene symbol")

# Integer fields
chromosome_count: int = Field(description="Number of chromosomes")

# Boolean fields
is_correct: bool = Field(description="Whether the statement is true")

# List fields
proteins: List[str] = Field(description="List of proteins mentioned")

# Float fields with constraints
score: float = Field(description="Accuracy score 0.0-1.0", ge=0.0, le=1.0)
```

---

## When to Use Which Approach

### Use Automatic Generation When:

✅ You have many questions to process
✅ Questions follow standard patterns (factual recall, numerical answers, multiple choice)
✅ You want consistency across templates
✅ You're prototyping or testing quickly

### Use Manual Creation When:

✅ You need very specific verification logic
✅ The question requires multi-step validation
✅ You want to implement tolerance ranges or fuzzy matching
✅ You're creating reusable template libraries
✅ Automatic generation doesn't produce the desired structure

---

## Complete Example

Here's a complete workflow showing automatic template generation:

```python
from karenina import Benchmark
from karenina.schemas import ModelConfig, VerificationConfig

# 1. Create benchmark and add questions
benchmark = Benchmark.create(
    name="Genomics Knowledge Benchmark",
    description="Testing LLM knowledge of genomics",
    version="1.0.0"
)

# Add questions
questions = [
    ("How many chromosomes are in a human somatic cell?", "46"),
    ("What is the approved drug target of Venetoclax?", "BCL2"),
    ("How many protein subunits does hemoglobin A have?", "4")
]

for q, a in questions:
    benchmark.add_question(question=q, raw_answer=a, author={"name": "Bio Curator"})

# 2. Generate templates automatically
model_config = ModelConfig(
    id="gpt-4.1-mini",
    model_provider="openai",
    model_name="gpt-4.1-mini",
    temperature=0.1,
    interface="langchain"
)

print("Generating templates...")
results = benchmark.generate_all_templates(model_config=model_config)
print(f"✓ Generated {len(results)} templates")

# 3. Templates are now ready - proceed to verification
config = VerificationConfig(
    answering_models=[model_config],
    parsing_models=[model_config]
)

results = benchmark.run_verification(config)
print(f"✓ Verification complete: {len(results)} questions evaluated")

# 4. Save benchmark
benchmark.save("genomics_benchmark.jsonld")
```

---

## Next Steps

Once you have templates set up for your questions, you can:

- [Create rubrics](rubrics.md) for qualitative assessment criteria
- [Run verification](verification.md) to evaluate LLM responses
- [Analyze results](verification.md#analyzing-results) to assess model performance
- [Save your benchmark](saving-loading.md) using checkpoints or database

---

## Related Documentation

- [Adding Questions](adding-questions.md) - Populate your benchmark with questions
- [Rubrics](rubrics.md) - Assess qualitative aspects beyond factual correctness
- [Verification](verification.md) - Run evaluations with multiple models
- [Quick Start](../quickstart.md) - End-to-end workflow example
