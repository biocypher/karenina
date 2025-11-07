# Answer Templates

Templates define how to evaluate LLM responses programmatically. This guide covers what templates are, why they're useful, and how to create them automatically or manually.

**Quick Navigation:**

- [What Are Templates?](#what-are-templates) - Core concepts and structure
- [Why Use Templates?](#why-use-templates) - Benefits and use cases
- [Automatic Template Generation](#automatic-template-generation-recommended) - Recommended LLM-based approach
- [Manual Template Creation](#manual-template-creation-advanced) - Advanced custom templates
- [When to Use Which Approach](#when-to-use-which-approach) - Decision guide for automatic vs manual
- [Complete Example](#complete-example) - End-to-end workflow

---

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

### How Automatic Template Generation Works

Understanding the template generation process helps you troubleshoot issues and make informed decisions about when to use automatic vs manual templates.

**High-Level Process:**

When you call `generate_all_templates(model_config)`, Karenina performs a **three-phase structured generation** for each question:

**Phase 1: Ground Truth Extraction**

The LLM analyzes the question-answer pair and generates a JSON specification defining the attributes needed for verification.

Example for "What is the approved drug target of Venetoclax?" (answer: "BCL2"):

```json
{
  "attributes": [
    {
      "name": "mentions_bcl2_protein",
      "type": "bool",
      "ground_truth": true
    },
    {
      "name": "mentions_apoptosis_regulation",
      "type": "bool",
      "ground_truth": false
    }
  ]
}
```

**Important design principle:** The system strongly **suggests boolean-based evaluation** rather than free-text string matching. Text-based assessment is typically converted to boolean checks for concept presence.

**Phase 2: Field Description Generation**

Using the ground truth specification, the LLM generates clear instructions for each attribute that will guide the judge model during response parsing.

Example output:

```json
{
  "field_descriptions": {
    "mentions_bcl2_protein": "Answer with true if the response mentions BCL2 or semantically related terms; otherwise answer false.",
    "mentions_apoptosis_regulation": "Answer with true if the response discusses apoptosis regulation mechanisms; otherwise answer false."
  }
}
```

These descriptions emphasize **semantic equivalence** over exact string matching.

**Phase 3: Code Generation**

Karenina programmatically builds the Pydantic class using the structured outputs from Phases 1 and 2. The generated code includes:

- Field definitions with judge instructions from Phase 2
- The `model_post_init()` method with ground truth values from Phase 1
- The `verify()` method with type-appropriate comparison logic
- The `verify_granular()` method for partial credit (multi-attribute templates only)

**Validation and Storage**

After generation, Karenina validates the Python code and stores it with the question. If validation fails, the system retries with error context (up to 3 attempts total).

**What Makes This Approach Effective:**

- **Structured Outputs**: JSON schema validation ensures consistent, parseable results from the LLM
- **Semantic Evaluation**: Boolean attributes capture concept presence, making verification robust to paraphrasing
- **Type Safety**: Enforced constraints prevent ambiguous evaluation strategies
- **Retry Logic**: Failed validations trigger automatic regeneration with error context
- **Partial Credit**: Multi-attribute templates support granular scoring automatically

**Why Boolean Attributes?**

The system strongly prefers boolean attributes over string extraction because:

- **Flexibility**: Judges check if concepts are present, not exact phrases
- **Deterministic**: `true`/`false` comparisons are unambiguous
- **Robust**: Handles paraphrasing, synonyms, and variations naturally
- **Avoids pitfalls**: No need for case normalization, fuzzy matching, or string similarity thresholds

**Trade-off: Speed vs. Rigor**

The current approach **may expose ground truth to the judge model** through field descriptions. For example, asking "Answer with true if the response mentions BCL2" reveals that BCL2 is the expected answer. The judge becomes aware of what's "correct" rather than acting as a pure information extractor.

**Alternative approach (more rigorous but requires manual curation):**

- Have the judge extract information **without knowing the correct answer**
- Field descriptions would be neutral (e.g., "Extract the protein target mentioned")
- All verification logic stays in the `verify()` method
- Judge models act as pure parsers, not evaluators

**Current approach (faster but less rigorous):**

- Field descriptions include hints about correctness
- Allows automated template generation with minimal manual curation
- Judge models perform some evaluation during extraction
- Faster to deploy at scale

This is a **design trade-off**: a more rigorous benchmark requires more manual template curation, while the current automated approach prioritizes speed and scalability at the cost of some methodological purity.

If you need the more rigorous approach, see [Manual Template Creation](#manual-template-creation-advanced) for how to write templates with neutral field descriptions.

**When Generation Might Fail:**

Template generation works well for most questions, but may struggle with:

- **Highly ambiguous questions** where even the ground truth is unclear
- **Complex compositional logic** requiring interdependent attribute checks
- **Domain-specific tolerance requirements** (e.g., "within 10% is acceptable")
- **Unusual answer formats** that don't fit the structured attribute model

In these cases, you can fall back to [manual template creation](#manual-template-creation-advanced).

---

## Manual Template Creation (Advanced)

For full control over evaluation logic, you can write templates manually. This is useful for complex verification requirements or custom validation rules.

### Basic Template Structure

Templates inherit from `BaseAnswer` and must include these **three required components**:

**1. Field Definitions**

Fields specify what data to extract. Each field should have a clear description that guides the judge LLM:

```python
from karenina.domain.answers import BaseAnswer
from pydantic import Field

# String fields
target: str = Field(description="The protein target mentioned in the response")

# Integer/Float fields
count: int = Field(description="The number of items mentioned")
score: float = Field(description="Accuracy score 0.0-1.0", ge=0.0, le=1.0)

# Boolean fields (recommended for rigorous evaluation - see trade-off discussion above)
mentions_bcl2: bool = Field(description="Extract the protein target mentioned in the response")

# List fields
proteins: List[str] = Field(description="List of proteins mentioned")
```

**2. `model_post_init(self, __context)` Method** (required)

- **Purpose**: Initialize the ground truth values after Pydantic constructs the model
- **Returns**: `None` (no return value)
- **Usage**: Set `self.correct` dictionary with expected values

```python
def model_post_init(self, __context):
    self.correct = {"count": 46}
```

**3. `verify(self) -> bool` Method** (required)

- **Purpose**: Determine if the extracted answer is correct
- **Returns**: `bool` - `True` if correct, `False` if incorrect
- **Usage**: Compare extracted field values against `self.correct`

```python
def verify(self) -> bool:
    return self.count == self.correct["count"]
```

**Complete Example:**

```python
class Answer(BaseAnswer):
    count: int = Field(description="The number of chromosomes mentioned in the response")

    def model_post_init(self, __context):
        self.correct = {"count": 46}

    def verify(self) -> bool:
        return self.count == self.correct["count"]
```

### Optional Method: Granular Scoring

**`verify_granular(self) -> float`** (optional)

- **Purpose**: Calculate partial credit for multi-attribute templates
- **Returns**: `float` between 0.0 and 1.0 representing the fraction of correct attributes
- **Usage**: Count matching attributes and return the percentage
- **Note**: Automatically generated for multi-attribute templates; rarely needed for manual templates

```python
def verify_granular(self) -> float:
    correct_count = 0
    total_count = 2

    if self.field1 == self.correct["field1"]:
        correct_count += 1
    if self.field2 == self.correct["field2"]:
        correct_count += 1

    return correct_count / total_count
```

### Adding Manual Templates to Questions

You can add templates in three ways:

**Option 1: Pass an Answer class directly (recommended)**

You can pass the class object directly, and Karenina will **automatically extract the source code**:

```python
from karenina.domain.answers import BaseAnswer
from pydantic import Field

# Use any descriptive class name you like
class VenetoclaxAnswer(BaseAnswer):
    target: str = Field(description="The protein target mentioned")

    def model_post_init(self, __context):
        self.correct = {"target": "BCL2"}

    def verify(self) -> bool:
        return self.target.strip().upper() == self.correct["target"].upper()

# Pass the class directly - source code extraction happens automatically
benchmark.add_question(
    question="What is the approved drug target of Venetoclax?",
    raw_answer="BCL2",
    answer_template=VenetoclaxAnswer
    # finished=True is automatically set when answer_template is provided
)
```

**How automatic source code extraction works:**

- For classes defined in files: `inspect.getsource()` captures the source code automatically
- For classes defined in notebooks: Use `YourClassName.set_source_code_from_notebook()` or the `@capture_answer_source` decorator
- For exec-created classes: Set `YourClassName._source_code` manually

**Option 2: Pass template code as a string**

```python
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
```

**Option 3: Add template to existing question**

```python
question_id = benchmark.add_question(
    question="How many protein subunits does hemoglobin A have?",
    raw_answer="4"
)

# Later, add the template using add_answer_template
template_code = '''class Answer(BaseAnswer):
    count: int = Field(description="The number of subunits mentioned")

    def model_post_init(self, __context):
        self.correct = {"count": 4}

    def verify(self) -> bool:
        return self.count == self.correct["count"]
'''

benchmark.add_answer_template(question_id, template_code)
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

## When to Use Which Approach

### Use Automatic Generation When:

- You have many questions to process
- Questions follow standard patterns (factual recall, numerical answers, multiple choice)
- You're prototyping or testing quickly

### Use Manual Creation When:

- You need very specific verification logic
- You want to implement tolerance ranges or fuzzy matching
- You're creating reusable template libraries
- Automatic generation doesn't produce the desired structure

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
- [Analyze results](verification.md#accessing-verification-results) to assess model performance
- [Save your benchmark](saving-loading.md) using checkpoints or database

---

## Related Documentation

- [Adding Questions](adding-questions.md) - Populate your benchmark with questions
- [Rubrics](rubrics.md) - Assess qualitative aspects beyond factual correctness
- [Verification](verification.md) - Run evaluations with multiple models
- [Quick Start](../quickstart.md) - End-to-end workflow example
