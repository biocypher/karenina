# About Karenina

## Understanding Karenina's Approach

### The Problem

Consider a simple multiple-choice question:

```python
question = "What is the capital of Italy?"
possible_answers = ["Rome", "Milan", "Paris", "New York"]
```

When we query a standard LLM, it usually responds in free text (e.g., `"I think the answer is Rome, because it is the capital of Italy."`). To evaluate such an answer programmatically, we have three approaches:

#### 1. Constrain the Answering Model's Output

We directly instruct the answering model to return a response in a machine-friendly format.

**Example prompt:**

```text
You are answering a multiple-choice question.
Return only the letter of your choice.

Question: What is the capital of Italy?
Options:
A) Rome
B) Milan
C) Paris
D) New York

Answer:
```

**Model output:** `A`

**Pros**: Simple and reliable when models comply

**Cons**: Fragile prompt adherence; requires different strategies for different formats

---

#### 2. Use an LLM as a Judge (Free-Text Evaluation)

Instead of constraining the answering model, we keep its output free-form and rely on a **judge LLM** to interpret it.

**Example:**

* **Answering model output:** `"The capital of Italy is Rome, of course."`
* **Judge model prompt:**
  ```text
  The following is a student's answer to a multiple-choice question.
  Question: What is the capital of Italy?
  Options: Rome, Milan, Paris, New York.
  Student's answer: "The capital of Italy is Rome, of course."
  Which option does this correspond to? Provide a justification.
  ```
* **Judge model output:** `"The student clearly selected Rome, which is correct."`

**Pros**: Flexible, allows natural answering
**Cons**: Judge response is also free text, requiring parsing; potential inconsistencies

---

### The Karenina Strategy

Karenina adopts a **third approach** that combines the advantages of both:

* The **answering model** remains unconstrained, generating natural free text
* The **judge model** returns results in a **structured format** (validated through Pydantic classes)

This setup allows the judge to flexibly interpret free text while ensuring that its own output remains standardized and machine-readable.

#### Example Workflow

**1. Define a Pydantic template:**

```python
from karenina.domain.answers import BaseAnswer
from pydantic import Field

class Answer(BaseAnswer):
    answer: str = Field(description="The name of the city in the response")

    def model_post_init(self, __context):
        self.correct = {"answer": "Rome"}

    def verify(self) -> bool:
        return self.answer.strip().lower() == self.correct["answer"].strip().lower()
```

**Key aspects:**
- The `answer` attribute uses `Field` description to guide the judge
- The `verify` method implements custom validation logic

**2. Answering model generates free text:**

```
"The capital of Italy is Rome."
```

**3. Judge model parses into structured format:**

```python
from langchain_core.output_parsers import PydanticOutputParser

parser = PydanticOutputParser(pydantic_object=Answer)
prompt = parser.get_format_instructions()
prompt += "\n LLM Answer: The capital of Italy is Rome."

judge_answer = llm.invoke(prompt)
```

**Judge output (structured JSON):**

```json
{"answer": "Rome"}
```

**4. Verification step:**

```python
populated_answer = Answer(**judge_answer)
result = populated_answer.verify()  # True
```

---

## Why Templates?

Templates play a central role in Karenina by standardizing how answers are parsed, verified, and evaluated. Their use provides several key benefits:

### 1. Unified Parsing and Evaluation

Templates allow parsing to happen **directly through the judge LLM**. The free-text answer is mapped into a structured format, ensuring that:

- Evaluation logic is **bundled with the question-answer pair**
- The same benchmark can accommodate **different answer formats** without custom code

### 2. Streamlined Benchmark Creation

Since LLMs are proficient at code generation, they can **auto-generate Pydantic classes** from raw question-answer pairs. This means large portions of benchmark creation can be automated, reducing manual effort while improving consistency.

### 3. Cognitive Offloading for the Judge

By embedding the evaluation schema in templates, the **judge LLM's task is simplified**. Instead of reasoning about both content and evaluation logic, the judge focuses only on interpreting the free-text answer and filling in the template.

### 4. Extensibility and Reusability

Templates make it straightforward to extend benchmarks:

- New tasks can be added by defining new templates
- The same evaluation logic can be reused across multiple benchmarks

### 5. Transparency and Debuggability

By encoding evaluation criteria into explicit, inspectable templates, benchmarks become more transparent. This allows developers to:

- **Audit** evaluation rules directly
- **Debug** failures by inspecting structured outputs

---

## Beyond Templates: Rubrics for Quality Assessment

While templates excel at verifying **factual correctness**, many evaluation scenarios require assessing **qualitative traits**, format compliance, or quantitative metrics. This is where **rubrics** come in.

Karenina rubrics support **three types of traits**, each suited for different evaluation needs:

### 1. LLM-Based Traits

AI-evaluated qualitative assessments where a Judge LLM evaluates subjective qualities of answers.

**Examples:**
- **Score-based (1-5):** "Rate the conciseness of the answer" or "How clear is the explanation?"
- **Binary (pass/fail):** "Does the answer mention safety concerns?" or "Is the response complete?"

### 2. Regex Pattern Traits

Deterministic validation using regular expressions for format compliance and keyword detection.

**Examples:**

- "Answer must include a valid email format"
- "Answer should not contain URLs" (inverted matching)
- "Must mention specific technical terms"

### 3. Metric-Based Traits

Quantitative evaluation using **confusion matrix metrics**. You specify lists of terms that should or should not appear in the answer, and the system extracts these terms to compute standard metrics like precision, recall, and F1 score.

**How it works:**

- Define **True Positive (TP) instructions**: Terms that SHOULD be present and correctly identified
- Define **False Positive (FP) instructions**: Terms that SHOULD NOT appear (incorrect terms in same domain)
- The system extracts term mentions from the answer and computes metrics based on what was found

**Example:**

- "Identify inflammatory lung diseases": Specify correct diseases (asthma, bronchitis, pneumonia) as TP instructions and incorrect diseases (emphysema, pulmonary fibrosis) as FP instructions
- Metrics computed: precision (accuracy of mentioned diseases), recall (coverage of expected diseases), F1 score

> **Learn more:** See the [Rubrics User Guide](using-karenina/rubrics.md) for detailed examples, configuration options, and best practices for all three trait types.

---

## Feature Comparison: Templates vs Rubrics

Karenina uses **two complementary evaluation units** to provide comprehensive assessment:

| Aspect | Answer Templates | Rubrics |
|--------|-----------------|---------|
| **Purpose** | Verify factual correctness | Assess qualitative traits, format, and metrics |
| **Evaluation Method** | Programmatic field comparison | Three approaches:<br>• LLM judgment<br>• Regex patterns<br>• Term extraction + metrics |
| **Best for** | Precise, unambiguous answers | Subjective qualities, format validation, quantitative analysis |
| **Trait Types** | Single verification method | **Three types:**<br>• LLM-based (qualitative)<br>• Regex-based (format)<br>• Metric-based (term extraction) |
| **Output** | Pass/fail per field | • Boolean (binary traits)<br>• Scores 1-5 (score traits)<br>• Precision/Recall/F1 (metric traits) |
| **Examples** | `"BCL2"`, `"46 chromosomes"` | • "Is the answer concise?" (LLM)<br>• Match email pattern (regex)<br>• Extract diseases for F1 score (metric) |
| **Scope** | Per question | Global or per question |
| **Automation** | Auto-generated from Q&A pairs | Reusable across benchmarks |

**When to use what:**

- Use **templates** when you need to verify specific factual content or structured data
- Use **LLM-based rubrics** for subjective quality assessment (clarity, conciseness, tone)
- Use **regex rubrics** for format compliance and deterministic keyword checks
- Use **metric rubrics** when evaluating classification accuracy by extracting and measuring term coverage
- Use **both together** for comprehensive evaluation covering correctness AND quality

[Learn more about Templates →](using-karenina/templates.md) | [Learn more about Rubrics →](using-karenina/rubrics.md)
