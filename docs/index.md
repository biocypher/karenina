# Karenina Documentation

!!! warning "Under Construction"
    The documentation is currently under construction. Some features and documentation may be incomplete or subject to change.

Welcome to the Karenina documentation. Karenina is a comprehensive benchmarking system for Large Language Models (LLMs).

## About

**A few words about what *Karenina* is:**

- A framework designed to standardize domain expertise and concepts into runnable benchmarks. The core challenge Karenina addresses is: *making the formulation of domain-specific benchmarks accessible to non-LLM-technical experts, allowing them to focus their time and expertise on knowledge rather than infrastructure.*
- Benchmarks in Karenina are expressed as **parametrizable code templates**, which can be evaluated with an **LLM-as-a-judge** model to evaluate performance.
- Karenina defines a **standardized schema** (building on existing standards such as *schema.org*) to enable rich, consistent, and extensible benchmark definitions.
- Karenina aims at providing tools to the end users to generate benchmarks **at scale**.
- Karenina also includes **utilities to run and manage benchmarks**, although its primary focus remains on standardization and accessibility rather than execution infrastructure.

At the heart of Karenina is the notion of a **template**. A template describes both the **question/task** to be posed to a model and the **structure of the expected answer**. By enforcing structured outputs, Karenina ensures that benchmarks are not only reproducible but also programmatically evaluable.

## The problem

Let us instroduce how `karenina` approaches the problem of LLM benchmarking by consider a simple example: want to task an LLM with a simple **multiple-choice question**:

```python
question = "What is the capital of Italy?"
possible_answers = ["Rome", "Milan", "Paris", "New York"]
```

When we query a standard LLM, it usually responds in **free text** (e.g., `"I think the answer is Rome, because it is the capital of Italy."`). To evaluate such an answer programmatically we could use the following apporaches:

### 1. Constrain the answering model's output

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

**Model output:**

```
A
```

This output is trivially parseable (`"Rome"`) and can be matched against the expected answer by writing dedicated parsing code.

The main advantage of this approach is its simplicity and reliability: once the model respects the instruction, evaluation can be fully automated with minimal overhead. However, its weakness lies in the fragility of prompt adherence. Many general-purpose LLMs do not always comply with rigid output constraints, especially across diverse domains or when questions become complex. In practice, this means users must design very careful prompts and may still face occasional formatting failures. Morevoer, everytime we have a different answer/question format we may need to come up with different dedicated prompting and parsing strategies.

---

### 2. Use an LLM as a judge (free-text evaluation)

Instead of constraining the answering model, we can keep its output free-form and rely on a **judge LLM** to interpret it.

**Example:**

* **Answering model output:**
  `"The capital of Italy is Rome, of course."`

* **Judge model prompt:**

  ```text
  The following is a student's answer to a multiple-choice question.
  Question: What is the capital of Italy?
  Options: Rome, Milan, Paris, New York.
  Student's answer: "The capital of Italy is Rome, of course."
  Which option does this correspond to? Provide a justification.
  ```

* **Judge model output (free text):**
  `"The student clearly selected Rome, which is correct."`

The advantage here is flexibility: the answering model is free to behave naturally, without tight formatting constraints, which is particularly useful in open-ended or exploratory settings. However, this shifts the ambiguity to the judge's response, which is also often free text. While the judge usually interprets correctly, the result again requires parsing, and subtle differences in wording may cause errors or inconsistencies. Thus, while this strategy increases robustness to different kinds of answers, it does so at the cost of reintroducing unstructured evaluation one step later.


## The Karenina strategy

To reduce ambiguity, `karenina` adopts a **third option** that combines the advantages of both approaches:

* the **answering model** remains unconstrained, generating natural free text;
* the **judge model** is instead required to return results in a **structured format** (e.g., JSON), validated through a `pydantic` class.

This setup allows the judge to flexibly interpret free text while ensuring that its own output remains standardized and machine-readable.


### Example Workflow

**Answering model output (free text):**

```
"The capital of Italy is Rome."
```


**Definition of the `pydantic` class:**

```python
class Answer(BaseAnswer):
    answer: str = Field(description="The name of the city in the response")

    def model_post_init(self, __context):
        self.correct = {"answer": "Paris"}

    def verify(self) -> bool:
        return self.answer == self.correct["answer"]
```

Two key aspects are worth highlighting:

* The `answer` attribute is decorated with a `Field` description, which provides additional context to the judge about what it should extract.
* The `verify` method implements a custom validation logic, comparing the parsed answer against the ground truth.


**Prompting the judge model:**

```python
from langchain_core.output_parsers import PydanticOutputParser

parser = PydanticOutputParser(pydantic_object=Answer)
prompt = parser.get_format_instructions()
prompt += "\n LLM Answer: The capital of Italy is Rome."
```

By leveraging LangChain utilities, the correct formatting instructions are automatically derived from the `pydantic` class itself, ensuring consistency between schema and prompt.


**Judge model output (structured JSON):**

```python
judge_answer = llm.invoke(prompt)
```

```json
{"answer": "Rome"}
```


**Verification step:**

```python
populated_answer = Answer(**judge_answer)
populated_answer.verify()
```

The result of `verify()` indicates whether the extracted answer matches the expected one.



## Why Templates

Templates play a central role in `karenina` by standardizing how answers are parsed, verified, and evaluated. Their use provides several key benefits:

### 1. Unified Parsing and Evaluation

Without templates, each benchmark would require bespoke parsing pipelines downstream of the inference process. By contrast, templates allow parsing to happen **directly through the judge LLM**. The free-text answer from the answering model is mapped into a structured format (e.g., a Pydantic class), ensuring that:

* Evaluation logic is **bundled with the question–answer pair** itself.
* The same benchmark can seamlessly accommodate **different answer formats** without custom code.

### 2. Streamlined Benchmark Creation

Since LLMs are proficient at code generation, they can often **auto-generate Pydantic classes** from raw question–answer pairs. This means that large portions of benchmark creation can be partially automated, reducing manual effort while improving consistency.


### 3. Cognitive Offloading for the Judge

By embedding the evaluation schema in templates, the **Judge LLM's task is simplified**. Instead of reasoning about both the content and the evaluation logic, the judge focuses only on interpreting the free-text answer and filling in the template. This reduces ambiguity, minimizes error, and makes evaluations more robust.


### 4. Extensibility and Reusability

Templates make it straightforward to extend benchmarks:

* New tasks can be added by defining new templates without re-engineering downstream code.
* The same evaluation logic can be reused across multiple benchmarks with minimal adaptation.


### 5. Transparency and Debuggability

By encoding evaluation criteria into explicit, inspectable templates, benchmarks become more transparent. This allows developers to:

* **Audit** the evaluation rules directly.
* **Debug** failures more easily by inspecting the structured outputs rather than opaque free text.

## Getting Started

Ready to get started with Karenina? Check out our documentation:

- [**Install**](install.md) - Set up Karenina on your system
- [**Quick Start**](quickstart.md) - Create your first benchmark in minutes
- [**Using Karenina**](using-karenina/defining-benchmark.md) - Comprehensive guide to all features
- [**API Reference**](api-reference.md) - Complete API documentation
