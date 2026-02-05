# LLM Evaluation Philosophy

This page explains the evaluation approach behind Karenina: why it uses LLMs as judges, why structured evaluation matters, and how it handles the reality of free-form language model outputs.

## The Evaluation Challenge

When evaluating LLM responses, a fundamental tension exists between **flexibility** and **reliability**:

- **Flexible evaluation** (human review, free-form LLM judgment) handles diverse outputs but is expensive, slow, and hard to reproduce.
- **Rigid evaluation** (exact string matching, regex) is fast and reproducible but breaks on the natural, varied language that LLMs produce.

Consider a simple question: *"What protein regulates apoptosis?"*

An LLM might answer:

- `"BCL2"`
- `"The BCL-2 protein regulates apoptosis."`
- `"BCL2 is an anti-apoptotic protein that prevents programmed cell death."`

All three are correct, but only the first would pass a naive string match against `"BCL2"`. The others require understanding of natural language to evaluate — and this is where Karenina's approach comes in.

## The LLM-as-Judge Approach

Karenina uses a **Judge LLM** to bridge the gap between free-form responses and structured evaluation. The key insight: instead of constraining how the answering model responds, constrain how the *judge* responds.

### How It Works

1. The **answering model** generates a natural, unconstrained response
2. A **Judge LLM** parses that response into a structured format (a Pydantic schema)
3. A programmatic **`verify()` method** checks the structured output against ground truth

The judge's task is deliberately narrow: given a response and a schema, fill in the schema fields. This is a much simpler task than open-ended evaluation, making it reliable across different judge models.

### Why Not Just Constrain the Answering Model?

A common alternative is to force the answering model into a rigid format ("respond with only the letter A, B, C, or D"). This has significant drawbacks:

- **Fragile compliance** — Models don't always follow format instructions, especially under complex reasoning
- **Unnatural evaluation** — Real-world usage rarely involves rigid formats; benchmarks should reflect how models are actually used
- **Lost signal** — Constrained formats discard valuable information about how the model reasons, hedges, or explains

By keeping the answering model unconstrained, Karenina evaluates models in conditions that match real deployment.

### Why Not Pure LLM Judgment?

Using an LLM judge to directly answer "is this response correct?" (without structured output) also has problems:

- **Inconsistency** — The same judge may give different assessments across runs
- **Opacity** — No clear audit trail of what the judge examined
- **Parsing difficulty** — The judge's own free-text verdict must be interpreted

Karenina solves this by having the judge fill in a **structured schema** rather than produce free-text verdicts. The schema becomes both the evaluation rubric and the audit trail.

## Why Structured Evaluation Matters

### Reproducibility

Because evaluation criteria are encoded as executable Python code (Pydantic models with `verify()` methods), benchmarks are **fully reproducible**. The same template evaluates answers identically across different runs, systems, and team members. There is no ambiguity about what "correct" means — it's defined in code.

### Systematic Comparison

Structured evaluation enables meaningful comparison across models:

- Run the **same benchmark** against multiple LLMs
- Get **consistent metrics** because the evaluation criteria don't change between runs
- **Track performance over time** as models are updated or fine-tuned

Without structure, every evaluation is an island — results from different sessions can't be meaningfully compared.

### Transparency and Debuggability

When a model fails a verification, structured output makes it straightforward to understand *why*:

- Inspect the judge's **parsed output** — what did it extract from the response?
- Compare against the **expected values** defined in the template
- Identify whether the issue was in the model's response, the judge's parsing, or the template's verification logic

This is far more actionable than a binary "the LLM judge said it was wrong."

## Natural Language Flexibility

Karenina embraces the reality that LLMs produce free-form text. Rather than fighting this with rigid output constraints, the framework works *with* natural language.

### Evaluating Unconstrained Outputs

The answering model is never asked to conform to a special format. It can:

- Explain its reasoning at length
- Provide caveats and qualifications
- Use domain-specific terminology naturally
- Structure its response however it sees fit

The Judge LLM handles the translation from free-form text to structured data. This means benchmarks evaluate models as they would actually be used in production.

### Complementary Evaluation Strategies

Not every evaluation needs LLM judgment. Karenina supports multiple strategies that work together:

- **LLM-as-judge** (templates) — For correctness verification requiring natural language understanding
- **Regex patterns** — For format compliance and deterministic keyword detection
- **Custom Python functions** — For arbitrary programmatic checks
- **Metric computation** — For precision, recall, and F1 over extracted terms

The right strategy depends on what you're evaluating. A benchmark might use templates to verify factual accuracy, regex traits to enforce citation format, and LLM rubric traits to assess explanation quality — all on the same set of questions.

## Separation of Concerns

Karenina maintains clear boundaries between three roles:

| Role | Responsibility |
|------|----------------|
| **Answering model** | Generate responses (the model being evaluated) |
| **Judge model** | Parse responses into structured data |
| **Template** | Define evaluation criteria and verify correctness |

This separation means each component can be independently configured, tested, and improved. You can change the judge model without rewriting templates, or add new templates without reconfiguring the judge.

---

## Learn More

- [Templates vs Rubrics](template-vs-rubric.md) — Understand the two complementary evaluation units
- [Answer Templates](../04-core-concepts/answer-templates.md) — Deep dive into template structure, `verify()`, and field types
- [Evaluation Modes](../04-core-concepts/evaluation-modes.md) — How template-only, rubric-only, and combined modes work

**Back to**: [Introduction](index.md)
