# Karenina Features

Karenina is a comprehensive benchmarking system for Large Language Models (LLMs) designed to make domain-specific benchmark creation accessible to experts without requiring LLM-technical infrastructure knowledge.

**Core challenge**: Standardizing domain expertise into runnable benchmarks through **templates** (verify factual correctness) and **rubrics** (assess qualitative traits, format compliance, and metrics).

**Architecture**: Standalone Python library with optional GUI integration for no-code accessibility.

This page provides an overview of all major features available in the karenina Python library.

## Quick Navigation

**Core Features**: [Question Management](#question-management) | [Answer Templates](#answer-template-generation) | [Verification](#benchmark-verification) | [Rubrics](#rubric-evaluation)

**Advanced Evaluation**: [Few-Shot Prompting](#few-shot-prompting) | [Deep-Judgment](#deep-judgment-parsing) | [Abstention Detection](#abstention-detection) | [Embedding Check](#embedding-check-semantic-fallback) | [Regex Validation](#regex-validation)

**Data Management**: [Database](#database-persistence) | [Checkpoints](#checkpoint-management) | [Export & Reporting](#export--reporting)

**Configuration**: [Settings](#configuration-management) | [Presets](#preset-management) | [MCP Integration](#mcp-integration) | [Manual Trace Upload](#manual-trace-upload)

**Other**: [Integration with Server & GUI](#integration-with-server-gui) | [Feature Comparison](#feature-comparison-templates-vs-rubrics) | [Workflow Overview](#workflow-overview) | [Next Steps](#next-steps)

---

## Core Workflow Features

### Question Management
Extract and manage benchmark questions with rich metadata support.

**What you can do:**

- Load questions from files (Excel, CSV, TSV)
- Map columns to question text, answers, and metadata
- Add questions programmatically
- Organize questions with keywords and author information
- Validate question data

**Learn more:** [Adding Questions](using-karenina/adding-questions.md)

---

### Answer Template Generation
Create structured evaluation templates that define what correct answers should contain.

**What templates are:**
Answer templates are Pydantic models that specify how to evaluate model outputs. They define what information should be extracted from answers and how to verify correctness programmatically.

**What you can do:**

- Generate templates automatically using LLMs
- Create templates for multiple questions in batches
- Define precise attribute types (strings, numbers, booleans, lists)
- Specify custom verification logic
- Include regex patterns for format enforcement

**Learn more:** [Templates](using-karenina/templates.md)

---

### Benchmark Verification
Evaluate LLM responses against your templates and rubrics.

**Supported Interfaces:**

- **`langchain`**: access all of the supported LLM providers via LangChain
- **`openrouter`**: access any model from the OpenRouter platform
- **`openai_endpoint`**: OpenAI-compatible endpoints
- **`manual`**: Manual trace replay for testing/debugging (no API calls)

**What you can do:**

- Run verification with any supported interface and provider
- Use different models for answering vs parsing (judge)
- Run multiple replications for statistical significance
- Enable optional advanced features (deep-judgment, abstention detection, embedding check)
- Track progress during verification
- Cache answers automatically for efficiency
- Save results to database

**Learn more:** [Verification](using-karenina/verification.md)

---

### Rubric Evaluation
Assess qualitative aspects of answers using LLM-based, regex-based, or metric-based criteria.

**What rubrics are:**

Rubrics define evaluation criteria for subjective qualities like clarity, safety, conciseness, or compliance with specific formats. Unlike templates (which verify factual correctness), rubrics assess answer quality.

**Three trait types:**

- **LLM-based traits**: AI evaluates qualities (binary pass/fail or 1-5 scale)
- **Regex-based traits**: Pattern matching for format validation
- **Metric-based traits**: Confusion matrix metrics (precision, recall, F1, accuracy)

**What you can do:**

- Create global rubrics applied to all questions
- Create question-specific rubrics
- Use LLM traits for subjective assessment (clarity, safety, etc.)
- Use regex traits for format validation (required patterns, prohibited content)
- Use metric traits for classification accuracy (TP/FP/TN/FN evaluation)
- Combine multiple trait types in a single rubric
- Export and import rubric configurations

**Learn more:** [Rubrics](using-karenina/rubrics.md)

---

## Advanced Evaluation Features

### Few-Shot Prompting
Guide LLM responses by providing examples.

**What you can do:**

- Configure few-shot examples globally or per question
- Use "all" mode (include all available examples)
- Use "k-shot" mode (include k random examples)
- Use "custom" mode (specify exact examples)
- Select examples by index or MD5 hash
- Ensure reproducible randomization

**Learn more:** [Advanced: Few-Shot Prompting](advanced/few-shot.md)

---

### Deep-Judgment Parsing
Extract detailed feedback with evidence, reasoning, and confidence scores.

**What it does:**

Deep-judgment uses a multi-stage parsing pipeline to extract specific excerpts from answers, reasoning traces explaining the evaluation, and confidence scores. This provides much richer feedback than standard pass/fail verification.

**What you can do:**

- Enable deep-judgment parsing in verification config
- Get verbatim excerpts with confidence scoring
- Access reasoning traces explaining why answers passed/failed
- Detect hallucinated evidence with fuzzy matching
- Auto-fail answers when required evidence is missing

**Learn more:** [Advanced: Deep-Judgment](advanced/deep-judgment.md)

---

### Abstention Detection
Identify when models refuse to answer questions.

**What it does:**

Detects patterns like "I don't know", "I cannot answer", or other refusals to provide information. When detected, the verification result is overridden and the abstention reasoning is stored.

**What you can do:**

- Enable via environment variable: `ABSTENTION_CHECK_ENABLED=true`
- Automatically detect common abstention patterns
- Store abstention reasoning for analysis
- Override verification results when abstentions are detected

**Learn more:** [Advanced: Abstention Detection](advanced/abstention-detection.md)

---

### Embedding Check (Semantic Fallback)
Catch semantically correct answers that fail format checks.

**What it does:**
When verification fails, embedding check uses semantic similarity (via SentenceTransformer) to determine if the answer is semantically equivalent to the expected answer. This prevents false negatives from format differences.

**What you can do:**

- Enable via environment variable: `EMBEDDING_CHECK=true`
- Configure similarity threshold and model
- Automatically override false negatives to true positives
- Combine with template verification for robust evaluation

**Trade-offs:**

- Only triggers on failed verifications
- Zero performance impact when disabled

**Learn more:** [Advanced: Embedding Check](advanced/embedding-check.md)

---

### Regex Validation
Validate answer formats with pattern matching.

**What it does:**

Check raw LLM response text against regex patterns to enforce specific formats (e.g., "answers must be enclosed in brackets").

**What you can do:**

- Define patterns in template fields
- Use match types: exact, contains, count, all
- Combine regex results with template verification
- Get detailed match reporting for debugging

**Learn more:** [Rubrics](using-karenina/rubrics.md#2-regex-based-traits-manualrubrictrait)

---

## Data Management Features

### Database Persistence
Store benchmarks, questions, and verification results in SQLite.

**What you can do:**

- Auto-save verification results to database
- Track verification run history
- Version questions with template_id
- List all saved benchmarks
- Load benchmarks by ID
- Delete old benchmarks

**Learn more:** [Database](using-karenina/defining-benchmark.md#database-persistence)

---

### Checkpoint Management
Save and load complete benchmark state as JSON-LD files.

**What checkpoints contain:**

- All questions with metadata
- Answer templates
- Rubrics (global and question-specific)
- Benchmark metadata
- Question versioning information

**What you can do:**

- Save benchmark state: `benchmark.save_checkpoint(path)`
- Load benchmark state: `Benchmark.load_checkpoint(path)`
- Share benchmarks across systems
- Version control your benchmarks

**Learn more:** [Saving & Loading](using-karenina/saving-loading.md)

---

### Export & Reporting
Export verification results and benchmarks in multiple formats.

**What you can do:**

- Export to JSON (complete data structure)
- Export to CSV (tabular format for analysis)
- Configure which columns to include
- Filter results before export
- Include/exclude rubric evaluation columns
- Export checkpoint files (JSON-LD)

**Learn more:** [Saving & Loading](using-karenina/saving-loading.md#exporting-verification-results)

---

## Configuration & Integration

### Configuration Management
Customize Karenina's behavior with environment variables and programmatic settings.

**What you can configure:**

- Default LLM model: `KARENINA_DEFAULT_MODEL`
- API keys: `OPENAI_API_KEY`, `ANTHROPIC_API_KEY`
- Database location: `KARENINA_DB_PATH`
- Feature flags: `EMBEDDING_CHECK`, `ABSTENTION_CHECK_ENABLED`
- Model temperature defaults
- Verification parameters (max_tokens, timeout, etc.)

**Learn more:** [Configuration](configuration.md)

---

### Preset Management
Save and reuse verification configurations across benchmarks.

**What presets contain:**

- Model configurations (answering, parsing, judging)
- Enabled features (deep-judgment, abstention, embedding check)
- Rubric settings
- Few-shot configuration
- MCP integration settings

**What you can do:**

- Save current config as named preset
- Load saved presets to restore configurations
- Update preset metadata
- Share presets across projects
- Use presets programmatically or via GUI

**Learn more:** [Advanced: Presets](advanced/presets.md)

---

### MCP Integration
Use Model Context Protocol servers for tool integration.

**What you can do:**

- Configure MCP servers for verification runs
- Support multiple MCP servers simultaneously
- Validate MCP server configurations
- Enable tool use in LLM responses

**Learn more:** [Advanced: MCP Integration](advanced/mcp-integration.md)

---

### Manual Trace Upload
Replay pre-recorded LLM responses for testing and debugging using the `manual` interface.

**What you can do:**

- Upload JSON-based trace files
- Bypass live LLM calls for reproducibility (uses `manual` interface, no API costs)
- Debug parsing issues without making real API calls
- Create regression tests with recorded responses
- Reproduce specific scenarios deterministically

**Learn more:** [Advanced: Manual Traces](advanced/manual-traces.md)

---

## Integration with Server & GUI

**Karenina as a standalone library:**
Everything documented here works independently as a Python library. You can use karenina programmatically in scripts, notebooks, or applications.

**Graphical User Interface (Optional):**
To guarantee **additional accessibility** to the framework, a **web-based graphical interface** is available for domain experts, curators, and non-technical users who prefer not to work with code.

**What the GUI provides:**

- **Visual question and metadata extraction** from files (Excel, CSV, TSV)
- **Template generation** with interactive preview and editing
- **No-code rubric curation** (LLM-based, regex, and metric traits)
- **Checkpointing and verification execution** with real-time progress monitoring
- **Results visualization** and export management

**Implementation:**
The GUI is built using two companion packages:

- **[karenina-server](https://github.com/biocypher/karenina-server)**: FastAPI-based REST API wrapper exposing karenina backend
- **[karenina-gui](https://github.com/biocypher/karenina-gui)**: TypeScript/React web application providing the user interface

**Note**: Coordination and deployment instructions for the full web-based stack are still a work in progress and will be released soon.

**Learn more:** [Advanced: Integration](advanced/integration.md)

---

## Feature Comparison: Templates vs Rubrics

| Aspect | Answer Templates | Rubrics |
|--------|-----------------|---------|
| **Purpose** | Verify factual correctness | Assess qualitative traits |
| **Evaluation** | Programmatic comparison | LLM judgment, regex, or metrics |
| **Best for** | Precise, unambiguous answers | Subjective qualities, format validation, classification metrics |
| **Trait Types** | N/A | LLM-based, regex-based, metric-based |
| **Output** | Pass/fail per field | Boolean, score (1-5), or metrics (precision/recall/F1) |
| **Examples** | "BCL2", "46 chromosomes", "True" | "Is the answer concise?", "Must match pattern", "Computed metrics: precision/recall/F1" |
| **Scope** | Per question | Global or per question |

---

## Workflow Overview

The typical Karenina workflow:

```
1. Create Benchmark
   ↓
2. Add Questions (from files or programmatically)
   ↓
3. Generate Templates (automated or manual)
   ↓
4. Create Rubrics (optional, for qualitative assessment)
   ↓
5. Run Verification (with chosen models and config)
   ↓
6. Analyze Results (export, filter, compare)
   ↓
7. Save Checkpoint (for reproducibility)
```

---

## Next Steps

**New to Karenina?**
- Start with [Quick Start](quickstart.md)
- Read [Core Concepts](index.md) for philosophy
- Follow [Defining Benchmarks](using-karenina/defining-benchmark.md) guide

**Ready to evaluate?**
- Set up [Configuration](configuration.md) with API keys
- Create [Templates](using-karenina/templates.md)
- Run [Verification](using-karenina/verification.md)

**Need advanced features?**
- Enable [Deep-Judgment](advanced/deep-judgment.md)
- Configure [Few-Shot](advanced/few-shot.md) examples
- Use [Presets](advanced/presets.md) for reproducibility

**Reference:**
- [API Reference](api-reference.md) for complete method documentation
- [Troubleshooting](troubleshooting.md) for common issues

---

[← Back to Documentation Index](index.md)
