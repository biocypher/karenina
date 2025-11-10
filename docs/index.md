# Karenina Documentation

Welcome to the Karenina documentation. Karenina is a Python library for systematic LLM benchmarking with structured, reproducible evaluation.

---

## What is Karenina?

Karenina is a framework for defining benchmarks in a rigorous and reproducible way. With Karenina, benchmarks can be created, shared, reviewed, and executed across a wide range of large language models (LLMs).

**Core Philosophy**: Enable domain experts to create benchmarks without deep LLM-technical expertise, allowing them to focus on knowledge rather than infrastructure.

**Key Capabilities**:

- Create benchmarks from scratch or existing question sets
- Define precise evaluation criteria using code-based templates
- Evaluate answers using both rule-based and LLM-as-judge strategies
- Support natural free-form responses and constrained formats
- Track performance across multiple models and configurations
- Share and reproduce benchmark results

---

## Quick Navigation

### 🚀 Getting Started
- **[Installation](install.md)** - Set up Karenina on your system
- **[Quick Start](quickstart.md)** - Create your first benchmark in 10 minutes
- **[Features Overview](features.md)** - Complete feature catalog

### 📖 User Guides
- **[Defining Benchmarks](using-karenina/defining-benchmark.md)** - Benchmark creation and metadata
- **[Adding Questions](using-karenina/adding-questions.md)** - File extraction and management
- **[Templates](using-karenina/templates.md)** - Creating and customizing answer templates
- **[Rubrics](using-karenina/rubrics.md)** - Evaluation criteria and trait types
- **[Verification](using-karenina/verification.md)** - Running evaluations and analyzing results
- **[CLI Verification](using-karenina/cli-verification.md)** - Command-line interface for automation
- **[Saving & Loading](using-karenina/saving-loading.md)** - Checkpoints, database, and export

### 🔬 Advanced Features
- **[Deep-Judgment](advanced/deep-judgment.md)** - Extract detailed feedback with excerpts
- **[Few-Shot Prompting](advanced/few-shot.md)** - Guide responses with examples
- **[Abstention Detection](advanced/abstention-detection.md)** - Handle model refusals
- **[Embedding Check](advanced/embedding-check.md)** - Semantic similarity fallback
- **[Presets](advanced/presets.md)** - Save and reuse verification configurations
- **[System Integration](advanced/integration.md)** - Server and GUI integration

### 📚 Reference
- **[API Reference](api-reference.md)** - Complete API documentation
- **[Configuration](configuration.md)** - Environment variables and defaults
- **[Troubleshooting](troubleshooting.md)** - Common issues and solutions
