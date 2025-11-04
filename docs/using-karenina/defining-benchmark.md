# Defining a Benchmark

The `Benchmark` class is the core component of Karenina. This page explains what it is, how to create benchmarks, what metadata can be associated with them, and how to persist them.

## The Benchmark Class

The `Benchmark` class is the central orchestrator for all benchmarking activities in Karenina. It:

- **Manages collections of questions** and their associated templates
- **Coordinates verification workflows** using LLM-as-a-judge patterns
- **Handles serialization and persistence** through JSON-LD checkpoints
- **Provides a unified interface** for benchmark creation, execution, and analysis

Think of a benchmark as a structured container that brings together questions, evaluation templates, and execution configuration into a cohesive evaluation framework.

## How to Create a Benchmark

### Basic Creation

Create a benchmark using the `Benchmark.create()` method:

```python
from karenina import Benchmark

# Create a basic benchmark
benchmark = Benchmark.create(
    name="Genomics Knowledge Benchmark"
)
```

### Creation with Metadata

You can attach rich metadata to provide context and organization:

```python
from karenina import Benchmark

benchmark = Benchmark.create(
    name="Genomics Knowledge Benchmark",
    description="Testing LLM knowledge of genomics and molecular biology",
    version="1.0.0",
    creator="Dr. Jane Smith",
    keywords=["genomics", "molecular-biology", "biology"],
    license="MIT"
)
```

**Key Parameters:**

- **`name`** (required): Unique identifier for the benchmark
- **`description`**: Human-readable explanation of the benchmark's purpose
- **`version`**: Version string for tracking benchmark evolution (e.g., "1.0.0")
- **`creator`**: Name or organization that created the benchmark
- **`keywords`**: List of searchable tags for categorization
- **`license`**: License type (e.g., "MIT", "CC-BY-4.0")

## Benchmark Metadata Attributes

### Standard Metadata

The following standard attributes can be set when creating a benchmark:

| Attribute | Type | Description |
|-----------|------|-------------|
| `name` | `str` | Unique identifier for the benchmark (required) |
| `description` | `str` | Human-readable description of the benchmark's purpose |
| `version` | `str` | Version string for tracking benchmark evolution |
| `creator` | `str` | Creator or maintainer of the benchmark |
| `keywords` | `List[str]` | Searchable tags for categorization |
| `license` | `str` | License type (e.g., "MIT", "CC-BY-4.0") |

### Accessing Metadata

```python
# Access standard attributes
print(benchmark.name)  # "Genomics Knowledge Benchmark"
print(benchmark.description)  # "Testing LLM knowledge of..."
print(benchmark.version)  # "1.0.0"
print(benchmark.keywords)  # ["genomics", "molecular-biology", "biology"]
```

## Benchmark Organization Patterns

### Domain-Specific Benchmarks

Organize benchmarks by domain to facilitate comparison and reuse:

```python
# Molecular biology benchmark
molecular_bio_benchmark = Benchmark.create(
    name="Molecular Biology Fundamentals",
    description="Tests understanding of core molecular biology concepts",
    version="1.0.0",
    creator="Biology Education Team",
    keywords=["molecular-biology", "genetics", "proteins", "dna"],
    license="CC-BY-4.0"
)

# Pharmacology benchmark
pharmacology_benchmark = Benchmark.create(
    name="Drug Mechanisms and Targets",
    description="Evaluates knowledge of drug targets and mechanisms of action",
    version="1.0.0",
    creator="Pharmacology Research Group",
    keywords=["pharmacology", "drug-targets", "clinical"],
    license="CC-BY-4.0"
)
```

### Multi-Version Benchmarks

Track benchmark evolution by versioning:

```python
# Version 1.0: Basic genomics questions
genomics_v1 = Benchmark.create(
    name="Genomics Knowledge Benchmark",
    version="1.0.0",
    description="Basic genomics questions covering chromosomes and DNA structure",
    creator="Dr. Jane Smith",
    keywords=["genomics", "basic"]
)

# Version 2.0: Expanded with advanced topics
genomics_v2 = Benchmark.create(
    name="Genomics Knowledge Benchmark",
    version="2.0.0",
    description="Expanded genomics benchmark including epigenetics and gene regulation",
    creator="Dr. Jane Smith",
    keywords=["genomics", "advanced", "epigenetics"]
)
```

## Database Persistence

Karenina provides SQLite database storage for persistent benchmark management.

### Save to Database

Save your benchmark to a database with an optional checkpoint file:

```python
from pathlib import Path

# Save to SQLite database
benchmark.save_to_db(
    storage="sqlite:///benchmarks.db",
    checkpoint_path=Path("genomics_benchmark.jsonld")
)
```

**Parameters:**

- **`storage`**: Database connection string (e.g., `"sqlite:///benchmarks.db"`)
- **`checkpoint_path`** (optional): Path to save a checkpoint file alongside the database entry

**What gets stored:**

- Benchmark metadata (name, description, version)
- All questions with their metadata
- Answer templates
- Rubrics (global and question-specific)
- Verification results (if available)

### Load from Database

Load a previously saved benchmark by name:

```python
from karenina import Benchmark

# Load from database
loaded_benchmark = Benchmark.load_from_db(
    benchmark_name="Genomics Knowledge Benchmark",
    storage="sqlite:///benchmarks.db"
)

print(f"Loaded {len(loaded_benchmark.questions)} questions")
```

**Parameters:**

- **`benchmark_name`**: Exact name of the benchmark to load
- **`storage`**: Database connection string

### Database Use Cases

**Version Control:**
Store multiple versions of the same benchmark with different version strings:

```python
# Save v1.0
benchmark_v1.save_to_db(storage="sqlite:///benchmarks.db")

# Later, save v2.0 with the same name but different version
benchmark_v2.save_to_db(storage="sqlite:///benchmarks.db")
```

**Shared Storage:**
Multiple team members can access the same database to collaborate on benchmarks.

**Automatic Verification Persistence:**
When you run verification, results are automatically saved to the database if you provide a `storage` parameter in your `VerificationConfig`.

---

## Checkpoint Files

Checkpoints are JSON-LD files that contain the complete state of a benchmark. Unlike database storage, checkpoints are portable files that can be easily shared, version-controlled, and inspected.

### Save Checkpoint

Save your benchmark to a JSON-LD checkpoint file:

```python
from pathlib import Path

# Save checkpoint
benchmark.save("genomics_benchmark.jsonld")

# Or use explicit method
benchmark.save_checkpoint(Path("checkpoints/genomics_benchmark.jsonld"))
```

### Load Checkpoint

Load a benchmark from a checkpoint file:

```python
from karenina import Benchmark

# Load from checkpoint
benchmark = Benchmark.load("genomics_benchmark.jsonld")

print(f"Loaded benchmark: {benchmark.name}")
print(f"Total questions: {len(benchmark.questions)}")
```

### Checkpoint Format

Checkpoints use JSON-LD format following schema.org conventions:

```json
{
  "@context": "https://schema.org/",
  "@type": "Dataset",
  "name": "Genomics Knowledge Benchmark",
  "description": "Testing LLM knowledge of genomics and molecular biology",
  "version": "1.0.0",
  "creator": {"@type": "Person", "name": "Dr. Jane Smith"},
  "keywords": ["genomics", "molecular-biology", "biology"],
  "license": "MIT",
  "hasPart": [
    {
      "@type": "Question",
      "text": "How many chromosomes are in a human somatic cell?",
      "acceptedAnswer": {
        "@type": "Answer",
        "text": "46"
      }
    }
  ]
}
```

### Checkpoint Use Cases

**Sharing Benchmarks:**
Send checkpoint files to collaborators or publish them in repositories.

**Version Control:**
Track checkpoint files in Git to monitor benchmark evolution over time.

**Portability:**
Move benchmarks between systems without database dependencies.

**Inspection:**
Open checkpoint files in text editors to review benchmark structure.

---

## Next Steps

Once you have a benchmark defined, you can:

- [Add questions](adding-questions.md) to populate it with evaluation content
- [Set up templates](templates.md) for structured evaluation
- [Configure verification](verification.md) to run assessments
- [Save and load](saving-loading.md) benchmarks using checkpoints or database

---

## Related Documentation

- [Adding Questions](adding-questions.md) - Load questions from files or add programmatically
- [Saving & Loading](saving-loading.md) - Complete guide to persistence options
- [Verification](verification.md) - Run evaluations and store results
- [Quick Start](../quickstart.md) - End-to-end workflow example
