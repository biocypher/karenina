# Defining a Benchmark

!!! warning "Under Construction"
    This section is currently under development. Some features and documentation may be incomplete or subject to change.


The `Benchmark` class is the core component of Karenina. This page explains what it is, how to create benchmarks, and what metadata can be associated with them.

## The Benchmark Class

The `Benchmark` class is the central orchestrator for all benchmarking activities in Karenina. It:

- **Manages collections of questions** and their associated templates
- **Coordinates verification workflows** using LLM-as-a-judge patterns
- **Handles serialization and persistence** through JSON-LD checkpoints
- **Provides a unified interface** for benchmark creation, execution, and analysis

Think of a benchmark as a structured container that brings together questions, evaluation templates, and execution configuration into a cohesive evaluation framework.

## How to Create a Benchmark

### Basic Creation

Create a benchmark using the simple constructor:

```python
from karenina import Benchmark

# Create a basic benchmark
benchmark = Benchmark(name="my-benchmark")
```

### Advanced Creation with Metadata

You can attach rich metadata to provide context and organization:

```python
from karenina import Benchmark

benchmark = Benchmark(
    name="medical-diagnosis-benchmark",
    description="Evaluates LLM performance on medical diagnostic scenarios",
    version="1.0.0",
    author="Dr. Jane Smith",
    tags=["medical", "diagnosis", "clinical-reasoning"],
    metadata={
        "domain": "healthcare",
        "target_models": ["gpt-4", "claude-3"],
        "difficulty_level": "expert",
        "created_date": "2024-01-15"
    }
)
```

## Benchmark Metadata Attributes

### Standard Metadata

The following standard attributes can be set on any benchmark:

| Attribute | Type | Description |
|-----------|------|-------------|
| `name` | `str` | Unique identifier for the benchmark |
| `description` | `str` | Human-readable description of the benchmark's purpose |
| `version` | `str` | Version string for tracking benchmark evolution |
| `author` | `str` | Creator or maintainer of the benchmark |
| `tags` | `List[str]` | Searchable tags for categorization |
| `metadata` | `Dict[str, Any]` | Custom key-value pairs for extended attributes |

### Accessing Metadata

```python
# Access standard attributes
print(benchmark.name)
print(benchmark.description)
print(benchmark.tags)

# Access custom metadata
domain = benchmark.metadata.get("domain")
difficulty = benchmark.metadata.get("difficulty_level")
```

### Modifying Metadata

```python
# Update standard attributes
benchmark.description = "Updated description"
benchmark.tags.append("updated")

# Update custom metadata
benchmark.metadata["last_modified"] = "2024-02-01"
benchmark.metadata["reviewer"] = "Dr. John Doe"
```

## Benchmark Organization Patterns

### Domain-Specific Benchmarks

```python
# Legal domain benchmark
legal_benchmark = Benchmark(
    name="legal-reasoning-v2",
    description="Tests legal reasoning and case analysis",
    tags=["legal", "reasoning", "case-law"],
    metadata={
        "jurisdiction": "US",
        "legal_areas": ["contract-law", "tort-law"],
        "complexity": "intermediate"
    }
)
```

### Multi-Version Benchmarks

```python
# Versioned benchmark series
benchmark_v1 = Benchmark(
    name="math-reasoning",
    version="1.0",
    metadata={"baseline": True}
)

benchmark_v2 = Benchmark(
    name="math-reasoning",
    version="2.0",
    metadata={"improvements": ["harder-problems", "better-rubrics"]}
)
```

## Next Steps

Once you have a benchmark defined, you can:

- [Add questions](adding-questions.md) to populate it with evaluation content
- [Set up templates](templates.md) for structured evaluation
- [Configure verification](verification.md) to run assessments
