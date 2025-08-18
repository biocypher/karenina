# Saving and Loading a Benchmark

This guide covers how to persist and restore benchmarks using Karenina's checkpoint system based on JSON-LD format.

## Understanding Checkpoints

Karenina uses **JSON-LD (JSON for Linked Data)** format for benchmark persistence:

- **JSON-LD** provides structured, semantic data representation
- **Checkpoints** capture complete benchmark state including questions, templates, and results
- **Cross-platform compatibility** ensures benchmarks work across different environments
- **Version compatibility** maintains backward compatibility as Karenina evolves

## Saving Benchmarks

### Basic Save Operation

```python
# Save benchmark to default location
benchmark.save("my-benchmark.jsonld")

# Save to specific directory
benchmark.save("/path/to/benchmarks/my-benchmark.jsonld")

# Save with auto-generated filename based on benchmark name
benchmark.save()  # Creates "benchmark-name-YYYYMMDD-HHMMSS.jsonld"
```

### Save with Metadata

```python
# Save with additional context
benchmark.save(
    "my-benchmark.jsonld",
    save_metadata={
        "created_by": "Jane Doe",
        "purpose": "Q1 2024 model evaluation",
        "notes": "Includes updated rubrics for math problems"
    }
)
```

### Incremental Saves / Checkpoints

```python
# Save checkpoint during long-running operations
def run_with_checkpoints(benchmark, model_config):
    # Generate templates
    benchmark.generate_answer_templates(model_config)
    benchmark.save_checkpoint("after-template-generation.jsonld")

    # Run verification
    results = benchmark.run_verification(model_config)
    benchmark.save_checkpoint("after-verification.jsonld")

    # Save final results
    benchmark.save("final-benchmark.jsonld")

    return results
```

### Selective Saving

```python
# Save only specific components
benchmark.save(
    "questions-only.jsonld",
    include_templates=False,     # Skip answer templates
    include_results=False,       # Skip verification results
    include_rubrics=True         # Include rubric definitions
)

# Save only verification results
benchmark.save_results("results-only.jsonld")
```

## Loading Benchmarks

### Basic Load Operation

```python
from karenina import Benchmark

# Load benchmark from file
benchmark = Benchmark.load("my-benchmark.jsonld")

print(f"Loaded benchmark: {benchmark.name}")
print(f"Questions: {len(benchmark.questions)}")
print(f"Has templates: {benchmark.all_questions_have_templates()}")
```

### Loading with Validation

```python
def load_with_validation(filepath):
    """Load benchmark with comprehensive validation"""
    try:
        benchmark = Benchmark.load(filepath)

        # Validate loaded data
        assert len(benchmark.questions) > 0, "No questions found"

        # Check for required metadata
        required_fields = ["name", "description"]
        for field in required_fields:
            assert getattr(benchmark, field), f"Missing required field: {field}"

        print(f"✓ Successfully loaded and validated: {benchmark.name}")
        return benchmark

    except Exception as e:
        print(f"✗ Failed to load benchmark: {str(e)}")
        return None

# Load with validation
benchmark = load_with_validation("my-benchmark.jsonld")
```

### Partial Loading

```python
# Load only metadata (for quick inspection)
metadata = Benchmark.load_metadata("my-benchmark.jsonld")
print(f"Benchmark: {metadata['name']}")
print(f"Questions: {metadata['question_count']}")
print(f"Created: {metadata['created_date']}")

# Load questions without templates (faster)
benchmark = Benchmark.load(
    "my-benchmark.jsonld",
    load_templates=False,
    load_results=False
)
```

## Checkpoint Management

### Creating Regular Checkpoints

```python
class CheckpointManager:
    def __init__(self, benchmark, base_path):
        self.benchmark = benchmark
        self.base_path = base_path
        self.checkpoint_counter = 0

    def checkpoint(self, label=None):
        """Create a checkpoint with optional label"""
        if label:
            filename = f"{self.benchmark.name}-{label}.jsonld"
        else:
            filename = f"{self.benchmark.name}-checkpoint-{self.checkpoint_counter:03d}.jsonld"
            self.checkpoint_counter += 1

        filepath = os.path.join(self.base_path, filename)
        self.benchmark.save(filepath)
        print(f"Checkpoint saved: {filename}")
        return filepath

# Usage
manager = CheckpointManager(benchmark, "checkpoints/")
manager.checkpoint("before-verification")

# Run operations...
results = benchmark.run_verification(model_config)

manager.checkpoint("after-verification")
```

### Checkpoint Comparison

```python
def compare_checkpoints(checkpoint1_path, checkpoint2_path):
    """Compare two benchmark checkpoints"""

    bench1 = Benchmark.load(checkpoint1_path)
    bench2 = Benchmark.load(checkpoint2_path)

    print(f"=== Checkpoint Comparison ===")
    print(f"Checkpoint 1: {checkpoint1_path}")
    print(f"Checkpoint 2: {checkpoint2_path}")
    print()

    # Compare basic metrics
    print(f"Questions: {len(bench1.questions)} vs {len(bench2.questions)}")

    # Compare template status
    bench1_templated = sum(1 for q in bench1.questions if q.answer_template)
    bench2_templated = sum(1 for q in bench2.questions if q.answer_template)
    print(f"Templated questions: {bench1_templated} vs {bench2_templated}")

    # Compare verification results
    bench1_verified = len(bench1.get_verification_results())
    bench2_verified = len(bench2.get_verification_results())
    print(f"Verified questions: {bench1_verified} vs {bench2_verified}")

# Compare checkpoints
compare_checkpoints("before-verification.jsonld", "after-verification.jsonld")
```

## Advanced Persistence Options

### Custom Serialization

```python
# Save with custom serialization options
benchmark.save(
    "custom-format.jsonld",
    compression=True,           # Compress JSON-LD output
    pretty_print=True,          # Human-readable formatting
    include_timestamps=True,    # Add creation/modification times
    embed_schemas=True          # Include schema definitions inline
)
```

### Backup and Recovery

```python
import shutil
from datetime import datetime

class BackupManager:
    def __init__(self, backup_dir="backups"):
        self.backup_dir = backup_dir
        os.makedirs(backup_dir, exist_ok=True)

    def create_backup(self, benchmark, label=None):
        """Create timestamped backup"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        if label:
            filename = f"{benchmark.name}_{label}_{timestamp}.jsonld"
        else:
            filename = f"{benchmark.name}_backup_{timestamp}.jsonld"

        backup_path = os.path.join(self.backup_dir, filename)
        benchmark.save(backup_path)

        return backup_path

    def list_backups(self, benchmark_name):
        """List all backups for a benchmark"""
        backups = []
        for filename in os.listdir(self.backup_dir):
            if filename.startswith(benchmark_name) and filename.endswith(".jsonld"):
                filepath = os.path.join(self.backup_dir, filename)
                stat = os.stat(filepath)
                backups.append({
                    "filename": filename,
                    "path": filepath,
                    "size": stat.st_size,
                    "modified": datetime.fromtimestamp(stat.st_mtime)
                })

        return sorted(backups, key=lambda x: x["modified"], reverse=True)

    def restore_backup(self, backup_path):
        """Restore benchmark from backup"""
        return Benchmark.load(backup_path)

# Usage
backup_manager = BackupManager()

# Create backup before major operations
backup_path = backup_manager.create_backup(benchmark, "before-major-changes")

# List available backups
backups = backup_manager.list_backups(benchmark.name)
for backup in backups:
    print(f"{backup['filename']} - {backup['modified']} ({backup['size']} bytes)")

# Restore from backup if needed
if something_went_wrong:
    benchmark = backup_manager.restore_backup(backup_path)
```

## Migration and Versioning

### Version Compatibility

```python
def check_version_compatibility(filepath):
    """Check if benchmark file is compatible with current Karenina version"""
    try:
        metadata = Benchmark.load_metadata(filepath)
        file_version = metadata.get("karenina_version", "unknown")
        current_version = karenina.__version__

        print(f"File version: {file_version}")
        print(f"Current version: {current_version}")

        # Add version compatibility logic here
        return True  # Simplified for example

    except Exception as e:
        print(f"Cannot determine compatibility: {str(e)}")
        return False

# Check before loading
if check_version_compatibility("old-benchmark.jsonld"):
    benchmark = Benchmark.load("old-benchmark.jsonld")
else:
    print("Benchmark may not be compatible with current version")
```

### Benchmark Merging

```python
def merge_benchmarks(*benchmark_paths, output_path):
    """Merge multiple benchmarks into a single benchmark"""

    benchmarks = [Benchmark.load(path) for path in benchmark_paths]

    # Create merged benchmark
    merged = Benchmark(
        name=f"merged-{datetime.now().strftime('%Y%m%d')}",
        description="Merged benchmark from multiple sources"
    )

    # Combine questions from all benchmarks
    for i, benchmark in enumerate(benchmarks):
        for question in benchmark.questions:
            # Add source information to metadata
            question.metadata["source_benchmark"] = benchmark.name
            question.metadata["source_index"] = i

            merged.add_question(question)

    # Save merged benchmark
    merged.save(output_path)

    print(f"Merged {len(benchmarks)} benchmarks into {output_path}")
    print(f"Total questions: {len(merged.questions)}")

    return merged

# Merge multiple benchmarks
merged = merge_benchmarks(
    "math-benchmark.jsonld",
    "science-benchmark.jsonld",
    "literature-benchmark.jsonld",
    output_path="comprehensive-benchmark.jsonld"
)
```

## Best Practices

### Naming Conventions

```python
# Use descriptive, consistent names
benchmark.save("medical-diagnosis-v2-20240315.jsonld")
benchmark.save("k12-math-assessment-final.jsonld")
benchmark.save("code-review-benchmark-pilot.jsonld")

# Include version and date information
benchmark.save(f"{benchmark.name}-v{benchmark.version}-{datetime.now():%Y%m%d}.jsonld")
```

### File Organization

```python
# Organize benchmarks in logical directories
benchmark.save("benchmarks/medical/diagnosis-v2.jsonld")
benchmark.save("benchmarks/education/k12-math.jsonld")
benchmark.save("checkpoints/2024-03/diagnosis-checkpoint-1.jsonld")
```

### Regular Maintenance

```python
def cleanup_old_checkpoints(directory, keep_days=30):
    """Remove checkpoint files older than specified days"""
    cutoff = datetime.now() - timedelta(days=keep_days)

    removed = 0
    for filename in os.listdir(directory):
        if "checkpoint" in filename and filename.endswith(".jsonld"):
            filepath = os.path.join(directory, filename)
            if datetime.fromtimestamp(os.path.getmtime(filepath)) < cutoff:
                os.remove(filepath)
                removed += 1

    print(f"Removed {removed} old checkpoint files")

# Regular cleanup
cleanup_old_checkpoints("checkpoints/", keep_days=30)
```

## Next Steps

With benchmarks saved and loaded:

- Share benchmarks with team members for collaborative evaluation
- Version control benchmark files for tracking changes over time
- Set up automated backup systems for important benchmark data
