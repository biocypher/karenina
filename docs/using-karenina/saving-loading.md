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

## Database Storage

Karenina supports persistent storage in relational databases (SQLite, PostgreSQL, MySQL) via SQLAlchemy. Database storage is ideal for:

- **Multi-user collaboration** with concurrent access
- **Query and analysis** of benchmark data using SQL
- **Production deployments** requiring robust persistence
- **Integration** with existing data infrastructure

### Quick Start with Database Storage

```python
from karenina import Benchmark, save_benchmark, load_benchmark

# Create and populate benchmark
benchmark = Benchmark.create(
    name="My Benchmark",
    description="Example benchmark",
    version="1.0.0"
)
benchmark.add_question("What is 2+2?", "4")

# Save to SQLite database
save_benchmark(benchmark, "sqlite:///benchmarks.db")

# Load from database
loaded = load_benchmark("My Benchmark", "sqlite:///benchmarks.db")
```

### Database Configuration

```python
from karenina import DBConfig

# SQLite (default, file-based)
db_config = DBConfig(
    storage_url="sqlite:///benchmarks.db",
    auto_create=True,       # Auto-create tables if they don't exist
    auto_commit=True,       # Auto-commit transactions
    echo=False             # Disable SQL query logging
)

# PostgreSQL (production)
db_config = DBConfig(
    storage_url="postgresql://user:password@localhost:5432/karenina",
    pool_size=10,           # Connection pool size
    max_overflow=20,        # Max connections beyond pool_size
    pool_recycle=3600,      # Recycle connections after 1 hour
    pool_pre_ping=True     # Verify connections before use
)

# MySQL
db_config = DBConfig(
    storage_url="mysql+pymysql://user:password@localhost:3306/karenina"
)

# Save using DBConfig
save_benchmark(benchmark, db_config)
```

### Save and Load Operations

```python
# Save benchmark to database
save_benchmark(
    benchmark,
    storage="sqlite:///benchmarks.db",
    checkpoint_path="backup.jsonld"  # Optional: also save JSON-LD checkpoint
)

# Load benchmark by name
benchmark = load_benchmark(
    benchmark_name="My Benchmark",
    storage="sqlite:///benchmarks.db"
)

# Load with DBConfig object
benchmark, db_config = load_benchmark(
    benchmark_name="My Benchmark",
    storage="sqlite:///benchmarks.db",
    load_config=True  # Returns tuple of (Benchmark, DBConfig)
)
```

### Benchmark Methods for Database

```python
# Create and save benchmark
benchmark = Benchmark.create(name="Test Benchmark")
benchmark.add_question("Question 1?", "Answer 1")

# Save using benchmark method (returns self for chaining)
benchmark.save_to_db("sqlite:///benchmarks.db")

# Load using class method
benchmark = Benchmark.load_from_db("Test Benchmark", "sqlite:///benchmarks.db")

# Method chaining
Benchmark.create(name="New Benchmark") \
    .add_question("Q?", "A") \
    .save_to_db("sqlite:///benchmarks.db")
```

### Updating Existing Benchmarks

```python
# Load existing benchmark
benchmark = load_benchmark("My Benchmark", "sqlite:///benchmarks.db")

# Make changes
benchmark.description = "Updated description"
benchmark.add_question("New question?", "New answer")

# Save updates (automatically updates existing record)
save_benchmark(benchmark, "sqlite:///benchmarks.db")
```

### Auto-Save Verification Results

```python
from karenina import VerificationConfig, DBConfig, ModelConfig

# Configure verification with database storage
db_config = DBConfig(storage_url="sqlite:///results.db")

verification_config = VerificationConfig(
    answering_model=ModelConfig(model_provider="openai", model_name="gpt-4.1-mini"),
    parsing_model=ModelConfig(model_provider="openai", model_name="gpt-4.1-mini"),
    db_config=db_config  # Auto-save results to database
)

# Run verification (results automatically saved to database)
results = benchmark.run_verification(verification_config)
```

### Querying Database Views

```python
from karenina.storage import (
    get_benchmark_summary,
    get_verification_run_summary,
    get_model_performance,
    get_failed_verifications,
    get_database_statistics
)

# Get benchmark summaries
summaries = get_benchmark_summary("sqlite:///benchmarks.db")
for summary in summaries:
    print(f"{summary['benchmark_name']}: {summary['total_questions']} questions")

# Get verification run statistics
runs = get_verification_run_summary("sqlite:///results.db")
for run in runs:
    print(f"{run['run_name']}: {run['success_rate']:.1f}% success rate")

# Get model performance statistics
performance = get_model_performance("sqlite:///results.db")
for model in performance:
    print(f"{model['model_name']}: {model['success_rate_pct']:.1f}% success")

# Get all failed verifications for debugging
failures = get_failed_verifications("sqlite:///results.db")
for failure in failures:
    print(f"Question: {failure['question_text']}")
    print(f"Error: {failure['error']}")

# Get overall database statistics
stats = get_database_statistics("sqlite:///benchmarks.db")
print(f"Total benchmarks: {stats['total_benchmarks']}")
print(f"Total questions: {stats['total_questions']}")
```

### Direct Database Access

```python
from karenina.storage import DBConfig, get_session
from karenina.storage.models import BenchmarkModel, QuestionModel
from sqlalchemy import select

db_config = DBConfig(storage_url="sqlite:///benchmarks.db")

# Use session context manager for custom queries
with get_session(db_config) as session:
    # Query all benchmarks
    benchmarks = session.execute(select(BenchmarkModel)).scalars().all()

    for b in benchmarks:
        print(f"Benchmark: {b.name} (ID: {b.id})")
        print(f"  Questions: {len(b.benchmark_questions)}")
        print(f"  Created: {b.created_at}")

    # Query specific questions
    questions = session.execute(
        select(QuestionModel).where(QuestionModel.question_text.like("%math%"))
    ).scalars().all()

    for q in questions:
        print(f"Question: {q.question_text}")
```

### Database Schema Overview

The database schema includes:

**Core Tables:**
- `benchmarks` - Benchmark metadata and configuration
- `questions` - Shared question pool (deduplicated by MD5 hash)
- `benchmark_questions` - Junction table with benchmark-specific data (templates, rubrics)
- `verification_runs` - Verification run metadata and statistics
- `verification_results` - Individual question verification results

**Pre-built Views:**
- `benchmark_summary_view` - Benchmark overviews with question counts
- `verification_run_summary_view` - Run statistics and success rates
- `question_usage_view` - Which benchmarks use each question
- `latest_verification_results_view` - Most recent result per question-model pair
- `model_performance_view` - Aggregated performance by model
- `failed_verifications_view` - All failures with error details
- `rubric_scores_aggregate_view` - Rubric statistics
- `verification_history_timeline_view` - Chronological run history
- `rubric_traits_by_type_view` - Rubric traits categorized by type

### Database vs JSON-LD Checkpoints

| Feature | Database Storage | JSON-LD Checkpoints |
|---------|-----------------|---------------------|
| **Query Support** | Full SQL queries | File-based search |
| **Concurrent Access** | Multi-user safe | Single user |
| **Deduplication** | Questions shared across benchmarks | Questions duplicated |
| **Views & Analytics** | Built-in aggregation views | Manual analysis |
| **Portability** | Database-dependent | Fully portable |
| **Versioning** | Database migrations | File versioning |
| **Best For** | Production, collaboration, analysis | Development, sharing, backups |

**Recommendation:** Use both! Database for primary storage and queries, JSON-LD checkpoints for backups and sharing.

```python
# Dual persistence strategy
benchmark.save_to_db("sqlite:///production.db")  # Primary storage
benchmark.save("backups/benchmark-20240315.jsonld")  # Backup + sharing
```

### Database Maintenance

```python
from karenina.storage import init_database, drop_database, reset_database

db_config = DBConfig(storage_url="sqlite:///benchmarks.db")

# Initialize database (creates tables and views)
init_database(db_config)

# Reset database (WARNING: destructive!)
reset_database(db_config)  # Drops and recreates all tables

# Drop database (WARNING: destructive!)
drop_database(db_config)  # Removes all tables and views
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
