# Saving and Loading Benchmarks

This guide covers how to persist, restore, and export benchmarks using Karenina's checkpoint and database systems.

## Understanding Persistence

Karenina provides two main approaches for persisting benchmarks:

1. **Checkpoints (JSON-LD files)**: Portable, human-readable files perfect for sharing and version control
2. **Database storage (SQLite)**: Structured storage with query capabilities for production use

You can use both approaches together: databases for primary storage and checkpoints for backups and sharing.

---

## Checkpoint Files (JSON-LD)

Checkpoints are JSON-LD files that capture the complete state of a benchmark.

### What Gets Saved

A checkpoint includes:
- Benchmark metadata (name, description, version)
- All questions with their metadata
- Answer templates
- Rubrics (global and question-specific)
- Verification results (if available)

### JSON-LD Format

Karenina uses **JSON-LD (JSON for Linked Data)** format following schema.org conventions:

**Benefits:**
- **Structured and semantic**: Machine-readable with clear data relationships
- **Human-readable**: Open in any text editor to inspect contents
- **Cross-platform**: Works across different environments
- **Version-compatible**: Maintains backward compatibility

---

## Saving Checkpoints

### Basic Save

Save your benchmark to a JSON-LD checkpoint file:

```python
from pathlib import Path

# Basic save
benchmark.save(Path("genomics_benchmark.jsonld"))

# Save to specific directory
benchmark.save(Path("benchmarks/genomics_benchmark.jsonld"))
```

### What Happens When You Save

```python
from karenina import Benchmark

# Create and populate benchmark
benchmark = Benchmark.create(
    name="Genomics Knowledge Benchmark",
    version="1.0.0"
)

benchmark.add_question(
    question="How many chromosomes are in a human somatic cell?",
    raw_answer="46",
    author={"name": "Bio Curator"}
)

benchmark.add_question(
    question="What is the approved drug target of Venetoclax?",
    raw_answer="BCL2",
    author={"name": "Bio Curator"}
)

# Save checkpoint
checkpoint_path = Path("genomics_benchmark.jsonld")
benchmark.save(checkpoint_path)

print(f"✓ Saved checkpoint to {checkpoint_path}")
print(f"  Questions: {len(benchmark.questions)}")
print(f"  Size: {checkpoint_path.stat().st_size} bytes")
```

**Output:**
```
✓ Saved checkpoint to genomics_benchmark.jsonld
  Questions: 2
  Size: 4532 bytes
```

---

## Loading Checkpoints

### Basic Load

Load a benchmark from a checkpoint file:

```python
from karenina import Benchmark
from pathlib import Path

# Load benchmark
benchmark = Benchmark.load(Path("genomics_benchmark.jsonld"))

print(f"Loaded benchmark: {benchmark.name}")
print(f"Version: {benchmark.version}")
print(f"Questions: {len(benchmark.questions)}")

# Access questions
for qid in list(benchmark.questions.keys())[:3]:
    question = benchmark.get_question(qid)
    print(f"  • {question.question[:50]}...")
```

**Output:**
```
Loaded benchmark: Genomics Knowledge Benchmark
Version: 1.0.0
Questions: 2
  • How many chromosomes are in a human somatic ce...
  • What is the approved drug target of Venetoclax...
```

### Verify Loaded Data

```python
def load_and_verify(checkpoint_path: Path):
    """Load benchmark with validation"""
    try:
        benchmark = Benchmark.load(checkpoint_path)

        # Basic validation
        assert len(benchmark.questions) > 0, "No questions found"
        assert benchmark.name, "Missing benchmark name"

        print(f"✓ Successfully loaded: {benchmark.name}")
        print(f"  Questions: {len(benchmark.questions)}")

        # Check templates
        questions_with_templates = sum(
            1 for q in benchmark.questions.values()
            if q.answer_template
        )
        print(f"  Templates: {questions_with_templates}/{len(benchmark.questions)}")

        return benchmark

    except Exception as e:
        print(f"✗ Failed to load: {str(e)}")
        return None

# Load with validation
benchmark = load_and_verify(Path("genomics_benchmark.jsonld"))
```

---

## Database Storage

Database storage provides structured persistence with query capabilities. For detailed database usage, see [Defining Benchmarks](defining-benchmark.md#database-persistence).

### Quick Database Example

```python
from karenina import Benchmark
from pathlib import Path

# Create benchmark
benchmark = Benchmark.create(
    name="Genomics Knowledge Benchmark",
    version="1.0.0"
)

# Add questions
benchmark.add_question(
    question="How many chromosomes are in a human somatic cell?",
    raw_answer="46"
)

# Save to database (with optional checkpoint backup)
benchmark.save_to_db(
    storage="sqlite:///benchmarks.db",
    checkpoint_path=Path("genomics_benchmark.jsonld")
)

print("✓ Saved to database and checkpoint")

# Load from database
loaded = Benchmark.load_from_db(
    benchmark_name="Genomics Knowledge Benchmark",
    storage="sqlite:///benchmarks.db"
)

print(f"✓ Loaded from database: {loaded.name}")
```

### When to Use Database vs Checkpoints

| Use Case | Recommended Approach |
|----------|---------------------|
| **Development and prototyping** | Checkpoints only |
| **Sharing benchmarks** | Checkpoints (portable files) |
| **Production deployment** | Database primary, checkpoints for backup |
| **Version control (Git)** | Checkpoints (diff-friendly) |
| **Multi-user collaboration** | Database with checkpoint backups |
| **Query and analytics** | Database |
| **Backups** | Checkpoints |

**Best Practice:** Use both!
```python
# Save to database for primary storage
benchmark.save_to_db("sqlite:///production.db")

# Also save checkpoint for backup/sharing
benchmark.save(Path("backups/genomics_v1.0.0.jsonld"))
```

---

## Exporting Verification Results

After running verification, export results for analysis and reporting.

### Export to CSV

CSV format is ideal for spreadsheet analysis:

```python
from pathlib import Path

# Run verification first
from karenina.schemas import VerificationConfig, ModelConfig

config = VerificationConfig(
    answering_models=[ModelConfig(
        id="gpt-4.1-mini",
        model_provider="openai",
        model_name="gpt-4.1-mini",
        interface="langchain"
    )],
    parsing_models=[ModelConfig(
        id="gpt-judge",
        model_provider="openai",
        model_name="gpt-4.1-mini",
        interface="langchain"
    )]
)

results = benchmark.run_verification(config)

# Export to CSV
benchmark.export_verification_results_to_file(
    file_path=Path("results.csv"),
    format="csv"
)

print("✓ Exported to results.csv")
```

**CSV Output Structure:**

| question_id | question | expected_answer | model_answer | template_passed | answering_model | parsing_model | timestamp |
|-------------|----------|-----------------|--------------|-----------------|-----------------|---------------|-----------|
| abc123... | How many chromosomes... | 46 | There are 46 chromosomes... | True | gpt-4.1-mini | gpt-judge | 2024-03-15 14:30:22 |

### Export to JSON

JSON format is ideal for programmatic analysis:

```python
# Export to JSON
benchmark.export_verification_results_to_file(
    file_path=Path("results.json"),
    format="json"
)

print("✓ Exported to results.json")
```

**JSON Output Structure:**
```json
{
  "benchmark_name": "Genomics Knowledge Benchmark",
  "export_timestamp": "2024-03-15T14:30:22",
  "total_results": 3,
  "results": [
    {
      "question_id": "abc123...",
      "question": "How many chromosomes are in a human somatic cell?",
      "expected_answer": "46",
      "raw_response": "There are 46 chromosomes in a human somatic cell.",
      "parsed_response": {"count": 46},
      "verify_result": true,
      "answering_model_id": "gpt-4.1-mini",
      "parsing_model_id": "gpt-judge",
      "timestamp": "2024-03-15T14:30:22"
    }
  ]
}
```

### Export Specific Questions

Export results for a subset of questions:

```python
# Get question IDs for chromosomes questions
chromosome_qids = [
    qid for qid in benchmark.questions.keys()
    if "chromosome" in benchmark.get_question(qid).question.lower()
]

# Export only chromosome questions
benchmark.export_verification_results_to_file(
    file_path=Path("chromosome_results.csv"),
    format="csv",
    question_ids=chromosome_qids
)

print(f"✓ Exported {len(chromosome_qids)} chromosome questions")
```

### Export with Rubric Scores

When rubrics are enabled, export includes rubric evaluations:

```python
# Export with rubric data
benchmark.export_verification_results_to_file(
    file_path=Path("results_with_rubrics.csv"),
    format="csv"
)
```

**CSV includes rubric columns:**

| question_id | template_passed | rubric_conciseness | rubric_clarity | rubric_bh3_mention |
|-------------|-----------------|--------------------|-----------------|--------------------|
| abc123... | True | 4 | 5 | True |

---

## Checkpoint Management

### Incremental Checkpoints

Save checkpoints at key stages of your workflow:

```python
from karenina import Benchmark
from karenina.schemas import ModelConfig
from pathlib import Path

# Create benchmark
benchmark = Benchmark.create(
    name="Genomics Knowledge Benchmark",
    version="1.0.0"
)

# Add questions
benchmark.add_question(
    question="How many chromosomes are in a human somatic cell?",
    raw_answer="46"
)
benchmark.add_question(
    question="What is the approved drug target of Venetoclax?",
    raw_answer="BCL2"
)

# Checkpoint 1: After adding questions
benchmark.save(Path("checkpoints/01_questions_added.jsonld"))
print("✓ Checkpoint 1: Questions added")

# Generate templates
model_config = ModelConfig(
    id="gpt-4.1-mini",
    model_provider="openai",
    model_name="gpt-4.1-mini",
    interface="langchain"
)
benchmark.generate_all_templates(model_config=model_config)

# Checkpoint 2: After template generation
benchmark.save(Path("checkpoints/02_templates_generated.jsonld"))
print("✓ Checkpoint 2: Templates generated")

# Create rubrics
from karenina.schemas import RubricTrait
benchmark.create_global_rubric(
    name="Answer Quality",
    traits=[
        RubricTrait(
            name="Conciseness",
            description="Rate conciseness 1-5",
            kind="score"
        )
    ]
)

# Checkpoint 3: After rubrics
benchmark.save(Path("checkpoints/03_rubrics_created.jsonld"))
print("✓ Checkpoint 3: Rubrics created")

# Run verification
from karenina.schemas import VerificationConfig
config = VerificationConfig(
    answering_models=[model_config],
    parsing_models=[model_config],
    rubric_enabled=True
)
results = benchmark.run_verification(config)

# Final checkpoint: After verification
benchmark.save(Path("checkpoints/04_verification_complete.jsonld"))
print("✓ Checkpoint 4: Verification complete")

# Save final version
benchmark.save(Path("genomics_benchmark_final.jsonld"))
print("✓ Final benchmark saved")
```

### Timestamped Backups

Create timestamped backups automatically:

```python
from datetime import datetime
from pathlib import Path

def save_with_timestamp(benchmark, base_name: str):
    """Save benchmark with timestamp"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{base_name}_{timestamp}.jsonld"
    path = Path(f"backups/{filename}")

    path.parent.mkdir(parents=True, exist_ok=True)
    benchmark.save(path)

    print(f"✓ Backup saved: {filename}")
    return path

# Save with timestamp
save_with_timestamp(benchmark, "genomics_benchmark")
```

**Output:**
```
✓ Backup saved: genomics_benchmark_20240315_143022.jsonld
```

---

## Comparing Checkpoints

Compare two checkpoints to see what changed:

```python
def compare_checkpoints(path1: Path, path2: Path):
    """Compare two benchmark checkpoints"""
    from karenina import Benchmark

    bench1 = Benchmark.load(path1)
    bench2 = Benchmark.load(path2)

    print(f"=== Comparing Checkpoints ===")
    print(f"Checkpoint 1: {path1.name}")
    print(f"Checkpoint 2: {path2.name}")
    print()

    # Compare questions
    print(f"Questions:")
    print(f"  {path1.name}: {len(bench1.questions)}")
    print(f"  {path2.name}: {len(bench2.questions)}")

    # Compare templates
    templates1 = sum(1 for q in bench1.questions.values() if q.answer_template)
    templates2 = sum(1 for q in bench2.questions.values() if q.answer_template)
    print(f"\nTemplates:")
    print(f"  {path1.name}: {templates1}")
    print(f"  {path2.name}: {templates2}")

    # Compare rubrics
    has_global1 = bench1.global_rubric is not None
    has_global2 = bench2.global_rubric is not None
    print(f"\nGlobal Rubric:")
    print(f"  {path1.name}: {'Yes' if has_global1 else 'No'}")
    print(f"  {path2.name}: {'Yes' if has_global2 else 'No'}")

# Compare before and after verification
compare_checkpoints(
    Path("checkpoints/02_templates_generated.jsonld"),
    Path("checkpoints/04_verification_complete.jsonld")
)
```

**Output:**
```
=== Comparing Checkpoints ===
Checkpoint 1: 02_templates_generated.jsonld
Checkpoint 2: 04_verification_complete.jsonld

Questions:
  02_templates_generated.jsonld: 2
  04_verification_complete.jsonld: 2

Templates:
  02_templates_generated.jsonld: 2
  04_verification_complete.jsonld: 2

Global Rubric:
  02_templates_generated.jsonld: No
  04_verification_complete.jsonld: Yes
```

---

## Portability and Sharing

### Sharing Benchmarks with Collaborators

Checkpoints are portable and can be easily shared:

```python
# Prepare benchmark for sharing
benchmark.save(Path("genomics_benchmark_v1.0.0.jsonld"))

# Collaborator loads it
from karenina import Benchmark
from pathlib import Path

benchmark = Benchmark.load(Path("genomics_benchmark_v1.0.0.jsonld"))
print(f"Loaded shared benchmark: {benchmark.name}")
print(f"Version: {benchmark.version}")
```

**Sharing checklist:**
- ✅ Save to descriptive filename with version
- ✅ Include README with benchmark purpose and usage
- ✅ Document any special requirements (API keys, models)
- ✅ Test loading on a different machine

### Version Control with Git

Checkpoints work well with Git:

```bash
# Add checkpoint to Git
git add genomics_benchmark_v1.0.0.jsonld
git commit -m "Add genomics benchmark v1.0.0"
git push

# Track benchmark evolution over time
git log -- genomics_benchmark_v1.0.0.jsonld
```

**Git best practices:**
- Use semantic versioning for checkpoint filenames
- Include descriptive commit messages
- Tag important versions: `git tag v1.0.0`
- Use `.gitignore` for temporary checkpoints

### Moving Benchmarks Between Environments

Checkpoints are fully portable:

```python
# Development environment
benchmark.save(Path("genomics_benchmark.jsonld"))

# Copy file to production environment
# scp genomics_benchmark.jsonld user@production:/data/

# Production environment
benchmark = Benchmark.load(Path("/data/genomics_benchmark.jsonld"))
```

---

## Complete Workflow Example

Here's a complete example showing checkpoints, database storage, and export:

```python
from karenina import Benchmark
from karenina.schemas import VerificationConfig, ModelConfig, RubricTrait
from pathlib import Path

# 1. Create benchmark
benchmark = Benchmark.create(
    name="Genomics Knowledge Benchmark",
    description="Testing LLM knowledge of genomics",
    version="1.0.0",
    creator="Bio Team"
)

# 2. Add questions
questions = [
    ("How many chromosomes are in a human somatic cell?", "46"),
    ("What is the approved drug target of Venetoclax?", "BCL2"),
    ("How many protein subunits does hemoglobin A have?", "4")
]

for q, a in questions:
    benchmark.add_question(question=q, raw_answer=a, author={"name": "Bio Curator"})

# Checkpoint 1
benchmark.save(Path("checkpoints/step1_questions.jsonld"))
print("✓ Checkpoint 1: Questions added")

# 3. Generate templates
model_config = ModelConfig(
    id="gpt-4.1-mini",
    model_provider="openai",
    model_name="gpt-4.1-mini",
    temperature=0.1,
    interface="langchain"
)
benchmark.generate_all_templates(model_config=model_config)

# Checkpoint 2
benchmark.save(Path("checkpoints/step2_templates.jsonld"))
print("✓ Checkpoint 2: Templates generated")

# 4. Create rubric
benchmark.create_global_rubric(
    name="Answer Quality",
    traits=[
        RubricTrait(
            name="Conciseness",
            description="Rate conciseness 1-5",
            kind="score"
        ),
        RubricTrait(
            name="Clarity",
            description="Is the answer clear?",
            kind="binary"
        )
    ]
)

# Checkpoint 3
benchmark.save(Path("checkpoints/step3_rubrics.jsonld"))
print("✓ Checkpoint 3: Rubrics created")

# 5. Run verification
config = VerificationConfig(
    answering_models=[model_config],
    parsing_models=[model_config],
    rubric_enabled=True
)
results = benchmark.run_verification(config)

# Checkpoint 4
benchmark.save(Path("checkpoints/step4_verified.jsonld"))
print("✓ Checkpoint 4: Verification complete")

# 6. Save to database with checkpoint
benchmark.save_to_db(
    storage="sqlite:///benchmarks.db",
    checkpoint_path=Path("genomics_benchmark_v1.0.0.jsonld")
)
print("✓ Saved to database with checkpoint")

# 7. Export results
benchmark.export_verification_results_to_file(
    file_path=Path("results.csv"),
    format="csv"
)
benchmark.export_verification_results_to_file(
    file_path=Path("results.json"),
    format="json"
)
print("✓ Exported results to CSV and JSON")

# 8. Create timestamped backup
from datetime import datetime
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
backup_path = Path(f"backups/genomics_{timestamp}.jsonld")
backup_path.parent.mkdir(parents=True, exist_ok=True)
benchmark.save(backup_path)
print(f"✓ Backup saved: {backup_path.name}")

print("\n=== Summary ===")
print(f"Benchmark: {benchmark.name} v{benchmark.version}")
print(f"Questions: {len(benchmark.questions)}")
print(f"Verification results: {len(results)}")
print(f"Checkpoints: 4")
print(f"Database: Yes (sqlite:///benchmarks.db)")
print(f"Exports: CSV and JSON")
print(f"Backup: Yes")
```

**Output:**
```
✓ Checkpoint 1: Questions added
✓ Checkpoint 2: Templates generated
✓ Checkpoint 3: Rubrics created
✓ Checkpoint 4: Verification complete
✓ Saved to database with checkpoint
✓ Exported results to CSV and JSON
✓ Backup saved: genomics_20240315_143022.jsonld

=== Summary ===
Benchmark: Genomics Knowledge Benchmark v1.0.0
Questions: 3
Verification results: 3
Checkpoints: 4
Database: Yes (sqlite:///benchmarks.db)
Exports: CSV and JSON
Backup: Yes
```

---

## Best Practices

### Naming Conventions

Use descriptive, versioned filenames:

```python
# ✅ Good: Descriptive with version and date
benchmark.save(Path("genomics_benchmark_v1.0.0_20240315.jsonld"))
benchmark.save(Path("drug_targets_benchmark_v2.1.0.jsonld"))

# ❌ Bad: Generic or unclear names
benchmark.save(Path("benchmark.jsonld"))
benchmark.save(Path("test.jsonld"))
```

### Directory Organization

Organize checkpoints logically:

```
project/
├── benchmarks/
│   ├── genomics_v1.0.0.jsonld
│   ├── drug_targets_v1.0.0.jsonld
│   └── proteins_v1.0.0.jsonld
├── checkpoints/
│   └── genomics/
│       ├── 01_questions.jsonld
│       ├── 02_templates.jsonld
│       ├── 03_rubrics.jsonld
│       └── 04_verified.jsonld
├── backups/
│   ├── genomics_20240315.jsonld
│   └── genomics_20240316.jsonld
└── exports/
    ├── results.csv
    └── results.json
```

### Backup Strategy

Implement a regular backup strategy:

```python
from pathlib import Path
from datetime import datetime

def backup_benchmark(benchmark, backup_dir: Path):
    """Create daily backup"""
    backup_dir.mkdir(parents=True, exist_ok=True)

    # Daily backup
    date_str = datetime.now().strftime("%Y%m%d")
    daily_backup = backup_dir / f"{benchmark.name}_{date_str}.jsonld"

    if not daily_backup.exists():
        benchmark.save(daily_backup)
        print(f"✓ Daily backup: {daily_backup.name}")
    else:
        print(f"  Daily backup already exists for {date_str}")

# Backup after important changes
backup_benchmark(benchmark, Path("backups"))
```

### Cleanup Old Checkpoints

Remove old temporary checkpoints periodically:

```python
from pathlib import Path
from datetime import datetime, timedelta

def cleanup_old_checkpoints(checkpoint_dir: Path, keep_days: int = 30):
    """Remove checkpoint files older than keep_days"""
    cutoff_date = datetime.now() - timedelta(days=keep_days)

    removed = 0
    for checkpoint in checkpoint_dir.glob("*.jsonld"):
        # Check if it's a temporary checkpoint (contains 'checkpoint' in name)
        if "checkpoint" in checkpoint.name.lower():
            mtime = datetime.fromtimestamp(checkpoint.stat().st_mtime)
            if mtime < cutoff_date:
                checkpoint.unlink()
                removed += 1
                print(f"  Removed: {checkpoint.name}")

    print(f"✓ Removed {removed} old checkpoints (older than {keep_days} days)")

# Cleanup old checkpoints
cleanup_old_checkpoints(Path("checkpoints"), keep_days=30)
```

---

## Next Steps

After saving and loading benchmarks:

- [Run Verification](verification.md) - Evaluate LLM responses
- [Advanced Features](../advanced/deep-judgment.md) - Use deep-judgment for detailed feedback
- [Share Benchmarks](#portability-and-sharing) - Collaborate with your team

---

## Related Documentation

- [Defining Benchmarks](defining-benchmark.md) - Benchmark creation and database persistence
- [Verification](verification.md) - Run evaluations
- [Templates](templates.md) - Structured answer evaluation
- [Rubrics](rubrics.md) - Qualitative assessment criteria
- [Quick Start](../quickstart.md) - End-to-end workflow example
