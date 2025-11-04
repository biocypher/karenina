# Configuration Presets

Configuration presets allow you to save, load, and share complete verification configurations, eliminating the need to manually reconfigure settings for recurring benchmark scenarios.

## What are Presets?

**Configuration presets** are saved snapshots of your verification settings that can be quickly reloaded for future benchmark runs. A preset captures:

- **Model configurations**: Answering and parsing models with all their settings
- **Evaluation settings**: Replication count, evaluation mode, parsing-only flag
- **Rubric settings**: Enabled status, trait selection, evaluation mode
- **Advanced features**: Deep-judgment, abstention detection, few-shot configuration

Presets make it easy to:
- **Reuse configurations**: Quickly switch between different benchmark setups
- **Ensure consistency**: Use the same configuration across multiple runs
- **Share setups**: Export and share configurations with teammates
- **Organize scenarios**: Maintain separate configs for testing, production, experiments

## Why Use Presets?

### 1. Save Time

Instead of manually reconfiguring models and settings each time:

```python
# Without presets: Manually configure every time ❌
config = VerificationConfig(
    answering_models=[model1, model2],
    parsing_models=[parser],
    replicate_count=3,
    rubric_enabled=True,
    deep_judgment_enabled=True,
    deep_judgment_max_excerpts_per_attribute=3,
    # ... 15 more parameters ...
)
```

```python
# With presets: Load saved configuration ✓
config = VerificationConfig.from_preset(Path("my-setup.json"))
```

### 2. Maintain Consistency

Presets ensure the same configuration is used across runs, eliminating configuration drift:

```python
# Run 1: Today
config = VerificationConfig.from_preset(Path("production-config.json"))
results1 = benchmark.run_verification(config)

# Run 2: Next week (identical configuration guaranteed)
config = VerificationConfig.from_preset(Path("production-config.json"))
results2 = benchmark.run_verification(config)
```

### 3. Share Configurations

Share preset files with teammates or across projects:

```bash
# Export preset
cp benchmark_presets/genomics-standard.json shared/

# Teammate imports preset
cp shared/genomics-standard.json benchmark_presets/
```

## Saving a Preset

### Basic Usage

Save your current configuration with a descriptive name:

```python
from karenina import Benchmark
from karenina.schemas import VerificationConfig, ModelConfig
from pathlib import Path

# Create and configure your verification setup
model_config = ModelConfig(
    id="gpt-4.1-mini",
    model_provider="openai",
    model_name="gpt-4.1-mini",
    temperature=0.0,
    interface="langchain"
)

config = VerificationConfig(
    answering_models=[model_config],
    parsing_models=[model_config],
    replicate_count=3,
    rubric_enabled=True,
    deep_judgment_enabled=True
)

# Save as a preset
metadata = config.save_preset(
    name="Genomics Standard Config",
    description="Standard setup for genomics benchmarks with deep-judgment"
)

print(f"Preset saved to: {metadata['filepath']}")
```

**Output:**
```
Preset saved to: benchmark_presets/genomics-standard-config.json
```

### What Gets Saved?

Presets include:

**✓ Included**:
- All model configurations (answering_models, parsing_models)
- Evaluation settings (replicate_count, parsing_only, evaluation_mode)
- Rubric settings (rubric_enabled, rubric_trait_names)
- Advanced features (abstention_enabled, deep_judgment_*, few_shot_config)

**✗ Excluded**:
- Job-specific metadata (run_name)
- Database configuration (db_config)

### Preset File Structure

Presets are saved as JSON files in the `benchmark_presets/` directory:

```
project_root/
├── benchmark_presets/
│   ├── genomics-standard-config.json
│   ├── quick-test.json
│   └── production-full.json
└── my_benchmark.py
```

Each preset file contains:

```json
{
  "id": "550e8400-e29b-41d4-a716-446655440000",
  "name": "Genomics Standard Config",
  "description": "Standard setup for genomics benchmarks with deep-judgment",
  "config": {
    "answering_models": [...],
    "parsing_models": [...],
    "replicate_count": 3,
    "rubric_enabled": true,
    "deep_judgment_enabled": true,
    ...
  },
  "created_at": "2025-11-04T10:30:00Z",
  "updated_at": "2025-11-04T10:30:00Z"
}
```

## Loading a Preset

### Basic Usage

Load a saved preset and use it for verification:

```python
from karenina import Benchmark
from karenina.schemas import VerificationConfig
from pathlib import Path

# Load benchmark
benchmark = Benchmark.load("genomics_benchmark.jsonld")

# Load preset by filepath
config = VerificationConfig.from_preset(
    Path("benchmark_presets/genomics-standard-config.json")
)

# Run verification with loaded configuration
results = benchmark.run_verification(config)
print(f"Verified {len(results)} questions")
```

### Custom Preset Directory

Specify a custom location for presets using an environment variable:

```python
import os
from pathlib import Path

# Set custom preset directory
os.environ["KARENINA_PRESETS_DIR"] = "/path/to/my/presets"

# Load preset from custom directory
config = VerificationConfig.from_preset(
    Path("/path/to/my/presets/genomics-standard-config.json")
)
```

## Complete Example

Here's an end-to-end workflow showing preset creation and usage:

```python
from karenina import Benchmark
from karenina.schemas import VerificationConfig, ModelConfig
from pathlib import Path

# ============================================================
# STEP 1: Create benchmark with genomics questions
# ============================================================

benchmark = Benchmark.create(
    name="Genomics Knowledge Benchmark",
    description="Testing LLM knowledge of genomics and molecular biology",
    version="1.0.0"
)

# Add questions
questions = [
    ("What is the approved drug target of Venetoclax?", "BCL2"),
    ("How many chromosomes are in a human somatic cell?", "46"),
    ("How many protein subunits does hemoglobin A have?", "4"),
]

for question, answer in questions:
    benchmark.add_question(
        question=question,
        raw_answer=answer,
        author={"name": "Genomics Curator"}
    )

# ============================================================
# STEP 2: Configure models and settings
# ============================================================

model_config = ModelConfig(
    id="gpt-4.1-mini",
    model_provider="openai",
    model_name="gpt-4.1-mini",
    temperature=0.0,
    interface="langchain"
)

# Configuration for testing (fast)
test_config = VerificationConfig(
    answering_models=[model_config],
    parsing_models=[model_config],
    replicate_count=1,
    rubric_enabled=False
)

# Configuration for production (comprehensive)
production_config = VerificationConfig(
    answering_models=[model_config],
    parsing_models=[model_config],
    replicate_count=3,
    rubric_enabled=True,
    deep_judgment_enabled=True,
    abstention_enabled=True
)

# ============================================================
# STEP 3: Save presets
# ============================================================

# Save test configuration
test_metadata = test_config.save_preset(
    name="Quick Test",
    description="Fast configuration for smoke tests"
)
print(f"✓ Test preset saved: {test_metadata['filepath']}")

# Save production configuration
prod_metadata = production_config.save_preset(
    name="Production Full",
    description="Comprehensive configuration with all features enabled"
)
print(f"✓ Production preset saved: {prod_metadata['filepath']}")

# ============================================================
# STEP 4: Generate templates
# ============================================================

print("\nGenerating templates...")
benchmark.generate_all_templates(model_config=model_config)
print("✓ Templates generated")

# ============================================================
# STEP 5: Run quick test using preset
# ============================================================

print("\nRunning quick test...")
test_config = VerificationConfig.from_preset(
    Path("benchmark_presets/quick-test.json")
)
test_results = benchmark.run_verification(test_config)
print(f"✓ Quick test complete: {len(test_results)} questions")

# ============================================================
# STEP 6: Run production verification using preset
# ============================================================

print("\nRunning production verification...")
prod_config = VerificationConfig.from_preset(
    Path("benchmark_presets/production-full.json")
)
prod_results = benchmark.run_verification(prod_config)
print(f"✓ Production verification complete: {len(prod_results)} questions")

# Analyze results
passed = sum(1 for r in prod_results.values() if r.verify_result)
print(f"Pass rate: {passed}/{len(prod_results)} ({passed/len(prod_results)*100:.1f}%)")

# Save final benchmark
benchmark.save("genomics_benchmark_final.jsonld")
print("\n✓ Benchmark saved with results")
```

**Example Output:**

```
✓ Test preset saved: benchmark_presets/quick-test.json
✓ Production preset saved: benchmark_presets/production-full.json

Generating templates...
✓ Templates generated

Running quick test...
✓ Quick test complete: 3 questions

Running production verification...
✓ Production verification complete: 3 questions
Pass rate: 3/3 (100.0%)

✓ Benchmark saved with results
```

## Managing Presets

### Listing Available Presets

List all presets in the presets directory:

```python
from pathlib import Path
import json

presets_dir = Path("benchmark_presets")

if presets_dir.exists():
    preset_files = list(presets_dir.glob("*.json"))
    print(f"Found {len(preset_files)} presets:")

    for preset_file in preset_files:
        with open(preset_file) as f:
            data = json.load(f)
            print(f"\n  {preset_file.name}")
            print(f"    Name: {data['name']}")
            print(f"    Description: {data.get('description', 'N/A')}")
            print(f"    Created: {data['created_at']}")
else:
    print("No presets directory found")
```

### Updating a Preset

To update a preset, load it, modify the configuration, and save it again:

```python
from pathlib import Path

# Load existing preset
preset_path = Path("benchmark_presets/genomics-standard-config.json")
config = VerificationConfig.from_preset(preset_path)

# Modify configuration
config.replicate_count = 5  # Increase replication
config.abstention_enabled = True  # Enable abstention detection

# Save updated configuration (overwrites existing file)
metadata = config.save_preset(
    name="Genomics Standard Config",  # Same name overwrites
    description="Updated with 5 replicates and abstention detection"
)

print(f"✓ Preset updated: {metadata['filepath']}")
```

### Deleting a Preset

Delete a preset file:

```python
from pathlib import Path

preset_path = Path("benchmark_presets/old-config.json")

if preset_path.exists():
    preset_path.unlink()
    print(f"✓ Deleted preset: {preset_path}")
else:
    print(f"Preset not found: {preset_path}")
```

## Common Preset Scenarios

### Scenario 1: Quick Test vs Full Evaluation

Create two presets for different thoroughness levels:

```python
# Quick test: Minimal configuration for fast feedback
quick_config = VerificationConfig(
    answering_models=[model_config],
    parsing_models=[model_config],
    replicate_count=1,
    rubric_enabled=False,
    deep_judgment_enabled=False,
    abstention_enabled=False
)

quick_config.save_preset(
    name="Quick Test",
    description="Fast smoke test configuration"
)

# Full evaluation: Comprehensive configuration
full_config = VerificationConfig(
    answering_models=[model_config],
    parsing_models=[model_config],
    replicate_count=5,
    rubric_enabled=True,
    deep_judgment_enabled=True,
    abstention_enabled=True,
    deep_judgment_max_excerpts_per_attribute=3
)

full_config.save_preset(
    name="Full Evaluation",
    description="Comprehensive configuration with all features"
)
```

**Usage:**

```python
# During development: Use quick test
config = VerificationConfig.from_preset(Path("benchmark_presets/quick-test.json"))
dev_results = benchmark.run_verification(config)

# Before release: Use full evaluation
config = VerificationConfig.from_preset(Path("benchmark_presets/full-evaluation.json"))
final_results = benchmark.run_verification(config)
```

### Scenario 2: Multi-Model Comparison

Create a preset for comparing multiple models:

```python
# Define models to compare
gpt4_mini = ModelConfig(
    id="gpt-4.1-mini",
    model_provider="openai",
    model_name="gpt-4.1-mini",
    temperature=0.0,
    interface="langchain"
)

claude_sonnet = ModelConfig(
    id="claude-sonnet-4",
    model_provider="anthropic",
    model_name="claude-sonnet-4",
    temperature=0.0,
    interface="langchain"
)

# Multi-model comparison configuration
comparison_config = VerificationConfig(
    answering_models=[gpt4_mini, claude_sonnet],  # Both models answer
    parsing_models=[gpt4_mini],  # One model judges
    replicate_count=3,
    rubric_enabled=True
)

comparison_config.save_preset(
    name="GPT-4 vs Claude Comparison",
    description="Compare GPT-4 and Claude on genomics questions"
)
```

**Usage:**

```python
# Load and run comparison
config = VerificationConfig.from_preset(
    Path("benchmark_presets/gpt-4-vs-claude-comparison.json")
)
results = benchmark.run_verification(config)

# Analyze by model
for question_id, result in results.items():
    print(f"\nQuestion: {result.question_text}")
    print(f"  GPT-4 answer: {result.answers.get('gpt-4.1-mini', 'N/A')}")
    print(f"  Claude answer: {result.answers.get('claude-sonnet-4', 'N/A')}")
```

### Scenario 3: Feature-Specific Configurations

Create presets for testing specific features:

```python
# Deep-judgment focused
deep_judgment_config = VerificationConfig(
    answering_models=[model_config],
    parsing_models=[model_config],
    replicate_count=1,
    deep_judgment_enabled=True,
    deep_judgment_max_excerpts_per_attribute=5,
    deep_judgment_fuzzy_match_threshold=0.80
)

deep_judgment_config.save_preset(
    name="Deep Judgment Test",
    description="Testing deep-judgment parsing with 5 excerpts per attribute"
)

# Abstention detection focused
abstention_config = VerificationConfig(
    answering_models=[model_config],
    parsing_models=[model_config],
    replicate_count=1,
    abstention_enabled=True
)

abstention_config.save_preset(
    name="Abstention Detection Test",
    description="Testing abstention detection on safety questions"
)
```

## Sharing Presets

### Exporting Presets

Share preset files with teammates or across projects:

```bash
# Copy preset to shared location
cp benchmark_presets/genomics-standard-config.json /shared/presets/

# Or commit to version control
git add benchmark_presets/production-full.json
git commit -m "Add production verification preset"
git push
```

### Importing Presets

Import presets shared by others:

```bash
# Copy from shared location
cp /shared/presets/team-standard.json benchmark_presets/

# Or pull from version control
git pull
# Preset files appear in benchmark_presets/
```

### Using Presets Across Projects

Use the same preset in multiple projects:

```python
# Project 1: Genomics benchmark
benchmark1 = Benchmark.load("genomics_benchmark.jsonld")
config = VerificationConfig.from_preset(
    Path("/shared/presets/standard-config.json")
)
results1 = benchmark1.run_verification(config)

# Project 2: Drug mechanism benchmark
benchmark2 = Benchmark.load("drug_mechanisms_benchmark.jsonld")
config = VerificationConfig.from_preset(
    Path("/shared/presets/standard-config.json")  # Same config
)
results2 = benchmark2.run_verification(config)
```

## Best Practices

### 1. Use Descriptive Names

**Good names:**
- "GPT-4 Production Config"
- "Quick Smoke Test"
- "Claude with Deep Judgment"
- "Multi-Model Comparison Setup"

**Avoid:**
- Vague names: "Test 1", "Config", "Setup"
- Timestamp-only names: "2025-11-03"
- Overly long names (keep under 50 characters)

### 2. Add Meaningful Descriptions

Include context about when and why to use the preset:

```python
config.save_preset(
    name="Production Genomics",
    description="Standard production configuration for genomics benchmarks. "
                "Uses 3 replicates, enables rubrics and deep-judgment. "
                "Suitable for final evaluations before publication."
)
```

### 3. Organize by Purpose

Create separate presets for different scenarios:

```python
# Development presets
quick_test_config.save_preset(name="Dev: Quick Test", description="...")
debug_config.save_preset(name="Dev: Debug Mode", description="...")

# Production presets
standard_config.save_preset(name="Prod: Standard", description="...")
comprehensive_config.save_preset(name="Prod: Comprehensive", description="...")

# Experiment presets
ablation_config.save_preset(name="Exp: Ablation Study", description="...")
```

### 4. Version Control Your Presets

Track preset files in version control:

```bash
git add benchmark_presets/
git commit -m "Add genomics benchmark presets"
```

This allows you to:
- Track changes to configurations over time
- Revert to previous configurations
- Share presets with teammates
- Document configuration evolution

### 5. Test Presets After Loading

Verify that loaded presets work as expected:

```python
# Load preset
config = VerificationConfig.from_preset(Path("my-preset.json"))

# Verify configuration
print(f"Answering models: {len(config.answering_models)}")
print(f"Parsing models: {len(config.parsing_models)}")
print(f"Replicate count: {config.replicate_count}")
print(f"Deep-judgment: {config.deep_judgment_enabled}")
print(f"Abstention: {config.abstention_enabled}")

# Run small test
test_results = benchmark.run_verification(
    config,
    question_ids=list(benchmark.questions.keys())[:2]  # Just 2 questions
)
print(f"Test passed: {len(test_results)} questions verified")
```

## Troubleshooting

### Issue 1: Preset File Not Found

**Symptom**: `FileNotFoundError` when loading preset

**Solution**:

```python
from pathlib import Path

preset_path = Path("benchmark_presets/my-config.json")

if not preset_path.exists():
    print(f"Preset not found: {preset_path}")
    print("Available presets:")
    for p in Path("benchmark_presets").glob("*.json"):
        print(f"  - {p.name}")
else:
    config = VerificationConfig.from_preset(preset_path)
```

### Issue 2: Invalid Preset Configuration

**Symptom**: `ValidationError` when loading preset

**Solution**:

```python
import json
from pathlib import Path

preset_path = Path("benchmark_presets/broken-config.json")

# Inspect preset file
with open(preset_path) as f:
    data = json.load(f)
    print("Preset contents:")
    print(json.dumps(data, indent=2))

# Check for common issues
if "config" not in data:
    print("ERROR: Missing 'config' field")
elif "answering_models" not in data["config"]:
    print("ERROR: Missing 'answering_models' in config")
```

### Issue 3: Preset Directory Not Found

**Symptom**: Preset directory doesn't exist

**Solution**:

```python
from pathlib import Path

presets_dir = Path("benchmark_presets")

if not presets_dir.exists():
    print("Creating presets directory...")
    presets_dir.mkdir(parents=True, exist_ok=True)
    print(f"✓ Created: {presets_dir}")
```

### Issue 4: Name Already Exists

**Symptom**: Trying to save a preset with a duplicate name

**Solution**:

```python
from pathlib import Path

name = "My Config"
preset_file = Path(f"benchmark_presets/{name.lower().replace(' ', '-')}.json")

if preset_file.exists():
    print(f"Preset '{name}' already exists")
    print("Options:")
    print("  1. Use a different name")
    print("  2. Delete the existing preset")
    print("  3. Use the same name to overwrite (current behavior)")

# Option: Delete existing and save new
preset_file.unlink()
config.save_preset(name=name, description="Updated version")
```

## Next Steps

Once you have presets configured, you can:

- [Verification](../using-karenina/verification.md) - Run verifications with presets
- [Deep-Judgment](deep-judgment.md) - Configure deep-judgment in presets
- [Abstention Detection](abstention-detection.md) - Configure abstention in presets
- [Few-Shot Prompting](few-shot.md) - Add few-shot configuration to presets

## Related Documentation

- [Verification](../using-karenina/verification.md) - Core verification workflow
- [Saving and Loading](../using-karenina/saving-loading.md) - Checkpoint management
- [Deep-Judgment](deep-judgment.md) - Multi-stage parsing
- [Abstention Detection](abstention-detection.md) - Refusal detection
