# Manual Traces

This guide explains how to use pre-recorded LLM responses for testing and debugging without making live API calls.

## What are Manual Traces?

**Manual traces** are pre-recorded LLM responses stored in JSON files that can be replayed during verification instead of making live API calls. This enables deterministic testing, debugging, and regression testing without API costs or response variability.

Think of manual traces as a "recording" of what an LLM would say, which you can replay as many times as needed.

## Why Use Manual Traces?

**Benefits**:
- **Zero API costs**: No LLM API calls during verification
- **Instant execution**: No network latency
- **100% reproducible**: Same input always produces same output
- **Debug specific cases**: Replay problematic responses for debugging
- **Regression testing**: Ensure verification logic doesn't break
- **Edge case testing**: Test unusual responses that are hard to trigger live

**Use Cases**:
- Debugging parsing issues with specific responses
- Testing verification logic changes without API costs
- Creating reproducible test suites
- Validating templates with ideal responses
- Testing edge cases (empty responses, malformed data, etc.)

## Trace File Format

Manual traces are stored as JSON files with this structure:

```json
{
  "question_id_1": {
    "model_name": "gpt-4.1-mini",
    "answering_response": "The capital of France is Paris.",
    "parsing_response": "{\"capital\": \"Paris\", \"country\": \"France\"}"
  },
  "question_id_2": {
    "model_name": "gpt-4.1-mini",
    "answering_response": "The answer is 42.",
    "parsing_response": "{\"answer\": 42}"
  }
}
```

**Field Descriptions**:
- **Top-level keys**: Question IDs (must match your benchmark)
- **model_name**: Model identifier (for metadata only)
- **answering_response**: Raw LLM response from answering model
- **parsing_response**: Raw LLM response from parsing model

## Creating Manual Traces

### Method 1: Write Traces Manually

Create a JSON file with your desired responses:

```python
import json
from pathlib import Path

# Define manual traces
traces = {
    "q1_venetoclax": {
        "model_name": "gpt-4.1-mini",
        "answering_response": "The approved drug target of Venetoclax is BCL-2 (B-cell lymphoma 2), an anti-apoptotic protein.",
        "parsing_response": '{"target_protein": "BCL-2", "full_name": "B-cell lymphoma 2"}'
    },
    "q2_hbb_location": {
        "model_name": "gpt-4.1-mini",
        "answering_response": "The HBB gene is located on chromosome 11, specifically at position 11p15.4.",
        "parsing_response": '{"chromosome": "Chromosome 11", "position": "11p15.4"}'
    }
}

# Save to file
trace_path = Path("traces/genomics_traces.json")
trace_path.parent.mkdir(parents=True, exist_ok=True)

with open(trace_path, "w") as f:
    json.dump(traces, f, indent=2)

print(f"Traces saved to {trace_path}")
```

### Method 2: Extract from Verification Results

Record responses from a live verification run:

```python
import json
from pathlib import Path
from karenina import Benchmark
from karenina.schemas import ModelConfig, VerificationConfig

# Run verification
benchmark = Benchmark.create(
    name="Genomics Benchmark",
    description="Testing genomics knowledge",
    version="1.0.0"
)

benchmark.add_question(
    question="What is the approved drug target of Venetoclax?",
    raw_answer="BCL2",
    author={"name": "Pharma Curator"}
)

model_config = ModelConfig(
    model_name="gpt-4.1-mini",
    model_provider="openai",
    temperature=0.0
)

benchmark.generate_all_templates(model_config=model_config)

config = VerificationConfig(
    answering_models=[model_config],
    parsing_models=[model_config],
    replicate_count=1
)

results = benchmark.run_verification(config)

# Extract traces from results
traces = {}
for question_id, result in results.items():
    traces[question_id] = {
        "model_name": result.models_used[0] if result.models_used else "unknown",
        "answering_response": result.raw_llm_response,
        "parsing_response": result.raw_llm_response  # Simplified
    }

# Save traces
trace_path = Path("traces/extracted_traces.json")
with open(trace_path, "w") as f:
    json.dump(traces, f, indent=2)

print(f"Extracted {len(traces)} traces from verification results")
```

## Using Manual Traces

### Basic Usage

Use manual traces by setting the model provider to `"manual"`:

```python
from pathlib import Path
from karenina import Benchmark
from karenina.schemas import ModelConfig, VerificationConfig

# Load benchmark
benchmark = Benchmark.load_checkpoint(Path("checkpoints/genomics.json"))

# Configure verification to use manual traces
model_config = ModelConfig(
    model_name="manual",  # Use manual traces
    model_provider="manual",  # Provider must be "manual"
    temperature=0.0
)

config = VerificationConfig(
    answering_models=[model_config],
    parsing_models=[model_config],
    replicate_count=1,
    # Provide trace file path
    manual_trace_file=Path("traces/genomics_traces.json")
)

# Run verification with manual traces (no API calls!)
results = benchmark.run_verification(config)

# Analyze results
success_count = sum(1 for r in results.values() if r.verify_result)
print(f"Success rate: {success_count}/{len(results)}")
print("✓ No API calls were made!")
```

### Complete Example

This example shows the full workflow from creating traces to using them:

```python
import json
from pathlib import Path
from karenina import Benchmark
from karenina.schemas import ModelConfig, VerificationConfig

# Step 1: Create benchmark
benchmark = Benchmark.create(
    name="Genomics Knowledge Test",
    description="Testing drug target knowledge",
    version="1.0.0"
)

benchmark.add_question(
    question="What is the approved drug target of Venetoclax?",
    raw_answer="BCL2",
    author={"name": "Pharma Curator"}
)

benchmark.add_question(
    question="Which chromosome contains the HBB gene?",
    raw_answer="Chromosome 11",
    author={"name": "Genetics Curator"}
)

# Step 2: Generate templates (one-time, using live API)
model_config_live = ModelConfig(
    model_name="gpt-4.1-mini",
    model_provider="openai",
    temperature=0.0
)

benchmark.generate_all_templates(model_config=model_config_live)

# Step 3: Create manual traces
question_ids = list(benchmark.questions.keys())

traces = {
    question_ids[0]: {
        "model_name": "gpt-4.1-mini",
        "answering_response": "Venetoclax is a BCL-2 inhibitor. The approved drug target is BCL-2 (B-cell lymphoma 2).",
        "parsing_response": '{"target_protein": "BCL2"}'
    },
    question_ids[1]: {
        "model_name": "gpt-4.1-mini",
        "answering_response": "The HBB gene encoding hemoglobin subunit beta is located on chromosome 11.",
        "parsing_response": '{"chromosome": "Chromosome 11"}'
    }
}

# Save traces
trace_path = Path("traces/genomics_test.json")
trace_path.parent.mkdir(parents=True, exist_ok=True)

with open(trace_path, "w") as f:
    json.dump(traces, f, indent=2)

# Step 4: Use manual traces for verification (no API calls)
model_config_manual = ModelConfig(
    model_name="manual",
    model_provider="manual",
    temperature=0.0
)

config = VerificationConfig(
    answering_models=[model_config_manual],
    parsing_models=[model_config_manual],
    replicate_count=1,
    manual_trace_file=trace_path
)

# Run with manual traces
results = benchmark.run_verification(config)

# Step 5: Analyze results
for question_id, result in results.items():
    question = benchmark.questions[question_id]
    print(f"\nQuestion: {question.question}")
    print(f"Expected: {question.raw_answer}")
    print(f"Verification: {'✓ PASS' if result.verify_result else '✗ FAIL'}")
    print(f"API calls: 0 (used manual traces)")
```

## Common Use Cases

### Use Case 1: Debugging Parsing Failures

When a specific response fails to parse, use manual traces to debug:

```python
import json
from pathlib import Path
from karenina import Benchmark
from karenina.schemas import ModelConfig, VerificationConfig

# Create trace with problematic response
problematic_traces = {
    "q1": {
        "model_name": "gpt-4.1-mini",
        # This response might fail parsing - let's debug it
        "answering_response": "The drug target is BCL2, also known as B-cell lymphoma 2 protein.",
        "parsing_response": '{"target": "BCL2"}'  # Wrong field name?
    }
}

trace_path = Path("traces/debug.json")
with open(trace_path, "w") as f:
    json.dump(problematic_traces, f, indent=2)

# Load benchmark
benchmark = Benchmark.load_checkpoint(Path("checkpoints/genomics.json"))

# Use manual trace to debug
model_config = ModelConfig(
    model_name="manual",
    model_provider="manual",
    temperature=0.0
)

config = VerificationConfig(
    answering_models=[model_config],
    parsing_models=[model_config],
    replicate_count=1,
    manual_trace_file=trace_path
)

# Run with manual trace - reproduces issue every time
results = benchmark.run_verification(config)

# Debug the failure
for result in results.values():
    if not result.verify_result:
        print("Parsing failed!")
        print(f"Raw response: {result.raw_llm_response}")
        print(f"Error: {result.error_message}")
        # Now you can fix the template or response
```

### Use Case 2: Regression Testing

Create a test suite that runs without API costs:

```python
import json
from pathlib import Path
from karenina import Benchmark
from karenina.schemas import ModelConfig, VerificationConfig

# Step 1: Create comprehensive trace file (do this once)
def create_regression_suite():
    """Create trace file for regression testing."""
    traces = {
        "venetoclax_target": {
            "model_name": "gpt-4.1-mini",
            "answering_response": "Venetoclax targets BCL-2.",
            "parsing_response": '{"target_protein": "BCL2"}'
        },
        "hbb_location": {
            "model_name": "gpt-4.1-mini",
            "answering_response": "HBB is on chromosome 11.",
            "parsing_response": '{"chromosome": "Chromosome 11"}'
        },
        # Add more test cases...
    }

    with open("traces/regression_suite.json", "w") as f:
        json.dump(traces, f, indent=2)

    return traces

# Step 2: Run regression tests
def run_regression_tests():
    """Run verification using manual traces."""
    benchmark = Benchmark.load_checkpoint(
        Path("checkpoints/genomics.json")
    )

    model_config = ModelConfig(
        model_name="manual",
        model_provider="manual",
        temperature=0.0
    )

    config = VerificationConfig(
        answering_models=[model_config],
        parsing_models=[model_config],
        replicate_count=1,
        manual_trace_file=Path("traces/regression_suite.json")
    )

    results = benchmark.run_verification(config)

    # Check all pass
    all_passed = all(r.verify_result for r in results.values())

    if all_passed:
        print("✓ All regression tests passed!")
    else:
        failures = [
            q_id for q_id, r in results.items()
            if not r.verify_result
        ]
        print(f"✗ {len(failures)} regression tests failed:")
        for q_id in failures:
            print(f"  - {q_id}")

    return all_passed

# Create suite once
create_regression_suite()

# Run tests (zero API costs!)
run_regression_tests()
```

### Use Case 3: Edge Case Testing

Test unusual responses that are hard to trigger with live LLMs:

```python
import json
from pathlib import Path

# Create traces for edge cases
edge_case_traces = {
    "empty_response": {
        "model_name": "gpt-4.1-mini",
        "answering_response": "",  # Empty response
        "parsing_response": "{}"
    },
    "malformed_json": {
        "model_name": "gpt-4.1-mini",
        "answering_response": "BCL2",
        "parsing_response": "{invalid json"  # Malformed
    },
    "unexpected_format": {
        "model_name": "gpt-4.1-mini",
        "answering_response": "I don't know",  # Refusal
        "parsing_response": '{"target_protein": null}'
    },
    "very_long_response": {
        "model_name": "gpt-4.1-mini",
        "answering_response": "BCL2 " * 1000,  # Very long
        "parsing_response": '{"target_protein": "BCL2"}'
    }
}

# Save edge cases
trace_path = Path("traces/edge_cases.json")
with open(trace_path, "w") as f:
    json.dump(edge_case_traces, f, indent=2)

print("Edge case traces created")
print("Run verification with these traces to test error handling")
```

### Use Case 4: Template Validation

Validate templates parse ideal responses correctly:

```python
import json
from pathlib import Path
from karenina import Benchmark
from karenina.schemas import ModelConfig, VerificationConfig

# Create ideal responses for template validation
ideal_traces = {
    "q1": {
        "model_name": "gpt-4.1-mini",
        # Perfect format - should definitely pass
        "answering_response": "The drug target is BCL2.",
        "parsing_response": '{"target_protein": "BCL2"}'
    },
    "q2": {
        "model_name": "gpt-4.1-mini",
        # Perfect format
        "answering_response": "The HBB gene is on chromosome 11.",
        "parsing_response": '{"chromosome": "Chromosome 11"}'
    }
}

trace_path = Path("traces/ideal_responses.json")
with open(trace_path, "w") as f:
    json.dump(ideal_traces, f, indent=2)

# Validate templates
benchmark = Benchmark.load_checkpoint(Path("checkpoints/genomics.json"))

model_config = ModelConfig(
    model_name="manual",
    model_provider="manual",
    temperature=0.0
)

config = VerificationConfig(
    answering_models=[model_config],
    parsing_models=[model_config],
    replicate_count=1,
    manual_trace_file=trace_path
)

results = benchmark.run_verification(config)

# All should pass if templates are correct
all_passed = all(r.verify_result for r in results.values())

if all_passed:
    print("✓ Templates validated - all ideal responses parsed correctly")
else:
    print("✗ Template validation failed - fix templates")
    for q_id, result in results.items():
        if not result.verify_result:
            print(f"  Failed: {q_id}")
```

## Troubleshooting

### Issue: Question ID Mismatch

**Error**: `KeyError: 'question_id_xyz'` or missing trace warnings

**Cause**: Trace file question IDs don't match benchmark question IDs.

**Solution**:
```python
import json
from pathlib import Path
from karenina import Benchmark

# Load benchmark and traces
benchmark = Benchmark.load_checkpoint(Path("checkpoints/genomics.json"))

with open("traces/genomics_traces.json", "r") as f:
    traces = json.load(f)

# Compare IDs
benchmark_ids = set(benchmark.questions.keys())
trace_ids = set(traces.keys())

missing = benchmark_ids - trace_ids
extra = trace_ids - benchmark_ids

if missing:
    print("Missing traces for questions:")
    for q_id in missing:
        q = benchmark.questions[q_id]
        print(f"  {q_id}: {q.question}")

if extra:
    print("Extra traces (not in benchmark):")
    for q_id in extra:
        print(f"  {q_id}")
```

### Issue: Invalid JSON Format

**Error**: `JSONDecodeError: Expecting property name`

**Cause**: Trace file has invalid JSON syntax.

**Solution**:
```bash
# Validate JSON before use
python -m json.tool traces/genomics_traces.json

# If valid, prints formatted JSON
# If invalid, shows syntax error location
```

### Issue: Missing Required Fields

**Error**: `KeyError: 'answering_response'`

**Cause**: Trace entries missing required fields.

**Solution**:
```python
import json

def validate_trace_file(trace_path):
    """Validate trace file has all required fields."""
    with open(trace_path, "r") as f:
        traces = json.load(f)

    required = {"model_name", "answering_response", "parsing_response"}

    for q_id, trace in traces.items():
        missing = required - set(trace.keys())
        if missing:
            print(f"Question {q_id} missing: {missing}")
            return False

    print("✓ All traces have required fields")
    return True

# Validate before use
validate_trace_file("traces/genomics_traces.json")
```

### Issue: Verification Still Uses API

**Error**: Unexpected API costs or latency

**Cause**: Model provider not set to `"manual"`.

**Solution**:
```python
# ❌ Wrong - will use live API
model_config = ModelConfig(
    model_name="gpt-4.1-mini",  # Wrong!
    model_provider="openai",     # Wrong!
    temperature=0.0
)

# ✅ Correct - uses manual traces
model_config = ModelConfig(
    model_name="manual",         # Correct
    model_provider="manual",     # Correct
    temperature=0.0
)

config = VerificationConfig(
    answering_models=[model_config],
    parsing_models=[model_config],
    manual_trace_file=Path("traces/genomics_traces.json")
)
```

## Best Practices

### Creating Traces

**Do**:
- Use descriptive question IDs
- Include model name for documentation
- Format JSON with indentation for readability
- Validate JSON syntax before use
- Version control trace files with code

**Don't**:
- Use generic question IDs (q1, q2, q3)
- Leave out model_name (helps with debugging)
- Commit sensitive data in traces
- Create traces with incorrect question IDs

### Using Traces

**Do**:
- Verify question IDs match before running
- Use manual traces for testing and debugging
- Save traces from successful verification runs
- Document what each trace file tests
- Keep trace files organized by purpose

**Don't**:
- Use manual traces for production benchmarking
- Forget to update traces when templates change
- Mix live and manual verification without documentation
- Delete trace files after first use

### Testing

**Do**:
- Create regression test suites with traces
- Test edge cases that are hard to trigger live
- Validate templates with ideal responses
- Use traces to reproduce bugs
- Run trace-based tests in CI/CD

**Don't**:
- Rely only on manual traces (also test live)
- Skip validation of trace file format
- Use outdated traces with new templates
- Test only happy path scenarios

## Performance

**Benefits of Manual Traces**:

| Metric | Manual Traces | Live API |
|--------|--------------|----------|
| **Cost** | $0 | Variable |
| **Speed** | Instant | 1-10 seconds per call |
| **Reproducibility** | 100% | Variable |
| **Rate limits** | None | Provider dependent |
| **Network dependency** | None | Required |

**When to Use**:
- Development and debugging
- Regression testing
- CI/CD pipelines
- Template validation
- Edge case testing

**When NOT to Use**:
- Production benchmarking (need real LLM responses)
- Exploring new questions (no traces exist yet)
- Evaluating LLM capabilities (need live responses)

## Related Documentation

- **Quick Start**: Basic verification workflow
- **Verification**: Complete verification documentation
- **Configuration**: Model and provider configuration
- **Troubleshooting**: Common issues and solutions

## Summary

Manual traces enable:

1. **Zero-cost testing** - No API calls means no costs
2. **Deterministic results** - Same input always produces same output
3. **Fast debugging** - Instantly reproduce any scenario
4. **Regression testing** - Ensure verification logic doesn't break
5. **Edge case coverage** - Test unusual responses easily

**Create traces** from live verification or write them manually, then use by setting `model_provider="manual"` in your configuration.
