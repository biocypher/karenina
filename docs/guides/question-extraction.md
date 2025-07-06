# Question Extraction Guide

This guide covers the complete workflow for extracting questions from files and preparing them for benchmarking.

## Overview

Question extraction transforms raw data files (Excel, CSV, TSV) into structured `Question` objects with unique identifiers, enabling consistent benchmarking workflows.

## Supported File Formats

### Excel Files (.xlsx, .xls)
```python
from karenina.questions.extractor import extract_questions_from_file

# Single sheet extraction
questions = extract_questions_from_file(
    file_path="data/benchmark.xlsx",
    question_column="Question",
    answer_column="Answer"
)

# Specific sheet
questions = extract_questions_from_file(
    file_path="data/workbook.xlsx",
    question_column="Question",
    answer_column="Answer",
    sheet_name="Hard_Questions"
)
```

### CSV Files (.csv)
```python
# Standard CSV
questions = extract_questions_from_file(
    file_path="data/qa_pairs.csv",
    question_column="query",
    answer_column="response"
)
```

### TSV Files (.tsv, .txt)
```python
# Tab-separated values
questions = extract_questions_from_file(
    file_path="data/questions.tsv",
    question_column="question_text",
    answer_column="expected_answer"
)
```

## Complete Extraction Workflow

### Step 1: File Preview and Validation

```python
from karenina.questions.extractor import get_file_preview

# Preview file structure
preview = get_file_preview("data/questions.xlsx", max_rows=5)

if preview["success"]:
    print(f"File contains {preview['total_rows']} rows")
    print(f"Available columns: {preview['columns']}")

    # Display sample data
    for i, row in enumerate(preview['data'][:3]):
        print(f"Row {i+1}: {row}")
else:
    print(f"Error reading file: {preview['error']}")
```

### Step 2: Column Mapping

```python
# Map your file's columns to standard names
file_columns = preview['columns']
print("Available columns:", file_columns)

# Example mappings for different file structures
mappings = {
    "survey_format": {
        "question_column": "Survey_Question",
        "answer_column": "Expected_Response"
    },
    "qa_format": {
        "question_column": "Q",
        "answer_column": "A"
    },
    "benchmark_format": {
        "question_column": "question_text",
        "answer_column": "ground_truth"
    }
}

# Select appropriate mapping
selected_mapping = mappings["benchmark_format"]
```

### Step 3: Extract Questions

```python
from karenina.questions.extractor import extract_questions_from_file

questions = extract_questions_from_file(
    file_path="data/benchmark.xlsx",
    question_column=selected_mapping["question_column"],
    answer_column=selected_mapping["answer_column"],
    sheet_name="Questions"  # Optional for Excel
)

print(f"Extracted {len(questions)} questions")

# Inspect first few questions
for i, q in enumerate(questions[:3]):
    print(f"\nQuestion {i+1}:")
    print(f"  ID: {q.id}")
    print(f"  Question: {q.question[:100]}...")
    print(f"  Answer: {q.raw_answer[:50]}...")
    print(f"  Tags: {q.tags}")
```

### Step 4: Generate Python File

```python
from karenina.questions.extractor import generate_questions_file

# Generate executable Python file
generate_questions_file(questions, "extracted_questions.py")

print("Generated extracted_questions.py with Question objects")
```

**Generated File Structure:**
```python
from karenina.schemas.question_class import Question

# Auto-generated questions from file

question_1 = Question(
    id="5f4dcc3b5aa765d61d8327deb882cf99",
    question="What is the capital of France?",
    raw_answer="Paris",
    tags=[]
)

question_2 = Question(
    id="098f6bcd4621d373cade4e832627b4f6",
    question="What is 2 + 2?",
    raw_answer="4",
    tags=[]
)

# List of all questions
all_questions = [
    question_1,
    question_2,
]
```

## Data Processing Details

### Hash Generation

Questions receive unique MD5 hash identifiers:

```python
from karenina.questions.extractor import hash_question

question_text = "What is the capital of France?"
question_id = hash_question(question_text)
print(question_id)  # "5f4dcc3b5aa765d61d8327deb882cf99"

# Same question always produces same hash
assert hash_question(question_text) == question_id
```

### Data Cleaning

The extraction process automatically:

1. **Removes NaN values** from required columns
2. **Strips whitespace** from text fields
3. **Filters empty strings** in questions or answers
4. **Converts to string type** for consistency

```python
# Example of data cleaning in action
import pandas as pd

# Raw data with issues
raw_data = pd.DataFrame({
    'Question': ['What is Paris?', '  ', None, 'Valid question?'],
    'Answer': ['Capital of France', 'No answer', '', 'Valid answer']
})

# After extraction (conceptual)
# Row 1: ✓ Kept (valid question and answer)
# Row 2: ✗ Removed (empty question after strip)
# Row 3: ✗ Removed (NaN question)
# Row 4: ✓ Kept (valid question and answer)
```

## Advanced Usage Patterns

### Batch Processing Multiple Files

```python
import os
from pathlib import Path

def batch_extract_questions(data_dir: str, output_dir: str):
    """Extract questions from all files in a directory."""

    data_path = Path(data_dir)
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)

    results = {}

    for file_path in data_path.glob("*.xlsx"):
        try:
            # Get file preview to understand structure
            preview = get_file_preview(str(file_path))

            if not preview["success"]:
                continue

            # Assume standard column names, adapt as needed
            questions = extract_questions_from_file(
                str(file_path),
                "Question",
                "Answer"
            )

            # Generate output file
            output_file = output_path / f"{file_path.stem}_questions.py"
            generate_questions_file(questions, str(output_file))

            results[file_path.name] = len(questions)

        except Exception as e:
            print(f"Error processing {file_path.name}: {e}")

    return results

# Usage
results = batch_extract_questions("data/raw", "data/processed")
print("Extraction results:", results)
```

### JSON Output for Web Applications

```python
from karenina.questions.extractor import extract_and_generate_questions

# Extract as JSON instead of Python file
questions_json = extract_and_generate_questions(
    file_path="data/questions.xlsx",
    output_path="",  # Not used when return_json=True
    question_column="Question",
    answer_column="Answer",
    return_json=True
)

print("JSON format:")
for q_id, q_data in list(questions_json.items())[:2]:
    print(f"  {q_id}: {q_data}")

# JSON structure:
# {
#   "question_hash": {
#     "question": "Question text",
#     "raw_answer": "Answer text"
#   }
# }
```

### Custom Validation and Filtering

```python
def extract_with_custom_validation(file_path: str) -> list:
    """Extract questions with custom validation rules."""

    questions = extract_questions_from_file(file_path, "Question", "Answer")

    validated_questions = []

    for q in questions:
        # Custom validation rules
        if len(q.question) < 10:
            print(f"Skipping short question: {q.question}")
            continue

        if not q.question.endswith('?'):
            print(f"Skipping non-question: {q.question}")
            continue

        if len(q.raw_answer) < 2:
            print(f"Skipping minimal answer: {q.raw_answer}")
            continue

        # Add custom tags based on content
        tags = []
        if any(word in q.question.lower() for word in ['what', 'which', 'where']):
            tags.append('factual')
        if any(word in q.question.lower() for word in ['how', 'why']):
            tags.append('explanatory')
        if any(word in q.question.lower() for word in ['calculate', 'compute', 'solve']):
            tags.append('computational')

        # Update question with tags
        q.tags = tags
        validated_questions.append(q)

    return validated_questions

# Usage
validated_questions = extract_with_custom_validation("data/questions.xlsx")
print(f"Validated {len(validated_questions)} questions")
```

## Error Handling and Troubleshooting

### Common Issues

```python
try:
    questions = extract_questions_from_file("data/questions.xlsx", "Q", "A")
except FileNotFoundError:
    print("File not found - check file path")
except ValueError as e:
    if "Missing columns" in str(e):
        print("Column names don't match file structure")
        # Show available columns
        preview = get_file_preview("data/questions.xlsx")
        print("Available columns:", preview.get('columns', []))
    elif "No valid questions" in str(e):
        print("File contains no usable question-answer pairs")
    else:
        print(f"Data validation error: {e}")
except Exception as e:
    print(f"Unexpected error: {e}")
```

### File Format Detection Issues

```python
from pathlib import Path

def safe_file_extraction(file_path: str):
    """Safely extract with format detection."""

    path_obj = Path(file_path)

    if not path_obj.exists():
        return {"error": "File not found"}

    # Check file size
    file_size = path_obj.stat().st_size
    if file_size == 0:
        return {"error": "Empty file"}

    if file_size > 100 * 1024 * 1024:  # 100MB
        return {"error": "File too large (>100MB)"}

    # Try extraction
    try:
        preview = get_file_preview(file_path, max_rows=10)

        if not preview["success"]:
            return {"error": f"Cannot read file: {preview['error']}"}

        # Suggest column mappings
        columns = preview["columns"]
        question_candidates = [col for col in columns if 'question' in col.lower()]
        answer_candidates = [col for col in columns if any(word in col.lower() for word in ['answer', 'response', 'reply'])]

        return {
            "success": True,
            "columns": columns,
            "question_candidates": question_candidates,
            "answer_candidates": answer_candidates,
            "preview": preview["data"][:3]
        }

    except Exception as e:
        return {"error": str(e)}

# Usage
result = safe_file_extraction("data/questions.xlsx")
if "error" in result:
    print(f"Error: {result['error']}")
else:
    print("Suggested question columns:", result["question_candidates"])
    print("Suggested answer columns:", result["answer_candidates"])
```

## Performance Optimization

### Large File Handling

```python
def extract_large_file(file_path: str, chunk_size: int = 1000):
    """Process large files in chunks."""

    import pandas as pd

    # Read file in chunks for memory efficiency
    chunk_reader = pd.read_csv(file_path, chunksize=chunk_size)

    all_questions = []

    for chunk_num, chunk in enumerate(chunk_reader):
        print(f"Processing chunk {chunk_num + 1}")

        # Process chunk as temporary file
        temp_file = f"temp_chunk_{chunk_num}.csv"
        chunk.to_csv(temp_file, index=False)

        try:
            chunk_questions = extract_questions_from_file(temp_file, "Question", "Answer")
            all_questions.extend(chunk_questions)
        finally:
            os.remove(temp_file)

    return all_questions
```

### Parallel Processing

```python
import concurrent.futures
from functools import partial

def parallel_file_extraction(file_paths: list, question_col: str, answer_col: str):
    """Extract questions from multiple files in parallel."""

    extract_func = partial(
        extract_questions_from_file,
        question_column=question_col,
        answer_column=answer_col
    )

    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
        future_to_file = {
            executor.submit(extract_func, file_path): file_path
            for file_path in file_paths
        }

        results = {}
        for future in concurrent.futures.as_completed(future_to_file):
            file_path = future_to_file[future]
            try:
                questions = future.result()
                results[file_path] = questions
            except Exception as e:
                print(f"Error processing {file_path}: {e}")
                results[file_path] = []

    return results

# Usage
file_list = ["data/set1.xlsx", "data/set2.xlsx", "data/set3.xlsx"]
all_results = parallel_file_extraction(file_list, "Question", "Answer")
```

## Integration with Pipeline

### With Answer Generation

```python
from karenina.questions.extractor import extract_and_generate_questions
from karenina.answers.generator import generate_answer_templates_from_questions_file

# Complete pipeline
def question_to_templates_pipeline(input_file: str, questions_file: str):
    """Full pipeline from file to answer templates."""

    # Step 1: Extract questions
    extract_and_generate_questions(
        file_path=input_file,
        output_path=questions_file,
        question_column="Question",
        answer_column="Answer"
    )

    # Step 2: Generate answer templates
    templates = generate_answer_templates_from_questions_file(questions_file)

    print(f"Generated {len(templates)} answer templates")
    return templates

# Usage
templates = question_to_templates_pipeline("data/benchmark.xlsx", "benchmark_questions.py")
```
