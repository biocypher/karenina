# Questions Module

The `karenina.questions.extractor` module handles file processing and question extraction from various formats (Excel, CSV, TSV).

## Core Functions

### extract_and_generate_questions

Main entry point for question extraction and Python file generation.

::: karenina.questions.extractor.extract_and_generate_questions

**Parameters:**
- `file_path` (str): Path to input file (Excel, CSV, or TSV)
- `output_path` (str): Destination for generated Python file
- `question_column` (str): Column name containing questions (default: "Question")
- `answer_column` (str): Column name containing answers (default: "Answer")
- `sheet_name` (Optional[str]): Excel sheet name (None for first sheet)
- `return_json` (bool): Return JSON format instead of generating file

**Returns:**
- If `return_json=True`: Dictionary in webapp format
- If `return_json=False`: None (generates Python file)

**Usage Examples:**

```python
# Basic extraction from Excel
extract_and_generate_questions(
    file_path="data/questions.xlsx",
    output_path="questions.py"
)

# Custom column mapping
extract_and_generate_questions(
    file_path="data/survey.csv",
    output_path="survey_questions.py",
    question_column="Query",
    answer_column="Expected_Response"
)

# Specific Excel sheet
extract_and_generate_questions(
    file_path="data/benchmark.xlsx",
    output_path="benchmark_questions.py",
    sheet_name="Hard"
)

# Return JSON format
question_data = extract_and_generate_questions(
    file_path="data/questions.csv",
    output_path="",
    return_json=True
)
```

### extract_questions_from_file

Extract Question objects from file with flexible column mapping.

::: karenina.questions.extractor.extract_questions_from_file

**Parameters:**
- `file_path` (str): Input file path
- `question_column` (str): Question column name
- `answer_column` (str): Answer column name  
- `sheet_name` (Optional[str]): Excel sheet name

**Returns:**
- `List[Question]`: List of extracted Question instances

**Processing Steps:**
1. Read file using pandas
2. Validate required columns exist
3. Filter and clean data (remove NaN, empty strings)
4. Generate MD5 hash IDs for questions
5. Create Question objects

**Usage Example:**

```python
questions = extract_questions_from_file(
    file_path="data/qa_pairs.xlsx",
    question_column="Question_Text",
    answer_column="Answer_Text",
    sheet_name="Dataset"
)

print(f"Extracted {len(questions)} questions")
for q in questions[:3]:
    print(f"ID: {q.id[:8]}... Question: {q.question[:50]}...")
```

### read_file_to_dataframe

Low-level file reading with format detection.

::: karenina.questions.extractor.read_file_to_dataframe

**Supported Formats:**
- Excel: `.xlsx`, `.xls`
- CSV: `.csv`
- TSV: `.tsv`, `.txt` (tab-separated)

**Parameters:**
- `file_path` (str): Path to file
- `sheet_name` (Optional[str]): Excel sheet name

**Returns:**
- `pd.DataFrame`: Loaded data

**Usage Example:**

```python
# Auto-detect format
df = read_file_to_dataframe("data/questions.csv")

# Excel with specific sheet
df = read_file_to_dataframe("data/workbook.xlsx", sheet_name="Questions")
```

### get_file_preview

Generate file preview with metadata for UI components.

::: karenina.questions.extractor.get_file_preview

**Parameters:**
- `file_path` (str): File to preview
- `sheet_name` (Optional[str]): Excel sheet name
- `max_rows` (int): Maximum preview rows (default: 100)

**Returns:**
```python
{
    "success": bool,
    "total_rows": int,
    "columns": List[str],
    "preview_rows": int,
    "data": List[Dict],
    "error": str  # Only if success=False
}
```

**Usage Example:**

```python
preview = get_file_preview("data/large_dataset.xlsx", max_rows=50)

if preview["success"]:
    print(f"File has {preview['total_rows']} rows")
    print(f"Columns: {preview['columns']}")
    print(f"Preview: {len(preview['data'])} rows")
else:
    print(f"Error: {preview['error']}")
```

## Utility Functions

### hash_question

Generate consistent MD5 hash for question identification.

::: karenina.questions.extractor.hash_question

**Parameters:**
- `question_text` (str): Question content

**Returns:**
- `str`: 32-character MD5 hash

**Usage Example:**

```python
q_id = hash_question("What is the capital of France?")
print(q_id)  # "5f4dcc3b5aa765d61d8327deb882cf99"

# Same question always produces same hash
assert hash_question("What is the capital of France?") == q_id
```

### generate_questions_file

Generate executable Python file from Question objects.

::: karenina.questions.extractor.generate_questions_file

**Generated File Structure:**
```python
from karenina.schemas.question_class import Question

# Auto-generated questions from file

question_1 = Question(
    id="hash1",
    question="Question text 1",
    raw_answer="Answer text 1",
    tags=[]
)

question_2 = Question(
    id="hash2", 
    question="Question text 2",
    raw_answer="Answer text 2",
    tags=[]
)

# List of all questions
all_questions = [
    question_1,
    question_2,
]
```

### questions_to_json

Convert Question objects to webapp-compatible JSON format.

::: karenina.questions.extractor.questions_to_json

**Parameters:**
- `questions` (List[Question]): Question instances

**Returns:**
```python
{
    "question_hash_1": {
        "question": "Question text",
        "raw_answer": "Answer text"
        # Note: answer_template added later by generator
    },
    "question_hash_2": {
        "question": "Question text 2", 
        "raw_answer": "Answer text 2"
    }
}
```

## Legacy Functions

### extract_questions_from_excel

Backward-compatible Excel extraction function.

::: karenina.questions.extractor.extract_questions_from_excel

**Note:** This function is deprecated. Use `extract_questions_from_file` with explicit parameters instead.

## Error Handling

Common exceptions and handling:

```python
try:
    questions = extract_questions_from_file("data/questions.xlsx", "Q", "A")
except FileNotFoundError:
    print("File not found")
except ValueError as e:
    if "Missing columns" in str(e):
        print(f"Column mapping error: {e}")
    elif "No valid questions" in str(e):
        print("File contains no valid question-answer pairs")
    else:
        print(f"Data validation error: {e}")
except Exception as e:
    print(f"Unexpected error: {e}")
```

## Data Validation

The module performs automatic data cleaning:

- Removes rows with NaN values in required columns
- Strips whitespace from text fields
- Filters out empty questions or answers
- Converts all text to string type
- Generates unique hash IDs for deduplication

## Performance Notes

- Uses pandas vectorized operations for data processing
- Hash generation is optimized for large datasets
- Memory-efficient processing with iterator patterns
- File preview limits data loading for UI responsiveness

## Integration Examples

### With Answer Generator

```python
# Extract questions
questions = extract_questions_from_file("data/benchmark.xlsx", "Question", "Answer")

# Generate Python file
generate_questions_file(questions, "benchmark_questions.py")

# Use with answer generator
from karenina.answers.generator import generate_answer_templates_from_questions_file
templates = generate_answer_templates_from_questions_file("benchmark_questions.py")
```

### With Web Interface

```python
# Get file preview for UI
preview = get_file_preview("uploaded_file.csv", max_rows=10)

# Extract with custom columns from UI selection
if preview["success"]:
    questions_json = extract_and_generate_questions(
        "uploaded_file.csv",
        "",
        question_column=selected_question_col,
        answer_column=selected_answer_col,
        return_json=True
    )
```