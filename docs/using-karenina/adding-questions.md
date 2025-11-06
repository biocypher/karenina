# Adding Questions to a Benchmark

This guide covers different methods for adding questions to your benchmark, including manual creation and automatic extraction from files.

## Manual Question Creation

### Basic Question Addition

Add questions directly using the `add_question` method:

```python
from karenina import Benchmark

benchmark = Benchmark.create(name="Genomics Knowledge Benchmark")

# Add a simple question
question_id = benchmark.add_question(
    question="How many chromosomes are in a human somatic cell?",
    raw_answer="46"
)
```

The `add_question` method returns a unique question ID that you can use to reference the question later.

### Questions with Author Information

Add author metadata to track question provenance:

```python
# Add question with author information
question_id = benchmark.add_question(
    question="What is the approved drug target of Venetoclax?",
    raw_answer="BCL2",
    author={
        "name": "Dr. Sarah Chen",
        "email": "schen@research.edu"
    }
)
```

## Automatic Question Extraction from Files

Karenina provides utilities to extract questions from various file formats automatically. This is useful when you have existing question sets in spreadsheets or structured files.

### Supported File Types

- **Excel** (.xlsx, .xls)
- **CSV** (Comma-separated values)
- **TSV** (Tab-separated values)

### Basic File Extraction

Extract questions from a file and add them to your benchmark:

```python
from karenina.questions.extractor import extract_questions_from_file

# Extract questions from an Excel file
questions = extract_questions_from_file(
    file_path="genomics_questions.xlsx",
    question_column="Question",
    answer_column="Answer"
)

# Add all extracted questions to benchmark
for q in questions:
    benchmark.add_question(**q)

print(f"Added {len(questions)} questions from file")
```

### Example Excel/CSV Format

Here's a sample spreadsheet structure that works well with the extraction utility:

| Question | Answer | Author | Keywords |
|----------|--------|--------|----------|
| How many chromosomes are in a human somatic cell? | 46 | Dr. Smith | genetics, chromosomes |
| What is the approved drug target of Venetoclax? | BCL2 | Dr. Chen | pharmacology, cancer |
| How many protein subunits does hemoglobin A have? | 4 | Dr. Smith | proteins, hemoglobin |

### Automatic Data Cleaning

The extraction process automatically performs several data cleaning steps:

1. **Whitespace normalization** - Removes leading/trailing spaces and normalizes internal spacing
2. **Empty row filtering** - Skips rows where essential fields (question content) are empty
3. **Encoding detection** - Automatically detects and handles different text encodings
4. **Type coercion** - Converts string representations to appropriate data types
5. **Null value handling** - Replaces various null indicators (`null`, `None`, `N/A`, empty strings) with proper null values
6. **Column name normalization** - Standardizes column headers to consistent naming conventions

### Advanced Extraction with Optional Columns

You can extract additional metadata by specifying optional column names:

```python
from karenina.questions.extractor import extract_questions_from_file

# Extract with author and keyword metadata
questions = extract_questions_from_file(
    file_path="comprehensive_questions.xlsx",
    question_column="Question",
    answer_column="Answer",
    author_name_column="Author",      # Optional: author name
    keywords_column="Keywords"         # Optional: comma-separated keywords
)

# Each extracted question will include author and keywords if available
for q in questions:
    benchmark.add_question(**q)
```

## Working with Questions

### Accessing Questions

Once you've added questions, you can access them using their question IDs:

```python
# Get a specific question by ID
question = benchmark.get_question(question_id)

# Access question attributes
print(f"Question text: {question.question}")
print(f"Expected answer: {question.raw_answer}")
print(f"Author: {question.author}")
print(f"Keywords: {question.keywords}")
```

### Listing All Questions

```python
# Get all question IDs
question_ids = list(benchmark.questions.keys())

# Iterate through all questions
for qid in question_ids:
    question = benchmark.get_question(qid)
    print(f"{qid}: {question.question}")
```

### Question Attributes

Each question in Karenina has the following key attributes:

| Attribute | Type | Description |
|-----------|------|-------------|
| `question` | `str` | The question text or prompt |
| `raw_answer` | `str` | The expected answer |
| `author` | `dict` | Author information (name, email) |
| `keywords` | `list[str]` | Searchable keywords or tags |
| `question_id` | `str` | Unique identifier (MD5 hash) |

## Question Organization

### Using Keywords

Keywords help organize and filter questions:

```python
# Add questions with keywords
benchmark.add_question(
    question="What is the role of telomerase in cell division?",
    raw_answer="Telomerase adds telomeric sequences to chromosome ends",
    author={"name": "Dr. Lee"},
    keywords=["cell-biology", "telomeres", "aging"]
)

benchmark.add_question(
    question="Describe the structure of a nucleosome",
    raw_answer="DNA wrapped around histone octamer",
    author={"name": "Dr. Lee"},
    keywords=["chromatin", "epigenetics", "dna-structure"]
)
```

### Batch Addition

Add multiple questions efficiently:

```python
# Prepare question data
genomics_questions = [
    {
        "question": "How many chromosomes are in a human somatic cell?",
        "raw_answer": "46",
        "author": {"name": "Bio Curator"},
        "keywords": ["genetics", "chromosomes"]
    },
    {
        "question": "What is the approved drug target of Venetoclax?",
        "raw_answer": "BCL2",
        "author": {"name": "Bio Curator"},
        "keywords": ["pharmacology", "cancer"]
    },
    {
        "question": "How many protein subunits does hemoglobin A have?",
        "raw_answer": "4",
        "author": {"name": "Bio Curator"},
        "keywords": ["proteins", "hemoglobin"]
    }
]

# Add all questions
for q in genomics_questions:
    benchmark.add_question(**q)
```

## Complete Example

Here's a complete workflow showing both manual and file-based question addition:

```python
from karenina import Benchmark
from karenina.questions.extractor import extract_questions_from_file

# 1. Create benchmark
benchmark = Benchmark.create(
    name="Genomics Knowledge Benchmark",
    description="Testing LLM knowledge of genomics and molecular biology",
    version="1.0.0"
)

# 2. Add questions manually
question_ids = []

qid1 = benchmark.add_question(
    question="How many chromosomes are in a human somatic cell?",
    raw_answer="46",
    author={"name": "Dr. Smith", "email": "smith@example.com"},
    keywords=["genetics", "chromosomes"]
)
question_ids.append(qid1)

qid2 = benchmark.add_question(
    question="What is the approved drug target of Venetoclax?",
    raw_answer="BCL2",
    author={"name": "Dr. Chen", "email": "chen@example.com"},
    keywords=["pharmacology", "cancer"]
)
question_ids.append(qid2)

# 3. Extract additional questions from file
file_questions = extract_questions_from_file(
    file_path="additional_questions.xlsx",
    question_column="Question",
    answer_column="Answer",
    author_name_column="Author",
    keywords_column="Keywords"
)

for q in file_questions:
    qid = benchmark.add_question(**q)
    question_ids.append(qid)

print(f"Total questions: {len(question_ids)}")

# 4. Verify questions were added
for qid in question_ids:
    question = benchmark.get_question(qid)
    print(f"✓ {question.question[:50]}...")
```

---

## Next Steps

Once you have questions in your benchmark, you can:

- [Generate templates](templates.md) to define evaluation structure
- [Set up rubrics](rubrics.md) for qualitative assessment
- [Run verification](verification.md) to evaluate LLM responses
- [Save your benchmark](saving-loading.md) using checkpoints or database

---

## Related Documentation

- [Defining Benchmarks](defining-benchmark.md) - Creating and configuring benchmarks
- [Templates](templates.md) - Structured answer evaluation
- [Rubrics](rubrics.md) - Qualitative assessment criteria
- [Quick Start](../quickstart.md) - End-to-end workflow example
