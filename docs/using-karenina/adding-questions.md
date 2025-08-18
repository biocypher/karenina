# Adding Questions to a Benchmark

This guide covers different methods for adding questions to your benchmark, including manual creation and automatic extraction from files.

## Manual Question Creation

### Basic Question Addition

Add questions directly using the `add_question` method:

```python
from karenina import Benchmark

benchmark = Benchmark(name="sample-benchmark")

# Add a simple question
benchmark.add_question(
    content="What is the capital of France?",
    expected_answer="Paris"
)
```

### Questions with Metadata

Enhance questions with rich metadata for better organization:

```python
benchmark.add_question(
    content="Explain the concept of photosynthesis",
    expected_answer="Photosynthesis is the process by which plants convert light energy into chemical energy",
    metadata={
        "category": "biology",
        "difficulty": "intermediate",
        "topic": "plant-biology",
        "estimated_time": "5-minutes",
        "source": "biology-textbook-ch3"
    }
)
```

## Automatic Question Extraction from Files

Karenina provides utilities to extract questions from various file formats automatically.

### Supported File Types

- **CSV** (Comma-separated values)
- **TSV** (Tab-separated values)
- **Excel** (.xlsx, .xls)
- **JSON** (structured question data)

### Basic File Extraction

```python
from karenina.questions import extract_questions_from_file

# Extract questions from a CSV file
questions = extract_questions_from_file("questions.csv")

# Add all extracted questions to benchmark
for question in questions:
    benchmark.add_question(**question)
```

### Example CSV Format

Here's a sample CSV structure that works well with the extraction utility:

```csv
question,expected_answer,category,difficulty,topic
"What is 2 + 2?","4","mathematics","easy","arithmetic"
"Explain Newton's first law","An object at rest stays at rest unless acted upon by a force","physics","medium","mechanics"
"What is the chemical formula for water?","H2O","chemistry","easy","basic-compounds"
```

### Automatic Data Cleaning

The extraction process automatically performs several data cleaning steps:

1. **Whitespace normalization** - Removes leading/trailing spaces and normalizes internal spacing
2. **Empty row filtering** - Skips rows where essential fields (question content) are empty
3. **Encoding detection** - Automatically detects and handles different text encodings
4. **Type coercion** - Converts string representations to appropriate data types
5. **Null value handling** - Replaces various null indicators (`null`, `None`, `N/A`, empty strings) with proper null values
6. **Column name normalization** - Standardizes column headers to consistent naming conventions

### Advanced Extraction Options

```python
# Extract with custom column mapping
questions = extract_questions_from_file(
    "custom_format.csv",
    column_mapping={
        "prompt": "content",
        "solution": "expected_answer",
        "subject": "category"
    }
)

# Extract with filtering
questions = extract_questions_from_file(
    "large_dataset.csv",
    filter_criteria={"difficulty": ["easy", "medium"]}
)
```

## Question Metadata Attributes

### Standard Metadata Attributes

| Attribute | Type | Description |
|-----------|------|-------------|
| `content` | `str` | The question text or prompt |
| `expected_answer` | `str` | The correct/expected response |
| `category` | `str` | Subject area or domain |
| `difficulty` | `str` | Difficulty level (easy, medium, hard) |
| `topic` | `str` | Specific topic within the category |
| `source` | `str` | Origin of the question |
| `tags` | `List[str]` | Searchable tags |

### Accessing Question Metadata

```python
# Get a specific question
question = benchmark.questions[0]

# Access metadata
print(f"Category: {question.metadata.get('category')}")
print(f"Difficulty: {question.metadata.get('difficulty')}")
print(f"Topic: {question.metadata.get('topic')}")
```

## Custom Metadata

### Adding Custom Fields

You can extend questions with any custom metadata:

```python
benchmark.add_question(
    content="Design a RESTful API for a library management system",
    metadata={
        "category": "software-engineering",
        "skill_type": "system-design",
        "estimated_duration": "30-minutes",
        "prerequisites": ["REST-concepts", "database-design"],
        "evaluation_criteria": ["scalability", "security", "clarity"],
        "industry_relevance": "high"
    }
)
```

### Batch Metadata Updates

Update metadata for multiple questions:

```python
# Add industry tag to all software engineering questions
for question in benchmark.questions:
    if question.metadata.get("category") == "software-engineering":
        question.metadata["industry_focus"] = "technology"
```

## Validation and Quality Control

### Question Content Validation

```python
# Validate questions meet minimum requirements
for question in benchmark.questions:
    assert len(question.content) > 10, "Question too short"
    assert question.metadata.get("category"), "Category required"
```

### Metadata Consistency

```python
# Ensure consistent category values
valid_categories = ["math", "science", "literature", "history"]
for question in benchmark.questions:
    category = question.metadata.get("category")
    assert category in valid_categories, f"Invalid category: {category}"
```

## Next Steps

Once you have questions in your benchmark:

- [Access and filter](accessing-filtering.md) questions for analysis
- [Add templates](templates.md) to define evaluation structure
- [Set up rubrics](rubrics.md) for scoring criteria
