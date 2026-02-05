# Accessing and Filtering Questions

This guide covers how to access, filter, and search through questions in your benchmark for analysis and management.

**Quick Navigation:**

- [Accessing Questions](#accessing-questions) - Basic access patterns and iteration
- [Filtering by Status](#filtering-by-status) - Finished vs unfinished, template status
- [Searching Questions by Content](#searching-questions-by-content) - Text search, regex, advanced search
- [Filtering by Metadata](#filtering-by-metadata) - Category, difficulty, multi-criteria filtering
- [Sorting Questions](#sorting-questions) - Sort by metadata, content length
- [Advanced Query Patterns](#advanced-query-patterns) - Complex filtering and statistics
- [Bulk Operations](#bulk-operations-on-filtered-questions) - Update metadata, generate templates

---

## Understanding Question Metadata

Each question in a Karenina benchmark has two types of metadata:

### System Metadata (Built-in Fields)

These are standard fields managed by Karenina:

- `id` - Unique question identifier
- `question` - The question text
- `raw_answer` - The expected answer
- `finished` - Boolean flag for template completion status
- `answer_template` - The Answer class code for verification
- `date_created` - Creation timestamp
- `date_modified` - Last modification timestamp
- `author` - Author information (optional dict)
- `sources` - Source documents (optional list)
- `question_rubric` - Question-specific rubric traits

**Access:** Direct fields on question dictionary

```python
question = benchmark.get_question(question_id)
question["question"]  # Question text
question["finished"]  # Completion status
question["author"]    # Author info
```

### Custom Metadata (User-defined Fields)

The `custom_metadata` field is a **dictionary** where you can store any arbitrary key-value pairs specific to your use case:

```python
# Your custom structure - completely flexible!
{
    "custom_metadata": {
        "category": "mathematics",
        "difficulty": "hard",
        "tags": ["algebra", "equations"],
        "source_chapter": 5,
        "estimated_time_minutes": 10,
        # ... any fields you need
    }
}
```

**Access:** Via `custom_metadata` dictionary

```python
question = benchmark.get_question(question_id)
custom = question.get("custom_metadata", {})
category = custom.get("category")
difficulty = custom.get("difficulty")
```

**Important:** Built-in filtering methods (`filter_questions`) work with system metadata. For custom metadata, use the generic filtering methods described below.

---

## Built-in Methods Overview

The Benchmark class provides several built-in methods for accessing and filtering questions:

### Access Methods
- `get_all_questions(ids_only)` - Get all questions (objects by default, IDs if `ids_only=True`)
- `get_question(question_id)` - Get a specific question by ID
- `get_question_ids()` - Get list of all question IDs (convenience wrapper for `get_all_questions(ids_only=True)`)

### System Metadata Filtering
These methods filter by built-in Karenina fields:
- `filter_questions(finished, has_template, has_rubric, author, custom_filter)` - Filter by system fields or custom lambda
- `get_unfinished_questions(ids_only)` - Get unfinished questions (objects by default, IDs if `ids_only=True`)
- `get_finished_questions(ids_only)` - Get finished questions (objects by default, IDs if `ids_only=True`)
- `get_questions_by_author(author)` - Filter by author name
- `get_questions_with_rubric()` - Get questions with rubrics

### Custom Metadata Filtering
These methods work with your `custom_metadata` dictionary:
- `filter_by_custom_metadata(**criteria)` - Filter by custom fields with AND/OR logic
- `filter_by_metadata(field_path, value, match_mode)` - Generic field filtering with dot notation
- `count_by_field(field_path)` - Count questions by any field value

### Search Methods
- `search_questions(query, match_all, fields, case_sensitive, regex)` - Unified search supporting:
  - Single term: `search_questions("machine learning")`
  - Multi-term AND: `search_questions(["quantum", "mechanics"])`
  - Multi-term OR: `search_questions(["python", "java"], match_all=False)`
  - Multi-field: `search_questions("algorithm", fields=["question", "raw_answer"])`
  - Regex: `search_questions(r"what (is|are)", regex=True)`

### Template Methods
- `has_template(question_id)` - Check if question has a template
- `get_missing_templates(ids_only)` - Get questions without templates (objects by default, IDs if `ids_only=True`)

---

## Accessing Questions

### Basic Access Patterns

```python
# Get all questions as dictionaries
all_questions = benchmark.get_all_questions()

# Get question count
question_count = len(benchmark)

# Get list of question IDs
question_ids = benchmark.get_question_ids()

# Get a specific question by ID
question = benchmark.get_question(question_ids[0])

# Iterate through questions
for question in benchmark.get_all_questions():
    print(f"Question: {question['question']}")
```

### Square Bracket Access

Karenina supports convenient square bracket notation for accessing questions:

```python
# Access by index
question = benchmark[0]

# Slice access
first_five = benchmark[0:5]

# Access multiple indices
selected = benchmark[[0, 2, 4]]
```

## Filtering by Status

### Finished vs Unfinished Questions

Questions are considered "finished" when they have both a template and verification results:

> **Note:** When adding questions through the backend API, questions are marked as "finished" by default. The frontend GUI behaves differently and marks questions as "unfinished" until templates are generated. This distinction is important when programmatically creating benchmarks versus using the web interface.

```python
# Get unfinished questions (returns list of question objects by default)
unfinished_questions = benchmark.get_unfinished_questions()

# Iterate directly over the question objects
for question in unfinished_questions:
    print(f"Unfinished: {question['question']}")
    print(f"Answer: {question['raw_answer']}")

# Get only question IDs if needed (for backward compatibility)
unfinished_ids = benchmark.get_unfinished_questions(ids_only=True)

# Get finished questions
finished_questions = benchmark.get_finished_questions()

# Check status for all questions
for question in benchmark.get_all_questions():
    is_finished = question.get("finished", False)
    has_template = benchmark.has_template(question["id"])
    status = "finished" if is_finished and has_template else "unfinished"
    print(f"{question['question'][:50]}... - {status}")
```

### Template Status Filtering

Use the built-in `filter_questions` method for template-based filtering:

```python
# Questions with generated templates
templated = benchmark.filter_questions(has_template=True)

# Questions needing templates
needs_templates = benchmark.filter_questions(has_template=False)

print(f"Templated: {len(templated)}, Needs templates: {len(needs_templates)}")
```

### Combined Status Filtering

The `filter_questions` method supports multiple criteria:

```python
# Filter by finished status, template, rubric, and author
filtered = benchmark.filter_questions(
    finished=True,
    has_template=True,
    has_rubric=True,
    author="John Doe"
)

# Get all finished questions without templates
needs_work = benchmark.filter_questions(finished=True, has_template=False)
```

## Searching Questions by Content

The `search_questions()` method provides flexible text search with support for single/multi-term queries, regex, and case-sensitive matching.

### Simple Text Search

```python
# Search in question text (default)
ml_questions = benchmark.search_questions("machine learning")

# The method returns a list of question dictionaries
for q in ml_questions:
    print(f"Found: {q['question'][:50]}...")
```

### Multi-term Search

```python
# AND logic: question must contain all terms
quantum_mechanics = benchmark.search_questions(["quantum", "mechanics"], match_all=True)

# OR logic: question contains any term
languages = benchmark.search_questions(["python", "javascript", "java"], match_all=False)
```

### Search in Multiple Fields

```python
# Search in both question and answer
algorithm_content = benchmark.search_questions(
    "algorithm",
    fields=["question", "raw_answer"]
)
```

### Advanced Search Options

```python
# Case-sensitive search
python_qs = benchmark.search_questions("Python", case_sensitive=True)

# Regex search
explanation_qs = benchmark.search_questions(r"\b(explain|describe|what is)\b", regex=True)

# Complex: multi-term regex in answers
dna_rna = benchmark.search_questions(
    [r"DNA.*replication", r"RNA.*synthesis"],
    match_all=False,
    fields=["raw_answer"],
    regex=True
)
```

## Filtering by Metadata

### Filtering by System Metadata

Filter by built-in Karenina fields using `filter_questions()`:

```python
# Filter by finished status
finished = benchmark.filter_questions(finished=True)
unfinished = benchmark.filter_questions(finished=False)

# Filter by author
johns_questions = benchmark.filter_questions(author="John Doe")
# Or use the convenience method
johns_questions = benchmark.get_questions_by_author("John Doe")

# Filter by template existence
needs_templates = benchmark.filter_questions(has_template=False)

# Combine multiple system filters
ready_to_verify = benchmark.filter_questions(
    finished=True,
    has_template=True,
    has_rubric=True
)
```

### Filtering by Custom Metadata

Use built-in methods to filter by your custom metadata fields:

```python
# If your custom_metadata has "category" and "difficulty" fields
math_hard = benchmark.filter_by_custom_metadata(category="math", difficulty="hard")

# OR logic for custom metadata (match any criterion)
stem_subjects = benchmark.filter_by_custom_metadata(
    match_all=False,
    category="math",
    subject="science"
)

# Using generic field path filtering with dot notation
math_qs = benchmark.filter_by_metadata("custom_metadata.category", "math")

# Filter by value in a list (for tags/arrays)
algebra_tagged = benchmark.filter_by_metadata("custom_metadata.tags", "algebra", match_mode="in")

# Substring matching
bio_qs = benchmark.filter_by_metadata("custom_metadata.domain", "bio", match_mode="contains")

# Regex matching on custom fields
advanced_qs = benchmark.filter_by_metadata("custom_metadata.level", r"(advanced|expert)", match_mode="regex")
```

### Complex Custom Filtering with Lambda

For complex logic, use the `custom_filter` parameter:

```python
# Complex logic on custom metadata
high_priority_recent = benchmark.filter_questions(
    custom_filter=lambda q: (
        q.get("custom_metadata", {}).get("priority") in ["high", "critical"] and
        q.get("custom_metadata", {}).get("year", 0) >= 2023
    )
)

# Combine system and custom metadata filtering
hard_finished = benchmark.filter_questions(
    finished=True,
    has_template=True,
    custom_filter=lambda q: q.get("custom_metadata", {}).get("difficulty") == "hard"
)

# Filter by custom score threshold
high_scorers = benchmark.filter_questions(
    custom_filter=lambda q: (
        q.get("custom_metadata", {}).get("score", 0) > 7 and
        len(q["question"]) < 200
    )
)
```

### Statistics with Custom Metadata

Use `count_by_field()` for statistics on any field:

```python
# Count by custom metadata field
category_counts = benchmark.count_by_field("custom_metadata.category")
# Result: {"math": 45, "science": 32, "history": 18}

# Count finished vs unfinished
status_counts = benchmark.count_by_field("finished")
# Result: {True: 67, False: 28}

# Count on filtered subset
math_qs = benchmark.filter_by_custom_metadata(category="math")
difficulty_counts = benchmark.count_by_field("custom_metadata.difficulty", questions=math_qs)
# Result: {"easy": 12, "medium": 20, "hard": 13}
```

## Sorting Questions

You can sort questions using Python's `sorted()` function with custom key functions:

```python
# Get all questions first
questions = benchmark.get_all_questions()

# Sort by custom metadata with custom order
difficulty_order = {"easy": 1, "medium": 2, "hard": 3, "expert": 4}
sorted_by_difficulty = sorted(
    questions,
    key=lambda q: difficulty_order.get(q.get("custom_metadata", {}).get("difficulty", "medium"), 2)
)

# Sort by category alphabetically
sorted_by_category = sorted(
    questions,
    key=lambda q: q.get("custom_metadata", {}).get("category", "")
)

# Sort by question length
sorted_by_length = sorted(questions, key=lambda q: len(q.get("question", "")))

# Sort by date created
sorted_by_date = sorted(questions, key=lambda q: q.get("date_created", ""))
```

## Advanced Query Patterns

### Combining Filters and Search

```python
# First filter, then search within results
math_questions = benchmark.filter_by_custom_metadata(category="math")
hard_math_with_calculus = [
    q for q in math_questions
    if q.get("custom_metadata", {}).get("difficulty") == "hard"
    and "calculus" in q["question"].lower()
]

# Or use lambda for the same thing
hard_math_calculus = benchmark.filter_questions(
    custom_filter=lambda q: (
        q.get("custom_metadata", {}).get("category") == "math" and
        q.get("custom_metadata", {}).get("difficulty") == "hard" and
        "calculus" in q["question"].lower()
    )
)
```

### Question Statistics

Use the built-in `count_by_field()` method for statistics:

```python
# Get distribution of any field
category_dist = benchmark.count_by_field("custom_metadata.category")
print("Category distribution:", category_dist)

difficulty_dist = benchmark.count_by_field("custom_metadata.difficulty")
print("Difficulty distribution:", difficulty_dist)

# Statistics on filtered subsets
math_qs = benchmark.filter_by_custom_metadata(category="math")
math_difficulty_dist = benchmark.count_by_field("custom_metadata.difficulty", questions=math_qs)
print("Math questions by difficulty:", math_difficulty_dist)
```

## Bulk Operations on Filtered Questions

### Update System Metadata

```python
# Mark all finished questions as unfinished
# Use ids_only=True since mark_unfinished_batch expects IDs
finished_ids = benchmark.get_finished_questions(ids_only=True)
benchmark.mark_unfinished_batch(finished_ids)

# Update author for specific questions (using default object return)
math_qs = benchmark.filter_by_custom_metadata(category="math")
for q in math_qs:
    benchmark.set_question_author(q["id"], {"name": "Math Team", "email": "math@example.com"})
```

### Update Custom Metadata

```python
# Add tags to all science questions
science_qs = benchmark.filter_by_custom_metadata(category="science")

for question in science_qs:
    question_id = question["id"]
    # Get current custom metadata
    custom_meta = benchmark.get_question_metadata(question_id).get("custom_metadata", {})

    # Add tags
    if "tags" not in custom_meta:
        custom_meta["tags"] = []
    if "STEM" not in custom_meta["tags"]:
        custom_meta["tags"].append("STEM")

    # Update the question
    benchmark.update_question_metadata(question_id, custom_metadata=custom_meta)

# Or use the convenience method for single properties
for q in science_qs:
    benchmark.set_question_custom_property(q["id"], "reviewed", True)
```

### Generate Templates for Filtered Questions

```python
# Generate templates only for unfinished questions
# Use ids_only=True since generate_templates expects a list of IDs
unfinished_ids = benchmark.get_unfinished_questions(ids_only=True)

# Use the bulk generation method
results = benchmark.generate_templates(
    question_ids=unfinished_ids,
    model="gemini-2.0-flash",
    model_provider="google_genai",
    temperature=0
)

# Check results
successful = sum(1 for r in results.values() if r["success"])
print(f"Generated {successful}/{len(unfinished_ids)} templates")

# Alternative: Work with question objects directly
unfinished_questions = benchmark.get_unfinished_questions()
for question in unfinished_questions:
    print(f"Need template for: {question['question']}")
```

## Next Steps

Once you can effectively access and filter questions:

- [Set up templates](templates.md) for evaluation structure
- [Configure rubrics](rubrics.md) for assessment criteria
- [Run verification](verification.md) to evaluate responses
