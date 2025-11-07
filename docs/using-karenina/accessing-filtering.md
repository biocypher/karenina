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

## Accessing Questions

### Basic Access Patterns

```python
# Get all questions
all_questions = benchmark.questions

# Get question count
question_count = len(benchmark)

# Access by index
first_question = benchmark.questions[0]
last_question = benchmark.questions[-1]

# Iterate through questions
for question in benchmark.questions:
    print(f"Question: {question.content}")
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

```python
# Get questions with templates assigned
finished_questions = benchmark.get_finished_questions()

# Get questions without templates
unfinished_questions = benchmark.get_unfinished_questions()

# Check status
for question in benchmark.questions:
    status = "finished" if question.has_template() else "unfinished"
    print(f"{question.content[:50]}... - {status}")
```

### Template Status Filtering

```python
# Questions with generated templates
templated = [q for q in benchmark.questions if q.answer_template is not None]

# Questions needing templates
needs_templates = [q for q in benchmark.questions if q.answer_template is None]

print(f"Templated: {len(templated)}, Needs templates: {len(needs_templates)}")
```

## Searching Questions by Content

### Simple Text Search

```python
# Search in question content
def search_questions(benchmark, search_term):
    return [q for q in benchmark.questions
            if search_term.lower() in q.content.lower()]

# Find questions about "machine learning"
ml_questions = search_questions(benchmark, "machine learning")
```

### Regular Expression Search

```python
import re

def regex_search(benchmark, pattern):
    compiled_pattern = re.compile(pattern, re.IGNORECASE)
    return [q for q in benchmark.questions
            if compiled_pattern.search(q.content)]

# Find questions that are asking for explanations
explanation_questions = regex_search(benchmark, r"\b(explain|describe|what is)\b")
```

### Advanced Text Search

```python
def advanced_search(benchmark, terms, match_all=False):
    """
    Search for questions containing specific terms.

    Args:
        terms: List of search terms
        match_all: If True, question must contain ALL terms; if False, ANY term
    """
    results = []
    for question in benchmark.questions:
        content_lower = question.content.lower()
        if match_all:
            # All terms must be present
            if all(term.lower() in content_lower for term in terms):
                results.append(question)
        else:
            # Any term can be present
            if any(term.lower() in content_lower for term in terms):
                results.append(question)
    return results

# Find questions about physics OR chemistry
science_questions = advanced_search(benchmark, ["physics", "chemistry"], match_all=False)

# Find questions about "quantum" AND "mechanics"
quantum_mechanics = advanced_search(benchmark, ["quantum", "mechanics"], match_all=True)
```

## Filtering by Metadata

### Category-based Filtering

```python
# Filter by category
def filter_by_category(benchmark, category):
    return [q for q in benchmark.questions
            if q.metadata.get("category") == category]

math_questions = filter_by_category(benchmark, "mathematics")
science_questions = filter_by_category(benchmark, "science")
```

### Difficulty-based Filtering

```python
# Filter by difficulty level
def filter_by_difficulty(benchmark, difficulty_levels):
    if isinstance(difficulty_levels, str):
        difficulty_levels = [difficulty_levels]

    return [q for q in benchmark.questions
            if q.metadata.get("difficulty") in difficulty_levels]

easy_questions = filter_by_difficulty(benchmark, "easy")
hard_questions = filter_by_difficulty(benchmark, ["hard", "expert"])
```

### Multi-criteria Filtering

```python
def filter_questions(benchmark, **criteria):
    """Filter questions by multiple metadata criteria."""
    results = []
    for question in benchmark.questions:
        match = True
        for key, value in criteria.items():
            if isinstance(value, list):
                if question.metadata.get(key) not in value:
                    match = False
                    break
            else:
                if question.metadata.get(key) != value:
                    match = False
                    break
        if match:
            results.append(question)
    return results

# Find intermediate-level math questions
intermediate_math = filter_questions(
    benchmark,
    category="mathematics",
    difficulty="intermediate"
)

# Find questions from specific sources
textbook_questions = filter_questions(
    benchmark,
    source=["textbook-ch1", "textbook-ch2", "textbook-ch3"]
)
```

## Sorting Questions

### Sort by Metadata

```python
# Sort by difficulty (custom order)
difficulty_order = {"easy": 1, "medium": 2, "hard": 3, "expert": 4}

sorted_by_difficulty = sorted(
    benchmark.questions,
    key=lambda q: difficulty_order.get(q.metadata.get("difficulty", "medium"), 2)
)

# Sort by category alphabetically
sorted_by_category = sorted(
    benchmark.questions,
    key=lambda q: q.metadata.get("category", "")
)
```

### Sort by Content Length

```python
# Sort by question length
sorted_by_length = sorted(benchmark.questions, key=lambda q: len(q.content))

# Get shortest and longest questions
shortest = sorted_by_length[0]
longest = sorted_by_length[-1]

print(f"Shortest: {len(shortest.content)} chars")
print(f"Longest: {len(longest.content)} chars")
```

## Advanced Query Patterns

### Complex Filtering with Lambda

```python
# Complex filtering logic
complex_filter = filter(
    lambda q: (
        q.metadata.get("difficulty") in ["medium", "hard"] and
        q.metadata.get("category") == "science" and
        len(q.content) > 100 and
        "experiment" in q.content.lower()
    ),
    benchmark.questions
)

experiment_questions = list(complex_filter)
```

### Question Statistics

```python
# Analyze question distribution
from collections import Counter

# Category distribution
categories = [q.metadata.get("category") for q in benchmark.questions]
category_counts = Counter(categories)

# Difficulty distribution
difficulties = [q.metadata.get("difficulty") for q in benchmark.questions]
difficulty_counts = Counter(difficulties)

print("Category distribution:", dict(category_counts))
print("Difficulty distribution:", dict(difficulty_counts))
```

## Bulk Operations on Filtered Questions

### Update Metadata for Filtered Questions

```python
# Add tags to all science questions
science_questions = filter_by_category(benchmark, "science")
for question in science_questions:
    if "tags" not in question.metadata:
        question.metadata["tags"] = []
    question.metadata["tags"].append("STEM")
```

### Generate Templates for Filtered Questions

```python
# Generate templates only for unfinished questions
unfinished = benchmark.get_unfinished_questions()

for question in unfinished:
    question.generate_template(
        model_config=model_config,
        system_prompt="Create evaluation template"
    )
```

## Next Steps

Once you can effectively access and filter questions:

- [Set up templates](templates.md) for evaluation structure
- [Configure rubrics](rubrics.md) for assessment criteria
- [Run verification](verification.md) to evaluate responses
