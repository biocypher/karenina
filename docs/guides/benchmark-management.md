# Benchmark Management Guide

This guide covers the comprehensive Benchmark class, which provides a high-level interface for creating, managing, and executing benchmarks in Karenina. The Benchmark class uses JSON-LD format for full compatibility with the Karenina GUI.

## Table of Contents

- [Overview](#overview)
- [Creating Benchmarks](#creating-benchmarks)
- [Loading and Saving](#loading-and-saving)
- [Managing Questions](#managing-questions)
- [Template Management](#template-management)
- [Metadata Management](#metadata-management)
- [Rubric Management](#rubric-management)
- [Validation and Health Checks](#validation-and-health-checks)
- [Export Capabilities](#export-capabilities)
- [Advanced Features](#advanced-features)
- [GUI Interoperability](#gui-interoperability)

## Overview

The Benchmark class serves as the central orchestrator for all benchmarking operations. It provides:

- **JSON-LD Storage**: Uses schema.org vocabulary for standardized linked data
- **GUI Compatibility**: Full bidirectional interoperability with Karenina GUI
- **Comprehensive Metadata**: Track everything from creation to evaluation
- **Built-in Validation**: Ensure benchmark integrity and readiness
- **Multiple Export Formats**: CSV, Markdown, JSON, and more

### Basic Usage

```python
from karenina.benchmark import Benchmark

# Create a new benchmark
benchmark = Benchmark.create(
    name="Python Programming Assessment",
    description="Test understanding of Python concepts",
    version="1.0.0",
    creator="Assessment Team"
)

# Add questions
benchmark.add_question(
    question="What is a Python decorator?",
    raw_answer="A decorator modifies or extends function behavior"
)

# Save for GUI compatibility
benchmark.save("python_assessment.jsonld")
```

## Creating Benchmarks

### Create from Scratch

```python
from karenina.benchmark import Benchmark

# Basic creation
benchmark = Benchmark.create(
    name="Machine Learning Fundamentals",
    description="Assessment of ML concepts and applications"
)

# With full metadata
benchmark = Benchmark.create(
    name="Advanced Python Programming",
    description="Deep dive into Python advanced features",
    version="2.1.0",
    creator="Python Education Team"
)

# Access and modify properties
print(f"Benchmark: {benchmark.name}")
print(f"Version: {benchmark.version}")
print(f"Created: {benchmark.created_at}")

# Update metadata
benchmark.description = "Updated description"
benchmark.version = "2.2.0"
```

### Create from Extracted Questions

```python
from karenina.questions.extractor import extract_questions_from_file
from karenina.benchmark import Benchmark

# Extract questions from file
questions = extract_questions_from_file(
    "data/questions.xlsx",
    "Question",
    "Answer"
)

# Create benchmark and add questions
benchmark = Benchmark.create("Extracted Questions Benchmark")

for question in questions:
    benchmark.add_question(
        question=question.question,
        raw_answer=question.raw_answer,
        custom_metadata={
            "source_file": "questions.xlsx",
            "difficulty": "intermediate"
        }
    )

print(f"Added {len(benchmark)} questions to benchmark")
```

## Loading and Saving

### Save and Load JSON-LD Files

```python
# Save benchmark (GUI compatible)
benchmark.save("my_benchmark.jsonld")

# Load existing benchmark
loaded_benchmark = Benchmark.load("my_benchmark.jsonld")

# Verify loaded data
print(f"Loaded: {loaded_benchmark.name}")
print(f"Questions: {len(loaded_benchmark)}")
print(f"Version: {loaded_benchmark.version}")
```

### Working with GUI Exports

```python
# Load a benchmark exported from GUI
gui_benchmark = Benchmark.load("gui_export.jsonld")

# Inspect what was loaded
print(f"GUI Benchmark: {gui_benchmark.name}")
print(f"Questions: {gui_benchmark.question_count}")
print(f"Finished: {gui_benchmark.finished_count}")
print(f"Progress: {gui_benchmark.get_progress():.1f}%")

# Modify in Python
gui_benchmark.add_question(
    "What is machine learning?",
    "ML is a subset of AI that learns from data"
)

# Save back for GUI
gui_benchmark.save("enhanced_benchmark.jsonld")
```

## Managing Questions

### Adding Questions

```python
# Simple question addition
question_id = benchmark.add_question(
    question="What is polymorphism in OOP?",
    raw_answer="Polymorphism allows objects of different types to be treated uniformly"
)

# With metadata and custom properties
question_id = benchmark.add_question(
    question="Explain list comprehensions",
    raw_answer="List comprehensions provide concise syntax for creating lists",
    finished=True,  # Mark as ready for verification
    author={"name": "Dr. Smith", "email": "smith@university.edu"},
    custom_metadata={
        "difficulty": "intermediate",
        "topic": "data_structures",
        "estimated_time": 300  # seconds
    }
)

# Bulk addition
questions_data = [
    ("What is inheritance?", "Inheritance allows classes to inherit properties"),
    ("What is encapsulation?", "Encapsulation bundles data and methods together")
]

for question, answer in questions_data:
    benchmark.add_question(question, answer)
```

### Question Operations

```python
# Get question information
question_ids = benchmark.get_question_ids()
print(f"Total questions: {len(question_ids)}")

# Get specific question
first_q_id = question_ids[0]
question_data = benchmark.get_question(first_q_id)
print(f"Question: {question_data['question']}")
print(f"Answer: {question_data['raw_answer']}")

# Update question
benchmark.update_question_metadata(
    first_q_id,
    author={"name": "Updated Author"},
    custom_metadata={"reviewed": True, "review_date": "2024-01-15"}
)

# Remove question
# benchmark.remove_question(first_q_id)
```

### Question Filtering and Search

```python
# Filter questions by status
finished_questions = benchmark.filter_questions(finished=True)
unfinished_questions = benchmark.filter_questions(finished=False)
questions_with_templates = benchmark.filter_questions(has_template=True)

print(f"Finished: {len(finished_questions)}")
print(f"With templates: {len(questions_with_templates)}")

# Search questions by content
python_questions = benchmark.search_questions("Python")
oop_questions = benchmark.search_questions("object oriented")

print(f"Python-related: {len(python_questions)}")
print(f"OOP-related: {len(oop_questions)}")

# Complex filtering
intermediate_python = benchmark.filter_questions(
    custom_property="difficulty",
    custom_value="intermediate"
)
```

## Template Management

### Adding Templates

```python
# Add custom template
template_code = '''
class Answer(BaseAnswer):
    """Answer template for inheritance question."""

    definition: str = Field(description="Definition of inheritance")
    benefits: List[str] = Field(description="Benefits of inheritance")
    example: str = Field(description="Code example")

    def verify(self) -> bool:
        has_definition = len(self.definition) > 20
        has_benefits = len(self.benefits) >= 2
        has_example = "class" in self.example.lower()
        return has_definition and has_benefits and has_example
'''

benchmark.add_answer_template(question_id, template_code)
```

### Bulk Template Generation

```python
# Generate templates for all questions without them
unfinished_ids = benchmark.get_unfinished_question_ids()
print(f"Generating templates for {len(unfinished_ids)} questions...")

# This would use LLM to generate templates
# benchmark.generate_all_templates(
#     model="gpt-4",
#     model_provider="openai"
# )

# Check results
templates_ready = benchmark.get_finished_templates()
print(f"Templates ready: {len(templates_ready)}")
```

### Template Operations

```python
# Check if question has template
has_template = benchmark.has_template(question_id)
print(f"Has template: {has_template}")

# Get template
if has_template:
    template = benchmark.get_template(question_id)
    print(f"Template length: {len(template)} characters")

# Copy template between questions
benchmark.copy_template(source_question_id, target_question_id)

# Remove template
# benchmark.remove_template(question_id)
```

## Metadata Management

### Benchmark-Level Metadata

```python
# Property accessors
benchmark.name = "Updated Benchmark Name"
benchmark.description = "New description"
benchmark.version = "1.1.0"
benchmark.creator = "New Creator"

# Custom properties
benchmark.set_custom_property("domain", "computer_science")
benchmark.set_custom_property("difficulty_level", "advanced")
benchmark.set_custom_property("estimated_duration_minutes", 45)

# Multiple properties at once
benchmark.set_multiple_custom_properties({
    "tags": ["python", "programming", "assessment"],
    "license": "CC-BY-4.0",
    "language": "english",
    "target_audience": "university_students"
})

# Access custom properties
domain = benchmark.get_custom_property("domain")
all_custom = benchmark.get_all_custom_properties()
print(f"Domain: {domain}")
print(f"Custom properties: {list(all_custom.keys())}")
```

### Question-Level Metadata

```python
# Get comprehensive question metadata
metadata = benchmark.get_question_metadata(question_id)
print(f"Question ID: {metadata['id']}")
print(f"Created: {metadata['date_created']}")
print(f"Modified: {metadata['date_modified']}")
print(f"Finished: {metadata['finished']}")
print(f"Has template: {metadata['has_template']}")
print(f"Author: {metadata['author']}")

# Update question metadata
benchmark.update_question_metadata(
    question_id,
    author={"name": "Prof. Johnson", "institution": "Tech University"},
    sources=[
        {"title": "Python Documentation", "url": "https://docs.python.org"},
        {"title": "Programming Textbook", "isbn": "978-0123456789"}
    ],
    custom_metadata={
        "difficulty": "advanced",
        "bloom_taxonomy": "analysis",
        "estimated_time": 600,
        "prerequisites": ["basic_python", "oop_concepts"]
    }
)

# Access specific metadata
author = benchmark.get_question_author(question_id)
sources = benchmark.get_question_sources(question_id)
difficulty = benchmark.get_question_custom_property(question_id, "difficulty")
timestamps = benchmark.get_question_timestamps(question_id)

print(f"Author: {author['name']} ({author['institution']})")
print(f"Difficulty: {difficulty}")
print(f"Created: {timestamps['created']}")
```

## Rubric Management

### Global Rubrics

```python
from karenina.schemas.rubric_class import RubricTrait

# Add global rubric traits (apply to all questions)
benchmark.add_global_rubric_trait(
    RubricTrait(
        name="clarity",
        description="Is the explanation clear and understandable?",
        kind="boolean"
    )
)

benchmark.add_global_rubric_trait(
    RubricTrait(
        name="completeness",
        description="How complete is the answer?",
        kind="score",
        min_score=1,
        max_score=5
    )
)

# Get global rubric
global_rubric = benchmark.get_global_rubric()
if global_rubric:
    print(f"Global traits: {len(global_rubric.traits)}")
    for trait in global_rubric.traits:
        print(f"  - {trait.name}: {trait.description}")
```

### Question-Specific Rubrics

```python
# Add question-specific rubric trait
benchmark.add_question_rubric_trait(
    question_id,
    RubricTrait(
        name="code_quality",
        description="Is the provided code example well-written?",
        kind="boolean"
    )
)

# Get question's rubric traits
question_rubric = benchmark.get_question_rubric(question_id)
if question_rubric:
    print(f"Question-specific traits: {len(question_rubric.traits)}")

# Remove rubric
# benchmark.remove_question_rubric(question_id)
```

## Validation and Health Checks

### Benchmark Validation

```python
# Validate benchmark structure
is_valid, message = benchmark.validate()
print(f"Valid: {is_valid}")
if not is_valid:
    print(f"Error: {message}")

# Comprehensive health check
health = benchmark.get_health_report()
print(f"Health Score: {health['health_score']}/100")
print(f"Status: {health['health_status']}")

print("\nRecommendations:")
for i, rec in enumerate(health['recommendations'], 1):
    print(f"  {i}. {rec}")

print(f"\nIssues:")
for issue in health['issues']:
    print(f"  - {issue}")
```

### Readiness Assessment

```python
# Check if benchmark is ready for verification
print(f"Is complete: {benchmark.is_complete}")
print(f"Progress: {benchmark.get_progress():.1f}%")

# Get questions that need attention
unfinished = benchmark.get_unfinished_question_ids()
without_templates = benchmark.get_questions_without_templates()

print(f"Unfinished questions: {len(unfinished)}")
print(f"Missing templates: {len(without_templates)}")

# Get ready templates for verification
ready_templates = benchmark.get_finished_templates()
print(f"Ready for verification: {len(ready_templates)}")
```

## Export Capabilities

### Summary and Statistics

```python
# Get benchmark summary
summary = benchmark.get_summary()
print("Benchmark Summary:")
print(f"  Total questions: {summary['total_questions']}")
print(f"  Finished questions: {summary['finished_questions']}")
print(f"  Questions with templates: {summary['questions_with_templates']}")
print(f"  Global rubric traits: {summary['global_rubric_traits']}")
print(f"  Average template length: {summary['avg_template_length']}")
```

### Export to Different Formats

```python
# Export as CSV
csv_data = benchmark.to_csv()
with open("benchmark_export.csv", "w") as f:
    f.write(csv_data)

# Export as Markdown report
markdown_report = benchmark.to_markdown()
with open("benchmark_report.md", "w") as f:
    f.write(markdown_report)

# Export as dictionary (for JSON)
import json
benchmark_dict = benchmark.to_dict()
with open("benchmark_data.json", "w") as f:
    json.dump(benchmark_dict, f, indent=2)

# Export questions as Python file
benchmark.export_questions_python("exported_questions.py")
```

## Advanced Features

### Magic Methods and Iteration

```python
# Length and boolean evaluation
print(f"Benchmark has {len(benchmark)} questions")
print(f"Is empty: {not bool(benchmark)}")

# String representation
print(f"Benchmark: {benchmark}")

# Iteration over questions
for question_data in benchmark:
    print(f"Q: {question_data['question'][:50]}...")
    print(f"   ID: {question_data['id']}")
    print(f"   Finished: {question_data['finished']}")
```

### Bulk Operations

```python
# Mark multiple questions as finished
question_ids = benchmark.get_question_ids()[:5]
for q_id in question_ids:
    benchmark.mark_finished(q_id)

# Bulk template addition
template_code = "class Answer(BaseAnswer): pass"
updated_ids = benchmark.add_template_to_unfinished_questions(template_code)
print(f"Added templates to {len(updated_ids)} questions")

# Clear operations
# benchmark.clear_questions()  # Remove all questions
# benchmark.clear_global_rubric()  # Remove global rubric
```

### Advanced Filtering

```python
# Complex filtering with multiple criteria
advanced_questions = benchmark.filter_questions(
    finished=True,
    has_template=True,
    has_rubric=True,
    author="Dr. Smith"
)

# Custom filtering function
def custom_filter(question_data):
    """Custom filter for advanced questions."""
    return (
        len(question_data.get('question', '')) > 50 and
        question_data.get('custom_metadata', {}).get('difficulty') == 'advanced'
    )

filtered = [q for q in benchmark if custom_filter(q)]
print(f"Advanced questions: {len(filtered)}")
```

## GUI Interoperability

### Loading GUI Exports

```python
# Load benchmark created/exported from GUI
gui_benchmark = Benchmark.load("gui_exported_benchmark.jsonld")

# Inspect GUI-created content
print(f"GUI Benchmark: {gui_benchmark.name}")
print(f"Questions: {len(gui_benchmark)}")

# Show question details
for i, question_data in enumerate(gui_benchmark):
    if i >= 3:  # Show first 3
        break
    print(f"{i+1}. {question_data['question']}")
    print(f"   Finished: {question_data['finished']}")
    print(f"   Has template: {question_data['has_template']}")
```

### Preparing for GUI Import

```python
# Create benchmark for GUI consumption
python_benchmark = Benchmark.create(
    name="Python Benchmark Created in Code",
    description="Programmatically created for GUI import"
)

# Add structured content
python_benchmark.add_question(
    "What is a closure in Python?",
    "A closure is a function that captures variables from its enclosing scope",
    finished=True,
    author={"name": "Python Expert"},
    custom_metadata={"difficulty": "advanced", "topic": "functions"}
)

# Add global rubric for GUI evaluation
python_benchmark.add_global_rubric_trait(
    RubricTrait(name="accuracy", description="Is the answer factually correct?", kind="boolean")
)

# Save in GUI-compatible format
python_benchmark.save("for_gui_import.jsonld")
print("Saved benchmark ready for GUI import")
```

### Round-Trip Workflow

```python
# Complete round-trip example
def round_trip_workflow():
    """Demonstrate Python → GUI → Python workflow."""

    # 1. Create in Python
    original = Benchmark.create("Round Trip Test")
    original.add_question("Test question", "Test answer")
    original.save("step1_python_created.jsonld")

    # 2. Simulate GUI modifications (in practice, user works in GUI)
    # GUI loads, user adds questions/templates, exports again

    # 3. Load back in Python
    modified = Benchmark.load("step1_python_created.jsonld")

    # 4. Make Python modifications
    modified.add_question("Added in Python", "Python answer")
    modified.set_custom_property("modified_in_python", True)

    # 5. Save for next GUI session
    modified.save("step2_python_modified.jsonld")

    print("Round-trip complete - data preserved throughout")

# round_trip_workflow()
```

## Best Practices

### Performance Tips

```python
# Use bulk operations when possible
ids = benchmark.get_question_ids()
# Instead of: for qid in ids: benchmark.mark_finished(qid)
# Use bulk operation or batch processing

# Cache question data when doing multiple operations
question_data = benchmark.get_question(question_id)
# Use question_data dict instead of multiple method calls

# Use filtering for subset operations
python_questions = benchmark.filter_questions(custom_property="topic", custom_value="python")
# Operate on filtered subset instead of all questions
```

### Metadata Best Practices

```python
# Use consistent custom property naming
benchmark.set_custom_property("difficulty_level", "intermediate")  # Good
# benchmark.set_custom_property("diff", "inter")  # Avoid abbreviations

# Include structured metadata
benchmark.update_question_metadata(
    question_id,
    author={
        "name": "Dr. Jane Smith",
        "email": "jane.smith@university.edu",
        "institution": "Tech University",
        "department": "Computer Science"
    },
    sources=[
        {
            "type": "textbook",
            "title": "Python Programming",
            "author": "John Doe",
            "chapter": "Object-Oriented Programming",
            "pages": "45-67"
        }
    ]
)
```

### Error Handling

```python
try:
    benchmark = Benchmark.load("missing_file.jsonld")
except FileNotFoundError:
    print("Benchmark file not found")
    benchmark = Benchmark.create("New Benchmark")

try:
    question_id = benchmark.add_question("", "")  # Empty question
except ValueError as e:
    print(f"Invalid question: {e}")

# Validate before operations
is_valid, message = benchmark.validate()
if not is_valid:
    print(f"Benchmark validation failed: {message}")
    # Handle validation errors
```

This guide covers the comprehensive capabilities of the Benchmark class. The next step is typically to run verification using the templates and questions managed through this interface.
