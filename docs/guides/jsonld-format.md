# JSON-LD Format Guide

This guide explains the JSON-LD (JSON for Linked Data) format used by Karenina for benchmark storage and interchange between the Python library and GUI components.

## Table of Contents

- [Overview](#overview)
- [Why JSON-LD?](#why-json-ld)
- [Schema.org Vocabulary](#schemaorg-vocabulary)
- [Format Structure](#format-structure)
- [Context Definition](#context-definition)
- [Benchmark Level](#benchmark-level)
- [Question Level](#question-level)
- [Metadata and Properties](#metadata-and-properties)
- [Rubric Representation](#rubric-representation)
- [Version Compatibility](#version-compatibility)
- [Complete Examples](#complete-examples)
- [Validation](#validation)

## Overview

Karenina uses JSON-LD format for storing benchmarks, providing:

- **Standardized Format**: Based on W3C JSON-LD specification
- **Semantic Meaning**: Uses schema.org vocabulary for linked data
- **Interoperability**: Full compatibility between Python library and GUI
- **Self-Describing**: Data includes its own context and type information
- **Extensible**: Easy to add new properties while maintaining compatibility

## Why JSON-LD?

### Traditional JSON Problems
```json
{
  "name": "My Benchmark",
  "questions": [
    {
      "text": "What is Python?",
      "answer": "A programming language"
    }
  ]
}
```

**Issues:**
- No semantic meaning for fields
- Ambiguous data types
- No standard vocabulary
- Hard to extend without breaking compatibility

### JSON-LD Solution
```json
{
  "@context": {"@vocab": "http://schema.org/"},
  "@type": "Dataset",
  "@id": "urn:uuid:benchmark-123",
  "name": "My Benchmark",
  "hasPart": [
    {
      "@type": "DataFeedItem",
      "item": {
        "@type": "Question",
        "text": "What is Python?",
        "acceptedAnswer": {
          "@type": "Answer",
          "text": "A programming language"
        }
      }
    }
  ]
}
```

**Benefits:**
- Each field has semantic meaning from schema.org
- Clear type information
- Globally unique identifiers
- Extensible with custom properties
- Machine-readable relationships

## Schema.org Vocabulary

Karenina maps benchmark concepts to schema.org types:

| Karenina Concept | Schema.org Type | Description |
|------------------|-----------------|-------------|
| Benchmark | `Dataset` | Collection of questions |
| Question Container | `DataFeedItem` | Item within dataset |
| Question | `Question` | Individual question |
| Answer | `Answer` | Expected answer |
| Template | `SoftwareSourceCode` | Python validation code |
| Rubric Trait | `Rating` | Evaluation criteria |
| Custom Property | `PropertyValue` | Extensible metadata |

## Format Structure

### High-Level Structure

```json
{
  "@context": { /* Context definition */ },
  "@type": "Dataset",
  "@id": "urn:uuid:benchmark-id",

  // Benchmark metadata
  "name": "Benchmark Name",
  "description": "Benchmark Description",
  "version": "1.0.0",
  "creator": "Creator Name",
  "dateCreated": "2024-01-15T10:30:00",
  "dateModified": "2024-01-15T10:30:00",

  // Global rubric (optional)
  "rating": [ /* Global rubric traits */ ],

  // Questions
  "hasPart": [ /* Question items */ ],

  // Custom properties
  "additionalProperty": [ /* Custom metadata */ ]
}
```

## Context Definition

The `@context` defines how to interpret the JSON data:

```json
{
  "@context": {
    "@version": 1.1,
    "@vocab": "http://schema.org/",

    // Core types
    "Dataset": "Dataset",
    "DataFeedItem": "DataFeedItem",
    "Question": "Question",
    "Answer": "Answer",
    "SoftwareSourceCode": "SoftwareSourceCode",
    "Rating": "Rating",
    "PropertyValue": "PropertyValue",

    // Properties
    "version": "version",
    "name": "name",
    "description": "description",
    "creator": "creator",
    "dateCreated": "dateCreated",
    "dateModified": "dateModified",

    // Structured properties
    "hasPart": {
      "@id": "hasPart",
      "@container": "@set"
    },
    "item": {
      "@id": "item",
      "@type": "@id"
    },
    "acceptedAnswer": {
      "@id": "acceptedAnswer",
      "@type": "@id"
    },
    "rating": {
      "@id": "rating",
      "@container": "@set"
    },
    "additionalProperty": {
      "@id": "additionalProperty",
      "@container": "@set"
    },

    // Simple properties
    "text": "text",
    "programmingLanguage": "programmingLanguage",
    "codeRepository": "codeRepository",
    "bestRating": "bestRating",
    "worstRating": "worstRating",
    "ratingExplanation": "ratingExplanation",
    "additionalType": "additionalType",
    "value": "value",
    "url": "url",
    "identifier": "identifier"
  }
}
```

## Benchmark Level

### Basic Benchmark Properties

```json
{
  "@type": "Dataset",
  "@id": "urn:uuid:karenina-checkpoint-1234567890.123456",

  // Required properties
  "name": "Python Programming Assessment",
  "description": "Comprehensive test of Python programming knowledge",
  "version": "1.2.0",
  "creator": "Educational Team",

  // Timestamps (ISO 8601 format)
  "dateCreated": "2024-01-15T09:00:00.000Z",
  "dateModified": "2024-01-15T15:30:00.000Z",

  // Format version for compatibility
  "additionalProperty": [
    {
      "@type": "PropertyValue",
      "name": "benchmark_format_version",
      "value": "3.0.0-jsonld"
    }
  ]
}
```

### Global Rubric

```json
{
  "rating": [
    {
      "@type": "Rating",
      "name": "clarity",
      "description": "Is the explanation clear and understandable?",
      "bestRating": 1.0,
      "worstRating": 0.0,
      "additionalType": "GlobalRubricTrait"
    },
    {
      "@type": "Rating",
      "name": "completeness",
      "description": "How complete is the answer on a scale of 1-5?",
      "bestRating": 5.0,
      "worstRating": 1.0,
      "additionalType": "GlobalRubricTrait"
    }
  ]
}
```

## Question Level

### Question Structure

```json
{
  "@type": "DataFeedItem",
  "@id": "urn:uuid:question-what-is-python-abc123",
  "dateCreated": "2024-01-15T09:15:00.000Z",
  "dateModified": "2024-01-15T09:15:00.000Z",

  "item": {
    "@type": "Question",
    "text": "What is Python and what are its main features?",

    "acceptedAnswer": {
      "@type": "Answer",
      "text": "Python is a high-level programming language known for readability and simplicity."
    },

    "hasPart": {
      "@type": "SoftwareSourceCode",
      "name": "What is Python and what are its... Answer Template",
      "text": "class Answer(BaseAnswer):\n    \"\"\"Answer template for Python question.\"\"\"\n\n    definition: str = Field(description=\"Definition of Python\")\n    features: List[str] = Field(description=\"Main features\")\n    \n    def verify(self) -> bool:\n        return len(self.definition) > 10 and len(self.features) >= 3",
      "programmingLanguage": "Python",
      "codeRepository": "karenina-benchmarks"
    },

    "additionalProperty": [
      {
        "@type": "PropertyValue",
        "name": "finished",
        "value": true
      }
    ]
  }
}
```

### Question-Specific Rubric

```json
{
  "item": {
    "@type": "Question",
    "text": "Write a Python function to calculate factorial",

    "rating": [
      {
        "@type": "Rating",
        "name": "code_correctness",
        "description": "Is the code syntactically correct and working?",
        "bestRating": 1.0,
        "worstRating": 0.0,
        "additionalType": "QuestionSpecificRubricTrait"
      },
      {
        "@type": "Rating",
        "name": "efficiency",
        "description": "How efficient is the algorithm?",
        "bestRating": 5.0,
        "worstRating": 1.0,
        "additionalType": "QuestionSpecificRubricTrait"
      }
    ]
  }
}
```

## Metadata and Properties

### Custom Properties at Benchmark Level

```json
{
  "additionalProperty": [
    {
      "@type": "PropertyValue",
      "name": "benchmark_format_version",
      "value": "3.0.0-jsonld"
    },
    {
      "@type": "PropertyValue",
      "name": "domain",
      "value": "computer_science"
    },
    {
      "@type": "PropertyValue",
      "name": "difficulty_level",
      "value": "intermediate"
    },
    {
      "@type": "PropertyValue",
      "name": "estimated_duration_minutes",
      "value": 45
    },
    {
      "@type": "PropertyValue",
      "name": "tags",
      "value": ["python", "programming", "assessment"]
    }
  ]
}
```

### Question Metadata

```json
{
  "additionalProperty": [
    {
      "@type": "PropertyValue",
      "name": "finished",
      "value": true
    },
    {
      "@type": "PropertyValue",
      "name": "author",
      "value": "{\"name\": \"Dr. Smith\", \"email\": \"smith@university.edu\"}"
    },
    {
      "@type": "PropertyValue",
      "name": "sources",
      "value": "[{\"title\": \"Python Documentation\", \"url\": \"https://docs.python.org\"}]"
    },
    {
      "@type": "PropertyValue",
      "name": "custom_difficulty",
      "value": "advanced"
    },
    {
      "@type": "PropertyValue",
      "name": "custom_topic",
      "value": "object_oriented_programming"
    },
    {
      "@type": "PropertyValue",
      "name": "custom_estimated_time",
      "value": 300
    }
  ]
}
```

## Rubric Representation

### Boolean Rubric Trait

```json
{
  "@type": "Rating",
  "name": "accuracy",
  "description": "Is the answer factually accurate?",
  "bestRating": 1.0,
  "worstRating": 0.0,
  "additionalType": "GlobalRubricTrait"
}
```

### Score-Based Rubric Trait

```json
{
  "@type": "Rating",
  "name": "depth_of_explanation",
  "description": "How deep is the explanation on a scale of 1-10?",
  "bestRating": 10.0,
  "worstRating": 1.0,
  "additionalType": "QuestionSpecificRubricTrait"
}
```

## Version Compatibility

### Format Version Tracking

The format version is stored as a custom property:

```json
{
  "additionalProperty": [
    {
      "@type": "PropertyValue",
      "name": "benchmark_format_version",
      "value": "3.0.0-jsonld"
    }
  ]
}
```

### Version History

- **1.0.0**: Original format (not JSON-LD)
- **2.0.0**: Transition format
- **3.0.0-jsonld**: Full JSON-LD implementation with schema.org

### Backward Compatibility

When loading older formats, Karenina automatically converts them:

```python
# Automatic conversion when loading
benchmark = Benchmark.load("old_format_v2.json")  # Auto-converts
benchmark.save("new_format_v3.jsonld")  # Saves in latest format
```

## Complete Examples

### Minimal Benchmark

```json
{
  "@context": {
    "@version": 1.1,
    "@vocab": "http://schema.org/",
    "hasPart": {"@id": "hasPart", "@container": "@set"},
    "additionalProperty": {"@id": "additionalProperty", "@container": "@set"}
  },
  "@type": "Dataset",
  "@id": "urn:uuid:karenina-checkpoint-minimal",
  "name": "Minimal Benchmark",
  "description": "Simple example",
  "version": "1.0.0",
  "creator": "Example Creator",
  "dateCreated": "2024-01-15T10:00:00.000Z",
  "dateModified": "2024-01-15T10:00:00.000Z",
  "hasPart": [],
  "additionalProperty": [
    {
      "@type": "PropertyValue",
      "name": "benchmark_format_version",
      "value": "3.0.0-jsonld"
    }
  ]
}
```

### Complete Benchmark Example

```json
{
  "@context": {
    "@version": 1.1,
    "@vocab": "http://schema.org/",
    "Dataset": "Dataset",
    "DataFeedItem": "DataFeedItem",
    "Question": "Question",
    "Answer": "Answer",
    "SoftwareSourceCode": "SoftwareSourceCode",
    "Rating": "Rating",
    "PropertyValue": "PropertyValue",
    "hasPart": {"@id": "hasPart", "@container": "@set"},
    "item": {"@id": "item", "@type": "@id"},
    "acceptedAnswer": {"@id": "acceptedAnswer", "@type": "@id"},
    "rating": {"@id": "rating", "@container": "@set"},
    "additionalProperty": {"@id": "additionalProperty", "@container": "@set"}
  },
  "@type": "Dataset",
  "@id": "urn:uuid:karenina-checkpoint-complete-example",
  "name": "Python Programming Assessment",
  "description": "Comprehensive test of Python programming concepts",
  "version": "1.0.0",
  "creator": "Programming Education Team",
  "dateCreated": "2024-01-15T09:00:00.000Z",
  "dateModified": "2024-01-15T15:30:00.000Z",

  "rating": [
    {
      "@type": "Rating",
      "name": "clarity",
      "description": "Is the explanation clear and easy to understand?",
      "bestRating": 1.0,
      "worstRating": 0.0,
      "additionalType": "GlobalRubricTrait"
    }
  ],

  "hasPart": [
    {
      "@type": "DataFeedItem",
      "@id": "urn:uuid:question-python-basics-abc123",
      "dateCreated": "2024-01-15T09:15:00.000Z",
      "dateModified": "2024-01-15T14:20:00.000Z",

      "item": {
        "@type": "Question",
        "text": "What is Python and what are its main characteristics?",

        "acceptedAnswer": {
          "@type": "Answer",
          "text": "Python is a high-level, interpreted programming language known for its readability and simplicity."
        },

        "hasPart": {
          "@type": "SoftwareSourceCode",
          "name": "What is Python and what are its... Answer Template",
          "text": "class Answer(BaseAnswer):\n    \"\"\"Answer template for Python basics question.\"\"\"\n\n    definition: str = Field(description=\"Definition of Python\")\n    characteristics: List[str] = Field(description=\"Main characteristics\")\n    use_cases: List[str] = Field(description=\"Common use cases\")\n\n    def verify(self) -> bool:\n        has_definition = len(self.definition) > 20\n        has_characteristics = len(self.characteristics) >= 3\n        has_use_cases = len(self.use_cases) >= 2\n        return has_definition and has_characteristics and has_use_cases",
          "programmingLanguage": "Python",
          "codeRepository": "karenina-benchmarks"
        },

        "rating": [
          {
            "@type": "Rating",
            "name": "technical_accuracy",
            "description": "Is the technical information accurate?",
            "bestRating": 1.0,
            "worstRating": 0.0,
            "additionalType": "QuestionSpecificRubricTrait"
          }
        ],

        "additionalProperty": [
          {
            "@type": "PropertyValue",
            "name": "finished",
            "value": true
          },
          {
            "@type": "PropertyValue",
            "name": "author",
            "value": "{\"name\": \"Dr. Jane Smith\", \"email\": \"jane.smith@university.edu\"}"
          },
          {
            "@type": "PropertyValue",
            "name": "custom_difficulty",
            "value": "beginner"
          },
          {
            "@type": "PropertyValue",
            "name": "custom_topic",
            "value": "python_basics"
          }
        ]
      }
    }
  ],

  "additionalProperty": [
    {
      "@type": "PropertyValue",
      "name": "benchmark_format_version",
      "value": "3.0.0-jsonld"
    },
    {
      "@type": "PropertyValue",
      "name": "domain",
      "value": "computer_science"
    },
    {
      "@type": "PropertyValue",
      "name": "target_audience",
      "value": "programming_students"
    }
  ]
}
```

## Validation

### Schema Validation

Karenina validates JSON-LD structure:

```python
from karenina.benchmark import Benchmark

# Automatic validation on load
try:
    benchmark = Benchmark.load("benchmark.jsonld")
    print("✓ Valid JSON-LD format")
except Exception as e:
    print(f"✗ Invalid format: {e}")

# Manual validation
is_valid, message = benchmark.validate()
if is_valid:
    print("✓ Benchmark structure is valid")
else:
    print(f"✗ Validation error: {message}")
```

### Common Validation Errors

1. **Missing Required Fields**
   ```json
   {
     "@type": "Dataset",
     // Missing "name" field
     "description": "Test"
   }
   ```

2. **Invalid Context**
   ```json
   {
     "@context": "wrong-context",
     "@type": "Dataset"
   }
   ```

3. **Incorrect Type References**
   ```json
   {
     "item": {
       "@type": "WrongType",  // Should be "Question"
       "text": "Question text"
     }
   }
   ```

### Best Practices

1. **Always Include Context**: Ensure `@context` is complete and correct
2. **Use Proper Types**: Match schema.org types exactly
3. **Include Format Version**: Track compatibility with format version property
4. **Validate After Changes**: Run validation after programmatic modifications
5. **Preserve IDs**: Maintain stable `@id` values for consistency

### Tools for Validation

```python
# Built-in validation
is_valid, msg = benchmark.validate()

# JSON-LD context validation
import jsonschema
# Apply JSON-LD schema validation if needed

# Schema.org validation
# Use schema.org validation tools for semantic correctness
```

This JSON-LD format ensures that Karenina benchmarks are not just data files, but semantically rich, interoperable resources that can be understood and processed by any JSON-LD compatible system while maintaining full compatibility between Python and GUI components.
