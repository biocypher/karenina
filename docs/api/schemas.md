# Schemas Module

The `karenina.schemas` module defines core data models using Pydantic for type validation and serialization.

## Question Schema

### Question

::: karenina.schemas.question_class.Question
    options:
      show_root_heading: false
      show_source: false

**Fields:**
- `id` (str): MD5 hash identifier of the question text
- `question` (str): Question content (minimum length: 1)
- `raw_answer` (str): Expected answer text (minimum length: 1)  
- `tags` (list[str | None]): Classification tags for the question

**Usage Examples:**

```python
from karenina.schemas.question_class import Question

# Create a question instance
question = Question(
    id="5f4dcc3b5aa765d61d8327deb882cf99",
    question="What is the capital of France?",
    raw_answer="Paris",
    tags=["geography", "capitals", "europe"]
)

# Validation
print(question.id)          # "5f4dcc3b5aa765d61d8327deb882cf99"
print(question.question)    # "What is the capital of France?"
print(question.raw_answer)  # "Paris"
print(question.tags)        # ["geography", "capitals", "europe"]

# JSON serialization
question_json = question.model_dump_json()
print(question_json)
# {"id": "5f4dcc3b5aa765d61d8327deb882cf99", "question": "What is the capital of France?", ...}

# JSON deserialization
question_data = {
    "id": "abc123",
    "question": "What is 2 + 2?",
    "raw_answer": "4",
    "tags": ["math", "arithmetic"]
}
question_obj = Question(**question_data)
```

**Validation Rules:**

```python
# Valid questions
valid_question = Question(
    id="hash123",
    question="Valid question text",
    raw_answer="Valid answer",
    tags=["tag1", "tag2"]
)

# Invalid questions (will raise ValidationError)
try:
    Question(id="", question="", raw_answer="", tags=[])
except ValidationError as e:
    print(e)  # String should have at least 1 character

try:
    Question(id="hash", question="Q", raw_answer="", tags=[])
except ValidationError as e:
    print(e)  # raw_answer: String should have at least 1 character
```

**Hash Generation Integration:**

```python
from karenina.questions.extractor import hash_question

question_text = "What is the capital of France?"
question_id = hash_question(question_text)

question = Question(
    id=question_id,
    question=question_text,
    raw_answer="Paris",
    tags=[]
)

# Verify hash consistency
assert question.id == hash_question(question.question)
```

## Answer Schema

### BaseAnswer

::: karenina.schemas.answer_class.BaseAnswer
    options:
      show_root_heading: false
      show_source: false

**Configuration:**
- `model_config = ConfigDict(extra="allow")`: Allows additional fields beyond those defined in subclasses

**Usage Examples:**

```python
from karenina.schemas.answer_class import BaseAnswer
from pydantic import Field

# Create custom answer template
class GeographyAnswer(BaseAnswer):
    city_name: str = Field(description="Name of the city")
    country: str = Field(description="Country where the city is located")
    confidence: float = Field(ge=0.0, le=1.0, description="Confidence score")
    population: Optional[int] = Field(default=None, description="City population")

# Usage
answer = GeographyAnswer(
    city_name="Paris",
    country="France", 
    confidence=0.95,
    population=2161000
)

# Extra fields allowed due to ConfigDict(extra="allow")
answer_with_extra = GeographyAnswer(
    city_name="Paris",
    country="France",
    confidence=0.95,
    additional_context="Capital city of France"  # Extra field
)

print(answer_with_extra.additional_context)  # "Capital city of France"
```

**Dynamic Template Generation:**

Answer templates are typically generated dynamically by the `answers.generator` module:

```python
# Generated template example
generated_code = '''
from pydantic import BaseModel, Field
from karenina.schemas.answer_class import BaseAnswer

class Answer(BaseAnswer):
    capital_city: str = Field(description="The capital city name")
    country_name: str = Field(description="The country name") 
    continent: str = Field(description="The continent")
    confidence_score: float = Field(ge=0.0, le=1.0, description="Answer confidence")
'''

# Execute generated code
local_ns = {}
exec(generated_code, globals(), local_ns)
AnswerTemplate = local_ns["Answer"]

# Use generated template
validated_answer = AnswerTemplate(
    capital_city="Paris",
    country_name="France",
    continent="Europe",
    confidence_score=0.92
)
```

## Integration Examples

### With Question Extractor

```python
from karenina.questions.extractor import extract_questions_from_file

# Extract questions (returns List[Question])
questions = extract_questions_from_file(
    "data/benchmark.xlsx",
    question_column="Question",
    answer_column="Answer"
)

# Access question attributes
for q in questions:
    print(f"ID: {q.id}")
    print(f"Question: {q.question}")
    print(f"Expected: {q.raw_answer}")
    print(f"Tags: {q.tags}")
    print("---")
```

### With Answer Generator

```python
from karenina.answers.generator import generate_answer_template

# Use Question object for template generation
question_obj = Question(
    id="hash123",
    question="What programming language is best for data science?",
    raw_answer="Python",
    tags=["programming", "data-science"]
)

# Generate template using question schema
template_code = generate_answer_template(
    question=question_obj.question,
    question_json=question_obj.model_dump_json()
)

# Template will inherit from BaseAnswer
print("Generated template inherits from BaseAnswer")
```

### With Benchmark Runner

```python
from karenina.benchmark.runner import run_benchmark

# Questions dictionary (Question objects converted to strings)
questions_dict = {q.id: q.question for q in questions}

# Run benchmark with answer templates (inheriting from BaseAnswer)
results = run_benchmark(questions_dict, responses_dict, answer_templates)

# Results are validated BaseAnswer subclass instances
for q_id, result in results.items():
    assert isinstance(result, BaseAnswer)
    print(f"Question {q_id}: {result.model_dump()}")
```

## Validation and Serialization

### Pydantic Features

Both schemas leverage Pydantic's features:

```python
# JSON serialization
question_json = question.model_dump_json(indent=2)
answer_json = answer.model_dump_json(exclude_none=True)

# Dictionary conversion
question_dict = question.model_dump()
answer_dict = answer.model_dump()

# Validation
from pydantic import ValidationError

try:
    invalid_question = Question(id="", question="", raw_answer="")
except ValidationError as e:
    print(f"Validation failed: {e}")

# Field information
print(Question.model_fields)
# {'id': FieldInfo(...), 'question': FieldInfo(...), ...}
```

### Custom Validation

```python
from pydantic import validator

class ValidatedQuestion(Question):
    @validator('tags')
    def validate_tags(cls, v):
        # Remove None values and empty strings
        return [tag for tag in v if tag and tag.strip()]
    
    @validator('question')
    def validate_question_format(cls, v):
        if not v.endswith('?'):
            raise ValueError('Questions must end with a question mark')
        return v

# Usage
validated_q = ValidatedQuestion(
    id="hash123",
    question="What is the capital of France?",
    raw_answer="Paris",
    tags=["geography", "", None, "capitals"]  # Will be cleaned
)

print(validated_q.tags)  # ["geography", "capitals"]
```

## Schema Evolution

### Backward Compatibility

When updating schemas, maintain backward compatibility:

```python
# Version 1
class QuestionV1(BaseModel):
    id: str
    question: str
    raw_answer: str

# Version 2 (backward compatible)
class Question(BaseModel):
    id: str
    question: str = Field(min_length=1)
    raw_answer: str = Field(min_length=1)
    tags: list[str | None] = Field(default_factory=list)  # New field with default
    
    # Migration from V1
    @classmethod
    def from_v1(cls, v1_data: dict):
        return cls(
            id=v1_data["id"],
            question=v1_data["question"],
            raw_answer=v1_data["raw_answer"],
            tags=[]
        )
```

### Schema Registry

For large applications, consider a schema registry pattern:

```python
from typing import Dict, Type
from pydantic import BaseModel

class SchemaRegistry:
    """Registry for managing schema versions."""
    
    _schemas: Dict[str, Dict[str, Type[BaseModel]]] = {
        "question": {
            "v1": QuestionV1,
            "v2": Question
        },
        "answer": {
            "v1": BaseAnswer
        }
    }
    
    @classmethod
    def get_schema(cls, schema_type: str, version: str = "latest"):
        if schema_type not in cls._schemas:
            raise ValueError(f"Unknown schema type: {schema_type}")
        
        versions = cls._schemas[schema_type]
        if version == "latest":
            version = max(versions.keys())
        
        if version not in versions:
            raise ValueError(f"Unknown version {version} for {schema_type}")
        
        return versions[version]

# Usage
QuestionSchema = SchemaRegistry.get_schema("question", "v2")
question = QuestionSchema(id="hash", question="Q?", raw_answer="A", tags=[])
```