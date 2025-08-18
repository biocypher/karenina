# Associating Templates to Questions

Templates define the structure for evaluating LLM responses. This guide covers manual template creation and automatic generation using the `generate_answer_template` method.

## Understanding Templates

Templates in Karenina are Pydantic classes that:

- **Structure LLM responses** into parseable formats
- **Define evaluation criteria** through validation methods
- **Standardize assessment** across different question types
- **Enable programmatic verification** of answer correctness

## Manual Template Creation

### Basic Template Structure

Create templates manually by defining Pydantic classes that inherit from `BaseAnswer`:

```python
from karenina.schemas import BaseAnswer
from pydantic import Field

class MultipleChoiceAnswer(BaseAnswer):
    selected_option: str = Field(description="The letter or option selected by the model")
    reasoning: str = Field(description="The model's explanation for their choice")

    def model_post_init(self, __context):
        # Define the correct answer
        self.correct = {"selected_option": "A"}

    def verify(self) -> bool:
        """Check if the selected option is correct"""
        return self.selected_option == self.correct["selected_option"]
```

### Assigning Templates to Questions

```python
# Create a question
question = benchmark.add_question(
    content="What is the capital of France? A) Paris B) London C) Rome D) Madrid"
)

# Assign the template manually
question.answer_template = MultipleChoiceAnswer
```

### Complex Template Example

```python
class EssayAnswer(BaseAnswer):
    main_points: List[str] = Field(description="Key points mentioned in the essay")
    thesis_statement: str = Field(description="The main argument or thesis")
    evidence_quality: int = Field(description="Quality of evidence (1-5 scale)", ge=1, le=5)
    coherence: int = Field(description="Logical flow and coherence (1-5 scale)", ge=1, le=5)

    def model_post_init(self, __context):
        self.correct = {
            "required_points": ["point1", "point2", "point3"],
            "min_evidence_quality": 3,
            "min_coherence": 3
        }

    def verify(self) -> bool:
        # Check if at least 2 of 3 required points are mentioned
        points_covered = sum(1 for point in self.correct["required_points"]
                           if any(point.lower() in mp.lower() for mp in self.main_points))

        return (points_covered >= 2 and
                self.evidence_quality >= self.correct["min_evidence_quality"] and
                self.coherence >= self.correct["min_coherence"])
```

## Automatic Template Generation

### Basic Template Generation

Use the `generate_answer_template` method to automatically create templates:

```python
from karenina.schemas import ModelConfiguration

# Configure the model for template generation
model_config = ModelConfiguration(
    provider="openai",
    model="gpt-4",
    temperature=0.1
)

# Generate template for a single question
question.generate_answer_template(
    model_config=model_config,
    system_prompt="Create a Pydantic class to evaluate responses to this question."
)
```

### Batch Template Generation

Generate templates for multiple questions at once:

```python
# Generate templates for all unfinished questions
benchmark.generate_answer_templates(
    model_config=model_config,
    system_prompt="""
    Create a Pydantic class that inherits from BaseAnswer to evaluate responses to this question.
    Include appropriate fields with descriptions and a verify() method that checks correctness.
    Consider the question type and create appropriate evaluation criteria.
    """
)
```

### Custom System Prompts

Tailor the generation process with specific instructions:

```python
# For math problems
math_prompt = """
Create a Pydantic template for evaluating mathematical solutions.
Include fields for:
- final_answer: The numerical or algebraic result
- solution_method: The approach used
- steps_shown: Whether work is shown (boolean)
- calculation_accuracy: Correctness of calculations (1-5 scale)

The verify() method should check if the final answer matches the expected result.
"""

# For science questions
science_prompt = """
Create a Pydantic template for scientific explanations.
Include fields for:
- key_concepts: List of scientific concepts mentioned
- accuracy: Factual correctness (1-5 scale)
- completeness: How comprehensive the explanation is (1-5 scale)
- scientific_reasoning: Quality of logical reasoning (1-5 scale)

The verify() method should ensure key concepts are covered and accuracy is high.
"""

# Apply different prompts to different question types
for question in benchmark.questions:
    category = question.metadata.get("category")

    if category == "mathematics":
        question.generate_answer_template(model_config, math_prompt)
    elif category == "science":
        question.generate_answer_template(model_config, science_prompt)
    else:
        question.generate_answer_template(model_config, "Create appropriate template")
```

## Template Validation and Quality Control

### Validating Generated Templates

```python
def validate_template(question):
    """Validate that a template is properly structured"""
    if question.answer_template is None:
        return False, "No template assigned"

    # Check if template inherits from BaseAnswer
    if not issubclass(question.answer_template, BaseAnswer):
        return False, "Template doesn't inherit from BaseAnswer"

    # Check if verify method is implemented
    if not hasattr(question.answer_template, 'verify'):
        return False, "Template missing verify() method"

    return True, "Template is valid"

# Validate all templates
for question in benchmark.questions:
    is_valid, message = validate_template(question)
    if not is_valid:
        print(f"Question {question.id}: {message}")
```

### Testing Templates

```python
def test_template(question, sample_response):
    """Test a template with a sample response"""
    try:
        # Create template instance
        template_instance = question.answer_template()

        # Simulate LLM parsing (normally done by judge LLM)
        # This would be replaced by actual LLM parsing in practice
        parsed_response = {"field1": "value1", "field2": "value2"}

        # Populate template
        populated_template = question.answer_template(**parsed_response)

        # Test verification
        is_correct = populated_template.verify()

        return True, is_correct
    except Exception as e:
        return False, str(e)
```

## Template Reusability

### Shared Template Library

Create reusable templates for common question types:

```python
class TrueFalseAnswer(BaseAnswer):
    answer: bool = Field(description="True or False response")
    confidence: float = Field(description="Confidence level 0.0-1.0", ge=0.0, le=1.0)

    def model_post_init(self, __context):
        # Correct answer set per question
        pass

    def verify(self) -> bool:
        return self.answer == self.correct.get("answer", False)

class NumericalAnswer(BaseAnswer):
    value: float = Field(description="Numerical answer")
    units: str = Field(description="Units of measurement", default="")

    def model_post_init(self, __context):
        self.tolerance = self.correct.get("tolerance", 0.01)

    def verify(self) -> bool:
        expected = self.correct["value"]
        return abs(self.value - expected) <= self.tolerance

# Apply templates to appropriate questions
for question in benchmark.questions:
    question_type = question.metadata.get("type")

    if question_type == "true_false":
        question.answer_template = TrueFalseAnswer
    elif question_type == "numerical":
        question.answer_template = NumericalAnswer
```

### Template Inheritance

Create specialized templates through inheritance:

```python
class BasicMathAnswer(BaseAnswer):
    answer: float = Field(description="Numerical result")

    def verify(self) -> bool:
        return abs(self.answer - self.correct["answer"]) <= 0.01

class AlgebraAnswer(BasicMathAnswer):
    """Specialized for algebra problems"""
    expression: str = Field(description="Algebraic expression if applicable")
    steps: List[str] = Field(description="Solution steps", default=[])

    def verify(self) -> bool:
        # Check numerical answer first
        if not super().verify():
            return False

        # Additional checks for algebra
        return len(self.steps) >= 2  # Require showing work

class GeometryAnswer(BasicMathAnswer):
    """Specialized for geometry problems"""
    diagram_described: bool = Field(description="Whether spatial relationships are described")
    theorem_used: str = Field(description="Mathematical theorem applied", default="")

    def verify(self) -> bool:
        # Check numerical answer first
        if not super().verify():
            return False

        # Geometry-specific validation
        return self.diagram_described and len(self.theorem_used) > 0
```

## Managing Template Evolution

### Template Versioning

```python
class AnswerTemplateV1(BaseAnswer):
    answer: str = Field(description="Basic answer field")

    def verify(self) -> bool:
        return self.answer == self.correct["answer"]

class AnswerTemplateV2(BaseAnswer):
    answer: str = Field(description="Enhanced answer field")
    confidence: float = Field(description="Confidence score", default=0.0)
    reasoning: str = Field(description="Explanation", default="")

    def verify(self) -> bool:
        # Enhanced verification logic
        basic_check = self.answer == self.correct["answer"]
        confidence_check = self.confidence >= 0.7

        return basic_check and confidence_check
```

### Template Migration

```python
def migrate_templates(benchmark, old_template_class, new_template_class):
    """Migrate questions from old template to new template"""
    migrated_count = 0

    for question in benchmark.questions:
        if question.answer_template == old_template_class:
            question.answer_template = new_template_class
            migrated_count += 1

    print(f"Migrated {migrated_count} questions to new template")

# Example migration
migrate_templates(benchmark, AnswerTemplateV1, AnswerTemplateV2)
```

## Next Steps

Once you have templates set up for your questions:

- [Configure rubrics](rubrics.md) for additional scoring criteria
- [Run verification](verification.md) to evaluate LLM responses
- [Analyze results](../api-reference.md#verification-results) to assess performance
