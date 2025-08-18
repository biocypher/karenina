# Rubrics

Rubrics provide additional evaluation criteria beyond the basic template verification. They enable nuanced scoring and assessment across multiple dimensions.

## Understanding Rubrics

Rubrics in Karenina:

- **Supplement template verification** with qualitative scoring
- **Provide multi-dimensional assessment** across different criteria
- **Enable consistent evaluation** across different evaluators
- **Support both global and question-specific** evaluation standards

## Global Rubrics

Global rubrics apply to all questions in a benchmark unless overridden by question-specific rubrics.

### Creating a Global Rubric

```python
from karenina.schemas import RubricCriterion, Rubric

# Define evaluation criteria
criteria = [
    RubricCriterion(
        name="accuracy",
        description="Factual correctness of the response",
        scale_type="points",
        max_points=5,
        levels=[
            {"score": 1, "description": "Mostly incorrect information"},
            {"score": 2, "description": "Some correct elements, major errors"},
            {"score": 3, "description": "Generally correct with minor errors"},
            {"score": 4, "description": "Accurate with very minor issues"},
            {"score": 5, "description": "Completely accurate and precise"}
        ]
    ),
    RubricCriterion(
        name="clarity",
        description="Clarity and organization of the response",
        scale_type="points",
        max_points=5,
        levels=[
            {"score": 1, "description": "Very unclear, poorly organized"},
            {"score": 2, "description": "Somewhat unclear, weak organization"},
            {"score": 3, "description": "Moderately clear, adequate organization"},
            {"score": 4, "description": "Clear and well-organized"},
            {"score": 5, "description": "Exceptionally clear and well-structured"}
        ]
    ),
    RubricCriterion(
        name="completeness",
        description="How thoroughly the question is answered",
        scale_type="points",
        max_points=3,
        levels=[
            {"score": 1, "description": "Major aspects missing"},
            {"score": 2, "description": "Some important elements missing"},
            {"score": 3, "description": "All key aspects addressed"}
        ]
    )
]

# Create the global rubric
global_rubric = Rubric(
    name="general-assessment",
    description="Standard evaluation criteria for all responses",
    criteria=criteria
)

# Apply to benchmark
benchmark.set_global_rubric(global_rubric)
```

### Alternative Scale Types

```python
# Binary pass/fail criterion
binary_criterion = RubricCriterion(
    name="factual_accuracy",
    description="Are the facts presented correct?",
    scale_type="binary",
    levels=[
        {"score": 0, "description": "Contains factual errors"},
        {"score": 1, "description": "Factually accurate"}
    ]
)

# Percentage-based scoring
percentage_criterion = RubricCriterion(
    name="coverage",
    description="Percentage of key topics covered",
    scale_type="percentage",
    max_points=100,
    levels=[
        {"score": 0, "description": "No key topics covered"},
        {"score": 25, "description": "25% of topics covered"},
        {"score": 50, "description": "50% of topics covered"},
        {"score": 75, "description": "75% of topics covered"},
        {"score": 100, "description": "All key topics covered"}
    ]
)
```

## Question-Specific Rubrics

Override global rubrics for specific questions that require specialized evaluation.

### Creating Question-Specific Rubrics

```python
# Math-specific rubric
math_rubric = Rubric(
    name="mathematics-assessment",
    description="Evaluation criteria for mathematical problems",
    criteria=[
        RubricCriterion(
            name="solution_method",
            description="Appropriateness of the solution approach",
            scale_type="points",
            max_points=4,
            levels=[
                {"score": 1, "description": "Inappropriate or no clear method"},
                {"score": 2, "description": "Partially appropriate method"},
                {"score": 3, "description": "Appropriate method with minor issues"},
                {"score": 4, "description": "Optimal solution method"}
            ]
        ),
        RubricCriterion(
            name="work_shown",
            description="Quality and completeness of shown work",
            scale_type="points",
            max_points=3,
            levels=[
                {"score": 1, "description": "Little or no work shown"},
                {"score": 2, "description": "Some steps shown, gaps present"},
                {"score": 3, "description": "All major steps clearly shown"}
            ]
        ),
        RubricCriterion(
            name="final_answer",
            description="Correctness of final numerical answer",
            scale_type="binary",
            levels=[
                {"score": 0, "description": "Incorrect final answer"},
                {"score": 1, "description": "Correct final answer"}
            ]
        )
    ]
)

# Apply to specific math questions
math_questions = benchmark.filter_questions(category="mathematics")
for question in math_questions:
    question.set_rubric(math_rubric)
```

### Domain-Specific Rubric Examples

```python
# Essay evaluation rubric
essay_rubric = Rubric(
    name="essay-assessment",
    description="Comprehensive essay evaluation",
    criteria=[
        RubricCriterion(
            name="thesis_strength",
            description="Clarity and strength of thesis statement",
            scale_type="points",
            max_points=4
        ),
        RubricCriterion(
            name="evidence_quality",
            description="Quality and relevance of supporting evidence",
            scale_type="points",
            max_points=5
        ),
        RubricCriterion(
            name="organization",
            description="Logical structure and flow",
            scale_type="points",
            max_points=4
        ),
        RubricCriterion(
            name="writing_mechanics",
            description="Grammar, spelling, and style",
            scale_type="points",
            max_points=3
        )
    ]
)

# Code evaluation rubric
code_rubric = Rubric(
    name="code-assessment",
    description="Software code evaluation criteria",
    criteria=[
        RubricCriterion(
            name="correctness",
            description="Does the code produce correct output?",
            scale_type="binary"
        ),
        RubricCriterion(
            name="efficiency",
            description="Time and space complexity appropriateness",
            scale_type="points",
            max_points=4
        ),
        RubricCriterion(
            name="readability",
            description="Code clarity and documentation",
            scale_type="points",
            max_points=3
        ),
        RubricCriterion(
            name="best_practices",
            description="Adherence to coding standards",
            scale_type="points",
            max_points=3
        )
    ]
)
```

## Rubric Application Strategies

### Hierarchical Rubrics

Apply rubrics at different levels for comprehensive evaluation:

```python
# Set benchmark-level default
benchmark.set_global_rubric(global_rubric)

# Override for specific categories
science_questions = benchmark.filter_questions(category="science")
for question in science_questions:
    question.set_rubric(science_rubric)

# Further override for specialized topics
physics_questions = benchmark.filter_questions(category="science", topic="physics")
for question in physics_questions:
    question.set_rubric(physics_rubric)
```

### Conditional Rubric Assignment

```python
def assign_rubrics_by_type(benchmark):
    """Assign appropriate rubrics based on question characteristics"""

    for question in benchmark.questions:
        # Get question metadata
        category = question.metadata.get("category")
        question_type = question.metadata.get("type")
        difficulty = question.metadata.get("difficulty")

        # Assign rubric based on characteristics
        if category == "mathematics":
            if question_type == "word_problem":
                question.set_rubric(word_problem_rubric)
            elif question_type == "proof":
                question.set_rubric(proof_rubric)
            else:
                question.set_rubric(math_rubric)

        elif category == "science":
            if difficulty == "advanced":
                question.set_rubric(advanced_science_rubric)
            else:
                question.set_rubric(basic_science_rubric)

        elif category == "literature":
            if "analysis" in question.content.lower():
                question.set_rubric(literary_analysis_rubric)
            else:
                question.set_rubric(literature_rubric)

        # Fall back to global rubric for unspecified cases
        elif not question.has_rubric():
            question.use_global_rubric()

# Apply conditional rubrics
assign_rubrics_by_type(benchmark)
```

## Working with Rubric Results

### Accessing Rubric Scores

```python
# After running verification with rubrics enabled
for result in verification_results:
    question = result.question
    rubric_scores = result.rubric_evaluation

    if rubric_scores:
        print(f"Question: {question.content[:50]}...")
        for criterion_name, score in rubric_scores.items():
            criterion = question.get_rubric_criterion(criterion_name)
            max_score = criterion.max_points
            print(f"  {criterion_name}: {score}/{max_score}")

        # Calculate total score
        total = sum(rubric_scores.values())
        max_total = sum(c.max_points for c in question.rubric.criteria)
        print(f"  Total: {total}/{max_total} ({total/max_total*100:.1f}%)")
```

### Aggregating Rubric Results

```python
def analyze_rubric_performance(verification_results):
    """Analyze performance across rubric criteria"""

    criterion_scores = {}

    for result in verification_results:
        if result.rubric_evaluation:
            for criterion_name, score in result.rubric_evaluation.items():
                if criterion_name not in criterion_scores:
                    criterion_scores[criterion_name] = []
                criterion_scores[criterion_name].append(score)

    # Calculate statistics for each criterion
    for criterion_name, scores in criterion_scores.items():
        avg_score = sum(scores) / len(scores)
        min_score = min(scores)
        max_score = max(scores)

        print(f"{criterion_name}:")
        print(f"  Average: {avg_score:.2f}")
        print(f"  Range: {min_score} - {max_score}")
        print(f"  Count: {len(scores)}")

# Analyze results
analyze_rubric_performance(verification_results)
```

## Rubric Management

### Importing and Exporting Rubrics

```python
# Export rubric to JSON
rubric_json = global_rubric.to_json()
with open("global_rubric.json", "w") as f:
    f.write(rubric_json)

# Import rubric from JSON
with open("global_rubric.json", "r") as f:
    rubric_data = json.load(f)
imported_rubric = Rubric.from_json(rubric_data)
```

### Rubric Validation

```python
def validate_rubric(rubric):
    """Validate rubric structure and consistency"""
    issues = []

    # Check that all criteria have valid score ranges
    for criterion in rubric.criteria:
        if criterion.scale_type == "points":
            max_score = max(level["score"] for level in criterion.levels)
            if max_score != criterion.max_points:
                issues.append(f"Criterion '{criterion.name}': max level score doesn't match max_points")

        # Check level descriptions exist
        for level in criterion.levels:
            if not level.get("description"):
                issues.append(f"Criterion '{criterion.name}': level {level['score']} missing description")

    return len(issues) == 0, issues

# Validate rubrics
is_valid, issues = validate_rubric(global_rubric)
if not is_valid:
    print("Rubric validation issues:")
    for issue in issues:
        print(f"  - {issue}")
```

## Next Steps

Once you have rubrics configured:

- [Run verification](verification.md) to apply both template and rubric evaluation
- [Analyze results](../api-reference.md#verification-results) to understand performance across different criteria
- [Save and load benchmarks](saving-loading.md) to preserve your rubric configurations
