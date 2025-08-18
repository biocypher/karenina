# Quick Start

Get started with Karenina in just a few minutes! This guide will walk you through creating your first benchmark, adding questions, configuring models, and running verification.

## Create a Benchmark

First, let's create a new benchmark:

```python
from karenina import Benchmark

# Create a new benchmark
benchmark = Benchmark.create(
        name="Test benchmark",
        description="Simple quick intro",
        version="1.0.0",
        creator="Karenina Example",
)
```

## Add a Couple of Questions

Now let's add some questions to our benchmark:

```python
# Add questions manually
question = "What is the capital of France?"
raw_answer = "Paris"

# Define the answer template manually
template_code = '''class Answer(BaseAnswer):
    answer: str = Field(description="the name of the city in the response")

    def model_post_init(self, __context):
        self.correct = {"answer": "Paris"}

    def verify(self) -> bool:
        return self.answer == self.correct["answer"]'''

# Add the question to the benchmark
qid = benchmark.add_question(
    question=question,
    raw_answer=raw_answer,
    answer_template=template_code,
    finished=True,  # Mark as ready for verification
    author={"name": "Example Author", "email": "author@example.com"},
)
```

## Add a Configuration

Configure which model will be used for verification:

```python
from karenina.schemas import ModelConfig

# Set up model configuration
answering_models=[
        ModelConfig(
            id="gemini-2.5-flash",
            model_provider="google_genai",
            model_name="gemini-2.5-flash",
            temperature=0.1,
            interface="langchain",
            system_prompt="You are a helpful assistant."
        )
    ]

parsing_models=[
        ModelConfig(
            id="gemini-2.5-flash",
            model_provider="google_genai",
            model_name="gemini-2.5-flash",
            temperature=0.0,
            interface="langchain",
            system_prompt="You are an LLM judge and, given a template, will judge the answer to the question"
        )
    ]

config = VerificationConfig(
    answering_models=answering_models,
    parsing_models=parsing_models
)
```

## Run the Verification

Finally, let's run the verification process:

```python
# Run verification
results = benchmark.run_verification([qid],config)
```

## Next Steps

Congratulations! You've created your first Karenina benchmark. To learn more about:

- **Advanced benchmark creation**: See [Defining a Benchmark](using-karenina/defining-benchmark.md)
- **Question management**: Check out [Adding Questions](using-karenina/adding-questions.md)
- **Template customization**: Read about [Templates](using-karenina/templates.md)
- **Evaluation criteria**: Learn about [Rubrics](using-karenina/rubrics.md)
