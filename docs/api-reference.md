# Karenina API Reference

## Table of Contents

- [LLM Module](#llm-module)
- [Questions Module](#questions-module)
- [Answers Module](#answers-module)
- [Benchmark Module](#benchmark-module)
- [Schemas Module](#schemas-module)
- [Prompts Module](#prompts-module)
- [Utils Module](#utils-module)

## LLM Module

### `karenina.llm.interface`

#### Functions

##### `init_chat_model_unified(model, provider, interface='langchain', temperature=0.7, **kwargs)`

Initialize a chat model with unified interface across providers.

**Parameters:**
- `model` (str): Model name (e.g., "gpt-4", "gemini-2.0-flash")
- `provider` (str): Provider name ("openai", "google_genai", "anthropic")
- `interface` (str): Interface type ("langchain", "openrouter", "manual")
- `temperature` (float): Sampling temperature (0.0-2.0)
- `**kwargs`: Additional provider-specific parameters

**Returns:**
- `BaseChatModel`: Initialized chat model instance

**Example:**
```python
from karenina.llm.interface import init_chat_model_unified

llm = init_chat_model_unified(
    model="gpt-4",
    provider="openai",
    interface="langchain",
    temperature=0.1
)
```

##### `chat_with_llm(llm, message, system_message=None)`

Send a message to an LLM instance.

**Parameters:**
- `llm` (BaseChatModel): Initialized LLM instance
- `message` (str): User message
- `system_message` (str, optional): System prompt

**Returns:**
- `str`: LLM response

**Example:**
```python
response = chat_with_llm(
    llm=llm,
    message="What is the capital of France?",
    system_message="You are a helpful geography assistant."
)
```

##### `create_chat_session(model, provider, temperature=0.7)`

Create a new chat session for stateful conversations.

**Parameters:**
- `model` (str): Model name
- `provider` (str): Provider name
- `temperature` (float): Sampling temperature

**Returns:**
- `str`: Session ID

**Example:**
```python
session_id = create_chat_session(
    model="gpt-4",
    provider="openai",
    temperature=0.7
)
```

##### `chat_with_session(session_id, message, system_message=None)`

Send a message within an existing session.

**Parameters:**
- `session_id` (str): Session identifier
- `message` (str): User message
- `system_message` (str, optional): System prompt

**Returns:**
- `ChatResponse`: Response with session context

#### Classes

##### `ChatSession`

Manages a conversation session with an LLM.

**Attributes:**
- `session_id` (str): Unique session identifier
- `model` (str): Model name
- `provider` (str): Provider name
- `temperature` (float): Sampling temperature
- `messages` (List[BaseMessage]): Conversation history
- `created_at` (datetime): Session creation time
- `last_used` (datetime): Last activity time

**Methods:**
- `initialize_llm()`: Initialize the LLM instance
- `add_message(message, is_human=True)`: Add message to history
- `add_system_message(message)`: Set or update system prompt

##### `ChatRequest`

Request model for chat API.

**Attributes:**
- `model` (str): Model name
- `provider` (str): Provider name
- `message` (str): User message
- `session_id` (str, optional): Session ID for stateful chat
- `system_message` (str, optional): System prompt
- `temperature` (float, optional): Sampling temperature

##### `ChatResponse`

Response model for chat API.

**Attributes:**
- `session_id` (str): Session identifier
- `message` (str): LLM response
- `model` (str): Model used
- `provider` (str): Provider used
- `timestamp` (str): Response timestamp

#### Exceptions

##### `LLMError`

Base exception for LLM-related errors.

##### `LLMNotAvailableError`

Raised when LangChain dependencies are not available.

##### `SessionError`

Raised for session management errors.

### `karenina.llm.manual_llm`

#### Functions

##### `create_manual_llm(traces)`

Create a mock LLM instance for manual verification.

**Parameters:**
- `traces` (Dict): Pre-recorded responses keyed by question ID

**Returns:**
- `ManualLLM`: Mock LLM instance

**Example:**
```python
from karenina.llm.manual_llm import create_manual_llm

traces = {"q1": "Manual response for question 1"}
manual_llm = create_manual_llm(traces)
```

### `karenina.llm.manual_traces`

#### Functions

##### `save_manual_traces(traces, filepath)`

Save manual traces to a JSON file.

**Parameters:**
- `traces` (Dict): Trace data
- `filepath` (str): Output file path

##### `load_manual_traces(filepath)`

Load manual traces from a JSON file.

**Parameters:**
- `filepath` (str): Input file path

**Returns:**
- `Dict`: Loaded trace data

## Questions Module

### `karenina.questions.extractor`

#### Functions

##### `extract_and_generate_questions(file_path, output_path, question_column='Question', answer_column='Answer', sheet_name=None, tags_column=None)`

Complete pipeline for extracting questions from files.

**Parameters:**
- `file_path` (str): Input file path (Excel/CSV/TSV)
- `output_path` (str): Output Python file path
- `question_column` (str): Column name for questions
- `answer_column` (str): Column name for answers
- `sheet_name` (str, optional): Excel sheet name
- `tags_column` (str, optional): Column name for tags

**Returns:**
- `List[Question]`: Extracted questions

**Example:**
```python
from karenina.questions.extractor import extract_and_generate_questions

questions = extract_and_generate_questions(
    file_path="data/questions.xlsx",
    output_path="questions.py",
    question_column="Question",
    answer_column="Expected Answer",
    sheet_name="Sheet1"
)
```

##### `read_file_to_dataframe(file_path, sheet_name=None)`

Read various file formats into pandas DataFrame.

**Parameters:**
- `file_path` (str): Input file path
- `sheet_name` (str, optional): Excel sheet name

**Returns:**
- `pd.DataFrame`: Loaded data

**Raises:**
- `ValueError`: Unsupported file format
- `FileNotFoundError`: File doesn't exist

##### `get_file_preview(file_path, sheet_name=None, max_rows=100)`

Get a preview of file contents.

**Parameters:**
- `file_path` (str): Input file path
- `sheet_name` (str, optional): Excel sheet name
- `max_rows` (int): Maximum preview rows

**Returns:**
- `Dict`: Preview information with columns and sample data

##### `hash_question(question_text)`

Generate MD5 hash ID for a question.

**Parameters:**
- `question_text` (str): Question text

**Returns:**
- `str`: 32-character hexadecimal hash

##### `generate_questions_file(questions, output_path)`

Generate Python file containing Question objects.

**Parameters:**
- `questions` (List[Question]): Question instances
- `output_path` (str): Output file path

### `karenina.questions.reader`

#### Functions

##### `read_questions_from_file(file_path)`

Read questions from generated Python file.

**Parameters:**
- `file_path` (str): Path to questions.py file

**Returns:**
- `List[Question]`: Loaded questions

**Example:**
```python
from karenina.questions.reader import read_questions_from_file

questions = read_questions_from_file("questions.py")
```

## Answers Module

### `karenina.answers.generator`

#### Functions

##### `generate_answer_template(question, raw_answer, model='gemini-2.0-flash', model_provider='google_genai', temperature=0, custom_system_prompt=None, interface='langchain')`

Generate a single answer template.

**Parameters:**
- `question` (str): Question text
- `raw_answer` (str): Expected answer
- `model` (str): LLM model name
- `model_provider` (str): LLM provider
- `temperature` (float): Sampling temperature
- `custom_system_prompt` (str, optional): Custom system prompt
- `interface` (str): LLM interface type

**Returns:**
- `str`: Python code for answer template class

**Example:**
```python
from karenina.answers.generator import generate_answer_template

template = generate_answer_template(
    question="What is the capital of France?",
    raw_answer="Paris",
    model="gpt-4",
    model_provider="openai"
)
```

##### `generate_answer_templates_from_questions_file(questions_py_path, model='gemini-2.0-flash', model_provider='google_genai', interface='langchain', return_blocks=False)`

Generate templates for all questions in a file.

**Parameters:**
- `questions_py_path` (str): Path to questions.py file
- `model` (str): LLM model name
- `model_provider` (str): LLM provider
- `interface` (str): LLM interface type
- `return_blocks` (bool): Return raw code blocks

**Returns:**
- `Dict[str, Any]` or `Tuple[Dict[str, Any], Dict[str, str]]`: Generated templates

**Example:**
```python
templates = generate_answer_templates_from_questions_file(
    questions_py_path="questions.py",
    model="gpt-4",
    model_provider="openai",
    return_blocks=True
)
```

##### `inject_question_id_into_answer_class(answer_class, question_id)`

Inject question ID into an answer class programmatically.

**Parameters:**
- `answer_class` (type): Answer class
- `question_id` (str): Question identifier

**Returns:**
- `type`: Modified answer class with ID injection

### `karenina.answers.reader`

#### Functions

##### `read_answer_templates_from_file(file_path)`

Read answer templates from Python file.

**Parameters:**
- `file_path` (str): Path to templates file

**Returns:**
- `Dict[str, type]`: Loaded template classes

## Benchmark Module

### `karenina.benchmark.models`

#### Classes

##### `ModelConfiguration`

Configuration for a single model.

**Attributes:**
- `id` (str): Model identifier
- `model_provider` (str): Provider name
- `model_name` (str): Model name
- `temperature` (float): Sampling temperature
- `interface` (str): Interface type
- `system_prompt` (str): System prompt

**Example:**
```python
from karenina.benchmark.models import ModelConfiguration

config = ModelConfiguration(
    id="gpt4",
    model_provider="openai",
    model_name="gpt-4",
    temperature=0.1,
    interface="langchain",
    system_prompt="You are an expert assistant."
)
```

##### `VerificationConfig`

Configuration for verification run with multiple models.

**Attributes:**
- `answering_models` (List[ModelConfiguration]): Models for answering
- `parsing_models` (List[ModelConfiguration]): Models for parsing
- `replicate_count` (int): Number of replications
- `rubric_enabled` (bool): Enable rubric evaluation
- `rubric_trait_names` (List[str], optional): Filter specific traits

**Example:**
```python
from karenina.benchmark.models import VerificationConfig, ModelConfiguration

config = VerificationConfig(
    answering_models=[
        ModelConfiguration(
            id="gpt4",
            model_provider="openai",
            model_name="gpt-4",
            temperature=0.1,
            interface="langchain",
            system_prompt="Answer accurately."
        )
    ],
    parsing_models=[
        ModelConfiguration(
            id="gpt35",
            model_provider="openai",
            model_name="gpt-3.5-turbo",
            temperature=0.0,
            interface="langchain",
            system_prompt="Parse the response."
        )
    ],
    replicate_count=3,
    rubric_enabled=True
)
```

##### `VerificationResult`

Results from a single verification run.

**Attributes:**
- `question_id` (str): Question identifier
- `result_id` (str): Unique result identifier
- `answering_model_id` (str): Answering model ID
- `parsing_model_id` (str): Parsing model ID
- `raw_response` (str): Raw LLM response
- `parsed_response` (Dict): Parsed response data
- `is_correct` (bool): Correctness flag
- `granular_score` (float, optional): Granular score
- `rubric_evaluation` (Dict, optional): Rubric scores
- `error` (str, optional): Error message
- `timestamp` (str): Execution timestamp

### `karenina.benchmark.verification.orchestrator`

#### Functions

##### `run_question_verification(question_id, question_text, template_code, config, rubric=None)`

Run verification for a single question with all model combinations.

**Parameters:**
- `question_id` (str): Question identifier
- `question_text` (str): Question text
- `template_code` (str): Answer template code
- `config` (VerificationConfig): Verification configuration
- `rubric` (Rubric, optional): Evaluation rubric

**Returns:**
- `Dict[str, VerificationResult]`: Results keyed by combination ID

**Example:**
```python
from karenina.benchmark.verification.orchestrator import run_question_verification

results = run_question_verification(
    question_id="q1",
    question_text="What is the capital of France?",
    template_code=template_code,
    config=config,
    rubric=rubric
)
```

### `karenina.benchmark.verification.runner`

#### Functions

##### `run_single_model_verification(question_id, question_text, template_code, answering_model, parsing_model, run_name=None, job_id=None, answering_replicate=None, parsing_replicate=None, rubric=None)`

Run verification with a single model pair.

**Parameters:**
- `question_id` (str): Question identifier
- `question_text` (str): Question text
- `template_code` (str): Answer template code
- `answering_model` (ModelConfiguration): Model for answering
- `parsing_model` (ModelConfiguration): Model for parsing
- `run_name` (str, optional): Run identifier
- `job_id` (str, optional): Job identifier
- `answering_replicate` (int, optional): Answering replicate number
- `parsing_replicate` (int, optional): Parsing replicate number
- `rubric` (Rubric, optional): Evaluation rubric

**Returns:**
- `VerificationResult`: Verification result

### `karenina.benchmark.verification.validation`

#### Functions

##### `validate_answer_template(template_code)`

Validate answer template code.

**Parameters:**
- `template_code` (str): Python code for answer template

**Returns:**
- `Tuple[bool, str]`: (is_valid, error_message)

### `karenina.benchmark.verification.rubric_evaluator`

#### Classes

##### `RubricEvaluator`

Evaluates responses using rubric criteria.

**Methods:**
- `evaluate(question, response, rubric)`: Evaluate response against rubric
- `get_trait_score(trait, response)`: Get score for specific trait

### `karenina.benchmark.exporter`

#### Functions

##### `export_verification_results(results, output_path, format='json')`

Export verification results to file.

**Parameters:**
- `results` (Dict or List): Verification results
- `output_path` (str): Output file path
- `format` (str): Export format ("json" or "csv")

**Example:**
```python
from karenina.benchmark.exporter import export_verification_results

export_verification_results(
    results=results,
    output_path="results.json",
    format="json"
)
```

## Schemas Module

### `karenina.schemas.question_class`

#### Classes

##### `Question`

Represents a benchmark question with metadata.

**Attributes:**
- `id` (str): Hashed ID of the question
- `question` (str): Question text (min_length=1)
- `raw_answer` (str): Raw answer text (min_length=1)
- `tags` (List[str]): Question tags

**Example:**
```python
from karenina.schemas.question_class import Question

question = Question(
    id="abc123...",
    question="What is the capital of France?",
    raw_answer="Paris",
    tags=["geography", "capitals"]
)
```

### `karenina.schemas.answer_class`

#### Classes

##### `BaseAnswer`

Base class for all answer templates.

**Attributes:**
- `id` (str, optional): Question ID reference

**Methods:**
- `set_question_id(question_id)`: Set question ID programmatically

**Configuration:**
- `model_config`: Allows extra fields

**Example:**
```python
from karenina.schemas.answer_class import BaseAnswer
from pydantic import Field

class CustomAnswer(BaseAnswer):
    response: str = Field(description="The answer text")

    def verify(self) -> bool:
        return self.response.lower() == "paris"
```

### `karenina.schemas.rubric_class`

#### Classes

##### `RubricTrait`

Single evaluation trait for qualitative assessment.

**Attributes:**
- `name` (str): Human-readable trait identifier
- `description` (str, optional): Detailed description
- `kind` (TraitKind): "boolean" or "score"
- `min_score` (int, optional): Lower bound for score traits (default: 1)
- `max_score` (int, optional): Upper bound for score traits (default: 5)

**Methods:**
- `validate_score(value)`: Validate score value for this trait

##### `Rubric`

Collection of evaluation traits.

**Attributes:**
- `traits` (List[RubricTrait]): List of evaluation traits

**Methods:**
- `get_trait_names()`: Get list of trait names
- `validate_evaluation(evaluation)`: Validate evaluation result

##### `RubricEvaluation`

Result of applying a rubric to a question-answer pair.

**Attributes:**
- `trait_scores` (Dict[str, Union[int, bool]]): Scores for each trait

#### Functions

##### `merge_rubrics(global_rubric, question_rubric)`

Merge global and question-specific rubrics.

**Parameters:**
- `global_rubric` (Rubric, optional): Global rubric
- `question_rubric` (Rubric, optional): Question-specific rubric

**Returns:**
- `Rubric`: Merged rubric

**Raises:**
- `ValueError`: If trait names conflict

## Prompts Module

### `karenina.prompts.answer_generation`

#### Constants

##### `ANSWER_GENERATION_SYS`

System prompt for generating Pydantic answer templates.

Contains detailed instructions for:
- Creating appropriate Pydantic classes
- Including proper field types and descriptions
- Adding verification methods
- Implementing granular scoring

##### `ANSWER_GENERATION_USER`

User prompt template for answer generation.

Template variables:
- `{question}`: The question text
- `{raw_answer}`: The expected answer

### `karenina.prompts.answer_evaluation`

#### Constants

##### `RUBRIC_EVALUATION_SYS`

System prompt for rubric-based evaluation.

##### `RUBRIC_EVALUATION_USER`

User prompt template for rubric evaluation.

## Utils Module

### `karenina.utils.code_parser`

#### Functions

##### `extract_and_combine_codeblocks(text)`

Extract and combine code blocks from text.

**Parameters:**
- `text` (str): Input text containing code blocks

**Returns:**
- `str`: Combined code content

**Example:**
```python
from karenina.utils.code_parser import extract_and_combine_codeblocks

code = extract_and_combine_codeblocks("""
Here's some code:
```python
def hello():
    print("Hello")
```

And more:
```python
def world():
    print("World")
```
""")
```

## Error Handling

### Common Exceptions

All modules use consistent error handling patterns:

1. **LLMError**: Base class for LLM-related errors
2. **ValidationError**: Pydantic validation errors
3. **FileNotFoundError**: File system errors
4. **ValueError**: Invalid parameter values
5. **KeyError**: Missing configuration keys

### Error Response Format

Most functions return structured error information:

```python
{
    "success": False,
    "error": "Error description",
    "error_type": "ErrorClassName",
    "details": {...}  # Additional context
}
```

## Type Annotations

All public APIs include comprehensive type annotations using Python 3.11+ syntax:

- `str | None` for optional strings
- `List[Type]` for lists
- `Dict[str, Any]` for dictionaries
- `Union[Type1, Type2]` for unions
- `Optional[Type]` for optional values

## Async Support

Currently, the library is primarily synchronous, but async support is planned for future versions. Some functions may accept async callbacks or return awaitable objects where indicated in their documentation.
