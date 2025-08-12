# Karenina Architecture

## Table of Contents

- [Overview](#overview)
- [System Design](#system-design)
- [Core Components](#core-components)
- [Data Flow](#data-flow)
- [Design Patterns](#design-patterns)
- [Module Dependencies](#module-dependencies)
- [Key Architectural Decisions](#key-architectural-decisions)

## Overview

Karenina is built as a modular, extensible framework for LLM benchmarking. The architecture follows a three-stage pipeline pattern with clear separation of concerns between data extraction, template generation, and verification.

## System Design

### High-Level Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                         Karenina Framework                       │
├─────────────────────────────────────────────────────────────────┤
│                                                                   │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐          │
│  │   Questions   │  │   Answers    │  │  Benchmark   │          │
│  │   Module      │→ │   Module     │→ │   Module     │          │
│  └──────────────┘  └──────────────┘  └──────────────┘          │
│         ↑                  ↑                  ↑                  │
│         └──────────────────┼──────────────────┘                  │
│                            │                                     │
│                     ┌──────────────┐                            │
│                     │  LLM Module  │                            │
│                     └──────────────┘                            │
│                            ↑                                     │
│                     ┌──────────────┐                            │
│                     │   Schemas     │                            │
│                     └──────────────┘                            │
└─────────────────────────────────────────────────────────────────┘
```

### Pipeline Stages

1. **Question Extraction Stage**
   - Input: Excel/CSV/TSV files
   - Processing: Parse files, extract questions, generate unique IDs
   - Output: Python file with Question objects

2. **Template Generation Stage**
   - Input: Question objects
   - Processing: LLM generates Pydantic validation classes
   - Output: Answer template classes with verification methods

3. **Verification Stage**
   - Input: Questions, templates, LLM configurations
   - Processing: Multi-model testing with replicates
   - Output: Verification results with scores and rubric evaluations

## Core Components

### LLM Module (`llm/`)

The LLM module provides a unified interface for interacting with various language model providers.

```python
# Key Components
interface.py         # Unified LLM interface
├── init_chat_model_unified()  # Factory for creating LLM instances
├── ChatSession                # Stateful conversation management
└── chat_with_llm()            # Main chat interface

manual_llm.py       # Human-in-the-loop support
├── ManualLLM                  # Mock LLM for manual responses
└── create_manual_llm()        # Factory for manual instances

manual_traces.py    # Trace management
├── save_manual_traces()       # Persist human responses
└── load_manual_traces()       # Load recorded responses
```

**Key Features:**
- Provider abstraction (OpenAI, Google, Anthropic, OpenRouter)
- Session management for stateful conversations
- Error handling with custom exception hierarchy
- Manual mode for human verification

### Questions Module (`questions/`)

Handles extraction and processing of benchmark questions from various file formats.

```python
# Key Components
extractor.py        # File extraction logic
├── extract_and_generate_questions()  # Main extraction pipeline
├── read_file_to_dataframe()         # Multi-format file reader
├── hash_question()                   # Generate unique IDs
└── generate_questions_file()        # Create Python output

reader.py           # Question file readers
├── read_questions_from_file()       # Load questions from Python file
└── import_module_from_path()        # Dynamic module loading
```

**Key Features:**
- Multi-format support (Excel, CSV, TSV)
- Automatic ID generation using MD5 hashing
- Column mapping for flexible data structures
- Python code generation for question storage

### Answers Module (`answers/`)

Generates and manages answer templates using LLMs.

```python
# Key Components
generator.py        # Template generation
├── generate_answer_template()              # Single template generation
├── generate_answer_templates_from_questions_file()  # Batch generation
└── inject_question_id_into_answer_class()  # ID injection

reader.py           # Template file readers
├── read_answer_templates_from_file()       # Load templates
└── parse_answer_class()                    # Parse Python classes
```

**Key Features:**
- Automatic Pydantic class generation
- Custom verification methods (verify, verify_granular)
- Batch processing capabilities
- Template validation and parsing

### Benchmark Module (`benchmark/`)

Orchestrates verification workflows and manages results.

```python
# Structure
models.py           # Configuration models
├── ModelConfiguration         # Single model config
├── VerificationConfig        # Multi-model verification config
└── VerificationResult        # Result data model

verification/
├── orchestrator.py           # Multi-model orchestration
│   └── run_question_verification()
├── runner.py                 # Single model execution
│   └── run_single_model_verification()
├── validation.py             # Response validation
│   └── validate_answer_template()
└── rubric_evaluator.py       # Rubric-based evaluation
    └── RubricEvaluator

verifier.py         # Legacy verification interface
exporter.py         # Result export utilities
```

**Key Features:**
- Multi-model testing configurations
- Replicate support for statistical analysis
- Rubric-based qualitative evaluation
- Multiple export formats (JSON, CSV)

### Schemas Module (`schemas/`)

Defines data models and validation schemas used throughout the framework.

```python
# Core Models
question_class.py   # Question data model
├── Question
│   ├── id: str              # Unique MD5 hash
│   ├── question: str        # Question text
│   ├── raw_answer: str      # Expected answer
│   └── tags: List[str]      # Categorization

answer_class.py     # Base answer template
├── BaseAnswer
│   ├── id: Optional[str]    # Question ID reference
│   ├── set_question_id()    # Programmatic ID setting
│   └── model_config         # Pydantic configuration

rubric_class.py     # Rubric evaluation models
├── RubricTrait              # Single evaluation criterion
├── Rubric                   # Collection of traits
├── RubricEvaluation        # Evaluation results
└── merge_rubrics()         # Combine global and question rubrics
```

### Prompts Module (`prompts/`)

Contains carefully crafted prompts for LLM interactions.

```python
# Prompt Templates
answer_generation.py
├── ANSWER_GENERATION_SYS    # System prompt for template generation
└── ANSWER_GENERATION_USER   # User prompt template

answer_evaluation.py
├── RUBRIC_EVALUATION_SYS    # System prompt for rubric evaluation
└── RUBRIC_EVALUATION_USER   # User prompt template
```

## Data Flow

### Complete Pipeline Flow

```
1. File Input (Excel/CSV/TSV)
         ↓
2. Question Extraction
   - Parse file with pandas
   - Generate MD5 hash IDs
   - Create Question objects
         ↓
3. Python File Generation
   - Export questions as Python code
   - Create all_questions list
         ↓
4. Template Generation
   - Load questions from Python file
   - Generate Pydantic classes via LLM
   - Add verification methods
         ↓
5. Verification Configuration
   - Define answering models
   - Define parsing models
   - Set replicate count
   - Configure rubrics (optional)
         ↓
6. Verification Execution
   - Run question through answering model
   - Parse response with parsing model
   - Validate against template
   - Evaluate with rubrics
         ↓
7. Result Aggregation
   - Collect verification results
   - Calculate scores
   - Generate statistics
         ↓
8. Export Results
   - JSON format for analysis
   - CSV format for spreadsheets
```

### Session Management Flow

```
User Request → ChatSession Creation
                    ↓
              Store in CHAT_SESSIONS
                    ↓
              Initialize LLM
                    ↓
              Add System Message
                    ↓
              Process Conversation
                    ↓
              Return Response
```

## Design Patterns

### Factory Pattern

Used for creating LLM instances based on provider configuration:

```python
def init_chat_model_unified(
    model: str,
    provider: str,
    interface: str,
    temperature: float
) -> BaseChatModel:
    """Factory method for creating LLM instances"""
    if interface == "manual":
        return create_manual_llm(traces)
    elif interface == "openrouter":
        return create_openrouter_llm(model)
    else:  # langchain
        return init_chat_model(model, provider, temperature)
```

### Template Method Pattern

Used in answer verification with base class defining the structure:

```python
class BaseAnswer(BaseModel):
    def verify(self) -> bool:
        """Template method for verification"""
        raise NotImplementedError

    def verify_granular(self) -> float:
        """Template method for granular scoring"""
        raise NotImplementedError
```

### Strategy Pattern

Used for different verification strategies (automated, manual, rubric-based):

```python
class VerificationStrategy:
    def verify(self, question, answer, template):
        pass

class AutomatedVerification(VerificationStrategy):
    def verify(self, question, answer, template):
        # Automated verification logic

class ManualVerification(VerificationStrategy):
    def verify(self, question, answer, template):
        # Manual verification logic
```

### Dependency Injection

Used throughout for providing LLM instances and configurations:

```python
def run_single_model_verification(
    question_id: str,
    question_text: str,
    template_code: str,
    answering_model: ModelConfiguration,  # Injected
    parsing_model: ModelConfiguration,    # Injected
    rubric: Optional[Rubric] = None      # Injected
) -> VerificationResult:
    # Use injected dependencies
```

## Module Dependencies

### Dependency Graph

```
                    ┌─────────────┐
                    │   schemas   │
                    └─────────────┘
                           ↑
        ┌──────────────────┼──────────────────┐
        │                  │                  │
        ↓                  ↓                  ↓
┌─────────────┐    ┌─────────────┐    ┌─────────────┐
│     llm     │    │   prompts   │    │    utils    │
└─────────────┘    └─────────────┘    └─────────────┘
        ↑                  ↑                  ↑
        └──────────────────┼──────────────────┘
                           │
        ┌──────────────────┼──────────────────┐
        │                  │                  │
        ↓                  ↓                  ↓
┌─────────────┐    ┌─────────────┐    ┌─────────────┐
│  questions  │    │   answers   │    │ benchmark   │
└─────────────┘    └─────────────┘    └─────────────┘
```

### Import Relationships

- **schemas**: Base module with no dependencies
- **llm**: Depends on schemas for data models
- **prompts**: Standalone prompt templates
- **utils**: Utility functions with minimal dependencies
- **questions**: Depends on schemas for Question class
- **answers**: Depends on llm, schemas, prompts, utils
- **benchmark**: Depends on all other modules

## Key Architectural Decisions

### 1. Provider Abstraction

**Decision**: Use LangChain as the abstraction layer for LLM providers.

**Rationale**:
- Consistent interface across providers
- Built-in retry logic and error handling
- Easy to add new providers
- Community support and maintenance

### 2. Pydantic for Validation

**Decision**: Use Pydantic for all data validation and schema definition.

**Rationale**:
- Type safety and runtime validation
- Automatic JSON serialization/deserialization
- Clear error messages
- Integration with modern Python tooling

### 3. Code Generation Approach

**Decision**: Generate Python code files for questions and templates.

**Rationale**:
- Human-readable and editable output
- Version control friendly
- Easy debugging and inspection
- No runtime parsing overhead

### 4. Stateless Verification

**Decision**: Each verification run is independent and stateless.

**Rationale**:
- Easier parallelization
- Reproducible results
- Simpler error recovery
- No hidden state bugs

### 5. Rubric System Design

**Decision**: Separate rubric evaluation from correctness checking.

**Rationale**:
- Qualitative vs quantitative assessment
- Flexible evaluation criteria
- Domain-specific customization
- Clear separation of concerns

### 6. Manual Mode Integration

**Decision**: Support manual/human verification alongside automated testing.

**Rationale**:
- Human-in-the-loop validation
- Ground truth establishment
- Edge case handling
- Quality assurance

### 7. Session Management

**Decision**: In-memory session storage for chat interactions.

**Rationale**:
- Simplicity over persistence
- Fast access patterns
- No database dependencies
- Suitable for single-user scenarios

## Performance Considerations

### Optimization Strategies

1. **Batch Processing**
   - Process multiple questions in single LLM calls where possible
   - Use async operations for I/O bound tasks

2. **Caching**
   - Cache LLM responses for identical inputs
   - Store parsed templates in memory

3. **Parallelization**
   - Run multiple model combinations in parallel
   - Process independent questions concurrently

4. **Resource Management**
   - Lazy loading of large dependencies
   - Connection pooling for API clients
   - Memory-efficient data structures

### Scalability Considerations

1. **Horizontal Scaling**
   - Stateless design allows multiple instances
   - Questions can be distributed across workers
   - Results can be aggregated from parallel runs

2. **Vertical Scaling**
   - Memory usage scales with number of questions
   - LLM API rate limits may constrain throughput
   - Consider batching for large datasets

## Security Considerations

### API Key Management

- Environment variables for sensitive data
- No hardcoded credentials
- Support for key rotation
- Provider-specific authentication

### Code Execution Safety

- Generated code is not automatically executed
- Template validation before use
- Sandboxed evaluation environment
- Input sanitization for file paths

### Data Privacy

- No persistent storage of sensitive data
- Session data cleared on restart
- Optional manual mode for sensitive content
- Configurable data retention policies

## Future Enhancements

### Planned Improvements

1. **Async/Await Support**
   - Full async pipeline for better performance
   - Concurrent LLM calls
   - Streaming responses

2. **Distributed Processing**
   - Job queue integration
   - Worker pool architecture
   - Result aggregation service

3. **Enhanced Caching**
   - Redis integration for distributed cache
   - Persistent cache options
   - Smart cache invalidation

4. **Advanced Analytics**
   - Statistical analysis tools
   - Visualization components
   - Comparative benchmarking

5. **Plugin System**
   - Custom provider plugins
   - Extension points for processing
   - Third-party integrations
