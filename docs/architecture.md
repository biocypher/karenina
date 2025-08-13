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

Karenina is built as a modular, extensible framework for LLM benchmarking. The architecture follows a three-stage pipeline pattern with clear separation of concerns between data extraction, template generation, and verification. At the top level, the Benchmark class provides a unified interface that orchestrates the entire workflow while maintaining full interoperability with the GUI through JSON-LD format.

## System Design

### High-Level Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                         Karenina Framework                       │
├─────────────────────────────────────────────────────────────────┤
│                                                                   │
│                   ┌──────────────────────┐                       │
│                   │   Benchmark Class    │  ← High-level API     │
│                   │  (JSON-LD Orchestrator) │                    │
│                   └──────────────────────┘                       │
│                            │                                     │
│           ┌────────────────┼────────────────┐                    │
│           │                │                │                    │
│           ↓                ↓                ↓                    │
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
│                                                                   │
│ ←─→ JSON-LD Format for GUI Interoperability                      │
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

### Benchmark Class (High-Level Orchestration)

The Benchmark class serves as the primary interface for creating, managing, and executing benchmarks. It provides a comprehensive abstraction layer over the underlying modules.

```python
# Key Components
benchmark.py            # Main Benchmark class
├── create()                    # Factory method for new benchmarks
├── load() / save()            # JSON-LD persistence
├── add_question()             # Question management
├── add_answer_template()      # Template management
├── add_global_rubric_trait()  # Rubric configuration
├── run_verification()         # Orchestrate verification
├── get_health_report()        # Readiness assessment
├── export_questions_python()  # Export capabilities
├── to_csv() / to_markdown()   # Multiple export formats
└── filter_questions()         # Query and search

jsonld_converter.py     # JSON-LD serialization
├── to_jsonld()                # Convert to JSON-LD format
├── from_jsonld()              # Load from JSON-LD format
├── validate_schema()          # Schema validation
└── upgrade_format()           # Format migration
```

**Key Features:**
- **JSON-LD Storage**: Uses schema.org vocabulary for standardized linked data
- **GUI Interoperability**: Full bidirectional compatibility with Karenina GUI
- **Comprehensive Metadata Management**: Track creation, modification, authorship, and custom properties
- **Built-in Validation**: Health checks and readiness assessment
- **Multiple Export Formats**: CSV, Markdown, JSON, Python files
- **Query and Filtering**: Advanced search and filtering capabilities
- **Rubric Management**: Global and question-specific evaluation criteria

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

Orchestrates verification workflows and manages results. This module is now organized around the high-level Benchmark class with supporting components for verification execution.

```python
# Structure
benchmark.py        # Main Benchmark class (high-level orchestration)
├── Benchmark                  # Primary interface class
├── create()                   # Factory methods
├── load() / save()           # JSON-LD persistence
└── verification orchestration # Workflow management

jsonld_converter.py # JSON-LD format handling
├── to_jsonld()               # Serialization to JSON-LD
├── from_jsonld()             # Deserialization from JSON-LD
├── validate_jsonld_schema()  # Schema validation
└── format_migration()        # Version compatibility

models.py           # Configuration models
├── ModelConfiguration        # Single model config
├── VerificationConfig       # Multi-model verification config
└── VerificationResult       # Result data model

verification/
├── orchestrator.py          # Multi-model orchestration
│   └── run_question_verification()
├── runner.py                # Single model execution
│   └── run_single_model_verification()
├── validation.py            # Response validation
│   └── validate_answer_template()
└── rubric_evaluator.py      # Rubric-based evaluation
    └── RubricEvaluator

verifier.py         # Legacy verification interface
exporter.py         # Result export utilities
```

**Key Features:**
- **High-level Benchmark orchestration** with comprehensive workflow management
- **JSON-LD format support** for standardized data interchange
- **GUI interoperability** through shared format specification
- Multi-model testing configurations
- Replicate support for statistical analysis
- Rubric-based qualitative evaluation
- Multiple export formats (JSON, CSV, Markdown, Python)

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

### High-Level Benchmark Class Workflow

```
1. Benchmark Creation/Loading
   - Create new benchmark with Benchmark.create()
   - OR Load existing benchmark from JSON-LD file
   - OR Load GUI-exported benchmark
         ↓
2. Question Management
   - Add questions directly with add_question()
   - OR Extract from files and bulk add
   - Include metadata and custom properties
         ↓
3. Template Management
   - Generate templates with LLM integration
   - OR Add custom templates manually
   - Mark questions as finished
         ↓
4. Rubric Configuration
   - Add global rubric traits
   - Add question-specific rubrics
   - Configure evaluation criteria
         ↓
5. Validation & Health Checks
   - Assess benchmark readiness
   - Validate data integrity
   - Check completion status
         ↓
6. Verification Execution
   - Configure models and parameters
   - Run verification workflow
   - Process results
         ↓
7. Export & Persistence
   - Save as JSON-LD for GUI compatibility
   - Export as CSV, Markdown, or Python
   - Maintain metadata integrity
```

### Traditional Pipeline Flow (Lower-Level)

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

### JSON-LD Interoperability Flow

```
Python Benchmark Creation
         ↓
Save as JSON-LD (schema.org format)
         ↓
Load in GUI → User Modifications → Export from GUI
         ↓
Load back in Python → Enhance → Save
         ↓
Seamless round-trip data integrity
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
                                              ↑
                                              │
                                    ┌─────────────┐
                                    │ Benchmark   │
                                    │   Class     │  ← High-level API
                                    │(Orchestrator)│
                                    └─────────────┘
```

### Import Relationships

- **schemas**: Base module with no dependencies
- **llm**: Depends on schemas for data models
- **prompts**: Standalone prompt templates
- **utils**: Utility functions with minimal dependencies
- **questions**: Depends on schemas for Question class
- **answers**: Depends on llm, schemas, prompts, utils
- **benchmark** (module): Depends on all other modules for verification workflows
- **Benchmark** (class): High-level orchestrator that depends on benchmark module and provides unified API with JSON-LD format support

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

### 8. High-Level Benchmark Class

**Decision**: Create a comprehensive Benchmark class as the primary interface.

**Rationale**:
- Simplify complex multi-step workflows
- Provide consistent API for common operations
- Abstract away implementation details
- Enable better testing and validation
- Support both programmatic and interactive usage

### 9. JSON-LD Format Adoption

**Decision**: Use JSON-LD with schema.org vocabulary for data interchange.

**Rationale**:
- Standardized linked data format with semantic meaning
- Better interoperability between Python library and GUI
- Extensible format that supports custom properties
- Version compatibility and migration support
- Industry-standard format for structured data

### 10. GUI-Python Bidirectional Compatibility

**Decision**: Ensure full round-trip compatibility between Python library and GUI.

**Rationale**:
- Users can start work in either environment
- No data loss during transitions
- Leverage strengths of both interfaces
- Flexible workflow support
- Maximize user adoption and utility

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
