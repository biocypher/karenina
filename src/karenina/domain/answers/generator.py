"""
Structured answer template generator for Karenina benchmarks.

This module implements a two-phase structured generation approach:
1. Ground truth extraction from question/answer pairs
2. Field description generation for judge prompts
3. Pydantic class code generation

This replaces the previous example-based prompting system with a more
reliable and standardized approach.
"""

import json
from typing import TYPE_CHECKING, Any

from langchain_core.output_parsers import BaseOutputParser, PydanticOutputParser
from pydantic import BaseModel, Field, ValidationError, field_validator
from tqdm import tqdm

from karenina.domain.questions.reader import read_questions_from_file
from karenina.schemas.domain import BaseAnswer  # noqa: F401

# Import from extracted modules
from .generator_code import (
    format_ground_truth_value,
    generate_verification_logic,
    python_type_to_annotation,
)
from .generator_code import (
    generate_pydantic_class as _generate_pydantic_class,
)
from .generator_prompts import (
    FIELD_DESCRIPTION_SYSTEM_PROMPT,
    FIELD_DESCRIPTION_USER_PROMPT_TEMPLATE,
    GROUND_TRUTH_SYSTEM_PROMPT,
    GROUND_TRUTH_USER_PROMPT_TEMPLATE,
    build_generation_chain,
    build_retry_chain,
)

# Backward compatibility aliases for internal functions
_format_ground_truth_value = format_ground_truth_value
_python_type_to_annotation = python_type_to_annotation
_generate_verification_logic = generate_verification_logic

# Re-export for backward compatibility
__all__ = [
    # Schema classes
    "GroundTruthField",
    "GroundTruthSpec",
    "AttributeDescriptions",
    "JSONOnlyOutputParser",
    # Prompt constants (re-exported from generator_prompts)
    "GROUND_TRUTH_SYSTEM_PROMPT",
    "GROUND_TRUTH_USER_PROMPT_TEMPLATE",
    "FIELD_DESCRIPTION_SYSTEM_PROMPT",
    "FIELD_DESCRIPTION_USER_PROMPT_TEMPLATE",
    # Code generation (re-exported from generator_code)
    "python_type_to_annotation",
    "generate_verification_logic",
    "format_ground_truth_value",
    # Public API
    "generate_answer_template",
    "generate_answer_templates_from_questions_file",
    "load_answer_templates_from_json",
    "inject_question_id_into_answer_class",
]

if TYPE_CHECKING:
    from karenina.schemas.workflow import ModelConfig


class GroundTruthField(BaseModel):
    """Schema describing a single ground-truth attribute."""

    name: str = Field(..., description="Attribute identifier suitable for a Pydantic field name.")
    type: str = Field(..., description="Python/Pydantic type annotation such as 'bool', 'str', 'List[str]'.")
    ground_truth: Any = Field(
        ..., description="The expected correct value for this attribute based on the reference answer."
    )


class GroundTruthSpec(BaseModel):
    """Schema for the ground-truth dictionary definition."""

    attributes: list[GroundTruthField] = Field(
        ..., description="Ordered collection of attributes required for judgement."
    )

    @field_validator("attributes")
    @classmethod
    def validate_no_string_attributes(cls, v: list[GroundTruthField]) -> list[GroundTruthField]:
        """Ensure no free string attributes are allowed."""
        forbidden_types = ["str", "List[str]", "Dict[str, str]"]
        errors = []

        for attr in v:
            if attr.type in forbidden_types:
                errors.append(
                    f"Attribute '{attr.name}' uses forbidden type '{attr.type}'. Use boolean attributes instead."
                )
            elif "str" in attr.type and not attr.type.startswith("Literal"):
                errors.append(
                    f"Attribute '{attr.name}' uses type '{attr.type}' which contains strings. Use boolean attributes or Literal types instead."
                )

        if errors:
            raise ValueError("Validation failed:\n" + "\n".join(errors))

        return v


class AttributeDescriptions(BaseModel):
    """Schema capturing instructions for each attribute."""

    field_descriptions: dict[str, str] = Field(
        ..., description="Mapping from attribute name to guidance text for judge prompts."
    )


class JSONOnlyOutputParser(BaseOutputParser[Any]):
    """Parser ensuring output is valid JSON before delegating to Pydantic parser.

    This parser handles markdown-wrapped JSON responses by stripping code blocks
    before attempting JSON parsing.
    """

    def __init__(self, inner: PydanticOutputParser[Any]):
        self._inner = inner

    def parse(self, text: str) -> Any:
        from karenina.utils.code import extract_and_combine_codeblocks

        # Try parsing directly first
        try:
            json.loads(text)
            return self._inner.parse(text)
        except json.JSONDecodeError:
            # If direct parsing fails, try stripping markdown code blocks
            stripped = extract_and_combine_codeblocks(text)
            if stripped:
                try:
                    json.loads(stripped)
                    return self._inner.parse(stripped)
                except json.JSONDecodeError as exc:
                    raise ValueError(
                        f"Model response is not valid JSON even after stripping code blocks: {exc}"
                    ) from exc
            else:
                # No code blocks found, original text is invalid
                raise ValueError("Model response is not valid JSON") from None

    @property
    def _type(self) -> str:
        """Return the type of parser for LangChain compatibility."""
        return "json_only_output_parser"


def _build_chain(stage: str, config: "ModelConfig") -> Any:
    """Build generation chain for a specific stage.

    Delegates to generator_prompts.build_generation_chain.
    """
    return build_generation_chain(
        stage=stage,
        config=config,
        GroundTruthSpec=GroundTruthSpec,
        AttributeDescriptions=AttributeDescriptions,
        JSONOnlyOutputParser=JSONOnlyOutputParser,
    )


def _generate_with_retry(
    stage: str,
    inputs: dict[str, Any],
    config: "ModelConfig",
    max_retries: int = 2,
) -> Any:
    """Generate output with retry logic on validation failures."""
    last_error = None

    for attempt in range(max_retries + 1):
        try:
            # Build/rebuild chain for each attempt (for error injection)
            if attempt > 0 and last_error:
                # Add error context on retry
                error_context = f"\n\nPREVIOUS ATTEMPT FAILED with error: {last_error}\nPlease fix the validation issues and try again."
                chain = build_retry_chain(
                    stage=stage,
                    config=config,
                    error_context=error_context,
                    GroundTruthSpec=GroundTruthSpec,
                    AttributeDescriptions=AttributeDescriptions,
                    JSONOnlyOutputParser=JSONOnlyOutputParser,
                )
            else:
                chain = _build_chain(stage, config)

            result = chain.invoke(inputs)
            return result

        except (ValidationError, ValueError) as e:
            last_error = str(e)
            if attempt == max_retries:
                raise ValueError(f"Failed after {max_retries + 1} attempts. Last error: {last_error}") from None
            continue

    raise ValueError("Unexpected error in retry logic")


def _generate_structured_outputs(
    question: str,
    answer: str,
    config: "ModelConfig",
    max_retries: int = 2,
) -> dict[str, Any]:
    """Generate structured outputs (ground truth + field descriptions) using two-phase approach."""
    # Phase 1: Generate ground truth specification
    gt_result = _generate_with_retry("ground_truth", {"question": question, "answer": answer}, config, max_retries)

    gt_json = gt_result.model_dump() if isinstance(gt_result, BaseModel) else gt_result

    # Phase 2: Generate field descriptions
    # Remove ground_truth values from spec for field description generation
    spec_for_descriptions = {
        "attributes": [{k: v for k, v in attr.items() if k != "ground_truth"} for attr in gt_json["attributes"]]
    }

    fd_result = _generate_with_retry(
        "field_descriptions",
        {
            "question": question,
            "answer": answer,
            "spec_json": json.dumps(spec_for_descriptions, ensure_ascii=False),
        },
        config,
        max_retries,
    )

    return {
        "attributes": gt_json["attributes"],
        "field_descriptions": fd_result.field_descriptions,
    }


def inject_question_id_into_answer_class(answer_class: type, question_id: str) -> type:
    """
    Programmatically inject the question ID into an Answer class.

    This creates a new class that automatically sets the question ID when instantiated,
    replacing the need for LLM-generated ID assignment in model_post_init.

    Args:
        answer_class: The Answer class generated by LLM
        question_id: The question ID to inject

    Returns:
        A new Answer class with programmatic ID assignment
    """

    class AnswerWithID(answer_class):  # type: ignore[misc]
        def model_post_init(self, __context: Any) -> None:
            # Call the original model_post_init if it exists
            if hasattr(super(), "model_post_init"):
                super().model_post_init(__context)
            # Set the question ID programmatically
            self.id = question_id

    # Preserve the original class name and metadata
    AnswerWithID.__name__ = answer_class.__name__
    AnswerWithID.__qualname__ = answer_class.__qualname__

    # Preserve the original source code
    if hasattr(answer_class, "_source_code"):
        AnswerWithID._source_code = answer_class._source_code

    return AnswerWithID


def generate_answer_template(
    question: str,
    raw_answer: str,
    model: str | None = None,
    model_provider: str | None = None,
    temperature: float | None = None,
    interface: str | None = None,
    endpoint_base_url: str | None = None,
    endpoint_api_key: str | None = None,
    config: "ModelConfig | None" = None,
) -> str:
    """
    Generate an answer template using structured two-phase approach.

    Args:
        question: The question to generate an answer template for.
        raw_answer: The raw answer to the question.
        model: The model to use (when not using config).
        model_provider: The provider of the model (when not using config).
        temperature: The temperature of the model (when not using config).
        interface: The interface to use (when not using config).
        endpoint_base_url: The OpenAI-compatible endpoint base URL (when not using config).
        endpoint_api_key: The API key for the endpoint (when not using config).
        config: ModelConfig object (takes precedence over individual params).

    Returns:
        The python code for the answer template.
    """
    if config is not None:
        # Use provided ModelConfig
        model_config = config
    else:
        # Import ModelConfig and SecretStr dynamically to avoid circular imports
        from pydantic import SecretStr

        from karenina.schemas.workflow import ModelConfig

        # Create ModelConfig from individual parameters
        model_config = ModelConfig(
            id="template-generator",
            model_name=model or "gemini-2.0-flash",
            model_provider=model_provider or "google_genai",
            temperature=temperature if temperature is not None else 0.0,
            interface=interface or "langchain",  # type: ignore[arg-type]
            system_prompt="",  # Not used in structured generation
            endpoint_base_url=endpoint_base_url,
            endpoint_api_key=SecretStr(endpoint_api_key) if endpoint_api_key else None,
        )

    # Phase 1 & 2: Generate structured outputs (ground truth + field descriptions)
    max_retries = getattr(model_config, "max_retries", 2)  # Default to 2 if not set
    spec = _generate_structured_outputs(question, raw_answer, model_config, max_retries=max_retries)

    # Phase 3: Generate Pydantic class code
    template_code = _generate_pydantic_class(spec)

    return template_code


def generate_answer_templates_from_questions_file(
    questions_py_path: str,
    model: str = "gemini-2.0-flash",
    model_provider: str = "google_genai",
    interface: str = "langchain",
    return_blocks: bool = False,
) -> dict[str, Any] | tuple[dict[str, Any], dict[str, str]]:
    """
    Given a path to a questions.py file, dynamically import all_questions from it,
    generate answer templates for each question using the structured approach,
    and return a dictionary mapping question.id to the generated Answer class.
    """
    from karenina.utils.code import extract_and_combine_codeblocks

    # Use the reader function to get all questions
    all_questions = read_questions_from_file(questions_py_path)

    answer_templates = {}
    all_code_blocks = {}
    for _, question in tqdm(enumerate(all_questions)):
        answer_template = generate_answer_template(
            question.question,
            question.raw_answer or "",
            model=model,
            model_provider=model_provider,
            interface=interface,
        )
        # Try to extract code blocks (for old markdown-wrapped responses)
        # If no blocks found, use the answer_template directly (new plain Python responses)
        code_blocks = extract_and_combine_codeblocks(answer_template)
        if not code_blocks:
            code_blocks = answer_template

        # define the class in a local namespace
        local_ns: dict[str, Any] = {}
        exec(code_blocks, globals(), local_ns)
        Answer = local_ns["Answer"]

        # Store the template code for exec-created classes
        Answer._source_code = code_blocks

        # Inject the question ID programmatically
        AnswerWithID = inject_question_id_into_answer_class(Answer, question.id)
        answer_templates[question.id] = AnswerWithID

        if return_blocks:
            all_code_blocks[question.id] = code_blocks

    if return_blocks:
        return answer_templates, all_code_blocks
    else:
        return answer_templates


def load_answer_templates_from_json(
    json_file_path: str, return_blocks: bool = False
) -> dict[str, type] | tuple[dict[str, type], dict[str, str]]:
    """
    Load answer templates from a JSON file containing code blocks.

    Args:
        json_file_path: Path to the JSON file containing code blocks
        return_blocks: Whether to also return the code blocks dictionary

    Returns:
        If return_blocks is False: Dictionary mapping question IDs to Answer classes
        If return_blocks is True: Tuple of (answer_templates, code_blocks)
    """
    # Read the JSON file
    with open(json_file_path) as f:
        all_code_blocks = json.load(f)

    answer_templates = {}
    for question_id, code_blocks in all_code_blocks.items():
        # Define the class in a local namespace
        local_ns: dict[str, Any] = {}
        exec(code_blocks, globals(), local_ns)
        Answer = local_ns["Answer"]

        # Store the template code for exec-created classes
        Answer._source_code = code_blocks

        # Inject the question ID programmatically
        AnswerWithID = inject_question_id_into_answer_class(Answer, question_id)
        answer_templates[question_id] = AnswerWithID

    if return_blocks:
        return answer_templates, all_code_blocks
    else:
        return answer_templates
