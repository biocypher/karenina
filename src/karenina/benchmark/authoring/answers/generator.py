"""
Structured answer template generator for Karenina benchmarks.

This module implements a two-phase structured generation approach:
1. Ground truth extraction from question/answer pairs
2. Field description generation for judge prompts
3. Pydantic class code generation

This module uses the port/adapter pattern for LLM invocation,
decoupling from specific LLM backends like LangChain.
"""

import json
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel, Field, field_validator
from tqdm import tqdm

from karenina.adapters import get_llm
from karenina.benchmark.authoring.questions.reader import read_questions_from_file
from karenina.ports import Message
from karenina.schemas.entities import BaseAnswer  # noqa: F401

# Import from extracted modules
from .generator_code import format_ground_truth_value as _format_ground_truth_value
from .generator_code import generate_pydantic_class as _generate_pydantic_class
from .generator_prompts import (
    FIELD_DESCRIPTION_SYSTEM_PROMPT,
    FIELD_DESCRIPTION_USER_PROMPT_TEMPLATE,
    GROUND_TRUTH_SYSTEM_PROMPT,
    GROUND_TRUTH_USER_PROMPT_TEMPLATE,
)

__all__ = [
    # Schema classes (used by builder.py)
    "GroundTruthField",
    # Public API
    "generate_answer_template",
    "generate_answer_templates_from_questions_file",
    "load_answer_templates_from_json",
    "inject_question_id_into_answer_class",
    # Internal API used by builder.py
    "_generate_pydantic_class",
    "_format_ground_truth_value",
]

if TYPE_CHECKING:
    from karenina.schemas.workflow import ModelConfig


class GroundTruthField(BaseModel):
    """Schema describing a single ground-truth attribute."""

    name: str = Field(..., description="Attribute identifier suitable for a Pydantic field name.")
    type: str = Field(..., description="Python/Pydantic type annotation such as 'bool', 'str', 'List[str]'.")
    # Note: Avoid `Any` in unions as it creates invalid JSON schema for some providers (e.g., Anthropic beta.messages.parse).
    ground_truth: bool | int | float | str | list[bool | int | float | str] | dict[str, bool | int | float | str] = (
        Field(..., description="The expected correct value for this attribute based on the reference answer.")
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


class FieldDescriptionItem(BaseModel):
    """A single field description entry."""

    name: str = Field(..., description="The attribute name (must match an attribute from the ground-truth spec).")
    description: str = Field(..., description="Instructional text explaining what to extract from the response.")


class AttributeDescriptions(BaseModel):
    """Schema capturing instructions for each attribute.

    Note: We use a list instead of dict because Anthropic's beta.messages.parse
    does not properly handle dict[str, str] schemas (returns empty dicts).
    """

    field_descriptions: list[FieldDescriptionItem] = Field(
        ..., description="List of field descriptions for judge prompts."
    )


def _generate_structured_output(
    stage: str,
    inputs: dict[str, Any],
    config: "ModelConfig",
    output_schema: type[BaseModel],
    max_retries: int = 0,
) -> BaseModel:
    """Generate structured output using LLM adapter.

    Args:
        stage: The generation stage ("ground_truth" or "field_descriptions").
        inputs: Dictionary of inputs to format the prompt template.
        config: Model configuration.
        output_schema: Pydantic model class for structured output.
        max_retries: Maximum retry attempts on validation failure (default: 0).
            Retry logic with error feedback is handled by the adapter.

    Returns:
        Parsed Pydantic model instance.

    Raises:
        TypeError: If the adapter does not return a valid Pydantic model instance.
        ValueError: If generation fails after all retry attempts.
    """
    # Select prompts based on stage
    if stage == "ground_truth":
        system_prompt = GROUND_TRUTH_SYSTEM_PROMPT
        user_template = GROUND_TRUTH_USER_PROMPT_TEMPLATE
    else:
        system_prompt = FIELD_DESCRIPTION_SYSTEM_PROMPT
        user_template = FIELD_DESCRIPTION_USER_PROMPT_TEMPLATE

    # Format user message
    user_content = user_template.format(**inputs)

    # Get LLM with structured output and retry support
    llm = get_llm(config).with_structured_output(output_schema, max_retries=max_retries)

    # Build messages
    messages = [
        Message.system(system_prompt),
        Message.user(user_content),
    ]

    # Invoke and return parsed model from raw
    response = llm.invoke(messages)

    # The adapter guarantees response.raw is the Pydantic model instance
    if isinstance(response.raw, output_schema):
        return response.raw

    # Fail if the adapter did not return the expected type
    raise TypeError(
        f"Adapter returned invalid structured output. "
        f"Expected {output_schema.__name__}, got {type(response.raw).__name__}. "
        f"The adapter's with_structured_output() must guarantee a Pydantic model in response.raw."
    )


def _generate_structured_outputs(
    question: str,
    answer: str,
    config: "ModelConfig",
    max_retries: int = 2,
) -> dict[str, Any]:
    """Generate structured outputs (ground truth + field descriptions) using two-phase approach.

    Retry logic with error feedback is handled by the adapter's with_structured_output().
    """
    # Phase 1: Generate ground truth specification
    gt_result = _generate_structured_output(
        "ground_truth",
        {"question": question, "answer": answer},
        config,
        GroundTruthSpec,
        max_retries,
    )

    gt_json = gt_result.model_dump() if isinstance(gt_result, BaseModel) else gt_result

    # Phase 2: Generate field descriptions
    # Remove ground_truth values from spec for field description generation
    spec_for_descriptions = {
        "attributes": [{k: v for k, v in attr.items() if k != "ground_truth"} for attr in gt_json["attributes"]]
    }

    fd_result = _generate_structured_output(
        "field_descriptions",
        {
            "question": question,
            "answer": answer,
            "spec_json": json.dumps(spec_for_descriptions, ensure_ascii=False),
        },
        config,
        AttributeDescriptions,
        max_retries,
    )

    # Cast to AttributeDescriptions to access field_descriptions attribute
    fd_typed = (
        fd_result
        if isinstance(fd_result, AttributeDescriptions)
        else AttributeDescriptions.model_validate(fd_result.model_dump())
    )

    # Convert list of FieldDescriptionItem to dict[str, str]
    # (list format is needed for Anthropic's beta.messages.parse compatibility)
    field_descriptions_dict = {item.name: item.description for item in fd_typed.field_descriptions}

    return {
        "attributes": gt_json["attributes"],
        "field_descriptions": field_descriptions_dict,
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
            model_name=model,
            model_provider=model_provider,
            temperature=temperature if temperature is not None else 0.0,
            interface=interface,  # type: ignore[arg-type]
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
    model: str,
    model_provider: str,
    interface: str,
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
