"""
Structured answer template generator for Karenina benchmarks.

This module implements a multi-phase structured generation approach:
0. Planning: free-text field design reasoning (optional)
1. Ground truth extraction from question/answer pairs
2. Field description generation for judge prompts
3. Pydantic class code generation
4. Smoke test: exec() and verify() the generated code

This module uses the port/adapter pattern for LLM invocation,
decoupling from specific LLM backends like LangChain.
"""

import json
import logging
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel, Field, field_validator
from tqdm import tqdm

from karenina.adapters import get_llm
from karenina.benchmark.authoring.questions.reader import read_questions_from_file
from karenina.benchmark.verification.utils.class_discovery import find_answer_class
from karenina.benchmark.verification.utils.template_validation import (
    _build_exec_namespace,
)
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
    PLANNING_SYSTEM_PROMPT,
    PLANNING_USER_PROMPT_TEMPLATE,
)

logger = logging.getLogger(__name__)

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
    from karenina.schemas.config import ModelConfig
    from karenina.schemas.entities import Question


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
    def validate_attribute_types(cls, v: list[GroundTruthField]) -> list[GroundTruthField]:
        """Validate attribute types.

        Allows str and list[str] (verified via ExactMatch/SetContainment).
        Forbids Dict[str, str] which is not well-supported.
        Logs a warning for str fields that lack a Literal constraint.
        """
        forbidden_types = ["Dict[str, str]"]
        errors = []

        for attr in v:
            if attr.type in forbidden_types:
                errors.append(f"Attribute '{attr.name}' uses forbidden type '{attr.type}'.")
            elif attr.type == "str":
                logger.warning(
                    "Attribute '%s' uses free-text 'str' type. Consider using "
                    "Literal types for more precise verification.",
                    attr.name,
                )

        if errors:
            raise ValueError("Validation failed:\n" + "\n".join(errors))

        return v


class FieldDescriptionItem(BaseModel):
    """A single field description entry."""

    name: str = Field(..., description="The attribute name (must match an attribute from the ground-truth spec).")
    description: str = Field(..., description="Instructional text explaining what to extract from the response.")
    extraction_hint: str | None = Field(
        None,
        description="Optional hint for the extraction model about normalization or formatting concerns.",
    )


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


def _generate_plan(
    question: str,
    answer: str,
    config: "ModelConfig",
    answer_notes: str | None = None,
) -> str:
    """Phase 0: Generate a field design plan via free-text LLM reasoning.

    Args:
        question: The question text.
        answer: The reference answer text.
        config: Model configuration.
        answer_notes: Optional interpretation guidance.

    Returns:
        Free-text plan string with field design reasoning.
    """
    llm = get_llm(config)

    answer_notes_section = ""
    if answer_notes:
        answer_notes_section = f"\nAnswer Notes:\n{answer_notes}\n"

    messages = [
        Message.system(PLANNING_SYSTEM_PROMPT),
        Message.user(
            PLANNING_USER_PROMPT_TEMPLATE.format(
                question=question,
                answer=answer,
                answer_notes_section=answer_notes_section,
            )
        ),
    ]

    response = llm.invoke(messages)
    return response.content


def _smoke_test_generated_code(template_code: str) -> tuple[bool, str | None]:
    """Phase 4: Smoke test generated code by exec() and verify().

    Executes the generated template code, discovers the Answer class,
    instantiates it with ground truth values, and runs verify().

    Args:
        template_code: The generated Python code string.

    Returns:
        Tuple of (success, error_message).
    """
    ns = _build_exec_namespace()
    try:
        exec(template_code, ns)  # noqa: S102
    except Exception as e:
        return False, f"exec() failed: {e}"

    try:
        answer_cls = find_answer_class(ns)
    except ValueError as e:
        return False, f"Class discovery failed: {e}"

    try:
        verified_fields = answer_cls._get_verified_fields()  # type: ignore[attr-defined]
        if verified_fields:
            # Build kwargs from ground truth values
            kwargs = {}
            for field_name, meta in verified_fields.items():
                kwargs[field_name] = meta.ground_truth
            instance = answer_cls(**kwargs)
            result = instance.verify()
            if not result:
                return False, "verify() returned False with ground truth values"
    except Exception as e:
        return False, f"Smoke test failed: {e}"

    return True, None


def _generate_structured_outputs(
    question: str,
    answer: str,
    config: "ModelConfig",
    max_retries: int = 2,
    answer_notes: str | None = None,
    planning_enabled: bool = True,
) -> dict[str, Any]:
    """Generate structured outputs (ground truth + field descriptions) using multi-phase approach.

    When planning is enabled, Phase 0 produces free-text reasoning that is
    passed as context to Phase 1 and Phase 2.

    Retry logic with error feedback is handled by the adapter's with_structured_output().
    """
    answer_notes_section = ""
    if answer_notes:
        answer_notes_section = f"\nAnswer Notes (interpretation guidance):\n{answer_notes}\n"

    # Phase 0: Planning (optional)
    plan_section = ""
    if planning_enabled:
        logger.info("Phase 0: Generating field design plan")
        plan_text = _generate_plan(question, answer, config, answer_notes)
        plan_section = f"\nField Design Plan (use as guidance):\n{plan_text}\n"

    # Phase 1: Generate ground truth specification
    gt_result = _generate_structured_output(
        "ground_truth",
        {
            "question": question,
            "answer": answer,
            "answer_notes_section": answer_notes_section,
            "plan_section": plan_section,
        },
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
            "answer_notes_section": answer_notes_section,
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
    extraction_hints_dict = {
        item.name: item.extraction_hint for item in fd_typed.field_descriptions if item.extraction_hint is not None
    }

    return {
        "attributes": gt_json["attributes"],
        "field_descriptions": field_descriptions_dict,
        "extraction_hints": extraction_hints_dict,
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
    question_obj: "Question",
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
        question_obj: The Question object.
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

        from karenina.schemas.config import ModelConfig

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

    # Phase 0, 1, & 2: Generate structured outputs (plan + ground truth + field descriptions)
    max_retries = getattr(model_config, "max_retries", 2)  # Default to 2 if not set
    spec = _generate_structured_outputs(
        question_obj.question,
        question_obj.raw_answer or "",
        model_config,
        max_retries=max_retries,
        answer_notes=question_obj.answer_notes,
    )

    # Phase 3: Generate Pydantic class code
    template_code = _generate_pydantic_class(spec)

    # Phase 4: Smoke test the generated code
    success, error_msg = _smoke_test_generated_code(template_code)
    if not success:
        logger.warning(
            "Smoke test failed for question '%s': %s",
            question_obj.id,
            error_msg,
        )
        # TODO: retry logic (feed error back to Phase 0 and re-run all phases,
        # max 1 retry) is future work; for now, return the code as-is.

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
            question,
            model=model,
            model_provider=model_provider,
            interface=interface,
        )
        # Try to extract code blocks (for old markdown-wrapped responses)
        # If no blocks found, use the answer_template directly (new plain Python responses)
        code_blocks = extract_and_combine_codeblocks(answer_template)
        if not code_blocks:
            code_blocks = answer_template

        # Execute in a namespace with VerifiedField types available
        ns = _build_exec_namespace()
        exec(code_blocks, ns)  # noqa: S102
        Answer = find_answer_class(ns)

        # Store the template code for exec-created classes
        Answer._source_code = code_blocks  # type: ignore[attr-defined]

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
        # Execute in a namespace with VerifiedField types available
        ns = _build_exec_namespace()
        exec(code_blocks, ns)  # noqa: S102
        Answer = find_answer_class(ns)

        # Store the template code for exec-created classes
        Answer._source_code = code_blocks  # type: ignore[attr-defined]

        # Inject the question ID programmatically
        AnswerWithID = inject_question_id_into_answer_class(Answer, question_id)
        answer_templates[question_id] = AnswerWithID

    if return_blocks:
        return answer_templates, all_code_blocks
    else:
        return answer_templates
