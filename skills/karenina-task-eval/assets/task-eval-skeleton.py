"""Minimal karenina TaskEval skeleton."""
from karenina.benchmark.task_eval import TaskEval
from karenina.schemas.config.models import ModelConfig
from karenina.schemas.entities import BaseAnswer, VerifiedField
from karenina.schemas.primitives import BooleanMatch
from karenina.schemas.verification.config import VerificationConfig

task = TaskEval(task_id="my-evaluation")
task.log(
    "BCL2 is the primary pharmacological target of venetoclax. "
    "It works by inhibiting the anti-apoptotic BCL-2 protein."
)


class Answer(BaseAnswer):
    identifies_bcl2: bool = VerifiedField(
        description=(
            "True if BCL2 is identified as the primary target of venetoclax. "
            "False otherwise."
        ),
        ground_truth=True,
        verify_with=BooleanMatch(),
    )


task.add_template(Answer)

config = VerificationConfig(
    parsing_models=[
        ModelConfig(
            id="judge",
            model_provider="anthropic",
            model_name="claude-haiku-4-5-20250514",
        ),
    ],
    parsing_only=True,
)

result = task.evaluate(config)
print(result.summary())
