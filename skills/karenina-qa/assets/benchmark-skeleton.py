"""Minimal karenina QA benchmark skeleton."""
from karenina.benchmark import Benchmark
from karenina.schemas.config.models import ModelConfig
from karenina.schemas.entities import BaseAnswer, VerifiedField
from karenina.schemas.entities.question import Question
from karenina.schemas.primitives import BooleanMatch
from karenina.schemas.verification.config import VerificationConfig

question = Question(
    question="What is the primary pharmacological target of venetoclax?",
    raw_answer="BCL2 (B-cell lymphoma 2)",
)


class Answer(BaseAnswer):
    identifies_target: bool = VerifiedField(
        description=(
            "True if the response identifies BCL2 (including Bcl-2, BCL-2, "
            "or B-cell lymphoma 2) as the primary pharmacological target. "
            "False if BCL2 is not mentioned or a different protein is "
            "identified as the primary target."
        ),
        ground_truth=True,
        verify_with=BooleanMatch(),
    )


benchmark = Benchmark(name="drug-target-extraction")
benchmark.add_question(question, answer_template=Answer)

config = VerificationConfig(
    answering_models=[
        ModelConfig(
            id="answering",
            model_provider="anthropic",
            model_name="claude-sonnet-4-6",
        ),
    ],
    parsing_models=[
        ModelConfig(
            id="judge",
            model_provider="anthropic",
            model_name="claude-haiku-4-5",
        ),
    ],
    evaluation_mode="template_only",
)

results = benchmark.run_verification(config)
print(results.get_summary())
