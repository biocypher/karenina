"""Deep judgment prompt builders for rubric trait evaluation.

Canonical location for DeepJudgmentPromptBuilder, which constructs prompts
for the multi-stage deep judgment evaluation of rubric traits:

1. Extract verbatim excerpts from the answer
2. (Optional) Assess hallucination risk using external search
3. Generate reasoning explaining how excerpts support the trait
4. Extract final score based on reasoning
"""

from karenina.benchmark.verification.prompts.deep_judgment.rubric.deep_judgment import (
    DeepJudgmentPromptBuilder,
)

__all__ = ["DeepJudgmentPromptBuilder"]
