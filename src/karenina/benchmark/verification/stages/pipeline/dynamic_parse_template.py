"""Dynamic template parsing stage."""

from ..core.base import BaseVerificationStage, VerificationContext


class DynamicParseTemplateStage(BaseVerificationStage):
    @property
    def name(self) -> str:
        return "DynamicParseTemplate"

    def execute(self, context: VerificationContext) -> None:
        context.mark_error("DynamicParseTemplateStage is not implemented")
