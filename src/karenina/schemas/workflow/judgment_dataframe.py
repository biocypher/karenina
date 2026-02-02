"""JudgmentDataFrameBuilder for converting deep judgment results to pandas DataFrames.

DEPRECATED: Import from `karenina.schemas.dataframes` instead.
"""

# Re-export from new location for backward compatibility
from ..dataframes.judgment import JudgmentDataFrameBuilder

__all__ = ["JudgmentDataFrameBuilder"]
