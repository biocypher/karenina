"""Validated compact records and result views for scenario executions."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, PrivateAttr

from karenina.schemas.dataframes.scenario import ScenarioDataFrameBuilder


class ScenarioTurnRecord(BaseModel):
    """Compact saved representation of one executed scenario turn."""

    model_config = ConfigDict(extra="forbid")

    node_id: str
    question_text: str
    raw_response: str
    parsed_fields: dict[str, Any] = Field(default_factory=dict)
    verify_result: bool | None = None
    verification_result_id: str | None = None


class ScenarioTerminalFailureRecord(BaseModel):
    """Compact saved representation of a terminal scenario failure."""

    model_config = ConfigDict(extra="forbid")

    node_id: str | None = None
    category: str | None = None
    stage: str | None = None
    reason: str | None = None


class ScenarioResultRecord(BaseModel):
    """Validated compact record from a saved scenario result export."""

    model_config = ConfigDict(extra="forbid")

    scenario_id: str
    status: Literal["completed", "limit_reached", "error", "timeout"]
    path: list[str] = Field(default_factory=list)
    turn_count: int
    history: list[ScenarioTurnRecord] = Field(default_factory=list)
    outcome_results: dict[str, bool | int | float] = Field(default_factory=dict)
    terminal_failure: ScenarioTerminalFailureRecord | None = None
    replicate: int | None = None


class ScenarioResults(BaseModel):
    """Public analysis view over live or loaded scenario executions."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    results: list[Any] = Field(default_factory=list)
    _dataframe_builder: ScenarioDataFrameBuilder | None = PrivateAttr(default=None)

    @property
    def dataframe_builder(self) -> ScenarioDataFrameBuilder:
        """Return the lazily constructed scenario DataFrame builder."""
        if self._dataframe_builder is None:
            self._dataframe_builder = ScenarioDataFrameBuilder(self.results)
        return self._dataframe_builder

    def to_dataframe(self) -> Any:
        """Return one row per scenario execution."""
        return self.dataframe_builder.build_scenario_dataframe()

    def to_turn_dataframe(self) -> Any:
        """Return one row per executed scenario turn."""
        return self.dataframe_builder.build_turn_dataframe()

    def to_outcome_dataframe(self) -> Any:
        """Return one row per scenario outcome criterion."""
        return self.dataframe_builder.build_outcome_dataframe()
