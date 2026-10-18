"""Output record produced by the Cascade-XID pipeline.

One :class:`OutputRecord` is emitted per scored flow. The field set is fixed by
AGENTS.md and must not be extended without updating that document.

ATT&CK technique identifiers are validated against the official ``T####`` /
``T####.###`` shape so that malformed or invented identifiers cannot be written
to a report.
"""

from __future__ import annotations

import re
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

_ATTACK_ID_PATTERN = re.compile(r"^T\d{4}(?:\.\d{3})?$")


class OutputRecord(BaseModel):
    """Final output record for a single network flow."""

    model_config = ConfigDict(extra="forbid")

    flow_id: str = Field(min_length=1)
    final_classification: str = Field(min_length=1)
    explanation: str = Field(min_length=1)
    attack_technique: str | None = None
    confidence: float = Field(ge=0.0, le=1.0)
    recommended_action: str = Field(min_length=1)
    evidence_used: list[str] = Field(default_factory=list)
    tier_reached: Literal[1, 2]
    llm_invoked: bool

    @field_validator("attack_technique")
    @classmethod
    def _validate_attack_id(cls, value: str | None) -> str | None:
        if value is None:
            return None
        if not _ATTACK_ID_PATTERN.match(value):
            raise ValueError(
                f"invalid ATT&CK technique id {value!r}; expected T#### or T####.###"
            )
        return value

    @model_validator(mode="after")
    def _check_tier_consistency(self) -> "OutputRecord":
        if self.llm_invoked and self.tier_reached != 2:
            raise ValueError("llm_invoked=True requires tier_reached=2")
        if self.tier_reached == 1 and self.evidence_used:
            raise ValueError("tier_reached=1 must not cite retrieved evidence")
        return self
