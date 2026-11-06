"""Pydantic v2 output contracts for drone field operations.

Two layers of validation:
  1. *Syntactic* - the JSON schema (sent to Ollama as `format`) constrains decoding, so the
     model can only emit tokens that form JSON matching these types/enums.
  2. *Semantic* - model validators encode the ops manual's rules (e.g. a defect rated
     high/critical MUST ground the aircraft). Constrained decoding cannot enforce these, so a
     violation triggers the retry-with-feedback loop in `extractor.py`.
"""

from __future__ import annotations

import re
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

CheckStatus = Literal["ok", "issue", "not_checked"]
Component = Literal[
    "propeller",
    "motor",
    "battery",
    "esc",
    "gps",
    "compass",
    "camera",
    "gimbal",
    "frame",
    "landing_gear",
    "radio",
    "flight_controller",
    "other",
]
Severity = Literal["low", "medium", "high", "critical"]

_ID_RE = re.compile(r"[^A-Z0-9]+")


def normalise_aircraft_id(value: str | None) -> str | None:
    """'uav 07' / 'UAV_07' / 'uav-07' -> 'UAV-07'."""
    if value is None:
        return None
    cleaned = _ID_RE.sub("-", value.strip().upper()).strip("-")
    return cleaned or None


class _Base(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    @field_validator("aircraft_id", mode="before", check_fields=False)
    @classmethod
    def _norm_id(cls, v):
        return normalise_aircraft_id(v) if isinstance(v, str) else v


class PreflightReport(_Base):
    """Structured result of a pilot's (messy) pre-flight checklist note."""

    aircraft_id: str | None = Field(description="Aircraft identifier, e.g. XY-12")
    battery_pct: int | None = Field(
        default=None,
        ge=0,
        le=100,
        description="Battery state of charge in percent, null if not stated",
    )
    propellers: CheckStatus
    motors: CheckStatus
    gps: CheckStatus
    compass: CheckStatus
    weather_ok: bool | None = Field(
        default=None, description="True if weather within limits, false if not, null if not stated"
    )
    decision: Literal["go", "no_go"]
    issues: list[str] = Field(default_factory=list, description="Short list of problems found")

    @model_validator(mode="after")
    def _decision_follows_rules(self) -> PreflightReport:
        reasons = []
        for name in ("propellers", "motors", "gps", "compass"):
            if getattr(self, name) == "issue":
                reasons.append(f"{name} has an issue")
        if self.battery_pct is not None and self.battery_pct < 30:
            reasons.append(f"battery {self.battery_pct}% is below the 30% minimum")
        if self.weather_ok is False:
            reasons.append("weather is not ok")
        if reasons and self.decision == "go":
            raise ValueError(
                "decision must be 'no_go' because " + "; ".join(reasons) + " (ops manual rule)"
            )
        if not reasons and self.decision == "no_go" and not self.issues:
            raise ValueError(
                "decision is 'no_go' but no issue, low battery or bad weather was recorded"
            )
        return self


class DefectReport(_Base):
    """Structured maintenance defect report from a free-text log line."""

    aircraft_id: str | None
    component: Component
    severity: Severity
    grounded: bool = Field(description="True if the aircraft must not fly until fixed")
    symptom: str = Field(min_length=3, max_length=300)
    action_required: str = Field(min_length=3, max_length=300)

    @model_validator(mode="after")
    def _grounding_matches_severity(self) -> DefectReport:
        must_ground = self.severity in ("high", "critical")
        if must_ground != self.grounded:
            raise ValueError(
                f"grounded must be {str(must_ground).lower()} when severity is '{self.severity}' "
                "(ops manual: high and critical defects ground the aircraft)"
            )
        return self


SCHEMAS: dict[str, type[BaseModel]] = {
    "preflight": PreflightReport,
    "defect": DefectReport,
}


def json_schema_for(kind: str) -> dict:
    """JSON schema passed to Ollama's `format` parameter for constrained decoding."""
    return SCHEMAS[kind].model_json_schema()
