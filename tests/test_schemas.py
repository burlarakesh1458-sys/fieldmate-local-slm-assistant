import pytest
from pydantic import ValidationError

from fieldmate.schemas import DefectReport, PreflightReport, json_schema_for, normalise_aircraft_id

GOOD_PREFLIGHT = {
    "aircraft_id": "uav 07",
    "battery_pct": 92,
    "propellers": "ok",
    "motors": "ok",
    "gps": "ok",
    "compass": "ok",
    "weather_ok": True,
    "decision": "go",
    "issues": [],
}


@pytest.mark.parametrize(
    "raw,expected",
    [("uav-07", "UAV-07"), ("UAV_07", "UAV-07"), (" q4 alpha ", "Q4-ALPHA"), ("--", None)],
)
def test_normalise_aircraft_id(raw, expected):
    assert normalise_aircraft_id(raw) == expected


def test_preflight_valid_and_normalised():
    r = PreflightReport.model_validate(GOOD_PREFLIGHT)
    assert r.aircraft_id == "UAV-07"


@pytest.mark.parametrize(
    "patch",
    [{"propellers": "issue"}, {"battery_pct": 20}, {"weather_ok": False}],
)
def test_preflight_rule_forces_no_go(patch):
    with pytest.raises(ValidationError, match="no_go"):
        PreflightReport.model_validate({**GOOD_PREFLIGHT, **patch})


def test_preflight_no_go_with_reason_is_valid():
    r = PreflightReport.model_validate({**GOOD_PREFLIGHT, "battery_pct": 20, "decision": "no_go"})
    assert r.decision == "no_go"


def test_preflight_rejects_unknown_enum_and_extra_fields():
    with pytest.raises(ValidationError):
        PreflightReport.model_validate({**GOOD_PREFLIGHT, "gps": "great"})
    with pytest.raises(ValidationError):
        PreflightReport.model_validate({**GOOD_PREFLIGHT, "pilot": "raki"})


def test_battery_range_enforced():
    with pytest.raises(ValidationError):
        PreflightReport.model_validate({**GOOD_PREFLIGHT, "battery_pct": 140})


def test_defect_grounding_rule():
    base = {
        "aircraft_id": "UAV-03",
        "component": "frame",
        "severity": "critical",
        "grounded": True,
        "symptom": "hairline crack in arm",
        "action_required": "replace arm",
    }
    assert DefectReport.model_validate(base).grounded
    with pytest.raises(ValidationError, match="grounded must be true"):
        DefectReport.model_validate({**base, "grounded": False})
    with pytest.raises(ValidationError, match="grounded must be false"):
        DefectReport.model_validate({**base, "severity": "low", "grounded": True})


def test_json_schema_has_enums_for_constrained_decoding():
    schema = json_schema_for("defect")
    assert "critical" in str(schema["properties"]["severity"])
    assert set(schema["required"]) >= {"component", "severity", "grounded"}
