import json

from fieldmate.extractor import Extractor, parse_and_validate, total_metrics

VALID_DEFECT = json.dumps(
    {
        "aircraft_id": "uav-07",
        "component": "propeller",
        "severity": "high",
        "grounded": True,
        "symptom": "prop tip chipped",
        "action_required": "replace propeller",
    }
)
RULE_BREAKING_DEFECT = VALID_DEFECT.replace('"grounded": true', '"grounded": false')


def test_parse_and_validate_reports_bad_json():
    model, err = parse_and_validate("defect", "{not json")
    assert model is None and "not valid JSON" in err


def test_first_try_success(fake, make_client):
    fake.replies = [VALID_DEFECT]
    res = Extractor(make_client()).extract("defect", "UAV-07 prop chipped")
    assert res.ok and res.first_try_valid and res.n_attempts == 1
    assert res.data["aircraft_id"] == "UAV-07"
    # constrained decoding: the JSON schema is sent as `format`
    assert isinstance(fake.requests[0]["format"], dict)
    assert fake.requests[0]["format"]["title"] == "DefectReport"


def test_retry_with_error_feedback_recovers(fake, make_client):
    fake.replies = [RULE_BREAKING_DEFECT, VALID_DEFECT]
    res = Extractor(make_client()).extract("defect", "UAV-07 prop chipped")
    assert res.ok and not res.first_try_valid and res.n_attempts == 2
    retry_msgs = fake.requests[1]["messages"]
    assert retry_msgs[-2]["role"] == "assistant"
    assert "failed validation" in retry_msgs[-1]["content"]
    assert "grounded must be true" in retry_msgs[-1]["content"]
    assert total_metrics(res).output_tokens == 40  # retries are counted in latency/tokens


def test_graceful_failure_after_one_retry(fake, make_client):
    fake.replies = ["{oops", RULE_BREAKING_DEFECT, VALID_DEFECT]
    res = Extractor(make_client()).extract("defect", "x")
    assert not res.ok and res.n_attempts == 2
    assert res.data is None and "validation failed after 2" in res.error
    assert len(fake.replies) == 1  # did not call a third time


def test_unconstrained_mode_uses_plain_json_format(fake, make_client):
    fake.replies = [VALID_DEFECT]
    Extractor(make_client()).extract("defect", "x", constrained=False)
    assert fake.requests[0]["format"] == "json"


def test_infra_error_fails_fast_without_retry(fake, make_client):
    fake.fail_with = 500
    res = Extractor(make_client()).extract("defect", "x")
    assert not res.ok and res.error.startswith("ollama error") and res.n_attempts == 0


def test_zero_retries_configurable(fake, make_client):
    fake.replies = [RULE_BREAKING_DEFECT, VALID_DEFECT]
    res = Extractor(make_client(), max_retries=0).extract("defect", "x")
    assert not res.ok and res.n_attempts == 1
