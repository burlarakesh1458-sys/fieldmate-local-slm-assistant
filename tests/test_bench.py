import json
from pathlib import Path

import pytest

from fieldmate.bench.common import percentile, summarize
from fieldmate.bench.quality import load_testset, run_quality
from fieldmate.bench.scoring import (
    exact_match_rate,
    field_agreement,
    field_scores,
    mean_pairwise,
    rubric_score,
    text_similarity,
)
from fieldmate.schemas import SCHEMAS

TESTSET = Path(__file__).resolve().parents[1] / "data" / "testset.jsonl"


def test_percentiles_match_numpy_linear():
    xs = [1, 2, 3, 4, 10]
    assert percentile(xs, 50) == 3
    assert percentile(xs, 95) == pytest.approx(8.8)
    s = summarize([1.0, 2.0, 3.0])
    assert s["mean"] == 2.0 and s["p50"] == 2.0 and s["n"] == 3


def test_testset_is_well_formed_and_expected_outputs_obey_rules():
    rows = load_testset(TESTSET)
    assert 30 <= len(rows) <= 50
    assert len({r["id"] for r in rows}) == len(rows)
    for r in rows:
        if r["kind"] == "qa":
            assert r["rubric"] and r["context"]
            continue
        # expected outputs must themselves pass the schema + ops rules
        exp = dict(r["expected"])
        if r["kind"] == "preflight":
            exp.setdefault("issues", ["x"] if exp["decision"] == "no_go" else [])
        else:
            exp.update(symptom="sym", action_required="act")
        SCHEMAS[r["kind"]].model_validate(exp)


def test_field_scores_and_rubric():
    exp = {"aircraft_id": "UAV-07", "component": "propeller", "severity": "high", "grounded": True}
    pred = {**exp, "aircraft_id": "uav 07", "severity": "medium"}
    fs = field_scores("defect", pred, exp)
    assert fs == {"aircraft_id": True, "component": True, "severity": False, "grounded": True}
    assert all(v is False for v in field_scores("defect", None, exp).values())
    assert rubric_score("Q4-ALPHA is grounded; so is Mapper-5", [["Q4-ALPHA"], ["MAPPER-5"]]) == 1
    assert rubric_score("only Q4-ALPHA", [["Q4-ALPHA"], ["MAPPER-5"]]) == 0.5


def test_variance_metrics():
    assert exact_match_rate(["a", "a", "b", "a"]) == 0.75
    d = [{"x": 1, "y": 2}, {"x": 1, "y": 3}, None]
    assert field_agreement(d, ["x", "y"]) == pytest.approx((2 / 3 + 1 / 3) / 2)
    assert mean_pairwise(["abc", "abc"], text_similarity) == 1.0


def test_run_quality_end_to_end_with_fake(tmp_path, fake, make_client):
    items = [json.loads(line) for line in TESTSET.read_text().splitlines()]
    df = next(r for r in items if r["id"] == "df-01")
    qa = next(r for r in items if r["id"] == "qa-05")
    small = tmp_path / "ts.jsonl"
    small.write_text(json.dumps(df) + "\n" + json.dumps(qa) + "\n")
    fake.replies = [
        json.dumps({**df["expected"], "symptom": "chipped tip", "action_required": "replace"}),
        "Spare prop set P-14 was fitted.",
    ]
    s = run_quality(["test-model"], small, tmp_path, client=make_client())
    m = s["test-model"]
    assert m["field_accuracy"] == 1.0 and m["qa_rubric_score"] == 1.0
    assert m["schema_valid_first_try"] == 1.0
    assert (tmp_path / "quality_summary.json").exists()
