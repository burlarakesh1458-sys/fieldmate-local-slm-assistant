"""Automatic quality scoring against the expected outputs in data/testset.jsonl."""

from __future__ import annotations

import difflib
import json
import math
from collections import Counter
from typing import Any

from ..schemas import normalise_aircraft_id

SCORED_FIELDS = {
    "preflight": [
        "aircraft_id",
        "battery_pct",
        "propellers",
        "motors",
        "gps",
        "compass",
        "weather_ok",
        "decision",
    ],
    "defect": ["aircraft_id", "component", "severity", "grounded"],
}


def _norm(field: str, value: Any) -> Any:
    if field == "aircraft_id" and isinstance(value, str):
        return normalise_aircraft_id(value)
    return value


def field_scores(kind: str, predicted: dict | None, expected: dict) -> dict[str, bool]:
    """Per-field correctness. A failed extraction scores every field as wrong."""
    out = {}
    for f in SCORED_FIELDS[kind]:
        if predicted is None:
            out[f] = False
        else:
            out[f] = _norm(f, predicted.get(f)) == _norm(f, expected.get(f))
    return out


def rubric_score(answer: str, rubric: list[list[str]]) -> float:
    """Fraction of rubric groups satisfied; a group is satisfied if any synonym appears."""
    text = answer.lower()
    hits = sum(any(alt.lower() in text for alt in group) for group in rubric)
    return hits / len(rubric) if rubric else 0.0


# ----------------------------------------------------------- variance metrics (temperature)
def canonical(obj: Any) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"))


def exact_match_rate(outputs: list[str]) -> float:
    """Share of runs identical to the most common output (1.0 = fully deterministic)."""
    if not outputs:
        return float("nan")
    return Counter(outputs).most_common(1)[0][1] / len(outputs)


def field_agreement(dicts: list[dict | None], fields: list[str]) -> float:
    """Mean over fields of the share of runs agreeing with the modal value of that field."""
    valid = [d for d in dicts if d is not None]
    if not valid:
        return 0.0
    scores = []
    for f in fields:
        vals = [canonical(d.get(f)) for d in valid]
        scores.append(Counter(vals).most_common(1)[0][1] / len(dicts))
    return sum(scores) / len(scores)


def mean_pairwise(items: list, sim) -> float:
    pairs = [(a, b) for i, a in enumerate(items) for b in items[i + 1 :]]
    if not pairs:
        return 1.0
    return sum(sim(a, b) for a, b in pairs) / len(pairs)


def text_similarity(a: str, b: str) -> float:
    return difflib.SequenceMatcher(None, a, b).ratio()


def cosine(u: list[float], v: list[float]) -> float:
    dot = sum(x * y for x, y in zip(u, v, strict=True))
    nu = math.sqrt(sum(x * x for x in u))
    nv = math.sqrt(sum(y * y for y in v))
    return dot / (nu * nv) if nu and nv else 0.0
