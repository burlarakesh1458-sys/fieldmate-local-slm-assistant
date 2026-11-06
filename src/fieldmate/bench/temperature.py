"""Phase 2b: temperature study - same prompts at T=0 vs T=0.7, N runs each, measure variance."""

from __future__ import annotations

import json
import statistics
from pathlib import Path

from ..client import OllamaClient, OllamaError
from ..extractor import Extractor
from ..prompts import chat_messages
from .common import append_jsonl, read_jsonl, safe_name, write_json
from .quality import load_testset
from .scoring import (
    SCORED_FIELDS,
    canonical,
    cosine,
    exact_match_rate,
    field_agreement,
    field_scores,
    mean_pairwise,
    text_similarity,
)


def run_temperature(
    model: str,
    testset: Path,
    out_dir: Path,
    ids: list[str],
    temps: tuple[float, ...] = (0.0, 0.7),
    n_runs: int = 5,
    embed_model: str | None = "all-minilm",
    client: OllamaClient | None = None,
):
    client = client or OllamaClient()
    extractor = Extractor(client, max_retries=0)  # measure the raw model, not the retry loop
    items = {r["id"]: r for r in load_testset(testset)}
    raw = out_dir / f"temperature_{safe_name(model)}.jsonl"
    done = {(r["id"], r["temperature"], r["run"]) for r in read_jsonl(raw)}
    for t in temps:
        for run in range(n_runs):
            for item_id in ids:
                if (item_id, t, run) in done:
                    continue
                item = items[item_id]
                # no fixed seed: we want to observe sampling randomness at T>0
                opts = {"num_predict": 256}
                if item["kind"] == "qa":
                    res = client.chat(
                        chat_messages(item["input"], item.get("context")),
                        model=model,
                        temperature=t,
                        options={"num_predict": 160},
                    )
                    row = {"output": res.content, "valid": True, "data": None}
                else:
                    r = extractor.extract(
                        item["kind"], item["input"], model=model, temperature=t, options=opts
                    )
                    row = {
                        "output": r.attempts[0].raw if r.attempts else "",
                        "valid": r.ok,
                        "data": r.data,
                    }
                append_jsonl(
                    raw, {"id": item_id, "kind": item["kind"], "temperature": t, "run": run, **row}
                )
                print(f"[temp] T={t} run={run} {item_id} valid={row['valid']}", flush=True)
    return analyse_temperature(model, testset, out_dir, embed_model, client)


def analyse_temperature(model, testset, out_dir, embed_model="all-minilm", client=None):
    items = {r["id"]: r for r in load_testset(testset)}
    rows = read_jsonl(out_dir / f"temperature_{safe_name(model)}.jsonl")
    groups: dict[tuple[str, float], list[dict]] = {}
    for r in rows:
        groups.setdefault((r["id"], r["temperature"]), []).append(r)

    per_item = []
    for (item_id, t), rs in sorted(groups.items()):
        item = items[item_id]
        outs = [r["output"] for r in rs]
        rec = {"id": item_id, "kind": item["kind"], "temperature": t, "n": len(rs)}
        if item["kind"] == "qa":
            rec["exact_match_rate"] = exact_match_rate(outs)
        else:
            datas = [r["data"] for r in rs]
            rec["exact_match_rate"] = exact_match_rate(
                [canonical(d) if d else f"INVALID:{o}" for d, o in zip(datas, outs, strict=True)]
            )
            rec["schema_valid_rate"] = sum(r["valid"] for r in rs) / len(rs)
            rec["field_agreement"] = field_agreement(datas, SCORED_FIELDS[item["kind"]])
            accs = [
                sum(field_scores(item["kind"], d, item["expected"]).values())
                / len(SCORED_FIELDS[item["kind"]])
                for d in datas
            ]
            rec["field_accuracy_mean"] = statistics.fmean(accs)
            rec["field_accuracy_stdev"] = statistics.pstdev(accs)
        rec["text_similarity"] = mean_pairwise(outs, text_similarity)
        if embed_model and client is not None:
            try:
                vecs = client.embed(outs, model=embed_model)
                rec["embedding_similarity"] = mean_pairwise(vecs, cosine)
            except OllamaError as exc:  # embedding model optional
                rec["embedding_similarity"] = None
                rec["embedding_error"] = str(exc)
        per_item.append(rec)

    def agg(t, key, kinds=("preflight", "defect", "qa")):
        vals = [
            r[key]
            for r in per_item
            if r["temperature"] == t and r["kind"] in kinds and r.get(key) is not None
        ]
        return round(statistics.fmean(vals), 4) if vals else None

    temps = sorted({t for _, t in groups})
    summary = {
        "model": model,
        "n_runs": max((r["n"] for r in per_item), default=0),
        "items": sorted({r["id"] for r in per_item}),
        "by_temperature": {},
    }
    ext = ("preflight", "defect")
    for t in temps:
        summary["by_temperature"][str(t)] = {
            "exact_match_rate_extraction": agg(t, "exact_match_rate", ext),
            "exact_match_rate_qa": agg(t, "exact_match_rate", ("qa",)),
            "schema_valid_rate": agg(t, "schema_valid_rate", ext),
            "field_agreement": agg(t, "field_agreement", ext),
            "field_accuracy_mean": agg(t, "field_accuracy_mean", ext),
            "field_accuracy_stdev": agg(t, "field_accuracy_stdev", ext),
            "text_similarity_extraction": agg(t, "text_similarity", ext),
            "text_similarity_qa": agg(t, "text_similarity", ("qa",)),
            "embedding_similarity_extraction": agg(t, "embedding_similarity", ext),
            "embedding_similarity_qa": agg(t, "embedding_similarity", ("qa",)),
        }
    write_json(
        out_dir / f"temperature_{safe_name(model)}_summary.json",
        {"summary": summary, "per_item": per_item},
    )
    print(json.dumps(summary, indent=2))
    return summary
