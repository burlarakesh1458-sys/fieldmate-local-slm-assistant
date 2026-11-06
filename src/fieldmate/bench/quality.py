"""Phase 3: model comparison on the standardized test set (data/testset.jsonl).

Per item we record validity (first try / after retry), field accuracy or rubric score, latency
and throughput. Results are appended to JSONL as we go so a long run can be resumed.
"""

from __future__ import annotations

import json
import os
import statistics
from pathlib import Path

from ..client import OllamaClient, OllamaError
from ..extractor import Extractor, total_metrics
from ..prompts import chat_messages
from .common import (
    MemorySampler,
    append_jsonl,
    hardware_info,
    read_jsonl,
    safe_name,
    summarize,
    write_json,
)
from .scoring import field_scores, rubric_score


def load_testset(path: Path, limit: int | None = None, kinds: list[str] | None = None):
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    if kinds:
        rows = [r for r in rows if r["kind"] in kinds]
    return rows[:limit] if limit else rows


def run_item(client: OllamaClient, extractor: Extractor, model: str, item: dict, constrained: bool):
    if item["kind"] == "qa":
        try:
            res = client.chat(
                chat_messages(item["input"], item.get("context")),
                model=model,
                temperature=0.0,
                options={"num_predict": 200, "seed": 42},
            )
        except OllamaError as exc:
            return {
                "id": item["id"],
                "kind": "qa",
                "infra_error": f"ollama error: {exc}",
                "answer": "",
                "rubric_score": 0.0,
                "latency_s": 0.0,
                "tokens_per_s": 0.0,
                "output_tokens": 0,
                "prompt_tokens": 0,
            }
        m = res.metrics
        return {
            "id": item["id"],
            "kind": "qa",
            "answer": res.content,
            "rubric_score": rubric_score(res.content, item["rubric"]),
            "latency_s": m.wall_s,
            "tokens_per_s": m.tokens_per_s,
            "output_tokens": m.output_tokens,
            "prompt_tokens": m.prompt_tokens,
        }
    result = extractor.extract(
        item["kind"],
        item["input"],
        model=model,
        temperature=0.0,
        constrained=constrained,
        options={"num_predict": 256, "seed": 42},
    )
    fs = field_scores(item["kind"], result.data, item["expected"])
    m = total_metrics(result)
    return {
        "id": item["id"],
        "kind": item["kind"],
        "ok": result.ok,
        "first_try_valid": result.first_try_valid,
        "attempts": result.n_attempts,
        "errors": [a.error for a in result.attempts if a.error],
        "infra_error": result.error if result.error and "ollama error" in result.error else None,
        "data": result.data,
        "field_scores": fs,
        "field_accuracy": sum(fs.values()) / len(fs),
        "latency_s": m.wall_s,
        "tokens_per_s": m.tokens_per_s,
        "output_tokens": m.output_tokens,
        "prompt_tokens": m.prompt_tokens,
    }


def summarize_rows(rows: list[dict]) -> dict:
    ext = [r for r in rows if r["kind"] in ("preflight", "defect")]
    qa = [r for r in rows if r["kind"] == "qa"]
    out: dict = {"n_items": len(rows), "n_extraction": len(ext), "n_qa": len(qa)}
    if ext:
        out["schema_valid_first_try"] = round(sum(r["first_try_valid"] for r in ext) / len(ext), 4)
        out["schema_valid_after_retry"] = round(sum(r["ok"] for r in ext) / len(ext), 4)
        out["retry_rate"] = round(sum(r["attempts"] > 1 for r in ext) / len(ext), 4)
        out["field_accuracy"] = round(statistics.fmean(r["field_accuracy"] for r in ext), 4)
        for kind in ("preflight", "defect"):
            sub = [r for r in ext if r["kind"] == kind]
            if sub:
                out[f"field_accuracy_{kind}"] = round(
                    statistics.fmean(r["field_accuracy"] for r in sub), 4
                )
                per_field: dict[str, list[bool]] = {}
                for r in sub:
                    for f, v in r["field_scores"].items():
                        per_field.setdefault(f, []).append(v)
                out[f"per_field_{kind}"] = {
                    f: round(sum(v) / len(v), 4) for f, v in per_field.items()
                }
        dec = [r for r in ext if r["kind"] == "preflight"]
        if dec:
            out["go_no_go_accuracy"] = round(
                sum(r["field_scores"]["decision"] for r in dec) / len(dec), 4
            )
    if qa:
        out["qa_rubric_score"] = round(statistics.fmean(r["rubric_score"] for r in qa), 4)
    out["latency_s"] = summarize([r["latency_s"] for r in rows])
    out["tokens_per_s"] = summarize([r["tokens_per_s"] for r in rows if r["tokens_per_s"]])
    out["infra_errors"] = sum(1 for r in rows if r.get("infra_error"))
    out["load_avg_1m_during"] = summarize([r["load_avg_1m"] for r in rows if r.get("load_avg_1m")])
    return out


def run_quality(
    models: list[str],
    testset: Path,
    out_dir: Path,
    limit: int | None = None,
    constrained: bool = True,
    tag: str = "quality",
    client: OllamaClient | None = None,
    max_infra_errors: int = 2,
):
    client = client or OllamaClient()
    extractor = Extractor(client)
    items = load_testset(testset, limit)
    summary_path = out_dir / f"{tag}_summary.json"
    summary = json.loads(summary_path.read_text()) if summary_path.exists() else {}
    for model in models:
        raw = out_dir / f"{tag}_{safe_name(model)}.jsonl"
        done = {r["id"] for r in read_jsonl(raw) if not r.get("infra_error")}
        hw = hardware_info()
        consecutive_infra = 0
        with MemorySampler() as mem:
            for item in items:
                if item["id"] in done:
                    continue
                row = run_item(client, extractor, model, item, constrained)
                row["load_avg_1m"] = round(os.getloadavg()[0], 2)
                append_jsonl(raw, row)
                # circuit breaker: e.g. runner OOM-killed on a shared box -> stop, resume later
                consecutive_infra = consecutive_infra + 1 if row.get("infra_error") else 0
                if consecutive_infra >= max_infra_errors:
                    raise RuntimeError(
                        f"{model}: {consecutive_infra} consecutive Ollama errors "
                        f"({row['infra_error']}); aborting - rerun to resume"
                    )
                score = row.get("field_accuracy", row.get("rubric_score"))
                print(
                    f"[{tag}] {model} {item['id']} score={score:.2f} "
                    f"ok={row.get('ok', '-')} lat={row['latency_s']:.1f}s",
                    flush=True,
                )
        rows = [r for r in read_jsonl(raw) if not r.get("infra_error")]
        s = summarize_rows(rows)
        prev_peak = summary.get(model, {}).get("peak_rss_mb", 0.0)
        s["peak_rss_mb"] = round(max(mem.peak_mb, prev_peak), 1)
        s["constrained_decoding"] = constrained
        s["hardware"] = hw
        summary[model] = s
        write_json(summary_path, summary)
    return summary
