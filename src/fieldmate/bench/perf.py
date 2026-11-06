"""Phase 1: inference performance - TTFT, tokens/s, total latency, peak memory."""

from __future__ import annotations

import json
import time
from pathlib import Path

from ..client import GenerationMetrics, OllamaClient
from ..prompts import chat_messages, extraction_messages
from ..schemas import json_schema_for
from .common import MemorySampler, append_jsonl, hardware_info, safe_name, summarize, write_json

# Each warm run uses a *different* user message so llama.cpp's prompt cache can only reuse the
# shared system-prompt prefix - the same situation as a real deployment.
CHAT_QUESTIONS = [
    "In 3 bullet points, what should a pilot check on the propellers before flight?",
    "In 3 bullet points, how should a swollen LiPo battery be handled?",
    "In 3 bullet points, what causes compass errors on a quadcopter?",
    "In 3 bullet points, what should happen after an RC link failsafe?",
    "In 3 bullet points, how do you check motors before a flight?",
]
DEFECT_NOTES = [
    "UAV-07 after landing: one prop tip chipped ~2mm. no vibration noticed.",
    "AERO-2 battery pack B3 puffed up a bit after charge, smells weird",
    "Q4-ALPHA camera footage has jello, gimbal seems unbalanced",
    "MAPPER-5 lost RC link for 4 seconds at 600m, RTH failsafe kicked in",
    "UAV-15 landing gear leg loose and wobbles, one screw missing",
]


def _messages(kind: str, i: int):
    if kind == "chat":
        return chat_messages(CHAT_QUESTIONS[i % len(CHAT_QUESTIONS)])
    return extraction_messages("defect", DEFECT_NOTES[i % len(DEFECT_NOTES)])


MAX_TOKENS = 128  # cap decode length so runs are comparable across models


def _one_run(client: OllamaClient, model: str, kind: str, i: int = 0) -> GenerationMetrics:
    fmt = json_schema_for("defect") if kind == "extract" else None
    metrics = None
    for item in client.chat_stream(
        _messages(kind, i),
        model=model,
        fmt=fmt,
        temperature=0.0,
        options={"num_predict": MAX_TOKENS, "seed": 42},
    ):
        if isinstance(item, GenerationMetrics):
            metrics = item
    assert metrics is not None
    return metrics


def run_perf(models: list[str], runs: int, out_dir: Path, client: OllamaClient | None = None):
    client = client or OllamaClient()
    out_dir.mkdir(parents=True, exist_ok=True)
    summary_path = out_dir / "perf_summary.json"
    summary = json.loads(summary_path.read_text()) if summary_path.exists() else {}
    summary.update({"runs_per_prompt": runs, "max_tokens": MAX_TOKENS})
    for model in models:
        raw_path = out_dir / f"perf_{safe_name(model)}.jsonl"
        raw_path.unlink(missing_ok=True)
        hw = hardware_info()
        client.unload(model)
        time.sleep(2)
        with MemorySampler() as mem:
            # cold start: includes loading weights from disk into RAM
            cold = _one_run(client, model, "chat")
            append_jsonl(raw_path, {"phase": "cold", "prompt": "chat", **cold.as_dict()})
            rows = {"chat": [], "extract": []}
            for i in range(runs):
                for kind in ("chat", "extract"):
                    m = _one_run(client, model, kind, i)
                    rows[kind].append(m)
                    append_jsonl(
                        raw_path, {"phase": "warm", "prompt": kind, "run": i, **m.as_dict()}
                    )
                    print(
                        f"[perf] {model} {kind} run {i}: {m.tokens_per_s:.2f} tok/s "
                        f"ttft {m.ttft_s:.2f}s wall {m.wall_s:.2f}s",
                        flush=True,
                    )
        model_sum = {
            "cold_load_s": round(cold.load_s, 3),
            "cold_ttft_s": round(cold.ttft_s or 0, 3),
            "peak_rss_mb": round(mem.peak_mb, 1),
        }
        for kind, ms in rows.items():
            model_sum[kind] = {
                "decode_tok_s": summarize([m.tokens_per_s for m in ms]),
                "prefill_tok_s": summarize([m.prompt_tokens_per_s for m in ms if m.prompt_eval_s]),
                "ttft_s": summarize([m.ttft_s for m in ms]),
                "latency_s": summarize([m.wall_s for m in ms]),
                "output_tokens": summarize([m.output_tokens for m in ms]),
                "prompt_tokens": summarize([m.prompt_tokens for m in ms]),
            }
        model_sum["hardware"] = hw
        summary[model] = model_sum
        write_json(summary_path, summary)
    return summary
