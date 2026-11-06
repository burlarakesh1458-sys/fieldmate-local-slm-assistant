"""Generate markdown result tables straight from results/*.json (no hand-copied numbers).

usage: python scripts/make_tables.py > docs/RESULTS_TABLES.md
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

RES = Path(__file__).resolve().parents[1] / "results"


def load(name: str) -> dict:
    p = RES / name
    return json.loads(p.read_text()) if p.exists() else {}


def models_of(s: dict) -> list[str]:
    return [k for k, v in s.items() if isinstance(v, dict) and "hardware" in v]


Q8_BASE = "llama3.2:1b"  # the default llama3.2:1b tag is the Q8_0 build (same digest)


def with_base(s: dict, base: dict, label: str) -> dict:
    """Prepend the main-study llama3.2:1b row (Q8_0 / constrained) as the reference point."""
    if not s or Q8_BASE not in base or any("q8_0" in k for k in s):
        return s
    return {label: base[Q8_BASE], **s}


def pct(x) -> str:
    return "-" if x is None else f"{100 * x:.1f}%"


def f(x, nd=2) -> str:
    return "-" if x is None else f"{x:.{nd}f}"


def perf_table(s: dict, title: str) -> str:
    ms = models_of(s)
    if not ms:
        return ""
    out = [
        f"### {title}",
        "",
        f"Warm runs per prompt type: {s.get('runs_per_prompt')}, decode capped at "
        f"{s.get('max_tokens')} tokens, T=0.",
        "",
        "| Model | Decode tok/s (mean) | Prefill tok/s (mean) | TTFT p50 / p95 (s) "
        "| Latency chat p50 / p95 (s) | Latency extract p50 / p95 (s) | Cold load (s) "
        "| Peak RSS (GiB) | Load avg at start |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for m in ms:
        d = s[m]
        c, e = d["chat"], d["extract"]
        out.append(
            f"| `{m}` | {f(c['decode_tok_s']['mean'])} | {f(e['prefill_tok_s'].get('mean'), 1)} "
            f"| {f(c['ttft_s']['p50'])} / {f(c['ttft_s']['p95'])} "
            f"| {f(c['latency_s']['p50'], 1)} / {f(c['latency_s']['p95'], 1)} "
            f"| {f(e['latency_s']['p50'], 1)} / {f(e['latency_s']['p95'], 1)} "
            f"| {f(d['cold_load_s'], 1)} | {f(d['peak_rss_mb'] / 1024)} "
            f"| {d['hardware']['load_avg_1m_at_start']} |"
        )
    return "\n".join(out) + "\n"


def quality_table(s: dict, title: str) -> str:
    ms = models_of(s)
    if not ms:
        return ""
    out = [
        f"### {title}",
        "",
        "| Model | Items | Valid 1st try | Valid after retry | Retry rate | Field acc. "
        "| Preflight acc. | Defect acc. | Go/No-go acc. | Q&A rubric | p50 latency/item (s) "
        "| Load avg (mean) |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for m in ms:
        d = s[m]
        out.append(
            f"| `{m}` | {d['n_items']} | {pct(d.get('schema_valid_first_try'))} "
            f"| {pct(d.get('schema_valid_after_retry'))} | {pct(d.get('retry_rate'))} "
            f"| {pct(d.get('field_accuracy'))} | {pct(d.get('field_accuracy_preflight'))} "
            f"| {pct(d.get('field_accuracy_defect'))} | {pct(d.get('go_no_go_accuracy'))} "
            f"| {pct(d.get('qa_rubric_score'))} | {f(d['latency_s'].get('p50'), 1)} "
            f"| {f((d.get('load_avg_1m_during') or {}).get('mean'), 1)} |"
        )
    return "\n".join(out) + "\n"


def per_field_table(s: dict) -> str:
    ms = models_of(s)
    if not ms:
        return ""
    out = []
    for kind in ("preflight", "defect"):
        fields = list(s[ms[0]].get(f"per_field_{kind}", {}))
        if not fields:
            continue
        out += [
            f"### Per-field accuracy - {kind}",
            "",
            "| Model | " + " | ".join(fields) + " |",
            "|---|" + "---|" * len(fields),
        ]
        for m in ms:
            pf = s[m].get(f"per_field_{kind}", {})
            out.append(f"| `{m}` | " + " | ".join(pct(pf.get(x)) for x in fields) + " |")
        out.append("")
    return "\n".join(out)


def temperature_table() -> str:
    files = sorted(RES.glob("temperature_*_summary.json"))
    if not files:
        return ""
    s = json.loads(files[0].read_text())["summary"]
    temps = sorted(s["by_temperature"], key=float)
    keys = list(s["by_temperature"][temps[0]])
    out = [
        f"### Temperature study - `{s['model']}` ({len(s['items'])} prompts x {s['n_runs']} runs "
        "per temperature, no retry)",
        "",
        "| Metric | " + " | ".join(f"T={t}" for t in temps) + " |",
        "|---|" + "---|" * len(temps),
    ]
    for k in keys:
        out.append(
            f"| {k} | " + " | ".join(f(s["by_temperature"][t].get(k), 3) for t in temps) + " |"
        )
    return "\n".join(out) + "\n"


Q8 = "llama3.2:1b (Q8_0)"


def quant_perf_table() -> str:
    qp = load("quant_perf/perf_summary.json")
    if any("q8_0" in k for k in models_of(qp)):  # Q8_0 re-measured back to back with Q4/Q5
        return perf_table(qp, "Quantization study - speed (all three back to back)")
    return perf_table(
        with_base(qp, load("perf_summary.json"), Q8),
        "Quantization study - speed (Q8_0 row from the main perf run)",
    )


def main() -> None:
    parts = [
        "<!-- generated by scripts/make_tables.py from results/*.json - do not edit by hand -->",
        "",
        perf_table(load("perf_summary.json"), "Inference performance (Phase 1)"),
        quality_table(load("quality_summary.json"), "Model comparison - quality (Phase 3)"),
        per_field_table(load("quality_summary.json")),
        temperature_table(),
        quality_table(
            with_base(
                load("ablation_unconstrained_summary.json"),
                load("quality_summary.json"),
                "llama3.2:1b (schema-constrained, main study)",
            ),
            "Ablation - format='json' without schema (unconstrained) vs schema-constrained",
        ),
        quality_table(
            with_base(load("quant_summary.json"), load("quality_summary.json"), Q8),
            "Quantization study - quality",
        ),
        perf_table(
            with_base(load("quant_perf/perf_summary.json"), load("perf_summary.json"), Q8),
            "Quantization study - speed (Q8_0 row from the main perf run)",
        ),
    ]
    sys.stdout.write("\n".join(p for p in parts if p))


if __name__ == "__main__":
    main()
