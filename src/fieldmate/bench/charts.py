"""Render PNG charts from results/*.json (matplotlib, headless)."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

COLORS = ["#2563eb", "#16a34a", "#ea580c", "#9333ea", "#0891b2", "#dc2626"]


def _load(p: Path) -> dict:
    return json.loads(p.read_text()) if p.exists() else {}


def _models(summary: dict) -> list[str]:
    return [k for k, v in summary.items() if isinstance(v, dict) and "hardware" in v]


def _bar(ax, labels, values, title, ylabel, fmt="{:.2f}", errs=None):
    bars = ax.bar(labels, values, color=COLORS[: len(labels)], yerr=errs, capsize=4)
    ax.set_title(title, fontsize=11)
    ax.set_ylabel(ylabel)
    ax.tick_params(axis="x", labelrotation=20, labelsize=8)
    for b, v in zip(bars, values, strict=True):
        ax.annotate(
            fmt.format(v),
            (b.get_x() + b.get_width() / 2, b.get_height()),
            ha="center",
            va="bottom",
            fontsize=8,
        )
    ax.grid(axis="y", alpha=0.3)


def perf_chart(results: Path, out: Path) -> Path | None:
    s = _load(results / "perf_summary.json")
    models = _models(s)
    if not models:
        return None
    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    _bar(
        axes[0],
        models,
        [s[m]["chat"]["decode_tok_s"]["mean"] for m in models],
        "Decode speed (chat)",
        "tokens / s",
    )
    _bar(
        axes[1],
        models,
        [s[m]["chat"]["ttft_s"]["p50"] for m in models],
        "Time to first token p50 (chat)",
        "seconds",
    )
    _bar(
        axes[2],
        models,
        [s[m]["extract"]["latency_s"]["p50"] for m in models],
        "End-to-end latency p50 (extract)",
        "seconds",
        fmt="{:.1f}",
    )
    _bar(
        axes[3],
        models,
        [s[m]["peak_rss_mb"] / 1024 for m in models],
        "Peak RSS (Ollama + runner)",
        "GiB",
    )
    fig.suptitle("Inference performance - 2 vCPU ARM64, no GPU", fontsize=12)
    fig.tight_layout()
    path = out / "perf.png"
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return path


def quality_chart(results: Path, out: Path, tag: str = "quality") -> Path | None:
    s = _load(results / f"{tag}_summary.json")
    if tag == "quant" and s and not any("q8_0" in k for k in s):
        # the default llama3.2:1b tag is the Q8_0 build: reuse its main-study row as the Q8_0 point
        base = _load(results / "quality_summary.json").get("llama3.2:1b")
        if base:
            s = {"llama3.2:1b (Q8_0)": base, **s}
    models = _models(s)
    if not models:
        return None
    metrics = [
        ("schema_valid_first_try", "Valid JSON+rules, 1st try"),
        ("schema_valid_after_retry", "Valid after 1 retry"),
        ("field_accuracy", "Field accuracy"),
        ("go_no_go_accuracy", "Go/No-go accuracy"),
        ("qa_rubric_score", "Q&A rubric"),
    ]
    fig, ax = plt.subplots(figsize=(11, 4.5))
    width = 0.8 / len(models)
    for i, m in enumerate(models):
        vals = [s[m].get(k) or 0 for k, _ in metrics]
        xs = [j + i * width for j in range(len(metrics))]
        bars = ax.bar(xs, vals, width, label=m, color=COLORS[i % len(COLORS)])
        for b, v in zip(bars, vals, strict=True):
            ax.annotate(
                f"{v:.2f}", (b.get_x() + b.get_width() / 2, v), ha="center", va="bottom", fontsize=7
            )
    ax.set_xticks([j + 0.4 - width / 2 for j in range(len(metrics))])
    ax.set_xticklabels([lbl for _, lbl in metrics], fontsize=9)
    ax.set_ylim(0, 1.1)
    ax.set_ylabel("score (0-1)")
    ax.legend(fontsize=8, loc="lower right")
    ax.grid(axis="y", alpha=0.3)
    ax.set_title(f"Output quality on the {s[models[0]]['n_items']}-item drone-ops test set (T=0)")
    fig.tight_layout()
    path = out / f"{tag}.png"
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return path


def tradeoff_chart(results: Path, out: Path, tag: str, title: str, fname: str) -> Path | None:
    s = _load(results / f"{tag}_summary.json")
    if tag == "quant" and s and not any("q8_0" in k for k in s):
        # the default llama3.2:1b tag is the Q8_0 build: reuse its main-study row as the Q8_0 point
        base = _load(results / "quality_summary.json").get("llama3.2:1b")
        if base:
            s = {"llama3.2:1b (Q8_0)": base, **s}
    models = _models(s)
    if not models:
        return None
    # RAM comes from the perf runs, which sample one loaded model at a time (the quality runs can
    # overlap with the previous model's runner shutting down)
    rss = {
        **_load(results / "quant_perf" / "perf_summary.json"),
        **_load(results / "perf_summary.json"),
    }
    if "llama3.2:1b" in rss:
        rss["llama3.2:1b (Q8_0)"] = rss["llama3.2:1b"]

    def ram_mb(m: str) -> float:
        return (rss.get(m) or {}).get("peak_rss_mb") or s[m].get("peak_rss_mb", 500)

    def load_of(m: str) -> str:
        mean = (s[m].get("load_avg_1m_during") or {}).get("mean")
        return "?" if mean is None else f"{mean:.1f}"

    fig, ax = plt.subplots(figsize=(7, 4.5))
    for i, m in enumerate(models):
        x = s[m]["latency_s"]["p50"]
        y = s[m].get("field_accuracy", 0)
        size = max(ram_mb(m) / 6, 40)
        ax.scatter(x, y, s=size, color=COLORS[i % len(COLORS)], alpha=0.75, label=m)
        ax.annotate(
            f"{m}\n{ram_mb(m) / 1024:.1f} GiB, host load {load_of(m)}",
            (x, y),
            textcoords="offset points",
            xytext=(8, -4),
            fontsize=8,
        )
    ax.set_xlabel("p50 latency per test item (s)  - lower is better")
    ax.set_ylabel("field accuracy - higher is better")
    ax.set_title(title, fontsize=11)
    ax.grid(alpha=0.3)
    ax.margins(x=0.35, y=0.15)
    fig.tight_layout()
    path = out / fname
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return path


def temperature_chart(results: Path, out: Path) -> Path | None:
    files = sorted(results.glob("temperature_*_summary.json"))
    if not files:
        return None
    s = _load(files[0])["summary"]
    temps = sorted(s["by_temperature"], key=float)
    keys = [
        ("exact_match_rate_extraction", "Exact match\n(extraction)"),
        ("exact_match_rate_qa", "Exact match\n(Q&A)"),
        ("field_agreement", "Field\nagreement"),
        ("schema_valid_rate", "Schema-valid\n(no retry)"),
        ("text_similarity_qa", "Text sim.\n(Q&A)"),
        ("embedding_similarity_qa", "Embedding sim.\n(Q&A)"),
    ]
    fig, ax = plt.subplots(figsize=(11, 4.2))
    width = 0.38
    for i, t in enumerate(temps):
        vals = [s["by_temperature"][t].get(k) or 0 for k, _ in keys]
        xs = [j + i * width for j in range(len(keys))]
        bars = ax.bar(xs, vals, width, label=f"T={t}", color=COLORS[i])
        for b, v in zip(bars, vals, strict=True):
            ax.annotate(
                f"{v:.2f}", (b.get_x() + b.get_width() / 2, v), ha="center", va="bottom", fontsize=7
            )
    ax.set_xticks([j + width / 2 for j in range(len(keys))])
    ax.set_xticklabels([lbl for _, lbl in keys], fontsize=8)
    ax.set_ylim(0, 1.12)
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    ax.set_title(f"Output variance, {s['model']}: {len(s['items'])} prompts x {s['n_runs']} runs")
    fig.tight_layout()
    path = out / "temperature.png"
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return path


def make_all(results: Path, out: Path) -> list[Path]:
    out.mkdir(parents=True, exist_ok=True)
    made = [
        perf_chart(results, out),
        quality_chart(results, out),
        tradeoff_chart(
            results,
            out,
            "quality",
            "Model comparison: accuracy vs latency (bubble = peak RAM)",
            "model_tradeoff.png",
        ),
        tradeoff_chart(
            results,
            out,
            "quant",
            "Quantization: accuracy vs latency (bubble = peak RAM)",
            "quant_tradeoff.png",
        ),
        temperature_chart(results, out),
    ]
    return [p for p in made if p]
