"""`fieldmate` command-line interface (Typer)."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import typer

from .client import GenerationMetrics, OllamaClient, OllamaError
from .extractor import Extractor
from .prompts import chat_messages

app = typer.Typer(
    add_completion=False, help="FieldMate - offline drone-ops assistant on local SLMs."
)
bench_app = typer.Typer(help="Benchmarks (perf, quality, temperature, charts).")
app.add_typer(bench_app, name="bench")

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_TESTSET = ROOT / "data" / "testset.jsonl"
DEFAULT_RESULTS = ROOT / "results"


def _read_text(text: str | None, file: Path | None) -> str:
    if file:
        return file.read_text()
    if text:
        return text
    if not sys.stdin.isatty():
        return sys.stdin.read()
    raise typer.BadParameter("provide TEXT, --file, or pipe text on stdin")


@app.command()
def health() -> None:
    """Check that Ollama is reachable and list local models."""
    client = OllamaClient()
    try:
        typer.echo(f"ollama {client.version()} at {client.settings.ollama_host}")
        for m in client.list_models():
            mark = "*" if m == client.settings.model else " "
            typer.echo(f" {mark} {m}")
    except OllamaError as exc:
        typer.secho(str(exc), fg="red", err=True)
        raise typer.Exit(1) from exc


@app.command()
def chat(
    question: str = typer.Argument(..., help="Question for the assistant"),
    context_file: Path | None = typer.Option(
        None, "--context", "-c", help="Fleet log to ground on"
    ),
    model: str | None = typer.Option(None, "--model", "-m"),
    temperature: float | None = typer.Option(None, "--temperature", "-t"),
    stats: bool = typer.Option(False, "--stats", help="Print TTFT and tokens/s afterwards"),
) -> None:
    """Ask a question; the answer streams token by token."""
    client = OllamaClient()
    ctx = context_file.read_text() if context_file else None
    try:
        for item in client.chat_stream(
            chat_messages(question, ctx), model=model, temperature=temperature
        ):
            if isinstance(item, GenerationMetrics):
                typer.echo()
                if stats:
                    typer.secho(
                        f"[ttft {item.ttft_s:.2f}s | {item.tokens_per_s:.1f} tok/s | "
                        f"{item.output_tokens} tokens | {item.wall_s:.1f}s]",
                        fg="cyan",
                    )
            else:
                typer.echo(item, nl=False)
    except OllamaError as exc:
        typer.secho(f"\n{exc}", fg="red", err=True)
        raise typer.Exit(1) from exc


@app.command()
def extract(
    kind: str = typer.Argument(..., help="preflight | defect"),
    text: str | None = typer.Argument(None, help="Note text (or use --file / stdin)"),
    file: Path | None = typer.Option(None, "--file", "-f"),
    model: str | None = typer.Option(None, "--model", "-m"),
) -> None:
    """Turn a messy pilot / maintenance note into validated JSON."""
    result = Extractor(OllamaClient()).extract(kind, _read_text(text, file), model=model)
    typer.echo(json.dumps(result.data if result.ok else result.model_dump(), indent=2))
    if not result.ok:
        typer.secho(f"extraction failed: {result.error}", fg="red", err=True)
        raise typer.Exit(2)
    if result.n_attempts > 1:
        typer.secho("(recovered on retry)", fg="yellow", err=True)


@app.command()
def serve(host: str = "127.0.0.1", port: int = 8000) -> None:
    """Run the FastAPI server."""
    import uvicorn

    uvicorn.run("fieldmate.api:app", host=host, port=port)


@bench_app.command("perf")
def bench_perf(
    models: list[str] = typer.Option(..., "--model", "-m"),
    runs: int = typer.Option(5, help="Warm runs per prompt type"),
    out: Path = typer.Option(DEFAULT_RESULTS, "--out"),
) -> None:
    """TTFT, tokens/s, latency (mean/p50/p95) and peak RSS per model."""
    from .bench.perf import run_perf

    s = run_perf(models, runs, out)
    typer.echo(json.dumps({m: s[m] for m in models}, indent=2))


@bench_app.command("quality")
def bench_quality(
    models: list[str] = typer.Option(..., "--model", "-m"),
    testset: Path = typer.Option(DEFAULT_TESTSET),
    out: Path = typer.Option(DEFAULT_RESULTS, "--out"),
    limit: int | None = typer.Option(None, help="Only the first N test items"),
    unconstrained: bool = typer.Option(False, help="Ablation: format='json' without schema"),
    tag: str = typer.Option("quality", help="Result file prefix"),
) -> None:
    """Score models on the standardized test set (validity, field accuracy, rubric)."""
    from .bench.quality import run_quality

    s = run_quality(models, testset, out, limit, constrained=not unconstrained, tag=tag)
    typer.echo(
        json.dumps(
            {m: {k: v for k, v in s[m].items() if k != "hardware"} for m in models}, indent=2
        )
    )


@bench_app.command("temperature")
def bench_temperature(
    model: str = typer.Option("llama3.2:3b", "--model", "-m"),
    ids: str = typer.Option(
        "pf-03,pf-05,pf-09,pf-13,pf-18,df-01,df-03,df-08,df-11,df-12,qa-01,qa-02,qa-06",
        help="Comma-separated test item ids",
    ),
    runs: int = typer.Option(5),
    testset: Path = typer.Option(DEFAULT_TESTSET),
    out: Path = typer.Option(DEFAULT_RESULTS, "--out"),
    embed_model: str = typer.Option("all-minilm"),
) -> None:
    """Run the same prompts at T=0 and T=0.7 and measure output variance."""
    from .bench.temperature import run_temperature

    run_temperature(model, testset, out, ids.split(","), n_runs=runs, embed_model=embed_model)


@bench_app.command("charts")
def bench_charts(
    results: Path = typer.Option(DEFAULT_RESULTS, "--results"),
    out: Path = typer.Option(ROOT / "docs" / "charts", "--out"),
) -> None:
    """Render matplotlib charts from results/*.json."""
    from .bench.charts import make_all

    for p in make_all(results, out):
        typer.echo(p)


if __name__ == "__main__":  # pragma: no cover
    app()
