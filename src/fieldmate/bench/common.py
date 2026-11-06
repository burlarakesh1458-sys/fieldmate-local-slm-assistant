"""Shared benchmark utilities: stats, memory sampling, hardware fingerprint, I/O."""

from __future__ import annotations

import json
import os
import platform
import statistics
import threading
import time
from pathlib import Path
from typing import Any

import psutil

OLLAMA_PROC_MARKERS = ("ollama", "llama-server")


def percentile(values: list[float], p: float) -> float:
    """Linear-interpolated percentile (numpy 'linear' method) without numpy."""
    if not values:
        return float("nan")
    xs = sorted(values)
    k = (len(xs) - 1) * p / 100.0
    lo, hi = int(k), min(int(k) + 1, len(xs) - 1)
    return xs[lo] + (xs[hi] - xs[lo]) * (k - lo)


def summarize(values: list[float]) -> dict[str, float]:
    vals = [v for v in values if v is not None]
    if not vals:
        return {"n": 0}
    return {
        "n": len(vals),
        "mean": round(statistics.fmean(vals), 4),
        "p50": round(percentile(vals, 50), 4),
        "p95": round(percentile(vals, 95), 4),
        "min": round(min(vals), 4),
        "max": round(max(vals), 4),
        "stdev": round(statistics.stdev(vals), 4) if len(vals) > 1 else 0.0,
    }


def ollama_processes() -> list[psutil.Process]:
    procs = []
    for p in psutil.process_iter(["name", "cmdline"]):
        try:
            blob = " ".join([p.info.get("name") or "", *(p.info.get("cmdline") or [])])
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue
        if any(m in blob for m in OLLAMA_PROC_MARKERS) and "fieldmate" not in blob:
            procs.append(p)
    return procs


def ollama_rss_mb() -> float:
    total = 0
    for p in ollama_processes():
        try:
            total += p.memory_info().rss
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue
    return total / (1024 * 1024)


class MemorySampler:
    """Background thread that tracks peak RSS of the Ollama server + model runner processes.

    Ollama (CPU backend) loads weights into anonymous memory (mmap disabled), so RSS of the
    llama-server child is a fair proxy for the model's resident footprint.
    """

    def __init__(self, interval_s: float = 0.25) -> None:
        self.interval_s = interval_s
        self.peak_mb = 0.0
        self.samples: list[float] = []
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)

    def _run(self) -> None:
        while not self._stop.is_set():
            mb = ollama_rss_mb()
            self.samples.append(mb)
            self.peak_mb = max(self.peak_mb, mb)
            self._stop.wait(self.interval_s)

    def __enter__(self) -> MemorySampler:
        self._thread.start()
        return self

    def __exit__(self, *exc) -> None:
        self._stop.set()
        self._thread.join(timeout=2)


def hardware_info() -> dict[str, Any]:
    vm = psutil.virtual_memory()
    return {
        "platform": platform.platform(),
        "machine": platform.machine(),
        "python": platform.python_version(),
        "cpu_count_logical": psutil.cpu_count(logical=True),
        "ram_total_gb": round(vm.total / 1e9, 2),
        "ram_available_gb_at_start": round(vm.available / 1e9, 2),
        "load_avg_1m_at_start": round(os.getloadavg()[0], 2),
        "gpu": "none",
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }


def append_jsonl(path: Path, row: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as f:
        f.write(json.dumps(row) + "\n")


def read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2) + "\n")


def safe_name(model: str) -> str:
    return model.replace(":", "_").replace("/", "_")
