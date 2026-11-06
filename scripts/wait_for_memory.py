"""Block until the host has enough free RAM to load a model (shared/edge boxes).

usage: python scripts/wait_for_memory.py <needed_GiB> [timeout_minutes]
Requires 3 consecutive readings (10 s apart) above the threshold, because on a shared
machine free memory fluctuates and a single good reading led to OOM-killed runners.
Exit code 0 when available, 1 on timeout.

Before waiting it asks Ollama to unload whatever model is still resident: with keep_alive=30m
the previous model's runner (3.3 GB for phi4-mini) otherwise holds the very RAM we wait for.
Set FIELDMATE_GATE_NO_UNLOAD=1 to skip that.
"""

import json
import os
import sys
import time
import urllib.request

import psutil


def unload_resident() -> None:
    host = os.environ.get("OLLAMA_HOST", "http://127.0.0.1:11434")
    if not host.startswith("http"):
        host = "http://" + host
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))  # never proxy local
    try:
        with opener.open(f"{host}/api/ps", timeout=10) as r:
            models = [m["name"] for m in json.load(r).get("models", [])]
        for m in models:
            body = json.dumps({"model": m, "keep_alive": 0}).encode()
            req = urllib.request.Request(
                f"{host}/api/generate", body, {"content-type": "application/json"}
            )
            opener.open(req, timeout=60).read()
            print(f"[mem] unloaded resident model {m}", flush=True)
        if models:
            time.sleep(3)
    except OSError as exc:  # Ollama down: nothing resident to free
        print(f"[mem] could not query Ollama ({exc}); not unloading", flush=True)


if not os.environ.get("FIELDMATE_GATE_NO_UNLOAD"):
    unload_resident()

need = float(sys.argv[1]) * 1024**3
timeout = float(sys.argv[2]) * 60 if len(sys.argv) > 2 else 3600
t0 = time.time()
streak = 0
while True:
    avail = psutil.virtual_memory().available
    streak = streak + 1 if avail >= need else 0
    if streak >= 3:
        print(
            f"[mem] available {avail / 1024**3:.2f} GiB >= {need / 1024**3:.2f} GiB, go", flush=True
        )
        sys.exit(0)
    if time.time() - t0 > timeout:
        print(f"[mem] timeout: only {avail / 1024**3:.2f} GiB available", flush=True)
        sys.exit(1)
    time.sleep(10)
