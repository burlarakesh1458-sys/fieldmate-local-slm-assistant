"""Block until the host has enough free RAM to load a model (shared/edge boxes).

usage: python scripts/wait_for_memory.py <needed_GiB> [timeout_minutes]
Requires 3 consecutive readings (10 s apart) above the threshold, because on a shared
machine free memory fluctuates and a single good reading led to OOM-killed runners.
Exit code 0 when available, 1 on timeout.
"""

import sys
import time

import psutil

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
