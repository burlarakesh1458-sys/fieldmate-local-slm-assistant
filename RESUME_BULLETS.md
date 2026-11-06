# Resume bullets: FieldMate (offline drone-ops assistant on small language models)

All numbers are measured and traceable to `results/` (2-vCPU ARM64 CPU, no GPU, Ollama 0.40.15).

- Built **FieldMate**, an offline assistant (Python, FastAPI, Typer, Ollama, Pydantic v2) that
  turns drone pilots' field notes into validated pre-flight and defect JSON using schema-constrained
  decoding, ops-manual validators and one error-feedback retry, reaching **100% schema-valid output
  on 144/144 extractions** across 4 small models on a 2-vCPU CPU-only box.
- Benchmarked **4 SLMs (1B to 3.8B)** on a hand-labelled 42-item drone-ops test set; selected
  **qwen2.5:3b** as default at **85.8% field accuracy and 95% go/no-go accuracy** (vs 61.8% / 80%
  for llama3.2:1b) with **2.3 GiB RAM and 12.6 tok/s** decode on CPU.
- Showed by ablation that **grammar-constrained decoding raised valid output from 69.4% to 100%**
  and **cut median latency from 7.8 s to 3.7 s per item** (fewer retries) on llama3.2:1b, and found that it also
  pushes tiny models to emit `null` for numeric fields (battery null on 20/20 notes), a safety issue
  hidden by headline validity.
- Measured quantization and sampling trade-offs: **Q4_K_M cut peak RAM 31% (1.72 to 1.19 GiB)**
  for a **1.7-point** extraction-accuracy loss; outputs matched across repeats in **31/32 runs at T=0 vs 15/32 at T=0.7**;
  made the suite RAM-gated and resumable so it survived OOM kills and a host reboot on a shared box.
