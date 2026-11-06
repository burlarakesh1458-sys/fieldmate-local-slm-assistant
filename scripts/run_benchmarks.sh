#!/usr/bin/env bash
# Runs the full benchmark suite sequentially, gating each model on free RAM so a shared /
# edge machine does not OOM-kill the runner. Results land in results/, charts in docs/charts/.
#
#   PHASES="perf quality temp ablation quant charts" scripts/run_benchmarks.sh
set -uo pipefail
cd "$(dirname "$0")/.."
PY=${PY:-python}
FM=${FM:-fieldmate}
PHASES=${PHASES:-"perf quality temp ablation quant charts"}
PERF_RUNS=${PERF_RUNS:-5}
MEM_WAIT_MIN=${MEM_WAIT_MIN:-240}
TEMP_MODEL=${TEMP_MODEL:-llama3.2:3b}
TEMP_RUNS=${TEMP_RUNS:-5}
ABLATION_MODELS=${ABLATION_MODELS:-"llama3.2:1b"}
QUANT_MODELS=${QUANT_MODELS:-"llama3.2:1b-instruct-q4_K_M llama3.2:1b-instruct-q5_K_M llama3.2:1b-instruct-q8_0"}
# model -> GiB of free RAM to wait for before loading (weights + KV cache + headroom)
declare -A NEED=( ["llama3.2:1b"]=1.9 ["qwen2.5:3b"]=2.6 ["llama3.2:3b"]=2.7 ["phi4-mini"]=3.2
                  ["llama3.2:1b-instruct-q4_K_M"]=1.4 ["llama3.2:1b-instruct-q5_K_M"]=1.5
                  ["llama3.2:1b-instruct-q8_0"]=1.9 )
MODELS=${MODELS:-"llama3.2:1b qwen2.5:3b llama3.2:3b phi4-mini"}

gate() { $PY scripts/wait_for_memory.py "${NEED[$1]:-3}" "$MEM_WAIT_MIN"; }
stamp() { echo "=== $(date -u +%FT%TZ) $* (load: $(cut -d' ' -f1-3 /proc/loadavg))"; }

MAX_PASSES=${MAX_PASSES:-1}
for pass in $(seq 1 "$MAX_PASSES"); do
stamp "pass $pass/$MAX_PASSES"
for phase in $PHASES; do
  case $phase in
    perf)    for m in $MODELS; do stamp "perf $m"; gate "$m" && $FM bench perf -m "$m" --runs "$PERF_RUNS" --skip-done; done ;;
    quality) for m in $MODELS; do stamp "quality $m"; gate "$m" && $FM bench quality -m "$m"; done ;;
    temp)    stamp "temperature $TEMP_MODEL"; gate "$TEMP_MODEL" && $FM bench temperature -m "$TEMP_MODEL" --runs "$TEMP_RUNS" ;;
    ablation) for m in $ABLATION_MODELS; do stamp "ablation $m"; gate "$m" && \
                $FM bench quality -m "$m" --unconstrained --tag ablation_unconstrained; done ;;
    quant)   for m in $QUANT_MODELS; do stamp "quant $m"; gate "$m" && \
                $FM bench quality -m "$m" --tag quant && $FM bench perf -m "$m" --runs "$PERF_RUNS" --out results/quant_perf --skip-done; done ;;
    charts)  stamp charts; $FM bench charts ;;
  esac
done
done
stamp done
