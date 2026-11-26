#!/usr/bin/env bash
# validate_gpu.sh - the first paid session. Run one step at a time, read the
# result, then run the next:
#   bash scripts/validate_gpu.sh env          what machine and versions we got
#   bash scripts/validate_gpu.sh baseline     start the no-speculation server
#   bash scripts/validate_gpu.sh chat-check   is the chat template applied once?
#   bash scripts/validate_gpu.sh lengths      output length per Spec-Bench category
#   bash scripts/validate_gpu.sh sweep        how much load the baseline can take
#   bash scripts/validate_gpu.sh metrics      does the running-requests gauge work?
#   bash scripts/validate_gpu.sh static       static K=3 server, short run
#   bash scripts/validate_gpu.sh dynamic-test the F-003 prediction
#   bash scripts/validate_gpu.sh stop         stop the server
# If a step fails, stop the pod and fix it on the Mac.

set -eo pipefail
WORK="${WORK:-/workspace}"
source "$WORK/vllm-env/bin/activate"
HERE="$(cd "$(dirname "$0")" && pwd)"
source "$HERE/lib.sh"
exec > >(tee -a "$LOG_DIR/validate.log") 2>&1

step="${1:-}"
STARTUP_FACTS="non-default args|max_num_batched|KV cache|prefix caching|cudagraph|cuda graph|max_num_scheduled|maximum concurrency|chunked|speculat"
METRICS="python $HERE/metrics_tool.py"
SUMMARY="python $HERE/summarize_bench.py"

case "$step" in
  env)
    session_clock
    echo "== Environment (written by setup.sh) =="
    cat "$LOG_DIR/env.txt"
    nvidia-smi --query-gpu=name,memory.total,memory.used,driver_version --format=csv
    df -h / "$WORK"
    echo "Dataset lines: $(grep -c . "$DATA")"
    ;;

  baseline)
    echo "== Baseline server: no speculation =="
    start_server off
    echo "== Startup facts from the log =="
    grep -i -E "$STARTUP_FACTS" "$LOG_DIR/server_off.log" | head -40 || true
    ;;

  chat-check)
    echo "== Chat path check =="
    require_server off
    python "$HERE/chat_check.py"
    ;;

  lengths)
    # Why: how long are the answers per category when nothing is speculated?
    # Categories where most answers hit the 256-token cap, or end after a few
    # tokens, behave very differently for speculative decoding.
    echo "== Output length per category (baseline, concurrency 16) =="
    require_server off
    python - "$DATA" > "$LOG_DIR/categories.txt" <<'PY'
import collections, json, sys
counts = collections.Counter(json.loads(l)["category"] for l in open(sys.argv[1]) if l.strip())
for k, v in sorted(counts.items()):
    print(v, k)
PY
    cat "$LOG_DIR/categories.txt"
    while read -r count cat; do
      n=$count
      if [ "$n" -gt "${LENGTH_MAX_PER_CAT:-100}" ]; then n="${LENGTH_MAX_PER_CAT:-100}"; fi
      run_bench "len_${cat}" 16 "$n" --spec-bench-category "$cat"
    done < "$LOG_DIR/categories.txt"
    echo "== Summary =="
    $SUMMARY "$RESULTS"/len_*.json
    ;;

  sweep)
    # Why: find where the baseline stops scaling. The request rates for the
    # real evaluation are chosen from this, not guessed.
    echo "== Baseline capacity sweep (all categories mixed) =="
    require_server off
    total=$(grep -c . "$DATA")
    for pair in "1 24" "2 32" "4 48" "8 64" "16 96" "32 160" "64 256"; do
      read -r c n <<< "$pair"
      if [ "$n" -gt "$total" ]; then n="$total"; fi
      run_bench "sweep_c$(printf '%02d' "$c")" "$c" "$n"
    done
    echo "== Summary =="
    $SUMMARY "$RESULTS"/sweep_c*.json
    ;;

  metrics)
    # Why: the Dynamic SD table is keyed by requests per step, but we can only
    # sample how many requests are running. Check the gauge exists and moves.
    echo "== Metrics check (baseline, concurrency 16) =="
    require_server off
    $METRICS list > "$LOG_DIR/metric_names_off.txt"
    echo "Metric names that matter here:"
    grep -E "running|waiting|kv_cache|prefix_cache|spec_decode" "$LOG_DIR/metric_names_off.txt" || true
    run_bench_watched metrics_load 16 64
    ;;

  static)
    echo "== Static K=3 server =="
    stop_server
    start_server k3 --speculative-config "$SPEC_K3"
    grep -i -E "$STARTUP_FACTS" "$LOG_DIR/server_k3.log" | head -40 || true
    $METRICS snap "$LOG_DIR/snap_k3_before.json"
    run_bench static_c08 8 64
    $METRICS snap "$LOG_DIR/snap_k3_after.json"
    $METRICS delta "$LOG_DIR/snap_k3_before.json" "$LOG_DIR/snap_k3_after.json" --label "static K=3, concurrency 8"
    echo "== Speed against the baseline at the same concurrency =="
    $SUMMARY "$RESULTS/sweep_c08.json" "$RESULTS/static_c08.json"
    echo "== Same text as the baseline? (temperature 0) =="
    $SUMMARY --compare "$RESULTS/sweep_c08.json" "$RESULTS/static_c08.json"
    ;;

  dynamic-test)
    # The F-003 prediction: with 32 requests in flight the table should pick
    # K=1, so draft tokens per draft should be near 1 (not near 3).
    echo "== Dynamic SD server, throwaway table [[1,16,3],[17,64,1]] =="
    stop_server
    start_server dynamic --speculative-config "$SPEC_DYNAMIC"
    grep -i -E "$STARTUP_FACTS" "$LOG_DIR/server_dynamic.log" | head -40 || true
    $METRICS list > "$LOG_DIR/metric_names_dynamic.txt"
    $METRICS snap "$LOG_DIR/snap_dyn_0.json"
    run_bench_watched dyn_c32 32 160
    $METRICS snap "$LOG_DIR/snap_dyn_1.json"
    $METRICS delta "$LOG_DIR/snap_dyn_0.json" "$LOG_DIR/snap_dyn_1.json" --label "concurrency 32 (the prediction run)" --prediction
    run_bench_watched dyn_c04 4 48
    $METRICS snap "$LOG_DIR/snap_dyn_2.json"
    $METRICS delta "$LOG_DIR/snap_dyn_1.json" "$LOG_DIR/snap_dyn_2.json" --label "concurrency 4 (control, F-002 gave 3.00)"
    $SUMMARY "$RESULTS/dyn_c32.json" "$RESULTS/dyn_c04.json"
    ;;

  stop)
    stop_server
    echo "Server stopped."
    ;;

  *)
    sed -n '2,15p' "$0"
    exit 1
    ;;
esac
echo
session_clock
