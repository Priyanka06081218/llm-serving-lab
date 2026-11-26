# lib.sh - small helpers shared by the session scripts. Source it, do not run it.
# shellcheck shell=bash

WORK="${WORK:-/workspace}"
LOG_DIR="$WORK/logs"
RESULTS="$WORK/results/validate"
DATA="$WORK/data/question.jsonl"
# Port and model are exported so the Python tools (metrics_tool.py,
# chat_check.py) use the same values. Override with LAB_PORT / LAB_MODEL.
PORT="${LAB_PORT:-8000}"
MODEL="${LAB_MODEL:-Qwen/Qwen3-8B}"
SPECULATOR="RedHatAI/Qwen3-8B-speculator.eagle3"
export LAB_PORT="$PORT" LAB_MODEL="$MODEL"
PID_FILE="$LOG_DIR/server.pid"
NAME_FILE="$LOG_DIR/server.name"
CLOCK_FILE="$LOG_DIR/session_start"

# Session limits. These match configs/experiment.yaml (budget section).
PRICE_PER_HOUR="${PRICE_PER_HOUR:-0.59}"
MAX_HOURS="${MAX_HOURS:-3}"

# The two speculative-decoding settings used in this session.
# static K=3, and Dynamic SD with the throwaway table from the smoke test.
# validate_gpu.sh uses them after sourcing this file.
# shellcheck disable=SC2034
SPEC_K3="{\"model\":\"$SPECULATOR\",\"num_speculative_tokens\":3,\"method\":\"eagle3\"}"
# shellcheck disable=SC2034
SPEC_DYNAMIC="{\"model\":\"$SPECULATOR\",\"num_speculative_tokens\":3,\"method\":\"eagle3\",\"num_speculative_tokens_per_batch_size\":[[1,16,3],[17,64,1]]}"

LIB_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

mkdir -p "$LOG_DIR" "$RESULTS"

# log_cmd CMD...: add one line to commands.log. printf %q keeps the quoting,
# so a logged line can be copied and run again exactly as it was.
log_cmd() {
  { printf '[%s] ' "$(date -u +%FT%TZ)"; printf '%q ' "$@"; echo; } >> "$LOG_DIR/commands.log"
}

# ---------------------------------------------------------------- session clock
# The clock starts the first time a step runs. If setup.log exists we start it
# from there, so the time spent on setup is counted too.
session_clock() {
  if [ ! -f "$CLOCK_FILE" ]; then
    local born
    born=$(stat -c %W "$LOG_DIR/setup.log" 2>/dev/null || echo 0)
    if [ "${born:-0}" -gt 0 ] 2>/dev/null; then echo "$born" > "$CLOCK_FILE"; else date +%s > "$CLOCK_FILE"; fi
  fi
  awk -v s="$(cat "$CLOCK_FILE")" -v n="$(date +%s)" -v p="$PRICE_PER_HOUR" \
    'BEGIN { e = n - s; printf "Session clock: %d min used, about $%.2f of GPU time at list price (the RunPod console shows the real bill).\n", e/60, e/3600*p }'
}

# session_guard: refuse to start new work after MAX_HOURS. It does not stop
# anything that is already running, and it does not stop the pod. You do that.
session_guard() {
  session_clock
  if awk -v s="$(cat "$CLOCK_FILE")" -v n="$(date +%s)" -v m="$MAX_HOURS" 'BEGIN { exit !((n - s) > m * 3600) }'; then
    echo "The ${MAX_HOURS}-hour limit is used up. Not starting anything new. Stop the pod in the RunPod console."
    return 1
  fi
}

# ---------------------------------------------------------------- server
# cache_flag: the prefix-caching flag every server gets. Caching is OFF for
# every arm, so a fresh server and a warm one behave the same and the arms
# differ only in speculation. Session 1 ran with the vLLM default (caching on);
# LAB_PREFIX_CACHING=on gives that behaviour back.
cache_flag() {
  case "${LAB_PREFIX_CACHING:-off}" in
    off) echo "--disable-prefix-caching" ;;
    on) echo "--enable-prefix-caching" ;;
    *) echo "LAB_PREFIX_CACHING must be 'on' or 'off', not '${LAB_PREFIX_CACHING}'." >&2; return 1 ;;
  esac
}

# runner_env: which model runner every server uses. V2 for every arm, because on V1
# vLLM 0.33.0 forces Dynamic SD into PIECEWISE graphs while the other arms keep full
# graphs (vllm/config/vllm.py, _maybe_override_dynamic_sd_cudagraph_mode). The value
# is always set, never left to the default, so a server that V2 cannot run fails
# loudly instead of quietly falling back to V1. Not yet checked on the A40.
# LAB_RUNNER=v1 gives the session 1 runner back.
runner_env() {
  case "${LAB_RUNNER:-v2}" in
    v2) echo "VLLM_USE_V2_MODEL_RUNNER=1" ;;
    v1) echo "VLLM_USE_V2_MODEL_RUNNER=0" ;;
    *) echo "LAB_RUNNER must be 'v1' or 'v2', not '${LAB_RUNNER}'." >&2; return 1 ;;
  esac
}

# start_server NAME [extra vllm arguments]
# Starts vLLM in the background, saves its output to a log file, and waits
# until the server answers. Every arm gets the same base settings, so the
# arms differ only in the arguments passed after NAME.
start_server() {
  local name="$1"; shift
  session_guard || return 1
  local cache runner
  cache=$(cache_flag) || return 1
  runner=$(runner_env) || return 1
  if [ -f "$PID_FILE" ] && kill -0 "$(cat "$PID_FILE")" 2>/dev/null; then
    echo "A server is already running (pid $(cat "$PID_FILE")). Run the stop step first."
    return 1
  fi
  local log="$LOG_DIR/server_${name}.log"
  local cmd=(env "$runner" vllm serve "$MODEL"
    --max-model-len 4096 --gpu-memory-utilization 0.90 --max-num-seqs 64
    --max-num-batched-tokens 2048
    --port "$PORT" "$cache" "$@")
  log_cmd "${cmd[@]}"
  echo "Starting server '$name'. Log: $log"
  nohup "${cmd[@]}" > "$log" 2>&1 < /dev/null &
  local pid=$!
  echo "$pid" > "$PID_FILE"
  echo "$name" > "$NAME_FILE"
  local waited=0
  while [ "$waited" -lt "${START_TIMEOUT:-900}" ]; do
    if curl -sf "localhost:${PORT}/health" > /dev/null; then
      echo "Server '$name' is up after about ${waited} seconds."
      return 0
    fi
    if ! kill -0 "$pid" 2>/dev/null; then
      echo "Server '$name' stopped early. Last lines of its log:"
      tail -n 30 "$log"
      return 1
    fi
    sleep 5
    waited=$((waited + 5))
  done
  echo "Server '$name' did not answer in time. Stopping it."
  stop_server || true
  return 1
}

# stop_server: stop the server and wait until the GPU memory is free again.
stop_server() {
  if [ -f "$PID_FILE" ]; then
    local pid
    pid=$(cat "$PID_FILE")
    kill "$pid" 2>/dev/null || true
    rm -f "$PID_FILE"
  fi
  rm -f "$NAME_FILE"
  local i used
  for i in $(seq 1 24); do
    used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1 | tr -dc '0-9')
    if [ "${used:-0}" -lt 1000 ]; then
      return 0
    fi
    if [ "$i" -eq 12 ]; then
      pkill -f "vllm serve" || true
    fi
    sleep 5
  done
  echo "GPU memory is still in use after 2 minutes (${used} MiB). Check nvidia-smi."
  return 1
}

# require_server NAME: stop early, with a clear message, if the wrong server (or none) is up.
require_server() {
  local want="$1" have="none"
  [ -f "$NAME_FILE" ] && have=$(cat "$NAME_FILE")
  if [ "$have" != "$want" ] || ! curl -sf "localhost:${PORT}/health" > /dev/null; then
    echo "This step needs the '$want' server, but the running server is '$have' (or it is not answering)."
    echo "Start it first, then run this step again."
    return 1
  fi
}

# ---------------------------------------------------------------- benchmark
# run_bench NAME CONCURRENCY NUM_PROMPTS [extra bench arguments]
# One closed-loop run: CONCURRENCY requests are always in flight.
# Writes $RESULTS/NAME.json (with per-request details) and $LOG_DIR/bench_NAME.log.
run_bench() {
  local name="$1" conc="$2" n="$3"; shift 3
  session_guard || return 1
  local logf="$LOG_DIR/bench_${name}.log"
  # The commas below are inside single arguments (ttft,tpot,itl,e2el), on purpose.
  # shellcheck disable=SC2054
  local cmd=(vllm bench serve
    --backend openai-chat --endpoint /v1/chat/completions
    --model "$MODEL" --port "$PORT"
    --dataset-name spec_bench --dataset-path "$DATA"
    --skip-chat-template
    --extra-body '{"chat_template_kwargs":{"enable_thinking":false}}'
    --temperature 0 --seed 0
    --output-len 256 --spec-bench-output-len 256
    --no-oversample --num-prompts "$n" --max-concurrency "$conc"
    --num-warmups 8
    --percentile-metrics ttft,tpot,itl,e2el --metric-percentiles 50,90,95,99
    --save-result --save-detailed
    --result-dir "$RESULTS" --result-filename "${name}.json"
    --disable-tqdm "$@")
  log_cmd "${cmd[@]}"
  echo "Running '$name': concurrency $conc, $n prompts ..."
  # < /dev/null stops the benchmark from reading (and eating) the input of a surrounding loop.
  if ! "${cmd[@]}" > "$logf" 2>&1 < /dev/null; then
    echo "The benchmark failed. Last lines of $logf:"
    tail -n 30 "$logf"
    return 1
  fi
  grep -i -E "successful|failed|duration|throughput|concurren|TTFT|TPOT" "$logf" || true
}

# run_bench_watched NAME CONCURRENCY NUM_PROMPTS [extra bench arguments]
# Same as run_bench. At the same time it samples the server's running and
# waiting request gauges once a second, so we can compare the load we asked
# for with what the server saw. If the benchmark fails, this fails too.
run_bench_watched() {
  local name="$1"
  local flag="$LOG_DIR/${name}.done"
  rm -f "$flag" "$flag.tmp"
  ( if run_bench "$@"; then echo 0 > "$flag.tmp"; else echo 1 > "$flag.tmp"; fi; mv "$flag.tmp" "$flag" ) &
  local bench_pid=$!
  python "$LIB_DIR/metrics_tool.py" watch --stop-file "$flag" --out "$LOG_DIR/running_${name}.csv"
  wait "$bench_pid" || true
  local status
  status=$(cat "$flag" 2>/dev/null || echo 1)
  if [ "$status" != "0" ]; then
    echo "The benchmark '$name' FAILED. Its results are not valid."
    return 1
  fi
}
