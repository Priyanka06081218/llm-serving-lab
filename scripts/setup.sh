#!/usr/bin/env bash
# setup.sh - get a fresh pod ready for the experiments.
# Run it once per pod, from the repo folder:  bash scripts/setup.sh
# It installs vLLM, downloads the pinned dataset and the two models,
# and writes down what the machine looks like.

set -eo pipefail

VLLM_VERSION="0.33.0"
MIN_FREE_GB=70
WORK=/workspace
VENV="$WORK/vllm-env"
LOG_DIR="$WORK/logs"
DATA_DIR="$WORK/data"
DATASET_URL="https://raw.githubusercontent.com/hemingkx/Spec-Bench/refs/heads/main/data/spec_bench/question.jsonl"
DATASET_SHA256="4b6d33e79484f9841c487ee87d1cf6aa8c6066f61d5d482ff09e5a007fafdf04"

mkdir -p "$LOG_DIR" "$DATA_DIR"
# Everything printed below is also saved to a log file.
exec > >(tee -a "$LOG_DIR/setup.log") 2>&1

echo "== 1. Disk space =="
df -h / "$WORK"
free_gb=$(df --output=avail -BG "$WORK" | tail -1 | tr -dc '0-9')
if [ "$free_gb" -lt "$MIN_FREE_GB" ]; then
  echo "Only ${free_gb} GB free on $WORK, need at least ${MIN_FREE_GB} GB. Stopping."
  exit 1
fi

echo "== 2. vLLM ${VLLM_VERSION} =="
pip install --quiet uv
if [ ! -d "$VENV" ]; then
  uv venv --python 3.12 "$VENV"
fi
source "$VENV/bin/activate"
if uv pip show vllm 2>/dev/null | grep -q "^Version: ${VLLM_VERSION}$"; then
  echo "vLLM ${VLLM_VERSION} is already installed."
else
  uv pip install "vllm==${VLLM_VERSION}" --torch-backend=auto
fi
vllm download-kernels

echo "== 3. Dataset, pinned by checksum =="
if [ ! -f "$DATA_DIR/question.jsonl" ]; then
  curl -L -o "$DATA_DIR/question.jsonl" "$DATASET_URL"
fi
echo "${DATASET_SHA256}  $DATA_DIR/question.jsonl" | sha256sum -c -

echo "== 4. Models =="
python - <<'EOF' | tee "$LOG_DIR/model-revisions.txt"
from huggingface_hub import snapshot_download
for repo in ["Qwen/Qwen3-8B", "RedHatAI/Qwen3-8B-speculator.eagle3"]:
    path = snapshot_download(repo)
    print(repo, "->", path)
EOF

echo "== 5. Record the environment =="
{
  echo "date (UTC): $(date -u +%FT%TZ)"
  nvidia-smi
  df -h / "$WORK"
  echo "vllm: $(uv pip show vllm | grep '^Version')"
  python -c "import torch, transformers; print('torch', torch.__version__, '| transformers', transformers.__version__, '| cuda', torch.version.cuda)"
  echo "repo commit: $(git -C "$(dirname "$0")/.." rev-parse HEAD 2>/dev/null || echo unknown)"
} > "$LOG_DIR/env.txt" 2>&1
uv pip freeze > "$LOG_DIR/pip-freeze.txt"
cat "$LOG_DIR/env.txt"
echo "Setup finished. Next: source $VENV/bin/activate"
