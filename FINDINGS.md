# Findings

Entry format: Fact → Hypothesis → Experiment → Result → Upstream State → Re-check

## F-001: KV cache size per token (Qwen3-8B, BF16)

**Fact.** Qwen3-8B has 36 layers, 8 KV heads and a head size of 128 (from its `config.json`). In BF16, every cached token needs 2 (keys and values) × 36 × 8 × 128 × 2 bytes = 147,456 bytes.

**Hypothesis.** I stated this formula before the run: vLLM's reported cache size divided by its reported token count should come out near 147,456 bytes per token.

**Experiment.** vLLM 0.33.0, one NVIDIA A40 (RunPod Secure Cloud, EU-SE-1), V1 model runner (`VLLM_USE_V2_MODEL_RUNNER=0`), started with `vllm serve Qwen/Qwen3-8B --max-model-len 4096 --gpu-memory-utilization 0.90`.

**Result.** The startup log reported 22.96 GiB of KV cache and 167,216 tokens. That is about 147,433 bytes per token, within 0.02% of the formula. (The log rounds to two decimals, so this is as close as the log lets me check.)

With the EAGLE-3 speculator added at K=3, the log reported 20.89 GiB and 148,000 tokens, about 151,557 bytes per token. One extra layer would give 147,456 × 37/36 = 151,552. This fits, but I haven't opened the speculator's config, so "one extra layer" is a guess for now.

**Upstream state.** Measured on vLLM v0.33.0 (tagged 31 Oct 2026).

**Re-check.** After any vLLM upgrade, or if the KV cache layout or block allocation changes.

**What this means for my experiments.** The A40 can hold about 167,000 tokens of KV cache for Qwen3-8B (about 148,000 with the speculator), so memory is not the limit at the short requests I plan to use. It only becomes a limit near 4,096 tokens per request, where about 36 requests fit at once. The startup log, not the formula alone, will set the concurrency ceiling for every run.
