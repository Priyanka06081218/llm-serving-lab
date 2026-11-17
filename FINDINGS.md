# Findings

Each entry says what I found, how I checked it, what I still need to check, which vLLM version it is true for, and when to look at it again.

## F-001: KV cache size per token (Qwen3-8B, BF16)

**Fact.** Qwen3-8B has 36 layers, 8 KV heads and a head size of 128 (from its `config.json`). In BF16, every cached token needs 2 (keys and values) × 36 × 8 × 128 × 2 bytes = 147,456 bytes.

**Hypothesis.** I stated this formula before the run: vLLM's reported cache size divided by its reported token count should come out near 147,456 bytes per token.

**Experiment.** vLLM 0.33.0, one NVIDIA A40 (RunPod Secure Cloud, EU-SE-1), V1 model runner (`VLLM_USE_V2_MODEL_RUNNER=0`), started with `vllm serve Qwen/Qwen3-8B --max-model-len 4096 --gpu-memory-utilization 0.90`.

**Result.** The startup log reported 22.96 GiB of KV cache and 167,216 tokens. That is about 147,433 bytes per token, within 0.02% of the formula. (The log rounds to two decimals, so this is as close as the log lets me check.)

With the EAGLE-3 speculator added at K=3, the log reported 20.89 GiB and 148,000 tokens, about 151,557 bytes per token. One extra layer would give 147,456 × 37/36 = 151,552. This fits, but I haven't opened the speculator's config, so "one extra layer" is a guess for now.

**Upstream state.** Measured on vLLM v0.33.0 (tagged 31 Oct 2026).

**Re-check.** After any vLLM upgrade, or if the KV cache layout or block allocation changes.

**What this means for my experiments.** The A40 can hold about 167,000 tokens of KV cache for Qwen3-8B (about 148,000 with the speculator), so memory is not the limit at the short requests I plan to use. It only becomes a limit near 4,096 tokens per request, where about 36 requests fit at once. The startup log, not the formula alone, will set the concurrency ceiling for every run.


## F-002: Dynamic SD smoke test (vLLM 0.33.0, one A40)

**Fact.** In vLLM 0.33.0, Dynamic SD is set with a list of `[start, end, K]` ranges called `num_speculative_tokens_per_batch_size`, inside `--speculative-config`. The docs say the range is picked by "concurrency".

**Hypothesis.** I used a throwaway table, `[[1,16,3],[17,64,1]]`, with `--max-num-seqs 64`. I expected that at a concurrency of 4, K would always be 3, so the number of draft tokens divided by the number of draft rounds would be 3.0.

**Experiment.** Same A40 and same flags as F-001, plus the EAGLE-3 speculator (`RedHatAI/Qwen3-8B-speculator.eagle3`, 3 draft tokens), `--max-num-seqs 64` and the table above. Then `vllm bench serve` with the random dataset, 128 tokens in and 128 out, 40 requests, `--max-concurrency 4`, `--temperature 0`.

**Result.** The server took the table without a validation error. The benchmark reported 2,781 draft rounds and 8,343 draft tokens, which is exactly 3.00 per round, so K was 3 as the table says for 1 to 16 requests. The counts also add up: 2,342 accepted tokens plus one token per round (2,781) is 5,123, and the run generated 5,120.

One more thing I noticed: mean inter-token latency was 36.09 ms and mean time per output token was 19.76 ms, a ratio of 1.83. The acceptance length was 1.84. I think the match is because speculation delivers several tokens at once, but I haven't tested that explanation. The acceptance rate (28%) probably says little, because the random dataset is made-up text. I will measure acceptance on real text later.

**What the startup logs told me.**
- With Dynamic SD, vLLM printed: "Overriding cudagraph_mode from FULL_AND_PIECEWISE to PIECEWISE for reliability". I haven't checked which mode the static speculative server used.
- The static K=3 server printed that `max_num_scheduled_tokens` was set to 2048 "based on the speculative decoding settings". I haven't checked what the no-speculation server used.
- The no-speculation server printed that vLLM's default sampling settings were replaced by the model's own (`temperature` 0.6, `top_k` 20, `top_p` 0.95). The log suggests `--generation-config vllm` to stop that. I will set temperature 0 on every request.
- The Dynamic SD docs' examples all start with `VLLM_USE_V2_MODEL_RUNNER=0`, so all my servers did too. The speculator's model card says it was tested on vLLM 0.11.0. I ran 0.33.0 and it worked in this one test.

**Not checked yet.**
- A run at concurrency 32, where the table says K=1.
- What the other servers used for the CUDA graph mode and `max_num_scheduled_tokens`.
- Why `vllm bench serve` reported "Peak concurrent requests: 7" with `--max-concurrency 4`. I need to read its code before saying anything about it.

**Upstream state.** vLLM v0.33.0 (tagged 31 Oct 2026). Run on 4 Nov 2026.

**Re-check.** After a vLLM upgrade, and after the concurrency 32 run.

## F-003: How vLLM picks K for Dynamic SD (code reading)

**What I found.** Each time vLLM plans one step of work, it first decides which requests run in that step. Dynamic SD uses how many requests that is to look up K in my table. It counts requests, not tokens, so ten short requests and ten long ones both count as 10.

The count includes requests that are still reading their prompt, not only the ones already generating. The scheduler adds requests to the step's list in two places. One is the requests that are already running, and some of those are still working through a long prompt in pieces. The other is the requests taken from the waiting line, which are new ones and ones that were paused and are now resuming. The K lookup happens after both.

**How I checked.** I only read the code. It is in vLLM v0.33.0, file `vllm/v1/core/sched/scheduler.py`, inside the function `schedule()`. The two places are at lines 1001 and 1569, and the K lookup is at lines 1709 to 1714. I did not run anything on a GPU for this one.

**Not checked yet.**
- Whether a request that is still reading its prompt gets draft tokens in that step. If it doesn't, my table's count includes requests that K doesn't really apply to.
- What the code does between lines 1569 and 1709. I only know the lookup comes after the two places because of the line numbers.
- Whether the table picks the right K at concurrency 32. That needs a GPU run.

**Upstream state.** This is true for vLLM v0.33.0.

**Re-check.** If I change the vLLM version, read these lines again first, because the scheduler changes often and this may no longer be true.

**Prediction to test (not run yet).** If K really depends on the number of requests in a step, then with my table (K=3 for 1 to 16 requests, K=1 for 17 to 64), a run with 32 requests in flight should show about 1 draft token per draft. I get that number from the `/metrics` page by dividing total draft tokens by total drafts. It won't be exactly 1.00, because at the start and end of the run fewer than 17 requests are in a step, and those steps use K=3. I'll call the prediction confirmed if the ratio is below 1.5, and wrong if it is near 3.00 like the concurrency-4 run. Anything in between means I have to find out why before I conclude anything. I'm writing the threshold down now, before the run, so I can't move it afterwards. The run needs a fresh server so counters from earlier runs aren't mixed in.

The question about requests that are still reading their prompt needs its own test, which I haven't designed yet.
