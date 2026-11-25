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


**Update, 24 Nov 2026.** In the first validation session the server without speculation reported 169,568 tokens of KV cache. The K=3 server reported 149,104 and the Dynamic SD server 149,232. F-001 measured 167,216 and about 148,000. They are close but not the same. The new runs added `--max-num-seqs 64`, which may explain it, but I have not tested that. So I read the cache size from each server's startup log every time and do not reuse an old number. Evidence: `evidence/session1/logs/server_off.log`, `server_k3.log`, `server_dynamic.log`.

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

**Update, 24 Nov 2026.** Answers from the first validation session's logs (`evidence/session1/logs/`):

- CUDA graph mode: the baseline and the static K=3 servers both used `FULL_AND_PIECEWISE`. Only the Dynamic SD server was lowered to `PIECEWISE`. So the confound is real. Any difference between the dynamic arm and the other arms includes this setting.
- `max_num_scheduled_tokens`: the static K=3 and Dynamic SD servers both printed that it was set to 2048 because of the speculative settings. The baseline log has no such line. What the baseline used is still unknown.
- The concurrency 32 run is done. See F-003.
- "Peak concurrent requests" in the benchmark output: in v0.33.0 `serve.py`, line 753, the value is `int(np.max(concurrent_requests_per_second))`. I think this counts per-second bins, not requests alive at one moment, but I have not tested that. It read 13 at a limit of 8 and 42 at a limit of 32. I do not use it. I use the server gauge `vllm:num_requests_running`. In the saved gauge files the running count never went above the limit (maximum 4 at limit 4, 16 at 16, 32 at 32).

## F-003: How vLLM picks K for Dynamic SD (code reading)

**What I found.** Each time vLLM plans one step of work, it first decides which requests run in that step. Dynamic SD uses how many requests that is to look up K in my table. It counts requests, not tokens, so ten short requests and ten long ones both count as 10.

The count includes requests that are still reading their prompt, not only the ones already generating. The scheduler adds requests to the step's list in two places. One is the requests that are already running, and some of those are still working through a long prompt in pieces. The other is the requests taken from the waiting line, which are new ones and ones that were paused and are now resuming. The K lookup happens after both.

**How I checked.** I only read the code. It is in vLLM v0.33.0, file `vllm/v1/core/sched/scheduler.py`, inside the function `schedule()`. The two places are at lines 1001 and 1569, and the K lookup is at lines 1709 to 1714. I did not run anything on a GPU for this one.

**Not checked yet.**
- Whether a request that is still reading its prompt gets draft tokens in that step. If it doesn't, my table's count includes requests that K doesn't really apply to.
- What the code does between lines 1569 and 1709. I only know the lookup comes after the two places because of the line numbers.
- Whether the table picks the right K at concurrency 32. That needs a GPU run. Answered on 24 Nov 2026: the server does use K=1 at concurrency 32 (see the result below). Whether K=1 is a good choice there is still open.

**Upstream state.** This is true for vLLM v0.33.0.

**Re-check.** If I change the vLLM version, read these lines again first, because the scheduler changes often and this may no longer be true.

**Prediction (written before the run; result below).** If K really depends on the number of requests in a step, then with my table (K=3 for 1 to 16 requests, K=1 for 17 to 64), a run with 32 requests in flight should show about 1 draft token per draft. I get that number from the `/metrics` page by dividing total draft tokens by total drafts. It won't be exactly 1.00, because at the start and end of the run fewer than 17 requests are in a step, and those steps use K=3. I'll call the prediction confirmed if the ratio is below 1.5, and wrong if it is near 3.00 like the concurrency-4 run. Anything in between means I have to find out why before I conclude anything. I'm writing the threshold down now, before the run, so I can't move it afterwards. The run needs a fresh server so counters from earlier runs aren't mixed in.

The question about requests that are still reading their prompt needs its own test, which I haven't designed yet.

**Result (run on 24 Nov 2026, fresh Dynamic SD server).** Same table `[[1,16,3],[17,64,1]]`, `--max-num-seqs 64`, 160 prompts at concurrency 32, 8 warmup requests.

- Concurrency 32: 16,362 draft rounds and 17,194 draft tokens, which is **1.05 draft tokens per draft**. My written limit was 1.5, so the prediction is **confirmed**.
- Control at concurrency 4 (48 prompts): 3,669 rounds and 11,007 draft tokens, exactly 3.00, as in F-002.
- If every round used K=1 or K=3, then about 416 of the 16,362 rounds (2.5%) used K=3. I think those are steps at the start and end of the run with fewer than 17 requests, but I did not check that.
- These counts include the warmup requests. The benchmark tool's own counters, which I think leave the warmups out, give 16,250 rounds and 16,858 draft tokens, which is 1.04.

Acceptance per draft token went from 0.42 at K=3 to 0.64 at K=1. That is averaging, not better drafts. The benchmark tool also saves acceptance by draft position. Position 1 was accepted 0.656 of the time in the static run at concurrency 8, 0.658 in the dynamic run at concurrency 4 and 0.663 in the dynamic run at concurrency 32. Positions 2 and 3 were about 0.41 and 0.21 in the K=3 runs. With K=1 only the best position is counted.

What this confirms: the table is looked up by the number of requests in a step, and above 16 the server really uses K=1. What it does not answer: whether requests still reading their prompt count in that number (still open), and whether K=1 at concurrency 32 is a good choice. That needs static K=1 and K=3 runs at the same load.

Evidence: `evidence/session1/logs/validate.log` (lines 790 to 795), `snap_dyn_*.json`, `results/validate/dyn_c32.json` and `dyn_c04.json`.

## F-004: Spec-Bench prompt lengths (measured with the Qwen3 tokenizer)

**What I found.** All 480 prompts fit comfortably in the 4,096-token limit. The longest is 1,500 tokens (a summarization prompt), well under the 3,840 I can allow once I leave 256 tokens for the answer. The counts include the chat template with thinking off, which is what the server will build.

The prompts fall into two groups. Most categories are short: qa has a median of 23 tokens, translation 51, math_reasoning 66, and the eight 10-question categories have medians between 31 and 219. Summarization (median 672, longest 1,500) and rag (median 735, longest 855) are much longer. That is the biggest difference between the workloads this dataset offers.

Estimate: even 64 requests of the longest size (1,500 prompt tokens plus 256 output tokens) need about 112,000 tokens of KV cache. F-001 measured about 148,000 with the speculator loaded, so memory should not be the limit. The server's startup log will confirm this.

**What this means for my experiments.** Any split into a profile workload and a held-out workload will mostly be a split between short-prompt and long-prompt traffic. If I put summarization and rag on one side, the two workloads differ in how much prompt each request carries, and not only in what kind of text it is. I will keep that in mind when I read any difference between the profile and held-out results.

**How I checked.** I ran `scripts/check_prompt_lengths.py` on my Mac with Transformers 4.57.1. The tokenizer repo revision was b968826d9c46dd6066d109eabc6255188de91218, and the dataset file matched SHA-256 4b6d33e79484f9841c487ee87d1cf6aa8c6066f61d5d482ff09e5a007fafdf04. The full output is in `docs/prompt-lengths.txt`.

**Not checked yet.**
- How long the answers are. Prompt lengths say nothing about output lengths. The qa questions may get very short answers, which would give speculation little to work with. I will measure output lengths in the validation session, using only the server without speculation.
- Which categories go into which workload. That is not decided.

**Upstream state.** This is about the dataset, not vLLM. The data is pinned by the hash above, and Spec-Bench's main branch can change.

**Re-check.** If the dataset file, the tokenizer revision or the context limit changes.

## F-005: First validation session (vLLM 0.33.0, one A40)

**What this was.** A validation run, not the experiment. I wanted to know that my scripts work, what the server does without speculation, and which settings could confuse a later comparison. Every number below comes from the raw files in `evidence/session1/` (checksums in `SHA256SUMS`, the commands in `logs/commands.log`). Each configuration was run once.

**Setup.** One NVIDIA A40 (46,068 MiB), driver 595.112.07, CUDA 13.2. vLLM 0.33.0, torch 2.14.0+cu132, transformers 5.20.2, Python 3.12.3, repo commit `c314ef5`. Models: `Qwen/Qwen3-8B` and the EAGLE-3 speculator `RedHatAI/Qwen3-8B-speculator.eagle3` (exact revisions in `logs/model-revisions.txt`). Every server used `--max-model-len 4096 --gpu-memory-utilization 0.90 --max-num-seqs 64` and `VLLM_USE_V2_MODEL_RUNNER=0`. Every benchmark used `vllm bench serve` with the Spec-Bench dataset, temperature 0, seed 0, 256 output tokens, 8 warmup requests, the chat template applied once and thinking off.

**Check 1: the prompt path.** `logs/validate.log` (lines 25 to 28) shows 24 tokens when the chat template is applied once and 36 if it were applied twice. The server path gave the once-applied result. Across the 1,496 generated texts in the result files, none contains a `<think>` or `</think>` tag.

**Check 2: output lengths** (concurrency 16, Spec-Bench categories):

| Category | Prompts | Median output tokens | Reached the 256 cap |
|---|---|---|---|
| humanities | 10 | 256 | 100% |
| stem | 10 | 256 | 100% |
| coding | 10 | 256 | 90% |
| math | 10 | 256 | 70% |
| math_reasoning | 80 | 239 | 42% |
| qa | 80 | 204 | 42% |
| reasoning | 10 | 230 | 50% |
| roleplay | 10 | 208 | 50% |
| rag | 80 | 195 | 36% |
| summarization | 80 | 197 | 19% |
| writing | 10 | 194 | 30% |
| extraction | 10 | 69 | 10% |
| translation | 80 | 24 | 0% (longest 70) |

Translation barely generates anything, and extraction is short. Humanities, stem, coding and math almost always run to the cap. The eight MT-bench categories have only 10 prompts each, so I make no claim about any single one. My summary script prints the lower of the two middle values for an even count, so its "len50" column reads 55 for extraction, where the true median is 69. I used the true median here. The script needs fixing.

**Check 3: baseline capacity** (no speculation, one run per level; the number of prompts grows with the level):

| Concurrency | Prompts | Output tok/s | Median first token (ms) | Median time per token (ms) | p99 time per token (ms) |
|---|---|---|---|---|---|
| 1 | 24 | 34 | 64 | 29.1 | 29.4 |
| 2 | 32 | 64 | 89 | 29.4 | 29.6 |
| 4 | 48 | 131 | 92 | 29.7 | 30.5 |
| 8 | 64 | 238 | 97 | 30.7 | 31.3 |
| 16 | 96 | 425 | 106 | 32.6 | 34.7 |
| 32 | 160 | 748 | 128 | 37.4 | 41.6 |
| 64 | 256 | 1,101 | 230 | 47.1 | 70.0 |

Throughput grows about 32 times from concurrency 1 to 64, and the time per token grows only about 1.6 times. So decoding on this GPU is far from saturated at these loads, which is where speculation has room to help. Concurrency 64 equals `--max-num-seqs`.

**Check 4: static K=3 against the baseline at concurrency 8** (64 prompts). Throughput 382 against 238 tok/s. Median time per token 18.6 against 30.7 ms. Acceptance 0.43 per draft token, 2.28 tokens per round, by position 0.656, 0.408 and 0.215. The first-token times (139 against 97 ms) are not comparable, because of the prefix cache (below).

**Check 5: text agreement.** At temperature 0 I compared the generated texts of runs on the same prompts:

| Comparison | Identical texts |
|---|---|
| Static K=3 at c=8 against baseline at c=8 | 30 of 64 (47%) |
| Baseline at c=8 against baseline at c=16 (control, no speculation) | 32 of 64 (50%) |
| Dynamic SD at c=4 against baseline at c=4 | 26 of 48 (54%) |
| Dynamic SD at c=32 against baseline at c=32 | 65 of 160 (41%) |

Two baseline runs with different batch sizes already disagree on half the prompts, so most of the disagreement is not about speculation. This does not show that speculation leaves the output unchanged either. It only shows this test cannot tell the two apart.

**A lead, not a result: Dynamic SD against the baseline.** Single runs, same prompts:

| | Baseline c=4 | Dynamic c=4 | Baseline c=32 | Dynamic c=32 |
|---|---|---|---|---|
| Output tok/s | 131 | 237 | 748 | 800 |
| Median time per token (ms) | 29.7 | 16.4 | 37.4 | 34.4 |
| p99 time per token (ms) | 30.5 | 21.2 | 41.6 | 68.1 |
| Median first token (ms) | 92 | 110 | 128 | 236 |
| p99 first token (ms) | 155 | 115 | 232 | 1,337 |

At concurrency 4 the dynamic server is clearly faster in this run. At concurrency 32 the average is a little better and the tail is worse. I do not claim that Dynamic SD hurts the tail, because these things differ between the arms:

- Prefix caching was on for all three servers (`enable_prefix_caching=True` in each log). The dynamic server was fresh, while the baseline sweep ran later on a server that had already seen many of the same prompts. I think this inflates the dynamic first-token times, but I did not measure it. A hit rate of about 52% was read from the live metrics page during the session, but that reading was not saved in the evidence files.
- CUDA graph mode: `PIECEWISE` for the dynamic server, `FULL_AND_PIECEWISE` for the others.
- `max_num_scheduled_tokens` was 2048 on the two speculative servers. The baseline's value is unknown. In the dynamic run at concurrency 32 the server's waiting queue reached 14 requests (`logs/running_dyn_c32.csv`). The baseline sweep has no gauge file, so I cannot say whether the baseline queued less.
- Each result is one run, and I have no repeat to tell me how large normal noise is. The only hint is the baseline at concurrency 16: 425 tok/s with 96 prompts and 436 with 64 prompts.
- There is no static K=3 or K=1 run at concurrency 32, so I cannot say whether the table `[[1,16,3],[17,64,1]]` is a good rule.

**Not checked yet.**
- Run-to-run noise. Every configuration needs repeats.
- What prefix caching does to the comparison. `--disable-prefix-caching` exists in the 0.33.0 help text. I have not yet started a server with it.
- The baseline's `max_num_scheduled_tokens` and its startup config.
- Whether the per-position acceptance changes with the kind of text. I only looked at the Spec-Bench mix.
- The F-004 prompt counts used transformers 4.57.1. The pod had 5.20.2. I did not repeat the counts on the pod.
- `scripts/setup.sh` does not install the benchmark extra. I installed `vllm[bench]==0.33.0` by hand on the pod. The script is not fixed yet.

**Upstream state.** vLLM v0.33.0 (tagged 31 Oct 2026). Run on 23 and 24 Nov 2026.

**Re-check.** After any vLLM upgrade, and before the experiment design is frozen.

**What this means for my experiments.** These are decisions I still have to make and log, not decisions already made:
- Whether to turn prefix caching off for every arm, so a fresh and a warm server behave the same.
- Whether to add a baseline arm in `PIECEWISE` mode, so the CUDA graph mode is separated from Dynamic SD itself.
- How many repeats each configuration gets, so I know the noise before I read any difference.
- Which categories go into the profile and the held-out workload. Translation and extraction make very little output, so they say little about speculation.
