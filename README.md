# LLM Serving Lab

**Status: In progress — setup and a smoke test are done, the benchmark harness is next. No benchmark results yet.**

## What am I trying to find out?

Speculative decoding is a technique used by vLLM to speed up LLM generation. A smaller draft model predicts a few tokens, and the main model checks those predictions together. When the predictions are accepted, generation can be faster.

But there's a trade-off. Checking draft tokens also takes compute, so speculative decoding doesn't always improve performance, especially when the server is under heavy load.

Dynamic Speculative Decoding (Dynamic SD) in vLLM tries to handle this by changing K, the number of draft tokens, at every step. It counts how many requests the scheduler is running in that step and reads K from a lookup table that has to be written by hand. That count includes requests that are still processing their prompt, not only the ones already generating.

In this project, I want to understand how well that approach works under different serving conditions. When does it help, when does it make little difference, and when can it actually hurt performance?

## How I'll test it

The plan is to run controlled experiments and compare the results rather than assume one configuration is always better.

- **Serving engine:** vLLM, running on a single GPU.
- **Model:** Qwen3-8B with an EAGLE-3 draft model. A smoke test ran this pair once on one A40, so it works, but that alone does not make it the right choice for the full experiments.
- **Configurations:** Standard generation without speculation, fixed K=1, fixed K=3, and Dynamic SD.
- **Workloads:** One workload to build the Dynamic SD lookup table and another held-out workload to test how well it generalizes.
- **Traffic:** Controlled request arrival rates using `vllm bench serve`, including Poisson arrivals.
- **Metrics:** Time to first token (TTFT), time per output token, end-to-end latency, throughput, goodput under a latency target, GPU memory usage, and speculative-token acceptance rate.

### Keeping the comparison fair

Before evaluating Dynamic SD, I'll define the comparison rules and record them in the lab notebook.

- Build the lookup table using one workload, then evaluate it on a separate workload.
- Compare Dynamic SD against the best fixed configuration tested, rather than an idealized best-case setting.
- Define what counts as a performance regression before looking at the results, while accounting for measurement variability.
- Record the exact software versions, model revisions, configurations, and experiment outputs so the results can be reproduced.

The goal isn't to prove that Dynamic SD is faster. It's to find out where it works well, where it doesn't, and why.

## Current progress

| Stage | Status |
|---|---|
| Set up the repo and start the `FINDINGS.md` notebook | Done. Entries get added as I verify things |
| Read how vLLM picks K for Dynamic SD | Code reading done (F-003). Runtime check at concurrency 32 still to do |
| GPU feasibility test | Done on one A40 (F-001, F-002). Concurrency 32 check and one benchmark oddity still open |
| Benchmark harness | Not started |
| Experiments and analysis | Not started |

## Keeping track of findings

I'll document the investigation in [`FINDINGS.md`](FINDINGS.md). Each entry will separate what I've verified from what I suspect, what I tested, what happened, and what needs to be checked again upstream.

This is an ongoing experiment, so I'll update the README as the implementation and results develop.