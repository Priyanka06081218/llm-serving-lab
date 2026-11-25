# LLM Serving Lab

**Status: In progress. Setup, a smoke test and a first validation session are done. The benchmark harness is next. No experiment results yet.**

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
- **Metrics:** Time to first token (TTFT), time per output token, end-to-end latency, throughput, goodput under a latency target, GPU memory usage (how to measure it over time is not decided yet), and speculative-token acceptance rate.

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
| Read how vLLM picks K for Dynamic SD | Code reading done (F-003). The runtime check at concurrency 32 confirmed the prediction (F-003, result) |
| GPU feasibility test | Done on one A40 (F-001, F-002) |
| Validation session: scripts, baseline capacity, static K=3, F-003 test | Done (F-005). Single runs, so no conclusions about Dynamic SD yet. Raw files and checksums in [`evidence/session1/`](evidence/session1/) |
| Benchmark harness | Not started |
| Experiments and analysis | Not started |

## What is in this repo

| Path | What it does |
|---|---|
| [`FINDINGS.md`](FINDINGS.md) | The lab notebook. Each entry says what was found, how it was checked and what is still open |
| [`configs/experiment.yaml`](configs/experiment.yaml) | The planned experiment settings. The harness that will read it is not written yet |
| [`scripts/setup.sh`](scripts/setup.sh) | Prepares a fresh GPU pod: installs vLLM 0.33.0 with its benchmark extra, downloads the dataset (checked by checksum) and both models, and records the machine and versions |
| [`scripts/lib.sh`](scripts/lib.sh) | Shared shell helpers: start and stop the server, run one benchmark, keep a log of every command, and refuse new work after the session time limit |
| [`scripts/validate_gpu.sh`](scripts/validate_gpu.sh) | The steps of the first GPU session, run one at a time |
| [`scripts/metrics_tool.py`](scripts/metrics_tool.py) | Reads the server's `/metrics` page: speculative-decoding counters, and the running and waiting request counts |
| [`scripts/summarize_bench.py`](scripts/summarize_bench.py) | Turns benchmark result files into one table, and checks whether two runs gave the same text |
| [`scripts/chat_check.py`](scripts/chat_check.py) | Checks that the server applies the chat template exactly once |
| [`scripts/check_prompt_lengths.py`](scripts/check_prompt_lengths.py) | Counts prompt tokens per category, on a laptop (F-004) |
| [`tests/`](tests/) | Checks for the Python tools and the command log. They run on a laptop |
| [`evidence/session1/`](evidence/session1/) | Raw files from the first GPU session, with checksums |
| [`docs/`](docs/) | Smoke test notes and the prompt length output |

## Running a GPU session

This needs a RunPod pod with one A40 and a 100 GB container disk. The container disk is erased when the pod stops, so the results must be downloaded before the pod is stopped.

In the pod's terminal, type one line at a time:

```
cd /workspace
git clone https://github.com/Priyanka06081218/llm-serving-lab.git
cd llm-serving-lab
bash scripts/setup.sh
```

Setup took about 5 minutes in session 1. Then run the session steps one at a time, and read each result before the next one. Run `bash scripts/validate_gpu.sh` with no step name to see the list. Every command a step runs is saved in `/workspace/logs/commands.log`, and the scripts refuse to start new work 3 hours after setup began.

Before stopping the pod, stop the server and pack the results:

```
bash scripts/validate_gpu.sh stop
cd /workspace && tar czf session.tgz logs results && sha256sum session.tgz && ls -lh session.tgz
```

Download `session.tgz` through JupyterLab. In session 1 no link for it could be found on the Connect tab, but it was running: open `https://<pod-id>-8888.proxy.runpod.net/` (the terminal's address with 8888 in place of its port), go to the `workspace` folder, right-click the file and choose Download. If it asks for a token, `ps aux | grep -o 'token=[^ ]*'` on the pod prints it. On the Mac, `shasum -a 256 session.tgz` must print the same checksum as the pod did. Only then terminate the pod.

## Reading the results

Each benchmark run writes `results/validate/NAME.json` (vLLM's result format, with per-request details) and its console output to `logs/bench_NAME.log`.

`python scripts/summarize_bench.py FILE ...` prints one row per run:

| Column | Meaning |
|---|---|
| ok, fail | Requests that finished, and requests that failed |
| req/s, tok/s | Requests finished per second, and output tokens per second |
| TTFT50, TTFT99 | Time to first token in ms, median and 99th percentile (vLLM's numbers) |
| TPOT50, TPOT99 | Time per output token after the first, in ms (vLLM's numbers). Checking vLLM's exact formula under speculation is still an open item |
| len50, len95, max | Output length in tokens: median, 95th percentile and longest |
| cap% | Share of outputs that reached the output limit (256 tokens) |

`python scripts/summarize_bench.py --compare A.json B.json` counts how many prompts got identical text in two runs. At temperature 0, batching alone changed about half of the texts in session 1 (F-005), so a mismatch is not proof of a problem.

`python scripts/metrics_tool.py delta BEFORE.json AFTER.json` gives draft tokens per draft and the acceptance rate between two counter snapshots.

One run is not a result. FINDINGS says for each number what it shows and what it does not.

## Keeping track of findings

I'll document the investigation in [`FINDINGS.md`](FINDINGS.md). Each entry will separate what I've verified from what I suspect, what I tested, what happened, and what needs to be checked again upstream.

This is an ongoing experiment, so I'll update the README as the implementation and results develop.

## Checking the code

The helper scripts have tests that run on a laptop, with no GPU and no model download:

```
pip install -r requirements-dev.txt
python -m pytest tests
```

GitHub runs the same tests, plus shell and lint checks and the evidence checksums, on every push (`.github/workflows/checks.yml`). Anything that needs a GPU is run by hand, one step at a time, with `scripts/validate_gpu.sh`.
