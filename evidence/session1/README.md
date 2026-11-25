# Session 1: validation run (raw evidence)

This folder holds the raw files from the first GPU session. It was a validation run. Its job was to check that the scripts work and to learn how the server behaves before the real experiment. It is not a set of findings by itself.

## What was run

- Date: 23 to 24 November 2026 (UTC).
- Hardware: one NVIDIA A40 (46,068 MiB) on RunPod. Driver 595.112.07, CUDA 13.2.
- Software: vLLM 0.33.0, torch 2.14.0+cu132, transformers 5.20.2, Python 3.12.3.
- Repo commit used on the pod: c314ef5.
- Models: Qwen/Qwen3-8B and RedHatAI/Qwen3-8B-speculator.eagle3. The exact revisions are in `logs/model-revisions.txt`. Every package version is in `logs/pip-freeze.txt`.
- Every command that was run is in `logs/commands.log`. The environment record is `logs/env.txt`.

## What the files are

- `results/validate/sweep_cNN.json`: baseline server (no speculation) at concurrency NN, from 1 to 64.
- `results/validate/static_c08.json`: static speculation with K=3 at concurrency 8.
- `results/validate/dyn_c04.json` and `dyn_c32.json`: Dynamic SD with the throwaway table `[[1,16,3],[17,64,1]]` at concurrency 4 and 32.
- `results/validate/metrics_load.json`: baseline at concurrency 16 with the server gauges sampled.
- `results/validate/len_<category>.json`: one run per SpecBench category at concurrency 16, to measure output lengths.
- `logs/bench_*.log`: the console output of each benchmark run. `logs/server_*.log`: the three server logs (baseline, static K=3, dynamic).
- `logs/snap_*.json`: speculative-decoding counters taken before and after runs. `logs/running_*.csv`: the running and waiting request gauge, sampled once a second.

## How to check the files

From this folder:

```
shasum -a 256 -c SHA256SUMS
```

Every line should say OK. The original archive downloaded from the pod was `session1.tgz`, with this SHA-256:

```
0e2d1d281d2ebf4fb5bb9bbd843e8caf5781b8827ac8a54ba7fd3e8044dbc5b8
```

## Known limits (read before using any number from here)

- Each configuration was run once. We have no measure of run-to-run noise yet.
- Prefix caching was on for every server (vLLM's default). All three server logs show `enable_prefix_caching=True`. A hit rate of about 52% was read from the live metrics page mid-session, but that reading was not saved in these files. Later runs reused earlier prompts, so first-token times are not comparable between a fresh server and a warm one.
- The dynamic server ran in CUDA graph mode `PIECEWISE`. The baseline and static servers ran `FULL_AND_PIECEWISE`. This is a known confound for any dynamic-versus-other comparison.
- The benchmark tool's "Peak concurrent requests" value is unreliable (it read 13 at a limit of 8). Use the server gauge in `running_*.csv` instead.
- The MT-bench categories (coding, extraction, humanities, math, reasoning, roleplay, stem, writing) have only 10 prompts each. Do not make per-category claims from them.
- The text-agreement check (outputs identical to the baseline) is not proof about speculation. Two baseline runs at different batch sizes matched each other on only 50% of prompts.
- transformers 5.20.2 was installed on the pod. The earlier token counts in FINDINGS (F-004) used 4.57.1.
- `scripts/setup.sh` did not install the benchmark extra on this run. It was installed by hand (`vllm[bench]==0.33.0`) after setup had finished. So `logs/pip-freeze.txt` does not list the 17 packages the extra added (among them pandas, datasets and scipy). Their versions are in `logs/bench_extra_dryrun.txt`, which is a dry run and not a record of what was installed. The script has since been changed to install the extra itself, so later freezes will include them.
