# Smoke test notes

These are my notes from the first time I used a GPU for this project, on 4 Nov 2026. I just wanted to see that everything runs before I build on top of it. It was not a speed test, so please don't read those numbers as results. What I actually measured is in F-001 and F-002 in `FINDINGS.md`.

## What I used

- GPU: one NVIDIA A40 (48 GB), rented on RunPod Secure Cloud with their PyTorch 2.8.0 template
- vLLM 0.33.0, installed in a Python 3.12 virtual environment made with uv
- Model: Qwen/Qwen3-8B, with the EAGLE-3 draft model RedHatAI/Qwen3-8B-speculator.eagle3
- Cost: about $0.69 in total

## How I installed it

    pip install uv
    uv venv --python 3.12 vllm-env
    source vllm-env/bin/activate
    uv pip install vllm==0.33.0 --torch-backend=auto
    vllm download-kernels

`--torch-backend=auto` lets uv pick the PyTorch build that fits the GPU driver. The last step, `vllm download-kernels`, is in vLLM's install docs.

## The servers

I started three servers, one at a time, in the background, and saved their output to log files. `VLLM_USE_V2_MODEL_RUNNER=0` is in the command because the Dynamic SD examples in the vLLM docs use it.

No speculation (the baseline):

    nohup env VLLM_USE_V2_MODEL_RUNNER=0 vllm serve Qwen/Qwen3-8B --max-model-len 4096 --gpu-memory-utilization 0.90 > /root/server_off.log 2>&1 &

Fixed K=3 is the same command, plus this:

    --speculative-config '{"model":"RedHatAI/Qwen3-8B-speculator.eagle3","num_speculative_tokens":3,"method":"eagle3"}'

Dynamic SD is the same as fixed K=3, plus `--max-num-seqs 64`. The table goes inside the same speculative config:

    "num_speculative_tokens_per_batch_size":[[1,16,3],[17,64,1]]

Each entry reads [fewest requests, most requests, K]. So when a step has 1 to 16 requests, K is 3, and when it has 17 to 64 requests, K is 1.

I sent a quick test request to the servers to check they were answering. It used temperature 0, and thinking was turned off with `chat_template_kwargs`. Then I ran `vllm bench serve` against the Dynamic SD server with 4 requests at a time (concurrency 4). The numbers are in F-002.

## What I forgot to save

I didn't save the exact `vllm bench serve` command or the test request, and the pod's terminal history is gone because the pod's disk is erased when it stops. Next session I'll re-run the concurrency 4 benchmark and write the exact command here. From now on, every command I run goes into a script or a log file.

## Things that went wrong

- **The disk filled up.** The pod had a 50 GB disk and ran out of space while the model was downloading ("No space left on device"). The uv cache was using about 18 GiB, and `uv cache clean` freed 17.9 GiB. For the real runs I plan to use about 100 GB or a persistent volume. Because the disk is erased when the pod stops, a new pod needs a full reinstall, which takes about 10 minutes.
- **Pasting into the web terminal.** When I pasted several commands at once, they got merged into one line (`activatecd /root`) and the environment didn't activate. After that I typed the commands one at a time.
- **Sampling settings.** The model has its own `generation_config.json`, and vLLM uses it instead of its own defaults (temperature 0.6, top_k 20, top_p 0.95). Adding `--generation-config vllm` turns that off. For the real runs I need to set sampling myself, so every configuration uses the same settings.
- **Stopping the pod.** After I stopped the pod, the console showed $0.00/hr.
