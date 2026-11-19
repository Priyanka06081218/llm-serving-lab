# Sends one raw prompt to the running server through the chat endpoint, thinking off,
# and checks that the server applied the chat template exactly once.
import json
import sys
import urllib.request

from transformers import AutoTokenizer

MODEL = "Qwen/Qwen3-8B"
URL = "http://localhost:8000/v1/chat/completions"
PROMPT = "What is the capital of France? Answer in one sentence."

body = {
    "model": MODEL,
    "messages": [{"role": "user", "content": PROMPT}],
    "max_completion_tokens": 64,
    "temperature": 0,
    "chat_template_kwargs": {"enable_thinking": False},
}
request = urllib.request.Request(
    URL,
    data=json.dumps(body).encode(),
    headers={"Content-Type": "application/json"},
)
with urllib.request.urlopen(request) as response:
    reply = json.load(response)

answer = reply["choices"][0]["message"]["content"]
server_prompt_tokens = reply["usage"]["prompt_tokens"]

# What the token count would be if the template were applied once, or twice.
tok = AutoTokenizer.from_pretrained(MODEL)
once = tok.apply_chat_template(
    [{"role": "user", "content": PROMPT}],
    add_generation_prompt=True,
    tokenize=False,
    enable_thinking=False,
)
twice = tok.apply_chat_template(
    [{"role": "user", "content": once}],
    add_generation_prompt=True,
    tokenize=False,
    enable_thinking=False,
)
count_once = len(tok(once, add_special_tokens=False).input_ids)
count_twice = len(tok(twice, add_special_tokens=False).input_ids)

print("answer:", answer)
print("prompt tokens reported by the server:", server_prompt_tokens)
print("tokens if the template is applied once :", count_once)
print("tokens if the template is applied twice:", count_twice)
print("answer contains <think> text:", "<think>" in answer)
if server_prompt_tokens == count_once:
    print("RESULT: the template was applied once. Good.")
elif server_prompt_tokens == count_twice:
    print("RESULT: the template was applied TWICE. Do not run benchmarks yet.")
    sys.exit(1)
else:
    print("RESULT: the count matches neither. Investigate before benchmarking.")
    sys.exit(1)
