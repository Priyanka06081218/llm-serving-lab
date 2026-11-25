# Prints prompt lengths in tokens for each Spec-Bench category,
# using the same chat template the server will apply (thinking off).
import collections
import hashlib
import json
import math
import statistics
import sys

import transformers
from transformers import AutoTokenizer

MODEL = "Qwen/Qwen3-8B"
MAX_MODEL_LEN = 4096
OUTPUT_LEN = 256
TOKENIZER_REVISION = "b968826d9c46dd6066d109eabc6255188de91218"
# Must match DATASET_SHA256 in scripts/setup.sh
EXPECTED_SHA256 = "4b6d33e79484f9841c487ee87d1cf6aa8c6066f61d5d482ff09e5a007fafdf04"

path = sys.argv[1]

with open(path, "rb") as f:
    actual_sha256 = hashlib.sha256(f.read()).hexdigest()
if actual_sha256 != EXPECTED_SHA256:
    raise ValueError(f"Dataset checksum mismatch: {actual_sha256}")
print("Dataset SHA-256 verified:", actual_sha256)

print("transformers version:", transformers.__version__)
print("tokenizer repo revision (pinned):", TOKENIZER_REVISION)

tok = AutoTokenizer.from_pretrained(MODEL, revision=TOKENIZER_REVISION)

by_category = collections.defaultdict(list)
with open(path) as f:
    for line in f:
        line = line.strip()
        if not line:
            continue
        row = json.loads(line)
        text = tok.apply_chat_template(
            [{"role": "user", "content": row["turns"][0]}],
            add_generation_prompt=True,
            tokenize=False,
            enable_thinking=False,
        )
        by_category[row["category"]].append(
            len(tok(text, add_special_tokens=False).input_ids)
        )

budget = MAX_MODEL_LEN - OUTPUT_LEN
for name, lengths in by_category.items():
    lengths.sort()
    p95 = lengths[math.ceil(0.95 * len(lengths)) - 1]
    too_long = sum(1 for n in lengths if n > budget)
    print(
        f"{name:15} n={len(lengths):3}  min={lengths[0]:5}  "
        f"median={int(statistics.median(lengths)):5}  p95={p95:5}  "
        f"max={lengths[-1]:5}  over_budget={too_long}"
    )
