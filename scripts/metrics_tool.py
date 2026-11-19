#!/usr/bin/env python3
"""Small reader for the server's /metrics page (Prometheus text format).

  list                      print every metric name the server exposes
  snap OUT.json             save the speculative-decoding counters right now
  delta A.json B.json       counters between two snapshots, plus tokens per draft
  watch --stop-file F ...   sample the running/waiting request gauges once a second
"""
import argparse
import json
import os
import re
import sys
import time
import urllib.request

URL = "http://localhost:8000/metrics"
NUM = r"([-+0-9.eE]+|nan|NaN|inf|Inf)"


def fetch():
    with urllib.request.urlopen(URL, timeout=10) as r:
        return r.read().decode()


def total(text, name):
    """Add up every label set of one metric. Returns None if the metric is absent."""
    pattern = re.compile(r"^" + re.escape(name) + r"(_total)?(\{[^}]*\})?\s+" + NUM + r"\s*$", re.M)
    values = [float(m.group(3)) for m in pattern.finditer(text)]
    return sum(values) if values else None


def cmd_list(_):
    names = set()
    for line in fetch().splitlines():
        if line and not line.startswith("#"):
            names.add(re.split(r"[{ ]", line, maxsplit=1)[0])
    for n in sorted(names):
        print(n)


COUNTERS = {
    "drafts": "vllm:spec_decode_num_drafts",
    "draft_tokens": "vllm:spec_decode_num_draft_tokens",
    "accepted_tokens": "vllm:spec_decode_num_accepted_tokens",
}


def cmd_snap(a):
    text = fetch()
    snap = {k: total(text, v) for k, v in COUNTERS.items()}
    json.dump(snap, open(a.out, "w"))
    print("snapshot saved:", snap)


def cmd_delta(a):
    before, after = json.load(open(a.before)), json.load(open(a.after))
    if any(after[k] is None for k in COUNTERS):
        print("The speculative-decoding counters were not found on /metrics.")
        print("Look at the name list (metrics_tool.py list) and fix COUNTERS in this file.")
        sys.exit(1)
    d = {k: after[k] - (before[k] or 0) for k in COUNTERS}
    print(f"== {a.label} ==")
    print(f"drafts: {d['drafts']:.0f}   draft tokens: {d['draft_tokens']:.0f}   accepted tokens: {d['accepted_tokens']:.0f}")
    if d["drafts"] <= 0:
        print("No drafts were made in this window. Nothing to divide.")
        sys.exit(1)
    per_draft = d["draft_tokens"] / d["drafts"]
    print(f"draft tokens per draft: {per_draft:.2f}")
    if d["draft_tokens"] > 0:
        print(f"acceptance rate (accepted / draft tokens): {d['accepted_tokens'] / d['draft_tokens']:.2f}")
    if a.prediction:
        # The rule written down in FINDINGS.md (F-003) before this run.
        if per_draft < 1.5:
            print("F-003 prediction: CONFIRMED (below 1.5).")
        elif per_draft >= 2.5:
            print("F-003 prediction: WRONG (close to 3). Re-read the scheduler code before going on.")
        else:
            print("F-003 prediction: IN BETWEEN (1.5 to 2.5). Investigate before trusting the table lookup.")


def cmd_watch(a):
    start = time.time()
    rows = []
    while not os.path.exists(a.stop_file) and time.time() - start < a.max_seconds:
        try:
            text = fetch()
            rows.append((time.time() - start, total(text, "vllm:num_requests_running"), total(text, "vllm:num_requests_waiting")))
        except Exception as e:  # server busy for a moment: skip this sample
            print("sample skipped:", e)
        time.sleep(a.every)
    with open(a.out, "w") as f:
        f.write("seconds,running,waiting\n")
        for t, r, w in rows:
            f.write(f"{t:.1f},{r},{w}\n")
    running = [r for _, r, _ in rows if r is not None]
    waiting = [w for _, _, w in rows if w is not None]
    print(f"samples: {len(rows)} (saved to {a.out})")
    if not running:
        print("vllm:num_requests_running was not found. Check the name list.")
        return
    print(f"running requests: min {min(running):.0f}, mean {sum(running)/len(running):.1f}, max {max(running):.0f}")
    if waiting:
        print(f"waiting requests: min {min(waiting):.0f}, mean {sum(waiting)/len(waiting):.1f}, max {max(waiting):.0f}")


p = argparse.ArgumentParser()
sub = p.add_subparsers(dest="cmd", required=True)
sub.add_parser("list").set_defaults(fn=cmd_list)
s = sub.add_parser("snap"); s.add_argument("out"); s.set_defaults(fn=cmd_snap)
s = sub.add_parser("delta"); s.add_argument("before"); s.add_argument("after")
s.add_argument("--label", default="window"); s.add_argument("--prediction", action="store_true"); s.set_defaults(fn=cmd_delta)
s = sub.add_parser("watch"); s.add_argument("--stop-file", required=True); s.add_argument("--out", required=True)
s.add_argument("--every", type=float, default=1.0); s.add_argument("--max-seconds", type=float, default=900); s.set_defaults(fn=cmd_watch)
args = p.parse_args()
args.fn(args)
