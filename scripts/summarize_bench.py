#!/usr/bin/env python3
"""Reads files written by `vllm bench serve --save-result --save-detailed`.

  summarize_bench.py FILE [FILE ...]        one row per file
  summarize_bench.py --compare A.json B.json   do two runs give the same text?

The TTFT and TPOT columns are vLLM's own numbers from the result file.
The output-length columns (len50, len95) are computed here with the same
percentile method vLLM uses (numpy's default, linear interpolation), so
len50 is the true median.
"""
import argparse
import json
import math
import os
import sys


def pct(values, p):
    """Percentile with linear interpolation, the same as numpy.percentile's default."""
    if not values:
        return None
    s = sorted(values)
    pos = p / 100 * (len(s) - 1)
    lo, hi = math.floor(pos), math.ceil(pos)
    return s[lo] + (s[hi] - s[lo]) * (pos - lo)


def fmt(x, spec=".0f"):
    return "-" if x is None else format(x, spec)


def load(path):
    with open(path) as f:
        return json.load(f)


def good_mask(d):
    """True for every request that finished without an error."""
    errors, lens = d.get("errors"), d.get("output_lens")
    if errors is not None:
        return [not e for e in errors]
    return [n > 0 for n in lens]


def summarize(paths, cap):
    head = f"{'run':<26}{'ok':>5}{'fail':>5}{'req/s':>7}{'tok/s':>8}{'TTFT50':>8}{'TTFT99':>8}{'TPOT50':>8}{'TPOT99':>8}{'len50':>7}{'len95':>7}{'max':>6}{'cap%':>6}"
    print(head)
    first = None
    for path in paths:
        d = load(path)
        first = first or d
        name = os.path.basename(path).removesuffix(".json")
        lens = d.get("output_lens")
        mask = good_mask(d) if lens else None
        good = [n for n, ok in zip(lens, mask, strict=True) if ok] if lens else []
        failed = (len(mask) - sum(mask)) if mask else None
        at_cap = 100 * sum(n >= cap for n in good) / len(good) if good else None
        print(f"{name:<26}{fmt(d.get('completed')):>5}{fmt(failed):>5}"
              f"{fmt(d.get('request_throughput'), '.2f'):>7}{fmt(d.get('output_throughput')):>8}"
              f"{fmt(d.get('median_ttft_ms')):>8}{fmt(d.get('p99_ttft_ms')):>8}"
              f"{fmt(d.get('median_tpot_ms'), '.1f'):>8}{fmt(d.get('p99_tpot_ms'), '.1f'):>8}"
              f"{fmt(pct(good, 50)):>7}{fmt(pct(good, 95)):>7}{fmt(max(good) if good else None):>6}{fmt(at_cap):>6}")
    if first:
        print()
        print("keys in the result file:", ", ".join(sorted(first.keys())))
        odd = {k: first[k] for k in first if "concurren" in k.lower()}
        if odd:
            print("concurrency-related values in the first file:", odd)
        if "output_lens" not in first:
            print("WARNING: no per-request details found. --save-detailed did not take effect.")
        elif max(first["output_lens"]) > cap:
            print(f"WARNING: an output is longer than {cap} tokens. The output-length cap is not working.")


def first_difference(x, y):
    """Character position where two texts first differ."""
    return next((j for j, (a, b) in enumerate(zip(x, y)) if a != b), min(len(x), len(y)))


def compare(a_path, b_path):
    """Print how many prompts got identical text in two runs. Returns the identical count, or None."""
    a, b = load(a_path), load(b_path)
    for path, d in ((a_path, a), (b_path, b)):
        if "generated_texts" not in d or "input_lens" not in d:
            print(f"{path} has no per-request texts. Re-run the benchmark with --save-detailed.")
            return None
    if a["input_lens"] != b["input_lens"]:
        print("The two runs did not use the same prompts (input lengths differ). Comparing them is not valid.")
        return None
    ta, tb = a["generated_texts"], b["generated_texts"]
    ma, mb = good_mask(a), good_mask(b)
    both = [i for i in range(len(ta)) if ma[i] and mb[i]]
    same = [i for i in both if ta[i] == tb[i]]
    print("Same prompts? The per-request input lengths match, in order. That is a strong sign, not proof.")
    print(f"prompts compared (both ok): {len(both)} of {len(ta)}")
    print(f"identical text: {len(same)} ({100 * len(same) / max(len(both), 1):.0f}%)")
    diffs = [first_difference(ta[i], tb[i]) for i in both if ta[i] != tb[i]]
    if diffs:
        print(f"where texts first differ (character position): median {pct(diffs, 50):.0f}, earliest {min(diffs)}")
        print("Differences are recorded, not assumed to be bugs. Float rounding can change a token choice.")
    return len(same)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("files", nargs="+")
    ap.add_argument("--compare", action="store_true")
    ap.add_argument("--cap", type=int, default=256)
    a = ap.parse_args(argv)
    if a.compare:
        if len(a.files) != 2:
            print("--compare needs exactly two result files.")
            sys.exit(2)
        compare(a.files[0], a.files[1])
    else:
        summarize(a.files, a.cap)


if __name__ == "__main__":
    main()
