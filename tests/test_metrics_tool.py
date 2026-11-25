"""Checks for scripts/metrics_tool.py. CPU only, no server needed."""
import json

import pytest

import metrics_tool as mt

# A cut-down /metrics page with the names that can trip up a parser:
# _total suffixes, _created timestamps, per-position counters and several label sets.
PAGE = """\
# HELP vllm:spec_decode_num_drafts_total Number of spec decoding drafts.
# TYPE vllm:spec_decode_num_drafts_total counter
vllm:spec_decode_num_drafts_total{engine="0",model_name="Qwen/Qwen3-8B"} 100.0
vllm:spec_decode_num_drafts_total{engine="1",model_name="Qwen/Qwen3-8B"} 20.0
vllm:spec_decode_num_drafts_created{engine="0",model_name="Qwen/Qwen3-8B"} 1.795e+09
vllm:spec_decode_num_draft_tokens_total{engine="0",model_name="Qwen/Qwen3-8B"} 300.0
vllm:spec_decode_num_accepted_tokens_total{engine="0",model_name="Qwen/Qwen3-8B"} 130.0
vllm:spec_decode_num_accepted_tokens_per_pos_total{engine="0",model_name="Qwen/Qwen3-8B",position="0"} 66.0
vllm:num_requests_running{engine="0",model_name="Qwen/Qwen3-8B"} 12.0
vllm:num_requests_waiting{engine="0",model_name="Qwen/Qwen3-8B"} 0.0
vllm:kv_cache_usage_perc{engine="0",model_name="Qwen/Qwen3-8B"} NaN
"""


def test_total_adds_up_label_sets_and_ignores_created():
    assert mt.total(PAGE, "vllm:spec_decode_num_drafts") == 120.0


def test_total_does_not_count_the_per_position_counter():
    assert mt.total(PAGE, "vllm:spec_decode_num_accepted_tokens") == 130.0


def test_total_reads_gauges_without_total_suffix():
    assert mt.total(PAGE, "vllm:num_requests_running") == 12.0


def test_total_of_a_missing_metric_is_none():
    assert mt.total(PAGE, "vllm:not_there") is None


def test_total_handles_nan():
    value = mt.total(PAGE, "vllm:kv_cache_usage_perc")
    assert value != value  # NaN is not equal to itself


# ---------------------------------------------------------------- delta and the F-003 rule
def run_delta(tmp_path, capsys, before, after, prediction=True):
    a, b = tmp_path / "a.json", tmp_path / "b.json"
    a.write_text(json.dumps(before))
    b.write_text(json.dumps(after))
    args = ["delta", str(a), str(b)] + (["--prediction"] if prediction else [])
    mt.main(args)
    return capsys.readouterr().out


def test_delta_session1_numbers_confirm_the_prediction(tmp_path, capsys):
    # The F-003 run: 16,362 drafts and 17,194 draft tokens on a fresh server.
    out = run_delta(tmp_path, capsys,
                    {"drafts": 0.0, "draft_tokens": 0.0, "accepted_tokens": 0.0},
                    {"drafts": 16362.0, "draft_tokens": 17194.0, "accepted_tokens": 11081.0})
    assert "draft tokens per draft: 1.05" in out
    assert "CONFIRMED" in out


@pytest.mark.parametrize(("tokens", "verdict"), [(3.0, "WRONG"), (2.0, "IN BETWEEN"), (1.49, "CONFIRMED")])
def test_delta_prediction_thresholds(tmp_path, capsys, tokens, verdict):
    out = run_delta(tmp_path, capsys,
                    {"drafts": 0.0, "draft_tokens": 0.0, "accepted_tokens": 0.0},
                    {"drafts": 100.0, "draft_tokens": 100.0 * tokens, "accepted_tokens": 10.0})
    assert verdict in out


def test_delta_stops_when_counters_are_missing(tmp_path, capsys):
    with pytest.raises(SystemExit):
        run_delta(tmp_path, capsys,
                  {"drafts": None, "draft_tokens": None, "accepted_tokens": None},
                  {"drafts": None, "draft_tokens": 5.0, "accepted_tokens": 1.0})


# ---------------------------------------------------------------- gauge summary
def test_busy_mean_leaves_out_idle_samples():
    rows = [(0, 0, 0.0, 0.0), (1, 1, 16.0, 0.0), (2, 2, 16.0, 2.0), (3, 3, 0.0, 0.0)]
    lines = mt.gauge_summary(rows)
    assert "mean 8.0" in lines[0]
    assert "mean 16.0 over 2 samples" in lines[1]
    assert "max 2" in lines[2]


def test_watch_warns_when_it_hits_the_time_limit(tmp_path, capsys, monkeypatch):
    monkeypatch.setattr(mt, "fetch", lambda: PAGE)
    out = tmp_path / "g.csv"
    mt.main(["watch", "--stop-file", str(tmp_path / "never"), "--out", str(out), "--every", "0.01", "--max-seconds", "0.05"])
    text = capsys.readouterr().out
    assert "WARNING: sampling stopped" in text
    header, first = out.read_text().splitlines()[:2]
    assert header == "seconds,running,waiting,unix_time"
    assert first.split(",")[1] == "12.0"
