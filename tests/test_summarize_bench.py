"""Checks for scripts/summarize_bench.py. CPU only, no server needed."""
import json

import pytest

import summarize_bench as sb


# ---------------------------------------------------------------- percentiles
def test_median_of_even_count_is_the_mean_of_the_two_middle_values():
    # The old nearest-rank version returned 55 here (see F-005).
    assert sb.pct([10, 55, 83, 200], 50) == 69


def test_median_of_odd_count_is_the_middle_value():
    assert sb.pct([3, 1, 2], 50) == 2


def test_percentiles_match_numpy_linear_method():
    np = pytest.importorskip("numpy")
    values = [7, 1, 256, 23, 50, 70, 256, 12, 199, 3, 88]
    for p in (0, 10, 50, 90, 95, 99, 100):
        assert sb.pct(values, p) == pytest.approx(float(np.percentile(values, p)))


def test_percentile_of_nothing_is_none():
    assert sb.pct([], 50) is None


# ---------------------------------------------------------------- failed requests
def test_errors_field_decides_which_requests_count():
    assert sb.good_mask({"errors": ["", "timeout", ""], "output_lens": [5, 0, 7]}) == [True, False, True]


def test_without_errors_field_an_empty_output_counts_as_failed():
    assert sb.good_mask({"output_lens": [5, 0, 7]}) == [True, False, True]


# ---------------------------------------------------------------- compare
def write(tmp_path, name, **fields):
    path = tmp_path / name
    path.write_text(json.dumps(fields))
    return str(path)


def test_compare_counts_identical_texts(tmp_path, capsys):
    a = write(tmp_path, "a.json", input_lens=[3, 4, 5], errors=["", "", ""], generated_texts=["x", "yy", "zzz"])
    b = write(tmp_path, "b.json", input_lens=[3, 4, 5], errors=["", "", ""], generated_texts=["x", "yQ", "zzz"])
    assert sb.compare(a, b) == 2
    out = capsys.readouterr().out
    assert "identical text: 2 (67%)" in out
    assert "earliest 1" in out


def test_compare_skips_failed_requests(tmp_path, capsys):
    a = write(tmp_path, "a.json", input_lens=[3, 4], errors=["", "boom"], generated_texts=["x", ""])
    b = write(tmp_path, "b.json", input_lens=[3, 4], errors=["", ""], generated_texts=["x", "y"])
    assert sb.compare(a, b) == 1
    assert "prompts compared (both ok): 1 of 2" in capsys.readouterr().out


def test_compare_refuses_different_prompts(tmp_path, capsys):
    a = write(tmp_path, "a.json", input_lens=[3, 4], errors=["", ""], generated_texts=["x", "y"])
    b = write(tmp_path, "b.json", input_lens=[3, 9], errors=["", ""], generated_texts=["x", "y"])
    assert sb.compare(a, b) is None
    assert "not valid" in capsys.readouterr().out


def test_compare_explains_missing_texts(tmp_path, capsys):
    a = write(tmp_path, "a.json", input_lens=[3])
    b = write(tmp_path, "b.json", input_lens=[3], generated_texts=["x"])
    assert sb.compare(a, b) is None
    assert "--save-detailed" in capsys.readouterr().out


def test_compare_needs_exactly_two_files(tmp_path):
    a = write(tmp_path, "a.json", input_lens=[3], generated_texts=["x"])
    with pytest.raises(SystemExit) as stop:
        sb.main(["--compare", a])
    assert stop.value.code == 2


def test_first_difference_when_one_text_is_a_prefix_of_the_other():
    assert sb.first_difference("abc", "abcdef") == 3


# ---------------------------------------------------------------- summary table
def test_summary_row_uses_true_median_and_cap_share(tmp_path, capsys):
    path = write(tmp_path, "run.json", completed=4, output_lens=[10, 55, 83, 256], errors=["", "", "", ""],
                 request_throughput=1.0, output_throughput=100.0, median_ttft_ms=50.0, p99_ttft_ms=90.0,
                 median_tpot_ms=20.0, p99_tpot_ms=30.0)
    sb.summarize([path], cap=256)
    row = capsys.readouterr().out.splitlines()[1].split()
    assert row[0] == "run"
    assert row[9] == "69"     # len50: true median of 10, 55, 83, 256
    assert row[-1] == "25"    # cap%: one of four outputs reached 256
