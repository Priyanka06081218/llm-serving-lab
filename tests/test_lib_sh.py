"""Checks for scripts/lib.sh that need only bash. CPU only, no server needed."""
import pathlib
import subprocess

LIB = pathlib.Path(__file__).resolve().parent.parent / "scripts" / "lib.sh"

# Arguments shaped like the real benchmark command: JSON with quotes and braces,
# a comma list, and one argument with a space in it.
TRICKY = [
    "vllm", "bench", "serve",
    "--extra-body", '{"chat_template_kwargs":{"enable_thinking":false}}',
    "--percentile-metrics", "ttft,tpot,itl,e2el",
    "--label", "two words",
]


def bash(script, work):
    return subprocess.run(["bash", "-c", script], check=True, capture_output=True, text=True,
                          env={"WORK": str(work), "PATH": "/usr/bin:/bin"})


def test_logged_command_can_be_replayed_exactly(tmp_path):
    args = " ".join("'" + a.replace("'", "'\\''") + "'" for a in TRICKY)
    bash(f'source "{LIB}"; log_cmd {args}', tmp_path)
    line = (tmp_path / "logs" / "commands.log").read_text().splitlines()[0]
    assert line.startswith("[")
    logged = line.split("] ", 1)[1]
    # Replay the logged line with printf standing in for the real program,
    # one argument per line, and compare with what went in.
    replay = bash(f"set -- {logged}; printf '%s\\n' \"$@\"", tmp_path).stdout.splitlines()
    assert replay == TRICKY


def test_port_and_model_are_exported_with_defaults(tmp_path):
    out = bash(f'source "{LIB}"; env | grep "^LAB_" | sort', tmp_path).stdout
    assert "LAB_PORT=8000" in out
    assert "LAB_MODEL=Qwen/Qwen3-8B" in out


def test_port_can_be_overridden(tmp_path):
    out = bash(f'LAB_PORT=9001 source "{LIB}"; echo "$PORT"', tmp_path).stdout.strip()
    assert out == "9001"


def test_prefix_caching_is_off_by_default(tmp_path):
    out = bash(f'source "{LIB}"; cache_flag', tmp_path).stdout.strip()
    assert out == "--disable-prefix-caching"


def test_prefix_caching_can_be_turned_on_to_repeat_session_1(tmp_path):
    out = bash(f'LAB_PREFIX_CACHING=on; source "{LIB}"; cache_flag', tmp_path).stdout.strip()
    assert out == "--enable-prefix-caching"


def test_prefix_caching_setting_must_be_on_or_off(tmp_path):
    result = subprocess.run(["bash", "-c", f'LAB_PREFIX_CACHING=true; source "{LIB}"; cache_flag'],
                            capture_output=True, text=True,
                            env={"WORK": str(tmp_path), "PATH": "/usr/bin:/bin"})
    assert result.returncode == 1
    assert result.stdout == ""
    assert "must be 'on' or 'off'" in result.stderr
