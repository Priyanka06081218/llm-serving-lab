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


# ---------------------------------------------------------------- the real server command
def logged_server_commands(tmp_path):
    """Run start_server for three arms with a stand-in `vllm`, and return the commands it logged."""
    fake = tmp_path / "fakebin"
    fake.mkdir()
    for name, body in {"vllm": "exec sleep 30", "curl": "exit 0", "nvidia-smi": "echo 0"}.items():
        script = fake / name
        script.write_text("#!/bin/bash\n" + body + "\n")
        script.chmod(0o755)
    steps = (
        'start_server off; stop_server; '
        'start_server k3 --speculative-config "$SPEC_K3"; stop_server; '
        'start_server dynamic --speculative-config "$SPEC_DYNAMIC"; stop_server'
    )
    subprocess.run(["bash", "-c", f'source "{LIB}"; {steps}'], check=True, capture_output=True, text=True,
                   env={"WORK": str(tmp_path), "PATH": f"{fake}:/usr/bin:/bin"})
    lines = (tmp_path / "logs" / "commands.log").read_text().splitlines()
    return [line.split("] ", 1)[1] for line in lines]


def test_every_arm_gets_the_same_controls(tmp_path):
    commands = logged_server_commands(tmp_path)
    assert len(commands) == 3
    for command in commands:
        assert "--max-num-batched-tokens 2048" in command
        assert "--disable-prefix-caching" in command
        assert "VLLM_USE_V2_MODEL_RUNNER=1" in command


def test_arms_differ_only_in_the_speculative_config(tmp_path):
    off, k3, dynamic = logged_server_commands(tmp_path)
    assert "--speculative-config" not in off
    assert off.rstrip() == k3.split("--speculative-config")[0].rstrip()
    assert off.rstrip() == dynamic.split("--speculative-config")[0].rstrip()


def test_model_runner_is_v2_by_default_and_always_set(tmp_path):
    assert bash(f'source "{LIB}"; runner_env', tmp_path).stdout.strip() == "VLLM_USE_V2_MODEL_RUNNER=1"


def test_model_runner_can_go_back_to_v1_to_repeat_session_1(tmp_path):
    out = bash(f'LAB_RUNNER=v1; source "{LIB}"; runner_env', tmp_path).stdout.strip()
    assert out == "VLLM_USE_V2_MODEL_RUNNER=0"


def test_model_runner_setting_must_be_v1_or_v2(tmp_path):
    result = subprocess.run(["bash", "-c", f'LAB_RUNNER=V3; source "{LIB}"; runner_env'],
                            capture_output=True, text=True,
                            env={"WORK": str(tmp_path), "PATH": "/usr/bin:/bin"})
    assert result.returncode == 1
    assert result.stdout == ""
    assert "must be 'v1' or 'v2'" in result.stderr
