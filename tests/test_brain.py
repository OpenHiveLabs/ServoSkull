"""Tests for software/skull_core/brain.py. No test runs the real `claude`: a fake one
(tests/fake_claude.py) replays a recorded, scrubbed stream (tests/fixtures/brain_stream_ok.jsonl)."""
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import pytest

from software.skull_core import brain as br

REPO_ROOT = Path(__file__).resolve().parents[1]
FAKE = REPO_ROOT / "tests" / "fake_claude.py"
FIXTURE = REPO_ROOT / "tests" / "fixtures" / "brain_stream_ok.jsonl"
FIXTURE_REPLY = ("On a clear day, the sky looks blue because air molecules scatter shorter blue "
                 "wavelengths of sunlight more than longer red ones.")


def fixture_events():
    return [json.loads(line) for line in FIXTURE.read_text().splitlines()]


def write_stream(path: Path, events) -> Path:
    path.write_text("".join(json.dumps(e) + "\n" for e in events))
    return path


@pytest.fixture
def shim(tmp_path) -> Path:
    """An executable that runs the fake claude with this Python, like a real binary on PATH."""
    path = tmp_path / "claude"
    path.write_text(f'#!/bin/sh\nexec "{sys.executable}" "{FAKE}" "$@"\n')
    path.chmod(0o755)
    return path


@pytest.fixture
def brain_dir(tmp_path) -> Path:
    path = tmp_path / "skull-brain"
    path.mkdir()
    return path


def make_brain(shim, brain_dir, tmp_path, fake_env=None, timeout_s=5.0, state=lambda: "voice state: listening",
               extra_env=None):
    environ = dict(os.environ)
    environ.pop("ANTHROPIC_API_KEY", None)
    environ.update(extra_env or {})
    environ.update(fake_env or {})
    config = br.BrainConfig(brain_dir=brain_dir, timeout_s=timeout_s, claude=str(shim))
    return br.Brain(config, state=state, log_path=tmp_path / "logs" / "turns.jsonl", environ=environ)


def log_rows(tmp_path):
    return [json.loads(line) for line in (tmp_path / "logs" / "turns.jsonl").read_text().splitlines()]


def wait_dead(pid: int, within_s: float = 3.0) -> bool:
    end = time.monotonic() + within_s
    while time.monotonic() < end:
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            return True
        status = Path(f"/proc/{pid}/stat")
        if status.exists() and status.read_text().split(")")[-1].split()[0] == "Z":
            return True   # a zombie is dead; it only waits to be reaped by its new parent
        time.sleep(0.05)
    return False


# --- the command, the environment, the prompt ---

def test_command_exact_argv():
    assert br.build_command(br.BrainConfig(claude="claude")) == [
        "claude", "-p",
        "--model", "claude-haiku-5-5",
        "--tools", "",
        "--permission-mode", "dontAsk",
        "--strict-mcp-config", "--mcp-config", '{"mcpServers":{}}',
        "--system-prompt", br.DEFAULT_SYSTEM_PROMPT,
        "--no-session-persistence",
        "--output-format", "stream-json",
        "--verbose",
        "--include-partial-messages",
    ]


def test_command_pins_model_id_and_disables_tools():
    argv = br.build_command(br.BrainConfig())
    assert argv[argv.index("--model") + 1] == "claude-haiku-5-5"
    assert argv[argv.index("--tools") + 1] == ""
    assert argv[argv.index("--permission-mode") + 1] == "dontAsk"
    assert "--strict-mcp-config" in argv
    assert not any("bypass" in a.lower() for a in argv)
    # a variadic flag (--tools, --mcp-config) must be followed by another flag, never by the prompt
    for flag in ("--tools", "--mcp-config"):
        assert argv[argv.index(flag) + 2].startswith("--")


def test_command_keeps_the_user_text_out_of_argv():
    assert not any("sky" in a for a in br.build_command(br.BrainConfig()))


def test_env_drops_api_key_and_keeps_base_untouched():
    base = {"PATH": "/bin", "ANTHROPIC_API_KEY": "sk-not-real", "CLAUDE_CODE_OAUTH_TOKEN": "tok-not-real"}
    env = br.build_env(base)
    assert "ANTHROPIC_API_KEY" not in env
    assert env["CLAUDE_CODE_OAUTH_TOKEN"] == "tok-not-real"   # passed through untouched, never read
    assert env["CLAUDE_CODE_DISABLE_CLAUDE_MDS"] == "1"
    assert env["CLAUDE_CODE_DISABLE_AUTO_MEMORY"] == "1"
    assert "MAX_THINKING_TOKENS" not in env
    assert base["ANTHROPIC_API_KEY"] == "sk-not-real"
    assert br.build_env(base, thinking_off=True)["MAX_THINKING_TOKENS"] == "0"


def test_prompt_puts_state_line_first():
    assert br.build_prompt("hello", "voice state: listening") == "[voice state: listening]\nhello"
    assert br.build_prompt("hello", "  ") == "hello"


# --- parsing ---

@pytest.mark.parametrize("line", ["", "not json", "{broken", "[1, 2]", '"text"', "42", "null"])
def test_parse_line_returns_none_for_garbage(line):
    assert br.parse_line(line) is None


def test_delta_text_only_takes_text_deltas():
    text = {"type": "stream_event", "event": {"type": "content_block_delta", "delta": {"type": "text_delta", "text": "Hi"}}}
    thinking = {"type": "stream_event", "event": {"type": "content_block_delta",
                                                  "delta": {"type": "thinking_delta", "thinking": "hmm"}}}
    assert br.delta_text(text) == "Hi"
    assert br.delta_text(thinking) is None
    assert br.delta_text({"type": "stream_event", "event": "odd"}) is None
    assert br.delta_text({"type": "assistant"}) is None


def test_fixture_parses_offline():
    record = br.TurnRecord(started_at="x", model="m")
    pieces = []
    for event in fixture_events():
        br.note_event(event, record)
        if (piece := br.delta_text(event)) is not None:
            pieces.append(piece)
        if event["type"] == "result":
            br.read_result(event, record)
    assert "".join(pieces) == FIXTURE_REPLY
    assert len(pieces) == 10
    assert record.duration_api_ms == 1511
    assert record.claude_code_version == "2.1.303"


# --- whole turns through the fake claude ---

def test_chunks_arrive_in_order(shim, brain_dir, tmp_path):
    brain = make_brain(shim, brain_dir, tmp_path, {"FAKE_CLAUDE_STREAM": str(FIXTURE)})
    pieces = list(brain.ask("What colour is the sky?"))
    expected = [br.delta_text(e) for e in fixture_events() if br.delta_text(e)]
    assert pieces == expected
    assert "".join(pieces) == FIXTURE_REPLY
    assert brain.last.ok and brain.last.fallback is None


def test_timing_and_usage_parse(shim, brain_dir, tmp_path):
    brain = make_brain(shim, brain_dir, tmp_path, {"FAKE_CLAUDE_STREAM": str(FIXTURE)})
    list(brain.ask("hi"))
    r = brain.last
    assert r.kind is None and r.exit_code == 0
    assert r.duration_api_ms == 1511 and r.duration_ms == 1657
    assert r.ttft_ms == 1607 and r.time_to_request_ms == 107
    assert (r.input_tokens, r.cache_creation_input_tokens, r.cache_read_input_tokens, r.output_tokens) == (2, 661, 0, 46)
    assert r.thinking_tokens == 0
    assert r.cost_usd == pytest.approx(0.0001554)
    assert r.models_reported == ["claude-haiku-5-5"] and r.model == "claude-haiku-5-5"
    assert r.stop_reason == "end_turn" and r.rate_limit_status == "allowed"
    assert r.claude_code_version == "2.1.303"
    assert r.chunks == 10 and r.reply_chars == len(FIXTURE_REPLY) and r.bad_lines == 0
    assert 0 < r.first_chunk_ms <= r.total_ms


def test_fake_gets_exact_argv_cwd_and_prompt(shim, brain_dir, tmp_path):
    seen = tmp_path / "seen.json"
    brain = make_brain(shim, brain_dir, tmp_path, {"FAKE_CLAUDE_STREAM": str(FIXTURE), "FAKE_CLAUDE_RECORD": str(seen)},
                       extra_env={"ANTHROPIC_API_KEY": "sk-not-real"})
    list(brain.ask("What colour is the sky?"))
    got = json.loads(seen.read_text())
    assert got["argv"] == br.build_command(brain.config)[1:]
    assert Path(got["cwd"]).resolve() == brain_dir.resolve()
    assert got["stdin"] == "[voice state: listening]\nWhat colour is the sky?"
    assert got["env"]["ANTHROPIC_API_KEY"] is None
    assert got["env"]["CLAUDE_CODE_DISABLE_CLAUDE_MDS"] == "1"


def test_state_function_is_called_each_turn(shim, brain_dir, tmp_path):
    seen = tmp_path / "seen.json"
    states = iter(["voice state: listening", "voice state: offline"])
    brain = make_brain(shim, brain_dir, tmp_path, {"FAKE_CLAUDE_STREAM": str(FIXTURE), "FAKE_CLAUDE_RECORD": str(seen)},
                       state=lambda: next(states))
    list(brain.ask("one"))
    list(brain.ask("two"))
    assert json.loads(seen.read_text())["stdin"] == "[voice state: offline]\ntwo"


def test_timeout_kills_child_and_grandchild(shim, brain_dir, tmp_path):
    pidfile, grandfile = tmp_path / "pid", tmp_path / "grandchild"
    brain = make_brain(shim, brain_dir, tmp_path, {"FAKE_CLAUDE_STREAM": str(FIXTURE), "FAKE_CLAUDE_HANG_AFTER": "3",
                                                   "FAKE_CLAUDE_PIDFILE": str(pidfile),
                                                   "FAKE_CLAUDE_GRANDCHILD": str(grandfile)}, timeout_s=1.0)
    start = time.monotonic()
    pieces = list(brain.ask("hi"))
    took = time.monotonic() - start
    assert took < 4.0
    assert brain.last.kind == "timeout"
    assert brain.last.exit_code is None
    assert pieces[-1] == br.FALLBACK_LINE
    assert wait_dead(int(pidfile.read_text()))
    assert wait_dead(int(grandfile.read_text()))


def test_timeout_before_any_output(shim, brain_dir, tmp_path):
    brain = make_brain(shim, brain_dir, tmp_path, {"FAKE_CLAUDE_HANG_AFTER": "0"}, timeout_s=0.5)
    assert list(brain.ask("hi")) == [br.FALLBACK_LINE]
    assert brain.last.kind == "timeout" and brain.last.first_chunk_ms is None


def test_missing_binary(brain_dir, tmp_path):
    config = br.BrainConfig(brain_dir=brain_dir, claude=str(tmp_path / "no-such-claude"))
    brain = br.Brain(config, log_path=tmp_path / "logs" / "turns.jsonl")
    assert list(brain.ask("hi")) == [br.FALLBACK_LINE]
    assert brain.last.kind == "not_found"
    assert "not found" in brain.last.message
    assert log_rows(tmp_path)[0]["kind"] == "not_found"


def test_missing_brain_dir(shim, tmp_path):
    config = br.BrainConfig(brain_dir=tmp_path / "absent", claude=str(shim))
    brain = br.Brain(config, log_path=tmp_path / "logs" / "turns.jsonl")
    assert list(brain.ask("hi")) == [br.FALLBACK_LINE]
    assert brain.last.kind == "setup" and "absent" in brain.last.message


def test_nonzero_exit_without_result(shim, brain_dir, tmp_path):
    brain = make_brain(shim, brain_dir, tmp_path, {"FAKE_CLAUDE_EXIT": "1", "FAKE_CLAUDE_STDERR": "Invalid API key"})
    assert list(brain.ask("hi")) == [br.FALLBACK_LINE]
    r = brain.last
    assert r.kind == "exit" and r.exit_code == 1
    assert "code 1" in r.message and "Invalid API key" in r.message


def test_error_result(shim, brain_dir, tmp_path):
    events = [e for e in fixture_events() if e["type"] not in ("result", "assistant") and not br.delta_text(e)]
    result = dict(fixture_events()[-1], is_error=True, subtype="error_during_execution", result="model not found")
    stream = write_stream(tmp_path / "err.jsonl", events + [result])
    brain = make_brain(shim, brain_dir, tmp_path, {"FAKE_CLAUDE_STREAM": str(stream), "FAKE_CLAUDE_EXIT": "1"})
    assert list(brain.ask("hi")) == [br.FALLBACK_LINE]
    assert brain.last.kind == "error" and "model not found" in brain.last.message
    assert brain.last.exit_code == 1


def test_refusal(shim, brain_dir, tmp_path):
    events = [e for e in fixture_events() if e["type"] != "result"]
    result = dict(fixture_events()[-1], stop_reason="refusal")
    stream = write_stream(tmp_path / "refusal.jsonl", events + [result])
    brain = make_brain(shim, brain_dir, tmp_path, {"FAKE_CLAUDE_STREAM": str(stream)})
    pieces = list(brain.ask("hi"))
    assert pieces[-1] == br.REFUSAL_LINE
    assert brain.last.kind == "refusal" and brain.last.fallback == br.REFUSAL_LINE


def test_garbage_line_is_counted_and_skipped(shim, brain_dir, tmp_path):
    brain = make_brain(shim, brain_dir, tmp_path, {"FAKE_CLAUDE_STREAM": str(FIXTURE),
                                                   "FAKE_CLAUDE_EXTRA": "Warning: {not json at all"})
    assert "".join(brain.ask("hi")) == FIXTURE_REPLY
    assert brain.last.ok and brain.last.bad_lines == 1


def test_stream_without_result_is_an_error(shim, brain_dir, tmp_path):
    stream = write_stream(tmp_path / "cut.jsonl", [e for e in fixture_events() if e["type"] != "result"])
    brain = make_brain(shim, brain_dir, tmp_path, {"FAKE_CLAUDE_STREAM": str(stream)})
    pieces = list(brain.ask("hi"))
    assert "".join(pieces[:-1]) == FIXTURE_REPLY and pieces[-1] == br.FALLBACK_LINE
    assert brain.last.kind == "error" and "without a result" in brain.last.message


def test_empty_reply(shim, brain_dir, tmp_path):
    events = [e for e in fixture_events() if not br.delta_text(e) and e["type"] not in ("assistant", "result")]
    stream = write_stream(tmp_path / "empty.jsonl", events + [dict(fixture_events()[-1], result="")])
    brain = make_brain(shim, brain_dir, tmp_path, {"FAKE_CLAUDE_STREAM": str(stream)})
    assert list(brain.ask("hi")) == [br.FALLBACK_LINE]
    assert brain.last.kind == "empty"


def test_whole_message_used_once_when_no_deltas(shim, brain_dir, tmp_path):
    stream = write_stream(tmp_path / "nodelta.jsonl", [e for e in fixture_events() if e["type"] != "stream_event"])
    brain = make_brain(shim, brain_dir, tmp_path, {"FAKE_CLAUDE_STREAM": str(stream)})
    assert list(brain.ask("hi")) == [FIXTURE_REPLY]
    assert brain.last.ok


def test_fallback_lines():
    assert br.FALLBACK_LINE == "I can't reach my brain."
    for kind in br.KINDS:
        expected = br.REFUSAL_LINE if kind == "refusal" else br.FALLBACK_LINE
        assert br.fallback_line(kind) == expected


def test_stopping_early_kills_child_and_logs_abandoned(shim, brain_dir, tmp_path):
    pidfile = tmp_path / "pid"
    brain = make_brain(shim, brain_dir, tmp_path, {"FAKE_CLAUDE_STREAM": str(FIXTURE), "FAKE_CLAUDE_HANG_AFTER": "10",
                                                   "FAKE_CLAUDE_PIDFILE": str(pidfile)})
    turn = brain.ask("hi")
    assert next(turn)
    turn.close()
    assert brain.last.kind == "abandoned"
    assert wait_dead(int(pidfile.read_text()))
    assert log_rows(tmp_path)[-1]["kind"] == "abandoned"


# --- the log ---

def test_one_log_line_per_turn_ok_or_not(shim, brain_dir, tmp_path):
    brain = make_brain(shim, brain_dir, tmp_path, {"FAKE_CLAUDE_STREAM": str(FIXTURE)})
    list(brain.ask("one"))
    list(brain.ask("two"))
    brain.config = br.BrainConfig(brain_dir=brain_dir, claude=str(tmp_path / "missing"))
    list(brain.ask("three"))
    rows = log_rows(tmp_path)
    assert [r["kind"] for r in rows] == [None, None, "not_found"]
    for key in ("first_chunk_ms", "total_ms", "duration_api_ms", "input_tokens", "output_tokens",
                "cost_usd", "model", "exit_code", "started_at"):
        assert key in rows[0]
    assert rows[0]["model"] == "claude-haiku-5-5"


def test_no_secret_reaches_log_or_output(shim, brain_dir, tmp_path, capsys):
    secrets = {"ANTHROPIC_API_KEY": "sk-secret-123", "CLAUDE_CODE_OAUTH_TOKEN": "tok-secret-456"}
    brain = make_brain(shim, brain_dir, tmp_path, {"FAKE_CLAUDE_STREAM": str(FIXTURE)}, extra_env=secrets)
    br.run_turn(brain, "hi")
    text = (tmp_path / "logs" / "turns.jsonl").read_text() + capsys.readouterr().out
    assert "secret" not in text


def test_summary_line():
    r = br.TurnRecord(started_at="x", model="claude-haiku-5-5", first_chunk_ms=812.4, total_ms=2601.0,
                      duration_api_ms=1006, input_tokens=2, cache_creation_input_tokens=661, output_tokens=12,
                      cost_usd=0.000155, exit_code=0)
    line = br.summary_line(r)
    assert line.startswith("turn ok | model claude-haiku-5-5 | first 812 ms | total 2601 ms | api 1006 ms")
    assert "in 2 (+661 cached) out 12" in line and "exit 0" in line
    failed = br.TurnRecord(started_at="x", model="m", kind="timeout", message="no answer")
    assert "FAILED timeout: no answer" in br.summary_line(failed)


def test_report_median_and_worst():
    rows = [{"kind": None, "first_chunk_ms": f, "total_ms": t, "duration_api_ms": a}
            for f, t, a in [(4000, 5000, 900), (1000, 2000, 1000), (1200, 2400, 1100), (1100, 2200, 1050)]]
    rows.append({"kind": "timeout", "first_chunk_ms": None, "total_ms": 10000, "duration_api_ms": None})
    text = br.report(rows)
    assert "turns 5, ok 4, failed 1 (timeout)" in text
    assert "all ok: n=4 | first median 1150 ms, worst 4000 ms | total median 2300 ms, worst 5000 ms" in text
    assert "warm (first turn left out): n=3 | first median 1100 ms, worst 1200 ms" in text


# --- the command line ---

def test_help_runs_without_claude():
    out = subprocess.run([sys.executable, "-m", "software.skull_core.brain", "--help"], cwd=REPO_ROOT,
                         capture_output=True, text=True, timeout=30)
    assert out.returncode == 0 and "--once" in out.stdout


def test_parse_args_rejects_bad_values():
    with pytest.raises(SystemExit):
        br.parse_args(["--timeout", "0"])
    with pytest.raises(SystemExit):
        br.parse_args(["--report", "0"])
    args = br.parse_args([])
    assert args.model == "claude-haiku-5-5" and args.timeout == br.DEFAULT_TIMEOUT_S


def test_main_once_ok_and_failed(shim, brain_dir, tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("FAKE_CLAUDE_STREAM", str(FIXTURE))
    log = tmp_path / "turns.jsonl"
    common = ["--brain-dir", str(brain_dir), "--log", str(log)]
    assert br.main(["--once", "hi", "--claude", str(shim)] + common) == 0
    assert br.main(["--once", "hi", "--claude", str(tmp_path / "missing")] + common) == 1
    out = capsys.readouterr().out
    assert FIXTURE_REPLY in out and "turn ok" in out and "FAILED not_found" in out
    assert br.main(["--report", "5", "--log", str(log)]) == 0
    assert "turns 2, ok 1, failed 1 (not_found)" in capsys.readouterr().out


def test_interactive_loop_quits_on_empty_line(shim, brain_dir, tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("FAKE_CLAUDE_STREAM", str(FIXTURE))
    lines = iter(["What colour is the sky?", ""])
    monkeypatch.setattr("builtins.input", lambda prompt="": next(lines))
    log = tmp_path / "turns.jsonl"
    assert br.main(["--claude", str(shim), "--brain-dir", str(brain_dir), "--log", str(log)]) == 0
    assert FIXTURE_REPLY in capsys.readouterr().out
    assert len(log.read_text().splitlines()) == 1


def test_interactive_loop_quits_on_eof(brain_dir, tmp_path, monkeypatch):
    def eof(prompt=""):
        raise EOFError
    monkeypatch.setattr("builtins.input", eof)
    assert br.main(["--brain-dir", str(brain_dir), "--log", str(tmp_path / "t.jsonl")]) == 0
