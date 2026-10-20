"""Tests for software/skull_core/exit_test.py: /proc reading, the sampler, the self-trigger rule,
the ten-turn verdict, the latency and memory report, the WiFi-off verdict and the run wrapper.

The turn records are built with the real voice_loop.turn_record, so a change in its fields breaks here.
"""
import json
import subprocess
import time
from pathlib import Path

import pytest

from software.skull_core import exit_test as et
from software.skull_core import voice_loop as vl
from software.skull_core import wake
from software.skull_core.brain import FALLBACK_LINE, TurnRecord
from software.skull_core.stt import Heard
from software.skull_core.voice_out import ClipLog, SpeechLog

QUESTIONS = et.read_questions(et.DEFAULT_QUESTIONS)


def turn(t0, question="What is the temperature of your brain?", reply="My brain is warm. It hums.",
         first_reply_s=13.0, trigger="enter", offline=False, kind=None, text_ok=True, end_state="idle",
         sound=True, queue_full=0):
    """One turn record as the loop writes it. Speech runs t0 .. t0+2; Cogitating at +0.7 s after its end."""
    end = t0 + 2.0
    heard = Heard(text=question if text_ok else None, raw_text=question, end_reason="detector", lines=1,
                  t_first_sample=t0, t_speech_end=end, t_text=end + 0.4, stt_s=0.4, detector_start=t0,
                  detector_end=end + 0.6, detector_end_to_text_s=0.35, queue_full=queue_full)
    speech = None
    if sound:
        speech = SpeechLog(t_start=end + 0.5)
        if not offline:                      # the brain's fallback comes through speak(): Cogitating first
            speech.clips.append(ClipLog("(cogitating)", "opener", 0.0, 1.2, t_ready=end + 0.6,
                                        t_play_start=end + 0.7, t_play_end=end + 1.9))
        line = FALLBACK_LINE if (offline or kind) else reply
        speech.clips.append(ClipLog(line, "bank" if (offline or kind) else "pocket", 12.0, 3.0,
                                    t_ready=end + first_reply_s - 0.1, t_play_start=end + first_reply_s,
                                    t_play_end=end + first_reply_s + 3.0))
    log = vl.TurnLog(t_enter=t0 - 0.5, heard=heard, asked=not offline, t_first_piece=end + 2.5,
                     t_done=end + first_reply_s + 3.0, speech=speech, offline=offline, end_state=end_state)
    brain = None if offline else TurnRecord(started_at="x", model="m", kind=kind, first_chunk_ms=2500.0)
    wake_heard = heard if trigger == "wake" else None
    return vl.turn_record(log, "detector", brain_record=brain, trigger=trigger,
                          wake_text=question if trigger == "wake" else None, wake_heard=wake_heard)


def ten_turns(t0=1000.0, spacing=25.0, **kwargs):
    return [turn(t0 + i * spacing, question=q, **kwargs) for i, q in enumerate(QUESTIONS)]


def make_run(turns, lines=(), samples=(), silent_s=70.0, footer=None, timeout_s=10.0):
    last = max(t["clock"]["last_sound_end"] or t["clock"]["done"] or 0 for t in turns) if turns else 0.0
    header = {"run_dir": "/tmp/r", "questions": QUESTIONS, "trigger": "enter", "mem_available_before_kb": 7_000_000,
              "loop": {"timeout_s": timeout_s, "model": "m", "end_rule": "detector"}}
    if footer is None:
        footer = {"outcome": "returned", "exit_code": 0, "t_end": last + silent_s, "pid": 1, "cpu_s": 60.0, "wall_s": 400.0}
    return et.Run(header, list(turns), list(lines), list(samples), footer)


# --- the question list ---------------------------------------------------------------------------

def test_question_list_has_ten_and_skips_comments(tmp_path):
    assert len(QUESTIONS) == et.REQUIRED_TURNS
    assert QUESTIONS[0] == "What is the temperature of your brain?"
    path = tmp_path / "q.txt"
    path.write_text("# header\n\nOne?\n  # indented comment\nTwo.\n")
    assert et.read_questions(path) == ["One?", "Two."]


# --- /proc and sensors ---------------------------------------------------------------------------

def test_parse_meminfo_and_mem_available(tmp_path):
    text = "MemTotal:        8245248 kB\nMemFree:  100 kB\nMemAvailable:    5123456 kB\nHugePages_Total:       0\n"
    assert et.parse_meminfo(text)["MemAvailable"] == 5123456
    assert et.parse_meminfo(text)["HugePages_Total"] == 0
    (tmp_path / "meminfo").write_text(text)
    assert et.mem_available_kb(tmp_path / "meminfo") == 5123456
    assert et.mem_available_kb(tmp_path / "missing") is None


def fake_proc(tmp_path, procs):
    """procs: {pid: (ppid, name, rss_kb or None)}"""
    for pid, (ppid, name, rss) in procs.items():
        folder = tmp_path / str(pid)
        folder.mkdir()
        (folder / "stat").write_text(f"{pid} ({name}) S {ppid} 1 1 0 -1 4194560 100 0 0 0")
        (folder / "status").write_text(f"Name:\t{name}\n" + ("" if rss is None else f"VmRSS:\t{rss} kB\n"))
    (tmp_path / "self").mkdir()           # not a pid: skipped
    return tmp_path


def test_process_tree_sums_descendants_only(tmp_path):
    proc = fake_proc(tmp_path, {
        100: (1, "python", 400_000),
        101: (100, "tts worker) (x", 1_100_000),   # a name with ") (" and spaces must not confuse the parser
        102: (101, "helper", 10_000),
        103: (100, "claude", 250_000),
        200: (1, "other", 999_999),                # not ours
        104: (100, "kthread-like", None),          # no VmRSS
    })
    assert et.process_tree_rss_kb(100, proc) == (1_760_000, 5)


def test_process_tree_on_this_process_is_positive():
    import os
    rss, count = et.process_tree_rss_kb(os.getpid())
    assert rss > 0 and count >= 1


def test_temperature_from_vcgencmd():
    def run(argv, **kwargs):
        out = {"measure_temp": "temp=47.2'C\n", "get_throttled": "throttled=0x50000\n"}[argv[1]]
        return subprocess.CompletedProcess(argv, 0, stdout=out, stderr="")
    assert et.read_temperature(run) == {"temp_c": 47.2, "throttled": "0x50000", "source": "vcgencmd"}


def test_temperature_falls_back_then_gives_none(tmp_path):
    def missing(argv, **kwargs):
        raise FileNotFoundError("vcgencmd")
    zone = tmp_path / "temp"
    zone.write_text("51234\n")
    assert et.read_temperature(missing, zone) == {"temp_c": 51.234, "throttled": None, "source": "thermal_zone0"}
    assert et.read_temperature(missing, tmp_path / "none") == {"temp_c": None, "throttled": None, "source": None}


def test_sampler_reads_temperature_every_ten_seconds_and_reports_write_errors_once(tmp_path):
    now = [0.0]
    temps = []
    said = []

    def temperature():
        temps.append(now[0])
        return {"temp_c": 50.0, "throttled": "0x0", "source": "fake"}

    blocker = tmp_path / "file"
    blocker.write_text("x")
    sampler = et.Sampler(blocker / "samples.jsonl", 1, clock=lambda: now[0], mem=lambda: 5_000_000,
                         tree=lambda pid: (123, 2), temperature=temperature, report=said.append)
    rows = []
    for second in range(25):
        now[0] = float(second)
        row = sampler.sample()
        rows.append(row)
        sampler.write(row)
    assert temps == [0.0, 10.0, 20.0]
    assert "temp_c" in rows[10] and "temp_c" not in rows[11]
    assert rows[3]["mem_available_kb"] == 5_000_000 and rows[3]["tree_rss_kb"] == 123
    assert len(said) == 1 and "Could not write" in said[0]
    assert sampler.rows == 25


def test_sampler_thread_writes_lines_and_stops(tmp_path):
    path = tmp_path / "samples.jsonl"
    sampler = et.Sampler(path, 1, period_s=0.01, mem=lambda: 1, tree=lambda pid: (2, 1),
                         temperature=lambda: {"temp_c": None, "throttled": None, "source": None}).start()
    time.sleep(0.1)
    sampler.stop()
    count = len(path.read_text().splitlines())
    assert count >= 2
    time.sleep(0.05)
    assert len(path.read_text().splitlines()) == count       # stopped means stopped


# --- the self-trigger rule -----------------------------------------------------------------------

def test_echo_run_needs_four_words_in_order():
    assert et.echo_run("so my brain is warm today", "My brain is warm. It hums.") == "my brain is warm"
    assert et.echo_run("brain warm is my", "My brain is warm.") is None
    assert et.echo_run("my brain is", "My brain is warm.") is None
    assert et.echo_run("", "anything at all here") is None


def test_self_trigger_by_timing_window():
    previous = turn(1000.0)
    last = previous["clock"]["last_sound_end"]
    assert et.self_trigger_reasons(last + 0.4, "hello there", previous, QUESTIONS)
    assert et.self_trigger_reasons(last + 1.49, "hello there", previous, QUESTIONS)
    assert et.self_trigger_reasons(last + 3.0, "hello there", previous, QUESTIONS) == []
    assert et.self_trigger_reasons(previous["clock"]["first_sound"] + 1, "x", previous, QUESTIONS)   # during the reply
    assert et.self_trigger_reasons(None, "hello there", previous, QUESTIONS) == []
    assert et.self_trigger_reasons(last + 0.4, "hello", None, QUESTIONS) == []        # the first turn


def test_self_trigger_by_text_but_not_when_the_reply_repeats_the_question():
    previous = turn(1000.0, reply="You ask what day comes after Friday? Saturday.")
    late = previous["clock"]["last_sound_end"] + 10
    assert et.self_trigger_reasons(late, "awaken machine spirit ask what day comes", previous, QUESTIONS)
    # the reply quoted question 4; the person reading question 4 must not be blamed
    assert et.self_trigger_reasons(late, "What day comes after Friday?", previous, QUESTIONS) == []


# --- the ten-turn verdict ------------------------------------------------------------------------

def test_ten_clean_turns_pass():
    passed, text = et.report_ten(make_run(ten_turns()))
    assert passed, text
    assert "PASS" in text and "clean turns in a row from the start: 10" in text
    assert "HEADLINE speech end -> first reply audio: median 13.000 s, worst 13.000 s, n=10" in text


def test_latency_median_and_worst_from_known_values():
    turns = [turn(1000.0 + i * 30, question=q, first_reply_s=s)
             for i, (q, s) in enumerate(zip(QUESTIONS, [12, 14, 13, 15, 11, 16, 12.5, 13.5, 14.5, 20]))]
    _, text = et.report_ten(make_run(turns))
    assert "HEADLINE speech end -> first reply audio: median 13.750 s, worst 20.000 s, n=10" in text
    assert "speech end -> Cogitating starts: median 0.700 s" in text


def test_lost_blocks_leave_the_latency_but_not_the_count():
    turns = ten_turns()
    turns[2] = turn(1050.0, question=QUESTIONS[2], queue_full=3)
    passed, text = et.report_ten(make_run(turns))
    assert passed
    assert "answered turns 10, with clean times 9" in text and "n=9" in text


def test_self_trigger_at_turn_four_ends_the_run():
    turns = ten_turns()
    prev_end = turns[2]["clock"]["last_sound_end"]
    turns[3] = turn(prev_end + 0.5, question=QUESTIONS[3])        # started 0.5 s after the skull stopped
    passed, text = et.report_ten(make_run(turns))
    assert not passed
    assert "clean turns in a row from the start: 3" in text
    assert "turn 4: self-trigger: started +0.50 s" in text


def test_misheard_and_fallback_are_counted_but_do_not_end_the_run():
    turns = ten_turns()
    turns[1] = turn(1025.0, question="How many eggs do you have?")
    turns[5] = turn(1125.0, question=QUESTIONS[5], kind="timeout")
    passed, text = et.report_ten(make_run(turns))
    assert passed, text
    assert "misheard 1," in text and "fallback lines 1," in text
    assert "answered turns 9" in text                 # the fallback turn is not in the headline


def test_a_turn_in_the_silent_minute_fails_unless_marked_as_a_false_wake():
    turns = ten_turns() + [turn(1000.0 + 10 * 25, question="what was that")]
    passed, text = et.report_ten(make_run(turns))
    assert not passed and "in the silent minute" in text and "--false-wake 11" in text
    passed, text = et.report_ten(make_run(turns), false_wakes={11})
    assert passed, text
    assert "FALSE WAKE (marked)" in text


def test_a_false_wake_mid_run_does_not_shift_the_questions():
    turns = ten_turns(spacing=60.0)
    extra = turn(1000.0 + 2 * 60 + 30, question="turn the telly down")    # someone else, between 3 and 4
    turns.insert(3, extra)
    passed, text = et.report_ten(make_run(turns), false_wakes={4})
    assert passed, text
    assert "misheard 0," in text


def test_idle_echo_line_is_a_self_trigger():
    turns = ten_turns()
    line = {"text": "it hums", "woke": False, "t_start": turns[4]["clock"]["last_sound_end"] + 0.8}
    wake_line = {"text": "awaken machine spirit", "woke": True, "t_start": turns[4]["clock"]["last_sound_end"] + 0.8}
    passed, text = et.report_ten(make_run(turns, lines=[line, wake_line]))
    assert not passed
    assert "idle line 'it hums' after turn 5: self-trigger" in text


def test_crash_short_run_and_missing_silent_minute_fail():
    passed, text = et.report_ten(make_run(ten_turns(), footer={"outcome": "exception", "exception": "RuntimeError('x')"}))
    assert not passed and "crash: RuntimeError('x')" in text
    passed, text = et.report_ten(make_run(ten_turns()[:7]))
    assert not passed and "only 7 turns logged" in text
    passed, text = et.report_ten(make_run(ten_turns(), silent_s=20))
    assert not passed and "only 20 s of 60 s" in text
    passed, text = et.report_ten(et.Run({}, ten_turns(), [], [], {}))
    assert not passed and "no footer.json" in text


def test_error_state_and_silence_past_the_timeout_end_the_run():
    turns = ten_turns()
    turns[6] = turn(1150.0, question=QUESTIONS[6], end_state="error")
    passed, text = et.report_ten(make_run(turns))
    assert not passed and "turn 7: error state" in text
    turns = ten_turns()
    turns[8] = turn(1200.0, question=QUESTIONS[8], offline=True, first_reply_s=12.0)   # an unbanked offline line
    passed, text = et.report_ten(make_run(turns))
    assert not passed and "turn 9: silence 12.00 s after the question, past the 10 s timeout" in text


def test_heard_text_but_no_sound_ends_the_run():
    turns = ten_turns()
    turns[0] = turn(1000.0, question=QUESTIONS[0], sound=False)
    passed, text = et.report_ten(make_run(turns))
    assert not passed and "turn 1: heard text but the skull made no sound" in text


# --- memory, heat, CPU ---------------------------------------------------------------------------

def test_memory_report_against_the_g9_limit():
    samples = [{"t": 1, "at": "a", "mem_available_kb": 6_500_000, "tree_rss_kb": 1_000_000, "tree_procs": 2,
                "temp_c": 55.0, "throttled": "0x0", "source": "vcgencmd"},
               {"t": 2, "at": "b", "mem_available_kb": 5_000_000, "tree_rss_kb": 2_070_937, "tree_procs": 3},
               {"t": 3, "at": "c", "mem_available_kb": 5_500_000, "tree_rss_kb": 1_500_000, "tree_procs": 3,
                "temp_c": 61.5, "throttled": "0x80000", "source": "vcgencmd"}]
    lines = et.memory_lines(make_run(ten_turns(), samples=samples))
    assert lines[0].startswith("G9, MemAvailable: before 7,000,000 kB, lowest 5,000,000 kB (at b), "
                               "drop 2,000,000 kB = 48.3 % of the 4,141,875 kB limit (within)")
    assert "peak 2,070,937 kB in 3 processes (at b) = 50.0 % of the limit (within" in lines[1]
    assert "peak 61.5 C" in lines[2] and "['0x80000']" in lines[2]
    assert "60.0 s over 400.0 s = 15.0 % of one core" in lines[3]


def test_memory_report_without_samples():
    lines = et.memory_lines(et.Run({}, [], [], [], {}))
    assert lines == ["G9, MemAvailable: no baseline or no samples", "temperature: no readings"]


# --- the WiFi-off verdict ------------------------------------------------------------------------

def wifi_turns():
    return [turn(1000.0, question=QUESTIONS[0], offline=True, first_reply_s=0.3),
            turn(1030.0, question=QUESTIONS[1], kind="timeout", first_reply_s=9.5),
            turn(1060.0, question=QUESTIONS[2])]


def test_wifi_pass():
    passed, text = et.report_wifi(make_run(wifi_turns()))
    assert passed, text
    assert "check 1 (a state command offline): NOT TESTABLE" in text
    assert "offline: end of question -> first sound: median 0.300 s" in text


def test_wifi_fails_without_a_fallback_or_a_normal_last_turn():
    passed, text = et.report_wifi(make_run(wifi_turns()[:1]))
    assert not passed
    assert "no fallback turn" in text and "the last turn was not a normal answer" in text


# --- the voice_loop and wake fields this tool reads ---------------------------------------------

def test_turn_record_clock_stamps():
    record = turn(1000.0, trigger="wake")
    clock = record["clock"]
    assert clock["speech_start"] == 1000.0 and clock["speech_end"] == 1002.0
    assert clock["wake_start"] == 1000.0 and clock["first_sound"] == pytest.approx(1002.7)
    assert clock["last_sound_end"] == pytest.approx(1018.0) and clock["enter"] == 999.5
    json.dumps(record)
    plain = vl.turn_record(vl.TurnLog(t_enter=5.0), "line")
    assert plain["clock"]["first_sound"] is None and plain["clock"]["wake_start"] is None


def test_wake_line_log_has_monotonic_stamps(tmp_path):
    utterance = Heard(text="hello", raw_text="hello", end_reason="detector", detector_start=50.0, detector_end=51.0)
    wake_up = Heard(text="x", raw_text="Awaken machine spirit. Hi there.", end_reason="detector",
                    detector_start=60.0, detector_end=62.0)
    script = [utterance, wake_up]
    log = wake.LineLog(tmp_path / "lines.jsonl")
    wake.wait_for_trigger(lambda interrupt: script.pop(0), None, list, log, 2, out=lambda text: None)
    rows = [json.loads(line) for line in (tmp_path / "lines.jsonl").read_text().splitlines()]
    assert [(r["t_start"], r["t_end"]) for r in rows] == [(50.0, 51.0), (60.0, 62.0)]


# --- the run wrapper -----------------------------------------------------------------------------

def run_args(tmp_path, *extra):
    return et.parse_args(["run", "--run-dir", str(tmp_path / "run"), "--trigger", "wake",
                          "--cooling", "heatsink and fan", "--", "--claude", str(tmp_path / "no-claude"), *extra])


def test_run_writes_header_samples_footer_and_report(tmp_path):
    seen = {}

    def fake_main(argv):
        seen["argv"] = argv
        log = Path(argv[argv.index("--log") + 1])
        t0 = time.monotonic() - 1000
        with open(log, "a") as out:
            for i, q in enumerate(QUESTIONS):
                out.write(json.dumps(turn(t0 + i * 25, question=q, trigger="wake")) + "\n")
        return 0

    said = []
    args, loop_argv = run_args(tmp_path, "--timeout", "12")
    assert et.run_test(args, loop_argv, main=fake_main, out=said.append) == 0
    run_dir = tmp_path / "run"
    header = json.loads((run_dir / "header.json").read_text())
    assert header["trigger"] == "wake" and header["cooling"] == "heatsink and fan"
    assert header["questions"] == QUESTIONS and header["loop"]["timeout_s"] == 12.0
    assert header["claude_version"] is None                      # missing program: recorded, not fatal
    assert header["voice_key"].endswith("/state/voice")          # the prefix comes from robot_params.yaml
    assert seen["argv"][-6:] == ["--trigger", "wake", "--log", str(run_dir / "turns.jsonl"),
                                 "--line-log", str(run_dir / "wake_lines.jsonl")]
    footer = json.loads((run_dir / "footer.json").read_text())
    assert footer["outcome"] == "returned" and footer["exit_code"] == 0 and footer["samples"] >= 1
    assert (run_dir / "samples.jsonl").exists()
    assert "M3-08 exit test" in said[-1] and "PASS" in said[-1]


def test_run_survives_a_crash_and_reports_it(tmp_path):
    def crashing(argv):
        raise RuntimeError("mic gone")
    said = []
    args, loop_argv = run_args(tmp_path)
    assert et.run_test(args, loop_argv, main=crashing, out=said.append) == 1
    footer = json.loads((tmp_path / "run" / "footer.json").read_text())
    assert footer["outcome"] == "exception" and "mic gone" in footer["exception"]
    assert "FAIL" in said[-1] and "crash: RuntimeError('mic gone')" in said[-1]


def test_run_refuses_a_used_folder_and_test_breaking_flags(tmp_path):
    (tmp_path / "run").mkdir()
    (tmp_path / "run" / "turns.jsonl").write_text("{}\n")
    said = []
    args, loop_argv = run_args(tmp_path)
    assert et.run_test(args, loop_argv, main=lambda argv: 0, out=said.append) == 1
    assert "not empty" in said[0]
    args, loop_argv = run_args(tmp_path / "b", "--no-brain")
    assert et.run_test(args, loop_argv, main=lambda argv: 0, out=said.append) == 1
    assert not (tmp_path / "b" / "run").exists()
    with pytest.raises(SystemExit):
        run_args(tmp_path, "--log", "x.jsonl")
    with pytest.raises(SystemExit):
        run_args(tmp_path, "--trigger=enter")


def test_report_command_exit_codes(tmp_path, capsys):
    run_dir = tmp_path / "r"
    run_dir.mkdir()
    run = make_run(ten_turns())
    (run_dir / "header.json").write_text(json.dumps(run.header))
    (run_dir / "turns.jsonl").write_text("".join(json.dumps(t) + "\n" for t in run.turns) + "{broken\n")
    (run_dir / "footer.json").write_text(json.dumps(run.footer))
    assert et.main(["report", "--run-dir", str(run_dir)]) == 0
    assert "PASS" in capsys.readouterr().out
    assert et.main(["report", "--run-dir", str(run_dir), "--wifi"]) == 2
    assert et.main(["report", "--run-dir", str(tmp_path / "none")]) == 1
