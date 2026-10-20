"""Tests for software/skull_core/voice_loop.py: the turn with fake parts, the log line, the report."""
import json
import socket
import sys
import threading
from types import SimpleNamespace

import numpy as np
import pytest

from software.skull_core import voice_loop as vl
from software.skull_core.brain import FALLBACK_LINE, TurnRecord
from software.skull_core.mic_stream import MicError, MicStats
from software.skull_core.stt import Heard
from software.skull_core.voice_console import HalfDuplexGuard
from software.skull_core.voice_out import ClipLog, PhraseBank, Speaker, SpeechLog, bank_voice


class FakeClock:
    """Moves forward by `step` every time it is read, so every stamp is distinct."""

    def __init__(self, t=10.0, step=1.0):
        self.t = t - step
        self.step = step

    def __call__(self):
        self.t += self.step
        return self.t


def heard(text="What time is it?", **kwargs):
    fields = dict(raw_text=text or "", end_reason="line", lines=1, t_first_sample=10.5, t_speech_end=12.0,
                  t_text=12.25, stt_s=0.25, first_line_text=text, first_line_stt_s=0.25)
    fields.update(kwargs)
    return Heard(text=text, **fields)


class Recorder:
    def __init__(self):
        self.states = []
        self.asked = []
        self.spoken = []
        self.closed = False

    def on_state(self, name):
        self.states.append(name)

    def ask(self, text):
        self.asked.append(text)
        return self._reply()

    def _reply(self):
        try:
            yield "It is "
            yield "noon."
        finally:
            self.closed = True

    def speak(self, pieces):
        self.spoken.extend(pieces)


# --- run_turn ------------------------------------------------------------------------------

def test_normal_turn_states_and_reply():
    rec = Recorder()
    turn = vl.run_turn(lambda: heard(), rec.ask, rec.speak, rec.on_state, clock=FakeClock())
    assert rec.states == ["listening", "thinking", "idle"]
    assert rec.asked == ["What time is it?"]
    assert rec.spoken == ["It is ", "noon."]
    assert turn.asked and turn.reply_chars == len("It is noon.")
    assert turn.t_enter < turn.t_first_piece < turn.t_done


def test_silent_turn_makes_no_brain_call():
    rec = Recorder()
    turn = vl.run_turn(lambda: heard(text=None, end_reason="no_speech"), rec.ask, rec.speak, rec.on_state)
    assert rec.states == ["listening", "idle"]
    assert rec.asked == [] and rec.spoken == []
    assert not turn.asked and turn.t_done is None


def test_no_brain_mode_stops_after_the_text():
    rec = Recorder()
    turn = vl.run_turn(lambda: heard(), None, rec.speak, rec.on_state)
    assert rec.states == ["listening", "idle"]
    assert turn.heard.text == "What time is it?" and not turn.asked


def test_turn_ends_on_error_when_speak_raises_and_the_reply_is_closed():
    rec = Recorder()

    def bad_speak(pieces):
        next(iter(pieces))
        raise OSError("amp gone")

    with pytest.raises(OSError):
        vl.run_turn(lambda: heard(), rec.ask, bad_speak, rec.on_state)
    assert rec.states == ["listening", "thinking", "error"]    # M3-07: no "idle" flash before the error
    assert rec.closed           # the reply generator's cleanup ran (the brain would kill its child here)


def test_turn_ends_on_error_when_listen_raises():
    rec = Recorder()

    def bad_listen():
        raise RuntimeError("moonshine failed")

    with pytest.raises(RuntimeError):
        vl.run_turn(bad_listen, rec.ask, rec.speak, rec.on_state)
    assert rec.states == ["listening", "error"]


def test_turn_ends_idle_on_ctrl_c_while_listening():
    rec = Recorder()

    def interrupted():
        raise KeyboardInterrupt

    with pytest.raises(KeyboardInterrupt):
        vl.run_turn(interrupted, rec.ask, rec.speak, rec.on_state)
    assert rec.states[-1] == "idle"


# --- the log line and the screen line ------------------------------------------------------

def test_turn_record_times_relative_and_word_errors():
    turn = vl.TurnLog(t_enter=10.0, heard=heard(text="What time is it", raw_text="What time is it"),
                      asked=True, t_first_piece=14.5, t_done=15.0, reply_chars=11)
    brain = TurnRecord(started_at="2026-10-20T10:00:00+03:00", model="m", first_chunk_ms=2400.0)
    record = vl.turn_record(turn, "line", expected="What time is it now?", brain_record=brain)
    assert record["enter_to_first_sample_s"] == 0.5
    assert record["enter_to_speech_end_s"] == 2.0
    assert record["speech_end_to_first_piece_s"] == 2.5
    assert record["speech_end_to_done_s"] == 3.0
    assert record["stt_s"] == 0.25
    assert record["expected_words"] == 5 and record["word_errors"] == 1
    assert record["first_line_word_errors"] == 1
    assert record["brain"]["first_chunk_ms"] == 2400.0
    json.dumps(record)       # it must be one valid JSON line


def test_turn_record_for_a_silent_turn():
    turn = vl.TurnLog(t_enter=10.0, heard=heard(text=None, raw_text="", end_reason="no_speech", lines=0,
                                                t_speech_end=None, stt_s=None, first_line_text=None,
                                                first_line_stt_s=None, t_text=None))
    record = vl.turn_record(turn, "detector", expected="Hello there")
    assert record["text"] is None and record["asked"] is False
    assert record["word_errors"] == 2 and record["brain"] is None
    assert record["speech_end_to_first_piece_s"] is None


def test_summary_line_shows_text_reason_time_and_errors():
    turn = vl.TurnLog(t_enter=10.0, heard=heard(raw_text="Awaken machine spirit. What time is it?",
                                                text="Awaken machine spirit. What time is it?", lines=2,
                                                first_line_text="Awaken machine spirit."))
    line = vl.summary_line(vl.turn_record(turn, "detector", expected="What time is it?"))
    assert "'Awaken machine spirit. What time is it?'" in line
    assert "end line" in line and "stt 0.25 s" in line and "lines 2" in line
    assert "first line 'Awaken machine spirit.'" in line
    assert "word errors 3/4" in line


# --- --report -------------------------------------------------------------------------------

def row(stt_s, errors=0, words=4, first=None, queue_full=0, text="x", asked=True, detector=None):
    return {"text": text, "stt_s": stt_s, "first_line_stt_s": stt_s if first is None else first,
            "detector_end_to_text_s": detector, "queue_full": queue_full, "gaps": 0, "asked": asked,
            "expected": "a b c d", "expected_words": words, "word_errors": errors,
            "first_line_word_errors": errors}


def test_report_median_worst_and_accuracy():
    rows = [row(0.2), row(0.4, errors=1), row(0.3), row(9.9, queue_full=3),
            row(None, text=None, asked=False, errors=4)]
    text = vl.report(rows)
    assert "turns 5: with text 4, nothing usable 1 (brain called on those: 0), lost blocks in 1" in text
    assert "stt_s (end of speech -> text): median 0.300 s, worst 0.400 s, n=3" in text
    assert "exact 3/5" in text and "at most 1 word wrong 4/5" in text
    assert "WER 5/20 = 0.250" in text      # the silent turn missed all 4 of its words


def test_report_without_expected_phrases_skips_accuracy():
    rows = [dict(row(0.2), expected=None)]
    assert "accuracy" not in vl.report(rows)


# --- arguments and main ----------------------------------------------------------------------

@pytest.mark.parametrize("argv", [["--start-timeout", "0"], ["--max", "-1"], ["--start-timeout", "20", "--max", "10"],
                                  ["--min-letters", "-1"], ["--report", "0"], ["--timeout", "0"],
                                  ["--end-rule", "wake"], ["--hang", "-0.1"], ["--hang", "nan"]])
def test_bad_arguments_are_refused(argv):
    with pytest.raises(SystemExit):
        vl.parse_args(argv)


def test_defaults():
    args = vl.parse_args([])
    assert args.end_rule == "detector" and args.start_timeout == 5.0 and args.max == 15.0 and args.min_letters == 2
    assert args.model == "claude-haiku-5-5" and args.mic_device == "UACDemoV1.0"
    assert args.hang == 0.6 and not args.no_speak and args.voice == "javert" and args.device == "MAX98357A"
    assert str(args.tts_python).endswith("m3-bench/.venv-tts/bin/python")
    assert str(args.bank).endswith("skull-voice/bank")


def test_line_rule_and_hang_still_passable():
    args = vl.parse_args(["--end-rule", "line", "--hang", "1.2", "--no-speak"])
    assert args.end_rule == "line" and args.hang == 1.2 and args.no_speak


def test_main_report_reads_the_log(tmp_path, capsys):
    log = tmp_path / "turns.jsonl"
    log.write_text("\n".join(json.dumps(r) for r in [row(0.2), row(0.4)]) + "\n")
    assert vl.main(["--report", "10", "--log", str(log)]) == 0
    assert "median 0.300 s, worst 0.400 s, n=2" in capsys.readouterr().out


def test_main_without_moonshine_explains_and_fails(monkeypatch, capsys):
    def missing(arch):
        raise ImportError("moonshine_voice is not installed in this Python.")

    monkeypatch.setattr(vl, "load_stream_maker", missing)
    monkeypatch.setattr(vl, "ERROR_HOLD_S", 0.0)
    assert vl.main(["--no-led"]) == 1
    assert "not installed" in capsys.readouterr().err


def test_read_phrases_skips_blank_lines(tmp_path):
    path = tmp_path / "p.txt"
    path.write_text("One two.\n\n  Three four?  \n")
    assert vl.read_phrases(path) == ["One two.", "Three four?"]


def test_importing_the_loop_needs_no_audio_or_model():
    assert "moonshine_voice" not in sys.modules


# --- M3-05: the spoken reply in the turn, the log and the report --------------------------------

def test_run_turn_keeps_what_speak_returned():
    rec = Recorder()
    marker = object()

    def speak(pieces):
        list(pieces)
        return marker

    turn = vl.run_turn(lambda: heard(), rec.ask, speak, rec.on_state, clock=FakeClock())
    assert turn.speech is marker


def test_run_turn_with_the_real_speaker_and_fakes(tmp_path):
    """The whole M3-05 path inside one turn: Cogitating, the sentences, the guard, the states."""
    rec = Recorder()
    guard = HalfDuplexGuard(tail_s=0.0)
    played = []

    def play(samples):
        played.append((len(samples), guard.is_speaking))

    def synth(text):
        return np.full(2400, 0.1, dtype=np.float32), 24_000

    speaker = Speaker(synth, PhraseBank(tmp_path, bank_voice("javert")), guard, play, on_state=rec.on_state,
                      opener=np.zeros((960, 1), dtype=np.float32))
    turn = vl.run_turn(lambda: heard(), rec.ask, speaker.speak, rec.on_state)
    assert played == [(960, True), (4800, True)]          # Cogitating, then "It is noon.", gate shut for both
    assert rec.states[:3] == ["listening", "thinking", "speaking"] and rec.states[-1] == "idle"
    assert [c.text for c in turn.speech.clips] == ["(cogitating)", "It is noon."]
    assert rec.closed and guard.mic_open()


def speech_log(opener=(13.0, 13.9), clips=((16.0, 17.5), (18.0, 19.0))):
    log = SpeechLog(t_start=12.9)
    log.clips.append(ClipLog("(cogitating)", "opener", t_play_start=opener[0], t_play_end=opener[1]))
    for i, (start, end) in enumerate(clips):
        log.clips.append(ClipLog(f"Sentence {i}.", "pocket", gen_s=2.0, audio_s=end - start, peak_dbfs=-9.0,
                                 t_ready=start, t_play_start=start, t_play_end=end))
    log.misses = len(clips)
    return log


def test_turn_record_has_the_voice_times():
    turn = vl.TurnLog(t_enter=10.0, heard=heard(detector_end=12.5), asked=True, speech=speech_log())
    record = vl.turn_record(turn, "detector")
    speech = record["speech"]
    assert speech["opener_start_s"] == 1.0                 # speech ended at 12.0
    assert speech["first_audio_s"] == 4.0
    assert record["detector_end_to_first_audio_s"] == 3.5
    assert speech["opener_end_to_first_audio_s"] == 2.1
    assert speech["gaps_s"] == [0.5]
    assert speech["sentences"] == 2 and speech["misses"] == 2
    line = vl.summary_line(record)
    assert "first audio 4.00 s" in line and "sentence gaps 0.50 s" in line
    json.dumps(record)


def test_turn_record_without_speech_keeps_the_keys():
    record = vl.turn_record(vl.TurnLog(t_enter=10.0, heard=heard()), "line")
    assert record["speech"] is None and record["detector_end_to_first_audio_s"] is None
    assert "first audio" not in vl.summary_line(record)


def test_report_gives_first_audio_and_gaps():
    rows = []
    for first, gaps in ((4.0, [0.5]), (6.0, [0.2, 1.0]), (5.0, [])):
        clips = [(12.0 + first, 13.0 + first)] + [(14.0 + first + i, 14.5 + first + i) for i in range(len(gaps))]
        turn = vl.TurnLog(t_enter=10.0, heard=heard(detector_end=12.5), asked=True, speech=speech_log(clips=clips))
        record = vl.turn_record(turn, "detector")
        record["speech"]["gaps_s"] = gaps
        rows.append(record)
    text = vl.report(rows)
    assert "speech end -> first reply audio: median 5.000 s, worst 6.000 s, n=3" in text
    assert "detector end -> first reply audio: median 4.500 s, worst 5.500 s, n=3" in text
    assert "gap between sentences (all gaps): median 0.500 s, worst 1.000 s, n=3" in text
    assert "spoken turns 3: with a speech error 0" in text


def test_main_with_voice_but_no_tts_python_fails_cleanly(monkeypatch, tmp_path, capsys):
    monkeypatch.setattr(vl, "load_stream_maker", lambda arch: (lambda: None))
    monkeypatch.setattr(vl, "ERROR_HOLD_S", 0.0)
    monkeypatch.setitem(sys.modules, "software.skull_core.record", type(sys)("fake_record"))
    sys.modules["software.skull_core.record"].SAMPLE_RATE = 48_000
    assert vl.main(["--no-led", "--tts-python", str(tmp_path / "no-python"), "--bank", str(tmp_path / "bank")]) == 1
    assert "cannot start the TTS worker" in capsys.readouterr().err


# --- M3-06 wiring and M3-07: the closing state, the offline path, the triggers --------------------

class Probe:
    """A fake OnlineProbe.start: records when it was started, answers `online`."""

    def __init__(self, rec, online):
        self.rec = rec
        self.online = online

    def __call__(self):
        self.rec.states.append("(probe started)")
        return lambda: self.online


def test_offline_turn_says_the_line_and_never_calls_the_brain():
    rec = Recorder()
    said = []

    def say_offline(text):
        said.append(text)
        return "offline speech"

    turn = vl.run_turn(lambda: heard(), rec.ask, rec.speak, rec.on_state, probe=Probe(rec, False),
                       say_offline=say_offline)
    assert rec.states == ["listening", "(probe started)", "offline", "offline"]    # never "idle"
    assert rec.asked == [] and said == ["What time is it?"]
    assert turn.offline and not turn.asked and turn.end_state == "offline" and turn.speech == "offline speech"


def test_online_turn_is_a_normal_turn_and_the_probe_starts_before_listening():
    rec = Recorder()
    order = []

    def listen():
        order.append(list(rec.states))
        return heard()

    turn = vl.run_turn(listen, rec.ask, rec.speak, rec.on_state, probe=Probe(rec, True))
    assert order == [["listening", "(probe started)"]]        # it runs while the person speaks
    assert rec.states[-2:] == ["thinking", "idle"] and rec.asked == ["What time is it?"] and not turn.offline


def test_no_probe_without_a_brain_or_without_text():
    rec = Recorder()
    vl.run_turn(lambda: heard(), None, rec.speak, rec.on_state, probe=Probe(rec, False))
    assert "(probe started)" not in rec.states and rec.states[-1] == "idle"
    rec = Recorder()
    vl.run_turn(lambda: heard(text=None), rec.ask, rec.speak, rec.on_state, probe=Probe(rec, False))
    assert rec.states == ["listening", "(probe started)", "idle"]       # silence: not an offline turn


def test_brain_failure_ends_on_its_state_with_no_idle_before_it():
    rec = Recorder()
    turn = vl.run_turn(lambda: heard(), rec.ask, rec.speak, rec.on_state, end_state=lambda t: "offline")
    assert rec.states == ["listening", "thinking", "offline"] and turn.end_state == "offline"


def test_end_state_none_means_idle():
    rec = Recorder()
    vl.run_turn(lambda: heard(), rec.ask, rec.speak, rec.on_state, end_state=lambda t: None)
    assert rec.states[-1] == "idle"


def test_offline_with_no_voice_still_ends_offline():
    rec = Recorder()
    turn = vl.run_turn(lambda: heard(), rec.ask, rec.speak, rec.on_state, probe=Probe(rec, False))
    assert turn.offline and turn.speech is None and rec.states[-1] == "offline"


def fake_brain(kind=None, rate_limit_status=None):
    record = TurnRecord(started_at="", model="m", kind=kind, rate_limit_status=rate_limit_status)
    if kind is not None:
        record.fallback = FALLBACK_LINE
    return SimpleNamespace(last=record)


@pytest.mark.parametrize("kind, status, state", [
    (None, None, None), ("timeout", None, "offline"), ("exit", None, "offline"), ("not_found", None, "error"),
    ("refusal", None, None), ("error", "rejected", None)])
def test_brain_end_state(kind, status, state):
    end_state = vl.brain_end_state(fake_brain(kind, status))
    assert end_state(vl.TurnLog(t_enter=0.0, asked=True)) == state
    assert end_state(vl.TurnLog(t_enter=0.0, asked=False)) is None


def test_fallback_clip_state_marks_only_the_fallback_line():
    clip_state = vl.fallback_clip_state(fake_brain("timeout"))
    assert clip_state("I can’t reach my brain.") == "offline"          # curly apostrophe, same line
    assert clip_state("It is noon.") == "speaking"
    assert vl.fallback_clip_state(fake_brain(None))(FALLBACK_LINE) == "speaking"     # a real reply
    assert vl.fallback_clip_state(fake_brain("refusal"))(FALLBACK_LINE) == "speaking"
    assert vl.fallback_clip_state(SimpleNamespace(last=None))("x") == "speaking"


def test_backstop_turn_with_the_real_speaker_shows_offline_during_the_fallback_line(tmp_path):
    """Probe said online, then the brain timed out: the fallback line plays under "offline", ending offline."""
    rec = Recorder()
    guard = HalfDuplexGuard(tail_s=0.0)
    brain = fake_brain()

    def ask(text):
        brain.last.kind, brain.last.fallback = "timeout", FALLBACK_LINE     # what Brain.ask does before yielding it
        yield FALLBACK_LINE

    bank = PhraseBank(tmp_path, bank_voice("javert"))
    bank.add(FALLBACK_LINE, np.zeros((4800, 1), dtype=np.float32))
    speaker = Speaker(None, bank, guard, lambda samples: None, on_state=rec.on_state,
                      opener=np.zeros((960, 1), dtype=np.float32), clip_state=vl.fallback_clip_state(brain))
    turn = vl.run_turn(lambda: heard(), ask, speaker.speak, rec.on_state, probe=Probe(rec, True),
                       end_state=vl.brain_end_state(brain))
    states = [s for s in rec.states if s != "(probe started)"]
    assert states[:3] == ["listening", "thinking", "speaking"]          # Cogitating is "speaking"
    assert "offline" in states and states[-1] == "offline" and "idle" not in states
    assert states.index("offline") > states.index("speaking")
    assert turn.speech.clips[-1].source == "bank"


def test_trigger_flags():
    assert vl.parse_args([]).trigger == "both"
    assert vl.parse_args(["--phrases", "p.txt"]).trigger == "enter"
    assert vl.parse_args(["--trigger", "wake"]).trigger == "wake"
    args = vl.parse_args([])
    assert args.wake_window == 6.0 and args.probe_host == "api.anthropic.com" and args.probe_timeout == 2.0
    assert args.pins == [5, 6, 13] and not args.no_led and not args.watch_state


@pytest.mark.parametrize("argv", [["--phrases", "p.txt", "--trigger", "wake"], ["--wake-window", "0"],
                                  ["--wake-window", "20"], ["--probe-timeout", "0"], ["--probe-port", "0"],
                                  ["--blink-period", "0"], ["--trigger", "clap"]])
def test_bad_m307_arguments_are_refused(argv):
    with pytest.raises(SystemExit):
        vl.parse_args(argv)


class FakeMicStream:
    fail = False

    def __init__(self, device, guard, rate, settle_s=0.0):
        self.guard = guard

    def open(self):
        if FakeMicStream.fail:
            raise MicError("no mic called UACDemoV1.0")

    def ready(self):
        return True

    def close(self):
        pass

    def stats(self):
        return MicStats()


def fake_hardware(monkeypatch):
    monkeypatch.setattr(vl, "load_stream_maker", lambda arch: (lambda: None))
    monkeypatch.setitem(sys.modules, "software.skull_core.record", type(sys)("fake_record"))
    sys.modules["software.skull_core.record"].SAMPLE_RATE = 48_000
    monkeypatch.setattr(vl, "MicStream", FakeMicStream)
    monkeypatch.setattr(vl, "ERROR_HOLD_S", 0.0)


def test_main_mic_failure_shows_error_after_thinking(monkeypatch, capsys):
    fake_hardware(monkeypatch)
    monkeypatch.setattr(FakeMicStream, "fail", True)
    assert vl.main(["--no-led", "--watch-state", "--no-brain"]) == 1
    captured = capsys.readouterr()
    shown = [line.split()[-1] for line in captured.out.splitlines() if line.startswith("[state")]
    assert shown == ["thinking", "error"]          # start-up, then the failure; the watcher is attached first
    assert "no mic called" in captured.err


def test_main_wake_one_breath_while_offline(monkeypatch, capsys, tmp_path):
    """Wake mode end to end with fakes: the wake + question in one breath, the probe refused, the canned line."""
    fake_hardware(monkeypatch)
    with socket.socket() as s:                       # a local port with nothing listening: refused at once
        s.bind(("127.0.0.1", 0))
        closed_port = s.getsockname()[1]
    script = [Heard(text="Awaken machine spirit. What time is it?",
                    raw_text="Awaken machine spirit. What time is it?", end_reason="detector", lines=2)]

    def fake_listen_once(*args, **kwargs):
        if not script:
            raise KeyboardInterrupt
        return script.pop(0)

    monkeypatch.setattr(vl, "listen_once", fake_listen_once)
    brain_dir = tmp_path / "brain"
    brain_dir.mkdir()
    log, lines = tmp_path / "turns.jsonl", tmp_path / "lines.jsonl"
    code = vl.main(["--no-led", "--watch-state", "--trigger", "wake", "--no-speak", "--brain-dir", str(brain_dir),
                    "--claude", str(tmp_path / "no-claude"), "--probe-host", "127.0.0.1",
                    "--probe-port", str(closed_port), "--log", str(log), "--line-log", str(lines)])
    out = capsys.readouterr().out
    assert code == 0
    watcher = [line for line in out.splitlines() if line.startswith("[state")]
    assert [line.split()[-1] for line in watcher] == ["thinking", "idle", "listening", "offline", "offline"]
    assert FALLBACK_LINE in out
    record = json.loads(log.read_text())
    assert record["trigger"] == "wake" and record["offline"] and not record["asked"]
    assert record["text"] == "What time is it?" and record["end_state"] == "offline"
    assert json.loads(lines.read_text())["woke"] is True


def test_main_without_gpiozero_explains_and_fails(monkeypatch, capsys):
    monkeypatch.setitem(sys.modules, "gpiozero", None)          # import gpiozero -> ImportError
    assert vl.main(["--no-brain"]) == 1
    assert "pass --no-led" in capsys.readouterr().err
