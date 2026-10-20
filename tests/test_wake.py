"""Tests for software/skull_core/wake.py: the wake rule, the trigger wait, the line log, the offline probe."""
import io
import json
import socket
import threading
import time

import pytest

from software.skull_core import wake
from software.skull_core.stt import Heard
from software.skull_core.wake import (KeyReader, LineLog, OnlineProbe, is_online, marked_text, split_wake,
                                      strip_wake, wait_for_trigger)


# --- split_wake: texts actually seen (M3-00 (d), F-027) and the cases the rule must refuse --------------

@pytest.mark.parametrize("text, woke, question", [
    ("Awaken machine spirit.", True, ""),
    ("Awaken machine pirate.", True, ""),                      # "spirit" heard as pirate (Pi, 2026-10-07)
    ("Awaken machine, a pirate.", True, ""),                   # the comma before the first word is not a cut
    ("Awaken machine, Sir Pirate.", True, ""),
    ("awakened machine spirit acknowledged.", True, ""),       # TTS take: "awakened", two words left
    ("Awaken machines spirit", True, ""),
    ("Awaken the machine spirit, what is the temperature of your brain?", True,
     "what is the temperature of your brain?"),
    ("Awaken machine spirit. What is the temperature of your brain?", True,
     "What is the temperature of your brain?"),
    ("Awaken machine, sir pirate. What time is it?", True, "What time is it?"),
    ("Awaken machine spirit what time is it", True, "spirit what time is it"),   # no mark: all of it
    ("  ...Awaken machine spirit!", True, ""),
    ("AWAKEN MACHINE SPIRIT. Hello there.", True, "Hello there."),
    ("I said awaken machine spirit yesterday.", False, ""),    # contains it, doesn't start with it
    ("Awaken the beast.", False, ""),
    ("Awaken machinery now.", False, ""),                      # \b: "machinery" is not "machine"
    ("Bugün hava çok güzel, İstanbul'a gidiyoruz.", False, ""),
    ("", False, ""),
])
def test_split_wake(text, woke, question):
    assert split_wake(text) == (woke, question)


def test_marked_text_turns_line_breaks_into_cut_marks():
    assert marked_text(["Awaken machine spirit", " What time is it? ", ""]) == \
        "Awaken machine spirit. What time is it?"
    woke, question = split_wake(marked_text(["Awaken machine spirit", "what time is it"]))
    assert woke and question == "what time is it."


def heard(raw, text="same", **kwargs):
    fields = dict(end_reason="detector", lines=1)
    fields.update(kwargs)
    return Heard(text=raw if text == "same" else text, raw_text=raw, **fields)


def test_strip_wake_removes_a_repeated_wake_from_the_window():
    assert strip_wake(heard("Awaken machine spirit. What time is it?"), 2).text == "What time is it?"
    assert strip_wake(heard("Awaken machine spirit."), 2).text is None        # said it again, nothing else
    assert strip_wake(heard("What time is it?"), 2).text == "What time is it?"
    assert strip_wake(heard("", text=None), 2).text is None


# --- wait_for_trigger ------------------------------------------------------------------------------

class FakeIdle:
    """listen_idle with a script of utterances; records the interrupt it was given."""

    def __init__(self, script):
        self.script = list(script)
        self.lines = []
        self.interrupts = []

    def __call__(self, interrupt):
        self.interrupts.append(interrupt)
        raw, lines = self.script.pop(0)
        self.lines = lines
        return heard(raw, text=raw or None, end_reason="detector" if raw else "no_speech")

    def line_texts(self):
        return self.lines


def test_idle_lines_are_logged_and_only_a_wake_starts_a_turn(tmp_path):
    idle = FakeIdle([("", []), ("Merhaba nasılsın", ["Merhaba nasılsın"]),
                     ("I want to awaken machine spirit", ["I want to awaken machine spirit"]),
                     ("Awaken machine spirit. What time is it?", ["Awaken machine spirit.", "What time is it?"])])
    log = LineLog(tmp_path / "lines.jsonl")
    trigger = wait_for_trigger(idle, None, idle.line_texts, log, 2, clock=lambda: 0.0, out=lambda text: None)
    assert trigger.kind == "wake"
    assert trigger.question.text == "What time is it?"
    assert trigger.heard.raw_text == "Awaken machine spirit. What time is it?"
    rows = [json.loads(line) for line in (tmp_path / "lines.jsonl").read_text().splitlines()]
    assert [r["woke"] for r in rows] == [False, False, True]          # the silent one is not logged
    assert rows[2]["question"] == "What time is it?" and rows[0]["question"] is None
    assert rows[2]["lines"] == ["Awaken machine spirit.", "What time is it?"]


def test_two_lines_without_marks_still_split(tmp_path):
    idle = FakeIdle([("Awaken machine spirit what time is it", ["Awaken machine spirit", "what time is it"])])
    trigger = wait_for_trigger(idle, None, idle.line_texts, LineLog(None), 2, out=lambda text: None)
    assert trigger.question.text == "what time is it."


def test_bare_wake_has_no_question():
    idle = FakeIdle([("Awaken machine pirate.", ["Awaken machine pirate."])])
    trigger = wait_for_trigger(idle, None, idle.line_texts, LineLog(None), 2, out=lambda text: None)
    assert trigger.kind == "wake" and trigger.question is None


def test_wake_with_too_short_a_question_counts_as_bare():
    idle = FakeIdle([("Awaken machine spirit. Uh. Hm.", ["Awaken machine spirit. Uh. Hm."])])
    trigger = wait_for_trigger(idle, None, idle.line_texts, LineLog(None), 5, out=lambda text: None)
    assert trigger.question is None


def test_enter_wins_when_a_key_is_waiting():
    keys = KeyReader(io.StringIO("\n")).start()
    keys._thread.join(1)
    idle = FakeIdle([])
    trigger = wait_for_trigger(idle, keys, idle.line_texts, LineLog(None), 2, out=lambda text: None)
    assert trigger.kind == "enter" and idle.interrupts == []


def test_q_quits_and_enter_only_mode_needs_no_mic():
    keys = KeyReader(io.StringIO("q\n")).start()
    assert wait_for_trigger(None, keys, list, LineLog(None), 2).kind == "quit"
    keys = KeyReader(io.StringIO("")).start()             # end of input in Enter-only mode = quit
    assert wait_for_trigger(None, keys, list, LineLog(None), 2).kind == "quit"


def test_end_of_input_in_both_mode_keeps_the_wake():
    keys = KeyReader(io.StringIO("")).start()
    keys._thread.join(1)
    idle = FakeIdle([("Awaken machine spirit.", ["Awaken machine spirit."])])
    trigger = wait_for_trigger(idle, keys, idle.line_texts, LineLog(None), 2, out=lambda text: None)
    assert trigger.kind == "wake" and keys.closed


def test_idle_listen_gets_the_key_check_as_interrupt():
    keys = KeyReader(io.StringIO("")).start()
    keys._thread.join(1)
    keys.lines.get_nowait()                     # pretend input is still open and empty
    idle = FakeIdle([("Awaken machine spirit.", ["Awaken machine spirit."])])
    wait_for_trigger(idle, keys, idle.line_texts, LineLog(None), 2, out=lambda text: None)
    assert idle.interrupts == [keys.pending]


def test_drop_stale_forgets_typed_lines_but_sees_q():
    keys = KeyReader(io.StringIO("\n\n")).start()
    keys._thread.join(1)
    assert keys.drop_stale() is False and keys.closed and not keys.pending()
    keys = KeyReader(io.StringIO("\nq\n")).start()
    keys._thread.join(1)
    assert keys.drop_stale() is True


def test_line_log_write_error_is_reported_not_raised(tmp_path):
    blocker = tmp_path / "file"
    blocker.write_text("x")
    said = []
    log = LineLog(blocker / "lines.jsonl", report=said.append)
    log.write({"text": "hi"})
    assert log.rows == 1 and "Could not write" in said[0]


def test_no_audio_is_reported_once():
    idle = FakeIdle([("", []), ("", []), ("Awaken machine spirit.", ["Awaken machine spirit."])])

    def no_audio(interrupt):
        result = idle(interrupt)
        if not result.raw_text:
            result.end_reason = "no_audio"
        return result

    said = []
    wait_for_trigger(no_audio, None, idle.line_texts, LineLog(None), 2, out=said.append)
    assert sum("No audio" in line for line in said) == 1


def test_needs_a_trigger():
    with pytest.raises(ValueError):
        wait_for_trigger(None, None, list, LineLog(None), 2)


# --- the offline probe -----------------------------------------------------------------------------

class FakeSocket:
    def __init__(self):
        self.closed = False

    def close(self):
        self.closed = True


def test_is_online_when_the_connect_succeeds_and_closes_the_socket():
    sock = FakeSocket()
    calls = []

    def connect(address, timeout):
        calls.append((address, timeout))
        return sock

    assert is_online("example.invalid", 443, 0.5, connect=connect) is True
    assert calls == [(("example.invalid", 443), 0.5)] and sock.closed


@pytest.mark.parametrize("error", [ConnectionRefusedError("refused"), socket.timeout("timed out"),
                                   socket.gaierror("no name"), OSError(101, "Network is unreachable")])
def test_is_online_false_on_every_network_error(error):
    def connect(address, timeout):
        raise error

    assert is_online("h", 1, 0.5, connect=connect) is False


def test_is_online_never_waits_longer_than_its_timeout():
    """A name lookup that hangs (WiFi up, no internet) must not hang the turn."""
    release = threading.Event()

    def stuck(address, timeout):
        release.wait(5)
        return FakeSocket()

    t0 = time.monotonic()
    assert is_online("h", 1, 0.2, connect=stuck) is False
    waited = time.monotonic() - t0
    release.set()
    assert 0.15 < waited < 1.0


def test_probe_runs_in_the_background_and_answers_later():
    started = threading.Event()
    release = threading.Event()

    def check():
        started.set()
        release.wait(5)
        return True

    answer = OnlineProbe(check).start()
    assert started.wait(1)                 # running while the caller does other things
    release.set()
    assert answer() is True


def test_a_raising_check_counts_as_offline():
    def broken():
        raise RuntimeError("bug")

    assert OnlineProbe(broken).start()() is False


def test_probe_defaults_are_documented_choices():
    assert wake.DEFAULT_PROBE_HOST == "api.anthropic.com" and wake.DEFAULT_PROBE_PORT == 443
    assert wake.DEFAULT_PROBE_TIMEOUT_S == 2.0 and wake.DEFAULT_WAKE_WINDOW_S == 6.0


def test_importing_wake_needs_no_audio_model_or_gpio():
    import subprocess
    import sys
    code = ("import sys, software.skull_core.wake; "
            "print([m for m in ('sounddevice', 'moonshine_voice', 'gpiozero', 'vosk') if m in sys.modules])")
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True).stdout
    assert out.strip() == "[]"
