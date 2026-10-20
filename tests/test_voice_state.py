"""Tests for software/skull_core/voice_state.py: publish_state, the LED follower, brain failures, the demo."""
import re
import sys
import threading
from pathlib import Path

import pytest

from software.skull_core import voice_state as vs
from software.skull_core.skull_bridge import Bridge, load_key_prefix
from software.skull_core.status_leds import STATES, StatusLight
from software.skull_core.voice_state import (eye_led, led_follows, print_watcher, publish_state, run_script,
                                             state_after_brain, voice_key)

KEY = "skull/skull-01/state/voice"


class FakeClock:
    def __init__(self, t=100.0):
        self.t = t

    def __call__(self):
        return self.t


class FakeLed:
    def __init__(self):
        self.value = None
        self.calls = []

    def off(self):
        self.calls.append("off")

    def close(self):
        self.calls.append("close")


def make_light():
    return StatusLight(FakeLed(), clock=FakeClock())


# --- key ------------------------------------------------------------------------------------------

def test_voice_key_comes_from_the_yaml(tmp_path):
    params = tmp_path / "robot_params.yaml"
    params.write_text("identity:\n  key_prefix: skull/skull-02\n", encoding="utf-8")
    assert voice_key(params) == "skull/skull-02/state/voice"


def test_repo_voice_key():
    assert voice_key() == load_key_prefix() + "/state/voice"


def test_no_robot_id_literal_in_the_source():
    assert not re.search(r"skull-\d+", Path(vs.__file__).read_text(encoding="utf-8"))


# --- publish_state and the LED --------------------------------------------------------------------

def test_publish_state_publishes_the_bare_name():
    bridge = Bridge()
    publish_state(bridge, KEY)("listening")
    assert bridge.last(KEY) == "listening"


@pytest.mark.parametrize("name", ["test", "quit", "Idle", "", "sleeping"])
def test_unknown_names_are_published_as_error(name):
    bridge = Bridge()
    publish_state(bridge, KEY)(name)
    assert bridge.last(KEY) == "error"


def test_led_follows_every_state():
    bridge = Bridge()
    light = make_light()
    led_follows(bridge, KEY, light)
    on_state = publish_state(bridge, KEY)
    for name in STATES:
        on_state(name)
        assert light.state == name


def test_led_ignores_other_keys():
    bridge = Bridge()
    light = make_light()
    led_follows(bridge, KEY, light)
    bridge.publish("skull/skull-02/state/voice", "speaking")
    assert light.state == "idle"


def test_a_broken_led_does_not_stop_the_publisher_or_the_watcher():
    reports, lines = [], []
    bridge = Bridge(report=reports.append)

    class BrokenLight:
        def set_state(self, _name):
            raise OSError("GPIO busy")

    led_follows(bridge, KEY, BrokenLight())
    print_watcher(bridge, KEY, clock=FakeClock(), out=lines.append)
    publish_state(bridge, KEY)("thinking")
    assert len(reports) == 1 and "GPIO busy" in reports[0]
    assert lines == [f"[state +  0.000s] {KEY} thinking"]


def test_led_from_two_threads_ends_on_the_last_published_state():
    bridge = Bridge()
    light = make_light()
    led_follows(bridge, KEY, light)
    on_state = publish_state(bridge, KEY)
    names = list(STATES)

    def worker(offset):
        for i in range(300):
            on_state(names[(i + offset) % len(names)])

    threads = [threading.Thread(target=worker, args=(n,)) for n in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert light.state == bridge.last(KEY)


# --- run_turn with the bridge (voice_loop.py is M3-04/M3-05 code, read only here) -----------------

def turn_states(listen_text, ask_reply=None, ask_raises=None):
    """Run voice_loop.run_turn with publish_state as on_state; return what the LED was set to."""
    from software.skull_core.stt import Heard
    from software.skull_core.voice_loop import run_turn

    bridge = Bridge()
    shown = []

    class RecordingLight:
        def set_state(self, name):
            shown.append(name)

    led_follows(bridge, KEY, RecordingLight())
    fields = dict(raw_text=listen_text or "", end_reason="line", lines=1, t_first_sample=0.0, t_speech_end=1.0,
                  t_text=1.2, stt_s=0.2, first_line_text=listen_text, first_line_stt_s=0.2)

    def listen():
        return Heard(text=listen_text, **fields)

    def ask(_text):
        if ask_raises is not None:
            raise ask_raises
        yield from ask_reply or ["Noon."]

    def speak(pieces):
        list(pieces)

    try:
        run_turn(listen, ask, speak, publish_state(bridge, KEY))
    except Exception:
        pass
    return shown, bridge.last(KEY)


def test_normal_turn_sequence_through_the_bridge():
    shown, last = turn_states("What time is it?")
    assert shown[0] == "listening" and "thinking" in shown and shown[-1] == "idle"
    assert last == "idle"


def test_silent_turn_sequence_through_the_bridge():
    shown, last = turn_states(None)
    assert shown == ["listening", "idle"] and last == "idle"


def test_turn_that_raises_ends_on_error():
    """M3-07 closed the M3-06 gap: the turn's last state is "error", with no "idle" just before it."""
    shown, last = turn_states("Hello", ask_raises=RuntimeError("brain blew up"))
    assert shown[-1] == "error" and last == "error" and "idle" not in shown


# --- brain failures (decision 3, provisional) -----------------------------------------------------

@pytest.mark.parametrize("kind, status, state", [
    (None, None, None),
    ("refusal", None, None),
    ("empty", None, None),
    ("abandoned", None, None),
    ("timeout", None, "offline"),
    ("exit", None, "offline"),
    ("error", None, "offline"),
    ("error", "rejected", None),          # over the limit: a spoken line, not offline
    ("setup", None, "error"),
    ("not_found", None, "error"),
    ("something_new", None, "error"),
])
def test_state_after_brain(kind, status, state):
    assert state_after_brain(kind, status) == state


def test_every_brain_kind_is_mapped_on_purpose():
    from software.skull_core.brain import KINDS
    mapped = set(vs.UNREACHABLE_KINDS) | set(vs.OWN_FAULT_KINDS) | {"refusal", "empty", "abandoned"}
    assert set(KINDS) == mapped


# --- demo -----------------------------------------------------------------------------------------

def test_run_script_publishes_in_order_and_speaking_from_another_thread():
    calls = []
    main_thread = threading.get_ident()

    def on_state(name):
        calls.append((name, threading.get_ident() == main_thread))

    slept = []
    run_script(on_state, vs.SCRIPTED_TURN, sleep=slept.append)
    assert [c[0] for c in calls] == [name for name, _ in vs.SCRIPTED_TURN]
    assert all(on_main == (name != "speaking") for name, on_main in calls)
    assert slept == [seconds for _, seconds in vs.SCRIPTED_TURN]


def test_main_no_led_states_demo(capsys):
    assert vs.main(["--demo", "states", "--no-led"], sleep=lambda s: None) == 0
    out = capsys.readouterr().out
    key = voice_key()
    assert f"Publishing on {key}" in out
    assert [line.split()[-1] for line in out.splitlines() if line.startswith("[state")] == list(STATES)


def test_main_no_led_needs_no_gpio(monkeypatch):
    monkeypatch.setitem(sys.modules, "gpiozero", None)   # importing it would fail
    assert vs.main(["--demo", "turn", "--no-led"], sleep=lambda s: None) == 0


@pytest.mark.parametrize("argv", [["--demo", "states", "--seconds", "0"], ["--demo", "nope"], [],
                                  ["--demo", "states", "--blink-period", "-1"]])
def test_bad_arguments_are_refused(argv):
    with pytest.raises(SystemExit):
        vs.parse_args(argv)


def test_bad_params_file_fails_cleanly(tmp_path, capsys):
    assert vs.main(["--demo", "states", "--no-led", "--params", str(tmp_path / "missing.yaml")]) == 1
    assert "cannot read" in capsys.readouterr().err


# --- eye_led --------------------------------------------------------------------------------------

def test_eye_led_follows_inside_and_releases_on_exit():
    bridge = Bridge()
    fake = FakeLed()
    with eye_led(bridge, KEY, make_led=lambda: fake) as light:
        assert light.state == "thinking"                  # nothing published yet: start-up
        publish_state(bridge, KEY)("listening")
        assert light.state == "listening"
        assert any(t.name == "eye-led" and t.is_alive() for t in threading.enumerate())
    assert fake.calls == ["off", "close"]
    assert not any(t.name == "eye-led" and t.is_alive() for t in threading.enumerate())


def test_eye_led_starts_on_the_last_published_state():
    bridge = Bridge()
    publish_state(bridge, KEY)("offline")
    with eye_led(bridge, KEY, make_led=FakeLed) as light:
        assert light.state == "offline"


def test_eye_led_releases_the_pins_when_the_block_raises():
    bridge = Bridge()
    fake = FakeLed()
    with pytest.raises(RuntimeError):
        with eye_led(bridge, KEY, make_led=lambda: fake):
            raise RuntimeError("mic gone")
    assert fake.calls == ["off", "close"]
