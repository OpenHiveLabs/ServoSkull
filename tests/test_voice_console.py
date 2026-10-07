import json
import math
import subprocess
import sys
import types
import wave
from pathlib import Path

import numpy as np
import pytest

from software.skull_core import voice_console as vc
from software.skull_core.status_leds import STATES, StatusLight

REPO_ROOT = Path(__file__).resolve().parents[1]


class FakeClock:
    def __init__(self, t=100.0):
        self.t = t

    def __call__(self):
        return self.t


class FakeRecognizer:
    """Stands in for vosk.KaldiRecognizer: remembers what it was fed, returns a canned result."""

    def __init__(self, text):
        self.text = text
        self.fed = []
        self.final_calls = 0

    def AcceptWaveform(self, data):
        assert isinstance(data, bytes)
        self.fed.append(data)
        return 0

    def FinalResult(self):
        self.final_calls += 1
        return json.dumps({"text": self.text})


# --- importing needs no hardware ------------------------------------------------------------

def test_import_and_help_need_no_audio_model_or_gpio():
    code = (
        "import sys\n"
        "from software.skull_core import voice_console\n"
        "bad = [m for m in ('vosk', 'sounddevice', 'gpiozero') if m in sys.modules]\n"
        "assert not bad, bad\n"
    )
    subprocess.run([sys.executable, "-c", code], cwd=REPO_ROOT, check=True)
    out = subprocess.run([sys.executable, "-m", "software.skull_core.voice_console", "--help"],
                         cwd=REPO_ROOT, capture_output=True, text=True)
    assert out.returncode == 0 and "--model-dir" in out.stdout


# --- TODO 1: grammar -----------------------------------------------------------------------

def test_grammar_is_the_six_states_then_unk():
    assert json.loads(vc.grammar_json()) == [*STATES, "[unk]"]


def test_grammar_has_no_console_commands():
    words = json.loads(vc.grammar_json())
    assert "test" not in words and "quit" not in words


# --- TODO 2: parse_result ------------------------------------------------------------------

@pytest.mark.parametrize("state", list(STATES))
def test_each_state_name_is_accepted(state):
    assert vc.parse_result(json.dumps({"text": state})) == state


@pytest.mark.parametrize("text", ["", "[unk]", "idle thinking", "quit", "test", "hello"])
def test_non_states_give_none(text):
    assert vc.parse_result(json.dumps({"text": text})) is None


def test_surrounding_spaces_are_ignored():
    assert vc.parse_result('{"text": "  error "}') == "error"


@pytest.mark.parametrize("raw", ["", "not json", "{}", '{"partial": "idle"}'])
def test_malformed_results_give_none_not_a_crash(raw):
    assert vc.parse_result(raw) is None


# --- TODO 3: recognise ---------------------------------------------------------------------

def test_recognise_feeds_int16_bytes_and_reads_the_final_result():
    samples = np.array([1, -2, 300, -32768, 32767], dtype=np.int16)
    rec = FakeRecognizer("offline")
    assert vc.recognise(rec, samples) == "offline"
    assert b"".join(rec.fed) == samples.tobytes()
    assert rec.final_calls == 1


def test_recognise_unknown_is_none():
    assert vc.recognise(FakeRecognizer("[unk]"), np.zeros(480, dtype=np.int16)) is None


# --- TODO 4, 5: half-duplex guard ----------------------------------------------------------

def test_guard_starts_open():
    guard = vc.HalfDuplexGuard(0.3, clock=FakeClock())
    assert guard.mic_open() and guard.seconds_until_open() == 0.0


def test_guard_is_shut_while_speaking():
    guard = vc.HalfDuplexGuard(0.3, clock=FakeClock())
    guard.start_speaking()
    assert not guard.mic_open()
    assert guard.seconds_until_open() == math.inf


def test_guard_stays_shut_for_the_tail_then_opens():
    clock = FakeClock(10.0)
    guard = vc.HalfDuplexGuard(0.3, clock=clock)
    guard.start_speaking()
    guard.stop_speaking()
    assert not guard.is_speaking
    clock.t = 10.1
    assert not guard.mic_open()
    assert guard.seconds_until_open() == pytest.approx(0.2)
    clock.t = 10.3
    assert guard.mic_open()                     # at exactly the end of the tail it's open
    clock.t = 99.0
    assert guard.seconds_until_open() == 0.0    # never negative


def test_guard_reopens_even_if_playback_raises():
    clock = FakeClock(10.0)
    guard = vc.HalfDuplexGuard(0.0, clock=clock)
    with pytest.raises(RuntimeError):
        with guard.speaking():
            assert not guard.mic_open()
            raise RuntimeError("amp unplugged")
    assert guard.mic_open()


def test_guard_rejects_negative_tail():
    with pytest.raises(ValueError):
        vc.HalfDuplexGuard(-0.1)


# --- TODO 6: one push-to-talk turn ---------------------------------------------------------

def make_turn(heard, tail=0.3, t=100.0):
    clock = FakeClock(t)
    light = StatusLight(None, clock=clock)
    guard = vc.HalfDuplexGuard(tail, clock=clock)
    log = []

    def listen():
        log.append(("listen", light.state, guard.mic_open()))
        return heard

    def say(state):
        log.append(("say", state, light.state, guard.mic_open()))

    def sleep(s):
        log.append(("sleep", s))
        clock.t += s

    return clock, light, guard, log, listen, say, sleep


def test_turn_shows_listening_then_the_state_and_speaks_it():
    clock, light, guard, log, listen, say, sleep = make_turn("thinking")
    assert vc.push_to_talk_turn(light, guard, listen, say, sleep) == "thinking"
    assert log == [("listen", "listening", True), ("say", "thinking", "thinking", False)]
    assert light.state == "thinking"


def test_mic_is_shut_right_after_the_skull_speaks():
    clock, light, guard, log, listen, say, sleep = make_turn("error", tail=0.3)
    vc.push_to_talk_turn(light, guard, listen, say, sleep)
    assert not guard.is_speaking
    assert guard.seconds_until_open() == pytest.approx(0.3)


def test_turn_waits_for_the_tail_before_listening():
    clock, light, guard, log, listen, say, sleep = make_turn("idle", tail=0.3)
    vc.push_to_talk_turn(light, guard, listen, say, sleep)     # speaks: mic shut for 0.3 s
    clock.t += 0.1                                              # Enter pressed 0.1 s later
    log.clear()
    vc.push_to_talk_turn(light, guard, listen, say, sleep)
    assert log[0][0] == "sleep" and log[0][1] == pytest.approx(0.2)
    assert log[1] == ("listen", "listening", True)


def test_no_sleep_when_the_mic_is_already_open():
    clock, light, guard, log, listen, say, sleep = make_turn("idle")
    vc.push_to_talk_turn(light, guard, listen, say, sleep)
    assert not any(entry[0] == "sleep" for entry in log)


def test_unknown_word_restores_the_light_and_says_nothing():
    clock, light, guard, log, listen, say, sleep = make_turn(None)
    light.set_state("offline")
    assert vc.push_to_talk_turn(light, guard, listen, say, sleep) is None
    assert light.state == "offline"
    assert [entry[0] for entry in log] == ["listen"]
    assert guard.mic_open()


# --- main, with every device faked ---------------------------------------------------------

def write_wav(path, rate=48_000):
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(rate)
        w.writeframes(np.zeros(480, dtype=np.int16).tobytes())


@pytest.fixture
def fake_pi(tmp_path, monkeypatch):
    """A model folder, a voice folder, and fake vosk / sounddevice / gpiozero / record."""
    model = tmp_path / "model"
    (model / "conf").mkdir(parents=True)
    (model / "conf" / "model.conf").write_text("")
    voice = tmp_path / "voice"
    voice.mkdir()
    for state in STATES:
        write_wav(voice / f"{state}.wav")

    calls = {"play": [], "record": [], "grammar": None, "rate": None, "heard": "speaking"}

    vosk = types.ModuleType("vosk")
    vosk.SetLogLevel = lambda level: None
    vosk.Model = lambda path: ("model", path)

    def kaldi(model_obj, rate, grammar):
        calls["rate"], calls["grammar"] = rate, grammar
        return FakeRecognizer(calls["heard"])

    vosk.KaldiRecognizer = kaldi

    sd = types.ModuleType("sounddevice")
    sd.PortAudioError = type("PortAudioError", (Exception,), {})
    sd.play = lambda data, **kw: calls["play"].append((data, kw))
    sd.stop = lambda: None

    gpiozero = types.ModuleType("gpiozero")

    class RGBLED:
        def __init__(self, *pins, active_high=True):
            self.value = (0, 0, 0)

        def close(self):
            pass

    gpiozero.RGBLED = RGBLED

    monkeypatch.setitem(sys.modules, "vosk", vosk)
    monkeypatch.setitem(sys.modules, "sounddevice", sd)
    monkeypatch.setitem(sys.modules, "gpiozero", gpiozero)
    from software.skull_core import record as record_module

    def fake_record(seconds, device=None):
        calls["record"].append((seconds, device))
        return np.zeros(int(seconds * 48_000), dtype=np.int16)

    monkeypatch.setattr(record_module, "record", fake_record)
    return model, voice, calls


def test_main_without_a_model_says_where_to_get_it(tmp_path, capsys):
    assert vc.main(["--model-dir", str(tmp_path / "nowhere")]) == 1
    assert vc.MODEL_URL in capsys.readouterr().err


def test_main_one_turn_then_quit(fake_pi, monkeypatch):
    model, voice, calls = fake_pi
    lines = iter(["", "quit"])
    monkeypatch.setattr("builtins.input", lambda prompt="": next(lines))
    rc = vc.main(["--model-dir", str(model), "--voice-dir", str(voice),
                  "--mic-device", "1", "--seconds", "0.5", "--tail", "0"])
    assert rc == 0
    assert calls["rate"] == 48_000                       # Vosk downsamples itself; no resampling here
    assert json.loads(calls["grammar"]) == [*STATES, "[unk]"]
    assert calls["record"] == [(0.5, 1)]
    assert len(calls["play"]) == 1
    assert calls["play"][0][1]["device"] == "MAX98357A"
