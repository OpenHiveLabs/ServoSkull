"""Tests for software/skull_core/voice_out.py and tts_worker.py. No amp, no torch, no model.

The synth is a fake function or tests/fake_tts_worker.py (the real protocol with a tone engine);
`play` is a fake that records what it was given.
"""
import io
import json
import math
import os
import sys
import threading
import time
from pathlib import Path

import numpy as np
import pytest

from software.skull_core import brain, tts_worker
from software.skull_core import voice_out as vo
from software.skull_core.voice_console import HalfDuplexGuard

FAKE_WORKER = Path(__file__).with_name("fake_tts_worker.py")
VOICE = vo.bank_voice("javert")


# --- split_sentences --------------------------------------------------------------------------

def test_split_waits_for_the_space_after_the_mark():
    assert list(vo.split_sentences(["It is no", "on.", " The sk", "y is grey. Bye"])) == \
        ["It is noon.", "The sky is grey.", "Bye"]


def test_split_pieces_cut_mid_word_and_one_character_at_a_time():
    text = "Hello there. How are you? Fine!"
    assert list(vo.split_sentences(list(text))) == ["Hello there.", "How are you?", "Fine!"]


def test_split_a_decimal_point_is_not_an_end():
    assert list(vo.split_sentences(["The joint is at 3.5 degrees. ", "Done."])) == \
        ["The joint is at 3.5 degrees.", "Done."]


def test_split_ellipsis_and_mixed_marks():
    assert list(vo.split_sentences(["Wait... what?! Really."])) == ["Wait...", "what?!", "Really."]


def test_split_closing_quote_stays_with_its_sentence():
    assert list(vo.split_sentences(['He said "go." Then left.'])) == ['He said "go."', "Then left."]


def test_split_no_end_mark_and_empty_pieces():
    assert list(vo.split_sentences(["", "no end mark here", "", "  "])) == ["no end mark here"]
    assert list(vo.split_sentences([])) == []
    assert list(vo.split_sentences(["", "   ", "\n"])) == []


def test_split_yields_before_the_stream_ends():
    """Sentence one must come out while the pieces are still arriving (that is the whole point)."""
    seen = []

    def pieces():
        yield "First one. "
        seen.append("second piece asked for")
        yield "Second one."

    sentences = vo.split_sentences(pieces())
    assert next(sentences) == "First one."
    assert seen == []


# --- normalise and the bank ------------------------------------------------------------------

def test_normalise_rule():
    assert vo.normalise("I can’t   reach my BRAIN.") == "i can't reach my brain"
    assert vo.normalise("  Ready!  ") == "ready"
    assert vo.normalise("Ready?") == "ready?"            # a question keeps its mark
    assert vo.normalise("Ready") == vo.normalise("ready.") == vo.normalise("READY!")


def test_bank_add_then_lookup_round_trip(tmp_path):
    bank = vo.PhraseBank(tmp_path, VOICE)
    tone = np.full((4800, 1), 0.1, dtype=np.float32)
    path = bank.add("I can't reach my brain.", tone)
    assert bank.lookup("i can’t reach my brain") == path
    again = vo.PhraseBank(tmp_path, VOICE)               # a fresh object reads the index from disk
    assert again.lookup("I CAN'T reach my brain!") == path
    assert vo.load_wav(path).shape == (4800, 1)


def test_bank_miss_missing_file_and_other_voice(tmp_path):
    bank = vo.PhraseBank(tmp_path, VOICE)
    assert bank.lookup("Never stored.") is None
    path = bank.add("Stored.", np.zeros((480, 1), dtype=np.float32))
    assert vo.PhraseBank(tmp_path, vo.bank_voice("marius")).lookup("Stored.") is None
    path.unlink()
    assert bank.lookup("Stored.") is None                # index entry without its file


def test_bank_misses_count_across_objects(tmp_path):
    assert vo.PhraseBank(tmp_path, VOICE).note_miss("Hello there.") == 1
    assert vo.PhraseBank(tmp_path, VOICE).note_miss("hello there") == 2
    vo.PhraseBank(tmp_path, VOICE).note_miss("Other.")
    assert vo.PhraseBank(tmp_path, VOICE).top_misses(5) == [(2, "Hello there."), (1, "Other.")]


def test_bank_corrupt_index_is_a_clear_error(tmp_path):
    (tmp_path / "index.json").write_text("{not json")
    with pytest.raises(ValueError, match="index.json"):
        vo.PhraseBank(tmp_path, VOICE)


def test_bank_lines_file_has_the_brain_lines_and_nothing_invented():
    lines = vo.read_bank_lines(vo.BANK_LINES_FILE)
    assert lines == [brain.FALLBACK_LINE, brain.REFUSAL_LINE]


def test_build_bank_makes_only_missing_lines(tmp_path):
    bank = vo.PhraseBank(tmp_path, VOICE)
    calls = []

    def synth(text):
        calls.append(text)
        return np.full(2400, 0.9, dtype=np.float32), 24_000   # loud on purpose: the cap must apply

    assert vo.build_bank(["One.", "Two."], bank, synth, out=lambda _: None) == (2, 0)
    assert vo.build_bank(["One.", "Two.", "Three."], bank, synth, out=lambda _: None) == (1, 2)
    assert calls == ["One.", "Two.", "Three."]
    stored = vo.load_wav(bank.lookup("Three."))
    assert len(stored) == 4800
    assert vo.peak_dbfs(stored) <= vo.PEAK_CAP_DBFS + 0.01


# --- audio -------------------------------------------------------------------------------------

def test_to_48k_doubles_length_and_keeps_level():
    t = np.arange(24_000) / 24_000
    tone = 0.5 * np.sin(2 * np.pi * 1000 * t)
    out = vo.to_48k(tone, 24_000)
    assert out.shape == (48_000, 1) and out.dtype == np.float32
    middle = out[2000:-2000, 0]
    ideal = 0.5 * np.sin(2 * np.pi * 1000 * np.arange(48_000) / 48_000)[2000:-2000]
    assert np.max(np.abs(middle - ideal)) < 1e-4        # the same 1 kHz tone, now at 48 kHz


def test_to_48k_removes_the_mirror_image():
    t = np.arange(24_000) / 24_000
    out = vo.to_48k(0.5 * np.sin(2 * np.pi * 1000 * t), 24_000)[:, 0]
    spectrum = np.abs(np.fft.rfft(out * np.hanning(len(out))))
    freqs = np.fft.rfftfreq(len(out), 1 / 48_000)
    image = spectrum[np.argmin(np.abs(freqs - 23_000))] / spectrum[np.argmin(np.abs(freqs - 1000))]
    assert 20 * np.log10(image) < -80                     # repeating samples would leave -23.7 dB


def test_to_48k_matches_scipy_resample_poly():
    signal = pytest.importorskip("scipy.signal")
    # Tones below 10 kHz: above our 11 kHz cutoff the two filters differ on purpose.
    rng = np.random.default_rng(1)
    t = np.arange(4800) / 24_000
    x = sum(0.05 * np.sin(2 * np.pi * f * t + rng.uniform(0, 6.28)) for f in rng.uniform(100, 10_000, 12))
    ours = vo.to_48k(x, 24_000)[:, 0]
    theirs = signal.resample_poly(x, 2, 1)
    assert np.sqrt(np.mean((ours[200:-200] - theirs[200:-200]) ** 2)) < 0.01 * np.sqrt(np.mean(theirs ** 2))


def test_to_48k_passes_48k_and_refuses_other_rates():
    x = np.ones(10)
    assert vo.to_48k(x, 48_000).shape == (10, 1)
    with pytest.raises(ValueError, match="22050"):
        vo.to_48k(x, 22_050)
    assert vo.to_48k(np.zeros(0), 24_000).shape == (0, 1)


def test_cap_peak_only_turns_down():
    loud = np.full((10, 1), 1.0, dtype=np.float32)
    assert vo.peak_dbfs(vo.cap_peak(loud)) == pytest.approx(vo.PEAK_CAP_DBFS)
    quiet = np.full((10, 1), 0.1, dtype=np.float32)
    assert vo.cap_peak(quiet) is quiet
    assert vo.peak_dbfs(np.zeros(5)) == -math.inf


# --- Speaker.speak ---------------------------------------------------------------------------

class FakeClock:
    def __init__(self):
        self.t = 100.0
        self.lock = threading.Lock()

    def __call__(self):
        with self.lock:
            return self.t

    def advance(self, dt):
        with self.lock:
            self.t += dt


class Rig:
    """A Speaker with a fake synth, fake player and fake clock; records everything."""

    def __init__(self, tmp_path, fail_on=None, play_s=0.5, bank_lines=(), opener=True):
        self.clock = FakeClock()
        self.guard = HalfDuplexGuard(tail_s=0.3, clock=self.clock)
        self.events = []
        self.synth_calls = []
        self.fail_on = fail_on
        self.play_s = play_s
        self.bank = vo.PhraseBank(tmp_path, VOICE)
        for line in bank_lines:
            self.bank.add(line, np.full((4800, 1), 0.05, dtype=np.float32))
        self.speaker = vo.Speaker(self.synth, self.bank, self.guard, self.play, on_state=self.on_state,
                                  opener=np.full((960, 1), 0.05, dtype=np.float32) if opener else None,
                                  clock=self.clock)

    def synth(self, text):
        self.synth_calls.append(text)
        if self.fail_on and self.fail_on in text:
            raise vo.SynthError("worker said no")
        self.clock.advance(2.0)
        return np.full(2400, 0.1, dtype=np.float32), 24_000

    def play(self, samples):
        self.events.append(("play", len(samples), self.guard.is_speaking))
        self.clock.advance(self.play_s)
        time.sleep(0.02)          # real time too, so the producer thread gets to run

    def on_state(self, name):
        self.events.append(("state", name))


def test_speak_plays_opener_then_sentences_in_order_with_guard_shut(tmp_path):
    rig = Rig(tmp_path)
    log = rig.speaker.speak(["One. Two", ". Three."])
    plays = [e for e in rig.events if e[0] == "play"]
    assert plays[0] == ("play", 960, True)                      # Cogitating first, gate shut
    assert [p[1] for p in plays[1:]] == [4800, 4800, 4800]      # 2400 at 24 kHz -> 4800 at 48 kHz
    assert all(shut for _, _, shut in plays)
    assert [c.text for c in log.clips] == ["(cogitating)", "One.", "Two.", "Three."]
    assert rig.synth_calls == ["One.", "Two.", "Three."]
    assert not rig.guard.is_speaking and not rig.guard.mic_open()   # tail running after the reply
    rig.clock.advance(0.3)
    assert rig.guard.mic_open()


def test_bank_hit_never_calls_the_synth(tmp_path):
    rig = Rig(tmp_path, bank_lines=["I can't reach my brain."])
    log = rig.speaker.speak(["I can’t reach my brain."])
    assert rig.synth_calls == []
    assert [c.source for c in log.clips] == ["opener", "bank"]
    assert log.misses == 0 and rig.bank.top_misses(5) == []


def test_misses_are_counted(tmp_path):
    rig = Rig(tmp_path)
    log = rig.speaker.speak(["Fresh words. Fresh words."])
    assert log.misses == 2 and rig.bank.top_misses(1) == [(2, "Fresh words.")]


def test_states_for_one_sentence(tmp_path):
    rig = Rig(tmp_path, opener=False)
    rig.speaker.speak(["Only one."])
    states = [e[1] for e in rig.events if e[0] == "state"]
    assert states[0] == "speaking" and states.count("speaking") == 1


def test_states_for_three_sentences_with_waits(tmp_path):
    """The brain is slow: each sentence arrives after the previous one finished playing."""
    rig = Rig(tmp_path, opener=False)
    gate = [threading.Event() for _ in range(3)]

    def slow_pieces():
        for i, text in enumerate(["One. ", "Two. ", "Three."]):
            if i:
                assert gate[i - 1].wait(5)
            yield text

    def play(samples):
        rig.events.append(("play", len(samples), rig.guard.is_speaking))
        rig.clock.advance(0.5)
        gate[len([e for e in rig.events if e[0] == "play"]) - 1].set()

    rig.speaker.play = play
    rig.speaker.speak(slow_pieces())
    states = [e[1] for e in rig.events if e[0] == "state"]
    assert states[:5] == ["speaking", "thinking", "speaking", "thinking", "speaking"]


def test_synth_error_on_sentence_two_ends_cleanly(tmp_path):
    rig = Rig(tmp_path, fail_on="Two", bank_lines=["Four."])
    log = rig.speaker.speak(["One. Two. Three. Four."])
    by_text = {c.text: c for c in log.clips}
    assert by_text["One."].source == "pocket" and by_text["One."].t_play_start is not None
    assert by_text["Two."].source == "none" and "synth" in by_text["Two."].error
    assert by_text["Three."].error.startswith("skipped")        # the synth is not tried again this reply
    assert by_text["Four."].source == "bank"                    # bank lines still play
    assert rig.synth_calls == ["One.", "Two."]
    assert not rig.guard.is_speaking


def test_playback_error_is_logged_not_raised(tmp_path):
    rig = Rig(tmp_path, opener=False)

    def broken(samples):
        raise OSError("amp gone")

    rig.speaker.play = broken
    log = rig.speaker.speak(["One."])
    assert log.clips[0].error == "playback: amp gone"
    assert not rig.guard.is_speaking


def test_an_error_from_the_pieces_is_raised_after_the_guard_opens(tmp_path):
    rig = Rig(tmp_path)

    def broken_pieces():
        yield "One. "
        raise RuntimeError("brain blew up")

    with pytest.raises(RuntimeError, match="brain blew up"):
        rig.speaker.speak(broken_pieces())
    assert not rig.guard.is_speaking


def test_ctrl_c_while_playing_stops_the_producer_and_aborts_the_synth(tmp_path):
    rig = Rig(tmp_path, opener=False)
    aborted = []
    rig.synth = None

    class Synth:
        def __call__(self, text):
            return np.zeros(240, dtype=np.float32), 24_000

        def abort(self):
            aborted.append(True)

    rig.speaker.synth = Synth()

    def interrupted(samples):
        raise KeyboardInterrupt

    rig.speaker.play = interrupted
    with pytest.raises(KeyboardInterrupt):
        rig.speaker.speak(["One. Two."])
    assert aborted == [True] and not rig.guard.is_speaking


def test_speech_log_times_and_gaps(tmp_path):
    rig = Rig(tmp_path, play_s=1.0)
    start = rig.clock()
    log = rig.speaker.speak(["One. Two."])
    record = log.as_record(start)
    assert record["opener_start_s"] == 0.0
    assert record["first_audio_s"] is not None and record["first_audio_s"] >= 1.0   # after the 1 s opener
    assert len(record["gaps_s"]) == 1 and record["gaps_s"][0] >= 0
    assert record["sentences"] == 2 and record["bank_hits"] == 0 and record["misses"] == 2
    # A constant 0.1 (-20 dBFS) starts with a step; the low-pass filter rings over it (about -18.9 dBFS).
    assert -20.0 < record["clips"][1]["peak_dbfs"] < vo.PEAK_CAP_DBFS
    json.dumps(record)


def test_speak_without_synth_or_bank_skips_and_still_ends():
    guard = HalfDuplexGuard(tail_s=0.0)
    played = []
    log = vo.Speaker(None, None, guard, played.append).speak(["Hello."])
    assert played == [] and log.clips[0].error.startswith("skipped")


# --- the worker protocol ------------------------------------------------------------------------

class FakeEngine:
    rate = 24_000
    name = "fake"

    def generate(self, text):
        if text == "boom":
            raise RuntimeError("engine failed")
        return np.array([0.5, -0.5, 2.0], dtype=np.float32)


def test_serve_answers_audio_and_errors():
    requests = io.StringIO('{"text": "hi"}\n\nnot json\n{"text": ""}\n{"text": "boom"}\n')
    out = io.BytesIO()
    tts_worker.serve(FakeEngine(), requests, out, clock=iter([0.0, 0.25, 1.0]).__next__)
    data = out.getvalue()
    header, rest = data.split(b"\n", 1)
    assert json.loads(header) == {"ok": True, "frames": 3, "rate": 24000, "gen_s": 0.25}
    pcm = np.frombuffer(rest[:6], dtype="<i2")
    assert pcm.tolist() == [16383, -16383, 32767]           # 2.0 was clipped, not wrapped
    answers = [json.loads(line) for line in rest[6:].splitlines()]
    assert [a["ok"] for a in answers] == [False, False, False]
    assert "bad request" in answers[0]["error"] and "engine failed" in answers[2]["error"]


def fake_synth(tmp_path=None, **env):
    full_env = dict(os.environ, **env)
    return vo.WorkerSynth([sys.executable, str(FAKE_WORKER)], env=full_env,
                          ready_timeout_s=20.0, sentence_timeout_s=3.0)


def test_worker_round_trip_through_the_pipe():
    synth = fake_synth(FAKE_TTS_NOISE="library chatter on stdout")
    try:
        info = synth.wait_ready()
        assert info["ready"] and info["rate"] == 24000 and info["engine"] == "fake-tts 0"
        for text in ("Hello there.", "A second, longer sentence."):
            samples, rate = synth(text)
            assert rate == 24000 and len(samples) == int(0.01 * 24000 * len(text))
            assert 0.24 < np.max(np.abs(samples)) < 0.26
    finally:
        synth.close()
    assert synth.proc is None


def test_worker_error_answer_keeps_it_running():
    synth = fake_synth(FAKE_TTS_FAIL_WORD="nope")
    try:
        with pytest.raises(vo.SynthError, match="could not say it"):
            synth("nope nope.")
        samples, _ = synth("Fine.")
        assert len(samples) > 0
    finally:
        synth.close()


def test_worker_that_dies_is_reported_then_restarted():
    synth = fake_synth(FAKE_TTS_DIE_WORD="die")
    try:
        with pytest.raises(vo.SynthError, match="exited"):
            synth("Now die.")
        assert synth.proc is None
        samples, _ = synth("Back again.")                  # started again on the next call
        assert len(samples) > 0
    finally:
        synth.close()


def test_worker_that_hangs_is_killed(tmp_path):
    pidfile = tmp_path / "pid"
    synth = fake_synth(FAKE_TTS_HANG_WORD="hang", FAKE_TTS_PIDFILE=str(pidfile))
    synth.sentence_timeout_s = 0.5
    try:
        synth.wait_ready()
        pid = int(pidfile.read_text())
        t0 = time.monotonic()
        with pytest.raises(vo.SynthError, match="in time"):
            synth("hang here")
        assert time.monotonic() - t0 < 3.0
        with pytest.raises(ProcessLookupError):
            for _ in range(50):          # reaped by abort(); give the kernel a moment
                os.kill(pid, 0)
                time.sleep(0.05)
    finally:
        synth.close()


def test_worker_load_failure_and_missing_python():
    synth = fake_synth(FAKE_TTS_LOAD_FAIL="1")
    with pytest.raises(vo.SynthError, match="failed to load"):
        synth.wait_ready()
    missing = vo.WorkerSynth(["/nonexistent/python", "-m", "x"])
    with pytest.raises(vo.SynthError, match="cannot start"):
        missing.wait_ready()


def test_speaker_with_the_fake_worker_end_to_end(tmp_path):
    synth = fake_synth()
    guard = HalfDuplexGuard(tail_s=0.0)
    played = []
    try:
        speaker = vo.Speaker(synth, vo.PhraseBank(tmp_path, VOICE), guard, played.append)
        log = speaker.speak(["Hello there. ", "General."])
    finally:
        synth.close()
    assert [len(p) for p in played] == [2 * int(0.01 * 24000 * len(t)) for t in ("Hello there.", "General.")]
    assert [c.source for c in log.clips] == ["pocket", "pocket"]


# --- CLI and imports ---------------------------------------------------------------------------

def test_cli_needs_one_action_and_lists_misses(tmp_path, capsys):
    with pytest.raises(SystemExit):
        vo.parse_args([])
    with pytest.raises(SystemExit):
        vo.parse_args(["--say", "x", "--misses", "3"])
    vo.PhraseBank(tmp_path, VOICE).note_miss("Again.")
    assert vo.main(["--misses", "5", "--bank", str(tmp_path)]) == 0
    assert "1  Again." in capsys.readouterr().out


def test_cli_build_bank_with_a_missing_worker_python_fails_cleanly(tmp_path, capsys):
    lines = tmp_path / "lines.txt"
    lines.write_text("One.\n")
    assert vo.main(["--build-bank", str(lines), "--bank", str(tmp_path / "bank"),
                    "--tts-python", str(tmp_path / "no-python")]) == 1
    assert "cannot start" in capsys.readouterr().err


def test_importing_needs_no_audio_library_or_torch():
    """In a fresh Python: other test files put a stand-in sounddevice into sys.modules."""
    import subprocess

    code = ("import sys\n"
            "import software.skull_core.voice_out, software.skull_core.tts_worker, software.skull_core.voice_loop\n"
            "bad = [m for m in ('sounddevice', 'torch', 'pocket_tts', 'moonshine_voice') if m in sys.modules]\n"
            "print(bad)\n")
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True,
                            cwd=Path(__file__).resolve().parents[1], timeout=60)
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "[]"


# --- M3-06 decision 1 (literal) and M3-07: states per clip, the fixed line ------------------------

def test_cogitating_is_speaking_then_thinking_while_the_reply_is_not_ready(tmp_path):
    rig = Rig(tmp_path)
    opener_done = threading.Event()

    def slow_pieces():
        assert opener_done.wait(5)        # the brain answers only after Cogitating has played
        yield "Late reply."

    def play(samples):
        rig.events.append(("play", len(samples), rig.guard.is_speaking))
        opener_done.set()

    rig.speaker.play = play
    rig.speaker.speak(slow_pieces())
    states = [e[1] for e in rig.events if e[0] == "state"]
    assert states == ["speaking", "thinking", "speaking"]


def test_clip_state_decides_what_each_sentence_shows(tmp_path):
    rig = Rig(tmp_path, opener=False, bank_lines=["I can't reach my brain."])
    rig.speaker.clip_state = lambda sentence: "offline" if "reach my brain" in sentence else "speaking"
    gate = threading.Event()

    def pieces():
        yield "Partly. "
        assert gate.wait(5)
        yield "I can't reach my brain."

    def play(samples):
        rig.events.append(("play", len(samples), rig.guard.is_speaking))
        gate.set()

    rig.speaker.play = play
    rig.speaker.speak(pieces())
    states = [e[1] for e in rig.events if e[0] == "state"]
    assert states == ["speaking", "thinking", "offline"]


def test_say_fixed_plays_one_bank_line_under_its_state_with_the_guard_shut(tmp_path):
    rig = Rig(tmp_path, bank_lines=["I can't reach my brain."])
    log = rig.speaker.say_fixed("I can't reach my brain.", state="offline")
    assert rig.events == [("state", "offline"), ("play", 4800, True)]     # no Cogitating
    assert rig.synth_calls == [] and [c.source for c in log.clips] == ["bank"]
    assert not rig.guard.is_speaking


def test_say_fixed_publishes_its_state_again_on_a_second_call(tmp_path):
    rig = Rig(tmp_path, bank_lines=["I can't reach my brain."])
    rig.speaker.say_fixed("I can't reach my brain.", state="offline")
    rig.speaker.say_fixed("I can't reach my brain.", state="offline")
    assert [e[1] for e in rig.events if e[0] == "state"] == ["offline", "offline"]


def test_say_fixed_falls_back_to_the_synth(tmp_path):
    rig = Rig(tmp_path)
    log = rig.speaker.say_fixed("Not in the bank.")
    assert rig.synth_calls == ["Not in the bank."] and log.clips[0].source == "pocket"
    assert ("state", "speaking") in rig.events
