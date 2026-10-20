"""Tests for software/skull_core/stt.py. No mic, no Moonshine model: a fake stream fires line events."""
import sys
import types
from types import SimpleNamespace

import numpy as np
import pytest

# block_dbfs -> record.py imports sounddevice at the top; the Surface venv doesn't have it.
try:
    import sounddevice  # noqa: F401
except (ImportError, OSError):
    sys.modules["sounddevice"] = types.ModuleType("sounddevice")

from software.skull_core import stt
from software.skull_core.mic_stream import Block, MicStats, SpeechDetector, SpeechEvent

# Small round numbers: 8 blocks a second, every time exact in binary.
RATE = 8000
BLOCK = 1000
BLOCK_S = 0.125
T0 = 100.0
LOUD = np.full(BLOCK, 3277, dtype=np.int16)    # -20.0 dBFS
QUIET = np.zeros(BLOCK, dtype=np.int16)


class FakeClock:
    def __init__(self, t=T0):
        self.t = t

    def __call__(self):
        return self.t


# --- a stand-in for moonshine_voice's Stream ------------------------------------------------
# Listener tells events apart by class name, as with the real LineStarted / LineCompleted.

class LineStarted:
    def __init__(self, line):
        self.line = line


class LineCompleted:
    def __init__(self, line):
        self.line = line


class LineTextChanged:
    def __init__(self, line):
        self.line = line


def line(text, start_time, duration, latency_ms=50):
    return SimpleNamespace(text=text, start_time=start_time, duration=duration,
                           last_transcription_latency_ms=latency_ms)


class FakeStream:
    """script: {number of blocks fed: [events]}, fired from inside add_audio, as Moonshine does.
    on_stop: events fired from inside stop() (the flush of an unfinished line)."""

    def __init__(self, script=None, on_stop=(), error_at=None):
        self.script = script or {}
        self.on_stop = list(on_stop)
        self.error_at = error_at
        self.listeners = []
        self.fed = []          # (dtype, length, rate, min, max) of every add_audio call
        self.calls = []

    def add_listener(self, callback):
        self.listeners.append(callback)

    def start(self):
        self.calls.append("start")

    def add_audio(self, samples, rate):
        self.fed.append((samples.dtype, len(samples), rate, float(samples.min()), float(samples.max())))
        if self.error_at == len(self.fed):
            raise RuntimeError("moonshine failed")
        for event in self.script.get(len(self.fed), []):
            for callback in self.listeners:
                callback(event)

    def stop(self):
        self.calls.append("stop")
        for event in self.on_stop:
            for callback in self.listeners:
                callback(event)

    def close(self):
        self.calls.append("close")


class StreamMaker:
    """Hands out the given streams one per turn, like Transcriber.create_stream."""

    def __init__(self, *streams):
        self.streams = list(streams)
        self.made = []

    def __call__(self):
        stream = self.streams.pop(0) if self.streams else FakeStream()
        self.made.append(stream)
        return stream


# --- a stand-in for MicStream ---------------------------------------------------------------

class FakeMic:
    """read(0) hands out the stale blocks first (what queued before Enter), then None.
    read(timeout) hands out the next live block and moves the clock to its t_end; with no
    blocks left it lets `timeout` pass and returns None."""

    def __init__(self, clock, blocks, stale=(), queue_full_during=0):
        self.clock = clock
        self.blocks = list(blocks)
        self.stale = list(stale)
        self.queue_full = 5             # drops from before the turn: must not be counted in it
        self.queue_full_during = queue_full_during

    def read(self, timeout):
        if timeout == 0:
            return self.stale.pop(0) if self.stale else None
        if not self.blocks:
            self.clock.t += timeout
            return None
        block = self.blocks.pop(0)
        self.clock.t = block.t_end
        if not self.blocks:
            self.queue_full += self.queue_full_during
        return block

    def stats(self):
        return MicStats(queue_full=self.queue_full)


class ScriptedDetector:
    """Stands in for SpeechDetector: events by block number; in_speech between "start" and "end"."""

    def __init__(self, events=None):
        self.events = events or {}
        self.in_speech = False
        self.updates = 0
        self.resets = 0

    def reset(self):
        self.resets += 1
        self.in_speech = False

    def update(self, block):
        self.updates += 1
        event = self.events.get(self.updates)
        if event is not None:
            self.in_speech = event.kind == "start"
        return event


def blocks(count, t0=T0, samples=LOUD):
    return [Block(samples, t0 + (n + 1) * BLOCK_S) for n in range(count)]


def listen(mic_blocks, stream, detector=None, clock=None, **kwargs):
    clock = clock or FakeClock()
    mic = kwargs.pop("mic", None) or FakeMic(clock, mic_blocks)
    listener = stt.Listener(StreamMaker(stream), RATE, clock=clock)
    heard = stt.listen_once(mic, listener, detector or ScriptedDetector(), clock=clock, **kwargs)
    return heard, listener


# --- to_float ------------------------------------------------------------------------------

def test_to_float_gives_float32_in_minus_one_to_one():
    samples = np.array([-32768, -16384, 0, 16384, 32767], dtype=np.int16)
    out = stt.to_float(samples)
    assert out.dtype == np.float32
    assert out[0] == -1.0 and out[2] == 0.0 and out[1] == -0.5 and out[3] == 0.5
    assert out[4] < 1.0 and out[4] == pytest.approx(32767 / 32768)


def test_to_float_leaves_the_block_alone():
    samples = np.array([1000, -1000], dtype=np.int16)
    out = stt.to_float(samples)
    out[0] = 0.0
    assert samples[0] == 1000


# --- usable_text and word errors ------------------------------------------------------------

@pytest.mark.parametrize("text", ["", "   ", ".", "A.", " ?! "])
def test_usable_text_rejects_nothing_said(text):
    assert stt.usable_text(text, 2) is None


def test_usable_text_strips_and_keeps_real_text():
    assert stt.usable_text("  Hi.  ", 2) == "Hi."
    assert stt.usable_text("What is the time?", 2) == "What is the time?"


def test_usable_text_threshold_is_a_parameter():
    assert stt.usable_text("Uh.", 3) is None
    assert stt.usable_text("Uh.", 2) == "Uh."
    assert stt.usable_text("", 0) == ""


def test_words_lowercase_without_punctuation():
    assert stt.words("What's the TEMPERATURE, of your brain?") == ["what's", "the", "temperature", "of", "your", "brain"]


@pytest.mark.parametrize("heard, errors", [
    ("What is the temperature of your brain?", 0),
    ("what is the temperature of your brain", 0),        # case and punctuation don't count
    ("What is the temperature of you brain?", 1),         # one wrong word
    ("What is temperature of your brain?", 1),            # one missing
    ("So what is the temperature of your brain?", 1),     # one extra
    ("", 7),                                              # nothing heard: every word missing
    ("Awaken machines spirit", 7),
])
def test_word_errors(heard, errors):
    assert stt.word_errors("What is the temperature of your brain?", heard) == errors


# --- Listener ------------------------------------------------------------------------------

def test_listener_feeds_float32_at_the_mic_rate():
    stream = FakeStream()
    listener = stt.Listener(StreamMaker(stream), 48000)
    listener.start_turn()
    listener.feed(np.array([-32768, 0, 16384], dtype=np.int16))
    assert stream.fed == [(np.float32, 3, 48000, -1.0, 0.5)]
    assert stream.calls == ["start"]
    assert listener.samples_fed == 3


def test_listener_stores_finished_lines_with_their_arrival_time():
    clock = FakeClock(5.0)
    done = line("Hello there.", 0.5, 1.0)
    stream = FakeStream({1: [LineStarted(done), LineTextChanged(done)], 2: [LineCompleted(done)]})
    listener = stt.Listener(StreamMaker(stream), RATE, clock=clock)
    listener.start_turn()
    listener.feed(LOUD)
    assert listener.speech_started and listener.lines == []
    clock.t = 6.0
    listener.feed(LOUD)
    assert listener.lines == [done] and listener.line_times == [6.0]
    assert listener.speech_end(10.0) == 11.5
    assert listener.text() == "Hello there."


def test_listener_new_stream_per_turn_closes_the_old_one():
    first, second = FakeStream(), FakeStream()
    maker = StreamMaker(first, second)
    listener = stt.Listener(maker, RATE)
    listener.start_turn()
    listener.start_turn()
    assert first.calls == ["start", "close"]
    assert second.calls == ["start"]
    listener.close()
    listener.close()     # twice is fine
    assert second.calls == ["start", "close"]


def test_drain_throws_away_stale_blocks():
    mic = FakeMic(FakeClock(), [], stale=blocks(3))
    assert stt.drain(mic) == 3
    assert stt.drain(mic) == 0


# --- listen_once: the "line" rule (Moonshine's VAD ends the turn) ----------------------------

def test_line_rule_ends_at_the_first_finished_line():
    said = line("What is the temperature of your brain?", 0.25, 0.75, latency_ms=42)
    stream = FakeStream({2: [LineStarted(said)], 10: [LineCompleted(said)]})
    heard, _ = listen(blocks(20), stream, end_rule="line")
    assert heard.text == "What is the temperature of your brain?"
    assert heard.end_reason == "line"
    assert heard.blocks_fed == 10                  # stopped reading at the line
    assert heard.t_first_sample == T0              # first block ends at 100.125 and is 0.125 s long
    assert heard.t_speech_end == pytest.approx(T0 + 1.0)
    assert heard.t_text == pytest.approx(T0 + 10 * BLOCK_S)
    assert heard.stt_s == pytest.approx(0.25)
    assert heard.first_line_text == heard.text and heard.first_line_stt_s == pytest.approx(0.25)
    assert heard.moonshine_latency_ms == 42
    assert heard.lines == 1
    assert stream.calls == ["start", "close"]       # no flush needed


def test_line_rule_sends_only_the_first_sentence_of_two():
    """The trap measured on the Surface (2026-10-20): a 0.555 s pause made Moonshine close a line."""
    one, two = line("Awaken machine spirit.", 0.5, 1.0), line("What is the time?", 2.0, 1.0)
    stream = FakeStream({13: [LineCompleted(one)], 25: [LineCompleted(two)]})
    heard, _ = listen(blocks(30), stream, end_rule="line")
    assert heard.text == "Awaken machine spirit."
    assert heard.end_reason == "line"


# --- listen_once: the "detector" rule (our SpeechDetector ends the turn) -----------------------

def test_detector_rule_keeps_listening_through_a_pause_then_flushes():
    one, two = line("Awaken machine spirit.", 0.5, 1.0), line("What is the time?", 2.0, 1.0)
    stream = FakeStream({13: [LineCompleted(one)]}, on_stop=[LineCompleted(two)])
    detector = ScriptedDetector({4: SpeechEvent("start", T0 + 0.5), 30: SpeechEvent("end", T0 + 3.0)})
    clock = FakeClock()
    heard, _ = listen(blocks(40), stream, detector, clock=clock, end_rule="detector")
    assert heard.end_reason == "detector"
    assert heard.raw_text == heard.text == "Awaken machine spirit. What is the time?"
    assert heard.lines == 2
    assert stream.calls == ["start", "stop", "close"]
    assert heard.blocks_fed == 30
    assert heard.first_line_text == "Awaken machine spirit."
    assert heard.first_line_stt_s == pytest.approx(13 * BLOCK_S - 1.5)
    assert heard.t_text == pytest.approx(T0 + 30 * BLOCK_S)       # after the flush
    assert heard.stt_s == pytest.approx(30 * BLOCK_S - 3.0)
    assert heard.detector_start == T0 + 0.5 and heard.detector_end == T0 + 3.0
    assert heard.detector_end_to_text_s == pytest.approx(30 * BLOCK_S - 3.0)


def test_detector_rule_falls_back_to_the_line_when_our_detector_heard_nothing():
    """A quiet voice under the start threshold: Moonshine hears it, our detector does not."""
    said = line("Hello.", 0.25, 0.5)
    stream = FakeStream({8: [LineCompleted(said)]})
    heard, _ = listen(blocks(20), stream, ScriptedDetector(), end_rule="detector")
    assert heard.end_reason == "line"
    assert heard.text == "Hello."
    assert stream.calls == ["start", "close"]


def test_detector_rule_flush_that_gives_the_only_line():
    said = line("Short one.", 0.25, 0.5)
    stream = FakeStream(on_stop=[LineCompleted(said)])
    detector = ScriptedDetector({3: SpeechEvent("start", T0 + 0.25), 12: SpeechEvent("end", T0 + 0.75)})
    heard, _ = listen(blocks(20), stream, detector, end_rule="detector")
    assert heard.text == "Short one."
    assert heard.first_line_text == "Short one."
    assert heard.first_line_stt_s == heard.stt_s == pytest.approx(12 * BLOCK_S - 0.75)


def test_detector_rule_with_the_real_detector_and_a_short_pause():
    """Real SpeechDetector (hang 0.5 s): a 0.25 s pause between two sentences is one utterance."""
    clock = FakeClock()
    audio = [LOUD] * 8 + [QUIET] * 2 + [LOUD] * 8 + [QUIET] * 8
    mic_blocks = [Block(samples, T0 + (n + 1) * BLOCK_S) for n, samples in enumerate(audio)]
    one, two = line("First.", 0.0, 1.0), line("Second.", 1.25, 1.0)
    stream = FakeStream({10: [LineCompleted(one)]}, on_stop=[LineCompleted(two)])
    detector = SpeechDetector(start_dbfs=-35, end_dbfs=-45, hang_s=0.5, min_speech_s=0.125)
    heard, _ = listen(mic_blocks, stream, detector, clock=clock, end_rule="detector")
    assert heard.end_reason == "detector"
    assert heard.text == "First. Second."
    assert heard.detector_end == pytest.approx(T0 + 18 * BLOCK_S)     # last loud block
    assert heard.blocks_fed == 22                                     # 18 + hang 0.5 s = 4 quiet blocks


# --- listen_once: nobody spoke, never-ending, nothing usable, lost blocks --------------------

def test_no_speech_gives_up_at_the_start_timeout():
    stream = FakeStream()
    heard, _ = listen(blocks(60, samples=QUIET), stream, end_rule="line", start_timeout_s=2.0)
    assert heard.text is None and heard.end_reason == "no_speech"
    assert heard.blocks_fed == 16          # 2.0 s of blocks; the 17th read never happens
    assert heard.stt_s is None and heard.raw_text == ""
    assert stream.calls == ["start", "close"]


def test_speech_that_began_is_not_cut_by_the_start_timeout():
    said = line("Long question.", 1.0, 3.0)
    stream = FakeStream({1: [LineStarted(said)], 40: [LineCompleted(said)]})
    heard, _ = listen(blocks(60), stream, end_rule="line", start_timeout_s=2.0)
    assert heard.end_reason == "line" and heard.text == "Long question."


def test_a_line_that_never_ends_stops_at_max_s_and_is_flushed():
    said = line("Blah blah blah", 0.0, 4.0)
    stream = FakeStream({1: [LineStarted(said)]}, on_stop=[LineCompleted(said)])
    heard, _ = listen(blocks(100), stream, end_rule="line", start_timeout_s=1.0, max_s=4.0)
    assert heard.end_reason == "max_s"
    assert heard.blocks_fed == 32                       # 4.0 s
    assert heard.text == "Blah blah blah"
    assert stream.calls == ["start", "stop", "close"]


def test_nothing_usable_keeps_the_raw_text():
    said = line("Uh.", 0.25, 0.25)
    stream = FakeStream({4: [LineCompleted(said)]})
    heard, _ = listen(blocks(10), stream, end_rule="line", min_letters=3)
    assert heard.text is None and heard.raw_text == "Uh."
    assert heard.end_reason == "line"


def test_mic_that_stops_delivering_ends_the_turn():
    stream = FakeStream({1: [LineStarted(line("", 0.0, 0.0))]})
    heard, _ = listen(blocks(2), stream, end_rule="line", stall_s=1.0)
    assert heard.end_reason == "no_audio" and heard.text is None
    assert stream.calls == ["start", "close"]


def test_stale_blocks_are_drained_and_only_this_turns_drops_counted():
    clock = FakeClock()
    said = line("Hello.", 0.25, 0.5)
    mic = FakeMic(clock, blocks(10), stale=blocks(4, t0=T0 - 1.0), queue_full_during=2)
    stream = FakeStream({8: [LineCompleted(said)]})
    heard, _ = listen(None, stream, clock=clock, mic=mic, end_rule="line")
    assert heard.blocks_drained == 4
    assert heard.t_first_sample == T0           # timing starts at the first live block
    assert heard.queue_full == 0                 # the 5 from before the turn don't count
    mic2 = FakeMic(FakeClock(), blocks(8), queue_full_during=2)
    heard2, _ = listen(None, FakeStream({8: [LineCompleted(said)]}), clock=mic2.clock, mic=mic2, end_rule="line")
    assert heard2.queue_full == 2


def test_a_hole_in_the_blocks_is_counted_and_resets_the_detector():
    clock = FakeClock()
    mic_blocks = blocks(4) + blocks(4, t0=T0 + 1.0)       # 0.5 s missing between them
    said = line("Hi there.", 0.0, 0.5)
    stream = FakeStream({8: [LineCompleted(said)]})
    detector = ScriptedDetector()
    heard, _ = listen(mic_blocks, stream, detector, clock=clock, end_rule="line")
    assert heard.gaps == 1
    assert detector.resets == 2          # once at the start of the turn, once at the hole


def test_stream_is_closed_even_when_moonshine_raises():
    stream = FakeStream(error_at=3)
    with pytest.raises(RuntimeError):
        listen(blocks(10), stream, end_rule="line")
    assert stream.calls == ["start", "close"]


def test_unknown_end_rule_is_refused():
    with pytest.raises(ValueError):
        listen(blocks(1), FakeStream(), end_rule="wake")


def test_importing_stt_needs_no_moonshine():
    assert "moonshine_voice" not in sys.modules


def test_load_stream_maker_without_moonshine_says_what_to_do(monkeypatch):
    monkeypatch.setitem(sys.modules, "moonshine_voice", None)     # makes the import fail
    with pytest.raises(ImportError, match="m3-bench/.venv-stt"):
        stt.load_stream_maker()


# --- interrupt (M3-07: Enter noticed while the wake listener waits) ------------------------------

def test_interrupt_ends_a_silent_wait():
    stream = FakeStream()
    asked = []

    def interrupt():
        asked.append(1)
        return len(asked) >= 4           # a key arrives before the 4th block

    heard, _ = listen(blocks(60, samples=QUIET), stream, end_rule="detector", start_timeout_s=30.0,
                      interrupt=interrupt)
    assert heard.end_reason == "interrupted" and heard.text is None
    assert heard.blocks_fed == 3
    assert stream.calls == ["start", "close"]


def test_interrupt_is_ignored_once_speech_began():
    said = line("Awaken machine spirit.", 0.0, 1.0)
    stream = FakeStream({1: [LineStarted(said)], 5: [LineCompleted(said)]})
    heard, _ = listen(blocks(10), stream, end_rule="line", start_timeout_s=30.0, interrupt=lambda: True)
    assert heard.end_reason == "interrupted"     # checked before block 1: nobody was speaking yet

    stream = FakeStream({1: [LineStarted(said)], 5: [LineCompleted(said)]})
    calls = []

    def after_first_block():
        calls.append(1)
        return len(calls) > 1

    heard, _ = listen(blocks(10), stream, end_rule="line", start_timeout_s=30.0, interrupt=after_first_block)
    assert heard.end_reason == "line" and heard.text == "Awaken machine spirit."
    assert len(calls) == 1                       # never asked again after speech started


def test_no_interrupt_keeps_the_old_behaviour():
    heard, _ = listen(blocks(60, samples=QUIET), FakeStream(), end_rule="line", start_timeout_s=2.0)
    assert heard.end_reason == "no_speech"
