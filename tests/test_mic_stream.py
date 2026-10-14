import signal
import subprocess
import sys
import types
import wave
from pathlib import Path

import numpy as np
import pytest

# record.py imports sounddevice at the top. The Surface venv doesn't have it,
# and these tests never touch audio hardware, so a stand-in module is enough.
try:
    import sounddevice  # noqa: F401
except (ImportError, OSError):
    sys.modules["sounddevice"] = types.ModuleType("sounddevice")

from software.skull_core import mic_stream as ms
from software.skull_core.record import SAMPLE_RATE
from software.skull_core.voice_console import HalfDuplexGuard

REPO_ROOT = Path(__file__).resolve().parents[1]

# Small round numbers for the stream tests: 8 blocks a second, every time exact in binary.
RATE = 8000
BLOCK = 1000
BLOCK_S = 0.125

DEVICES = [
    {"name": "MAX98357A: HiFi (hw:3,0)", "index": 0, "max_input_channels": 0, "max_output_channels": 2},
    {"name": "USB Camera: Audio (hw:1,0)", "index": 1, "max_input_channels": 2, "max_output_channels": 0},
    {"name": "UACDemoV1.0: USB Audio (hw:2,0)", "index": 2, "max_input_channels": 1, "max_output_channels": 0},
    {"name": "sysdefault", "index": 3, "max_input_channels": 0, "max_output_channels": 32},
]


class FakeClock:
    def __init__(self, t=100.0):
        self.t = t

    def __call__(self):
        return self.t


class Status:
    """Stands in for sounddevice.CallbackFlags."""

    def __init__(self, input_overflow=False):
        self.input_overflow = input_overflow


@pytest.fixture
def fake_sd(monkeypatch):
    """A stand-in sounddevice module. Its InputStream never runs: the test calls the callback by hand."""
    sd = types.ModuleType("sounddevice")
    sd.PortAudioError = type("PortAudioError", (Exception,), {})
    sd.streams = []
    sd.on_start = None        # a test may set this to feed blocks, as PortAudio would after start()
    sd.open_error = None      # a test may set this to make InputStream() fail
    sd.query_devices = lambda: list(DEVICES)

    class FakeInputStream:
        def __init__(self, **kwargs):
            if sd.open_error is not None:
                raise sd.open_error
            self.kwargs = kwargs
            self.callback = kwargs["callback"]
            self.started = False
            self.stops = 0
            self.closes = 0
            sd.streams.append(self)

        def start(self):
            self.started = True
            if sd.on_start is not None:
                sd.on_start(self)

        def stop(self):
            self.stops += 1

        def close(self):
            self.closes += 1

    sd.InputStream = FakeInputStream
    monkeypatch.setitem(sys.modules, "sounddevice", sd)
    return sd


def feed(stream, samples, overflow=False):
    """Call the stream's callback the way PortAudio does: int16, shape (frames, 1). Returns that array."""
    indata = np.asarray(samples, dtype=np.int16).reshape(-1, 1)
    stream.callback(indata, len(indata), None, Status(overflow))
    return indata


def ramp(first, frames=BLOCK):
    return np.arange(first, first + frames, dtype=np.int16)


def open_mic(fake_sd, clock, settle_s=0.0, tail_s=0.25, queue_s=1.0, ring_s=1.0, device=2):
    guard = HalfDuplexGuard(tail_s, clock=clock)
    mic = ms.MicStream(device, guard, RATE, block=BLOCK, settle_s=settle_s, ring_s=ring_s,
                       queue_s=queue_s, clock=clock)
    mic.open()
    return mic, guard, fake_sd.streams[-1]


# --- importing needs no hardware ------------------------------------------------------------

def test_import_and_help_need_no_audio_hardware():
    code = (
        "import sys\n"
        "from software.skull_core import mic_stream\n"
        "bad = [m for m in ('sounddevice', 'vosk', 'gpiozero') if m in sys.modules]\n"
        "assert not bad, bad\n"
    )
    subprocess.run([sys.executable, "-c", code], cwd=REPO_ROOT, check=True)
    out = subprocess.run([sys.executable, "-m", "software.skull_core.mic_stream", "--help"],
                         cwd=REPO_ROOT, capture_output=True, text=True)
    assert out.returncode == 0
    for flag in ("--monitor", "--dump-last", "--mic-device", "--detector"):
        assert flag in out.stdout


def test_block_size_is_the_approved_one():
    assert ms.BLOCK_FRAMES == 1024


# --- TODO 1, 2: the ring buffer (8 samples long: rate 8, one second) -------------------------

def ring8():
    return ms.RingBuffer(1.0, rate=8)


def int16(*values):
    return np.array(values, dtype=np.int16)


def test_ring_rejects_a_buffer_with_no_samples():
    for seconds in (0.0, -1.0, 0.01):
        with pytest.raises(ValueError):
            ms.RingBuffer(seconds, rate=8)


def test_ring_returns_what_was_written():
    ring = ring8()
    ring.write(int16(1, 2, 3))
    assert ring.last(1.0).tolist() == [1, 2, 3]
    assert ring.last(0.25).tolist() == [2, 3]            # the newest two, oldest first
    assert ring.last(1.0).dtype == np.int16


def test_ring_wraps_and_keeps_the_newest():
    ring = ring8()
    ring.write(int16(1, 2, 3, 4, 5, 6))
    ring.write(int16(7, 8, 9, 10, 11))                   # runs past the end of the storage
    assert ring.last(1.0).tolist() == [4, 5, 6, 7, 8, 9, 10, 11]
    assert ring.last(0.5).tolist() == [8, 9, 10, 11]
    ring.write(int16(12))
    assert ring.last(0.375).tolist() == [10, 11, 12]


def test_ring_filled_exactly_to_the_end_starts_again_at_the_front():
    ring = ring8()
    ring.write(int16(1, 2, 3, 4, 5, 6, 7, 8))
    assert ring.last(1.0).tolist() == [1, 2, 3, 4, 5, 6, 7, 8]
    ring.write(int16(9, 10))
    assert ring.last(1.0).tolist() == [3, 4, 5, 6, 7, 8, 9, 10]


def test_ring_takes_a_block_longer_than_itself():
    ring = ring8()
    ring.write(int16(-1, -2, -3))
    ring.write(np.arange(20, dtype=np.int16))
    assert ring.last(1.0).tolist() == list(range(12, 20))
    ring.write(int16(100, 101))                          # and it still knows where it is
    assert ring.last(1.0).tolist() == [14, 15, 16, 17, 18, 19, 100, 101]


def test_ring_last_before_it_is_full_is_shorter():
    ring = ring8()
    assert len(ring.last(1.0)) == 0 and ring.last(1.0).dtype == np.int16
    ring.write(int16(5, 6, 7))
    assert ring.last(1.0).tolist() == [5, 6, 7]          # no zeros in front


def test_ring_last_never_gives_more_than_it_holds():
    ring = ring8()
    ring.write(np.arange(1, 13, dtype=np.int16))
    assert ring.last(60.0).tolist() == list(range(5, 13))


def test_ring_last_of_zero_seconds_is_empty():
    ring = ring8()
    ring.write(int16(1, 2, 3))
    assert len(ring.last(0.0)) == 0


def test_ring_last_returns_a_copy():
    ring = ring8()
    ring.write(int16(1, 2, 3))
    got = ring.last(1.0)
    got[:] = 99
    assert ring.last(1.0).tolist() == [1, 2, 3]


def test_ring_keeps_its_own_copy_of_what_was_written():
    ring = ring8()
    block = int16(1, 2, 3)
    ring.write(block)
    block[:] = 99                                        # PortAudio reuses its buffer
    assert ring.last(1.0).tolist() == [1, 2, 3]


def test_ring_rejects_negative_seconds():
    with pytest.raises(ValueError):
        ring8().last(-0.1)


# --- TODO 3: find the mic by name -----------------------------------------------------------

def test_device_is_found_by_part_of_its_name():
    assert ms.find_input_device("UACDemoV1.0", DEVICES) == 2
    assert ms.find_input_device("Camera", DEVICES) == 1


def test_device_without_inputs_is_never_picked():
    with pytest.raises(ValueError):
        ms.find_input_device("MAX98357A", DEVICES)        # it exists, but it is an output
    devices = [
        {"name": "USB Audio out", "index": 0, "max_input_channels": 0, "max_output_channels": 2},
        {"name": "USB Audio in", "index": 1, "max_input_channels": 1, "max_output_channels": 0},
    ]
    assert ms.find_input_device("USB Audio", devices) == 1


def test_device_missing_lists_the_inputs():
    with pytest.raises(ValueError) as error:
        ms.find_input_device("NoSuchMic", DEVICES)
    text = str(error.value)
    assert "NoSuchMic" in text
    assert "UACDemoV1.0: USB Audio (hw:2,0)" in text and "USB Camera: Audio (hw:1,0)" in text
    assert "MAX98357A" not in text and "sysdefault" not in text    # outputs are not listed


def test_device_ambiguous_is_an_error():
    with pytest.raises(ValueError) as error:
        ms.find_input_device("USB", DEVICES)              # the camera's mic and the USB mic
    assert "USB" in str(error.value)


def test_device_name_is_resolved_when_the_stream_opens(fake_sd):
    mic, guard, stream = open_mic(fake_sd, FakeClock(), device="UACDemo")
    assert stream.kwargs["device"] == 2


def test_device_missing_becomes_a_mic_error_on_open(fake_sd):
    mic = ms.MicStream("NoSuchMic", HalfDuplexGuard(0.0), RATE)
    with pytest.raises(ms.MicError) as error:
        mic.open()
    assert "NoSuchMic" in str(error.value) and "UACDemoV1.0" in str(error.value)
    assert fake_sd.streams == []


def test_device_missing_ends_main_with_a_message_and_exit_code_1(fake_sd, capsys):
    assert ms.main(["--monitor", "--mic-device", "NoSuchMic"]) == 1
    err = capsys.readouterr().err
    assert "NoSuchMic" in err and "UACDemoV1.0" in err
    assert fake_sd.streams == []


# --- open() and close(): written already ----------------------------------------------------

def test_stream_opens_with_the_approved_format(fake_sd):
    mic = ms.MicStream(2, HalfDuplexGuard(0.0), SAMPLE_RATE)
    mic.open()
    (stream,) = fake_sd.streams
    assert stream.kwargs["samplerate"] == 48_000
    assert stream.kwargs["channels"] == 1
    assert stream.kwargs["dtype"] == "int16"
    assert stream.kwargs["blocksize"] == 1024
    assert stream.kwargs["device"] == 2
    assert stream.callback == mic._on_audio
    assert stream.started


def test_open_twice_keeps_the_one_stream(fake_sd):
    mic, guard, stream = open_mic(fake_sd, FakeClock())
    mic.open()
    assert fake_sd.streams == [stream]


def test_close_stops_and_closes_and_twice_is_safe(fake_sd):
    mic, guard, stream = open_mic(fake_sd, FakeClock())
    mic.close()
    mic.close()
    assert (stream.stops, stream.closes) == (1, 1)


def test_close_before_open_is_safe():
    ms.MicStream(2, HalfDuplexGuard(0.0), RATE).close()


def test_port_audio_error_becomes_a_mic_error(fake_sd):
    fake_sd.open_error = fake_sd.PortAudioError("Invalid sample rate")
    mic = ms.MicStream(2, HalfDuplexGuard(0.0), RATE)
    with pytest.raises(ms.MicError) as error:
        mic.open()
    assert "Invalid sample rate" in str(error.value)


def test_stats_start_at_zero(fake_sd):
    mic, guard, stream = open_mic(fake_sd, FakeClock())
    assert mic.stats() == ms.MicStats(0, 0, 0, 0, 0)


def test_queue_length_comes_from_seconds():
    mic = ms.MicStream(2, HalfDuplexGuard(0.0), RATE, block=BLOCK, queue_s=0.25)
    assert mic.queue_blocks == 2


# --- TODO 4: ready() ------------------------------------------------------------------------

def test_not_ready_before_open_or_after_close(fake_sd):
    clock = FakeClock(100.0)
    mic = ms.MicStream(2, HalfDuplexGuard(0.0, clock=clock), RATE, settle_s=0.0, clock=clock)
    assert not mic.ready()
    mic.open()
    assert mic.ready()
    mic.close()
    assert not mic.ready()


def test_not_ready_until_settle_time_has_passed(fake_sd):
    clock = FakeClock(100.0)
    mic, guard, stream = open_mic(fake_sd, clock, settle_s=3.0)
    assert not mic.ready()
    clock.t = 102.5
    assert not mic.ready()
    clock.t = 103.0
    assert mic.ready()                                   # at exactly settle_s it is ready
    clock.t = 500.0
    assert mic.ready()


# --- TODO 5: the callback (checked through the counters and the ring buffer) ----------------

def test_callback_counts_each_block_and_keeps_it_for_looking_back(fake_sd):
    clock = FakeClock(100.0)
    mic, guard, stream = open_mic(fake_sd, clock)
    for n in range(3):
        clock.t += BLOCK_S
        feed(stream, ramp(n * BLOCK))
    assert mic.stats() == ms.MicStats(blocks_seen=3, blocks_queued=3)
    assert mic.ring.last(3 * BLOCK_S).tolist() == list(range(3 * BLOCK))


def test_callback_queues_nothing_while_settling(fake_sd):
    clock = FakeClock(100.0)
    mic, guard, stream = open_mic(fake_sd, clock, settle_s=3.0)
    clock.t = 101.0
    feed(stream, ramp(0))
    assert mic.stats() == ms.MicStats(blocks_seen=1, blocks_gated=1)
    assert mic.ring.last(BLOCK_S).tolist() == list(range(BLOCK))      # the audio is kept all the same
    clock.t = 103.0
    feed(stream, ramp(BLOCK))
    assert mic.stats() == ms.MicStats(blocks_seen=2, blocks_queued=1, blocks_gated=1)


def test_callback_queues_nothing_while_the_skull_speaks_or_in_the_tail(fake_sd):
    clock = FakeClock(100.0)
    mic, guard, stream = open_mic(fake_sd, clock, tail_s=0.25)
    guard.start_speaking()
    clock.t = 100.125
    feed(stream, ramp(0))                                # the skull is speaking
    guard.stop_speaking()                                # at 100.125: the mic reopens at 100.375
    clock.t = 100.25
    feed(stream, ramp(BLOCK))                            # inside the tail
    assert mic.stats() == ms.MicStats(blocks_seen=2, blocks_gated=2)
    assert mic.ring.last(2 * BLOCK_S).tolist() == list(range(2 * BLOCK))


def test_callback_queues_again_after_the_tail(fake_sd):
    clock = FakeClock(100.0)
    mic, guard, stream = open_mic(fake_sd, clock, tail_s=0.25)
    with guard.speaking():
        feed(stream, ramp(0))
    clock.t = 100.25                                     # exactly the end of the tail
    feed(stream, ramp(BLOCK))
    assert mic.stats() == ms.MicStats(blocks_seen=2, blocks_queued=1, blocks_gated=1)


def test_callback_full_queue_drops_and_counts_and_never_raises(fake_sd):
    clock = FakeClock(100.0)
    mic, guard, stream = open_mic(fake_sd, clock, queue_s=0.25)      # room for 2 blocks
    for n in range(5):
        clock.t += BLOCK_S
        feed(stream, ramp(n * BLOCK))                    # nobody reads
    assert mic.stats() == ms.MicStats(blocks_seen=5, blocks_queued=2, queue_full=3)
    assert mic.ring.last(BLOCK_S).tolist() == list(range(4 * BLOCK, 5 * BLOCK))   # the newest block is kept


def test_callback_counts_the_overflow_status(fake_sd):
    clock = FakeClock(100.0)
    mic, guard, stream = open_mic(fake_sd, clock, settle_s=3.0)
    feed(stream, ramp(0), overflow=True)                 # counted even while the gate is shut
    feed(stream, ramp(0))
    clock.t = 110.0
    feed(stream, ramp(0), overflow=True)
    stats = mic.stats()
    assert stats.input_overflows == 2 and stats.blocks_seen == 3


# --- TODO 6: read() -------------------------------------------------------------------------

def test_read_times_out_with_none():
    mic = ms.MicStream(2, HalfDuplexGuard(0.0), RATE)
    assert mic.read(0.01) is None


def test_read_returns_the_blocks_in_order(fake_sd):
    clock = FakeClock(100.0)
    mic, guard, stream = open_mic(fake_sd, clock)
    for n in range(3):
        clock.t += BLOCK_S
        feed(stream, ramp(n * BLOCK))
    firsts = [int(mic.read(0.01).samples[0]) for _ in range(3)]
    assert firsts == [0, BLOCK, 2 * BLOCK]
    assert mic.read(0.01) is None


def test_read_gives_one_dimensional_int16(fake_sd):
    mic, guard, stream = open_mic(fake_sd, FakeClock())
    feed(stream, ramp(0))                                # arrives with shape (frames, 1)
    block = mic.read(0.01)
    assert isinstance(block, ms.Block)
    assert block.samples.shape == (BLOCK,) and block.samples.dtype == np.int16


def test_read_block_does_not_change_when_the_input_array_is_reused(fake_sd):
    mic, guard, stream = open_mic(fake_sd, FakeClock())
    indata = feed(stream, ramp(0))
    indata[:] = -1                                       # PortAudio writes the next block into the same memory
    assert mic.read(0.01).samples.tolist() == list(range(BLOCK))
    assert mic.ring.last(BLOCK_S).tolist() == list(range(BLOCK))


def test_read_block_time_comes_from_the_injected_clock(fake_sd):
    clock = FakeClock(100.0)
    mic, guard, stream = open_mic(fake_sd, clock)
    clock.t = 107.5
    feed(stream, ramp(0))
    clock.t = 200.0                                      # reading later does not change the stamp
    assert mic.read(0.01).t_end == 107.5


def test_read_gives_nothing_that_was_gated(fake_sd):
    clock = FakeClock(100.0)
    mic, guard, stream = open_mic(fake_sd, clock, tail_s=0.25)
    with guard.speaking():
        feed(stream, ramp(0))
    clock.t = 100.125
    feed(stream, ramp(BLOCK))                            # still in the tail
    clock.t = 110.0                                      # the gate is open now, but those blocks were captured shut
    assert mic.read(0.01) is None


# --- TODO 7: the end-of-speech detector -----------------------------------------------------
# Thresholds: start -30 dBFS, end -50 dBFS. Blocks come every 0.125 s; the detector only sees t_end.

LOUD = np.full(256, 3277, dtype=np.int16)     # about -20 dBFS
BETWEEN = np.full(256, 328, dtype=np.int16)   # about -40 dBFS
QUIET = np.full(256, 33, dtype=np.int16)      # about -60 dBFS


def make_detector(hang_s=0.25, min_speech_s=0.0):
    return ms.SpeechDetector(start_dbfs=-30.0, end_dbfs=-50.0, hang_s=hang_s, min_speech_s=min_speech_s)


def run(detector, *steps):
    """steps: (samples, t_end) pairs. Returns what update() gave for each, as (kind, t) or None."""
    results = []
    for samples, t_end in steps:
        event = detector.update(ms.Block(samples, t_end))
        results.append(None if event is None else (event.kind, event.t))
    return results


def test_detector_rejects_bad_settings():
    with pytest.raises(ValueError):
        ms.SpeechDetector(start_dbfs=-50.0, end_dbfs=-30.0, hang_s=0.25, min_speech_s=0.0)
    with pytest.raises(ValueError):
        ms.SpeechDetector(start_dbfs=-30.0, end_dbfs=-50.0, hang_s=-0.1, min_speech_s=0.0)
    with pytest.raises(ValueError):
        ms.SpeechDetector(start_dbfs=-30.0, end_dbfs=-50.0, hang_s=0.25, min_speech_s=-0.1)


def test_detector_test_levels_are_what_the_comments_say():
    assert ms.block_dbfs(LOUD) == pytest.approx(-20.0, abs=0.1)
    assert ms.block_dbfs(BETWEEN) == pytest.approx(-40.0, abs=0.1)
    assert ms.block_dbfs(QUIET) == pytest.approx(-60.0, abs=0.3)


def test_detector_silence_gives_no_events():
    detector = make_detector()
    assert run(detector, (QUIET, 1.0), (QUIET, 1.125), (BETWEEN, 1.25), (BETWEEN, 1.375), (QUIET, 1.5)) == [None] * 5
    assert not detector.in_speech


def test_detector_start_then_end_after_the_hang_time():
    detector = make_detector(hang_s=0.25)
    assert run(detector, (QUIET, 1.0), (LOUD, 1.125)) == [None, ("start", 1.125)]
    assert detector.in_speech
    assert run(detector, (LOUD, 1.25), (LOUD, 1.375)) == [None, None]      # one start per utterance
    assert run(detector, (QUIET, 1.5)) == [None]                           # quiet for 0.125 s: too early
    assert detector.in_speech
    assert run(detector, (QUIET, 1.625)) == [("end", 1.375)]               # quiet for exactly hang_s
    assert not detector.in_speech
    assert run(detector, (QUIET, 1.75), (QUIET, 1.875)) == [None, None]    # one end per utterance


def test_detector_end_time_is_when_speech_stopped():
    detector = make_detector(hang_s=0.5)
    results = run(detector, (LOUD, 1.0), (LOUD, 1.125), (QUIET, 1.25), (QUIET, 1.375), (QUIET, 1.5), (QUIET, 1.625))
    assert results == [("start", 1.0), None, None, None, None, ("end", 1.125)]   # not 1.625


def test_detector_short_pause_does_not_end_the_utterance():
    detector = make_detector(hang_s=0.25)
    results = run(detector, (LOUD, 1.0), (QUIET, 1.125), (LOUD, 1.25), (QUIET, 1.375), (LOUD, 1.5),
                  (QUIET, 1.625), (QUIET, 1.75))
    assert results == [("start", 1.0), None, None, None, None, None, ("end", 1.5)]


def test_detector_between_the_two_thresholds_keeps_its_state():
    detector = make_detector(hang_s=0.25)
    assert run(detector, (BETWEEN, 1.0), (BETWEEN, 1.125)) == [None, None]         # not loud enough to begin
    assert run(detector, (LOUD, 1.25)) == [("start", 1.25)]
    hovering = [(BETWEEN, 1.375 + n * 0.125) for n in range(8)]                    # up to 2.25
    assert run(detector, *hovering) == [None] * 8                                  # not quiet enough to end
    assert detector.in_speech
    assert run(detector, (QUIET, 2.375), (QUIET, 2.5)) == [None, ("end", 2.25)]    # the last block that was not quiet


def test_detector_ignores_a_burst_shorter_than_the_minimum():
    detector = make_detector(hang_s=0.25, min_speech_s=0.25)
    results = run(detector, (LOUD, 1.0), (QUIET, 1.125), (QUIET, 1.25), (QUIET, 1.375), (QUIET, 1.5))
    assert results == [None] * 5                                                   # a click: no start, no end
    assert not detector.in_speech


def test_detector_start_waits_for_the_minimum_and_names_when_speech_began():
    detector = make_detector(hang_s=0.25, min_speech_s=0.25)
    assert run(detector, (LOUD, 1.0), (LOUD, 1.125)) == [None, None]
    assert not detector.in_speech
    assert run(detector, (LOUD, 1.25)) == [("start", 1.0)]                         # lasted 0.25 s; it began at 1.0
    assert detector.in_speech
    assert run(detector, (QUIET, 1.375), (QUIET, 1.5)) == [None, ("end", 1.25)]


def test_detector_forgets_an_ignored_burst():
    detector = make_detector(hang_s=0.25, min_speech_s=0.25)
    run(detector, (LOUD, 1.0), (QUIET, 1.125), (QUIET, 1.25))                      # a click, dropped
    results = run(detector, (QUIET, 3.0), (LOUD, 3.125), (LOUD, 3.25), (LOUD, 3.375))
    assert results == [None, None, None, ("start", 3.125)]                         # not 1.0


def test_detector_second_utterance_starts_again():
    detector = make_detector(hang_s=0.25)
    run(detector, (LOUD, 1.0), (QUIET, 1.125), (QUIET, 1.25))                      # start ... end
    assert run(detector, (LOUD, 2.0), (QUIET, 2.125), (QUIET, 2.25)) == [("start", 2.0), None, ("end", 2.0)]


def test_detector_reset_returns_to_silence():
    detector = make_detector(hang_s=0.25)
    assert run(detector, (LOUD, 1.0)) == [("start", 1.0)]
    detector.reset()
    assert not detector.in_speech
    assert run(detector, (QUIET, 1.125), (QUIET, 1.25), (QUIET, 1.375)) == [None] * 3   # no end for the old one
    assert run(detector, (LOUD, 1.5)) == [("start", 1.5)]


def test_detector_reset_drops_a_burst_that_had_not_started_yet():
    detector = make_detector(hang_s=0.25, min_speech_s=0.25)
    run(detector, (LOUD, 1.0), (LOUD, 1.125))
    detector.reset()
    assert run(detector, (LOUD, 1.25), (LOUD, 1.375), (LOUD, 1.5)) == [None, None, ("start", 1.25)]


# --- --monitor, with a stand-in mic: written already ----------------------------------------

class FakeRing:
    def last(self, seconds):
        return LOUD


class FakeMic:
    """Stands in for MicStream. Each read() hands out the next block and moves the clock to its t_end.
    When the blocks run out it sets `stop`, or, with idle_step, lets time pass with no audio."""

    def __init__(self, clock, blocks, stop=None, idle_step=None):
        self.clock = clock
        self.blocks = list(blocks)
        self.stop = stop
        self.idle_step = idle_step
        self.guard = HalfDuplexGuard(0.0, clock=clock)
        self.ring = FakeRing()
        self.seen = 0

    def ready(self):
        return True

    def read(self, timeout):
        if not self.blocks:
            if self.idle_step is not None:
                self.clock.t += self.idle_step
            elif self.stop is not None:
                self.stop.set()
            return None
        block = self.blocks.pop(0)
        self.clock.t = block.t_end
        self.seen += 1
        return block

    def stats(self):
        return ms.MicStats(blocks_seen=self.seen, blocks_queued=self.seen)


class Stop:
    """A threading.Event, as far as monitor() needs one."""

    def __init__(self):
        self.flag = False

    def set(self):
        self.flag = True

    def is_set(self):
        return self.flag


class ScriptedSpeech:
    """Stands in for SpeechDetector: gives the events it was told to, by block number."""

    def __init__(self, events=None, error=None):
        self.events = events or {}
        self.error = error
        self.log = []

    def reset(self):
        self.log.append("reset")

    def update(self, block):
        self.log.append(block.t_end)
        if self.error is not None:
            raise self.error
        return self.events.get(sum(1 for entry in self.log if entry != "reset"))


def blocks_from(t0, count):
    return [ms.Block(LOUD, t0 + (n + 1) * BLOCK_S) for n in range(count)]


def run_monitor(blocks, speech=None, **kwargs):
    clock = FakeClock(100.0)
    stop = Stop()
    mic = FakeMic(clock, blocks, stop, idle_step=kwargs.pop("idle_step", None))
    lines = []
    code = ms.monitor(mic, speech, stop, clock=clock, out=lines.append, **kwargs)
    return code, lines, clock


def test_monitor_prints_level_gate_and_counters_once_a_period():
    code, lines, clock = run_monitor(blocks_from(100.0, 16), period_s=1.0)     # two seconds of blocks
    assert code == 0
    status = [line for line in lines if "level" in line]
    assert len(status) == 2
    assert "-20.0" in status[0] and "dBFS" in status[0] and "gate open" in status[0]
    assert "seen 8 queued 8 gated 0 queue_full 0 overflows 0" in status[0]
    assert "seen 16" in status[1]
    assert lines[-1].startswith("Final") and "seen 16" in lines[-1]


def test_monitor_prints_speech_events():
    speech = ScriptedSpeech({2: ms.SpeechEvent("start", 100.25), 6: ms.SpeechEvent("end", 100.5)})
    code, lines, clock = run_monitor(blocks_from(100.0, 8), speech)
    assert len([line for line in lines if "speech start" in line]) == 1
    (end,) = [line for line in lines if "speech end" in line]
    assert "lasted 0.25 s" in end and "reported 0.25 s later" in end


def test_monitor_carries_on_when_todo_7_is_not_filled_in():
    speech = ScriptedSpeech(error=NotImplementedError("TODO 7"))
    code, lines, clock = run_monitor(blocks_from(100.0, 16), speech, period_s=1.0)
    assert code == 0
    assert len([line for line in lines if "TODO 7" in line]) == 1                # said once
    assert len(speech.log) == 1                                                  # and never asked again
    assert len([line for line in lines if "level" in line]) == 2                 # the rest still works


def test_monitor_resets_speech_detection_after_a_hole_in_the_blocks():
    blocks = blocks_from(100.0, 2) + blocks_from(101.0, 2)                       # the gate was shut in between
    speech = ScriptedSpeech()
    run_monitor(blocks, speech)
    assert speech.log == [100.125, 100.25, "reset", 101.125, 101.25]


def test_monitor_ends_with_code_1_when_the_mic_goes_silent():
    code, lines, clock = run_monitor(blocks_from(100.0, 2), idle_step=0.5, stall_s=2.0)
    assert code == 1
    assert any("No audio" in line for line in lines)
    assert lines[-1].startswith("Final")
    assert clock.t == pytest.approx(102.25)


def test_monitor_stops_by_itself_after_run_s():
    code, lines, clock = run_monitor(blocks_from(100.0, 80), run_s=1.0)
    assert code == 0
    assert clock.t == 101.0


def test_gate_name_says_why_blocks_are_held_back():
    clock = FakeClock()
    mic = FakeMic(clock, [])
    assert ms.gate_name(mic) == "open"
    mic.guard.start_speaking()
    assert ms.gate_name(mic) == "shut"
    mic.ready = lambda: False
    assert ms.gate_name(mic) == "settling"


def test_say_loop_shuts_the_gate_around_each_playback():
    guard = HalfDuplexGuard(0.0, clock=FakeClock())
    waits, played = [], []

    class StopAfterTwo:
        def wait(self, seconds):
            waits.append(seconds)
            return len(waits) > 2

    ms.say_loop("sound", guard, StopAfterTwo(), lambda sound: played.append((sound, guard.mic_open())), 7.0)
    assert played == [("sound", False), ("sound", False)]
    assert waits == [7.0, 7.0, 7.0]
    assert guard.mic_open()


# --- main, with the audio library faked -----------------------------------------------------

def test_main_needs_a_mode():
    with pytest.raises(SystemExit) as error:
        ms.main([])
    assert error.value.code == 2


@pytest.mark.parametrize("argv", [
    ["--dump-last", "0"],
    ["--dump-last", "11"],                               # longer than the default 10 s of history
    ["--monitor", "--seconds", "0"],
    ["--monitor", "--period", "0"],
    ["--monitor", "--window", "nan"],
    ["--monitor", "--settle", "-1"],
    ["--monitor", "--queue", "0"],
    ["--monitor", "--tail", "-0.1"],
    ["--monitor", "--start-dbfs", "-50", "--end-dbfs", "-30"],
    ["--monitor", "--hang", "-1"],
    ["--monitor", "--min-speech", "-1"],
    ["--dump-last", "1", "--say", "idle"],
    ["--monitor", "--say", "shouting"],
])
def test_main_rejects_bad_values(argv):
    with pytest.raises(SystemExit) as error:
        ms.main(argv)
    assert error.value.code == 2


def test_main_say_without_its_wav_is_a_message_not_a_traceback(tmp_path, capsys):
    assert ms.main(["--monitor", "--say", "idle", "--voice-dir", str(tmp_path)]) == 1
    assert "Missing WAV file" in capsys.readouterr().err


def test_main_port_audio_error_is_a_message_and_exit_code_1(fake_sd, capsys):
    fake_sd.open_error = fake_sd.PortAudioError("Device unavailable")
    assert ms.main(["--monitor", "--mic-device", "2"]) == 1
    assert "Device unavailable" in capsys.readouterr().err
    assert signal.getsignal(signal.SIGINT) is signal.default_int_handler    # handlers are put back


def test_main_dump_last_saves_the_newest_seconds(fake_sd, tmp_path, capsys):
    fed = np.arange(12 * 1024, dtype=np.int16)                               # 0.256 s at 48 kHz

    def on_start(stream):
        for n in range(12):
            feed(stream, fed[n * 1024:(n + 1) * 1024])

    fake_sd.on_start = on_start
    assert ms.main(["--dump-last", "0.25", "--mic-device", "2", "--settle", "0", "--out", str(tmp_path)]) == 0
    (path,) = tmp_path.glob("mic_*.wav")
    with wave.open(str(path), "rb") as w:
        assert (w.getnchannels(), w.getsampwidth(), w.getframerate()) == (1, 2, 48_000)
        saved = np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16)
    assert np.array_equal(saved, fed[-12_000:])
    assert str(path) in capsys.readouterr().out
    assert fake_sd.streams[0].closes == 1


def test_main_dump_last_with_no_audio_is_exit_code_1(fake_sd, tmp_path, capsys):
    assert ms.main(["--dump-last", "0.05", "--mic-device", "2", "--out", str(tmp_path)]) == 1
    assert "no audio" in capsys.readouterr().err
    assert list(tmp_path.glob("*.wav")) == []


def test_main_sigint_ends_the_monitor_with_a_final_line_and_a_closed_stream(fake_sd, capsys):
    def on_start(stream):
        for n in range(3):
            feed(stream, np.full(1024, 3277, dtype=np.int16))
        signal.raise_signal(signal.SIGINT)                                   # Ctrl+C, or `timeout -s INT`

    fake_sd.on_start = on_start
    try:
        code = ms.main(["--monitor", "--mic-device", "2", "--settle", "0"])
    except KeyboardInterrupt:
        pytest.fail("SIGINT reached Python's default handler: main() did not take it")
    assert code == 0
    out = capsys.readouterr().out
    assert "Final" in out and "seen 3 queued 3 gated 0 queue_full 0 overflows 0" in out
    assert (fake_sd.streams[0].stops, fake_sd.streams[0].closes) == (1, 1)
    assert signal.getsignal(signal.SIGINT) is signal.default_int_handler


def test_main_monitor_with_seconds_stops_by_itself(fake_sd, capsys):
    def on_start(stream):
        for n in range(3):
            feed(stream, np.full(1024, 3277, dtype=np.int16))

    fake_sd.on_start = on_start
    code = ms.main(["--monitor", "--mic-device", "2", "--settle", "0", "--seconds", "0.3",
                    "--period", "0.1", "--window", "0.05"])
    assert code == 0
    out = capsys.readouterr().out
    assert "level" in out and "-20.0" in out and "gate open" in out
    assert "Final" in out and "seen 3 queued 3" in out
