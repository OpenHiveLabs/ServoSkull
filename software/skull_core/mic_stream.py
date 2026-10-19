"""Always-open mic stream: one input stream for the whole run, a ring buffer, an end-of-speech detector.

The stream is opened once and never closed between turns, so the USB mic's level settles once
per run. A PortAudio callback copies every block into a ring buffer (for looking back) and hands
it to the listener through a bounded queue. Half-duplex: while the skull speaks, for a short tail
after, and while the mic is still settling, blocks reach the ring but not the queue.

Usage (from the repo root; -m is required):
    python -m software.skull_core.mic_stream --monitor [--mic-device UACDemoV1.0] [--seconds 300]
           [--period 1.0] [--window 0.5] [--settle 3.0] [--ring 10] [--queue 2.0] [--tail 0.3]
           [--detector] [--start-dbfs -35] [--end-dbfs -45] [--hang 0.6] [--min-speech 0.15]
           [--say STATE] [--say-every 10] [--device MAX98357A] [--voice-dir software/skull_core/voice]
           [--dump-last N] [--out captures/]
    python -m software.skull_core.mic_stream --dump-last 5 [--mic-device UACDemoV1.0] [--out captures/]

--monitor     prints the level, the gate and the counters about once a second, until Ctrl+C
              (or for --seconds).
--detector    also feeds the end-of-speech detector and prints its events.
--say STATE   plays that state's word every --say-every seconds with the mic gate shut around it.
--dump-last N saves the ring buffer's last N seconds to a WAV when the run ends. Without
              --monitor the run lasts --seconds, or N seconds.
"""
import argparse
import math
import queue
import signal
import sys
import threading
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np

from software.skull_core.state_console import DEFAULT_VOLUME, VOICE_DIR, WAV_RATE, load_wav, wav_path
from software.skull_core.status_leds import STATES
from software.skull_core.voice_console import DEFAULT_TAIL_S, HalfDuplexGuard

# The sample rate is record.SAMPLE_RATE. record.py imports sounddevice at the top, so it is
# imported inside functions only: importing this file (the tests, --help) needs no audio library.
# That is also why `rate` has no default value anywhere below.

DEFAULT_MIC_NAME = "UACDemoV1.0"   # the USB mic, picked by name: card numbers move when devices are plugged in
BLOCK_FRAMES = 1024                # frames per callback: 21.33 ms at 48 kHz
DEFAULT_SETTLE_S = 3.0             # estimate ("about 2 to 3 s", by ear). Measure on the Pi (check 3).
DEFAULT_RING_S = 10.0              # estimate of how far back anything needs to look. 0.96 MB at 48 kHz int16.
DEFAULT_QUEUE_S = 2.0              # placeholder, tune on the Pi (check 2): audio the listener may fall behind by
DEFAULT_START_DBFS = -35.0         # placeholder, tune on the Pi (check 3): a block this loud starts speech
DEFAULT_END_DBFS = -45.0           # placeholder, tune on the Pi (check 3): below this a block counts as quiet
DEFAULT_HANG_S = 0.6               # placeholder, tune on the Pi (check 6): quiet for this long ends the speech
DEFAULT_MIN_SPEECH_S = 0.15        # placeholder, tune on the Pi (check 5): shorter bursts (a click) are ignored

DEFAULT_PERIOD_S = 1.0             # --monitor: one status line per period
DEFAULT_WINDOW_S = 0.5             # --monitor: the level is taken over the last window
DEFAULT_SAY_EVERY_S = 10.0         # --say: seconds between two playbacks
READ_TIMEOUT_S = 0.1               # --monitor: longest wait for one block, so Ctrl+C is noticed quickly
STALL_S = 2.0                      # estimate: no new block for this long means the mic is gone (unplugged)
RESET_GAP_S = 0.15                 # estimate: a hole this long between two queued blocks means the gate was shut


class MicError(Exception):
    """The mic could not be found or opened. The text is meant to be shown to the user as it is."""


@dataclass(frozen=True, eq=False)
class Block:
    """One callback's worth of audio. samples: 1-D int16, mono. t_end: clock time of its last sample."""
    samples: np.ndarray
    t_end: float


@dataclass(frozen=True)
class MicStats:
    """Counters since open(). blocks_seen = blocks_queued + blocks_gated + queue_full, always.

    blocks_gated: not queued because the mic was settling or the guard was shut.
    queue_full: dropped because the listener was too slow.
    input_overflows: callbacks whose status said PortAudio itself lost input.
    """
    blocks_seen: int = 0
    blocks_queued: int = 0
    blocks_gated: int = 0
    queue_full: int = 0
    input_overflows: int = 0


@dataclass(frozen=True)
class SpeechEvent:
    """kind: "start" or "end". t: when the speech began or stopped, taken from the blocks' t_end."""
    kind: str
    t: float


def block_dbfs(samples: np.ndarray) -> float:
    """Level of some samples in dBFS (record.rms_dbfs). Call it on the listener's side, never in the callback."""
    from software.skull_core.record import rms_dbfs

    return float(rms_dbfs(samples))


class RingBuffer:
    """Keeps the last `seconds` of audio. Old samples are overwritten. Written by one thread, read by another.

    The storage is ready: self._data holds `capacity` int16 samples, self._write_at is where the next
    sample goes, self._filled is how many samples are valid so far (never more than capacity).
    Hold self._lock while you touch them: write() runs on the audio thread, last() on another one.
    """

    def __init__(self, seconds: float, rate: int):
        if not (math.isfinite(seconds) and seconds > 0):
            raise ValueError("seconds must be a finite value greater than 0")
        if rate <= 0:
            raise ValueError("rate must be positive")
        self.rate = rate
        self.capacity = int(seconds * rate)   # in samples
        if self.capacity < 1:
            raise ValueError("the buffer would hold no samples")
        self._data = np.zeros(self.capacity, dtype=np.int16)
        self._write_at = 0
        self._filled = 0
        self._lock = threading.Lock()

    def write(self, samples: np.ndarray) -> None:
        """Append a 1-D int16 block. The newest samples always win.

        A block may wrap past the end of the storage, and may be longer than the whole buffer
        (then only its newest `capacity` samples are kept). The buffer keeps its own copy.
        """
        samples = np.asarray(samples, dtype=np.int16)
        with self._lock:
            n = len(samples)
            if n >= self.capacity:
                self._data[:] = samples[-self.capacity:]
                self._write_at = 0
                self._filled = self.capacity
                return
            end = self._write_at + n
            if end <= self.capacity:
                self._data[self._write_at:end] = samples
            else:
                first = self.capacity - self._write_at
                self._data[self._write_at:] = samples[:first]
                self._data[:end % self.capacity] = samples[first:]
            self._write_at = end % self.capacity
            self._filled = min(self.capacity, self._filled + n)

    def last(self, seconds: float) -> np.ndarray:
        """The newest int(seconds * rate) samples, oldest first, as a new int16 array.

        Shorter if less than that has been written (no zero padding), and never more than `capacity`.
        ValueError if seconds < 0.
        """
        if not (math.isfinite(seconds) and seconds >= 0):
            raise ValueError("seconds must be a finite value >= 0")
        with self._lock:
            n = min(int(seconds * self.rate), self._filled)
            if n == 0:
                return np.zeros(0, dtype=np.int16)
            start = self._write_at - n
            if start >= 0:
                return self._data[start:self._write_at].copy()
            # two pieces: the end of the storage, then its front (np.concatenate makes a new array)
            return np.concatenate((self._data[start:], self._data[:self._write_at]))


def find_input_device(name: str, devices: list[dict]) -> int:
    """Index of the one input device whose name contains `name`.

    devices: what sounddevice.query_devices() gives, as a list of dicts with at least "name" and
    "max_input_channels". A device's index is its position in that list. Devices that have no input
    channels are never candidates. ValueError if no input matches or if two or more do; the message
    names `name` and lists the names of the inputs that do exist.
    """
    matches = [i for i, d in enumerate(devices) if d["max_input_channels"] > 0 and name in d["name"]]
    if len(matches) == 1:
        return matches[0]
    names = [d["name"] for d in devices if d["max_input_channels"] > 0]
    if len(matches) == 0:
        raise ValueError(f"no input device contains {name!r}; available inputs: {names}")
    raise ValueError(f"multiple input devices contain {name!r}: {[devices[i]['name'] for i in matches]}; available inputs: {names}")


class SpeechDetector:
    """End-of-speech detector on block levels. It only observes: it never decides what a listener gets.

    A block is loud when block_dbfs(block.samples) >= start_dbfs, quiet when it is < end_dbfs.
    Anything in between changes nothing: it cannot begin speech, and it cannot end it either.

    - Speech begins with a loud block. Its "start" event is returned once the speech has lasted
      min_speech_s (first loud block to the latest block that was not quiet). With min_speech_s = 0
      that is the loud block itself. The event's t is the first loud block's t_end.
    - Speech ends when every block has been quiet for hang_s: the block whose t_end is hang_s or
      more after the last non-quiet block returns "end". The event's t is that last non-quiet
      block's t_end, so the hang time is not part of it.
    - A burst that goes quiet for hang_s before it has lasted min_speech_s gives no events at all.

    All times come from the blocks' t_end. The detector never reads a clock.
    """

    def __init__(self, start_dbfs: float, end_dbfs: float, hang_s: float, min_speech_s: float):
        if end_dbfs > start_dbfs:
            raise ValueError("end_dbfs must not be above start_dbfs")
        if hang_s < 0 or min_speech_s < 0:
            raise ValueError("hang_s and min_speech_s must be >= 0")
        self.start_dbfs = start_dbfs
        self.end_dbfs = end_dbfs
        self.hang_s = hang_s
        self.min_speech_s = min_speech_s
        self.reset()

    def reset(self) -> None:
        """Back to silence, with nothing remembered. Called when the mic gate reopens, so the skull's
        own voice and its tail never count as speech."""
        self.in_speech = False        # True from a returned "start" until the matching "end"
        self._began_at = None         # t_end of the first loud block of the current burst, or None
        self._last_voiced_at = None   # t_end of the latest block of that burst that was not quiet
        # If update() needs to remember anything else between blocks, clear it here as well.

    def update(self, block: Block) -> SpeechEvent | None:
        """Feed one block. Returns a "start" or an "end" SpeechEvent when one is due, else None.

        Keeps self.in_speech true between the two events. See the class docstring for the rules.

        Three states, read from the fields that reset() names:
          silence  _began_at is None                     waiting for a loud block
          burst    _began_at is set, in_speech False     begun, not long enough yet to call it speech
          speech   in_speech True                        "start" returned, "end" not yet
        """
        level = block_dbfs(block.samples)
        loud = level >= self.start_dbfs
        quiet = level < self.end_dbfs   # a block that is neither sits in the dead band: it moves nothing
        t = block.t_end

        # Silence: only a loud block can begin something. Between-level blocks are ignored here,
        # so a voice hovering just under start_dbfs never starts speech.
        if self._began_at is None and not loud:
            return None

        # Burst or speech, and this block is not quiet: the sound goes on.
        if not quiet:
            if self._began_at is None:
                self._began_at = t          # silence -> burst
            self._last_voiced_at = t
            # burst -> speech once it has lasted min_speech_s (at once when that is 0).
            if not self.in_speech and t - self._began_at >= self.min_speech_s:
                self.in_speech = True
                return SpeechEvent("start", self._began_at)
            return None

        # A quiet block in a burst or in speech: wait out the hang time, so a pause between words
        # does not end the utterance.
        if t - self._last_voiced_at < self.hang_s:
            return None
        # Quiet for hang_s: back to silence. Only speech reports an end; a burst that never
        # became speech (a click) is forgotten without any event.
        was_speech = self.in_speech
        stopped_at = self._last_voiced_at
        self.reset()
        return SpeechEvent("end", stopped_at) if was_speech else None


class MicStream:
    """The one input stream. open() at start-up, close() at exit, nothing in between.

    device: an input device name (or a part of one), or a PortAudio device index.
    guard: the half-duplex guard shared with whatever plays sound.
    clock: returns seconds. time.monotonic on the Pi; the tests pass a fake they move by hand.
    """

    def __init__(self, device, guard: HalfDuplexGuard, rate: int, block: int = BLOCK_FRAMES,
                 settle_s: float = DEFAULT_SETTLE_S, ring_s: float = DEFAULT_RING_S,
                 queue_s: float = DEFAULT_QUEUE_S, clock=time.monotonic):
        if block < 1:
            raise ValueError("block must be at least 1 frame")
        if settle_s < 0:
            raise ValueError("settle_s must be >= 0")
        if queue_s <= 0:
            raise ValueError("queue_s must be greater than 0")
        self.device = device
        self.guard = guard
        self.rate = rate
        self.block = block
        self.settle_s = settle_s
        self.clock = clock
        self.ring = RingBuffer(ring_s, rate)
        self.queue_blocks = max(1, math.ceil(queue_s * rate / block))   # how many blocks may wait
        self._queue: queue.Queue = queue.Queue(maxsize=self.queue_blocks)
        self._stream = None
        self._opened_at = None        # clock time of open(); None while the stream is closed
        # Counters. Only the callback changes them; stats() reads them.
        self._blocks_seen = 0
        self._blocks_queued = 0
        self._blocks_gated = 0
        self._queue_full = 0
        self._input_overflows = 0

    def open(self) -> None:
        """Open and start the input stream. MicError, with a text fit to print, if that fails."""
        if self._stream is not None:
            return
        import sounddevice as sd   # only here: importing this file needs no audio library

        stream = None
        try:
            if isinstance(self.device, str):
                index = find_input_device(self.device, list(sd.query_devices()))
            else:
                index = self.device
            stream = sd.InputStream(samplerate=self.rate, channels=1, dtype="int16", blocksize=self.block,
                                    device=index, callback=self._on_audio)
            self._opened_at = self.clock()   # before start(): the callback may run at once
            stream.start()
        except (ValueError, sd.PortAudioError) as exc:
            self._opened_at = None
            if stream is not None:
                try:
                    stream.close()
                except sd.PortAudioError:
                    pass
            raise MicError(f"Cannot open the mic {self.device!r}: {exc}") from exc
        self._stream = stream

    def close(self) -> None:
        """Stop and close the stream. Safe to call twice, and before open()."""
        stream, self._stream = self._stream, None
        self._opened_at = None
        if stream is None:
            return
        import sounddevice as sd

        try:
            stream.stop()
        except sd.PortAudioError:
            pass   # an unplugged mic cannot be stopped; it is closed all the same
        finally:
            try:
                stream.close()
            except sd.PortAudioError:
                pass

    def ready(self) -> bool:
        """True once settle_s has passed since open(). The mic's level is not trustworthy before.

        False while the stream is closed (self._opened_at is None). At exactly settle_s it is ready.
        """
        clock = self.clock()
        if self._opened_at is None:
            return False
        if clock - self._opened_at >= self.settle_s:
            return True
        return False

    def _on_audio(self, indata, frames, time_info, status) -> None:
        """The PortAudio callback: runs on the audio thread, once per block.

        indata is int16 with shape (frames, 1), and PortAudio reuses its memory after the call.
        Every block: count it, count status.input_overflow if it is set, and write a 1-D copy to self.ring.
        Then exactly one of: gated (not ready(), or the guard is shut), queued as a Block stamped with
        self.clock(), or dropped because the queue was full. Each has its counter.
        Never waits, never prints, never raises.
        """
        samples = np.array(indata, dtype=np.int16).reshape(-1)  # np.array copies; PortAudio reuses indata
        self._blocks_seen += 1
        if status.input_overflow:
            self._input_overflows += 1
        self.ring.write(samples)
        now = self.clock()  # stamp the block with the time of its last sample
        if not self.ready() or not self.guard.mic_open():
            self._blocks_gated += 1
            return
        block = Block(samples, now)
        try:
            self._queue.put_nowait(block)
        except queue.Full:
            self._queue_full += 1
        else:
            self._blocks_queued += 1

    def read(self, timeout: float) -> Block | None:
        """The next queued block, oldest first, or None if none arrived within `timeout` seconds."""
        if timeout < 0:
            raise ValueError("timeout must be >= 0")
        try:
            return self._queue.get(timeout=timeout)
        except queue.Empty:
            return None

    def stats(self) -> MicStats:
        return MicStats(self._blocks_seen, self._blocks_queued, self._blocks_gated,
                        self._queue_full, self._input_overflows)


def gate_name(mic) -> str:
    """Why blocks are, or are not, reaching the listener right now."""
    if not mic.ready():
        return "settling"
    return "open" if mic.guard.mic_open() else "shut"


def counters_text(stats: MicStats) -> str:
    return (f"seen {stats.blocks_seen} queued {stats.blocks_queued} gated {stats.blocks_gated} "
            f"queue_full {stats.queue_full} overflows {stats.input_overflows}")


def print_now(text: str) -> None:
    """print() with flush, so each --monitor line shows at once when piped into `tee`."""
    print(text, flush=True)


def monitor(mic, detector=None, stop=None, *, run_s=None, period_s=DEFAULT_PERIOD_S,
            window_s=DEFAULT_WINDOW_S, stall_s=STALL_S, clock=time.monotonic, out=print_now) -> int:
    """Read the stream and report on it until `stop` is set, or for run_s seconds.

    Every period_s: the level of the ring's last window_s, the gate and the counters.
    With a detector: its events as they happen. It is reset after a hole in the queued blocks
    (the gate was shut), so the skull's own voice never counts as speech.
    Returns 0, or 1 if the mic stopped delivering blocks for stall_s.
    """
    stop = stop if stop is not None else threading.Event()
    started = clock()
    next_line = started + period_s
    seen = mic.stats().blocks_seen
    last_audio = started
    last_t_end = None
    speech_began = None
    code = 0

    while not stop.is_set():
        block = mic.read(READ_TIMEOUT_S)
        now = clock()
        stamp = f"[{now - started:7.1f} s]"

        stats = mic.stats()
        if stats.blocks_seen != seen:
            seen, last_audio = stats.blocks_seen, now
        elif now - last_audio >= stall_s:
            out(f"{stamp} No audio for {stall_s:g} s: the mic stopped delivering blocks (unplugged?).")
            code = 1
            break

        if block is not None and detector is not None:
            if last_t_end is not None and block.t_end - last_t_end > RESET_GAP_S:
                detector.reset()
                speech_began = None
            try:
                event = detector.update(block)
            except NotImplementedError:
                out(f"{stamp} SpeechDetector.update is not filled in yet (TODO 7): carrying on without it.")
                detector = None
            else:
                if event is not None and event.kind == "start":
                    speech_began = event.t
                    out(f"{stamp} speech start at {event.t - started:.2f} s")
                elif event is not None:
                    lasted = "" if speech_began is None else f", lasted {event.t - speech_began:.2f} s"
                    out(f"{stamp} speech end at {event.t - started:.2f} s{lasted}, "
                        f"reported {block.t_end - event.t:.2f} s later")
                    speech_began = None
        if block is not None:
            last_t_end = block.t_end

        if now >= next_line:
            recent = mic.ring.last(window_s)
            level = f"{block_dbfs(recent):7.2f} dBFS" if len(recent) else "no audio yet"
            out(f"{stamp} level {level} | gate {gate_name(mic)} | {counters_text(stats)}")
            while next_line <= now:
                next_line += period_s

        if run_s is not None and now - started >= run_s:
            break

    out(f"Final after {clock() - started:.1f} s: {counters_text(mic.stats())}")
    return code


def say_loop(sound, guard: HalfDuplexGuard, stop, play, every_s: float) -> None:
    """Every every_s seconds, until `stop` is set: play(sound) with the mic gate shut around it."""
    while not stop.wait(every_s):
        with guard.speaking():
            play(sound)


def parse_args(argv=None) -> argparse.Namespace:
    int_or_name = lambda value: int(value) if value.isdigit() else value  # noqa: E731
    parser = argparse.ArgumentParser(description="Always-open mic stream: watch it (--monitor) or save its "
                                                 "last seconds (--dump-last)")
    parser.add_argument("--monitor", action="store_true", help="print level, gate and counters until Ctrl+C")
    parser.add_argument("--dump-last", type=float, default=None, metavar="N",
                        help="when the run ends, save the ring's last N seconds as a WAV")
    parser.add_argument("--seconds", type=float, default=None, help="stop by itself after this long")
    parser.add_argument("--mic-device", type=int_or_name, default=DEFAULT_MIC_NAME,
                        help="input: a part of the sounddevice name, or an index")
    parser.add_argument("--out", type=Path, default=Path("captures"), help="folder for --dump-last")
    parser.add_argument("--period", type=float, default=DEFAULT_PERIOD_S, help="seconds between status lines")
    parser.add_argument("--window", type=float, default=DEFAULT_WINDOW_S, help="seconds the level is taken over")
    parser.add_argument("--settle", type=float, default=DEFAULT_SETTLE_S,
                        help="seconds after opening before blocks reach the listener")
    parser.add_argument("--ring", type=float, default=DEFAULT_RING_S, help="ring buffer length in seconds")
    parser.add_argument("--queue", type=float, default=DEFAULT_QUEUE_S,
                        help="seconds of audio that may wait for the listener")
    parser.add_argument("--tail", type=float, default=DEFAULT_TAIL_S, help="mic stays shut this long after speaking")
    parser.add_argument("--detector", action="store_true", help="feed the end-of-speech detector, print its events")
    parser.add_argument("--start-dbfs", type=float, default=DEFAULT_START_DBFS)
    parser.add_argument("--end-dbfs", type=float, default=DEFAULT_END_DBFS)
    parser.add_argument("--hang", type=float, default=DEFAULT_HANG_S, help="quiet seconds that end the speech")
    parser.add_argument("--min-speech", type=float, default=DEFAULT_MIN_SPEECH_S,
                        help="bursts shorter than this are ignored")
    parser.add_argument("--say", choices=list(STATES), default=None,
                        help="with --monitor: play this state's word again and again, the gate shut around it")
    parser.add_argument("--say-every", type=float, default=DEFAULT_SAY_EVERY_S, help="seconds between playbacks")
    parser.add_argument("--device", type=int_or_name, default="MAX98357A", help="output for --say: name or index")
    parser.add_argument("--voice-dir", type=Path, default=VOICE_DIR)
    args = parser.parse_args(argv)

    def check(ok: bool, message: str) -> None:
        if not ok:
            parser.error(message)

    positive = lambda value: math.isfinite(value) and value > 0     # noqa: E731
    not_negative = lambda value: math.isfinite(value) and value >= 0  # noqa: E731

    check(args.monitor or args.dump_last is not None, "nothing to do: pass --monitor, --dump-last N, or both")
    check(positive(args.ring), "--ring must be a finite value greater than 0")
    check(args.dump_last is None or (positive(args.dump_last) and args.dump_last <= args.ring),
          "--dump-last must be greater than 0 and no longer than --ring")
    check(args.seconds is None or positive(args.seconds), "--seconds must be a finite value greater than 0")
    check(positive(args.period), "--period must be a finite value greater than 0")
    check(positive(args.window) and args.window <= args.ring,
          "--window must be greater than 0 and no longer than --ring")
    check(not_negative(args.settle), "--settle must be >= 0")
    check(positive(args.queue), "--queue must be a finite value greater than 0")
    check(not_negative(args.tail), "--tail must be >= 0")
    check(math.isfinite(args.start_dbfs) and math.isfinite(args.end_dbfs) and args.end_dbfs <= args.start_dbfs,
          "--end-dbfs must not be above --start-dbfs")
    check(not_negative(args.hang), "--hang must be >= 0")
    check(not_negative(args.min_speech), "--min-speech must be >= 0")
    check(args.say is None or args.monitor, "--say needs --monitor")
    check(positive(args.say_every), "--say-every must be a finite value greater than 0")
    return args


def main(argv=None) -> int:
    args = parse_args(argv)

    sound = None
    if args.say is not None:
        path = wav_path(args.say, args.voice_dir)
        if not path.exists():
            print(f"Missing WAV file: {path}", file=sys.stderr)
            return 1
        try:
            sound = load_wav(path) * DEFAULT_VOLUME
        except ValueError as exc:
            print(exc, file=sys.stderr)
            return 1

    # Audio imports only here: importing this file (the tests, --help) needs none of them.
    import sounddevice as sd

    from software.skull_core.record import SAMPLE_RATE, save_wav

    guard = HalfDuplexGuard(args.tail)
    mic = MicStream(args.mic_device, guard, SAMPLE_RATE, settle_s=args.settle, ring_s=args.ring, queue_s=args.queue)
    detector = None
    if args.detector:
        detector = SpeechDetector(args.start_dbfs, args.end_dbfs, args.hang, args.min_speech)

    stop = threading.Event()
    talker = None
    if sound is not None:
        def play(samples):
            try:
                sd.play(samples, samplerate=WAV_RATE, device=args.device, blocking=True)
            except sd.PortAudioError as exc:
                print(f"Audio playback failed: {exc}")

        talker = threading.Thread(target=say_loop, args=(sound, guard, stop, play, args.say_every), daemon=True)

    # Ctrl+C and `timeout -s INT` both end the run through `stop`, so the final line is always printed.
    old_handlers = {}
    for signum in (signal.SIGINT, signal.SIGTERM):
        old_handlers[signum] = signal.signal(signum, lambda *_: stop.set())

    code = 0
    try:
        try:
            mic.open()
        except MicError as exc:
            print(exc, file=sys.stderr)
            return 1
        print(f"Mic {args.mic_device!r} open: {SAMPLE_RATE} Hz, mono, int16, blocks of {BLOCK_FRAMES} frames. "
              f"Settling for {args.settle:g} s.")
        if talker is not None:
            talker.start()
        if args.monitor:
            print("Ctrl+C to stop.")
            code = monitor(mic, detector, stop, run_s=args.seconds, period_s=args.period, window_s=args.window)
        else:
            stop.wait(args.seconds if args.seconds is not None else args.dump_last)
    finally:
        stop.set()
        mic.close()
        if talker is not None and talker.ident is not None:
            talker.join()
        for signum, handler in old_handlers.items():
            signal.signal(signum, handler)

    if args.dump_last is not None:
        samples = mic.ring.last(args.dump_last)
        if len(samples) == 0:
            print("Nothing to save: no audio arrived.", file=sys.stderr)
            return 1
        args.out.mkdir(parents=True, exist_ok=True)
        out_path = args.out / f"mic_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}.wav"
        save_wav(out_path, samples, SAMPLE_RATE)
        print(f"Saved the last {len(samples) / SAMPLE_RATE:.2f} s to {out_path}")
        print(f"Level: {block_dbfs(samples):.2f} dBFS")
    return code


if __name__ == "__main__":
    sys.exit(main())
