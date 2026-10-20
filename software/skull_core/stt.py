"""Speech to text: one utterance from the always-open mic stream -> Moonshine Tiny -> text.

The mic stream (mic_stream.MicStream) delivers 48 kHz mono int16 blocks. Each block is converted
to float and handed to a Moonshine stream as it arrives, at its own 48 kHz rate: Moonshine's
library resamples to the 16 kHz its model works at. There is no resampling code here.

Moonshine does not return text from a call. It calls us back from inside add_audio() when a line
of speech is finished (its built-in voice activity detector decided the line ended). The callback
only stores the line; the loop that feeds the audio looks at what was stored after each block.

Our SpeechDetector (M3-02) watches the same blocks. Which of the two ends the turn is END_RULES:
  "line"      the first line Moonshine finishes ends the turn (the M3-02 / ADR-013 plan).
  "detector"  our detector's "end" ends the turn; Moonshine's unfinished line is then flushed with
              stream.stop(). If our detector hears no speech when Moonshine finishes a line (a quiet
              voice under its start threshold), that line ends the turn instead.
Either way the turn also ends after max_s, and gives up after start_timeout_s if nobody spoke.

moonshine_voice is imported only inside load_stream_maker(), so the tests and --help need no model.
"""
import re
import time
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

from software.skull_core.mic_stream import RESET_GAP_S, STALL_S

DEFAULT_MODEL_ARCH = 2          # moonshine_voice ModelArch: 2 = tiny-streaming (ADR-013, M3-00 (d))
DEFAULT_START_TIMEOUT_S = 5.0   # placeholder (M3-04 note, decision 3): wait this long for speech to begin
DEFAULT_MAX_S = 15.0            # placeholder (decision 3): the longest a turn listens, from its start
DEFAULT_MIN_LETTERS = 2         # placeholder (decision 2): fewer letters than this counts as nothing said
READ_TIMEOUT_S = 0.1            # longest wait for one block, so the deadlines are checked often
END_RULES = ("line", "detector")
DEFAULT_END_RULE = "line"       # the approved M3-02 plan: Moonshine's VAD ends the turn

MODEL_MISSING = ("moonshine_voice is not installed in this Python. Run with the venv that has it "
                 "(on the Pi: ~/m3-bench/.venv-stt/bin/python), or pip install moonshine-voice==0.1.5.")


@dataclass
class Heard:
    """What one listening turn produced. All t_* are clock times (time.monotonic on the Pi), or None.

    text: the usable text (see usable_text), or None when nothing usable was said.
    raw_text: every line Moonshine finished in this turn, joined with spaces, exactly as written.
    end_reason: "line", "detector", "max_s", "no_speech", "no_audio" (the mic stopped delivering)
      or "interrupted" (listen_once's interrupt() said stop before anyone spoke).
    t_speech_end: where Moonshine's last line ends, moved onto the clock (see Listener.speech_end).
    stt_s: t_text - t_speech_end, the same measure as M3-00's 0.223 s (speech end -> text in hand).
      Quote it for the "line" rule. Under the "detector" rule the last line comes from the flush,
      and Moonshine can place that line's end later than the speech really ended, so stt_s reads
      short there: quote detector_end_to_text_s instead (Surface, 2026-10-20: 0.33-0.38 s vs 0.66-0.71 s).
    first_line_*: Moonshine's first finished line and its stt_s. Under the "detector" rule this shows
      what the "line" rule would have sent, so one run measures both rules.
    detector_*: our SpeechDetector's start and end, observed on the same blocks.
    queue_full / gaps: blocks lost during the turn. Either one > 0 makes the times unreliable,
      because Moonshine's line times count the samples it was fed, not the clock.
    """
    text: str | None
    raw_text: str = ""
    end_reason: str = "no_speech"
    lines: int = 0
    t_listen: float | None = None
    t_first_sample: float | None = None
    t_speech_end: float | None = None
    t_text: float | None = None
    stt_s: float | None = None
    first_line_text: str | None = None
    first_line_stt_s: float | None = None
    detector_start: float | None = None
    detector_end: float | None = None
    detector_end_to_text_s: float | None = None
    moonshine_latency_ms: int | None = None
    blocks_fed: int = 0
    blocks_drained: int = 0
    queue_full: int = 0
    gaps: int = 0


def to_float(samples: np.ndarray) -> np.ndarray:
    """int16 -> float32 in -1.0 .. just under 1.0, the format Moonshine's add_audio takes.

    Divide by 32768 (2**15), not 32767: then -32768 maps to exactly -1.0 and nothing goes over.
    astype() makes a new array first, so the caller's block is never changed.
    """
    return np.asarray(samples, dtype=np.int16).astype(np.float32) / np.float32(32768.0)


def usable_text(text: str, min_letters: int = DEFAULT_MIN_LETTERS) -> str | None:
    """The stripped text, or None when it has fewer than min_letters letters (a cough, "Uh.", "")."""
    text = text.strip()
    letters = sum(1 for ch in text if ch.isalpha())
    return text if letters >= min_letters else None


def words(text: str) -> list[str]:
    """Lower-case words, punctuation dropped (apostrophes kept), for comparing what was said."""
    return re.findall(r"[a-z0-9']+", text.lower())


def word_errors(expected: str, heard: str) -> int:
    """How many words must be swapped, added or removed to turn `heard` into `expected`.

    The word-level edit distance (Levenshtein). Divided by the number of expected words it is the
    word error rate (WER), the usual accuracy measure for speech to text.
    """
    want, got = words(expected), words(heard)
    previous = list(range(len(got) + 1))   # row 0: turning nothing into got[:j] takes j insertions
    for i, w in enumerate(want, start=1):
        current = [i]
        for j, g in enumerate(got, start=1):
            current.append(min(previous[j] + 1,                  # a word of `expected` is missing
                               current[j - 1] + 1,               # an extra word was heard
                               previous[j - 1] + (w != g)))      # same word, or a wrong one
        previous = current
    return previous[-1]


def load_stream_maker(model_arch: int = DEFAULT_MODEL_ARCH) -> Callable[[], object]:
    """Load Moonshine once and return its create_stream, for one new stream per turn.

    ImportError with MODEL_MISSING if moonshine_voice is not installed. If the model is not in
    ~/.cache/moonshine_voice yet, Moonshine downloads it here (needs the network once).
    """
    try:
        from moonshine_voice import ModelArch, Transcriber, get_model_for_language
    except ImportError as exc:
        raise ImportError(MODEL_MISSING) from exc
    path, arch = get_model_for_language("en", ModelArch(model_arch))
    transcriber = Transcriber(model_path=path, model_arch=arch)
    return transcriber.create_stream


class Listener:
    """Wraps one Moonshine stream per turn. feed() hands it a block; finished lines collect in self.lines.

    make_stream: returns a new, unstarted Moonshine stream (Transcriber.create_stream, or a fake).
    The stream calls _on_event from inside add_audio() or stop(), on the thread that called them.
    """

    def __init__(self, make_stream: Callable[[], object], rate: int, clock=time.monotonic):
        self.make_stream = make_stream
        self.rate = rate
        self.clock = clock
        self._stream = None
        self.lines: list = []            # finished lines (Moonshine's TranscriptLine), oldest first
        self.line_times: list = []       # clock time each one reached us
        self.speech_started = False      # Moonshine began a line in this turn
        self.samples_fed = 0

    def start_turn(self) -> None:
        """A fresh stream, so line times count from this turn's first sample."""
        self.close()
        self.lines, self.line_times = [], []
        self.speech_started = False
        self.samples_fed = 0
        self._stream = self.make_stream()
        self._stream.add_listener(self._on_event)
        self._stream.start()

    def _on_event(self, event) -> None:
        """Moonshine's callback. Only store what arrived; the feeding loop acts on it."""
        kind = type(event).__name__
        if kind == "LineStarted":
            self.speech_started = True
        elif kind == "LineCompleted":
            self.speech_started = True
            self.lines.append(event.line)
            self.line_times.append(self.clock())

    def feed(self, samples: np.ndarray) -> None:
        """One block into Moonshine, at the mic's own rate. A transcription pass may run inside."""
        self._stream.add_audio(to_float(samples), self.rate)
        self.samples_fed += len(samples)

    def finish(self) -> None:
        """Flush: stop() makes Moonshine finish the line it is still in, if any."""
        if self._stream is not None:
            self._stream.stop()

    def close(self) -> None:
        stream, self._stream = self._stream, None
        if stream is not None:
            stream.close()

    def text(self) -> str:
        return " ".join(line.text.strip() for line in self.lines if line.text.strip())

    def speech_end(self, t_first_sample: float, index: int = -1) -> float:
        """Clock time where a finished line's speech ends: the turn's first sample plus the line's
        start_time + duration, which Moonshine counts in seconds of audio fed to this stream."""
        line = self.lines[index]
        return t_first_sample + line.start_time + line.duration


def drain(mic) -> int:
    """Throw away the blocks that queued up while nobody was listening (before Enter). Returns how many."""
    count = 0
    while mic.read(0) is not None:
        count += 1
    return count


def listen_once(mic, listener: Listener, detector, *, end_rule: str = DEFAULT_END_RULE,
                start_timeout_s: float = DEFAULT_START_TIMEOUT_S, max_s: float = DEFAULT_MAX_S,
                min_letters: int = DEFAULT_MIN_LETTERS, stall_s: float = STALL_S,
                clock=time.monotonic, interrupt: Callable[[], bool] | None = None) -> Heard:
    """Read blocks from the mic and feed Moonshine and our detector until the turn ends (see END_RULES).

    Never raises for "nobody spoke": that is a Heard with text None. The mic gate is not checked
    here; MicStream already keeps gated blocks out of the queue.
    interrupt (M3-07): checked before each block while nobody is speaking; True ends the turn with
    end_reason "interrupted" (the wake loop uses it to notice Enter). Once speech began it is ignored.
    """
    if end_rule not in END_RULES:
        raise ValueError(f"end_rule must be one of {END_RULES}")
    drained = drain(mic)
    stats_before = mic.stats()
    listener.start_turn()
    detector.reset()
    heard = Heard(text=None, blocks_drained=drained)
    t_listen = heard.t_listen = clock()
    last_audio = t_listen
    last_t_end = None
    try:
        while True:
            now = clock()
            if now - t_listen >= max_s:
                listener.finish()
                heard.end_reason = "max_s"
                break
            speech_seen = listener.speech_started or detector.in_speech or heard.detector_start is not None
            if not speech_seen and now - t_listen >= start_timeout_s:
                heard.end_reason = "no_speech"
                break
            if not speech_seen and interrupt is not None and interrupt():
                heard.end_reason = "interrupted"
                break
            block = mic.read(READ_TIMEOUT_S)
            if block is None:
                if clock() - last_audio >= stall_s:
                    heard.end_reason = "no_audio"
                    break
                continue
            last_audio = clock()
            if heard.t_first_sample is None:
                heard.t_first_sample = block.t_end - len(block.samples) / listener.rate
            if last_t_end is not None and block.t_end - last_t_end > RESET_GAP_S:
                heard.gaps += 1          # a hole in the audio: Moonshine's line times shift
                detector.reset()
            last_t_end = block.t_end

            listener.feed(block.samples)
            heard.blocks_fed += 1
            event = detector.update(block)
            if event is not None and event.kind == "start":
                heard.detector_start = event.t
            elif event is not None:
                heard.detector_end = event.t

            if listener.lines and heard.first_line_text is None:
                heard.first_line_text = listener.lines[0].text.strip()
                heard.first_line_stt_s = listener.line_times[0] - listener.speech_end(heard.t_first_sample, 0)
            if end_rule == "line" and listener.lines:
                heard.end_reason = "line"
                break
            if end_rule == "detector":
                if event is not None and event.kind == "end":
                    listener.finish()
                    heard.end_reason = "detector"
                    break
                if listener.lines and not detector.in_speech:
                    heard.end_reason = "line"
                    break
    finally:
        listener.close()

    if listener.lines:
        heard.t_text = clock() if heard.end_reason in ("detector", "max_s") else listener.line_times[-1]
        heard.t_speech_end = listener.speech_end(heard.t_first_sample)
        heard.stt_s = heard.t_text - heard.t_speech_end
        heard.moonshine_latency_ms = listener.lines[-1].last_transcription_latency_ms
        if heard.first_line_text is None:      # the only line came from the flush in finish()
            heard.first_line_text = listener.lines[0].text.strip()
            heard.first_line_stt_s = heard.stt_s if len(listener.lines) == 1 else None
    if heard.detector_end is not None and heard.t_text is not None:
        heard.detector_end_to_text_s = heard.t_text - heard.detector_end
    heard.lines = len(listener.lines)
    heard.raw_text = listener.text()
    heard.text = usable_text(heard.raw_text, min_letters)
    heard.queue_full = mic.stats().queue_full - stats_before.queue_full
    return heard
