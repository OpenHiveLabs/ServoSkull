"""Voice loop: wake phrase or Enter -> listen (mic stream + Moonshine) -> text -> brain -> reply spoken.

One turn is run_turn(listen, ask, speak, on_state): four parts handed in from outside, so the
turn can be tested with fakes and later steps swap one part without rewriting it (M3-05 swaps
speak for a voice, M3-06 swaps on_state for a publish, M3-07 swaps listen for the wake phrase).

Every turn appends one JSON line to the log: what was heard, Moonshine's times, our detector's
times, the brain's record, the voice's times (M3-05: Cogitating, first reply audio, the gaps between
sentences, each clip), and, with --phrases, what should have been heard and the word errors.

The reply is spoken by voice_out.Speaker: Cogitating at once, then sentence by sentence through the
amp, the mic gate shut throughout. Pocket TTS runs in its own process from --tts-python (on the Pi
~/m3-bench/.venv-tts/bin/python). --no-speak prints the reply instead, as in M3-04.

M3-06: every state (idle, listening, thinking, speaking, offline, error) is published on
<key_prefix>/state/voice through skull_bridge; the eye LED is one subscriber (voice_state.eye_led),
--watch-state prints another. The loop itself never touches the LED.

M3-07: --trigger wake|enter|both (default both). In wake mode Moonshine listens all the time and a
turn starts when an utterance begins with "Awaken machine spirit" (wake.split_wake). The question can
follow in the same breath, or within --wake-window seconds after a bare wake. Every utterance heard
while idle goes to --line-log with whether it woke. Before the brain is called, a TCP probe to
--probe-host says whether the network is there; offline -> the canned line from the bank, LED
"offline", no brain call. A brain failure (timeout, no network) is the backstop: its fallback line
plays under "offline" too. Enter stays as a second trigger.

Usage (from the repo root; -m is required; Moonshine must be importable, see stt.MODEL_MISSING):
    python -m software.skull_core.voice_loop [--mic-device UACDemoV1.0] [--end-rule detector|line] [--hang 0.6]
           [--phrases FILE] [--no-brain] [--start-timeout 5] [--max 15] [--min-letters 2]
           [--log ~/m3-bench/logs/m304_turns.jsonl] [--settle 3.0] [--model-arch 2]
           [--brain-dir ~/skull-brain] [--model claude-haiku-5-5] [--timeout 10]
           [--system-prompt-file FILE] [--no-thinking] [--claude claude]
           [--no-speak] [--tts-python ~/m3-bench/.venv-tts/bin/python] [--voice javert]
           [--bank ~/skull-voice/bank] [--device MAX98357A] [--voice-dir software/skull_core/voice]
           [--params robot_params.yaml] [--pins 5 6 13] [--common-anode] [--blink-period 1.0] [--no-led]
           [--watch-state] [--trigger both] [--wake-window 6] [--line-log ~/m3-bench/logs/m307_wake_lines.jsonl]
           [--probe-host api.anthropic.com] [--probe-port 443] [--probe-timeout 2] [--no-probe]
    python -m software.skull_core.voice_loop --report N [--log ...]

Without --phrases: say the wake phrase or press Enter, speak, see the text and the reply; "q" + Enter
quits (Ctrl+C in --trigger wake).
With --phrases FILE (one phrase per line): each turn shows the phrase to say, and the log
records the word errors against it. --no-brain stops after the text (no claude call).
--report N prints speech-to-text times and accuracy over the last N logged turns, then exits.
"""
import argparse
import contextlib
import json
import math
import statistics
import sys
import time
from collections.abc import Callable, Iterable, Iterator
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path

from software.skull_core.brain import (DEFAULT_BRAIN_DIR, DEFAULT_CLAUDE, DEFAULT_MODEL, DEFAULT_SYSTEM_PROMPT,
                                       DEFAULT_TIMEOUT_S, FALLBACK_LINE, Brain, BrainConfig, print_now)
from software.skull_core.mic_stream import (DEFAULT_END_DBFS, DEFAULT_HANG_S, DEFAULT_MIC_NAME, DEFAULT_MIN_SPEECH_S,
                                            DEFAULT_SETTLE_S, DEFAULT_START_DBFS, MicError, MicStream, SpeechDetector)
from software.skull_core.skull_bridge import DEFAULT_PARAMS, Bridge
from software.skull_core.state_console import VOICE_DIR
from software.skull_core.status_leds import DEFAULT_BLINK_PERIOD_S
from software.skull_core.stt import (DEFAULT_MAX_S, DEFAULT_MIN_LETTERS, DEFAULT_MODEL_ARCH,
                                     DEFAULT_START_TIMEOUT_S, END_RULES, Heard, Listener, listen_once,
                                     load_stream_maker, word_errors, words)
from software.skull_core.voice_console import HalfDuplexGuard
from software.skull_core.voice_out import (DEFAULT_BANK_DIR, DEFAULT_TTS_PYTHON, DEFAULT_VOICE, PhraseBank,
                                           Speaker, SynthError, WorkerSynth, bank_voice, load_opener, make_player,
                                           normalise, show_clip, worker_argv)
from software.skull_core.voice_state import eye_led, print_watcher, publish_state, state_after_brain, voice_key
from software.skull_core.wake import (DEFAULT_LINE_LOG, DEFAULT_PROBE_HOST, DEFAULT_PROBE_PORT,
                                      DEFAULT_PROBE_TIMEOUT_S, DEFAULT_WAKE_WINDOW_S, IDLE_LISTEN_S, KeyReader,
                                      LineLog, OnlineProbe, is_online, strip_wake, wait_for_trigger)

DEFAULT_LOG = Path.home() / "m3-bench" / "logs" / "m304_turns.jsonl"
# Rule B (2026-10-20, provisional): our detector ends the turn. stt.DEFAULT_END_RULE stays "line".
LOOP_END_RULE = "detector"
ERROR_HOLD_S = 3.0          # a choice, not a measurement: long enough to see the error blink before the exit
TRIGGERS = ("enter", "wake", "both")


@dataclass
class TurnLog:
    """One turn's clock times (time.monotonic), filled in by run_turn as it goes."""
    t_enter: float
    heard: Heard | None = None
    asked: bool = False
    t_first_piece: float | None = None
    t_done: float | None = None
    reply_chars: int = 0
    speech: object = None      # what speak() returned: a voice_out.SpeechLog, or None (printed reply)
    offline: bool = False      # the probe said offline: the brain was never called (M3-07)
    end_state: str = "idle"    # the state published when the turn ended


def run_turn(listen: Callable[[], Heard], ask: Callable[[str], Iterator[str]] | None,
             speak: Callable[[Iterable[str]], object], on_state: Callable[[str], None],
             clock: Callable[[], float] = time.monotonic, *,
             probe: Callable[[], Callable[[], bool]] | None = None,
             say_offline: Callable[[str], object] | None = None,
             end_state: Callable[["TurnLog"], str | None] | None = None) -> TurnLog:
    """One turn of the skull, start to finish. Every part is passed in.

    1. on_state("listening"); the offline probe starts (if given); heard = listen().
       Nothing usable (heard.text is None) -> no brain call.
    2. The probe says offline -> on_state("offline"), say_offline(text), no brain call (M3-07).
    3. on_state("thinking"); the reply pieces from ask(text) go to speak() as they arrive.
       ask=None means speech to text only (--no-brain).
    4. The turn ends with exactly one closing state, published last: "offline" after an offline
       turn, end_state(turn) if it names one (the brain failed: see voice_state.state_after_brain),
       "error" if a part raised (the error still goes up), else "idle". Ctrl+C ends on "idle".
    """
    turn = TurnLog(t_enter=clock())
    try:
        on_state("listening")
        answer = probe() if probe is not None and ask is not None else None
        turn.heard = listen()
        if turn.heard.text is None or ask is None:
            return turn
        if answer is not None and not answer():
            turn.offline = True
            turn.end_state = "offline"
            on_state("offline")
            if say_offline is not None:
                turn.speech = say_offline(turn.heard.text)
            turn.t_done = clock()
            return turn
        on_state("thinking")
        turn.asked = True
        reply = ask(turn.heard.text)
        try:
            turn.speech = speak(_stamped(reply, turn, clock))
        finally:
            close = getattr(reply, "close", None)
            if close is not None:
                close()          # a generator stopped early still runs its cleanup (the brain kills its child)
        turn.t_done = clock()
        if end_state is not None:
            turn.end_state = end_state(turn) or "idle"
        return turn
    except Exception:
        turn.end_state = "error"
        raise
    finally:
        on_state(turn.end_state)


def _stamped(pieces: Iterable[str], turn: TurnLog, clock) -> Iterator[str]:
    """Pass the pieces on unchanged, noting when the first one came and how much text there was."""
    for piece in pieces:
        if turn.t_first_piece is None:
            turn.t_first_piece = clock()
        turn.reply_chars += len(piece)
        yield piece


def _since(t: float | None, origin: float | None) -> float | None:
    return None if t is None or origin is None else round(t - origin, 3)


def turn_record(turn: TurnLog, end_rule: str, expected: str | None = None, brain_record=None,
                trigger: str | None = None, wake_text: str | None = None, wake_heard: Heard | None = None) -> dict:
    """The JSON line for one turn. Times become seconds, relative to Enter or to the end of speech.

    "clock" (M3-08) keeps the raw time.monotonic stamps as well, so times can be compared ACROSS
    turns (did this turn start right after the skull's last reply ended?). wake_heard is the
    utterance that woke the skull, when a wake started the turn.
    """
    heard = turn.heard or Heard(text=None)
    record = {
        "started_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "end_rule": end_rule,
        "end_reason": heard.end_reason,
        "text": heard.text,
        "raw_text": heard.raw_text,
        "lines": heard.lines,
        "stt_s": None if heard.stt_s is None else round(heard.stt_s, 3),
        "first_line_text": heard.first_line_text,
        "first_line_stt_s": None if heard.first_line_stt_s is None else round(heard.first_line_stt_s, 3),
        "detector_end_to_text_s": None if heard.detector_end_to_text_s is None else round(heard.detector_end_to_text_s, 3),
        "moonshine_latency_ms": heard.moonshine_latency_ms,
        "enter_to_first_sample_s": _since(heard.t_first_sample, turn.t_enter),
        "enter_to_speech_end_s": _since(heard.t_speech_end, turn.t_enter),
        "detector_speech_s": _since(heard.detector_end, heard.detector_start),
        "speech_end_to_first_piece_s": _since(turn.t_first_piece, heard.t_speech_end),
        "speech_end_to_done_s": _since(turn.t_done, heard.t_speech_end),
        "blocks_fed": heard.blocks_fed,
        "blocks_drained": heard.blocks_drained,
        "queue_full": heard.queue_full,
        "gaps": heard.gaps,
        "asked": turn.asked,
        "reply_chars": turn.reply_chars,
        "expected": expected,
        "expected_words": None,
        "word_errors": None,
        "first_line_word_errors": None,
        "brain": None if brain_record is None else asdict(brain_record),
        "detector_end_to_first_audio_s": None,
        "speech": None,
        "trigger": trigger,          # "enter" or "wake" (M3-07)
        "wake_text": wake_text,      # the utterance that woke the skull
        "offline": turn.offline,
        "end_state": turn.end_state,
    }
    speech = turn.speech
    if hasattr(speech, "as_record"):
        record["speech"] = speech.as_record(heard.t_speech_end)     # times from the end of speech
        record["detector_end_to_first_audio_s"] = _since(speech.first_audio(), heard.detector_end)
    record["clock"] = clock_stamps(turn, wake_heard)
    if expected is not None:
        record["expected_words"] = len(words(expected))
        record["word_errors"] = word_errors(expected, heard.raw_text)
        record["first_line_word_errors"] = word_errors(expected, heard.first_line_text or "")
    return record


def clock_stamps(turn: TurnLog, wake_heard: Heard | None = None) -> dict:
    """The turn's raw time.monotonic stamps (seconds since boot on Linux, one clock for every process).

    first_sound / last_sound_end: the first clip that started and the last one that ended, the
    opener (Cogitating) included: the span in which the skull's own voice was in the room.
    """
    heard = turn.heard or Heard(text=None)
    clips = list(getattr(turn.speech, "clips", None) or [])
    starts = [c.t_play_start for c in clips if c.t_play_start is not None]
    ends = [c.t_play_end for c in clips if c.t_play_end is not None]
    return {
        "enter": turn.t_enter,
        "wake_start": None if wake_heard is None else wake_heard.detector_start,
        "wake_end": None if wake_heard is None else wake_heard.detector_end,
        "speech_start": heard.detector_start,
        "speech_end": heard.t_speech_end,
        "first_sound": min(starts) if starts else None,
        "last_sound_end": max(ends) if ends else None,
        "done": turn.t_done,
        "logged": time.monotonic(),
    }


def summary_line(record: dict) -> str:
    """One line per turn for the screen, e.g. "heard 'What is…' | end line | stt 0.38 s | …"."""
    def sec(value):
        return "-" if value is None else f"{value:.2f} s"

    parts = [f"heard {record['raw_text']!r}" + ("" if record["text"] else " (nothing usable)")
             + (" OFFLINE" if record.get("offline") else ""),
             f"end {record['end_reason']}", f"stt {sec(record['stt_s'])}",
             f"lines {record['lines']}"]
    if record["lines"] > 1 or record["end_rule"] == "detector":
        parts.append(f"first line {record['first_line_text']!r} {sec(record['first_line_stt_s'])}")
    parts.append(f"detector end->text {sec(record['detector_end_to_text_s'])}")
    parts.append(f"queue_full {record['queue_full']} gaps {record['gaps']}")
    speech = record.get("speech")
    if speech:
        gaps = ", ".join(f"{g:.2f}" for g in speech["gaps_s"]) or "-"
        parts.append(f"first audio {sec(speech['first_audio_s'])} (detector end "
                     f"{sec(record['detector_end_to_first_audio_s'])}) | sentences {speech['sentences']} "
                     f"bank {speech['bank_hits']} | sentence gaps {gaps} s")
    if record["expected"] is not None:
        parts.append(f"word errors {record['word_errors']}/{record['expected_words']} "
                     f"(first line {record['first_line_word_errors']})")
    return " | ".join(parts)


def _median_worst(values: list[float]) -> str:
    if not values:
        return "n=0"
    return f"median {statistics.median(values):.3f} s, worst {max(values):.3f} s, n={len(values)}"


def report(records: list[dict]) -> str:
    """Speech-to-text times and accuracy over some logged turns.

    Times are taken only from turns with text and no lost blocks (queue_full and gaps both 0).
    """
    heard = [r for r in records if r.get("text")]
    clean = [r for r in heard if r.get("queue_full") == 0 and r.get("gaps") == 0]
    silent = [r for r in records if not r.get("text")]
    lines = [f"turns {len(records)}: with text {len(heard)}, nothing usable {len(silent)} "
             f"(brain called on those: {sum(1 for r in silent if r.get('asked'))}), "
             f"lost blocks in {len(heard) - len(clean)}",
             "stt_s (end of speech -> text): " + _median_worst([r["stt_s"] for r in clean if r.get("stt_s") is not None]),
             "first line stt_s: " + _median_worst([r["first_line_stt_s"] for r in clean
                                                   if r.get("first_line_stt_s") is not None]),
             "detector end -> text: " + _median_worst([r["detector_end_to_text_s"] for r in clean
                                                      if r.get("detector_end_to_text_s") is not None])]
    spoken = [r for r in records if r.get("speech")]
    if spoken:
        speech = [r["speech"] for r in spoken]
        lines += [f"spoken turns {len(spoken)}: with a speech error {sum(1 for s in speech if s.get('error'))}, "
                  f"clips with an error {sum(1 for s in speech for c in s['clips'] if c.get('error'))}, "
                  f"bank hits {sum(s['bank_hits'] for s in speech)} of {sum(s['sentences'] for s in speech)} sentences",
                  "speech end -> Cogitating: " + _median_worst([s["opener_start_s"] for s in speech
                                                                 if s.get("opener_start_s") is not None]),
                  "speech end -> first reply audio: " + _median_worst([s["first_audio_s"] for s in speech
                                                                        if s.get("first_audio_s") is not None]),
                  "detector end -> first reply audio: " + _median_worst(
                      [r["detector_end_to_first_audio_s"] for r in spoken
                       if r.get("detector_end_to_first_audio_s") is not None]),
                  "Cogitating end -> first reply audio: " + _median_worst(
                      [s["opener_end_to_first_audio_s"] for s in speech
                       if s.get("opener_end_to_first_audio_s") is not None]),
                  "gap between sentences (all gaps): " + _median_worst([g for s in speech for g in s["gaps_s"]])]
    scored = [r for r in records if r.get("expected") is not None]
    if scored:
        for label, key in (("all lines", "word_errors"), ("first line only", "first_line_word_errors")):
            errors = [r[key] for r in scored]
            total = sum(r["expected_words"] for r in scored)
            lines.append(f"accuracy, {label}: exact {sum(1 for e in errors if e == 0)}/{len(scored)}, "
                         f"at most 1 word wrong {sum(1 for e in errors if e <= 1)}/{len(scored)}, "
                         f"WER {sum(errors)}/{total} = {sum(errors) / total:.3f}" if total else
                         f"accuracy, {label}: no expected words")
    return "\n".join(lines)


def read_phrases(path: Path) -> list[str]:
    return [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def speak_printed(pieces: Iterable[str]) -> None:
    """M3-04's voice: print each piece as it arrives. Kept for --no-speak."""
    for piece in pieces:
        print_now(piece, end="")
    print_now("")


def brain_end_state(brain: Brain) -> Callable[[TurnLog], str | None]:
    """run_turn's end_state for a real brain: "offline" or "error" after a failed brain turn, else None."""
    def end_state(turn: TurnLog) -> str | None:
        record = brain.last
        if not turn.asked or record is None:
            return None
        return state_after_brain(record.kind, record.rate_limit_status)
    return end_state


def fallback_clip_state(brain: Brain) -> Callable[[str], str]:
    """The Speaker's clip_state: the brain's fallback line plays under its failure state, all else "speaking".

    The brain sets record.kind and record.fallback before it yields the fallback line, and the
    Speaker asks here when it makes the clip, so the answer is already known (the backstop path).
    """
    def clip_state(sentence: str) -> str:
        record = brain.last
        if record is None or record.kind is None or not record.fallback:
            return "speaking"
        if normalise(sentence) != normalise(record.fallback):
            return "speaking"
        return state_after_brain(record.kind, record.rate_limit_status) or "speaking"
    return clip_state


def parse_args(argv=None) -> argparse.Namespace:
    int_or_name = lambda value: int(value) if value.isdigit() else value  # noqa: E731
    parser = argparse.ArgumentParser(description="Push-to-talk voice loop: mic stream -> Moonshine -> brain.")
    parser.add_argument("--mic-device", type=int_or_name, default=DEFAULT_MIC_NAME,
                        help="input: a part of the sounddevice name, or an index")
    parser.add_argument("--settle", type=float, default=DEFAULT_SETTLE_S, help="seconds before the mic is used")
    parser.add_argument("--end-rule", choices=END_RULES, default=LOOP_END_RULE,
                        help="what ends a turn: our detector's end (default), or Moonshine's first finished line")
    parser.add_argument("--hang", type=float, default=DEFAULT_HANG_S,
                        help="seconds of quiet that end the speech (our detector)")
    parser.add_argument("--start-timeout", type=float, default=DEFAULT_START_TIMEOUT_S,
                        help="seconds to wait for speech to begin")
    parser.add_argument("--max", type=float, default=DEFAULT_MAX_S, help="longest a turn listens, in seconds")
    parser.add_argument("--min-letters", type=int, default=DEFAULT_MIN_LETTERS,
                        help="text with fewer letters counts as nothing said")
    parser.add_argument("--model-arch", type=int, default=DEFAULT_MODEL_ARCH, help="Moonshine ModelArch number")
    parser.add_argument("--phrases", type=Path, default=None, help="file with one phrase to say per turn")
    parser.add_argument("--no-brain", action="store_true", help="stop after the text: no claude call")
    parser.add_argument("--log", type=Path, default=DEFAULT_LOG, help="JSON-lines turn log (appended)")
    parser.add_argument("--report", type=int, default=None, metavar="N",
                        help="print times and accuracy of the last N logged turns, then exit")
    parser.add_argument("--brain-dir", type=Path, default=DEFAULT_BRAIN_DIR)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--timeout", type=float, default=DEFAULT_TIMEOUT_S)
    parser.add_argument("--system-prompt-file", type=Path, default=None)
    parser.add_argument("--no-thinking", action="store_true")
    parser.add_argument("--claude", default=DEFAULT_CLAUDE)
    parser.add_argument("--no-speak", action="store_true", help="print the reply instead of speaking it")
    parser.add_argument("--tts-python", type=Path, default=DEFAULT_TTS_PYTHON,
                        help="python of the venv with pocket-tts (the TTS worker runs there)")
    parser.add_argument("--voice", default=DEFAULT_VOICE, help="Pocket TTS voice")
    parser.add_argument("--bank", type=Path, default=DEFAULT_BANK_DIR, help="phrase bank folder")
    parser.add_argument("--device", type=int_or_name, default="MAX98357A", help="output: sounddevice name or index")
    parser.add_argument("--voice-dir", type=Path, default=VOICE_DIR, help="folder with thinking.wav (Cogitating)")
    # M3-06: the voice state on the bridge, the eye LED follows it.
    parser.add_argument("--params", type=Path, default=DEFAULT_PARAMS, help="robot_params.yaml (the key prefix)")
    parser.add_argument("--pins", nargs=3, type=int, default=[5, 6, 13], metavar=("R", "G", "B"),
                        help="eye LED pins, BCM")
    parser.add_argument("--common-anode", action="store_true")
    parser.add_argument("--blink-period", type=float, default=DEFAULT_BLINK_PERIOD_S)
    parser.add_argument("--no-led", action="store_true", help="no GPIO: the state is published, nothing lights")
    parser.add_argument("--watch-state", action="store_true", help="print every published state with its time")
    # M3-07: the wake phrase and the offline path.
    parser.add_argument("--trigger", choices=TRIGGERS, default=None,
                        help="what starts a turn: Enter, the wake phrase, or both (default: both; enter with --phrases)")
    parser.add_argument("--wake-window", type=float, default=DEFAULT_WAKE_WINDOW_S,
                        help="seconds to wait for the question after a bare wake")
    parser.add_argument("--line-log", type=Path, default=DEFAULT_LINE_LOG,
                        help="JSON-lines log of every utterance heard while idle (wake mode)")
    parser.add_argument("--probe-host", default=DEFAULT_PROBE_HOST, help="host the offline probe connects to")
    parser.add_argument("--probe-port", type=int, default=DEFAULT_PROBE_PORT)
    parser.add_argument("--probe-timeout", type=float, default=DEFAULT_PROBE_TIMEOUT_S)
    parser.add_argument("--no-probe", action="store_true", help="no probe: only the brain's failure says offline")
    args = parser.parse_args(argv)

    def positive(value):
        return math.isfinite(value) and value > 0

    if not (math.isfinite(args.settle) and args.settle >= 0):
        parser.error("--settle must be >= 0")
    if not (math.isfinite(args.hang) and args.hang >= 0):
        parser.error("--hang must be >= 0")
    if not positive(args.start_timeout) or not positive(args.max) or args.start_timeout > args.max:
        parser.error("--start-timeout and --max must be greater than 0, and --start-timeout no more than --max")
    if args.min_letters < 0:
        parser.error("--min-letters must be >= 0")
    if not positive(args.timeout):
        parser.error("--timeout must be greater than 0")
    if args.report is not None and args.report < 1:
        parser.error("--report needs N >= 1")
    if not positive(args.blink_period):
        parser.error("--blink-period must be greater than 0")
    if args.trigger is None:
        args.trigger = "enter" if args.phrases is not None else "both"
    elif args.phrases is not None and args.trigger != "enter":
        parser.error("--phrases works with --trigger enter only")
    if not positive(args.wake_window) or args.wake_window > args.max:
        parser.error("--wake-window must be greater than 0 and no more than --max")
    if not positive(args.probe_timeout):
        parser.error("--probe-timeout must be greater than 0")
    if not 0 < args.probe_port < 65536:
        parser.error("--probe-port must be 1..65535")
    return args


def main(argv=None) -> int:
    """Parse, then run the loop with the voice state published on the bridge and the eye LED following it.

    Any exit code other than 0, and any unexpected exception, first shows "error" on the LED for
    ERROR_HOLD_S, so a failure on the bench is visible and not just a dark eye.
    """
    args = parse_args(argv)
    log_path = args.log.expanduser()
    if args.report is not None:
        try:
            rows = [json.loads(line) for line in log_path.read_text(encoding="utf-8").splitlines() if line.strip()]
        except (OSError, ValueError) as exc:
            print(f"Could not read {log_path}: {exc}", file=sys.stderr)
            return 1
        print_now(report(rows[-args.report:]))
        return 0

    bridge = Bridge()
    try:
        key = voice_key(args.params)
    except ValueError as exc:
        print(exc, file=sys.stderr)
        return 1
    if not args.no_led:
        try:
            import gpiozero  # noqa: F401  (checked here so a missing package is one clear line, not a traceback)
        except ImportError:
            print("gpiozero is not importable in this Python: install it in this venv, or pass --no-led.",
                  file=sys.stderr)
            return 1
    if args.watch_state:
        print_watcher(bridge, key)
    on_state = publish_state(bridge, key)
    on_state("thinking")           # start-up: the models are loading (M3-06 decision 2)
    led = (contextlib.nullcontext() if args.no_led else
           eye_led(bridge, key, args.pins, args.common_anode, args.blink_period))
    with led:
        try:
            code = run_loop(args, log_path, on_state)
        except Exception:
            on_state("error")
            time.sleep(ERROR_HOLD_S)
            raise
        if code != 0:
            on_state("error")
            time.sleep(ERROR_HOLD_S)
        return code


def run_loop(args: argparse.Namespace, log_path: Path, on_state: Callable[[str], None]) -> int:
    """Everything after the bridge is up: load the parts, open the mic, then turn after turn until quit."""
    phrases = None
    if args.phrases is not None:
        try:
            phrases = read_phrases(args.phrases)
        except OSError as exc:
            print(f"Could not read {args.phrases}: {exc}", file=sys.stderr)
            return 1
    system_prompt = DEFAULT_SYSTEM_PROMPT
    if args.system_prompt_file is not None:
        try:
            system_prompt = args.system_prompt_file.read_text(encoding="utf-8").strip()
        except OSError as exc:
            print(f"Could not read {args.system_prompt_file}: {exc}", file=sys.stderr)
            return 1

    try:
        make_stream = load_stream_maker(args.model_arch)
    except ImportError as exc:
        print(exc, file=sys.stderr)
        return 1

    from software.skull_core.record import SAMPLE_RATE   # imports sounddevice: only here, never in the tests

    brain = None
    if not args.no_brain:
        config = BrainConfig(model=args.model, brain_dir=args.brain_dir.expanduser(), timeout_s=args.timeout,
                             system_prompt=system_prompt, claude=args.claude, thinking_off=args.no_thinking)
        brain = Brain(config, log_path=None)   # its record goes into this loop's own log line

    guard = HalfDuplexGuard()      # one guard: the mic checks it, the speaker shuts it while it plays
    mic = MicStream(args.mic_device, guard, SAMPLE_RATE, settle_s=args.settle)
    listener = Listener(make_stream, SAMPLE_RATE)
    detector = SpeechDetector(DEFAULT_START_DBFS, DEFAULT_END_DBFS, args.hang, DEFAULT_MIN_SPEECH_S)

    def listen() -> Heard:
        return listen_once(mic, listener, detector, end_rule=args.end_rule, start_timeout_s=args.start_timeout,
                           max_s=args.max, min_letters=args.min_letters)

    def listen_idle(interrupt) -> Heard:
        """Wake mode, while idle: one long Moonshine stream; a fresh one after IDLE_LISTEN_S of silence."""
        return listen_once(mic, listener, detector, end_rule=args.end_rule, start_timeout_s=IDLE_LISTEN_S,
                           max_s=IDLE_LISTEN_S + args.max, min_letters=args.min_letters, interrupt=interrupt)

    def listen_after_bare_wake() -> Heard:
        heard = listen_once(mic, listener, detector, end_rule=args.end_rule, start_timeout_s=args.wake_window,
                            max_s=args.max, min_letters=args.min_letters)
        return strip_wake(heard, args.min_letters)

    speak, synth, speaker = speak_printed, None, None
    if brain is not None and not args.no_speak:
        try:
            bank = PhraseBank(args.bank.expanduser(), bank_voice(args.voice))
        except ValueError as exc:
            print(exc, file=sys.stderr)
            return 1
        synth = WorkerSynth(worker_argv(args.tts_python.expanduser(), args.voice))
        try:
            synth.start()          # the model loads (about 14 s on the Pi) while the mic settles
        except SynthError as exc:
            print(exc, file=sys.stderr)
            return 1
        speaker = Speaker(synth, bank, guard, make_player(args.device), on_state=on_state,
                          opener=load_opener(args.voice_dir), show=show_clip, clip_state=fallback_clip_state(brain))
        speak = speaker.speak

    def say_offline(_text: str):
        """Offline: the canned line from the bank, LED "offline", no Cogitating (PM decision 6).

        TODO (M3-07, decision 6, left on purpose): the six Vosk state commands offline (D3). Not wired
        because (1) listen_once keeps no audio, so the question's samples would have to be cut out of
        MicStream.ring with boundaries we don't track yet; (2) a second model (Vosk, ~40 MB) loads next
        to Moonshine and Pocket TTS while G9 (memory) is still open; (3) how long a commanded state
        holds before the loop shows idle/offline again is undecided (it shares the voice state key).
        """
        if speaker is None:
            print_now(FALLBACK_LINE)
            return None
        return speaker.say_fixed(FALLBACK_LINE, state="offline")

    probe = None
    if brain is not None and not args.no_probe:
        probe = OnlineProbe(lambda: is_online(args.probe_host, args.probe_port, args.probe_timeout)).start
    end_state = brain_end_state(brain) if brain is not None else None
    ask = None if brain is None else brain.ask

    try:
        mic.open()
    except MicError as exc:
        print(exc, file=sys.stderr)
        if synth is not None:
            synth.close()
        return 1
    try:
        if synth is not None:
            print_now("Loading the TTS worker ...")
            info = synth.wait_ready()
            print_now(f"TTS worker ready: {info.get('engine')}, voice {info.get('voice')}, {info.get('rate')} Hz, "
                      f"loaded in {info.get('load_s')} s. Bank: {args.bank.expanduser()}")
        print_now(f"Mic {args.mic_device!r} open at {SAMPLE_RATE} Hz. Settling for {args.settle:g} s. "
                  f"End rule: {args.end_rule}, hang {args.hang:g} s. Brain: {'off' if brain is None else args.model}. "
                  f"Voice: {'printed' if synth is None else args.voice}. Trigger: {args.trigger}. "
                  f"Probe: {'off' if probe is None else f'{args.probe_host}:{args.probe_port}'}. Log: {log_path}")
        while not mic.ready():
            time.sleep(0.05)
        on_state("idle")

        keys = KeyReader(sys.stdin).start() if args.trigger in ("enter", "both") else None
        wake_on = args.trigger in ("wake", "both")
        line_log = LineLog(args.line_log.expanduser() if wake_on else None,
                           report=lambda text: print(text, file=sys.stderr, flush=True))
        if wake_on:
            print_now(f"Wake lines log: {args.line_log.expanduser()}")
        count = len(phrases) if phrases is not None else None
        turn_number = 0
        while count is None or turn_number < count:
            if keys is not None:
                if keys.drop_stale():
                    break
                if keys.closed:
                    if not wake_on:
                        break
                    keys = None    # no more input (e.g. under nohup): the wake phrase only
            expected = phrases[turn_number] if phrases is not None else None
            if expected is not None:
                prompt = f"\nPhrase {turn_number + 1}/{count}: say \"{expected}\". Enter, then speak (q quits): "
            elif keys is None:
                prompt = "\nSay \"Awaken machine spirit\" (Ctrl+C quits)\n"
            elif wake_on:
                prompt = "\nSay \"Awaken machine spirit\", or press Enter, then speak (q + Enter quits)\n"
            else:
                prompt = "\nEnter, then speak (q quits): "
            print_now(prompt, end="")
            trigger = wait_for_trigger(listen_idle if wake_on else None, keys,
                                       lambda: [line.text for line in listener.lines], line_log, args.min_letters,
                                       out=print_now)
            if trigger.kind == "quit":
                break
            if trigger.kind == "enter":
                turn_listen = listen
            elif trigger.question is not None:
                turn_listen = (lambda question=trigger.question: question)     # one breath: already heard
            else:
                print_now(f"  [awake: listening {args.wake_window:g} s for the question]")
                turn_listen = listen_after_bare_wake
            turn = run_turn(turn_listen, ask, speak, on_state, probe=probe, say_offline=say_offline,
                            end_state=end_state)
            record = turn_record(turn, args.end_rule, expected, brain.last if (brain and turn.asked) else None,
                                 trigger=trigger.kind, wake_text=trigger.heard.raw_text if trigger.heard else None,
                                 wake_heard=trigger.heard)
            print_now(summary_line(record))
            try:
                log_path.parent.mkdir(parents=True, exist_ok=True)
                with open(log_path, "a", encoding="utf-8") as log:
                    log.write(json.dumps(record) + "\n")
            except OSError as exc:
                print(f"Could not write the turn log {log_path}: {exc}", file=sys.stderr, flush=True)
            turn_number += 1
    except SynthError as exc:
        print(exc, file=sys.stderr, flush=True)
        return 1
    except KeyboardInterrupt:
        print_now("")
    finally:
        mic.close()
        listener.close()
        if synth is not None:
            synth.close()
        stats = mic.stats()
        print_now(f"Mic: seen {stats.blocks_seen} queued {stats.blocks_queued} queue_full {stats.queue_full} "
                  f"overflows {stats.input_overflows} (queue_full also counts blocks dropped between turns)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
