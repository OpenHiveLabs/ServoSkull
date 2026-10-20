"""Wake phrase and offline check for the voice loop (M3-07).

Wake: Moonshine listens all the time. Every finished utterance heard while the skull is idle is
checked by split_wake(): does it START with "Awaken machine spirit"? If yes, a turn begins. The
question can come in the same breath ("Awaken machine spirit. What time is it?"), or the skull
waits a few seconds for it after a bare wake. Every utterance heard while idle goes to a JSON-lines
log with whether it woke the skull, so the rule can be widened later from real misheard texts.

Offline: before the brain is called, a short TCP connect to the brain's host says whether the
network is there (OnlineProbe). It runs in the background while the question is being spoken, so
on a healthy network it costs nothing extra. The brain's own failure stays the backstop.

Nothing here imports Moonshine, sounddevice or GPIO: the tests run on the Surface with fakes.
"""
import json
import queue
import re
import socket
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass, replace
from datetime import datetime
from pathlib import Path

from software.skull_core.stt import Heard, usable_text

# ADR-013: the loosened prefix rule. The line must START with it (re.match, not re.search), so a
# sentence that only contains the phrase in the middle never wakes the skull. Measured 5/5 on
# files on the Pi (2026-10-07); live 2 of 4 (2026-10-08, Ersin's report).
WAKE_PATTERN = r"^\W*awaken(ed)?\W+(the\W+)?machines?\b"
BARE_WAKE_MAX_WORDS = 3        # step note proposal: at most this many words after "machine" = no question
CUT_MARKS = ".,!?;:"           # a question starts after the first of these that has text after it
DEFAULT_WAKE_WINDOW_S = 6.0    # PM decision 4 (estimate): after a bare wake, wait this long for the question
IDLE_LISTEN_S = 30.0           # how long one idle Moonshine stream waits for speech before a fresh one starts
DEFAULT_LINE_LOG = Path.home() / "m3-bench" / "logs" / "m307_wake_lines.jsonl"

# Offline probe target: the host the claude CLI sends its requests to (Claude Code docs, network
# access requirements, checked 2026-10-20: api.anthropic.com "Claude API requests"). Port 443 = HTTPS.
DEFAULT_PROBE_HOST = "api.anthropic.com"
DEFAULT_PROBE_PORT = 443
# 2.0 s: Linux resends a lost connection request (SYN) after 1 s, so 2 s survives one lost packet on
# a slow link instead of calling it offline. A healthy connect takes one round trip (not measured on
# the Pi). The full 2 s is only paid when the network is gone in a way that doesn't fail at once.
DEFAULT_PROBE_TIMEOUT_S = 2.0


# --- the wake rule -------------------------------------------------------------------------------

def marked_text(line_texts: list[str]) -> str:
    """Join Moonshine's lines so every line break becomes a cut mark.

    "Awaken machine spirit" + "What time is it?" -> "Awaken machine spirit. What time is it?".
    Without this, a first line with no full stop would glue the wake to the question.
    """
    parts = []
    for text in line_texts:
        text = text.strip()
        if not text:
            continue
        parts.append(text if text[-1] in CUT_MARKS else text + ".")
    return " ".join(parts)


def split_wake(text: str, pattern: str = WAKE_PATTERN) -> tuple[bool, str]:
    """(woke, question). woke: the line starts with the wake phrase. question: "" for a bare wake.

    What follows "machine" starts with "spirit", or with whatever it was heard as ("pirate",
    "a pirate", "Sir Pirate"), so the question is cut out like this (step note, design point 2):
      1. a cut mark after the first word of what follows, with text after it -> the question is
         the text after that mark. The mark must come after a word, so "Awaken machine, a pirate."
         is not cut at the comma (that would send "a pirate." to the brain);
      2. else at most BARE_WAKE_MAX_WORDS words left -> a bare wake;
      3. else everything after "machine" is the question ("spirit what time is it").
    """
    # IGNORECASE on the text itself, not text.lower(): lower() can change the length of Turkish
    # text ("İ" becomes two characters), and the match's end is used to cut the original text.
    match = re.match(pattern, text, re.IGNORECASE)
    if match is None:
        return False, ""
    tail = text[match.end():]
    first_word = re.search(r"[\w']+", tail)
    if first_word is None:
        return True, ""
    cut = re.compile(f"[{re.escape(CUT_MARKS)}]+\\s*(?=\\w)").search(tail, first_word.end())
    if cut is not None:
        return True, tail[cut.end():].strip()
    words = re.findall(r"[\w']+", tail)
    if len(words) <= BARE_WAKE_MAX_WORDS:
        return True, ""
    return True, tail.strip(" ,.;:")


def strip_wake(heard: Heard, min_letters: int) -> Heard:
    """The window's Heard, with a repeated wake phrase removed (people say it again when unsure)."""
    if not heard.raw_text:
        return heard
    woke, question = split_wake(heard.raw_text)
    if not woke:
        return heard
    return replace(heard, text=usable_text(question, min_letters))


# --- the line log ---------------------------------------------------------------------------------

class LineLog:
    """Appends one JSON line per utterance heard while idle. A write error is reported, never raised."""

    def __init__(self, path: Path | None, report: Callable[[str], None] = print):
        self.path = path
        self.report = report
        self.rows = 0

    def write(self, row: dict) -> None:
        self.rows += 1
        if self.path is None:
            return
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.path, "a", encoding="utf-8") as log:
                log.write(json.dumps(row) + "\n")
        except OSError as exc:
            self.report(f"Could not write the wake line log {self.path}: {exc}")


# --- waiting for a trigger -----------------------------------------------------------------------

@dataclass
class Trigger:
    """What started a turn. kind: "enter", "wake" or "quit".

    For a wake: heard is the waking utterance; question is a Heard holding the question when it came
    in the same breath, or None for a bare wake (the loop then listens for wake_window_s).
    """
    kind: str
    heard: Heard | None = None
    question: Heard | None = None


class KeyReader:
    """Reads stdin lines on a daemon thread, so the mic loop can watch for Enter without blocking.

    Each line goes on a queue; end of input puts None (then .closed is True).
    """

    def __init__(self, stream):
        self.stream = stream
        self.lines: queue.Queue = queue.Queue()
        self.closed = False
        self._thread = threading.Thread(target=self._read, name="keys", daemon=True)

    def start(self) -> "KeyReader":
        self._thread.start()
        return self

    def _read(self) -> None:
        try:
            for line in self.stream:
                self.lines.put(line.rstrip("\n"))
        except (OSError, ValueError):
            pass
        self.lines.put(None)

    def pending(self) -> bool:
        return not self.lines.empty()

    def get(self, timeout: float | None = None) -> str | None:
        """The next line, or None at end of input. Raises queue.Empty after timeout."""
        line = self.lines.get(timeout=timeout)
        if line is None:
            self.closed = True
        return line

    def drop_stale(self) -> bool:
        """Forget the lines typed during the last turn. True if one of them was "q" (quit).

        End of input is not a quit here: it sets .closed, and the caller decides.
        """
        quit_seen = False
        while True:
            try:
                line = self.lines.get_nowait()
            except queue.Empty:
                return quit_seen
            if line is None:
                self.closed = True
            elif line.strip().lower() == "q":
                quit_seen = True


def wait_for_trigger(listen_idle: Callable[[Callable[[], bool] | None], Heard] | None, keys: KeyReader | None,
                     line_texts: Callable[[], list[str]], line_log: LineLog, min_letters: int,
                     clock: Callable[[], float] = time.time, out: Callable[[str], None] = print) -> Trigger:
    """Block until Enter (keys), the wake phrase (listen_idle), or quit. Ctrl+C goes up to the caller.

    listen_idle(interrupt) is listen_once with a long start timeout; interrupt() is True when a key
    line is waiting, so a key press is noticed within one block read. line_texts() gives the texts
    of the lines Moonshine finished in the last call. Every utterance with text is logged.
    """
    if listen_idle is None and keys is None:
        raise ValueError("need at least one trigger")
    no_audio_reported = False
    while True:
        if keys is not None and (keys.pending() or listen_idle is None):
            line = keys.get()
            if line is None:
                if listen_idle is None:
                    return Trigger("quit")
                keys = None          # input ended (e.g. run under nohup): carry on with the wake only
                continue
            return Trigger("quit" if line.strip().lower() == "q" else "enter")
        heard = listen_idle(keys.pending if keys is not None else None)
        if heard.end_reason == "no_audio" and not no_audio_reported:
            out("No audio from the mic while waiting for the wake phrase (unplugged?). Still trying.")
            no_audio_reported = True
        if not heard.raw_text:
            continue
        no_audio_reported = False
        texts = line_texts()
        woke, question = split_wake(marked_text(texts) if texts else heard.raw_text)
        line_log.write({
            "at": datetime.fromtimestamp(clock()).astimezone().isoformat(timespec="seconds"),
            "text": heard.raw_text, "lines": texts, "woke": woke, "question": question if woke else None,
            "end_reason": heard.end_reason, "stt_s": None if heard.stt_s is None else round(heard.stt_s, 3),
            "detector_end_to_text_s": (None if heard.detector_end_to_text_s is None
                                       else round(heard.detector_end_to_text_s, 3)),
            "queue_full": heard.queue_full, "gaps": heard.gaps,
            # M3-08: time.monotonic stamps, the same clock as the turn log's "clock" (self-trigger check)
            "t_start": heard.detector_start, "t_end": heard.detector_end})
        out(f"  [{'WAKE' if woke else 'idle'}] {heard.raw_text!r}" + (f" -> question {question!r}" if question else ""))
        if woke:
            asked = replace(heard, text=usable_text(question, min_letters)) if question else None
            if asked is not None and asked.text is None:
                asked = None          # "Awaken machine spirit. Uh." is a bare wake
            return Trigger("wake", heard=heard, question=asked)


# --- the offline probe ---------------------------------------------------------------------------

def is_online(host: str = DEFAULT_PROBE_HOST, port: int = DEFAULT_PROBE_PORT,
              timeout_s: float = DEFAULT_PROBE_TIMEOUT_S, connect=socket.create_connection) -> bool:
    """True if a TCP connection to host:port opens within timeout_s. Never raises, never waits longer.

    connect is socket.create_connection (a fake in the tests). Its own timeout does not cover the
    name lookup (DNS), which can hang for seconds with the WiFi up but no internet, so the connect
    runs on a daemon thread and we stop waiting for it at timeout_s. A thread still stuck in DNS is
    left to finish on its own; it holds nothing we need.
    """
    result: list[bool] = []

    def attempt() -> None:
        try:
            sock = connect((host, port), timeout_s)
        except (OSError, ValueError):
            result.append(False)
            return
        try:
            sock.close()
        except OSError:
            pass
        result.append(True)

    worker = threading.Thread(target=attempt, name="online-probe", daemon=True)
    worker.start()
    worker.join(timeout_s)
    return bool(result) and result[0]


class OnlineProbe:
    """One probe per turn: start() when the turn starts listening, answer() when the text is ready.

    Started early, the probe runs while the person is still speaking, so it is usually finished
    by the time the brain would be called.
    """

    def __init__(self, check: Callable[[], bool] = is_online):
        self.check = check

    def start(self) -> Callable[[], bool]:
        """Begin the check in the background. Returns answer(): blocks until it is done, then True/False."""
        result: list[bool] = []
        done = threading.Event()

        def run() -> None:
            try:
                result.append(bool(self.check()))
            except Exception:       # a broken check must not end the turn: treat it as offline
                result.append(False)
            finally:
                done.set()

        threading.Thread(target=run, name="online-probe-turn", daemon=True).start()

        def answer() -> bool:
            done.wait()             # is_online itself never takes longer than its timeout
            return result[0]
        return answer
