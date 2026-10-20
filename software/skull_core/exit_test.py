"""M3-08 exit test: run the voice loop once with a written header and memory/temperature samples,
then analyse the run: ten turns in a row, the latency, self-triggers, G9, and the WiFi-off checks.

Why a wrapper and not a second script beside the loop: the header (commit, trigger, cooling, the
question list) is written by the same command that starts the loop, the samples share the loop's
clock (time.monotonic), and the analysis reads one folder per run, so old turns never mix in.

One folder per run (it must not exist yet, or be empty):
    header.json      what was run and how: commit, claude --version, model, trigger, cooling, the questions
    turns.jsonl      the loop's turn log (voice_loop.turn_record, with its "clock" stamps)
    wake_lines.jsonl every utterance heard while idle (wake mode only)
    samples.jsonl    MemAvailable and the loop's process-tree RSS every second, temperature every 10 s
    footer.json      how the loop ended (exit code, exception, Ctrl+C), wall time, CPU time

Usage (from the repo root; the arguments after -- go to voice_loop unchanged):
    python -m software.skull_core.exit_test run --run-dir ~/m3-bench/logs/m308_run1 --trigger wake
           --cooling "heatsink and fan" [--room-temp "22 C (estimate)"] [--amp-volume ...]
           [--mic-distance "30 cm"] [--people "nobody else"] [--note ...] [--questions FILE]
           -- --mic-device UACDemoV1.0 --tts-python ~/m3-bench/.venv-tts/bin/python ...
    python -m software.skull_core.exit_test report --run-dir ~/m3-bench/logs/m308_run1 [--false-wake 4]
    python -m software.skull_core.exit_test report --run-dir ~/m3-bench/logs/m308_wifi --wifi

Nothing here imports numpy, Moonshine, sounddevice or GPIO at the top: the tests run on the
Surface, and the memory baseline is read before the loop's heavy imports.
"""
import argparse
import json
import os
import re
import socket
import statistics
import subprocess
import sys
import threading
import time
import traceback
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

from software.skull_core.stt import word_errors, words

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_QUESTIONS = Path(__file__).parent / "voice" / "m3_exit_questions.txt"
REQUIRED_TURNS = 10                 # M3 exit criterion: ten turns in a row from one start
SILENT_MINUTE_S = 60.0              # step note, protocol step 5: one silent minute after turn 10
G9_LIMIT_KB = 4_141_875             # 50 % of 7.9 GiB (computed 2026-10-12, M3-02 "G4 and G9")
MEM_PERIOD_S = 1.0                  # /proc reads are cheap (well under a millisecond each, not measured on the Pi)
TEMP_PERIOD_S = 10.0                # as M3-00 (e): vcgencmd starts a process, so not every second

# Self-trigger rule (see find_self_triggers). The guard keeps the mic shut while the skull plays and
# for DEFAULT_TAIL_S = 0.3 s after. On the Pi (2026-10-20, M3-02 check 7) the level after the tail
# stayed raised (-61 to -65 dBFS) for about 1 s. 0.3 + 1.0 + 0.2 margin = 1.5 s. The protocol has
# Ersin wait for "idle" and then three seconds, so his own speech starts well after this window.
SELF_TRIGGER_WINDOW_S = 1.5
ECHO_WORDS = 4                      # a run of this many words from the skull's own reply = its echo

RUN_FILES = ("header.json", "turns.jsonl", "wake_lines.jsonl", "samples.jsonl", "footer.json")
LOOP_FLAGS_SET_HERE = ("--log", "--line-log", "--trigger")


# --- reading /proc and the Pi's sensors ------------------------------------------------------------

def parse_meminfo(text: str) -> dict[str, int]:
    """/proc/meminfo as {name: kB}. Lines look like "MemAvailable:    5123456 kB"."""
    values = {}
    for line in text.splitlines():
        name, _, rest = line.partition(":")
        parts = rest.split()
        if parts and parts[0].isdigit():
            values[name.strip()] = int(parts[0])
    return values


def mem_available_kb(meminfo: Path = Path("/proc/meminfo")) -> int | None:
    """MemAvailable in kB: the kernel's estimate of what new programs can use without swapping."""
    try:
        return parse_meminfo(meminfo.read_text()).get("MemAvailable")
    except OSError:
        return None


def _parent_pid(stat_text: str) -> int:
    """The parent pid from /proc/<pid>/stat. The name (field 2) may hold spaces and ")", so split after the last ")"."""
    return int(stat_text.rsplit(")", 1)[1].split()[1])


def process_tree_rss_kb(root_pid: int, proc: Path = Path("/proc")) -> tuple[int, int]:
    """(sum of VmRSS in kB, number of processes) for root_pid and all its descendants, right now.

    Here that is the loop's own process (Moonshine, the mic), the TTS worker and any running
    `claude`. Shared libraries are counted once per process, so the sum is an upper bound, the same
    kind of number as G9's "maxRSS together". A process that exits while we read is skipped.
    """
    children: dict[int, list[int]] = {}
    for entry in proc.iterdir():
        if not entry.name.isdigit():
            continue
        try:
            parent = _parent_pid((entry / "stat").read_text())
        except (OSError, IndexError, ValueError):
            continue
        children.setdefault(parent, []).append(int(entry.name))
    total, count, todo = 0, 0, [root_pid]
    while todo:
        pid = todo.pop()
        todo.extend(children.get(pid, []))
        try:
            status = parse_meminfo((proc / str(pid) / "status").read_text())
        except OSError:
            continue
        total += status.get("VmRSS", 0)      # kernel threads have no VmRSS
        count += 1
    return total, count


def read_temperature(run: Callable = subprocess.run,
                     thermal: Path = Path("/sys/class/thermal/thermal_zone0/temp")) -> dict:
    """{"temp_c", "throttled", "source"}. Never raises: missing tools give None.

    The Pi: `vcgencmd measure_temp` ("temp=47.2'C") and `get_throttled` ("throttled=0x0").
    Elsewhere (the Surface): no vcgencmd, so the kernel's thermal zone 0 (millidegrees), no throttle flag.
    """
    def vcgencmd(command: str) -> str | None:
        try:
            done = run(["vcgencmd", command], capture_output=True, text=True, timeout=2)
        except (OSError, subprocess.SubprocessError):
            return None
        return done.stdout.strip() if done.returncode == 0 else None

    temp = vcgencmd("measure_temp")
    match = re.search(r"temp=([\d.]+)", temp or "")
    if match:
        throttled = vcgencmd("get_throttled")
        return {"temp_c": float(match.group(1)), "source": "vcgencmd",
                "throttled": throttled.partition("=")[2] if throttled else None}
    try:
        return {"temp_c": int(thermal.read_text().strip()) / 1000, "throttled": None, "source": "thermal_zone0"}
    except (OSError, ValueError):
        return {"temp_c": None, "throttled": None, "source": None}


class Sampler:
    """Writes one JSON line every period_s on a daemon thread: memory always, temperature every temp_period_s.

    A write error is reported once and sampling goes on in memory, so a full disk never stops the loop.
    """

    def __init__(self, path: Path, root_pid: int, period_s: float = MEM_PERIOD_S,
                 temp_period_s: float = TEMP_PERIOD_S, clock=time.monotonic,
                 mem=mem_available_kb, tree=process_tree_rss_kb, temperature=read_temperature,
                 report: Callable[[str], None] = print):
        self.path, self.root_pid = path, root_pid
        self.period_s, self.temp_period_s = period_s, temp_period_s
        self.clock, self.mem, self.tree, self.temperature, self.report = clock, mem, tree, temperature, report
        self.rows = 0
        self._last_temp: float | None = None
        self._write_failed = False
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, name="exit-test-sampler", daemon=True)

    def sample(self) -> dict:
        now = self.clock()
        rss, procs = self.tree(self.root_pid)
        row = {"t": now, "at": datetime.now().astimezone().isoformat(timespec="seconds"),
               "mem_available_kb": self.mem(), "tree_rss_kb": rss, "tree_procs": procs}
        if self._last_temp is None or now - self._last_temp >= self.temp_period_s:
            self._last_temp = now
            row.update(self.temperature())
        return row

    def write(self, row: dict) -> None:
        self.rows += 1
        try:
            with open(self.path, "a", encoding="utf-8") as out:
                out.write(json.dumps(row) + "\n")
        except OSError as exc:
            if not self._write_failed:
                self._write_failed = True
                self.report(f"Could not write the samples {self.path}: {exc}")

    def _run(self) -> None:
        while True:
            try:
                self.write(self.sample())
            except Exception as exc:          # a sampling bug must never end the test run
                self.report(f"Sampler error (sampling goes on): {exc!r}")
            if self._stop.wait(self.period_s):
                return

    def start(self) -> "Sampler":
        self._thread.start()
        return self

    def stop(self) -> None:
        self._stop.set()
        self._thread.join(timeout=5)


# --- the run's files -------------------------------------------------------------------------------

def read_questions(path: Path) -> list[str]:
    """One question per line; blank lines and lines starting with # are skipped."""
    lines = Path(path).read_text(encoding="utf-8").splitlines()
    return [line.strip() for line in lines if line.strip() and not line.strip().startswith("#")]


def _read_jsonl(path: Path) -> list[dict]:
    """Every JSON line of a file; a missing file is empty; a broken line (a kill mid-write) is skipped."""
    if not path.exists():
        return []
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        try:
            rows.append(json.loads(line))
        except ValueError:
            continue
    return rows


def _read_json(path: Path) -> dict:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}


@dataclass
class Run:
    header: dict
    turns: list[dict]
    lines: list[dict]
    samples: list[dict]
    footer: dict


def load_run(run_dir: Path) -> Run:
    return Run(header=_read_json(run_dir / "header.json"), turns=_read_jsonl(run_dir / "turns.jsonl"),
               lines=_read_jsonl(run_dir / "wake_lines.jsonl"), samples=_read_jsonl(run_dir / "samples.jsonl"),
               footer=_read_json(run_dir / "footer.json"))


# --- the analysis ------------------------------------------------------------------------------------

def _clock(record: dict, name: str) -> float | None:
    return (record.get("clock") or {}).get(name)


def reply_text(record: dict) -> str:
    """Everything the skull said in one turn (Cogitating included), from the speech log's clips."""
    clips = (record.get("speech") or {}).get("clips") or []
    return " ".join(clip.get("text", "") for clip in clips)


def echo_run(heard: str, said: str, n: int = ECHO_WORDS) -> str | None:
    """The first run of n consecutive words of `heard` that also appears, in order, in `said`; else None."""
    heard_words, said_words = words(heard or ""), words(said or "")
    said_runs = {tuple(said_words[i:i + n]) for i in range(len(said_words) - n + 1)}
    for i in range(len(heard_words) - n + 1):
        if tuple(heard_words[i:i + n]) in said_runs:
            return " ".join(heard_words[i:i + n])
    return None


def self_trigger_reasons(start: float | None, text: str, previous: dict | None, questions: list[str],
                         window_s: float = SELF_TRIGGER_WINDOW_S) -> list[str]:
    """Why an utterance looks like the skull hearing itself. Empty list = it doesn't.

    The rule, two independent tests (either one is enough):
    1. timing: the utterance started between the first sound of the previous reply and window_s after
       its last sound. The guard keeps the mic shut over that span plus 0.3 s, so speech found
       there got past the guard (echo, a late tail), and a person following the protocol waits
       3 s after "idle". start is the detector's start of speech on time.monotonic.
    2. text: the words heard contain ECHO_WORDS consecutive words of the previous reply, and that
       run is not part of any question on the list (a reply that repeats the question must not
       blame the person who asked it).
    """
    if previous is None:
        return []
    reasons = []
    first, last = _clock(previous, "first_sound"), _clock(previous, "last_sound_end")
    if start is not None and first is not None and last is not None and first <= start <= last + window_s:
        reasons.append(f"started {start - last:+.2f} s after the skull's last sound (window {window_s:g} s)")
    run = echo_run(text, reply_text(previous))
    if run is not None and not any(f" {run} " in f" {' '.join(words(q))} " for q in questions):
        reasons.append(f"heard the skull's own words {run!r}")
    return reasons


@dataclass
class TurnCheck:
    """One logged turn, judged."""
    number: int
    record: dict
    outcome: str                         # answered, fallback, offline, nothing heard, error
    expected: str | None = None
    errors: int | None = None
    false_wake: bool = False
    self_trigger: list[str] = field(default_factory=list)
    ends_run: str | None = None
    silence_s: float | None = None       # end of speech -> first sound of any kind


def outcome_of(record: dict) -> str:
    if record.get("end_state") == "error":
        return "error"
    if not record.get("text"):
        return "nothing heard"
    if record.get("offline"):
        return "offline"
    brain = record.get("brain") or {}
    if brain.get("kind"):
        return f"fallback ({brain['kind']})"
    return "answered"


def check_turns(run: Run, false_wakes: set[int] = frozenset()) -> tuple[list[TurnCheck], list[str]]:
    """Judge every turn in log order, and the idle lines. Returns (turn checks, run-level events).

    A turn ends the run (decision 6) on: an error state, heard text but no sound at all, silence
    past the brain timeout, or a self-trigger. Turns after the tenth counted one happened in the
    silent minute: they are self-triggers unless --false-wake marks them (the video decides).
    """
    questions = run.header.get("questions") or []
    timeout_s = (run.header.get("loop") or {}).get("timeout_s")
    checks, events, previous, asked = [], [], None, 0
    for number, record in enumerate(run.turns, start=1):
        check = TurnCheck(number, record, outcome_of(record))
        start = _clock(record, "wake_start") if record.get("trigger") == "wake" else None
        start = start if start is not None else _clock(record, "speech_start")
        heard = " ".join(t for t in (record.get("wake_text"), record.get("raw_text")) if t)
        check.self_trigger = self_trigger_reasons(start, heard, previous, questions)
        speech_end, first = _clock(record, "speech_end"), _clock(record, "first_sound")
        if speech_end is not None and first is not None:
            check.silence_s = round(first - speech_end, 3)
        previous = record if _clock(record, "first_sound") is not None else previous
        if number in false_wakes:
            check.false_wake = True
            checks.append(check)
            continue
        asked += 1
        if asked > REQUIRED_TURNS:
            check.self_trigger.append(f"turn {number} came after the {REQUIRED_TURNS}th question, in the silent "
                                      f"minute (if the video shows someone else spoke: --false-wake {number})")
        elif asked <= len(questions):
            check.expected = questions[asked - 1]
            check.errors = word_errors(check.expected, record.get("raw_text") or "")
        if check.outcome == "error":
            check.ends_run = "error state (a crash inside the turn)"
        elif check.self_trigger:
            check.ends_run = "self-trigger: " + "; ".join(check.self_trigger)
        elif record.get("text") and first is None:
            check.ends_run = "heard text but the skull made no sound"
        elif timeout_s is not None and check.silence_s is not None and check.silence_s > timeout_s:
            check.ends_run = f"silence {check.silence_s:.2f} s after the question, past the {timeout_s:g} s timeout"
        checks.append(check)
        if check.ends_run:
            events.append(f"turn {number}: {check.ends_run}")
    for line in run.lines:
        if line.get("woke"):
            continue                       # a wake utterance: judged with its turn above
        start = line.get("t_start")
        before = [r for r in run.turns if start is not None and _clock(r, "first_sound") is not None
                  and _clock(r, "first_sound") <= start]
        reasons = self_trigger_reasons(start, line.get("text") or "", before[-1] if before else None, questions)
        if reasons:
            events.append(f"idle line {line.get('text')!r} after turn {len(before)}: self-trigger: " + "; ".join(reasons))
    return checks, events


def footer_problem(footer: dict) -> str | None:
    """A crash, as the footer tells it; None when the loop ended normally (q, Ctrl+C or end of input)."""
    if not footer:
        return "no footer.json: the run did not end through the tool (killed, or power lost?)"
    if footer.get("outcome") == "exception":
        return f"crash: {footer.get('exception')}"
    if footer.get("outcome") == "returned" and footer.get("exit_code") not in (0, None):
        return f"the loop exited with code {footer.get('exit_code')}"
    return None


def median_worst(values: list[float]) -> str:
    """Median, worst and n, plus every value: with n = 10 a percentile beyond the median says little."""
    if not values:
        return "n=0"
    ordered = sorted(values)
    return (f"median {statistics.median(ordered):.3f} s, worst {ordered[-1]:.3f} s, n={len(ordered)} "
            f"[{', '.join(f'{v:.2f}' for v in ordered)}]")


def latency_lines(checks: list[TurnCheck]) -> list[str]:
    """The headline (end of speech -> first reply audio) and the stages, over clean answered turns."""
    answered = [c.record for c in checks if c.outcome == "answered" and not c.false_wake and c.record.get("speech")]
    clean = [r for r in answered if r.get("queue_full") == 0 and r.get("gaps") == 0]
    speech = [r["speech"] for r in clean]

    def first_reply_clip(s: dict) -> dict:
        return next((c for c in s.get("clips", []) if c.get("source") != "opener" and c.get("t_play_start") is not None), {})

    def values(get) -> list[float]:
        return [v for v in (get(r) for r in clean) if v is not None]

    brain = [r.get("brain") or {} for r in clean]
    return [
        f"answered turns {len(answered)}, with clean times {len(clean)} (lost blocks in {len(answered) - len(clean)})",
        "HEADLINE speech end -> first reply audio: " + median_worst([s["first_audio_s"] for s in speech
                                                                     if s.get("first_audio_s") is not None]),
        "speech end -> Cogitating starts: " + median_worst([s["opener_start_s"] for s in speech
                                                            if s.get("opener_start_s") is not None]),
        "detector end -> first reply audio (the detector ends ~hang after the real end): "
        + median_worst(values(lambda r: r.get("detector_end_to_first_audio_s"))),
        "  stage, speech end -> text: " + median_worst(values(lambda r: r.get("stt_s"))),
        "  stage, detector end -> text: " + median_worst(values(lambda r: r.get("detector_end_to_text_s"))),
        "  stage, speech end -> brain first piece: " + median_worst(values(lambda r: r.get("speech_end_to_first_piece_s"))),
        "  stage, speech end -> sentence 1 generated: " + median_worst(
            [c["t_ready"] for c in map(first_reply_clip, speech) if c.get("t_ready") is not None]),
        "  stage, speech end -> sentence 1 playing: " + median_worst(
            [c["t_play_start"] for c in map(first_reply_clip, speech) if c.get("t_play_start") is not None]),
        "  stage, speech end -> turn done: " + median_worst(values(lambda r: r.get("speech_end_to_done_s"))),
        "  brain first chunk (its own clock): " + median_worst([b["first_chunk_ms"] / 1000 for b in brain
                                                                if b.get("first_chunk_ms") is not None]),
        "  gaps between sentences: " + median_worst([g for s in speech for g in s.get("gaps_s", [])]),
        f"  bank hits {sum(s.get('bank_hits', 0) for s in speech)} of {sum(s.get('sentences', 0) for s in speech)} "
        f"sentences; hit lines: {sorted({c['text'] for s in speech for c in s.get('clips', []) if c.get('source') == 'bank'})}",
    ]


def memory_lines(run: Run) -> list[str]:
    """G9 at the same moment: the lowest MemAvailable against the value before start, and the peak tree RSS."""
    base = run.header.get("mem_available_before_kb")
    mem = [s for s in run.samples if s.get("mem_available_kb") is not None]
    rss = [s for s in run.samples if s.get("tree_rss_kb") is not None]
    lines = []
    if base is not None and mem:
        low = min(mem, key=lambda s: s["mem_available_kb"])
        drop = base - low["mem_available_kb"]
        lines.append(f"G9, MemAvailable: before {base:,} kB, lowest {low['mem_available_kb']:,} kB (at {low.get('at')}), "
                     f"drop {drop:,} kB = {100 * drop / G9_LIMIT_KB:.1f} % of the {G9_LIMIT_KB:,} kB limit "
                     f"({'within' if drop <= G9_LIMIT_KB else 'OVER'}); n={len(mem)} samples")
    else:
        lines.append("G9, MemAvailable: no baseline or no samples")
    if rss:
        peak = max(rss, key=lambda s: s["tree_rss_kb"])
        lines.append(f"G9, loop process tree RSS: peak {peak['tree_rss_kb']:,} kB in {peak.get('tree_procs')} processes "
                     f"(at {peak.get('at')}) = {100 * peak['tree_rss_kb'] / G9_LIMIT_KB:.1f} % of the limit "
                     f"({'within' if peak['tree_rss_kb'] <= G9_LIMIT_KB else 'OVER'}; an upper bound: shared pages count twice)")
    temps = [s for s in run.samples if s.get("temp_c") is not None]
    if temps:
        flags = sorted({s["throttled"] for s in temps if s.get("throttled") not in (None, "0x0")})
        lines.append(f"temperature: peak {max(s['temp_c'] for s in temps):.1f} C, median "
                     f"{statistics.median(s['temp_c'] for s in temps):.1f} C, n={len(temps)} ({temps[0].get('source')}); "
                     f"throttled other than 0x0: {flags or 'none'}")
    else:
        lines.append("temperature: no readings")
    cpu, wall = run.footer.get("cpu_s"), run.footer.get("wall_s")
    if cpu is not None and wall:
        lines.append(f"CPU, whole run (loop, TTS worker, claude): {cpu:.1f} s over {wall:.1f} s = "
                     f"{100 * cpu / wall:.1f} % of one core on average (not G4's quiet/talking split)")
    return lines


def _turn_row(check: TurnCheck) -> str:
    record = check.record
    first = (record.get("speech") or {}).get("first_audio_s")
    parts = [f"{check.number:>2}", record.get("trigger") or "-", check.outcome,
             f"heard {record.get('raw_text')!r}"]
    if check.expected is not None:
        parts.append(f"word errors {check.errors}")
    parts.append("first reply audio " + ("-" if first is None else f"{first:.2f} s"))
    parts.append("silence " + ("-" if check.silence_s is None else f"{check.silence_s:.2f} s"))
    if check.false_wake:
        parts.append("FALSE WAKE (marked)")
    if check.ends_run:
        parts.append("ENDS THE RUN: " + check.ends_run)
    return " | ".join(parts)


def report_ten(run: Run, false_wakes: set[int] = frozenset()) -> tuple[bool, str]:
    """The exit-test verdict and its report. Pass: ten counted turns, no event, no crash, a silent minute after."""
    checks, events = check_turns(run, false_wakes)
    crash = footer_problem(run.footer)
    counted = [c for c in checks if not c.false_wake]
    streak = 0
    for check in counted:
        if check.ends_run:
            break
        streak += 1
    last_end = max((_clock(c.record, "last_sound_end") or _clock(c.record, "done") or 0.0 for c in checks), default=None)
    silent = (run.footer.get("t_end") - last_end) if last_end and run.footer.get("t_end") else None
    problems = list(events)
    if crash:
        problems.append(crash)
    if len(counted) < REQUIRED_TURNS:
        problems.append(f"only {len(counted)} turns logged, {REQUIRED_TURNS} needed (a hang or a quit: write the cause in the log)")
    if silent is None or silent < SILENT_MINUTE_S:
        problems.append("silent minute after the last turn: "
                        + ("not measured" if silent is None else f"only {silent:.0f} s of {SILENT_MINUTE_S:.0f} s"))
    passed = not problems
    header, loop = run.header, run.header.get("loop") or {}
    misheard = [c for c in counted if c.errors]
    lines = [
        f"M3-08 exit test, run {header.get('run_dir')}: {'PASS' if passed else 'FAIL'}",
        f"  started {header.get('started_at')}, commit {header.get('commit')} ({header.get('dirty_files')} files changed), "
        f"{header.get('claude_version')}, model {loop.get('model')}, thinking {'off' if loop.get('thinking_off') else 'on'}, "
        f"per-turn claude process, timeout {loop.get('timeout_s')} s",
        f"  trigger {header.get('trigger')}, voice {loop.get('voice')}, cooling {header.get('cooling')!r}, room "
        f"{header.get('room_temp')!r}, amp {header.get('amp_volume')!r}, mic {header.get('mic_distance')!r}, "
        f"people {header.get('people')!r}, end rule {loop.get('end_rule')}, hang {loop.get('hang')} s",
        f"  turns logged {len(checks)} (false wakes marked {len(checks) - len(counted)}); clean turns in a row from the start: {streak}",
        f"  misheard {len(misheard)}, nothing heard {sum(1 for c in counted if c.outcome == 'nothing heard')}, "
        f"fallback lines {sum(1 for c in counted if c.outcome.startswith('fallback'))}, "
        f"offline {sum(1 for c in counted if c.outcome == 'offline')}",
    ]
    lines += ["  " + _turn_row(c) for c in checks]
    lines += ["  why it failed: " + p for p in problems]
    lines += ["latency:"] + ["  " + line for line in latency_lines(checks)]
    lines += ["memory, heat, CPU:"] + ["  " + line for line in memory_lines(run)]
    lines.append(f"self-trigger rule: speech starting within {SELF_TRIGGER_WINDOW_S:g} s of the skull's last sound, "
                 f"or {ECHO_WORDS} words in a row of its own reply; idle lines checked: "
                 f"{sum(1 for line in run.lines if not line.get('woke'))}")
    return passed, "\n".join(lines)


def report_wifi(run: Run) -> tuple[bool, str]:
    """The WiFi-off checks from one run: offline turns, a fallback during a brain call, then a normal turn.

    Pass: no crash, no error state, no silence past the timeout, every heard question got a sound,
    at least one offline turn and one fallback turn, and the last turn answered normally.
    Check (1), a state command offline, is not testable: the Vosk path is not wired (M3-07 TODO).
    """
    checks, _ = check_turns(run)
    timeout_s = (run.header.get("loop") or {}).get("timeout_s")
    problems = []
    crash = footer_problem(run.footer)
    if crash:
        problems.append(crash)
    for check in checks:
        if check.ends_run and not check.ends_run.startswith("self-trigger"):
            problems.append(f"turn {check.number}: {check.ends_run}")
    if not any(c.outcome == "offline" for c in checks):
        problems.append("no offline turn (check 2 not shown)")
    if not any(c.outcome.startswith("fallback") for c in checks):
        problems.append("no fallback turn (check 3, WiFi cut during a brain call, not shown: try again)")
    if not checks or checks[-1].outcome != "answered":
        problems.append("the last turn was not a normal answer (check 4, WiFi back)")
    lines = [f"M3-08 WiFi-off test, run {run.header.get('run_dir')}: {'PASS' if not problems else 'FAIL'}",
             "  check 1 (a state command offline): NOT TESTABLE, the offline Vosk commands are not wired (M3-07 TODO)",
             f"  one process from start to end: pid {run.footer.get('pid')}, {len(checks)} turns, timeout {timeout_s} s"]
    lines += ["  " + _turn_row(c) for c in checks]
    lines += ["  why it failed: " + p for p in problems]
    offline = [c.silence_s for c in checks if c.outcome == "offline" and c.silence_s is not None]
    lines.append("  offline: end of question -> first sound: " + median_worst(offline))
    return not problems, "\n".join(lines)


# --- running the loop ----------------------------------------------------------------------------

def _command_output(argv: list[str], cwd: Path | None = None) -> str | None:
    try:
        done = subprocess.run(argv, cwd=cwd, capture_output=True, text=True, timeout=20)
    except (OSError, subprocess.SubprocessError):
        return None
    return done.stdout.strip() if done.returncode == 0 else None


def make_header(args: argparse.Namespace, loop_args: argparse.Namespace, loop_argv: list[str],
                questions: list[str], mem_before: int | None) -> dict:
    """What was run and how, written before the loop starts (the method note of the step note)."""
    from software.skull_core.voice_console import DEFAULT_TAIL_S
    from software.skull_core.voice_state import voice_key
    status = _command_output(["git", "status", "--porcelain"], REPO_ROOT)
    bank_index = _read_json(Path(loop_args.bank).expanduser() / "index.json")
    try:
        key = voice_key(loop_args.params)
    except (OSError, ValueError) as exc:
        key = f"unreadable: {exc}"
    return {
        "run_dir": str(args.run_dir.expanduser()), "started_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "t_start": time.monotonic(), "host": socket.gethostname(), "pid": os.getpid(),
        "python": sys.version.split()[0], "commit": _command_output(["git", "rev-parse", "--short", "HEAD"], REPO_ROOT),
        "dirty_files": None if status is None else len(status.splitlines()),
        "claude_version": _command_output([loop_args.claude, "--version"]),
        "trigger": args.trigger, "cooling": args.cooling, "room_temp": args.room_temp,
        "amp_volume": args.amp_volume, "mic_distance": args.mic_distance, "people": args.people, "note": args.note,
        "questions_file": str(args.questions), "questions": questions,
        "mem_available_before_kb": mem_before, "g9_limit_kb": G9_LIMIT_KB,
        "self_trigger_window_s": SELF_TRIGGER_WINDOW_S, "guard_tail_s": DEFAULT_TAIL_S, "voice_key": key,
        "loop_argv": loop_argv,
        "loop": {"model": loop_args.model, "thinking_off": loop_args.no_thinking, "timeout_s": loop_args.timeout,
                 "voice": loop_args.voice, "bank": str(loop_args.bank), "bank_lines": sorted(
                     entry.get("text", "") for entry in bank_index.values() if isinstance(entry, dict)),
                 "end_rule": loop_args.end_rule, "hang": loop_args.hang, "probe": not loop_args.no_probe,
                 "probe_host": loop_args.probe_host, "wake_window": loop_args.wake_window,
                 "system_prompt_file": None if loop_args.system_prompt_file is None else str(loop_args.system_prompt_file)},
    }


def run_test(args: argparse.Namespace, loop_argv: list[str], main=None, out=print) -> int:
    """Write the header, sample while voice_loop.main runs, write the footer, print the report."""
    mem_before = mem_available_kb()           # first, before the loop's imports and models
    run_dir = args.run_dir.expanduser()
    if run_dir.exists() and any(run_dir.iterdir()):
        out(f"{run_dir} is not empty: every run gets a new folder, so old turns never mix in.")
        return 1
    try:
        questions = read_questions(args.questions)
    except OSError as exc:
        out(f"Could not read the questions {args.questions}: {exc}")
        return 1
    from software.skull_core import voice_loop
    full_argv = list(loop_argv) + ["--trigger", args.trigger, "--log", str(run_dir / "turns.jsonl"),
                                   "--line-log", str(run_dir / "wake_lines.jsonl")]
    loop_args = voice_loop.parse_args(full_argv)        # a bad flag stops here, before anything is written
    if loop_args.no_brain or loop_args.no_speak or loop_args.phrases is not None or loop_args.report is not None:
        out("The exit test needs the brain and the voice: drop --no-brain, --no-speak, --phrases and --report.")
        return 1
    run_dir.mkdir(parents=True, exist_ok=True)
    header = make_header(args, loop_args, full_argv, questions, mem_before)
    (run_dir / "header.json").write_text(json.dumps(header, indent=1), encoding="utf-8")
    out(f"Run folder {run_dir}. Trigger {args.trigger}. MemAvailable before: {mem_before} kB.")
    out("Questions, in order:\n" + "\n".join(f"  {i}. {q}" for i, q in enumerate(questions, start=1)))
    out(f"After question {REQUIRED_TURNS}: stay silent for {SILENT_MINUTE_S:.0f} s, then quit (q + Enter, or Ctrl+C).")
    sampler = Sampler(run_dir / "samples.jsonl", os.getpid(), report=out).start()
    t0, cpu0 = time.monotonic(), os.times()
    footer = {"pid": os.getpid(), "outcome": "returned", "exit_code": None, "exception": None}
    try:
        footer["exit_code"] = (main or voice_loop.main)(full_argv)
    except KeyboardInterrupt:
        footer["outcome"] = "interrupted"
    except BaseException as exc:                # also SystemExit: the run still gets its footer and report
        footer["outcome"], footer["exception"] = "exception", repr(exc)
        traceback.print_exc()
    finally:
        sampler.stop()
        cpu1 = os.times()
        footer.update({"t_end": time.monotonic(), "ended_at": datetime.now().astimezone().isoformat(timespec="seconds"),
                       "wall_s": round(time.monotonic() - t0, 3), "samples": sampler.rows,
                       # children_*: only children that were waited for (each claude call, the TTS worker at close)
                       "cpu_s": round(sum(cpu1[:4]) - sum(cpu0[:4]), 3)})
        (run_dir / "footer.json").write_text(json.dumps(footer, indent=1), encoding="utf-8")
    passed, text = report_ten(load_run(run_dir))
    out(text)
    return 0 if footer["outcome"] != "exception" and footer["exit_code"] in (0, None) else 1


def parse_args(argv: list[str]) -> tuple[argparse.Namespace, list[str]]:
    """Split at "--": ours before it, voice_loop's after it."""
    ours, loop_argv = (argv[:argv.index("--")], argv[argv.index("--") + 1:]) if "--" in argv else (argv, [])
    parser = argparse.ArgumentParser(description="M3-08 exit test: run the voice loop with samples, or report a run.")
    sub = parser.add_subparsers(dest="command", required=True)
    run = sub.add_parser("run", help="run the voice loop once into a new run folder")
    run.add_argument("--run-dir", type=Path, required=True)
    run.add_argument("--trigger", choices=("wake", "enter"), required=True, help="decision 4: recorded in the header")
    run.add_argument("--cooling", required=True, help='e.g. "heatsink and fan"')
    run.add_argument("--room-temp", default="not recorded")
    run.add_argument("--amp-volume", default="not recorded")
    run.add_argument("--mic-distance", default="about 30 cm")
    run.add_argument("--people", default="not recorded")
    run.add_argument("--note", default="")
    run.add_argument("--questions", type=Path, default=DEFAULT_QUESTIONS)
    rep = sub.add_parser("report", help="analyse a run folder")
    rep.add_argument("--run-dir", type=Path, required=True)
    rep.add_argument("--wifi", action="store_true", help="judge the WiFi-off checks instead of the ten turns")
    rep.add_argument("--false-wake", type=int, action="append", default=[], metavar="TURN",
                     help="a turn the video shows was started by someone else (not counted, not a self-trigger)")
    args = parser.parse_args(ours)
    if args.command == "report" and loop_argv:
        parser.error("report takes no voice_loop arguments")
    for flag in LOOP_FLAGS_SET_HERE:
        if any(item == flag or item.startswith(flag + "=") for item in loop_argv):
            parser.error(f"{flag} is set by the exit test; leave it out after --")
    return args, loop_argv


def main(argv=None) -> int:
    args, loop_argv = parse_args(sys.argv[1:] if argv is None else argv)
    if args.command == "run":
        return run_test(args, loop_argv, out=lambda text: print(text, flush=True))
    run_dir = args.run_dir.expanduser()
    if not (run_dir / "header.json").exists():
        print(f"No header.json in {run_dir}: not a run folder.", file=sys.stderr)
        return 1
    run = load_run(run_dir)
    passed, text = report_wifi(run) if args.wifi else report_ten(run, set(args.false_wake))
    print(text, flush=True)
    return 0 if passed else 2


if __name__ == "__main__":
    sys.exit(main())
