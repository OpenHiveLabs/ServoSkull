"""Voice out: the reply's text pieces -> sentences -> audio, sentence one playing while sentence two is made.

How a reply is spoken (Speaker.speak):
  - A producer thread reads the text pieces, cuts them into sentences (split_sentences) and turns
    each one into a Clip: from the phrase bank if the exact line is stored there, else from the
    TTS worker (Pocket TTS, voice javert, in its own process: tts_worker.py). It puts the Clips on
    a queue, then one END marker, also when something failed.
  - The calling thread first plays the "Cogitating" clip (Ersin's voice/thinking.wav), so the
    wait for the brain starts with a sound, then plays the Clips in order as they arrive.
  - The half-duplex guard is shut for all of it, Cogitating included, and opens again (after its
    tail) only when the last clip has played: the mic never hears the skull.
  - Every clip's times go into a SpeechLog: time to first audio, the gap between sentences.

Pocket TTS makes 24 kHz audio; the amp runs at 48 kHz (PortAudio refuses 24 kHz on it, M3-00 (g)).
to_48k doubles the rate in code. No clip is played louder than PEAK_CAP_DBFS (F-030).

Usage (from the repo root; -m is required; on the Pi run it with ~/m3-bench/.venv-stt/bin/python):
    python -m software.skull_core.voice_out --build-bank software/skull_core/voice/bank_lines.txt
           [--bank ~/skull-voice/bank] [--tts-python ~/m3-bench/.venv-tts/bin/python] [--voice javert]
    python -m software.skull_core.voice_out --say "Text to speak." [--opener] [--repeat 1]
           [--device MAX98357A] [--log ~/m3-bench/logs/m305_say.jsonl] [--bank ...] [--tts-python ...]
    python -m software.skull_core.voice_out --misses 20 [--bank ...]

--build-bank FILE  generates every line of FILE that the bank doesn't have yet (once; about 2.5x
                   the line's length each on the Pi). Blank lines and lines starting with # are skipped.
--say TEXT         speaks TEXT through the same path as the loop (bank first, then the worker).
--misses N         lists the N sentences generated most often (candidates for the bank file).
"""
import argparse
import hashlib
import json
import math
import os
import queue
import re
import select
import subprocess
import sys
import threading
import time
import wave
from collections.abc import Callable, Iterable, Iterator
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path

import numpy as np

from software.skull_core.state_console import VOICE_DIR, WAV_RATE, load_wav
from software.skull_core.tts_worker import DEFAULT_VOICE, SAMPLE_BYTES
from software.skull_core.voice_console import HalfDuplexGuard

OUT_RATE = WAV_RATE                     # 48 kHz: the rate the amp path is tested at (M2-03)
ENGINE = "pocket-tts"                   # the bank keys its lines by engine and voice, e.g. "pocket-tts/javert"
SOURCE_SYNTH = "pocket"                 # ClipLog.source for a generated sentence
DEFAULT_BANK_DIR = Path.home() / "skull-voice" / "bank"      # generated audio: outside the repo, never committed
DEFAULT_TTS_PYTHON = Path.home() / "m3-bench" / ".venv-tts" / "bin" / "python"   # the venv of M3-00 (b)
DEFAULT_SAY_LOG = Path.home() / "m3-bench" / "logs" / "m305_say.jsonl"
BANK_LINES_FILE = Path(__file__).resolve().parent / "voice" / "bank_lines.txt"
OPENER_FILE = "thinking.wav"            # "Cogitating", Ersin's own recording
REPO_ROOT = Path(__file__).resolve().parents[2]

# Loudness cap. The loudest audio heard clean on the amp was Pocket TTS at -6.14 dBFS peak; Piper at
# full scale (0 dBFS) crackled (F-030, M3-00b). A clip whose peak is above the cap is scaled down
# as a whole (never clipped). Pocket's own peaks (-10.94 to -6.14 dBFS on the Surface files) pass untouched.
PEAK_CAP_DBFS = -6.0

READY_TIMEOUT_S = 180.0     # estimate: model + voice + warm-up took 13.7 s on the Pi; the first run downloads 216 MB
SENTENCE_TIMEOUT_S = 90.0   # estimate: a 30 s sentence at the Pi's RTF 2.52 takes about 76 s
GET_TIMEOUT_S = 0.2         # the playing thread wakes this often, so Ctrl+C is noticed

# 2:1 upsampling filter: a windowed-sinc low-pass at 11 kHz, 95 taps, Kaiser beta 8.6.
# Measured on the Surface 2026-10-20 (numpy FFT of the taps): passband to 10 kHz within -0.11 dB,
# everything at 13 kHz and above at least 89 dB down. Repeating every sample instead leaves the
# mirror image of a 1 kHz tone (at 23 kHz) only 23.7 dB down, and of a 5 kHz tone 9.4 dB down.
UPSAMPLE_TAPS = 95
UPSAMPLE_CUTOFF_HZ = 11_000
UPSAMPLE_BETA = 8.6


# --- sentences -------------------------------------------------------------------------------

# An end mark (. ! ? or a run of them, e.g. "..." or "?!"), maybe a closing quote or bracket,
# then whitespace. "3.5" is not an end: no space after the dot.
_SENTENCE_END = re.compile(r"[.!?]+[\"'”’)\]]*\s+")


def split_sentences(pieces: Iterable[str]) -> Iterator[str]:
    """Text pieces in (cut anywhere, even mid-word), whole sentences out, as early as possible.

    A sentence ends at an end mark followed by whitespace, so the splitter waits for the next
    piece before it knows "noon." is finished. Whatever is left when the pieces stop is the last
    sentence. Sentences are stripped; empty ones are never yielded.
    Known limit: "Dr. Smith" is cut after "Dr." (no list of abbreviations).
    """
    buffer = ""
    for piece in pieces:
        buffer += piece
        while True:
            match = _SENTENCE_END.search(buffer)
            if match is None:
                break
            sentence = buffer[:match.end()].strip()
            buffer = buffer[match.end():]
            if sentence:
                yield sentence
    rest = buffer.strip()
    if rest:
        yield rest


def normalise(text: str) -> str:
    """The bank's key for a sentence, used for both storing and lookup.

    Curly apostrophes and quotes become straight ones (the brain writes "can’t"), case is
    folded, runs of whitespace become one space, and a final "." or "!" is dropped. A final "?"
    is kept: a question is spoken differently.
    """
    text = text.replace("’", "'").replace("‘", "'").replace("“", '"').replace("”", '"')
    text = " ".join(text.casefold().split())
    return text.rstrip(".!").rstrip()


# --- audio -------------------------------------------------------------------------------------

def _upsample_taps() -> np.ndarray:
    n = np.arange(UPSAMPLE_TAPS) - (UPSAMPLE_TAPS - 1) / 2
    cutoff = UPSAMPLE_CUTOFF_HZ / OUT_RATE                    # cycles per output sample
    taps = 2 * cutoff * np.sinc(2 * cutoff * n) * np.kaiser(UPSAMPLE_TAPS, UPSAMPLE_BETA)
    return taps / taps.sum() * 2                              # gain 2: zero-stuffing halved the level


_TAPS = _upsample_taps()


def to_48k(samples: np.ndarray, rate: int) -> np.ndarray:
    """Engine audio at `rate` -> float32 at 48 kHz, shape (frames, 1), ready for the amp.

    48 kHz passes through. 24 kHz is doubled: a zero goes between every two samples, then the
    low-pass filter removes the mirror image that the zeros create above 12 kHz. Exactly 2N frames
    out for N in. Any other rate is a ValueError (Piper's 22,050 Hz would need 320/147: not built).
    """
    flat = np.asarray(samples, dtype=np.float64).reshape(-1)
    if rate == OUT_RATE:
        return flat.astype(np.float32).reshape(-1, 1)
    if rate * 2 != OUT_RATE:
        raise ValueError(f"cannot play {rate} Hz audio: only {OUT_RATE // 2} and {OUT_RATE} Hz are handled")
    if len(flat) == 0:
        return np.zeros((0, 1), dtype=np.float32)
    stuffed = np.zeros(2 * len(flat))
    stuffed[0::2] = flat
    delay = (UPSAMPLE_TAPS - 1) // 2                          # the filter's delay, in output samples
    filtered = np.convolve(stuffed, _TAPS)[delay:delay + len(stuffed)]
    return filtered.astype(np.float32).reshape(-1, 1)


def peak_dbfs(samples: np.ndarray) -> float:
    peak = float(np.max(np.abs(samples))) if len(samples) else 0.0
    return -math.inf if peak == 0.0 else 20 * math.log10(peak)


def cap_peak(samples: np.ndarray, cap_dbfs: float = PEAK_CAP_DBFS) -> np.ndarray:
    """The same audio, scaled down as a whole if its peak is above cap_dbfs. Never louder, never clipped."""
    level = peak_dbfs(samples)
    if level <= cap_dbfs:
        return samples
    return (samples * 10 ** ((cap_dbfs - level) / 20)).astype(np.float32)


def save_wav48(path: Path, samples: np.ndarray) -> None:
    """Write float audio in -1..1 as a 48 kHz, 16-bit, mono WAV (the format load_wav reads)."""
    pcm = (np.clip(np.asarray(samples, dtype=np.float32).reshape(-1), -1.0, 1.0) * 32767.0).astype("<i2")
    with wave.open(str(path), "wb") as out:
        out.setnchannels(1)
        out.setsampwidth(2)
        out.setframerate(OUT_RATE)
        out.writeframes(pcm.tobytes())


# --- the phrase bank ---------------------------------------------------------------------------

def _write_json_atomic(path: Path, data: dict) -> None:
    """Write to a temporary file, then rename over the old one: a crash leaves the old or the new file, never half."""
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(data, indent=1, ensure_ascii=False), encoding="utf-8")
    os.replace(tmp, path)


def _read_json(path: Path) -> dict:
    if not path.exists():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise ValueError(f"{path}: unreadable ({exc})") from exc
    if not isinstance(data, dict):
        raise ValueError(f"{path}: expected a JSON object")
    return data


def bank_voice(voice: str) -> str:
    return f"{ENGINE}/{voice}"


class PhraseBank:
    """Stored WAVs for exact sentences, in one folder outside the repo.

    index.json   {normalised text: {"file", "text", "voice"}}: a line made with another voice never matches.
    misses.json  {normalised text: {"text", "count"}}: how often each sentence had to be generated.
    The bank does not grow by itself in M3 (decision 5): lines are added with --build-bank.
    """
    INDEX = "index.json"
    MISSES = "misses.json"

    def __init__(self, folder: Path, voice: str):
        self.folder = Path(folder)
        self.voice = voice
        self._index = _read_json(self.folder / self.INDEX)

    def lookup(self, sentence: str) -> Path | None:
        """The stored WAV for this exact (normalised) text and this voice, or None."""
        entry = self._index.get(normalise(sentence))
        if not isinstance(entry, dict) or entry.get("voice") != self.voice:
            return None
        path = self.folder / str(entry.get("file", ""))
        return path if path.is_file() else None

    def add(self, sentence: str, samples48: np.ndarray) -> Path:
        """Store one line (48 kHz audio) and record it in the index."""
        key = normalise(sentence)
        slug = re.sub(r"[^a-z0-9]+", "-", key).strip("-")[:40] or "line"
        digest = hashlib.sha1(f"{self.voice}|{key}".encode("utf-8")).hexdigest()[:8]
        name = f"{slug}-{digest}.wav"
        self.folder.mkdir(parents=True, exist_ok=True)
        save_wav48(self.folder / name, samples48)
        self._index[key] = {"file": name, "text": sentence, "voice": self.voice}
        _write_json_atomic(self.folder / self.INDEX, self._index)
        return self.folder / name

    def note_miss(self, sentence: str) -> int:
        """Count this text as generated once more; return its new count. Survives a restart."""
        path = self.folder / self.MISSES
        misses = _read_json(path)
        key = normalise(sentence)
        entry = misses.get(key) if isinstance(misses.get(key), dict) else {"text": sentence, "count": 0}
        entry["count"] = int(entry.get("count", 0)) + 1
        misses[key] = entry
        self.folder.mkdir(parents=True, exist_ok=True)
        _write_json_atomic(path, misses)
        return entry["count"]

    def top_misses(self, n: int) -> list[tuple[int, str]]:
        misses = _read_json(self.folder / self.MISSES)
        rows = [(int(e.get("count", 0)), str(e.get("text", k))) for k, e in misses.items() if isinstance(e, dict)]
        return sorted(rows, key=lambda row: (-row[0], row[1]))[:n]


def read_bank_lines(path: Path) -> list[str]:
    """The fixed lines: one per line; blank lines and lines starting with # are skipped."""
    lines = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line and not line.startswith("#"):
            lines.append(line)
    return lines


# --- the TTS worker, seen from the loop ----------------------------------------------------------

class SynthError(Exception):
    """The TTS worker failed: it did not start, timed out, died, or answered with an error."""


def worker_argv(python: Path, voice: str) -> list[str]:
    return [str(python), "-u", "-m", "software.skull_core.tts_worker", "--voice", voice]


class WorkerSynth:
    """The loop's side of the tts_worker pipe protocol. synth(text) -> (samples, rate).

    One request at a time, from one thread (the producer). Every read has a deadline: a worker that
    hangs is killed instead of freezing the skull. A dead worker is started again on the next call
    (that costs the model load, about 14 s on the Pi).
    """

    def __init__(self, argv: list[str], cwd: Path = REPO_ROOT, env: dict | None = None,
                 ready_timeout_s: float = READY_TIMEOUT_S, sentence_timeout_s: float = SENTENCE_TIMEOUT_S):
        self.argv = argv
        self.cwd = cwd
        self.env = env
        self.ready_timeout_s = ready_timeout_s
        self.sentence_timeout_s = sentence_timeout_s
        self.proc: subprocess.Popen | None = None
        self.info: dict = {}
        self._buffer = b""

    def start(self) -> None:
        """Start the worker process (it loads the model in the background; wait_ready() waits for it)."""
        if self.proc is not None and self.proc.poll() is None:
            return
        self._buffer = b""
        try:
            self.proc = subprocess.Popen(self.argv, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                         cwd=self.cwd, env=self.env, bufsize=0)
        except OSError as exc:
            self.proc = None
            raise SynthError(f"cannot start the TTS worker {self.argv[0]}: {exc}") from exc

    def wait_ready(self) -> dict:
        """Block until the worker says it is ready; return its ready message (rate, engine, voice, load_s)."""
        self.start()
        message = self._read_json_line(time.monotonic() + self.ready_timeout_s, "the ready line")
        if not message.get("ready"):
            self.abort()
            raise SynthError(f"the TTS worker failed to load: {message.get('error', message)}")
        self.info = message
        return message

    def __call__(self, text: str) -> tuple[np.ndarray, int]:
        if self.proc is None or self.proc.poll() is not None or not self.info:
            self.wait_ready()
        try:
            self.proc.stdin.write((json.dumps({"text": text}) + "\n").encode("utf-8"))
            self.proc.stdin.flush()
        except (BrokenPipeError, OSError) as exc:
            self.abort()
            raise SynthError(f"the TTS worker is gone: {exc}") from exc
        deadline = time.monotonic() + self.sentence_timeout_s
        answer = self._read_json_line(deadline, "an answer")
        if not answer.get("ok"):
            raise SynthError(f"the TTS worker could not say it: {answer.get('error', answer)}")
        frames, rate = int(answer["frames"]), int(answer["rate"])
        pcm = self._read_exact(frames * SAMPLE_BYTES, deadline)
        samples = np.frombuffer(pcm, dtype="<i2").astype(np.float32) / 32768.0
        return samples, rate

    def abort(self) -> None:
        """Kill the worker now (a hang, or the loop is being stopped mid-sentence)."""
        proc, self.proc = self.proc, None
        self.info = {}
        if proc is None:
            return
        if proc.poll() is None:
            proc.kill()
        try:
            proc.wait(timeout=5.0)
        except subprocess.TimeoutExpired:
            pass
        for pipe in (proc.stdin, proc.stdout):
            try:
                pipe.close()
            except OSError:
                pass

    def close(self) -> None:
        """Polite stop: close its stdin (it exits at end of input), wait a little, then kill."""
        proc = self.proc
        if proc is None:
            return
        try:
            proc.stdin.close()
            proc.wait(timeout=5.0)
        except (OSError, subprocess.TimeoutExpired):
            pass
        self.abort()

    def _fill(self, deadline: float, what: str) -> None:
        """Read whatever the worker has sent into self._buffer, waiting at most until the deadline."""
        remaining = deadline - time.monotonic()
        fd = self.proc.stdout.fileno()
        ready, _, _ = select.select([fd], [], [], max(0.0, remaining))
        if not ready:
            self.abort()
            raise SynthError(f"the TTS worker sent no {what} in time; it was killed")
        chunk = os.read(fd, 65536)
        if not chunk:
            code = self.proc.poll()
            self.abort()
            raise SynthError(f"the TTS worker exited (code {code}) while we waited for {what}")
        self._buffer += chunk

    def _read_json_line(self, deadline: float, what: str) -> dict:
        if self.proc is None:
            raise SynthError("the TTS worker is not running")
        while b"\n" not in self._buffer:
            self._fill(deadline, what)
        line, self._buffer = self._buffer.split(b"\n", 1)
        try:
            message = json.loads(line)
        except ValueError as exc:
            self.abort()
            raise SynthError(f"the TTS worker sent a bad line: {line[:200]!r}") from exc
        if not isinstance(message, dict):
            self.abort()
            raise SynthError(f"the TTS worker sent a bad line: {line[:200]!r}")
        return message

    def _read_exact(self, size: int, deadline: float) -> bytes:
        while len(self._buffer) < size:
            self._fill(deadline, "the audio")
        data, self._buffer = self._buffer[:size], self._buffer[size:]
        return data


# --- speaking --------------------------------------------------------------------------------------

@dataclass
class ClipLog:
    """One clip's story. Times are clock times (time.monotonic on the Pi), or None."""
    text: str
    source: str                      # "opener", "bank", "pocket", or "none" (nothing to play: see error)
    gen_s: float | None = None       # bank: reading the file; pocket: the worker round trip plus resampling
    audio_s: float = 0.0
    peak_dbfs: float | None = None   # after the cap
    t_ready: float | None = None
    t_play_start: float | None = None
    t_play_end: float | None = None
    error: str | None = None


@dataclass
class Clip:
    """One sentence ready to play: 48 kHz float32 samples (or None if there is nothing to play) and its log."""
    samples: np.ndarray | None
    log: ClipLog
    synth_failed: bool = False       # the synth raised: the rest of this reply skips it
    state: str = "speaking"          # the voice state published while it plays (M3-07: "offline" for the fallback)


@dataclass
class SpeechLog:
    """Everything one speak() call did, in order: the opener first, then one ClipLog per sentence."""
    t_start: float
    clips: list[ClipLog] = field(default_factory=list)
    misses: int = 0
    error: str | None = None

    def _reply(self) -> list[ClipLog]:
        return [c for c in self.clips if c.source != "opener" and c.t_play_start is not None]

    def opener(self) -> ClipLog | None:
        return next((c for c in self.clips if c.source == "opener"), None)

    def first_audio(self) -> float | None:
        """When the first reply sentence started playing (not the opener)."""
        reply = self._reply()
        return reply[0].t_play_start if reply else None

    def gaps_s(self) -> list[float]:
        """Silence between consecutive reply sentences: next start minus previous end."""
        reply = self._reply()
        return [round(b.t_play_start - a.t_play_end, 3) for a, b in zip(reply, reply[1:])]

    def as_record(self, origin: float | None) -> dict:
        """JSON-ready, times in seconds relative to origin (the end of the user's speech)."""
        def rel(t):
            return None if t is None or origin is None else round(t - origin, 3)

        opener = self.opener()
        first = self.first_audio()
        clips = []
        for clip in self.clips:
            row = asdict(clip)
            for key in ("t_ready", "t_play_start", "t_play_end"):
                row[key] = rel(row[key])
            row["peak_dbfs"] = None if clip.peak_dbfs in (None, -math.inf) else round(clip.peak_dbfs, 2)
            row["gen_s"] = None if clip.gen_s is None else round(clip.gen_s, 3)
            row["audio_s"] = round(clip.audio_s, 3)
            clips.append(row)
        return {
            "opener_start_s": rel(opener.t_play_start) if opener else None,
            "first_audio_s": rel(first),
            "opener_end_to_first_audio_s": (round(first - opener.t_play_end, 3)
                                            if first is not None and opener and opener.t_play_end else None),
            "gaps_s": self.gaps_s(),
            "sentences": sum(1 for c in self.clips if c.source != "opener"),
            "bank_hits": sum(1 for c in self.clips if c.source == "bank"),
            "misses": self.misses,
            "error": self.error,
            "clips": clips,
        }


_END = object()   # the producer's last item: nothing more is coming


def _wait_for(clips: queue.Queue):
    """The next item. Waits in short steps, so Ctrl+C reaches the playing thread quickly."""
    while True:
        try:
            return clips.get(timeout=GET_TIMEOUT_S)
        except queue.Empty:
            continue


class Speaker:
    """Speaks replies. Holds the synth, the bank, the guard, the player and on_state; all passed in.

    synth(text) -> (samples, rate)   the TTS (WorkerSynth on the Pi, a fake in the tests)
    play(samples)                    plays 48 kHz float32 (frames, 1) and returns when it is done
    opener                           48 kHz samples played at the start of every reply, or None
    show(clip_log)                   called as each clip starts (the loop prints the sentence)
    clip_state(sentence)             the state to publish while that sentence plays; None = always
                                     "speaking". The loop uses it to show "offline" during the
                                     brain's fallback line (M3-07). Called on the producer thread.

    States (M3-06 decision 1, literal): "speaking" while any clip plays, Cogitating included, and
    "thinking" whenever a clip has ended and the next one isn't ready yet.
    """

    def __init__(self, synth: Callable[[str], tuple[np.ndarray, int]] | None, bank: PhraseBank | None, guard,
                 play: Callable[[np.ndarray], None], on_state: Callable[[str], None] = lambda name: None,
                 opener: np.ndarray | None = None, clock: Callable[[], float] = time.monotonic,
                 show: Callable[[ClipLog], None] = lambda clip: None,
                 clip_state: Callable[[str], str] | None = None):
        self.synth = synth
        self.bank = bank
        self.guard = guard
        self.play = play
        self.on_state = on_state
        self.opener = opener
        self.clock = clock
        self.show = show
        self.clip_state = clip_state
        self._state = None

    def speak(self, pieces: Iterable[str], use_opener: bool = True) -> SpeechLog:
        """Sentences from `pieces`, each from the bank or the synth, played in order. Returns the SpeechLog.

        A failed sentence (synth or file) is logged and skipped; the reply goes on. Once the synth
        has failed, the rest of this reply uses bank lines only. An exception from `pieces` itself is
        raised again here, after the guard has opened.
        """
        log = SpeechLog(t_start=self.clock())
        clips: queue.Queue = queue.Queue()        # unbounded: the producer never waits to put
        stop = threading.Event()
        failure: list[BaseException] = []
        producer = threading.Thread(target=self._produce, args=(pieces, clips, stop, log, failure),
                                    name="voice-producer", daemon=True)
        self._state = "thinking"                  # run_turn has just shown "thinking"
        with self.guard.speaking():
            producer.start()                      # the brain starts working now, while the opener plays
            try:
                if use_opener and self.opener is not None:
                    self._play_clip(Clip(self.opener, ClipLog("(cogitating)", "opener", gen_s=0.0,
                                                              audio_s=len(self.opener) / OUT_RATE,
                                                              t_ready=self.clock())), log)
                    item = self._next_clip(clips)
                else:
                    item = _wait_for(clips)               # still "thinking": nothing has played yet
                while item is not _END:
                    self._play_clip(item, log)
                    item = self._next_clip(clips)
            except BaseException:
                stop.set()
                abort = getattr(self.synth, "abort", None)
                if abort is not None:
                    abort()                       # a sentence being made is not waited for
                raise
            finally:
                producer.join()
        if failure:
            log.error = f"{type(failure[0]).__name__}: {failure[0]}"
            raise failure[0]
        return log

    def say_fixed(self, text: str, state: str = "speaking") -> SpeechLog:
        """One fixed line (the bank first, else the synth), no Cogitating, `state` published while it plays.

        M3-07 uses it for the offline line: the brain is never started, so there is nothing to wait for.
        The guard is shut while it plays, as for every reply.
        """
        log = SpeechLog(t_start=self.clock())
        self._state = None                        # publish `state` even if it was the last one shown
        clip = self._make_clip(text, self.synth is not None, log)
        clip.state = state
        with self.guard.speaking():
            self._play_clip(clip, log)
        return log

    def _next_clip(self, clips: queue.Queue):
        """The next queued item; if none is ready, show "thinking" and wait for it."""
        try:
            return clips.get_nowait()
        except queue.Empty:
            self._set_state("thinking")   # the next sentence isn't ready (or the end isn't known yet)
            return _wait_for(clips)

    def _set_state(self, name: str) -> None:
        if name != self._state:
            self._state = name
            self.on_state(name)

    def _play_clip(self, clip: Clip, log: SpeechLog) -> None:
        log.clips.append(clip.log)
        self.show(clip.log)
        if clip.samples is None:
            return
        self._set_state(clip.state)
        clip.log.t_play_start = self.clock()
        try:
            self.play(clip.samples)
        except Exception as exc:      # the amp failing must not end the turn; Ctrl+C still does
            clip.log.error = f"playback: {exc}"
        clip.log.t_play_end = self.clock()

    def _produce(self, pieces, clips: queue.Queue, stop: threading.Event, log: SpeechLog,
                 failure: list) -> None:
        """The producer thread: sentences -> Clips on the queue, then _END (always, even after an error)."""
        synth_ok = self.synth is not None
        try:
            for sentence in split_sentences(pieces):
                if stop.is_set():
                    break
                clip = self._make_clip(sentence, synth_ok, log)
                if self.clip_state is not None:
                    clip.state = self.clip_state(sentence)
                if clip.synth_failed:
                    synth_ok = False
                clips.put(clip)
        except BaseException as exc:      # handed to the playing thread, which raises it
            failure.append(exc)
        finally:
            clips.put(_END)

    def _make_clip(self, sentence: str, synth_ok: bool, log: SpeechLog) -> Clip:
        t0 = self.clock()
        clip_log = ClipLog(sentence, "none")
        samples = None
        synth_failed = False
        path = self.bank.lookup(sentence) if self.bank is not None else None
        if path is not None:
            try:
                samples = load_wav(path)[:, :1]
                clip_log.source = "bank"
            except ValueError as exc:
                clip_log.error = f"bank: {exc}"
        if samples is None:
            log.misses += 1
            if self.bank is not None:
                try:
                    self.bank.note_miss(sentence)
                except (OSError, ValueError) as exc:
                    print(f"Could not count the bank miss: {exc}", file=sys.stderr, flush=True)
            if not synth_ok:
                clip_log.error = clip_log.error or "skipped: no synth for this reply"
            else:
                try:
                    raw, rate = self.synth(sentence)
                    samples = to_48k(raw, rate)
                    clip_log.source = SOURCE_SYNTH
                    clip_log.error = None
                except Exception as exc:
                    clip_log.error = f"synth: {exc}"
                    synth_failed = True
        if samples is not None:
            samples = cap_peak(samples)
            clip_log.audio_s = len(samples) / OUT_RATE
            clip_log.peak_dbfs = peak_dbfs(samples)
        clip_log.gen_s = self.clock() - t0
        clip_log.t_ready = self.clock()
        return Clip(samples, clip_log, synth_failed)


# --- wiring shared with the voice loop -----------------------------------------------------------

def make_player(device):
    """sounddevice playback at 48 kHz on `device` (a name part or an index). Imports sounddevice here only."""
    import sounddevice as sd

    def play(samples: np.ndarray) -> None:
        sd.play(samples, samplerate=OUT_RATE, device=device, blocking=True)

    return play


def load_opener(voice_dir: Path) -> np.ndarray | None:
    """The "Cogitating" clip, or None (reported, not fatal) if it is missing or unreadable."""
    path = Path(voice_dir) / OPENER_FILE
    try:
        return cap_peak(load_wav(path)[:, :1])
    except ValueError as exc:
        print(f"No Cogitating clip, replies start without it: {exc}", file=sys.stderr, flush=True)
        return None


def show_clip(clip: ClipLog) -> None:
    """The loop's screen line for a clip as it starts."""
    if clip.source == "opener":
        print(f"  [cogitating {clip.audio_s:.2f} s]", flush=True)
        return
    gen = "-" if clip.gen_s is None else f"{clip.gen_s:.2f} s"
    extra = f" ERROR {clip.error}" if clip.error else ""
    print(f"  [{clip.source} gen {gen} audio {clip.audio_s:.2f} s] {clip.text}{extra}", flush=True)


def build_bank(lines: list[str], bank: PhraseBank, synth, out=print) -> tuple[int, int]:
    """Generate every line the bank doesn't have yet. Returns (made, already there)."""
    made = present = 0
    for line in lines:
        if bank.lookup(line) is not None:
            present += 1
            out(f"in bank: {line}")
            continue
        t0 = time.monotonic()
        raw, rate = synth(line)
        samples = cap_peak(to_48k(raw, rate))
        path = bank.add(line, samples)
        made += 1
        out(f"made in {time.monotonic() - t0:.2f} s ({len(samples) / OUT_RATE:.2f} s of audio, "
            f"peak {peak_dbfs(samples):.2f} dBFS): {line} -> {path.name}")
    return made, present


def say_summary(record: dict) -> str:
    gaps = record["gaps_s"]
    return (f"first audio {record['first_audio_s']} s after start | sentences {record['sentences']} | "
            f"bank hits {record['bank_hits']} | gaps {gaps if gaps else '-'} | error {record['error']}")


def parse_args(argv=None) -> argparse.Namespace:
    int_or_name = lambda value: int(value) if value.isdigit() else value  # noqa: E731
    parser = argparse.ArgumentParser(description="Voice out: build the phrase bank, or speak a text")
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--build-bank", type=Path, metavar="FILE", help="generate the lines of FILE missing from the bank")
    action.add_argument("--say", metavar="TEXT", help="speak TEXT through the amp")
    action.add_argument("--misses", type=int, metavar="N", help="list the N sentences generated most often")
    parser.add_argument("--bank", type=Path, default=DEFAULT_BANK_DIR)
    parser.add_argument("--tts-python", type=Path, default=DEFAULT_TTS_PYTHON, help="python of the venv with pocket-tts")
    parser.add_argument("--voice", default=DEFAULT_VOICE)
    parser.add_argument("--device", type=int_or_name, default="MAX98357A", help="output: sounddevice name or index")
    parser.add_argument("--voice-dir", type=Path, default=VOICE_DIR, help="folder with thinking.wav (--opener)")
    parser.add_argument("--opener", action="store_true", help="with --say: play Cogitating first, as the loop does")
    parser.add_argument("--repeat", type=int, default=1, help="with --say: speak it this many times")
    parser.add_argument("--log", type=Path, default=DEFAULT_SAY_LOG, help="with --say: JSON-lines log (appended)")
    args = parser.parse_args(argv)
    if args.repeat < 1:
        parser.error("--repeat must be >= 1")
    if args.misses is not None and args.misses < 1:
        parser.error("--misses needs N >= 1")
    return args


def main(argv=None) -> int:
    args = parse_args(argv)
    try:
        bank = PhraseBank(args.bank.expanduser(), bank_voice(args.voice))
    except ValueError as exc:
        print(exc, file=sys.stderr)
        return 1
    if args.misses is not None:
        for count, text in bank.top_misses(args.misses):
            print(f"{count:5d}  {text}")
        return 0

    lines = None
    if args.build_bank is not None:
        try:
            lines = read_bank_lines(args.build_bank)
        except OSError as exc:
            print(f"Could not read {args.build_bank}: {exc}", file=sys.stderr)
            return 1

    synth = WorkerSynth(worker_argv(args.tts_python.expanduser(), args.voice))
    try:
        info = synth.wait_ready()
        print(f"TTS worker ready: {info.get('engine')}, voice {info.get('voice')}, {info.get('rate')} Hz, "
              f"loaded in {info.get('load_s')} s", flush=True)
        if lines is not None:
            made, present = build_bank(lines, bank, synth, out=lambda text: print(text, flush=True))
            print(f"Bank {bank.folder}: made {made}, already there {present}", flush=True)
            return 0

        speaker = Speaker(synth, bank, HalfDuplexGuard(), make_player(args.device),
                          on_state=lambda name: print(f"[{name}]", flush=True),
                          opener=load_opener(args.voice_dir) if args.opener else None, show=show_clip)
        log_path = args.log.expanduser()
        for _ in range(args.repeat):
            speech = speaker.speak([args.say], use_opener=args.opener)
            record = {"started_at": datetime.now().astimezone().isoformat(timespec="seconds"),
                      "text": args.say, **speech.as_record(speech.t_start)}
            print(say_summary(record), flush=True)
            try:
                log_path.parent.mkdir(parents=True, exist_ok=True)
                with open(log_path, "a", encoding="utf-8") as log:
                    log.write(json.dumps(record) + "\n")
            except OSError as exc:
                print(f"Could not write {log_path}: {exc}", file=sys.stderr, flush=True)
        return 0
    except SynthError as exc:
        print(exc, file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        print("", flush=True)
        return 1
    finally:
        synth.close()


if __name__ == "__main__":
    sys.exit(main())
