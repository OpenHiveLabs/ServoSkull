"""Text-to-speech worker: a separate process that holds the Pocket TTS model and turns sentences into audio.

Why a separate process: Pocket TTS needs torch, and the voice loop runs in the Moonshine venv.
Putting both in one venv has never been tried on the Pi (M3-05 decision 6). The worker runs in
its own venv (on the Pi: ~/m3-bench/.venv-tts) and the loop talks to it through two pipes.

The pipe protocol (one request at a time, the loop waits for the answer before sending the next):
  start-up    worker -> loop: one JSON line, {"ready": true, "rate": 24000, "engine": "...", "voice": "...",
              "load_s": 13.7} or {"ready": false, "error": "..."} (then the worker exits).
  request     loop -> worker: one JSON line, {"text": "One sentence."}
  answer      worker -> loop: one JSON line, then the audio:
              {"ok": true, "frames": N, "rate": 24000, "gen_s": 1.23} followed by exactly N * 2 bytes,
              N mono samples as 16-bit little-endian integers; or {"ok": false, "error": "..."}, no bytes.
  end         the loop closes the worker's stdin; the worker exits.
The JSON line says how many bytes follow, so the reader always knows where the audio ends.

stdout carries the protocol and nothing else. Libraries print things; at start-up the worker moves
everything printed to stdout over to stderr (see protect_stdout), so a stray print can't break a message.

Usage (from the repo root, with the venv that has pocket-tts; the loop starts it for you):
    ~/m3-bench/.venv-tts/bin/python -u -m software.skull_core.tts_worker [--voice javert]
Only the standard library and numpy are imported at the top: the loop imports this file for the
protocol constants without needing torch.
"""
import argparse
import json
import os
import signal
import sys
import time
from typing import BinaryIO, TextIO

import numpy as np

DEFAULT_VOICE = "javert"        # CC0 voice (ADR-013); never cosette or jean (NC licences)
WARMUP_TEXT = "Ready."          # generated once after loading and thrown away, as in M3-00 (g)
SAMPLE_BYTES = 2                # int16 on the pipe
MAX_TEXT_CHARS = 2000           # a request longer than this is refused (a runaway reply, not a sentence)


def protect_stdout() -> BinaryIO:
    """Keep the real stdout for the protocol and send everything else that prints to stderr.

    os.dup(1) makes a second handle on the pipe to the loop; os.dup2(2, 1) then points file
    descriptor 1 (where print and C libraries write) at stderr. Only the returned file reaches the loop.
    """
    sys.stdout.flush()
    protocol_fd = os.dup(1)
    os.dup2(2, 1)
    sys.stdout = sys.stderr
    return os.fdopen(protocol_fd, "wb", buffering=0)


def send_json(out: BinaryIO, message: dict) -> None:
    out.write((json.dumps(message) + "\n").encode("utf-8"))
    out.flush()


def to_int16(samples: np.ndarray) -> np.ndarray:
    """Float audio in -1..1 -> int16, clipped first so a loud sample can't wrap around."""
    flat = np.asarray(samples, dtype=np.float32).reshape(-1)
    return (np.clip(flat, -1.0, 1.0) * 32767.0).astype("<i2")


def send_audio(out: BinaryIO, samples: np.ndarray, rate: int, gen_s: float) -> None:
    pcm = to_int16(samples)
    send_json(out, {"ok": True, "frames": int(len(pcm)), "rate": int(rate), "gen_s": round(gen_s, 3)})
    out.write(pcm.tobytes())
    out.flush()


def serve(engine, requests: TextIO, out: BinaryIO, clock=time.perf_counter) -> None:
    """Answer requests until stdin closes. A bad request or a failed sentence gets an error answer;
    the worker keeps running. engine has .rate and .generate(text) -> float samples."""
    for line in requests:
        if not line.strip():
            continue
        try:
            text = json.loads(line)["text"]
            if not isinstance(text, str) or not text.strip() or len(text) > MAX_TEXT_CHARS:
                raise ValueError("text must be a non-empty string of at most "
                                 f"{MAX_TEXT_CHARS} characters")
        except (ValueError, KeyError, TypeError) as exc:
            send_json(out, {"ok": False, "error": f"bad request: {exc}"})
            continue
        t0 = clock()
        try:
            samples = engine.generate(text)
        except Exception as exc:   # one failed sentence must not end the worker
            send_json(out, {"ok": False, "error": f"{type(exc).__name__}: {exc}"})
            continue
        send_audio(out, samples, engine.rate, clock() - t0)


class PocketEngine:
    """Pocket TTS, loaded once. generate_audio_stream is not thread-safe: this process has one thread."""

    def __init__(self, voice: str):
        from importlib.metadata import version

        from pocket_tts import TTSModel

        self.model = TTSModel.load_model()
        self.voice_state = self.model.get_state_for_audio_prompt(voice)
        self.rate = int(self.model.sample_rate)
        self.name = f"pocket-tts {version('pocket-tts')}"

    def generate(self, text: str) -> np.ndarray:
        chunks = [chunk.numpy().reshape(-1) for chunk in self.model.generate_audio_stream(self.voice_state, text)]
        return np.concatenate(chunks) if chunks else np.zeros(0, dtype=np.float32)


def run_worker(make_engine, voice: str, warmup: str | None = WARMUP_TEXT) -> int:
    """Shared by the real worker and the test fake: protect stdout, load, say ready, serve."""
    # Ctrl+C in the terminal reaches every process in the group. The loop decides when the
    # worker stops (it closes stdin, or kills it), so the worker ignores SIGINT.
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    out = protect_stdout()
    t0 = time.perf_counter()
    try:
        engine = make_engine(voice)
        if warmup:
            engine.generate(warmup)
    except Exception as exc:
        send_json(out, {"ready": False, "error": f"{type(exc).__name__}: {exc}"})
        return 1
    send_json(out, {"ready": True, "rate": engine.rate, "engine": engine.name, "voice": voice,
                    "load_s": round(time.perf_counter() - t0, 2)})
    try:
        serve(engine, sys.stdin, out)
    except BrokenPipeError:
        pass   # the loop went away; nothing left to answer
    return 0


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Pocket TTS worker: JSON line in, 16-bit audio out (see the docstring)")
    parser.add_argument("--voice", default=DEFAULT_VOICE)
    parser.add_argument("--no-warmup", action="store_true", help="skip the warm-up sentence after loading")
    args = parser.parse_args(argv)
    return run_worker(PocketEngine, args.voice, None if args.no_warmup else WARMUP_TEXT)


if __name__ == "__main__":
    sys.exit(main())
