"""A stand-in for the Pocket TTS worker, for tests/test_voice_out.py. No torch, no model.

It runs the real protocol code (tts_worker.run_worker and serve) with a fake engine that makes a
quiet 440 Hz tone, 10 ms per character of text, at 24 kHz. What it does is set by environment variables:
    FAKE_TTS_LOAD_FAIL   fail while loading (the ready line says so)
    FAKE_TTS_FAIL_WORD   a sentence containing this word gets an error answer
    FAKE_TTS_DIE_WORD    a sentence containing this word makes the worker exit at once (code 3)
    FAKE_TTS_HANG_WORD   a sentence containing this word makes the worker sleep 60 s
    FAKE_TTS_NOISE       print this to stdout (via print) before every answer: it must not reach the loop
    FAKE_TTS_PIDFILE     write our pid here
"""
import os
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))   # the repo root, so `software.` imports work

from software.skull_core import tts_worker  # noqa: E402

RATE = 24_000


class FakeEngine:
    name = "fake-tts 0"
    rate = RATE

    def __init__(self, voice):
        if os.environ.get("FAKE_TTS_LOAD_FAIL"):
            raise RuntimeError("fake load failure")
        self.voice = voice

    def generate(self, text):
        env = os.environ
        if env.get("FAKE_TTS_NOISE"):
            print(env["FAKE_TTS_NOISE"])           # a chatty library; protect_stdout sends it to stderr
            sys.stdout.flush()
        for key, action in (("FAKE_TTS_DIE_WORD", "die"), ("FAKE_TTS_HANG_WORD", "hang"),
                            ("FAKE_TTS_FAIL_WORD", "fail")):
            word = env.get(key)
            if word and word in text:
                if action == "die":
                    os._exit(3)
                if action == "hang":
                    time.sleep(60)
                raise ValueError(f"cannot say {word!r}")
        frames = int(0.01 * RATE * len(text))
        return 0.25 * np.sin(2 * np.pi * 440 * np.arange(frames) / RATE).astype(np.float32)


if __name__ == "__main__":
    if os.environ.get("FAKE_TTS_PIDFILE"):
        Path(os.environ["FAKE_TTS_PIDFILE"]).write_text(str(os.getpid()))
    sys.exit(tts_worker.run_worker(FakeEngine, "fakevoice", warmup=None))
