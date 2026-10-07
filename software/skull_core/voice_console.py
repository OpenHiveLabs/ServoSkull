"""Voice state console: press Enter, say a state name; the LED shows it and the skull says its word.

Push-to-talk: Enter -> record a short window -> recognise (Vosk, offline) -> act.
The recogniser only knows the six state names (plus "[unk]" for anything else).
Half-duplex: the mic is never opened while the skull is speaking, or just after.

Usage (from the repo root; -m is required):
    python -m software.skull_core.voice_console [--model-dir ~/models/vosk-model-small-en-us-0.15]
           [--seconds 2.0] [--mic-device MIC] [--device MAX98357A] [--voice-dir software/skull_core/voice]
           [--volume 1.0] [--tail 0.3] [--pins 5 6 13] [--common-anode] [--blink-period 1.0]
"""
import argparse
import contextlib
import json
import math
import sys
import threading
import time
from pathlib import Path

from software.skull_core.status_leds import (
    DEFAULT_BLINK_PERIOD_S,
    QUIT,
    STATES,
    StatusLight,
    parse_command,
    run_ticker,
)
from software.skull_core.state_console import (
    DEFAULT_VOLUME,
    MAX_VOLUME,
    VOICE_DIR,
    WAV_RATE,
    load_wav,
    missing_wavs,
    wav_path,
)

# Kept outside the repo: the model is ~40 MB and isn't ours to publish.
DEFAULT_MODEL_DIR = Path.home() / "models" / "vosk-model-small-en-us-0.15"
MODEL_URL = "https://alphacephei.com/vosk/models/vosk-model-small-en-us-0.15.zip"
UNKNOWN = "[unk]"            # Vosk's catch-all word for a grammar recogniser
DEFAULT_LISTEN_S = 2.0       # push-to-talk window; one word fits easily. Tune on the bench.
DEFAULT_TAIL_S = 0.3         # mic stays closed this long after playback ends (amp/room tail). Estimate.
LISTENING = "listening"      # LED while the mic is open
THINKING = "thinking"        # LED while Vosk decodes


def grammar_json() -> str:
    """The Vosk grammar: a JSON list of the six state names, in STATES order, then UNKNOWN.

    Only STATES. "test" and "quit" are console commands, not things you can say.
    """
    # TODO 1: one line with json.dumps. Build the list from STATES, don't type the words again.
    raise NotImplementedError("TODO 1")


def parse_result(result_json: str) -> str | None:
    """Turn a Vosk result like '{"text": "thinking"}' into a state name, or None.

    None for: empty text, UNKNOWN, two or more words, a console command ("quit", "test"),
    and malformed JSON or a missing "text" key (a bad message must not crash the console).
    """
    # TODO 2: json.loads inside try/except, get "text" with a default, then reuse parse_command()
    #         from M2-06. parse_command also accepts "quit" and "test": only a name in STATES counts.
    raise NotImplementedError("TODO 2")


def recognise(recognizer, samples) -> str | None:
    """Feed one recorded window (int16 numpy array) to the recogniser, return a state or None.

    recognizer is a vosk.KaldiRecognizer (a fake in the tests). AcceptWaveform wants bytes;
    FinalResult() returns the JSON text for everything fed so far and resets it for the next turn.
    """
    # TODO 3: AcceptWaveform(<the samples as bytes>), then FinalResult(), then parse_result(). 2-3 lines.
    raise NotImplementedError("TODO 3")


class HalfDuplexGuard:
    """Keeps the mic shut while the skull speaks, and for tail_s after, so it can't hear itself.

    clock: returns seconds. time.monotonic on the Pi; the tests pass a fake they move by hand.
    """

    def __init__(self, tail_s: float = DEFAULT_TAIL_S, clock=time.monotonic):
        if tail_s < 0:
            raise ValueError("tail_s must be >= 0")
        self.tail_s = tail_s
        self.clock = clock
        self.is_speaking = False
        self._open_at = -math.inf   # the mic starts open

    def start_speaking(self) -> None:
        self.is_speaking = True

    def stop_speaking(self) -> None:
        """Playback has ended: the mic opens again tail_s from now."""
        # TODO 4: two lines. Clear the flag; set self._open_at from self.clock() (not time.monotonic).
        raise NotImplementedError("TODO 4")

    def seconds_until_open(self) -> float:
        """0.0 when the mic may listen now. While speaking: math.inf. After: the time left of the tail.

        Same convention as color_at in M2-06: at exactly _open_at the mic is already open.
        """
        # TODO 5: three cases, never negative.
        raise NotImplementedError("TODO 5")

    def mic_open(self) -> bool:
        return self.seconds_until_open() == 0.0

    @contextlib.contextmanager
    def speaking(self):
        """with guard.speaking(): play(...)  -- the mic reopens (after the tail) even if play raises."""
        self.start_speaking()
        try:
            yield
        finally:
            self.stop_speaking()


def push_to_talk_turn(light: StatusLight, guard: HalfDuplexGuard, listen, say, sleep=time.sleep) -> str | None:
    """One Enter press. Returns the state that was acted on, or None.

    listen() records a window and returns a state name or None (main builds it from record + recognise).
    say(state) plays that state's WAV and blocks until it's done.

    1. If the mic isn't open yet, sleep(guard.seconds_until_open()) first. Never call listen() while it's shut.
    2. Remember light.state, show LISTENING, and call listen().
    3. None: put the light back to the remembered state and return None. Say nothing.
    4. A state: show it, then call say(state) INSIDE `with guard.speaking():`, and return the state.
    """
    # TODO 6: about 8 lines, steps 1-4 above. (THINKING while Vosk decodes is done in main's listen().)
    raise NotImplementedError("TODO 6")


def parse_args(argv=None) -> argparse.Namespace:
    int_or_name = lambda value: int(value) if value.isdigit() else value  # noqa: E731
    parser = argparse.ArgumentParser(description="Voice state console (push-to-talk, Vosk, offline)")
    parser.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL_DIR, help="unzipped Vosk model folder")
    parser.add_argument("--seconds", type=float, default=DEFAULT_LISTEN_S, help="push-to-talk window")
    parser.add_argument("--mic-device", type=int_or_name, default=None, help="input: sounddevice name or index")
    parser.add_argument("--device", type=int_or_name, default="MAX98357A", help="output: sounddevice name or index")
    parser.add_argument("--voice-dir", type=Path, default=VOICE_DIR)
    parser.add_argument("--volume", type=float, default=DEFAULT_VOLUME)
    parser.add_argument("--tail", type=float, default=DEFAULT_TAIL_S, help="mic stays shut this long after speaking")
    parser.add_argument("--pins", nargs=3, type=int, default=[5, 6, 13], metavar=("R", "G", "B"))
    parser.add_argument("--common-anode", action="store_true")
    parser.add_argument("--blink-period", type=float, default=DEFAULT_BLINK_PERIOD_S)
    args = parser.parse_args(argv)

    if not (math.isfinite(args.seconds) and args.seconds > 0):
        parser.error("--seconds must be a finite value greater than 0")
    if not 0 <= args.volume <= MAX_VOLUME:
        parser.error(f"--volume must be between 0 and {MAX_VOLUME}")
    if not (math.isfinite(args.tail) and args.tail >= 0):
        parser.error("--tail must be >= 0")
    if args.blink_period <= 0:
        parser.error("--blink-period must be positive")
    return args


def main(argv=None) -> int:
    args = parse_args(argv)

    if not (args.model_dir / "conf" / "model.conf").exists():
        print(f"Vosk model not found in {args.model_dir}", file=sys.stderr)
        print(f"Download and unzip {MODEL_URL} there (see the M2-08 install steps),", file=sys.stderr)
        print("or pass --model-dir.", file=sys.stderr)
        return 1

    missing = missing_wavs(args.voice_dir)
    if missing:
        print("Missing required WAV files:", file=sys.stderr)
        for path in missing:
            print(f"  {path}", file=sys.stderr)
        return 1
    try:
        sounds = {state: load_wav(wav_path(state, args.voice_dir)) * args.volume for state in STATES}
    except ValueError as exc:
        print(exc, file=sys.stderr)
        return 1

    # Hardware and model imports only here: importing this file (the tests, --help) needs none of them.
    import sounddevice as sd
    import vosk
    from gpiozero import RGBLED

    from software.skull_core.record import SAMPLE_RATE, record

    vosk.SetLogLevel(-1)
    print(f"Loading {args.model_dir} ...")
    recognizer = vosk.KaldiRecognizer(vosk.Model(str(args.model_dir)), SAMPLE_RATE, grammar_json())

    led = RGBLED(*args.pins, active_high=not args.common_anode)
    light = StatusLight(led, blink_period_s=args.blink_period)
    guard = HalfDuplexGuard(args.tail)
    stop = threading.Event()
    ticker = threading.Thread(target=run_ticker, args=(light, stop), daemon=True)

    def listen():
        samples = record(args.seconds, device=args.mic_device)
        light.set_state(THINKING)
        return recognise(recognizer, samples)

    def say(state):
        try:
            sd.play(sounds[state], samplerate=WAV_RATE, device=args.device, blocking=True)
        except sd.PortAudioError as exc:
            print(f"Audio playback failed: {exc}")

    try:
        ticker.start()
        print(f"Press Enter, then say one of: {', '.join(STATES)}.  Type quit to leave.")
        while True:
            try:
                line = input("> ")
            except EOFError:
                break
            if parse_command(line) == QUIT:
                break
            print(f"Listening for {args.seconds:g} s ...")
            state = push_to_talk_turn(light, guard, listen, say)
            if state is None:
                print("Didn't catch a state name.")
            else:
                print(f"Heard: {state}")
    except KeyboardInterrupt:
        print()
    finally:
        try:
            sd.stop()
        finally:
            stop.set()
            if ticker.ident is not None:
                ticker.join()
            led.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
