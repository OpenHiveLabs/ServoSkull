"""Bench state console: type a state; the status LED shows it and the skull says the word.

A bench stand-in: M3 replaces the WAV files with Piper TTS (ADR-003). set_state() stays.

Usage (from the repo root; -m is required):
    python -m software.skull_core.state_console [--voice-dir software/skull_core/voice] [--device MAX98357A]
           [--volume 1.0] [--common-anode] [--pins 5 6 13] [--blink-period 1.0]
           [--demo STATE|all] [--seconds 3]
"""
import argparse
import sys
import threading
import time
import wave

from pathlib import Path

import numpy as np

from software.skull_core.status_leds import (
    DEFAULT_BLINK_PERIOD_S,
    QUIT,
    STATES,
    TEST,
    TEST_SHOWS,
    StatusLight,
    parse_command,
    run_ticker,
)

VOICE_DIR = Path("software/skull_core/voice")  # relative to the repo root
WAV_RATE = 48_000                 # Hz: the rate the amp path was tested at (M2-03, speaker-test)
WAV_SAMPLE_BYTES = 2              # 16-bit signed PCM (S16_LE)
WAV_CHANNELS = (1, 2)             # mono recommended; PortAudio copies mono onto the stereo pair
DEFAULT_VOLUME = 1.0              # your WAVs already peak at -18 dBFS (measured 2026-10-01): 1.0 = what aplay played
MAX_VOLUME = 1.0                  # never amplify in code; for more level, re-export the WAVs louder


def wav_path(state: str, folder: Path) -> Path:
    if state not in STATES and state != TEST:
        raise ValueError(f"Invalid state: {state}")
    return folder / f"{state}.wav"


def demo_names(choice: str) -> list[str]:
    
    """The states a --demo run shows, in STATES order.

    "all" -> every state in STATES order; a state name -> [that name].
    Anything else ("test", "quit", "", "ERROR") -> ValueError naming the bad choice.
    """
    all_names = list(STATES)
    if choice == "all":
        return all_names
    if choice in all_names:
        return [choice]
    raise ValueError(f"Invalid demo choice: {choice}. Must be one of {all_names} or 'all'.")

def missing_wavs(folder: Path) -> list[Path]:
    """Return missing required WAV paths in state-table order."""
    return [wav_path(state, folder) for state in STATES if not wav_path(state, folder).exists()]


def missing_optional_wavs(folder: Path) -> list[Path]:
    """The OPTIONAL WAV paths that don't exist: [folder / "test.wav"] or [].
    main() only warns about these; they never stop the console.
    """
    state = TEST
    path = wav_path(state, folder)
    if not path.exists():
        return [path]
    return []


def load_wav(path: Path) -> np.ndarray:
    """Read one WAV. Return float32 samples in -1..1, shape (frames, channels).

    Raise ValueError, with the path and what was found, if the file isn't a readable WAV,
    isn't WAV_RATE, isn't 16-bit, or doesn't have 1 or 2 channels.
    Don't resample here: a wrong file is fixed once with ffmpeg, not on every start.
    """
    try:
        with path.open("rb") as source:
            with wave.open(source, "rb") as wave_file:
                rate = wave_file.getframerate()
                sample_width = wave_file.getsampwidth()
                channels = wave_file.getnchannels()
                if rate != WAV_RATE:
                    raise ValueError(f"{path}: wrong sample rate {rate} (expected {WAV_RATE})")
                if sample_width != WAV_SAMPLE_BYTES:
                    raise ValueError(
                        f"{path}: wrong sample width {sample_width} (expected {WAV_SAMPLE_BYTES})"
                    )
                if channels not in WAV_CHANNELS:
                    raise ValueError(f"{path}: wrong channel count {channels} (expected 1 or 2)")
                pcm_bytes = wave_file.readframes(wave_file.getnframes())
    except (wave.Error, EOFError, OSError) as exc:
        raise ValueError(f"{path}: unreadable WAV ({exc})") from exc

    bytes_per_frame = WAV_SAMPLE_BYTES * channels
    if len(pcm_bytes) % bytes_per_frame:
        raise ValueError(f"{path}: incomplete PCM frame ({len(pcm_bytes)} audio bytes)")

    samples = np.frombuffer(pcm_bytes, dtype="<i2").astype(np.float32)
    return (samples / 32768.0).reshape(-1, channels)


def main() -> int:
    parser = argparse.ArgumentParser(description="Bench state console")
    parser.add_argument("--demo", help="a state name or 'all' (default: the prompt)")
    parser.add_argument("--seconds", type=float, default=3.0, help="how long each demo state holds")
    parser.add_argument("--voice-dir", type=Path, default=VOICE_DIR)
    parser.add_argument(
        "--device",
        type=lambda value: int(value) if value.isdigit() else value,
        default="MAX98357A",
        help="sounddevice name or numeric index",
    )
    parser.add_argument("--volume", type=float, default=DEFAULT_VOLUME)
    parser.add_argument("--common-anode", action="store_true")
    parser.add_argument("--pins", nargs=3, type=int, default=[5, 6, 13], metavar=("R", "G", "B"))
    parser.add_argument("--blink-period", type=float, default=DEFAULT_BLINK_PERIOD_S)
    args = parser.parse_args()
    
    names = None
    if args.demo is not None:
        try:
            names = demo_names(args.demo)
        except ValueError as exc:
            parser.error(str(exc))

    if not 0 <= args.volume <= MAX_VOLUME:
        parser.error(f"--volume must be between 0 and {MAX_VOLUME}")
    if args.seconds <= 0:
        parser.error("--seconds must be positive")
    if args.blink_period <= 0:
        parser.error("--blink-period must be positive")

    missing = missing_wavs(args.voice_dir)
    if missing:
        print("Missing required WAV files:", file=sys.stderr)
        for path in missing:
            print(f"  {path}", file=sys.stderr)
        print(
            "Convert raw/<state>.* to 48000 Hz, mono, 16-bit PCM with ffmpeg; "
            "see software/skull_core/voice/README.md.",
            file=sys.stderr,
        )
        return 1

    optional_missing = missing_optional_wavs(args.voice_dir)
    if optional_missing:
        print(f"Warning: optional WAV missing: {optional_missing[0]}; test audio is disabled.")

    try:
        sounds = {
            state: load_wav(wav_path(state, args.voice_dir)) * args.volume
            for state in STATES
        }
        test_path = wav_path(TEST, args.voice_dir)
        if not optional_missing:
            sounds[TEST] = load_wav(test_path) * args.volume
    except ValueError as exc:
        print(exc, file=sys.stderr)
        return 1
    
    import sounddevice as sd
    from gpiozero import RGBLED

    led = RGBLED(*args.pins, active_high=not args.common_anode)
    light = StatusLight(led, blink_period_s=args.blink_period)
    stop = threading.Event()
    ticker = threading.Thread(target=run_ticker, args=(light, stop), daemon=True)

    try:
        ticker.start()
        if names is not None:
            for state in names:
                print(f"Showing state '{state}' for {args.seconds} seconds...")
                light.set_state(state)
                try:
                    sd.play(sounds[state], samplerate=WAV_RATE, device=args.device)
                except sd.PortAudioError as exc:
                    print(f"Audio playback failed: {exc}")
                time.sleep(args.seconds)
        else:
            while True:
                try:
                    line = input("> ")
                except EOFError:
                    break
                command = parse_command(line)
                if command is None:
                    print("Unknown command. Enter a state, 'test', or 'quit'.")
                    continue
                if command == QUIT:
                    break

                if command == TEST:
                    light.set_state(TEST_SHOWS)
                    if TEST not in sounds:
                        print(f"Optional test WAV is missing: {test_path}")
                        continue
                    sound = sounds[TEST]
                else:
                    light.set_state(command)
                    sound = sounds[command]

                try:
                    sd.play(sound, samplerate=WAV_RATE, device=args.device, blocking=True)
                except sd.PortAudioError as exc:
                    print(f"Audio playback failed: {exc}")
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