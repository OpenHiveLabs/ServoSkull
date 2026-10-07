"""Record a short clip from the USB mic, save it as WAV, print its level.

Usage:  python software/skull_core/record.py [--seconds 5] [--warmup 3] [--out captures/] [--device MIC_NAME_OR_INDEX]
                                             [--play] [--out-device MAX98357A]
"""
import argparse
from datetime import datetime
import math
import wave
from pathlib import Path
import numpy as np
import sounddevice as sd

SAMPLE_RATE = 48_000  # Hz; the rate the amp path was tested at (M2-03). STT's 16 kHz is M3's resampling job
WARMUP_S = 3.0  # s; the USB mic's firmware gain is loud for the first seconds after the stream opens, then settles
# TODO: set from the 2026-10-07 steady-tone measurement


def record(seconds: float, device=None, rate: int = SAMPLE_RATE, warmup: float = WARMUP_S) -> np.ndarray:
    """Record mono audio and return it as int16 samples.

    The first `warmup` seconds after the stream opens are read and thrown away,
    so the mic's gain has settled before the kept part starts.
    """
    # TODO 1: open ONE sd.InputStream (keyword args: samplerate=rate, channels=1, dtype='int16', device=device)
    #   as a context manager. Inside it, read int(warmup * rate) frames and discard them, then read
    #   int(seconds * rate) frames and return them flattened to 1-D. stream.read(n) returns a pair
    #   (data, overflowed); data has shape (n, 1). warmup=0 must behave like today (nothing discarded).
    #   Your sd.rec version, for reference:
    #     recorded = sd.rec(frames=int(seconds * rate), samplerate=rate, channels=1, dtype='int16', device=device)
    #     sd.wait()
    #     return recorded.flatten()
    raise NotImplementedError("TODO 1")


def rms_dbfs(samples: np.ndarray) -> float:
    """Loudness as dBFS: 0 = full scale, silence is very negative."""
    samples = samples.astype(np.float32)  # convert to float for RMS calculation
    samples /= np.iinfo(np.int16).max  # normalize to -1..1
    rms = np.sqrt(np.mean(samples**2))  # root-mean-square
    if rms == 0:
        rms = 1 / (np.iinfo(np.int16).max + 1)
    log_rms = 20 * np.log10(rms)  # convert to dBFS
    return log_rms


def save_wav(path: Path, samples: np.ndarray, rate: int = SAMPLE_RATE) -> None:
    """Write mono 16-bit WAV."""
    with wave.open(str(path), 'wb') as wf:
        wf.setnchannels(1)  # mono
        wf.setsampwidth(2)  # 16-bit
        wf.setframerate(rate)
        wf.writeframes(samples.tobytes())


def main() -> int:
    argument_parser = argparse.ArgumentParser(description="Record a short clip from the USB mic, save it as WAV, print its level.")
    argument_parser.add_argument("--seconds", type=float, default=5.0, help="Duration of recording in seconds (default: 5)")
    argument_parser.add_argument("--warmup", type=float, default=WARMUP_S, help=f"Seconds read and discarded before recording, while the mic's gain settles (default: {WARMUP_S})")
    argument_parser.add_argument("--out", type=Path, default=Path("captures"), help="Output directory for WAV files (default: captures/)")
    argument_parser.add_argument("--device", type=lambda value: int(value) if value.isdigit() else value, help="Audio device to use (default: first input device)")
    argument_parser.add_argument("--play", action="store_true", help="Play the recorded audio back through the amp")
    argument_parser.add_argument("--out-device", type=lambda value: int(value) if value.isdigit() else value, help="Output audio device to use (default: first output device)")
    args = argument_parser.parse_args() 

    if not math.isfinite(args.seconds) or args.seconds <= 0:
        argument_parser.error("--seconds must be a finite value greater than 0")
    if not math.isfinite(args.warmup) or args.warmup < 0:
        argument_parser.error("--warmup must be a finite value of 0 or more")

    args.out.mkdir(parents=True, exist_ok=True)
    samples = record(args.seconds, device=args.device, warmup=args.warmup)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    output_path = args.out / f"recording_{timestamp}.wav"
    save_wav(output_path, samples)
    level = rms_dbfs(samples)
    print(f"Saved recording to {output_path}")
    print(f"Level: {level:.2f} dBFS")

    if args.play:
        sd.play(samples, samplerate=SAMPLE_RATE, device=args.out_device, blocking=True)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())