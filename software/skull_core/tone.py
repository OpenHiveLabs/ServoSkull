"""Play a sine test tone through the skull speaker.

Usage:  python tools/bringup/tone.py [--freq 440] [--seconds 1.0] [--volume 0.2] [--device NAME_OR_INDEX]
"""
import argparse
import numpy as np
import sounddevice as sd

SAMPLE_RATE = 48_000  # Hz


def sine(freq_hz: float, seconds: float, volume: float, rate: int = SAMPLE_RATE) -> np.ndarray:

    t = np.arange(int(seconds * rate)) / rate  # time axis
    wave = volume * np.sin(2 * np.pi * freq_hz * t)  # sine wave
    fade_samples = int(0.001 * rate)  # 10 ms fade
    fade_in = np.linspace(0, 1, fade_samples)  # fade in
    fade_out = np.linspace(1, 0, fade_samples)  # fade out
    wave[:fade_samples] *= fade_in  # apply fade in
    wave[-fade_samples:] *= fade_out  # apply fade out
    return wave.astype(np.float32)  # return as float32

def main() -> int:
    argument_parser = argparse.ArgumentParser(description="Play a sine test tone through the skull speaker.")
    argument_parser.add_argument("--freq", type=float, default=440, help="Frequency in Hz")
    argument_parser.add_argument("--seconds", type=float, default=1.0, help="Duration in seconds")
    argument_parser.add_argument("--volume", type=float, default=0.2, help="Volume (0..1)")
    argument_parser.add_argument("--device", type=lambda x: int(x) if x.isdigit() else x, help="Device name or index")
    args = argument_parser.parse_args()

    if args.volume > 0.5 or np.isnan(args.volume):
        print("Volume is too high or Not a Number! Please use a value between 0 and 0.5.")
        return 1
    
    if args.device == "list":
        print("Available devices:")
        for i, info in enumerate(sd.query_devices()):
            print(f"  {i}: {info['name']}")
        return 0

    wave = sine(args.freq, args.seconds, args.volume)
    sd.play(wave, samplerate=SAMPLE_RATE,device=args.device)
    sd.wait()
    return 0

if __name__ == "__main__":
    raise SystemExit(main())