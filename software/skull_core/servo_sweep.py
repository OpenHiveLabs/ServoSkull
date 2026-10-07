"""Sweep one SG90 servo on a Pi GPIO, in microseconds (software PWM via lgpio).

Usage (from the repo root):
    python -m software.skull_core.servo_sweep [--pin 23] [--chip 0]
           [--pulse-lo 1400] [--pulse-hi 1600] [--step-us 10] [--delay 0.05] [--dry-run]

Temporary bench set-up (M2-05): GPIO23 signal, Pi 5 V power. Moves to PCA9685 ch 12 later.
Works in microseconds only; converting to degrees is a later calibration step.
"""
import argparse
import sys
import time

HARD_MIN_US = 1000  # the clamp: nothing outside this ever reaches the pin
HARD_MAX_US = 2000
CENTRE_US = 1500    # nominal servo centre, only used for the dry-run "offset" column
FRAME_S = 0.020     # 50 Hz servo frame. A delay shorter than one frame just queues pulses.
SERVO_HZ = 50


def clamp_us(us: float, lo: float = HARD_MIN_US, hi: float = HARD_MAX_US) -> float:
    """Keep a pulse width inside [lo, hi] microseconds."""
    # TODO 1: return us, pushed back inside [lo, hi]. One line with min() and max().
    raise NotImplementedError("TODO 1")


def sweep_pulses(lo: int, hi: int, step: int) -> list[int]:
    """lo -> hi -> lo in `step` us increments.

    Both ends are included, even when `step` doesn't divide (hi - lo), and there is
    no duplicate at the turn. The way down mirrors the way up:
        sweep_pulses(1400, 1650, 100) == [1400, 1500, 1600, 1650, 1600, 1500, 1400]
    lo == hi gives [lo]. Raise ValueError if step <= 0 or lo > hi.
    """
    # TODO 2: check the arguments: raise ValueError (with a message) if step <= 0 or lo > hi.
    raise NotImplementedError("TODO 2")
    # TODO 3: build the way up so it ends exactly on hi, then append the way down.
    #         range(lo, hi, step) stops before hi. Watch out for lo == hi.
    raise NotImplementedError("TODO 3")


def to_lgpio_width(us: float) -> int:
    """Turn a pulse width into the argument lgpio.tx_servo wants.

    tx_servo takes a whole number of us (500-2500, or 0 = off). Round to the nearest
    us and clamp to HARD_MIN_US..HARD_MAX_US. This function must never return 0:
    stopping the servo is done on purpose in main(), not by accident here.
    """
    # TODO 4: round, clamp, and make sure the result is an int (not 1500.0).
    raise NotImplementedError("TODO 4")


def check_delay(delay_s: float) -> float:
    """Return delay_s if it's at least one servo frame (FRAME_S), else raise ValueError."""
    # TODO 5: compare against FRAME_S; the error message should say what the minimum is.
    raise NotImplementedError("TODO 5")


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Sweep one SG90 on a Pi GPIO, in microseconds.")
    parser.add_argument("--pin", type=int, default=23, help="BCM GPIO number (default: 23)")
    parser.add_argument("--chip", type=int, default=0, help="gpiochip number; check with gpiodetect (default: 0)")
    parser.add_argument("--pulse-lo", type=int, default=1400, help="low end of the sweep, us (default: 1400)")
    parser.add_argument("--pulse-hi", type=int, default=1600, help="high end of the sweep, us (default: 1600)")
    parser.add_argument("--step-us", type=int, default=10, help="step between pulses, us (default: 10)")
    parser.add_argument("--delay", type=float, default=0.05, help="seconds per step, at least one frame (default: 0.05)")
    parser.add_argument("--dry-run", action="store_true", help="print the pulse table, touch no pins")
    return parser.parse_args(argv)


def print_table(pulses: list[int], delay_s: float) -> None:
    print(f"{'#':>4}  {'pulse us':>8}  {'lgpio':>6}  {'offset':>7}")
    for i, p in enumerate(pulses):
        width = to_lgpio_width(p)
        print(f"{i:>4}  {p:>8}  {width:>6}  {width - CENTRE_US:>+7}")
    print(f"{len(pulses)} pulses, {len(pulses) * delay_s:.2f} s")


def main(argv=None) -> int:
    args = parse_args(argv)

    lo = int(clamp_us(args.pulse_lo))
    hi = int(clamp_us(args.pulse_hi))
    if (lo, hi) != (args.pulse_lo, args.pulse_hi):
        print(f"Warning: range {args.pulse_lo}-{args.pulse_hi} us clamped to {lo}-{hi} us "
              f"(hard limits {HARD_MIN_US}-{HARD_MAX_US})")
    try:
        delay = check_delay(args.delay)
        pulses = sweep_pulses(lo, hi, args.step_us)
    except ValueError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    if args.dry_run:
        print_table(pulses, delay)
        return 0

    import lgpio  # only here: the Surface (and the tests) have no lgpio

    handle = None
    try:
        # TODO 6: open the gpiochip (args.chip) into `handle`, then claim args.pin as an output.
        raise NotImplementedError("TODO 6")
        print(f"Sweeping GPIO{args.pin} on gpiochip{args.chip}: {lo}-{hi} us, "
              f"step {args.step_us} us, {len(pulses)} pulses, {len(pulses) * delay:.2f} s")
        for p in pulses:
            # TODO 7: send this pulse width at SERVO_HZ. Only to_lgpio_width() output may reach the pin.
            raise NotImplementedError("TODO 7")
            time.sleep(delay)
        print("Done.")
    except KeyboardInterrupt:
        print("\nStopped (Ctrl-C).")
    except lgpio.error as exc:
        print(f"lgpio error: {exc}", file=sys.stderr)
        return 1
    finally:
        if handle is not None:
            # TODO 8: make the servo limp (width 0 = no pulses), free the pin, close the chip.
            #         This runs on success, on Ctrl-C and on an error.
            raise NotImplementedError("TODO 8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
