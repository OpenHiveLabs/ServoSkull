"""Show the skull's voice state on a status LED (three bench LEDs or one RGB LED).

Usage:  python software/skull_core/status_leds.py --test
        python software/skull_core/status_leds.py --demo thinking [--seconds 6]
        python software/skull_core/status_leds.py --demo all
        python software/skull_core/status_leds.py --interactive
Options: --pins 5 6 13 (R G B, BCM numbers)  --common-anode  --blink-period 1.0
"""
import argparse
import sys
import time

Color = tuple[float, float, float]
Frame = tuple[Color, float]  # (colour, seconds)

OFF: Color = (0, 0, 0)
RED: Color = (1, 0, 0)
GREEN: Color = (0, 1, 0)
BLUE: Color = (0, 0, 1)

DEFAULT_BLINK_PERIOD_S = 1.0  # SUGGESTION: 0.5 s on + 0.5 s off. Tune it on the bench.
CYCLE_STEP_S = 0.5            # thinking: each colour for 0.5 s (Ersin, 2026-09-29)

# The one table. Keys are the exact payloads M3 will publish on skull/skull-01/state/voice.
# Value: (kind, colours). kind is "solid", "blink" or "cycle".
STATES: dict[str, tuple[str, tuple[Color, ...]]] = {
    "idle": ("solid", (GREEN,)),
    "thinking": ("cycle", (RED, GREEN, BLUE)),
    "listening": ("blink", (BLUE,)),
    "offline": ("solid", (RED,)),
    "error": ("blink", (RED,)),
    "speaking": ("solid", (BLUE,)),
}


def pattern_for(state: str, blink_period_s: float = DEFAULT_BLINK_PERIOD_S) -> list[Frame]:
    """Return one period of the state's pattern as (colour, seconds) frames.

    solid → one frame of that colour (its length doesn't matter; use blink_period_s)
    blink → colour for half the period, then OFF for the other half
    cycle → each colour for CYCLE_STEP_S, in order
    An unknown state returns the "error" pattern: a status light must not crash on a bad message.
    Raise ValueError if blink_period_s <= 0.
    """
    
    if blink_period_s <= 0:
        raise ValueError("blink_period_s must be positive")

    if state not in STATES:
        state = "error"

    kind, colors = STATES[state]

    if kind == "solid":
        return [(colors[0], blink_period_s)]
    elif kind == "blink":
        half_period = blink_period_s / 2
        return [(colors[0], half_period), (OFF, half_period)]
    elif kind == "cycle":
        frames = []
        for color in colors:
            frames.append((color, CYCLE_STEP_S))
        return frames
    else:
        raise ValueError(f"Unknown state kind: {kind}")


def color_at(frames: list[Frame], t: float) -> Color:
    """The colour shown t seconds after the pattern started. The pattern repeats forever.

    Each frame covers [start, end): at exactly t = 0.5 s the thinking cycle is already GREEN.
    Raise ValueError if t < 0.
    """
    if t < 0:
        raise ValueError("t must be non-negative")

    period = sum(duration for _, duration in frames)
    if period <= 0:
        raise ValueError("frames must have a positive total duration")
    t %= period

    end = 0.0
    for color, duration in frames:
        end += duration
        if t < end:
            return color

    raise ValueError("frames must have a positive total duration")


QUIT = "quit"
TEST = "test"              # a console command, NOT a state: never add it to STATES (ADR-006 contract)
TEST_SHOWS = "speaking"    # the pattern the test command shows


def parse_command(line: str) -> str | None:
    """Turn one typed line into a state name, QUIT, TEST, or None.

    Case and surrounding whitespace don't matter: "  Thinking " -> "thinking", "QUIT" -> QUIT, "Test" -> TEST.
    Anything else returns None, an empty line included. The caller then lists the valid states and
    KEEPS the current one. Not "error": a typo is not a fault of the skull (pattern_for's fallback
    is for messages, not for the keyboard).
    Ctrl-D is not a line: input() raises EOFError. Handle that in main, not here.
    """
    # TODO 11: normalise the line, then check it against STATES, QUIT and TEST
    raise NotImplementedError


class StatusLight:
    """The skull's status light. set_state() is the ONLY way to change what it shows.

    Callers today: --demo and --interactive (and M2-07's state console). In M3, a subscriber on
    skull/skull-01/state/voice (ADR-006) calls the same set_state() with the message payload.

    led:   anything with a .value attribute: gpiozero's RGBLED on the Pi, a fake in the tests, or None.
    clock: returns seconds. time.monotonic on the Pi; the tests pass a fake they can move by hand.
    """

    def __init__(self, led=None, blink_period_s: float = DEFAULT_BLINK_PERIOD_S,
                 clock=time.monotonic, initial_state: str = "idle") -> None:
        # TODO 12: store led, blink_period_s and clock, then go through set_state(initial_state).
        #          (Don't copy set_state's logic here: the constructor uses the entry point too.)
        raise NotImplementedError

    def set_state(self, state: str) -> None:
        """Show `state` from now on. Afterwards self.state is the name actually shown.

        - A new state: build its frames with pattern_for and restart the pattern clock.
        - The SAME state again: change nothing. The clock keeps running, so a re-published
          "thinking" doesn't freeze the cycle on RED.
        - An unknown state: show "error", the same fallback as pattern_for.
        """
        # TODO 13: resolve unknown names to "error" first, THEN compare with the current state
        # TODO 14: store the new frames and the start time (think about the ticker thread reading them)
        raise NotImplementedError

    def color(self) -> Color:
        """The colour to show right now: color_at(frames, clock() - start)."""
        # TODO 15
        raise NotImplementedError

    def update(self) -> Color:
        """Push color() to the LED (if there is one) and return it. The ticker calls this every tick."""
        # TODO 16
        raise NotImplementedError


def main() -> int:
    parser = argparse.ArgumentParser(description="Status LED demo")
    parser.add_argument("--demo", required=True, help="a state name or 'all'")
    parser.add_argument("--seconds", type=float, default=4)          # float, default 4
    parser.add_argument("--pins", nargs=3, type=int, default=[5, 6, 13])             # 3 ints: nargs=3, type=int, default [5, 6, 13]
    parser.add_argument("--blink-period", type=float, default=DEFAULT_BLINK_PERIOD_S)     # float, default DEFAULT_BLINK_PERIOD_S
    args = parser.parse_args()
    
    if args.demo != "all" and args.demo not in STATES:
        parser.error(f"Invalid state: {args.demo}")
    
    names = list(STATES) if args.demo == "all" else [args.demo]
    
    from gpiozero import RGBLED
    led = RGBLED(*args.pins)
    try:
        for name in names:
            print(name)
            frames = pattern_for(name, args.blink_period)
            start = time.monotonic()
            while time.monotonic() - start < args.seconds:  # time since start < args.seconds
                led.value = color_at(frames, time.monotonic() - start)  # color_at(frames, ?)
                time.sleep(0.02)
    finally:
        led.off()
        led.close()
    return 0
if __name__ == "__main__":
    sys.exit(main())