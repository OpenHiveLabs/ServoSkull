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
    word = line.strip().lower()        # two string methods: trim the ends, then lower case
    if word in STATES:                     # the six states
        return word
    if word in ("quit", "test"):              # the two console commands (use the constants, not strings)
        return word
    return None

class StatusLight:
    """The skull's status light. set_state() is the ONLY way to change what it shows.

    led:   anything with a .value attribute: gpiozero's RGBLED on the Pi, a fake in the tests, or None.
    clock: returns seconds. time.monotonic on the Pi; the tests pass a fake they can move by hand.
    """

    def __init__(self, led=None, blink_period_s=DEFAULT_BLINK_PERIOD_S, clock=time.monotonic, initial_state="idle"):
        self.led = led
        self.blink_period_s = blink_period_s
        self.clock = clock
        self.state = "planning"                    # never a real state name
        self.set_state(initial_state)

    def set_state(self, state):
        if state not in STATES:
            state = "error"                       # a status light must not crash on a bad message
        if state == self.state:
            return                          # same state: the clock keeps running
        self._frames = pattern_for(state, self.blink_period_s)    # don't forget the blink period
        self._start = self.clock()                   # "now", from self.clock (not time.monotonic: the tests fake it)
        self.state = state

    def color(self):
        return color_at(self._frames, self.clock() - self._start)

    def update(self):
        c = self.color()
        if self.led is not None:
            self.led.value = c
        return c

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
    light = StatusLight(led, blink_period_s=args.blink_period)
    try:
        for name in names:
            print(name)
            light.set_state(name)
            t_end = time.monotonic() + args.seconds
            while time.monotonic() < t_end:
                light.update()
                time.sleep(0.02)
    except KeyboardInterrupt:
        print()                           # Ctrl-C: a newline instead of a traceback
    finally:
        led.off()
        led.close()   
    return 0
if __name__ == "__main__":
    sys.exit(main())