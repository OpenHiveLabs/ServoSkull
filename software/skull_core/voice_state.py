"""The voice state on the bridge: the pipeline publishes it, the eye LED follows it (M3-06).

The voice pipeline never touches the LED. It calls on_state(name); here on_state is a publish
on <robot prefix>/state/voice, and the LED is one subscriber. The payload is the bare state name,
one of the six in status_leds.STATES (ADR-012). Anything else is published as "error".

Usage (from the repo root; -m is required):
    python -u -m software.skull_core.voice_state --demo states [--seconds 3] [--no-led]
    python -u -m software.skull_core.voice_state --demo turn [--no-led]
Options: --pins 5 6 13 (R G B, BCM)  --common-anode  --blink-period 1.0  --params robot_params.yaml
"""
import argparse
import contextlib
import sys
import threading
import time
from collections.abc import Callable
from pathlib import Path

from software.skull_core.skull_bridge import DEFAULT_PARAMS, Bridge, load_key_prefix, make_key
from software.skull_core.status_leds import DEFAULT_BLINK_PERIOD_S, STATES, StatusLight, run_ticker

VOICE_STATE = "state/voice"     # the part after the robot's prefix

# Brain failure kinds (brain.KINDS) and the state shown after the fallback line has played.
# Provisional (PM, 2026-10-20, decision 3): unreachable -> offline; our own fault -> error;
# a refusal, an empty reply or an abandoned turn -> nothing special (back to idle).
UNREACHABLE_KINDS = ("timeout", "exit", "error")
OWN_FAULT_KINDS = ("setup", "not_found")
LIMIT_STATUS = "rejected"   # rate_limit_status meaning "over the limit". Unverified: not yet seen in a real log.


def voice_key(params_path: Path = DEFAULT_PARAMS) -> str:
    """This robot's voice state key, "<key_prefix>/state/voice", from robot_params.yaml."""
    return make_key(load_key_prefix(params_path), VOICE_STATE)


def publish_state(bridge: Bridge, key: str) -> Callable[[str], None]:
    """Return an on_state(name) for run_turn and the speaker: it publishes name on key.

    A name that is not one of the six states is published as "error" (same rule as
    StatusLight.set_state), so no subscriber ever receives a name outside the contract.
    """
    def on_state(name: str) -> None:
        bridge.publish(key, name if name in STATES else "error")
    return on_state


def led_follows(bridge: Bridge, key: str, light: StatusLight) -> None:
    """Subscribe the light to key: every payload goes to light.set_state. Its only caller from now on."""
    def show_on_led(_key: str, payload: str) -> None:
        light.set_state(payload)
    bridge.subscribe(key, show_on_led)


@contextlib.contextmanager
def eye_led(bridge: Bridge, key: str, pins=(5, 6, 13), common_anode: bool = False,
            blink_period_s: float = DEFAULT_BLINK_PERIOD_S, make_led=None):
    """While inside the with-block, the eye LED follows key; on exit the ticker stops and the pins are released.

    The light starts on bridge.last(key) (a late subscriber catches up, decision 6), or "thinking"
    if nothing was published yet (start-up, decision 2). make_led() builds the LED; the default is
    gpiozero's RGBLED, imported only here so the tests need no GPIO.
    After exit the pins are released: GPIO6 may glow green faintly from its default pull-up (F-022).
    """
    if make_led is None:
        from gpiozero import RGBLED

        def make_led():
            return RGBLED(*pins, active_high=not common_anode)
    led = make_led()
    stop = threading.Event()
    ticker = None
    try:
        light = StatusLight(led, blink_period_s=blink_period_s, initial_state=bridge.last(key) or "thinking")
        led_follows(bridge, key, light)
        ticker = threading.Thread(target=run_ticker, args=(light, stop), name="eye-led", daemon=True)
        ticker.start()
        yield light
    finally:
        stop.set()
        if ticker is not None:
            ticker.join()
        led.off()
        led.close()


def print_watcher(bridge: Bridge, key: str, clock: Callable[[], float] = time.monotonic,
                  out: Callable[[str], None] | None = None) -> None:
    """Subscribe a printer: one line per message, "[state +12.345s] <key_prefix>/state/voice idle".

    Times are seconds since the watcher was attached, for comparing a run against the state map.
    """
    start = clock()
    write = out if out is not None else (lambda line: print(line, flush=True))

    def print_message(msg_key: str, payload: str) -> None:
        write(f"[state +{clock() - start:7.3f}s] {msg_key} {payload}")
    bridge.subscribe(key, print_message)


def state_after_brain(kind: str | None, rate_limit_status: str | None = None) -> str | None:
    """The state to publish after a failed brain turn's fallback line, or None for "carry on as normal".

    kind is TurnRecord.kind from brain.py. A refusal or a limit is just a spoken line (None).
    """
    if kind is None or kind in ("refusal", "empty", "abandoned"):
        return None
    if kind in OWN_FAULT_KINDS:
        return "error"
    if kind in UNREACHABLE_KINDS:
        if rate_limit_status == LIMIT_STATUS:
            return None
        return "offline"
    return "error"    # a kind this code doesn't know yet: show it rather than hide it


# --- Bench demo ----------------------------------------------------------------------------------

# A scripted turn (state, seconds). Times are made up for the bench, not measured.
# Startup "thinking", then a turn with two sentences; the gap between them is "thinking" (decision 1).
SCRIPTED_TURN = (
    ("thinking", 3.0),    # models loading
    ("idle", 3.0),        # waiting for Enter
    ("listening", 3.0),
    ("thinking", 4.0),    # Moonshine + brain + first sentence
    ("speaking", 2.0),    # sentence 1 (published from a second thread, like the playback thread)
    ("thinking", 1.0),    # gap: sentence 2 not ready
    ("speaking", 2.0),    # sentence 2
    ("idle", 2.0),
)
SPEAKER_THREAD_STATES = ("speaking",)


def run_script(on_state: Callable[[str], None], script, sleep=time.sleep) -> None:
    """Publish each state of the script in turn; 'speaking' from a separate thread, as the speaker will."""
    for name, seconds in script:
        if name in SPEAKER_THREAD_STATES:
            worker = threading.Thread(target=on_state, args=(name,), name="fake-speaker")
            worker.start()
            worker.join()
        else:
            on_state(name)
        sleep(seconds)


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Voice state on the bridge: LED follows the published state.")
    parser.add_argument("--demo", choices=("states", "turn"), required=True,
                        help="states: the six states in turn; turn: a scripted turn with two sentences")
    parser.add_argument("--seconds", type=float, default=3.0, help="per state, for --demo states")
    parser.add_argument("--params", type=Path, default=DEFAULT_PARAMS)
    parser.add_argument("--pins", nargs=3, type=int, default=[5, 6, 13], metavar=("R", "G", "B"))
    parser.add_argument("--common-anode", action="store_true")
    parser.add_argument("--blink-period", type=float, default=DEFAULT_BLINK_PERIOD_S)
    parser.add_argument("--no-led", action="store_true", help="no GPIO: print the messages only")
    args = parser.parse_args(argv)
    if not args.seconds > 0:
        parser.error("--seconds must be greater than 0")
    if not args.blink_period > 0:
        parser.error("--blink-period must be greater than 0")
    return args


def main(argv=None, sleep=time.sleep) -> int:
    args = parse_args(argv)
    try:
        key = voice_key(args.params)
    except ValueError as exc:
        print(exc, file=sys.stderr)
        return 1
    script = tuple((name, args.seconds) for name in STATES) if args.demo == "states" else SCRIPTED_TURN

    bridge = Bridge()
    print_watcher(bridge, key)
    led = (contextlib.nullcontext() if args.no_led else
           eye_led(bridge, key, args.pins, args.common_anode, args.blink_period))
    with led:
        try:
            print(f"Publishing on {key}", flush=True)
            run_script(publish_state(bridge, key), script, sleep)
        except KeyboardInterrupt:
            print("", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
