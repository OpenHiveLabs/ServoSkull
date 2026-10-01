import pytest

from software.skull_core.status_leds import (
    BLUE,
    CYCLE_STEP_S,
    DEFAULT_BLINK_PERIOD_S,
    GREEN,
    OFF,
    QUIT,
    RED,
    STATES,
    TEST,
    TEST_SHOWS,
    StatusLight,
    color_at,
    parse_command,
    pattern_for,
)


def colours(frames):
    return [c for c, _ in frames]


def durations(frames):
    return [d for _, d in frames]


def test_the_table_has_exactly_the_six_states():
    assert set(STATES) == {"idle", "listening", "thinking", "speaking", "offline", "error"}


@pytest.mark.parametrize("state, colour", [("idle", GREEN), ("speaking", BLUE), ("offline", RED)])
def test_solid_states(state, colour):
    assert colours(pattern_for(state)) == [colour]


def test_thinking_cycles_red_green_blue_half_a_second_each():
    frames = pattern_for("thinking")
    assert colours(frames) == [RED, GREEN, BLUE]
    assert durations(frames) == pytest.approx([0.5, 0.5, 0.5])
    assert CYCLE_STEP_S == 0.5


def test_thinking_colour_over_time():
    frames = pattern_for("thinking")
    assert color_at(frames, 0.0) == RED
    assert color_at(frames, 0.49) == RED
    assert color_at(frames, 0.5) == GREEN
    assert color_at(frames, 1.0) == BLUE
    assert color_at(frames, 1.49) == BLUE
    assert color_at(frames, 1.5) == RED  # wraps around
    assert color_at(frames, 3.2) == RED  # 3.2 s = two cycles + 0.2 s


@pytest.mark.parametrize("state, colour", [("listening", BLUE), ("error", RED)])
def test_blink_is_half_on_half_off(state, colour):
    frames = pattern_for(state)
    assert colours(frames) == [colour, OFF]
    assert durations(frames) == pytest.approx([DEFAULT_BLINK_PERIOD_S / 2] * 2)


def test_blink_period_is_a_parameter():
    frames = pattern_for("listening", blink_period_s=0.4)
    assert durations(frames) == pytest.approx([0.2, 0.2])
    assert color_at(frames, 0.1) == BLUE
    assert color_at(frames, 0.3) == OFF
    assert color_at(frames, 0.5) == BLUE


def test_unknown_state_shows_error():
    assert pattern_for("dancing") == pattern_for("error")
    assert pattern_for("") == pattern_for("error")


def test_bad_inputs_raise():
    with pytest.raises(ValueError):
        pattern_for("idle", blink_period_s=0)
    with pytest.raises(ValueError):
        color_at(pattern_for("idle"), -0.1)


# ── --interactive: parse_command and StatusLight ─────────────────────────────

class FakeClock:
    """A clock the test moves by hand: set .t, then ask the light."""
    def __init__(self, t=100.0):
        self.t = t

    def __call__(self):
        return self.t


class FakeLed:
    value = None


@pytest.mark.parametrize("line, expected", [
    ("idle", "idle"),
    ("  Thinking \n", "thinking"),
    ("ERROR", "error"),
    ("quit", QUIT),
    (" Quit\n", QUIT),
    ("test", TEST),
    (" Test\n", TEST),
    ("dancing", None),  # a typo: None, NOT "error"
    ("", None),
    ("   ", None),
    ("idle now", None),
])
def test_parse_command(line, expected):
    assert parse_command(line) == expected


def test_parse_command_knows_every_state():
    for state in STATES:
        assert parse_command(state) == state


def test_test_is_a_command_not_a_state():
    assert TEST not in STATES          # STATES is what M3 publishes; "test" must never go out
    assert TEST_SHOWS == "speaking"
    light = StatusLight(clock=FakeClock())
    light.set_state(TEST_SHOWS)
    assert light.color() == BLUE       # the speaking pattern, not error's red


def test_light_starts_idle():
    light = StatusLight(clock=FakeClock())
    assert light.state == "idle"
    assert light.color() == GREEN


def test_new_state_restarts_the_pattern_clock():
    clock = FakeClock(100.0)
    light = StatusLight(clock=clock)
    clock.t = 100.3
    light.set_state("thinking")
    assert light.color() == RED    # 0.0 s into the cycle, not 0.3 s
    clock.t = 100.8
    assert light.color() == GREEN  # 0.5 s in


def test_same_state_again_keeps_the_pattern_running():
    clock = FakeClock(100.0)
    light = StatusLight(clock=clock, initial_state="thinking")
    clock.t = 100.6
    light.set_state("thinking")    # e.g. M3 re-publishes the same state
    assert light.color() == GREEN  # still 0.6 s in; a restart would show RED


def test_unknown_state_via_set_state_shows_error():
    light = StatusLight(clock=FakeClock())
    light.set_state("dancing")
    assert light.state == "error"
    assert light.color() == RED    # first half of the blink


def test_update_pushes_the_colour_to_the_led():
    clock = FakeClock(100.0)
    led = FakeLed()
    light = StatusLight(led=led, clock=clock, initial_state="listening")
    assert light.update() == BLUE
    assert led.value == BLUE
    clock.t = 100.0 + 0.75 * DEFAULT_BLINK_PERIOD_S
    light.update()
    assert led.value == OFF


def test_blink_period_reaches_the_light():
    clock = FakeClock(100.0)
    light = StatusLight(clock=clock, blink_period_s=0.4, initial_state="error")
    clock.t = 100.3
    assert light.color() == OFF


def test_update_without_an_led_does_not_crash():
    assert StatusLight(clock=FakeClock()).update() == GREEN