import pytest

from software.skull_core.servo_sweep import (
    HARD_MAX_US,
    HARD_MIN_US,
    check_delay,
    clamp_us,
    sweep_pulses,
    to_lgpio_width,
)


def test_clamp_inside_and_outside():
    assert clamp_us(1500) == 1500
    assert clamp_us(400) == HARD_MIN_US
    assert clamp_us(2600) == HARD_MAX_US
    assert clamp_us(1300, lo=1400, hi=1600) == 1400


def test_sweep_goes_there_and_back():
    assert sweep_pulses(1400, 1600, 100) == [1400, 1500, 1600, 1500, 1400]


def test_sweep_includes_the_end_when_step_does_not_divide():
    assert sweep_pulses(1400, 1650, 100) == [1400, 1500, 1600, 1650, 1600, 1500, 1400]


def test_sweep_has_no_duplicate_neighbours():
    pulses = sweep_pulses(1400, 1600, 10)
    assert len(pulses) == 41
    assert all(a != b for a, b in zip(pulses, pulses[1:]))


def test_sweep_single_point():
    assert sweep_pulses(1500, 1500, 10) == [1500]


@pytest.mark.parametrize("lo, hi, step", [(1400, 1600, 0), (1400, 1600, -10), (1600, 1400, 10)])
def test_sweep_rejects_bad_arguments(lo, hi, step):
    with pytest.raises(ValueError):
        sweep_pulses(lo, hi, step)


def test_lgpio_width_is_a_rounded_int():
    width = to_lgpio_width(1500.4)
    assert width == 1500 and isinstance(width, int)
    assert to_lgpio_width(1500.6) == 1501


def test_lgpio_width_is_clamped_and_never_zero():
    assert to_lgpio_width(0) == HARD_MIN_US
    assert to_lgpio_width(2500) == HARD_MAX_US


def test_nothing_outside_the_clamp_reaches_the_pin():
    widths = [to_lgpio_width(p) for p in sweep_pulses(900, 2100, 50)]
    assert min(widths) == HARD_MIN_US and max(widths) == HARD_MAX_US


def test_delay_must_cover_one_frame():
    assert check_delay(0.05) == 0.05
    with pytest.raises(ValueError):
        check_delay(0.005)
