import pytest

from software.skull_core.state_console import demo_names
from software.skull_core.status_leds import STATES, TEST


def test_all_is_every_state_in_states_order():
    assert demo_names("all") == list(STATES)
    assert len(demo_names("all")) == 6


def test_all_returns_a_fresh_list():
    names = demo_names("all")
    names.append("dancing")
    assert demo_names("all") == list(STATES)


@pytest.mark.parametrize("state", list(STATES))
def test_one_state_shows_only_that_state(state):
    assert demo_names(state) == [state]


def test_the_test_command_is_not_a_demo_state():
    with pytest.raises(ValueError):
        demo_names(TEST)


@pytest.mark.parametrize("bad", ["quit", "", "dancing", "ERROR", "All", " idle"])
def test_anything_else_is_rejected(bad):
    with pytest.raises(ValueError):
        demo_names(bad)


@pytest.mark.parametrize("bad", ["dancing", "ERROR", TEST])
def test_error_message_names_the_bad_choice(bad):
    with pytest.raises(ValueError) as err:
        demo_names(bad)
    assert bad in str(err.value)
