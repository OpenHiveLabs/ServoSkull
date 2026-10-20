"""Tests for software/skull_core/skull_bridge.py: keys, publish/subscribe, last value, threads."""
import re
import threading
import time
from pathlib import Path

import pytest

from software.skull_core import skull_bridge as sb
from software.skull_core.skull_bridge import Bridge, check_key, load_key_prefix, make_key

BRIDGE_SOURCE = Path(sb.__file__)


# --- keys -----------------------------------------------------------------------------------------

@pytest.mark.parametrize("key", ["skull/skull-01/state/voice", "a", "skull/skull_02/state/voice"])
def test_good_keys_pass_unchanged(key):
    assert check_key(key) == key


@pytest.mark.parametrize("key, why", [
    ("", "empty"),
    ("/skull/skull-01", "leading or trailing slash"),
    ("skull/skull-01/", "leading or trailing slash"),
    ("skull//state", "empty chunk"),
    ("skull/*/state", "reserved character"),
    ("skull/**", "reserved character"),
    ("skull/a$b", "reserved character"),
    ("skull/a?b", "reserved character"),
    ("skull/a#b", "reserved character"),
    ("skull/skull 01", "whitespace"),
    ("skull/skull-01\t", "whitespace"),
])
def test_bad_keys_are_refused_with_the_key_in_the_message(key, why):
    with pytest.raises(ValueError) as err:
        check_key(key)
    assert why in str(err.value)
    assert repr(key) in str(err.value)


def test_make_key_joins_with_one_slash():
    assert make_key("skull/skull-02", "state/voice") == "skull/skull-02/state/voice"


def test_make_key_refuses_a_bad_suffix():
    with pytest.raises(ValueError):
        make_key("skull/skull-01", "/state/voice")


def test_prefix_comes_from_the_yaml(tmp_path):
    params = tmp_path / "robot_params.yaml"
    params.write_text("identity:\n  robot_id: skull-07\n  key_prefix: \"skull/skull-07\"\n", encoding="utf-8")
    assert load_key_prefix(params) == "skull/skull-07"


def test_the_repo_yaml_has_a_valid_prefix():
    assert check_key(load_key_prefix()) == load_key_prefix()


@pytest.mark.parametrize("text", ["identity:\n  robot_id: x\n", "other: 1\n", "", "identity: [1, 2]\n",
                                  "identity:\n  key_prefix: \"skull//x\"\n"])
def test_missing_or_bad_prefix_is_a_clear_error(tmp_path, text):
    params = tmp_path / "robot_params.yaml"
    params.write_text(text, encoding="utf-8")
    with pytest.raises(ValueError, match="robot_params.yaml|bad key"):
        load_key_prefix(params)


def test_missing_file_is_a_clear_error(tmp_path):
    with pytest.raises(ValueError, match="cannot read"):
        load_key_prefix(tmp_path / "nope.yaml")


def test_no_robot_id_literal_in_the_bridge_source():
    """ADR-006 rule 1: the ID lives in robot_params.yaml only."""
    assert not re.search(r"skull-\d+", BRIDGE_SOURCE.read_text(encoding="utf-8"))


# --- publish / subscribe --------------------------------------------------------------------------

KEY = "skull/skull-01/state/voice"


def test_two_subscribers_both_called_in_order():
    bridge = Bridge()
    calls = []
    bridge.subscribe(KEY, lambda k, p: calls.append(("first", k, p)))
    bridge.subscribe(KEY, lambda k, p: calls.append(("second", k, p)))
    bridge.publish(KEY, "idle")
    assert calls == [("first", KEY, "idle"), ("second", KEY, "idle")]


def test_exact_keys_only():
    bridge = Bridge()
    calls = []
    bridge.subscribe(KEY, lambda k, p: calls.append(p))
    bridge.publish("skull/skull-02/state/voice", "idle")
    bridge.publish(KEY + "/extra", "idle")
    assert calls == []


def test_subscribe_refuses_a_wildcard():
    with pytest.raises(ValueError):
        Bridge().subscribe("skull/*/state/voice", lambda k, p: None)


def test_publish_refuses_a_non_string_payload():
    with pytest.raises(TypeError):
        Bridge().publish(KEY, 3)


def test_a_raising_subscriber_is_reported_and_skipped():
    reports = []
    bridge = Bridge(report=reports.append)
    calls = []

    def broken(_k, _p):
        raise RuntimeError("LED gone")

    bridge.subscribe(KEY, broken)
    bridge.subscribe(KEY, lambda k, p: calls.append(p))
    bridge.publish(KEY, "speaking")          # must not raise
    assert calls == ["speaking"]
    assert len(reports) == 1 and "LED gone" in reports[0] and KEY in reports[0]


def test_last_before_and_after():
    bridge = Bridge()
    assert bridge.last(KEY) is None
    bridge.publish(KEY, "thinking")
    bridge.publish(KEY, "idle")
    assert bridge.last(KEY) == "idle"
    assert bridge.last("skull/skull-01/other") is None


def test_late_subscriber_gets_only_later_messages_but_can_read_last():
    bridge = Bridge()
    bridge.publish(KEY, "thinking")
    calls = []
    bridge.subscribe(KEY, lambda k, p: calls.append(p))
    assert calls == []                       # no replay on subscribe (decision 6: use last())
    assert bridge.last(KEY) == "thinking"


def test_a_subscriber_may_publish_without_deadlock():
    bridge = Bridge()
    echoed = []
    bridge.subscribe(KEY, lambda k, p: bridge.publish("skull/skull-01/echo", p))
    bridge.subscribe("skull/skull-01/echo", lambda k, p: echoed.append(p))
    worker = threading.Thread(target=bridge.publish, args=(KEY, "idle"))
    worker.start()
    worker.join(timeout=2.0)
    assert not worker.is_alive()
    assert echoed == ["idle"]


# --- threads --------------------------------------------------------------------------------------

def test_publishers_on_many_threads_lose_nothing_and_last_matches_what_subscribers_saw():
    bridge = Bridge()
    seen_a, seen_b = [], []
    inside = [0]
    overlap = [False]

    def subscriber_a(_k, payload):
        inside[0] += 1
        if inside[0] > 1:
            overlap[0] = True                # two deliveries at once would mean no serialisation
        seen_a.append(payload)
        time.sleep(0.0002)                   # hold the window open so an unlocked bridge would overlap
        inside[0] -= 1

    bridge.subscribe(KEY, subscriber_a)
    bridge.subscribe(KEY, lambda k, p: seen_b.append(p))

    threads_n, per_thread = 8, 100
    start = threading.Barrier(threads_n)

    def hammer(index):
        start.wait()
        for i in range(per_thread):
            bridge.publish(KEY, f"{index}:{i}")

    workers = [threading.Thread(target=hammer, args=(n,)) for n in range(threads_n)]
    for w in workers:
        w.start()
    for w in workers:
        w.join()

    assert len(seen_a) == threads_n * per_thread
    assert seen_a == seen_b                  # every subscriber saw the same order
    assert not overlap[0]
    assert bridge.last(KEY) == seen_a[-1]
    for n in range(threads_n):               # each thread's own messages arrive in its own order
        own = [int(p.split(":")[1]) for p in seen_a if p.startswith(f"{n}:")]
        assert own == list(range(per_thread))


def test_main_prints_the_full_key(capsys):
    assert sb.main(["--key-of", "state/voice"]) == 0
    assert capsys.readouterr().out.strip() == load_key_prefix() + "/state/voice"


def test_main_reports_a_bad_suffix(capsys):
    assert sb.main(["--key-of", "state//voice"]) == 1
    assert "bad key" in capsys.readouterr().err
