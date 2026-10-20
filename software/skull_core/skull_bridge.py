"""skull_bridge: an in-process publish/subscribe with Zenoh-shaped keys (ADR-006).

The pipeline says what happened (publish) without knowing who listens (subscribe). Today
everything stays inside one Python process. Later the same two verbs go out over Zenoh, and
the jaw (M4), a log or another skull can subscribe without the voice code changing.

Keys:
- Every key starts with the robot's own prefix from robot_params.yaml (identity.key_prefix,
  e.g. "skull/<robot_id>"). The robot ID is never typed into the code: a second skull only
  needs a different YAML.
- Zenoh-shaped: chunks separated by single slashes, no leading or trailing slash, no empty
  chunk. "*", "**", "$", "?" and "#" are reserved (Zenoh uses them for patterns), so a
  published key may not contain them. Subscriptions use exact keys only (no wildcards yet).

Threads: publish() may be called from any thread (the turn, the playback thread, main).
Callbacks run on the publisher's thread, one publish at a time: a lock is held while the value
is stored AND delivered, so every subscriber sees the messages in the same order and the last
one it saw is always last(key). The price: a slow subscriber delays the publisher. Keep
subscribers short (setting an LED state is microseconds); push slow work to a queue.

Usage (from the repo root; -m is required):
    python -m software.skull_core.skull_bridge --key-of state/voice [--params robot_params.yaml]
"""
import argparse
import sys
import threading
import traceback
from collections import defaultdict
from collections.abc import Callable
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PARAMS = REPO_ROOT / "robot_params.yaml"

# Characters Zenoh key expressions reserve for patterns and selectors. Never inside a chunk.
RESERVED_CHARS = ("*", "$", "?", "#")

Callback = Callable[[str, str], None]   # callback(key, payload)


def load_key_prefix(params_path: Path = DEFAULT_PARAMS) -> str:
    """identity.key_prefix from robot_params.yaml, checked as a key.

    Raises ValueError with the file name when the file, the section or the field is missing,
    so a misconfigured robot stops at start-up instead of publishing on a wrong key.
    """
    params_path = Path(params_path)
    try:
        data = yaml.safe_load(params_path.read_text(encoding="utf-8"))
    except OSError as exc:
        raise ValueError(f"cannot read {params_path}: {exc}") from exc
    identity = data.get("identity") if isinstance(data, dict) else None
    prefix = identity.get("key_prefix") if isinstance(identity, dict) else None
    if not isinstance(prefix, str):
        raise ValueError(f"{params_path} has no identity.key_prefix string")
    check_key(prefix)
    return prefix


def check_key(key: str) -> str:
    """Return key unchanged if it is a valid concrete key, else raise ValueError naming it.

    Valid: non-empty, no leading/trailing slash, no empty chunk (no "//"), no whitespace,
    no reserved character. These are keys Zenoh will accept later as plain (non-pattern) keys.
    """
    if not isinstance(key, str) or key == "":
        raise ValueError(f"bad key {key!r}: empty")
    if key.startswith("/") or key.endswith("/"):
        raise ValueError(f"bad key {key!r}: leading or trailing slash")
    for chunk in key.split("/"):
        if chunk == "":
            raise ValueError(f"bad key {key!r}: empty chunk (double slash)")
        if any(ch.isspace() for ch in chunk):
            raise ValueError(f"bad key {key!r}: whitespace in chunk {chunk!r}")
        bad = [ch for ch in RESERVED_CHARS if ch in chunk]
        if bad:
            raise ValueError(f"bad key {key!r}: reserved character {bad[0]!r} in chunk {chunk!r}")
    return key


def make_key(prefix: str, suffix: str) -> str:
    """prefix + "/" + suffix, checked. make_key(prefix, "state/voice") gives "<prefix>/state/voice"."""
    return check_key(f"{prefix}/{suffix}")


class Bridge:
    """In-process publish/subscribe on exact keys, with the last value of each key kept.

    report: called with a one-line message when a subscriber raises (default: print to stderr).
    The publisher never sees a subscriber's error; the other subscribers still get the message.
    """

    def __init__(self, report: Callable[[str], None] | None = None):
        self._subscribers: dict[str, list[Callback]] = defaultdict(list)
        self._last: dict[str, str] = {}
        # RLock, not Lock: a subscriber that publishes (on any key) from inside its callback
        # must not deadlock its own thread.
        self._lock = threading.RLock()
        self._report = report if report is not None else _print_to_stderr

    def subscribe(self, key: str, callback: Callback) -> None:
        """callback(key, payload) is called for every later publish on exactly this key."""
        check_key(key)
        with self._lock:
            self._subscribers[key].append(callback)

    def publish(self, key: str, payload: str) -> None:
        """Store payload as the key's last value, then call each subscriber of the key, in order.

        Done under the lock, so publishes from different threads never interleave: what a
        subscriber saw last is always last(key).
        """
        check_key(key)
        if not isinstance(payload, str):
            raise TypeError(f"payload for {key!r} must be str, got {type(payload).__name__}")
        with self._lock:
            self._last[key] = payload
            for callback in list(self._subscribers.get(key, ())):
                try:
                    callback(key, payload)
                except Exception as exc:   # a broken subscriber must not break the publisher
                    where = traceback.extract_tb(exc.__traceback__)[-1]
                    self._report(f"subscriber {getattr(callback, '__name__', callback)!r} on {key!r} raised "
                                 f"{type(exc).__name__}: {exc} ({Path(where.filename).name}:{where.lineno})")

    def last(self, key: str) -> str | None:
        """The most recent payload on this key, or None if nothing was published on it yet."""
        with self._lock:
            return self._last.get(key)


def _print_to_stderr(message: str) -> None:
    print(message, file=sys.stderr, flush=True)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Print the full key for a suffix, using this robot's prefix.")
    parser.add_argument("--key-of", default="state/voice", help="the part after the robot's prefix")
    parser.add_argument("--params", type=Path, default=DEFAULT_PARAMS)
    args = parser.parse_args(argv)
    try:
        print(make_key(load_key_prefix(args.params), args.key_of), flush=True)
    except ValueError as exc:
        print(exc, file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
