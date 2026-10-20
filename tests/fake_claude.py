"""A stand-in for the `claude` CLI, for tests/test_brain.py. It never contacts anything.

It reads its whole stdin (the prompt), then replays a recorded stream-json file line by line.
What it does is set by environment variables:
    FAKE_CLAUDE_STREAM      file to replay (default: nothing is printed)
    FAKE_CLAUDE_RECORD      write {"argv", "cwd", "stdin", "env"} here as JSON
    FAKE_CLAUDE_EXTRA       one extra raw line printed first (e.g. garbage)
    FAKE_CLAUDE_HANG_AFTER  after this many lines, hang (sleep 60 s) instead of going on
    FAKE_CLAUDE_GRANDCHILD  start a `sleep 60` grandchild that shares stdout; its pid goes to this file
    FAKE_CLAUDE_PIDFILE     write our own pid here
    FAKE_CLAUDE_STDERR      print this to stderr
    FAKE_CLAUDE_EXIT        exit code (default 0)
"""
import json
import os
import subprocess
import sys
import time

ENV_SEEN = ("ANTHROPIC_API_KEY", "CLAUDE_CODE_DISABLE_CLAUDE_MDS", "CLAUDE_CODE_DISABLE_AUTO_MEMORY",
            "MAX_THINKING_TOKENS")


def main() -> int:
    env = os.environ
    prompt = sys.stdin.read()
    if env.get("FAKE_CLAUDE_PIDFILE"):
        with open(env["FAKE_CLAUDE_PIDFILE"], "w") as f:
            f.write(str(os.getpid()))
    if env.get("FAKE_CLAUDE_RECORD"):
        with open(env["FAKE_CLAUDE_RECORD"], "w") as f:
            json.dump({"argv": sys.argv[1:], "cwd": os.getcwd(), "stdin": prompt,
                       "env": {k: env.get(k) for k in ENV_SEEN}}, f)
    if env.get("FAKE_CLAUDE_GRANDCHILD"):
        child = subprocess.Popen(["sleep", "60"])
        with open(env["FAKE_CLAUDE_GRANDCHILD"], "w") as f:
            f.write(str(child.pid))
    if env.get("FAKE_CLAUDE_STDERR"):
        print(env["FAKE_CLAUDE_STDERR"], file=sys.stderr, flush=True)
    if env.get("FAKE_CLAUDE_EXTRA"):
        print(env["FAKE_CLAUDE_EXTRA"], flush=True)
    hang_after = int(env.get("FAKE_CLAUDE_HANG_AFTER", "-1"))
    lines = []
    if env.get("FAKE_CLAUDE_STREAM"):
        with open(env["FAKE_CLAUDE_STREAM"]) as f:
            lines = [line.rstrip("\n") for line in f]
    for i, line in enumerate(lines):
        if i == hang_after:
            time.sleep(60)
        print(line, flush=True)
    if hang_after >= len(lines):
        time.sleep(60)
    return int(env.get("FAKE_CLAUDE_EXIT", "0"))


if __name__ == "__main__":
    sys.exit(main())
