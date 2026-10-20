"""Text-mode brain: one line of text in, the reply streamed out, one log record per turn.

Each turn runs `claude -p` (headless Claude Code) once, as a child process, with the model pinned
by ID, no tools, no MCP servers and its own system prompt, in a working folder outside the repo.
The reply is read from the child's stdout as JSON lines (`--output-format stream-json`) and
handed on piece by piece while it is still being written, so a speech step can start early.

The skull's state goes into the prompt as a line of text (the brain has no tools to ask for it).
Every turn ends with a record: time to the first text piece, total time, the API time and token
counts the CLI reports, the exit status, and the failure kind if there was one. A failed turn
(timeout, `claude` missing, a non-zero exit, an error or refusal result, no text) still ends
cleanly: the child is killed and a fixed spoken-style line comes out instead.

Our code never reads, prints or stores an auth token. `claude` finds its own login (on the Pi,
export CLAUDE_CODE_OAUTH_TOKEN from ~/skull-brain/token.env in the shell first). ANTHROPIC_API_KEY
is removed from the child's environment: it would outrank the login and bill the API.

Usage (from the repo root; -m is required):
    python -m software.skull_core.brain [--once TEXT] [--model claude-haiku-5-5]
           [--brain-dir ~/skull-brain] [--timeout 10] [--log ~/m3-bench/logs/m303_turns.jsonl]
           [--system-prompt-file FILE] [--no-thinking] [--claude claude] [--raw FILE]
    python -m software.skull_core.brain --report N [--log ...]

Without --once: type a line, see the reply, repeat. An empty line or Ctrl+D quits.
--report N prints the median and worst times of the last N turns in the log, then exits.
"""
import argparse
import json
import os
import queue
import signal
import statistics
import subprocess
import sys
import threading
import time
from collections.abc import Callable, Iterator
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path

DEFAULT_MODEL = "claude-haiku-5-5"      # pinned by ID, never an alias (ADR-013)
DEFAULT_BRAIN_DIR = Path.home() / "skull-brain"
DEFAULT_LOG = Path.home() / "m3-bench" / "logs" / "m303_turns.jsonl"
DEFAULT_TIMEOUT_S = 10.0   # estimate: about 2.3x the one cold Pi run (4.34 s) and 3.3x the warm worst (3.04 s)
DEFAULT_CLAUDE = "claude"
# Placeholder persona, the same line as the M3-00 benchmark. M5 owns the real one.
DEFAULT_SYSTEM_PROMPT = ("You are the voice of a small robot skull on a desk. "
                         "Answer in one short spoken sentence, no lists, no markdown.")
EMPTY_MCP = '{"mcpServers":{}}'
STDERR_KEEP = 2000          # characters of the child's stderr kept for the error message

# Spoken-style lines for a turn that failed. Placeholders: the wording is Ersin's call.
FALLBACK_LINE = "I can't reach my brain."
REFUSAL_LINE = "My brain won't answer that."
KINDS = ("setup", "not_found", "timeout", "refusal", "error", "exit", "empty", "abandoned")


@dataclass(frozen=True)
class BrainConfig:
    model: str = DEFAULT_MODEL
    brain_dir: Path = DEFAULT_BRAIN_DIR
    timeout_s: float = DEFAULT_TIMEOUT_S
    system_prompt: str = DEFAULT_SYSTEM_PROMPT
    claude: str = DEFAULT_CLAUDE
    thinking_off: bool = False


@dataclass
class TurnRecord:
    """What one turn did. kind is None when it worked, else one of KINDS. Times are in ms."""
    started_at: str
    model: str
    kind: str | None = None
    message: str = ""
    first_chunk_ms: float | None = None
    total_ms: float | None = None
    duration_api_ms: float | None = None
    duration_ms: float | None = None          # the CLI's own wall time, as it reports it (excludes its start-up)
    ttft_ms: float | None = None              # the CLI's time to first token, as it reports it
    time_to_request_ms: float | None = None   # the CLI's time until it sent the request, as it reports it
    input_tokens: int | None = None
    cache_creation_input_tokens: int | None = None
    cache_read_input_tokens: int | None = None
    output_tokens: int | None = None
    thinking_tokens: int | None = None
    cost_usd: float | None = None             # notional on a subscription, logged as reported
    models_reported: list[str] = field(default_factory=list)
    stop_reason: str | None = None
    api_error_status: int | str | None = None
    rate_limit_status: str | None = None      # e.g. "allowed": watch it for subscription limits (ADR-013)
    claude_code_version: str | None = None
    exit_code: int | None = None
    chunks: int = 0
    reply_chars: int = 0
    bad_lines: int = 0
    fallback: str | None = None

    @property
    def ok(self) -> bool:
        return self.kind is None


def print_now(text: str, end: str = "\n") -> None:
    """print() and flush at once, so a piped run (`| tee`) shows it straight away (C-004)."""
    print(text, end=end, flush=True)


def stub_state() -> str:
    """Stand-in for the skull's state until M3-06 publishes the real voice state."""
    return "voice state: listening"


def build_command(config: BrainConfig) -> list[str]:
    """The argv for one turn. A list, never a shell string; the user's text goes in on stdin.

    Every safety flag of M3-00 section 6 is here. --tools and --mcp-config take several values,
    so each is followed by another flag, never by a bare value that could be swallowed.
    """
    return [
        config.claude, "-p",
        "--model", config.model,
        "--tools", "",
        "--permission-mode", "dontAsk",
        "--strict-mcp-config", "--mcp-config", EMPTY_MCP,
        "--system-prompt", config.system_prompt,
        "--no-session-persistence",
        "--output-format", "stream-json",
        "--verbose",
        "--include-partial-messages",
    ]


def build_env(base: dict[str, str], thinking_off: bool = False) -> dict[str, str]:
    """The child's environment: a copy of base, without the API key, without CLAUDE.md or memory."""
    env = dict(base)
    env.pop("ANTHROPIC_API_KEY", None)
    env["CLAUDE_CODE_DISABLE_CLAUDE_MDS"] = "1"
    env["CLAUDE_CODE_DISABLE_AUTO_MEMORY"] = "1"
    if thinking_off:
        env["MAX_THINKING_TOKENS"] = "0"
    return env


def build_prompt(text: str, state_line: str = "") -> str:
    """The user message: the state line first, then what was said."""
    state_line = state_line.strip()
    if not state_line:
        return text
    return f"[{state_line}]\n{text}"


def parse_line(line: str) -> dict | None:
    """One stream-json line -> the event dict, or None if it isn't a JSON object. Never raises."""
    try:
        event = json.loads(line)
    except ValueError:
        return None
    return event if isinstance(event, dict) else None


def delta_text(event: dict) -> str | None:
    """The text of a partial-message event (a text_delta), else None. Thinking deltas are skipped."""
    if event.get("type") != "stream_event":
        return None
    inner = event.get("event")
    if not isinstance(inner, dict) or inner.get("type") != "content_block_delta":
        return None
    delta = inner.get("delta")
    if not isinstance(delta, dict) or delta.get("type") != "text_delta":
        return None
    text = delta.get("text")
    return text if isinstance(text, str) and text else None


def message_text(event: dict) -> str | None:
    """The joined text blocks of a whole assistant message, else None. Used only if no deltas came."""
    if event.get("type") != "assistant":
        return None
    message = event.get("message")
    content = message.get("content") if isinstance(message, dict) else None
    if not isinstance(content, list):
        return None
    parts = [block.get("text") for block in content
             if isinstance(block, dict) and block.get("type") == "text" and isinstance(block.get("text"), str)]
    return "".join(parts) or None


def _number(value):
    return value if isinstance(value, (int, float)) and not isinstance(value, bool) else None


def read_result(event: dict, record: TurnRecord) -> dict:
    """Copy timing, usage and cost from the final `result` event into the record. Returns the event."""
    record.duration_api_ms = _number(event.get("duration_api_ms"))
    record.duration_ms = _number(event.get("duration_ms"))
    record.ttft_ms = _number(event.get("ttft_ms"))
    record.time_to_request_ms = _number(event.get("time_to_request_ms"))
    if isinstance(event.get("api_error_status"), (int, str)) and not isinstance(event.get("api_error_status"), bool):
        record.api_error_status = event["api_error_status"]
    record.cost_usd = _number(event.get("total_cost_usd"))
    usage = event.get("usage") if isinstance(event.get("usage"), dict) else {}
    record.input_tokens = _number(usage.get("input_tokens"))
    record.cache_creation_input_tokens = _number(usage.get("cache_creation_input_tokens"))
    record.cache_read_input_tokens = _number(usage.get("cache_read_input_tokens"))
    record.output_tokens = _number(usage.get("output_tokens"))
    details = usage.get("output_tokens_details")
    if isinstance(details, dict):
        record.thinking_tokens = _number(details.get("thinking_tokens"))
    models = event.get("modelUsage")
    if isinstance(models, dict):
        record.models_reported = sorted(str(name) for name in models)
    if isinstance(event.get("stop_reason"), str):
        record.stop_reason = event["stop_reason"]
    return event


def note_event(event: dict, record: TurnRecord) -> None:
    """Copy the small facts other events carry: the CLI version, the rate-limit status, a stop reason."""
    kind = event.get("type")
    if kind == "system" and event.get("subtype") == "init" and isinstance(event.get("claude_code_version"), str):
        record.claude_code_version = event["claude_code_version"]
    elif kind == "rate_limit_event" and isinstance(event.get("rate_limit_info"), dict):
        status = event["rate_limit_info"].get("status")
        record.rate_limit_status = status if isinstance(status, str) else None
    elif kind == "stream_event" and isinstance(event.get("event"), dict) and event["event"].get("type") == "message_delta":
        delta = event["event"].get("delta")
        if isinstance(delta, dict) and isinstance(delta.get("stop_reason"), str):
            record.stop_reason = delta["stop_reason"]
    elif kind == "assistant" and isinstance(event.get("message"), dict):
        reason = event["message"].get("stop_reason")
        if isinstance(reason, str):
            record.stop_reason = reason


def classify(record: TurnRecord, result: dict | None, timed_out: bool) -> tuple[str | None, str]:
    """(kind, message) for a finished child. Order matters: timeout, refusal, error, exit, empty."""
    if timed_out:
        return "timeout", "no answer within the timeout; the child was killed"
    if record.stop_reason == "refusal":
        return "refusal", "the model refused"
    if result is not None and (result.get("is_error") or result.get("subtype") not in (None, "success")):
        detail = result.get("result") if isinstance(result.get("result"), str) else result.get("subtype")
        return "error", f"error result: {str(detail)[:200]}"
    if record.exit_code not in (0, None):
        return "exit", f"claude exited with code {record.exit_code}"
    if result is None:
        return "error", "the stream ended without a result line"
    if record.reply_chars == 0:
        return "empty", "the reply had no text"
    return None, ""


def fallback_line(kind: str) -> str:
    return REFUSAL_LINE if kind == "refusal" else FALLBACK_LINE


def _pump(stream, lines: queue.Queue) -> None:
    """Reader thread: every line of the child's stdout onto the queue, then None for end of stream."""
    try:
        for line in stream:
            lines.put(line)
    except (OSError, ValueError):
        pass
    finally:
        lines.put(None)


def _collect(stream, keep: list[str]) -> None:
    """Reader thread for stderr, so a chatty child can never block on a full pipe."""
    try:
        for line in stream:
            keep.append(line)
    except (OSError, ValueError):
        pass


def _kill(proc: subprocess.Popen) -> None:
    """Kill the child and anything it started (it runs in its own process group)."""
    if proc.poll() is not None:
        return
    try:
        os.killpg(proc.pid, signal.SIGKILL)
    except (ProcessLookupError, PermissionError):
        proc.kill()
    try:
        proc.wait(timeout=2.0)
    except subprocess.TimeoutExpired:
        pass


class Brain:
    """ask(text) yields the reply's text pieces as they arrive; afterwards self.last is the TurnRecord.

    A failed turn yields the fallback line as its last piece, so a speaker always has something to
    say. If the caller stops reading early, the child is killed and the turn is logged as "abandoned".
    """

    def __init__(self, config: BrainConfig = BrainConfig(), state: Callable[[], str] = stub_state,
                 log_path: Path | None = DEFAULT_LOG, raw_path: Path | None = None,
                 clock: Callable[[], float] = time.monotonic, environ: dict[str, str] | None = None):
        self.config = config
        self.state = state
        self.log_path = log_path
        self.raw_path = raw_path
        self.clock = clock
        self.environ = os.environ if environ is None else environ
        self.last: TurnRecord | None = None

    def ask(self, text: str) -> Iterator[str]:
        record = TurnRecord(started_at=datetime.now().astimezone().isoformat(timespec="seconds"),
                            model=self.config.model)
        self.last = record
        t0 = self.clock()
        finished = False
        try:
            yield from self._run(text, record, t0)
            if record.kind is not None:
                record.fallback = fallback_line(record.kind)
                yield record.fallback
            finished = True
        finally:
            if not finished and record.kind is None:
                record.kind, record.message = "abandoned", "the caller stopped reading"
            if record.total_ms is None:
                record.total_ms = round((self.clock() - t0) * 1000, 1)
            self._log(record)

    def _run(self, text: str, record: TurnRecord, t0: float) -> Iterator[str]:
        config = self.config
        if not config.brain_dir.is_dir():
            record.kind = "setup"
            record.message = f"brain folder {config.brain_dir} is missing (mkdir -p it, chmod 700)"
            return
        prompt = build_prompt(text, self.state())
        try:
            proc = subprocess.Popen(build_command(config), cwd=config.brain_dir,
                                    env=build_env(self.environ, config.thinking_off),
                                    stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                    text=True, encoding="utf-8", errors="replace", bufsize=1,
                                    start_new_session=True)
        except FileNotFoundError:
            record.kind, record.message = "not_found", f"'{config.claude}' not found on PATH"
            return
        except OSError as exc:
            record.kind, record.message = "error", f"could not start {config.claude}: {exc.strerror or exc}"
            return

        lines: queue.Queue = queue.Queue()
        errors: list[str] = []
        readers = [threading.Thread(target=_pump, args=(proc.stdout, lines), daemon=True),
                   threading.Thread(target=_collect, args=(proc.stderr, errors), daemon=True)]
        raw = open(self.raw_path, "w", encoding="utf-8") if self.raw_path else None
        result = None
        streamed = False
        timed_out = False
        try:
            for reader in readers:
                reader.start()
            try:
                proc.stdin.write(prompt)
                proc.stdin.close()
            except (BrokenPipeError, OSError):
                pass   # the child died at once; its exit code tells why
            deadline = t0 + config.timeout_s
            while True:
                remaining = deadline - self.clock()
                if remaining <= 0:
                    timed_out = True
                    break
                try:
                    line = lines.get(timeout=remaining)
                except queue.Empty:
                    timed_out = True
                    break
                if line is None:
                    break
                if raw:
                    raw.write(line)
                if not line.strip():
                    continue
                event = parse_line(line)
                if event is None:
                    record.bad_lines += 1
                    continue
                piece = delta_text(event)
                if piece is not None:
                    streamed = True
                elif not streamed and record.chunks == 0:
                    piece = message_text(event)
                note_event(event, record)
                if event.get("type") == "result":
                    result = read_result(event, record)
                    if record.chunks == 0 and not result.get("is_error") and isinstance(result.get("result"), str):
                        piece = result["result"] or None
                if piece:
                    if record.first_chunk_ms is None:
                        record.first_chunk_ms = round((self.clock() - t0) * 1000, 1)
                    record.chunks += 1
                    record.reply_chars += len(piece)
                    yield piece
            if not timed_out:
                try:
                    proc.wait(timeout=max(deadline - self.clock(), 0.01))
                except subprocess.TimeoutExpired:
                    timed_out = True
        finally:
            _kill(proc)
            for reader in readers:
                reader.join(timeout=1.0)
            for pipe in (proc.stdout, proc.stderr):
                try:
                    pipe.close()
                except OSError:
                    pass
            if raw:
                raw.close()
        record.total_ms = round((self.clock() - t0) * 1000, 1)
        record.exit_code = None if timed_out else proc.returncode
        record.kind, record.message = classify(record, result, timed_out)
        if record.kind in ("exit", "error") and errors:
            first = next((e.strip() for e in errors if e.strip()), "")
            record.message += f" ({first[:200]})" if first else ""

    def _log(self, record: TurnRecord) -> None:
        if self.log_path is None:
            return
        try:
            self.log_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.log_path, "a", encoding="utf-8") as log:
                log.write(json.dumps(asdict(record)) + "\n")
        except OSError as exc:
            print(f"Could not write the turn log {self.log_path}: {exc}", file=sys.stderr, flush=True)


def summary_line(record: TurnRecord) -> str:
    """One line per turn, e.g. 'turn ok first 812 ms total 2601 ms api 1006 ms in 564 out 12 exit 0'."""
    def ms(value):
        return "-" if value is None else f"{value:.0f} ms"

    def num(value):
        return "-" if value is None else str(value)

    cached = (record.cache_read_input_tokens or 0) + (record.cache_creation_input_tokens or 0)
    status = "ok" if record.ok else f"FAILED {record.kind}: {record.message}"
    cost = "-" if record.cost_usd is None else f"${record.cost_usd:.6f}"
    return (f"turn {status} | model {record.model} | first {ms(record.first_chunk_ms)} | "
            f"total {ms(record.total_ms)} | api {ms(record.duration_api_ms)} | "
            f"in {num(record.input_tokens)} (+{cached} cached) out {num(record.output_tokens)} | "
            f"cost {cost} | exit {num(record.exit_code)}")


def report(records: list[dict]) -> str:
    """Median and worst of the time to first chunk and the total, over the turns that worked.

    "warm" leaves out the first turn of the list, as M3-00 did with its cold run 1.
    """
    ok = [r for r in records if r.get("kind") is None and r.get("first_chunk_ms") is not None]
    lines = [f"turns {len(records)}, ok {len(ok)}, failed {len(records) - len(ok)}"
             f" ({', '.join(sorted({str(r.get('kind')) for r in records if r.get('kind')})) or 'none'})"]
    for label, rows in (("all ok", ok), ("warm (first turn left out)", ok[1:] if records and records[0] in ok else ok)):
        if not rows:
            lines.append(f"{label}: n=0")
            continue
        first = [r["first_chunk_ms"] for r in rows]
        total = [r["total_ms"] for r in rows]
        api = [r["duration_api_ms"] for r in rows if r.get("duration_api_ms") is not None]
        api_text = f"api median {statistics.median(api):.0f} ms" if api else "api -"
        lines.append(f"{label}: n={len(rows)} | first median {statistics.median(first):.0f} ms, "
                     f"worst {max(first):.0f} ms | total median {statistics.median(total):.0f} ms, "
                     f"worst {max(total):.0f} ms | {api_text}")
    return "\n".join(lines)


def run_turn(brain: Brain, text: str, out: Callable[..., None] = print_now) -> TurnRecord:
    """Ask, print each piece as it comes, then the summary line. Returns the turn's record."""
    for piece in brain.ask(text):
        out(piece, end="")
    out("")
    out(summary_line(brain.last))
    return brain.last


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Text-mode brain: a typed line goes to claude -p, "
                                                 "the reply streams back, every turn is logged.")
    parser.add_argument("--once", metavar="TEXT", default=None, help="ask one question, then exit")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="model ID (pin an ID, not an alias)")
    parser.add_argument("--brain-dir", type=Path, default=DEFAULT_BRAIN_DIR,
                        help="working folder for claude, outside the repo")
    parser.add_argument("--timeout", type=float, default=DEFAULT_TIMEOUT_S,
                        help="seconds before the turn is killed and the fallback line is used")
    parser.add_argument("--log", type=Path, default=DEFAULT_LOG, help="JSON-lines turn log (appended)")
    parser.add_argument("--system-prompt-file", type=Path, default=None,
                        help="read the system prompt from this file (default: a one-line placeholder)")
    parser.add_argument("--no-thinking", action="store_true", help="set MAX_THINKING_TOKENS=0 for the child")
    parser.add_argument("--claude", default=DEFAULT_CLAUDE, help="the claude executable")
    parser.add_argument("--raw", type=Path, default=None, help="also save the last turn's raw stream here")
    parser.add_argument("--report", type=int, default=None, metavar="N",
                        help="print median and worst of the last N logged turns, then exit")
    args = parser.parse_args(argv)
    if not args.timeout > 0:
        parser.error("--timeout must be greater than 0")
    if args.report is not None and args.report < 1:
        parser.error("--report needs N >= 1")
    return args


def main(argv=None) -> int:
    args = parse_args(argv)
    if args.report is not None:
        try:
            rows = [json.loads(line) for line in args.log.read_text(encoding="utf-8").splitlines() if line.strip()]
        except (OSError, ValueError) as exc:
            print(f"Could not read {args.log}: {exc}", file=sys.stderr)
            return 1
        print_now(report(rows[-args.report:]))
        return 0

    system_prompt = DEFAULT_SYSTEM_PROMPT
    if args.system_prompt_file is not None:
        try:
            system_prompt = args.system_prompt_file.read_text(encoding="utf-8").strip()
        except OSError as exc:
            print(f"Could not read {args.system_prompt_file}: {exc}", file=sys.stderr)
            return 1
    config = BrainConfig(model=args.model, brain_dir=args.brain_dir.expanduser(), timeout_s=args.timeout,
                         system_prompt=system_prompt, claude=args.claude, thinking_off=args.no_thinking)
    brain = Brain(config, log_path=args.log.expanduser(), raw_path=args.raw)

    if args.once is not None:
        return 0 if run_turn(brain, args.once).ok else 1

    print_now(f"Brain: {config.model} in {config.brain_dir}, timeout {config.timeout_s:g} s. "
              f"Empty line or Ctrl+D quits.")
    try:
        while True:
            try:
                line = input("> ")
            except EOFError:
                break
            if not line.strip():
                break
            run_turn(brain, line)
    except KeyboardInterrupt:
        print_now("")
    return 0


if __name__ == "__main__":
    sys.exit(main())
