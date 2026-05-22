#!/usr/bin/env python3
"""
Homogeneous Needle-in-a-Haystack benchmark for llama-server (OpenAI-compatible API).

Generates random key=value pairs, embeds them as haystack, queries the model for
values at various depths, and writes results to CSV.

Usage:
  python haystack_test.py --haystack-num 5000 --needles-num 100 --repeat 3
"""

from __future__ import annotations

import argparse
import csv
import datetime
import json
import random
import secrets
import socket
import string
import sys
import time
import urllib.error
import urllib.request
from dataclasses import asdict, dataclass, field
from typing import TypedDict
from enum import Enum, auto


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

CSV_COLUMNS: list[str] = [
    "run", "haystack_size", "depth_pct", "correct", "expected", "actual",
    "prompt_tokens", "completion_tokens", "total_tokens", "latency_ms",
]


# ---------------------------------------------------------------------------
# Types
# ---------------------------------------------------------------------------

HaystackPair = tuple[str, int]
Message = dict[str, str]


class ChatCompletionChoiceMessage(TypedDict):
    content: str


class ChatCompletionChoice(TypedDict):
    message: ChatCompletionChoiceMessage


class ApiErrorResponse(TypedDict, total=False):
    error: dict[str, str]


class ApiUsage(TypedDict):
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int


class ApiSuccessResponse(TypedDict):
    model: str
    choices: list[ChatCompletionChoice]
    usage: ApiUsage


ApiResponseBody = ApiSuccessResponse | ApiErrorResponse


class Payload(TypedDict):
    model: str
    messages: list[Message]
    temperature: float
    max_tokens: int
    stream: bool


class RunStats(TypedDict, total=False):
    total_tokens: int
    total_latency_ms: int | float
    total_completion: int
    error: str | None
    truncated: bool | None
    model_name: str | None
    tokens_per_sec: float
    avg_latency_ms: float


@dataclass(frozen=True)
class ResultRow:
    run: int
    haystack_size: int
    depth_pct: float
    correct: int
    expected: int | str
    actual: str
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    total_tokens: int | None = None
    latency_ms: float | None = None


@dataclass
class Needle:
    index: int
    key: str
    expected: int | None
    is_distractor: bool


@dataclass
class Config:
    endpoint: str
    output_filename: str
    show: str
    model: str | None
    key_len: int
    val_min: int
    val_max: int
    haystack_num: int
    needles_num: int
    distractors_num: int
    temperature: float
    max_tokens: int
    timeout: int
    repeat: int
    full: bool
    seed: int
    fuzz: float = 0.49


@dataclass
class ModelResult:
    rows: list[ResultRow]
    model_name: str
    stats: RunStats


class GlobalResultType(Enum):
    TRUNCATED = auto()
    NO_RESULTS = auto()
    OK = auto()

@dataclass
class GlobalResult:
    all_rows: list[ResultRow] = field(default_factory=list)
    all_stats: list[RunStats] = field(default_factory=list)
    timed_out_runs: list[int] = field(default_factory=list)
    truncated_runs: list[int] = field(default_factory=list)

    def add_model_result(self, run_idx:int, model_result: ModelResult) -> GlobalResultType:
        self.all_rows.extend(model_result.rows)
        self.all_stats.append(model_result.stats)
        error = model_result.stats.get("error")
        truncated = model_result.stats.get("truncated")
        if error:
            self.timed_out_runs.append(run_idx + 1)
            raise Exception(error)
        elif truncated:
            self.truncated_runs.append(run_idx + 1)
            return GlobalResultType.TRUNCATED
        elif model_result.rows:
            return GlobalResultType.OK
        return GlobalResultType.NO_RESULTS


@dataclass
class ScoredNeedle:
    needle_key: str
    expected: int | str
    actual: str
    correct: int


@dataclass
class Summary:
    actual_real_needles: int
    actual_distractors: int
    actual_total: int


@dataclass
class QueryResult:
    parsed: dict[str, str]
    usage: ApiUsage | None
    latency_ms: float
    truncated: bool

@dataclass
class DebugContext:
    messages: list[Message] | None = None
    raw_response: ApiResponseBody | None = None
    response_text: str = ""
    model_name: str = ""


# ---------------------------------------------------------------------------
# Data generation
# ---------------------------------------------------------------------------

def generate_key(length: int = 8) -> str:
    """Generate a random uppercase string of fixed length."""
    return "".join(random.choices(string.ascii_uppercase, k=length))


def generate_value(val_min: int = 1000, val_max: int = 9999) -> int:
    """Generate a random integer, avoiding obviously round numbers."""
    while True:
        v = random.randint(val_min, val_max)
        if v % 100 != 0:
            return v


def build_haystack(
    n: int, key_len: int = 8, val_min: int = 10000, val_max: int = 99999
) -> tuple[str, list[HaystackPair]]:
    """Build a list of `n` random KEY = VALUE lines.

    Args:
    - n: number of pairs to generate
    - key_len: length of the random keys (default: 8)
    - val_min: minimum value (default: 10000)
    - val_max: maximum value (default: 99999)

    Returns:
    - haystack_text: a single string containing all pairs, one per line, in the format "KEY = VALUE
    - pairs: a list of (key, value) tuples for reference
    """
    pairs: list[HaystackPair] = []
    for _ in range(n):
        key = generate_key(key_len)
        value = generate_value(val_min, val_max)
        pairs.append((key, value))
    text = "\n".join(f"{k} = {v}" for k, v in pairs)
    return text, pairs


# ---------------------------------------------------------------------------
# Needle selection
# ---------------------------------------------------------------------------

def resolve_distractors_num(
    real_needles: int,
    distractors_num: int | None,
    distractor_pct: float | None,
) -> int:
    """Resolve the final distractor count from user-provided parameters.

    Priority: explicit num > pct > default 8%.
    """
    if distractors_num is not None:
        return distractors_num
    if distractor_pct is not None:
        return int(real_needles * distractor_pct)
    return int(real_needles * 0.08)


def shake_positions(positions: list[int], fuzz_pct: float) -> list[int]:
    """Add random jitter to each position as a fraction of the step size.

    Each position is shifted by ±(step * fuzz_pct), clamped to [0, max(positions)].
    The jitter is capped at fuzz_pct < 0.5 to guarantee no collisions:
    int() truncation ensures fuzz_abs < step/2, so two adjacent positions can never
    land on the same value or swap order. If fuzz_pct <= 0 or fewer than 2 positions,
    the original list is returned unchanged.

    Args:
        positions: Sorted list of positions to jitter.
        fuzz_pct: Fraction of the step size to use as jitter range (0.0-0.5 exclusive).

    Returns:
        New list of jittered positions. Order is preserved (no sorting needed)
        because fuzz_abs < step/2 guarantees no position swaps or collisions.

    Raises:
        ValueError: If fuzz_pct is outside [0, 0.5). Must be validated via
            validate_params() before calling this function.
    """
    if fuzz_pct < 0 or fuzz_pct >= 0.5:
        raise ValueError(
            f"fuzz_pct must be in [0, 0.5), got {fuzz_pct}. "
            "Use validate_params() to catch invalid values before running."
        )
    if fuzz_pct <= 0 or len(positions) < 2:
        return positions

    step = positions[1] - positions[0]
    fuzz_abs = int(step * fuzz_pct)

    max_pos = max(positions)
    result = []
    for p in positions:
        jitter = random.randint(-fuzz_abs, fuzz_abs)
        result.append(max(0, min(max_pos, p + jitter)))
    return result  # order guaranteed since fuzz_abs < step/2


def pick_needle_positions(n_total: int, n_needles: int) -> list[int]:
    """Pick `n_needles` indices uniformly spaced across [0, n_total).

    Args:
        n_total: Total number of positions in the haystack.
        n_needles: Number of needle positions to select.

    Returns:
        List of evenly-spaced needle positions.
    """
    if n_needles >= n_total:
        return list(range(n_total))
    step = n_total / n_needles
    return [int(i * step) for i in range(n_needles)]


def create_distractor_keys(
    pairs: list[HaystackPair],
    n_distractor: int,
    starting_index: int,
    key_length: int
) -> list[Needle]:
    """Create distractor needles with keys not present in the haystack.

    Args:
        pairs: List of haystack key-value pairs (used to avoid key collision).
        n_distractor: Number of distractor needles to create.
        starting_index: Starting index for distractor position assignment.
        key_length: Length of randomly generated distractor keys.

    Returns:
        List of Needle objects with is_distractor=True and expected=None.
    """
    # a set that contains all the keys of the haystack
    haystack_keys = set(k for k, _ in pairs)

    # contains the set of distractor keys: keys not in the haystack
    distractor_keys: set[str] = set()
    tries = 0
    while len(distractor_keys) < n_distractor and tries < n_distractor * 20:
        k = generate_key(key_length)
        if k not in haystack_keys:
            distractor_keys.add(k)
        tries += 1

    distractor_needles: list[Needle] = []

    # create the distractors using the remaining indexes of the positions
    idx = starting_index
    for key in distractor_keys:
        distractor_needles.append(Needle(
            index=idx,
            key=key,
            expected=None,
            is_distractor=True,
        ))
        idx = idx + 1

    return distractor_needles


def select_needles(
    pairs: list[HaystackPair],
    n_needles: int,
    fuzz: float = 0.49,
) -> list[Needle]:
    """Select real needles from haystack at uniformly spaced intervals.

    Args:
        pairs: List of all haystack key-value pairs.
        n_needles: Number of needles to extract from the haystack.
        fuzz: Fraction of the step size to jitter each needle position by (default: 0.49).
              Values >= 0.5 are silently skipped to avoid position collisions.

    Returns:
        List of Needle objects extracted from the haystack positions.
    """
    # list n_needles indexes to the pairs list taken at fixed intervals
    positions: list[int] = pick_needle_positions(len(pairs), n_needles)
    if fuzz != 0.0:
        positions = shake_positions(positions, fuzz)

    # extract the needles from the haystack according to the indexes in positions
    real_needles: list[Needle] = []
    for idx in positions:
        key, value = pairs[idx]
        real_needles.append(Needle(
            index=idx,
            key=key,
            expected=value,
            is_distractor=False,
        ))
    return real_needles


def generate_haystack_and_needles(
    haystack_num: int,
    needles_num: int,
    distractors_num: int,
    key_len: int,
    val_min: int,
    val_max: int,
    fuzz: float = 0.49,
) -> tuple[str, list[HaystackPair], list[Needle]]:
    """Generate haystack text, pair list, and shuffled needles."""
    # Build haystack
    haystack_text, pairs = build_haystack(
        haystack_num, key_len, val_min, val_max
    )

    # Select real needles from haystack at fixed intervals
    # Skip fuzz when needles are denser than 2 per haystack position —
    # positions are already fully packed, jitter would only cause collisions.
    effective_fuzz = 0.0 if needles_num > haystack_num * 2 else fuzz
    real_needles = select_needles(pairs, needles_num, effective_fuzz)

    # Create distractor needles (keys not in haystack)
    distractor_needles = create_distractor_keys(
        pairs,
        distractors_num,
        len(real_needles),
        key_len,
    )

    # Combine real and distractor needles, then shuffle
    all_needles = real_needles + distractor_needles
    random.shuffle(all_needles)

    return haystack_text, pairs, all_needles


# ---------------------------------------------------------------------------
# Prompt building
# ---------------------------------------------------------------------------

SYSTEM_PROMPT: str = (
    "You will be given a list of key-value pairs and later asked to retrieve "
    "the value for some of those keys. Read the pairs carefully."
)


def build_prompt(haystack_text: str, needles: list[Needle]) -> list[Message]:
    """Build a batch prompt with system message and all needle queries.

    Args:
        haystack_text: The full haystack text containing key=value pairs.
        needles: List of needles to query for.

    Returns:
        System and user messages forming the complete prompt.
    """
    lines: list[str] = [
        haystack_text,
        "",
        "Now answer for each key. Use the format KEY=VALUE on each line:",
    ]
    for needle in needles:
        lines.append(f"What is the value for {needle.key}?")
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": "\n".join(lines)},
    ]


# ---------------------------------------------------------------------------
# API query
# ---------------------------------------------------------------------------

def query_llama(
    endpoint: str,
    messages: list[Message],
    model: str | None = None,
    temperature: float = 0.0,
    max_tokens: int = 240000,
    timeout: int = 1200,
) -> tuple[ApiResponseBody, float]:
    """Send a request to the llama-server OpenAI-compatible endpoint.

    Args:
        endpoint: API URL for chat completions.
        messages: List of conversation messages.
        model: Model name (defaults to "local").
        temperature: Sampling temperature.
        max_tokens: Maximum tokens in the response.
        timeout: Request timeout in seconds.

    Returns:
        Tuple of (API response body, latency in milliseconds).
    Raises:
        HaystackQueryError: On HTTP errors, timeouts, or connection failures.
    """
    payload: Payload = {
        "model": model or "local",
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "stream": False,
    }
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        endpoint,
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    start = time.monotonic()
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            raw = resp.read().decode("utf-8")
            latency_ms = (time.monotonic() - start) * 1000
            result = json.loads(raw)
            return result, latency_ms
    except socket.timeout as e:
        latency_ms = (time.monotonic() - start) * 1000
        raise HaystackQueryError(f"Timed out after {latency_ms:.0f}ms") from e
    except urllib.error.HTTPError as e:
        latency_ms = (time.monotonic() - start) * 1000
        body = e.read().decode("utf-8", errors="replace")
        raise HaystackQueryError(f"HTTP {e.code}: {body}") from e
    except urllib.error.URLError as e:
        latency_ms = (time.monotonic() - start) * 1000
        raise HaystackQueryError(f"Connection failed: {e.reason}") from e


class HaystackQueryError(Exception):
    pass


# ---------------------------------------------------------------------------
# Response parsing
# ---------------------------------------------------------------------------

def parse_response(
    response_text: str,
    needle_keys: set[str],
) -> dict[str, str]:
    """Parse the model's response into a dict mapping key -> value.

    Only handles explicit KEY=VALUE lines. Non-integer values and keys
    not in `needle_keys` are ignored.

    Args:
        response_text: The raw text response from the model.
        needle_keys: Set of valid keys to accept.

    Returns:
        Dictionary mapping each recognized key to its integer string value.
    """
    if not response_text:
        return {}

    found: dict[str, str] = {}
    for line in response_text.strip().split("\n"):
        line = line.strip()
        if not line:
            continue
        if "=" in line:
            parts = line.split("=", 1)
            key = parts[0].strip()
            val = parts[1].strip()
            if key in needle_keys and val:
                try:
                    int(val)
                    found[key] = val
                except ValueError:
                    pass

    return found


def score_needles(needles: list[Needle], parsed: dict[str, str]) -> list[ScoredNeedle]:
    """Score each needle against parsed responses.

    For real needles, marks correct if actual matches expected value.
    For distractors, marks correct if actual is empty (no hallucination).

    Args:
        needles: List of needles to score.
        parsed: Dictionary of key -> actual value from the model.

    Returns:
        List of ScoredNeedle objects with correctness flags.
    """
    results: list[ScoredNeedle] = []
    for needle in needles:
        key = needle.key
        expected = needle.expected
        actual = parsed.get(key, "")
        if expected is not None:
            correct = 1 if actual == str(expected) else 0
        else:
            correct = 1 if actual == "" else 0
        results.append(ScoredNeedle(
            needle_key=key,
            expected=expected if expected is not None else "",
            actual=actual,
            correct=correct,
        ))
    return results


# ---------------------------------------------------------------------------
# Experiment runner
# ---------------------------------------------------------------------------

def _truncate(text: str, max_lines: int = 5, prefix: str = "...") -> str:
    """Show first and last N lines of a long text with ellipsis in between.

    Args:
        text: The text to truncate.
        max_lines: Number of lines to show at start and end.
        prefix: Text to insert between truncated sections.

    Returns:
        Truncated text, or the original if it fits within 2*max_lines.
    """
    lines = text.split("\n")
    if len(lines) <= max_lines * 2:
        return text
    first = "\n".join(lines[:max_lines])
    last = "\n".join(lines[-max_lines:])
    return f"{first}\n{prefix}\n{last}"


def query_model(
    config: Config,
    needles: list[Needle],
    haystack_text: str
) -> tuple[QueryResult, DebugContext]:
    """Query the model with all needles in a single prompt."""
    messages = build_prompt(haystack_text, needles)
    needle_keys: set[str] = {n.key for n in needles}

    raw_response, latency_ms = query_llama(
        config.endpoint,
        messages,
        model=config.model,
        temperature=config.temperature,
        max_tokens=config.max_tokens,
        timeout=config.timeout,
    )

    model_name = raw_response.get("model", "unknown") or "unknown"  # type: ignore[union-attr]
    response_text = raw_response["choices"][0]["message"].get("content", "")  # type: ignore[typeddict-item]
    debug = DebugContext(messages, raw_response, response_text, model_name)

    # Extract usage info if available (may be missing in error responses or older API versions)
    usage = raw_response.get("usage")

    # Consider the output truncated if the completion_tokens equals max_tokens,
    truncated: bool = (
        usage is not None
        and usage.get("completion_tokens") == config.max_tokens  # type: ignore[attr-defined]
    )

    # Parse the response text into a dict of key -> value for scoring.
    parsed: dict[str,str] = parse_response(response_text, needle_keys)
    result = QueryResult(
        parsed=parsed,
        usage=usage,
        latency_ms=latency_ms,
        truncated=truncated,
    )

    return result, debug


def _build_rows(
    run_index: int,
    needles: list[Needle],
    pairs: list[HaystackPair],
    parsed: dict[str, str],
    usage: ApiUsage | None = None,
    latency_ms: float | None = None,
) -> list[ResultRow]:
    """Build result rows from scored needles.

    Args:
        run_index: The experiment run number.
        needles: List of needles that were queried.
        pairs: Full list of haystack pairs (for depth calculation).
        parsed: Parsed key -> value results from the model.
        usage: Token usage info to attach to all rows.
        latency_ms: Total latency to attach to all rows.

    Returns:
        List of ResultRow objects with correctness, depth, and usage info.
    """
    rows: list[ResultRow] = []
    for needle, score in zip(needles, score_needles(needles, parsed)):
        depth_pct = round(
            needle.index / (len(pairs) - 1) * 100, 2
        ) if len(pairs) > 1 else 0.0
        rows.append(ResultRow(
            run=run_index,
            haystack_size=len(pairs),
            depth_pct=depth_pct,
            correct=score.correct,
            expected=score.expected,
            actual=score.actual,
            prompt_tokens=usage.get("prompt_tokens") if usage else None,
            completion_tokens=usage.get("completion_tokens") if usage else None,
            total_tokens=usage.get("total_tokens") if usage else None,
            latency_ms=latency_ms,
        ))
    return rows


def _print_results(
    run_index: int,
    needles: list[Needle],
    pairs: list[HaystackPair],
    parsed: dict[str, str],
    *,
    show: str | None = None,
    is_full: bool,
    debug: DebugContext,
) -> None:
    """Print experiment results and optional debug output.

    Args:
        run_index: The experiment run number.
        needles: List of needles that were queried.
        pairs: Full list of haystack pairs.
        parsed: Parsed key -> value results from the model.
        show: Which debug output to show ("prompt", "response", "all", or None).
        is_full: Whether to print full untruncated content.
        debug: Context containing messages, raw response, response text, and model name.
    """
    messages = debug.messages
    raw_response = debug.raw_response
    response_text = debug.response_text

    print()
    print("=" * 60)
    print(f"Run {run_index} — model={debug.model_name}")
    print("=" * 60)

    if messages is not None and (show in ("prompt", "all") or is_full):
        print(f"\n--- ALL MESSAGES ({len(messages)} messages) ---")
        for msg in messages:
            _print_message(msg, is_full)

    if (show in ("response", "all") or is_full):
        print("\n--- RAW RESPONSE (JSON) ---")
        if raw_response is not None:
            print(json.dumps(raw_response, indent=2, default=str))
        else:
            print("(no response received)")
        print("\n--- CONTENT FIELD ---")
        print(repr(response_text))

    print("\n--- PARSED RESULTS ---")
    scored = list(score_needles(needles, parsed))
    real_needles = []
    distractor_needles = []
    for needle, score in zip(needles, scored):
        if needle.is_distractor:
            distractor_needles.append((needle, score))
        else:
            real_needles.append((needle, score))

    for needle, score in sorted(real_needles, key=lambda x: x[0].index):
        depth = round(
            needle.index / (len(pairs) - 1) * 100, 1
        ) if len(pairs) > 1 else 0.0
        status = "OK" if score.correct else "FAIL"
        print(f"  [{status}] {needle.key} (depth {depth}%) "
              f"expected={score.expected!r} actual={score.actual!r}")

    if distractor_needles:
        print(f"\n  --- Distractors ({len(distractor_needles)}) ---")
        for needle, score in sorted(distractor_needles, key=lambda x: x[0].key):
            status = "OK" if score.correct else "FAIL"
            print(f"  [{status}] {needle.key} "
                  f"(distractor, expected=not_found actual={score.actual!r})")
    print("=" * 60)
    print()


def _print_message(msg: Message, is_full: bool) -> None:
    """Print a single message with optional truncation.

    Args:
        msg: The message to print (has role and content keys).
        is_full: If True, print full content; otherwise truncate to 5 lines.
    """
    role = msg["role"]
    content = msg["content"]
    print(f"\n--- MESSAGE ({role}, {len(content)} chars) ---")
    if is_full:
        print(content)
    else:
        print(_truncate(content, max_lines=5))


def run_single_experiment(
    run_index: int,
    config: Config,
    show: str | None = None
) -> ModelResult:
    """Run one complete experiment: generate, query, score, return rows."""

    # each run gets its onw seed
    random.seed(config.seed + run_index)

    haystack_text, pairs, needles = generate_haystack_and_needles(
        haystack_num=config.haystack_num,
        needles_num=config.needles_num,
        distractors_num=config.distractors_num,
        key_len=config.key_len,
        val_min=config.val_min,
        val_max=config.val_max,
        fuzz=config.fuzz,
    )

    result, debug = query_model(config, needles, haystack_text)

    rows: list[ResultRow] = _build_rows(
        run_index, needles, pairs, result.parsed, result.usage, result.latency_ms
    )
    total_tokens = result.usage["total_tokens"] if result.usage else 0
    total_latency = result.latency_ms
    total_completion = result.usage["completion_tokens"] if result.usage else 0

    stats: RunStats = {
        "total_tokens": total_tokens,
        "total_latency_ms": total_latency,
        "total_completion": total_completion,
        "error": None,
        "truncated": result.truncated,
        "model_name": debug.model_name,
    }
    if total_latency > 0:
        stats["tokens_per_sec"] = round(total_completion / (total_latency / 1000), 1)
    stats["avg_latency_ms"] = round(
        total_latency / len(needles), 1
    ) if needles else 0.0

    if show or config.full:
        _print_results(
            run_index, needles, pairs, result.parsed,
            show=show, is_full=config.full,
            debug=debug,
        )

    return ModelResult(rows, debug.model_name, stats)


def _validate_params(config: Config) -> None:
    """Validate parameters for haystack and needle generation."""
    if config.haystack_num < 1:
        raise ValueError(f"haystack_num must be >= 1, got {config.haystack_num}")
    if config.needles_num < 1:
        raise ValueError(f"needles_num must be >= 1, got {config.needles_num}")
    if config.distractors_num < 0:
        raise ValueError(
            f"distractors_num must be >= 0, got {config.distractors_num}"
        )
    if config.key_len < 1:
        raise ValueError(f"key_len must be >= 1, got {config.key_len}")
    if config.val_min > config.val_max:
        raise ValueError(f"val_min ({config.val_min}) > val_max ({config.val_max})")
    if config.fuzz < 0 or config.fuzz >= 0.5:
        raise ValueError(
            f"fuzz must be in [0, 0.5), got {config.fuzz}. "
            "Values >= 0.5 risk needle position collisions."
        )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def _parse_arguments() -> argparse.Namespace :
    """Parse CLI arguments, run experiments, and write results to CSV."""
    parser = argparse.ArgumentParser(
        description="Homogeneous Needle-in-a-Haystack benchmark"
    )
    parser.add_argument("--endpoint",
                        default="http://localhost:8080/v1/chat/completions",
                        help="llama-server endpoint (default: http://localhost:8080/v1/chat/completions)")
    parser.add_argument("--model", default=None,
                        help="Model name (auto-detected from API if omitted)")
    parser.add_argument("--key-len", type=int, default=8, choices=range(5, 13),
                        help="Key string length, 5-12 (default: 8)")
    parser.add_argument("--val-min", type=int, default=10000,
                        help="Minimum value (default: 10000)")
    parser.add_argument("--val-max", type=int, default=99999,
                        help="Maximum value (default: 99999)")
    parser.add_argument("--haystack-num", type=int, default=5000,
                        help="Total number of pairs in haystack (default: 5000)")
    parser.add_argument("--needles-num", type=int, default=100,
                        help="Number of needles to query (default: 100)")
    parser.add_argument("--distractor-pct", type=float, default=None,
                        help="Fraction of needles that are distractors (0.0-1.0, default: 0.08)")
    parser.add_argument("--distractors-num", type=int, default=None,
                        help="Exact number of distractors (mutually exclusive with --distractor-pct)")
    parser.add_argument("--temperature", type=float, default=0.0,
                        help="Sampling temperature (default: 0.0)")
    parser.add_argument("--max-tokens", type=int, default=240000,
                        help="Max tokens per response (default: 240000)")
    parser.add_argument("--timeout", type=int, default=7200,
                        help="Request timeout in seconds (default: 7200)")
    parser.add_argument("--show", choices=["prompt", "response", "all"],
                        help="Print prompt/response for debugging (prompt=response/all)")
    parser.add_argument("--full", action="store_true",
                        help="Print full untruncated prompt and response")
    parser.add_argument("--repeat", type=int, default=1,
                        help="Number of independent runs (default: 1)")
    parser.add_argument("--seed", type=int, default=None,
                        help="Base random seed (random if omitted)")
    parser.add_argument("--fuzz", type=float, default=0.49,
                        help="Jitter needle positions as fraction of step size (default: 0.49)")
    parser.add_argument("--output", default="results.csv",
                        help="Output CSV path (default: results.csv)")

    args: argparse.Namespace = parser.parse_args()
    return args


def _create_configuration() -> Config:
    args: argparse.Namespace = _parse_arguments()

    actual_real_needles = min(args.haystack_num, args.needles_num)
    actual_distractors = resolve_distractors_num(
        actual_real_needles, args.distractors_num, args.distractor_pct
    )
    actual_seed = args.seed if args.seed is not None else secrets.randbits(31)
    actual_show = args.show or "all"

    config = Config(
        endpoint=args.endpoint,
        output_filename=args.output,
        show=actual_show,
        model=args.model,
        key_len=args.key_len,
        val_min=args.val_min,
        val_max=args.val_max,
        haystack_num=args.haystack_num,
        needles_num=actual_real_needles,
        distractors_num=actual_distractors,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        timeout=args.timeout,
        repeat=args.repeat,
        full=args.full,
        seed=actual_seed,
        fuzz=args.fuzz,
    )

    _validate_params(config)

    return config


def _print_config(config: Config) -> None:
    """Print benchmark configuration parameters before starting runs."""
    print(f"Seed: {config.seed}")
    print(f"Endpoint: {config.endpoint}")
    total_needles = config.needles_num + config.distractors_num
    distractors_pct = int(config.distractors_num * 100.0 / total_needles)
    print(f"Haystack: {config.haystack_num} pairs, {total_needles} needles "
          f"({config.needles_num} real + {config.distractors_num} distractors, "
          f"{distractors_pct*100:.0f}% distractors)")
    print(f"Output: {config.output_filename}")
    print()


def main() -> None:
    start_time = datetime.datetime.now()
    print(f"Start time: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")

    config: Config = _create_configuration()
    _print_config(config)

    global_result = GlobalResult()

    for run_idx in range(config.repeat):
        print(f"Run {run_idx + 1}/{config.repeat}...", end=" ", flush=True)
        try:
            model_result: ModelResult = run_single_experiment(
                run_idx, config, config.show
            )

            status = global_result.add_model_result(run_idx, model_result)
            match status:
                case GlobalResultType.OK:
                    correct_count = sum(1 for r in model_result.rows if r.correct == 1)
                    print(f"model={model_result.model_name}, "
                        f"accuracy={correct_count}/{len(model_result.rows)} "
                        f"({100*correct_count/len(model_result.rows):.1f}%)")
                case GlobalResultType.TRUNCATED:
                    print(f"model={model_result.model_name}, OUTPUT TRUNCATED")
                case GlobalResultType.NO_RESULTS:
                    print(f"model={model_result.model_name}, NO RESULTS")

        except HaystackQueryError as e:
            global_result.timed_out_runs.append(run_idx + 1)
            print()
            print("=" * 60)
            print(f"QUERY ERROR: Run {run_idx + 1} — {e}")
            print("=" * 60)
            print()
        except Exception as e:
            print(f"FAILED: {e}", file=sys.stderr)

    with open(config.output_filename, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        writer.writerows(asdict(r) for r in global_result.all_rows)

    _print_summary(
        config=config,
        result=global_result,
        start_time=start_time,
    )


def _print_summary(
    config: Config,
    result: GlobalResult,
    start_time: datetime.datetime
) -> None:
    """Print a comprehensive summary of the experiment."""
    print()
    print("=" * 60)
    print("SUMMARY")
    print("=" * 60)

    # Configuration
    print("\nConfiguration:")
    print(f"  Haystack size:     {config.haystack_num}")
    print(f"  Num needles:       {config.needles_num + config.distractors_num} ({config.needles_num} real + {config.distractors_num} distractors)")
    distractor_pct = config.distractors_num / (config.needles_num + config.distractors_num)
    print(f"  Distractors:       {config.distractors_num} (exact count) ({distractor_pct * 100:.1f}%)")
    print(f"  Key length:        {config.key_len}")
    print(f"  Value range:       {config.val_min} - {config.val_max}")
    print(f"  Temperature:       {config.temperature}")
    print(f"  Max tokens:        {config.max_tokens}")
    print(f"  Timeout:           {config.timeout}s")
    print(f"  Seed:              {config.seed}")
    print(f"  Repeat:            {config.repeat}")
    print(f"  Endpoint:          {config.endpoint}")

    # Per-run results
    if result.all_stats:
        print("\nPer-run results:")
        run_rows = {}
        for r in result.all_rows:
            run_rows.setdefault(r.run, []).append(r)
        for i, stats in enumerate(result.all_stats):
            model_name = stats.get("model_name", "N/A") or "N/A"
            total_tokens = stats.get("total_tokens", 0) or 0
            avg_lat = stats.get("avg_latency_ms", 0) or 0
            tps = stats.get("tokens_per_sec", 0)
            error = stats.get("error")
            truncated = stats.get("truncated")
            run_rows_for_this = run_rows.get(i + 1, [])
            correct_count = sum(1 for r in run_rows_for_this if r.correct == 1)
            row_count = len(run_rows_for_this)

            print(f"  Run:       {i + 1}")
            print(f"  Model:     {model_name}")
            if error:
                print(f"  Status:    ERROR ({error})")
            elif truncated:
                print("  Status:    OUTPUT TRUNCATED")
            else:
                print(f"  Accuracy:    {correct_count}/{row_count} "
                      f"({100*correct_count/row_count:.1f}%)" if row_count > 0 else "  Accuracy:    N/A")
                print(f"  Tokens:    {total_tokens}")
                print(f"  Avg Lat:   {avg_lat:.0f}ms")
                if tps:
                    print(f"  TPS:       {tps:.1f}")
            print()

    # Overall results
    total = len(result.all_rows)
    correct = sum(1 for r in result.all_rows if r.correct == 1)
    if total > 0:
        total_tokens_all = sum((float(s.get("total_tokens") or 0) for s in result.all_stats), 0.0)
        total_latency_all = sum((float(s.get("total_latency_ms") or 0) for s in result.all_stats), 0.0)
        total_completion_all = sum((float(s.get("total_completion") or 0) for s in result.all_stats), 0.0)
        print("\nOverall results:")
        print(f"  Accuracy:            {correct}/{total} ({100*correct/total:.1f}%)")
        print(f"  Total tokens:        {total_tokens_all}")
        print(f"  Total latency:       {total_latency_all:.0f}ms"
              + (f" ({total_latency_all/60000:.1f} min)" if total_latency_all > 60000 else ""))
        if total_latency_all > 0:
            print(f"  Avg tokens/sec:      {total_completion_all/(total_latency_all/1000):.1f}")

    # Issues
    if result.timed_out_runs or result.truncated_runs:
        print("\nIssues:")
        if result.timed_out_runs:
            print(f"  Timed out runs:    {result.timed_out_runs}")
        if result.truncated_runs:
            print(f"  Truncated runs:    {result.truncated_runs}")
            print("  (completion_tokens reached max_tokens — model was cut off)")

    print(f"\nOutput: {config.output_filename}")

    elapsed = datetime.datetime.now() - start_time if start_time else datetime.timedelta(0)
    minutes, remainder = divmod(int(elapsed.total_seconds()), 60)
    hours, minutes = divmod(minutes, 60)
    if hours > 0:
        print(f"Elapsed: {hours}h {minutes}m {remainder}s")
    else:
        print(f"Elapsed: {minutes}m {remainder}s")
    print("=" * 60)


if __name__ == "__main__":
    main()
