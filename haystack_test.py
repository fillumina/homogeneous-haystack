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
import os
import urllib.error
import urllib.request
from dataclasses import asdict, dataclass, field
from typing import NamedTuple, TypedDict
from enum import Enum, auto


class HaystackPair(NamedTuple):
    key: str
    value: int


class Message(NamedTuple):
    role: str
    content: str


class Verbosity(Enum):
    MINIMAL = "minimal"
    MEDIUM = "medium"
    FULL = "full"
    DEBUG = "debug"


MAX_FUZZ: float = 0.5
MAX_DISTRACTOR_KEY_TRIES: int = 20

CSV_COLUMNS: list[str] = [
    "run", "haystack_size", "depth_pct", "correct", "expected", "actual",
    "prompt_tokens", "completion_tokens", "total_tokens", "latency_ms",
]

_SUMMARY_PREFIX = "ctx-pos-"
_SUMMARY_BUCKET_COUNT = 10
_SUMMARY_BUCKET_COLS = [f"{_SUMMARY_PREFIX}{i*10}" for i in range(_SUMMARY_BUCKET_COUNT)]

SUMMARY_CSV_COLUMNS: list[str] = [
    "timestamp",
    "model_name",
    "k_quant",
    "v_quant",
    "prompt_tokens",
    "completion_tokens",
    "total_tokens",
    "total_pairs",
    "needles_num",
    "distractors_num",
    "needle_success_pct",
    "distractor_success_pct",
    "tokens_per_sec",
    "total_time_sec",
    *_SUMMARY_BUCKET_COLS,
    "distractor_failures",
    "note",
]


class ApiUsage(TypedDict):
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int


class ApiResponseBody(TypedDict, total=False):
    """OpenAI-compatible API response body.

    Success responses contain model, choices, and usage. Error responses
    contain an error dict. All keys are optional since the API may return
    a subset at runtime.
    """
    model: str
    choices: list[dict]
    usage: ApiUsage
    error: dict[str, str]


class Payload(TypedDict):
    model: str
    messages: list[Message]
    temperature: float
    max_tokens: int
    stream: bool


class RunStats(TypedDict):
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
    is_distractor: bool = False
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
class ApiConfig:
    endpoint: str
    temperature: float
    max_tokens: int
    timeout: int


@dataclass
class HaystackConfig:
    haystack_num: int
    needles_num: int
    distractors_num: int
    key_len: int
    val_min: int
    val_max: int
    fuzz: float = 0.49
    seed: int = 42


@dataclass
class OutputConfig:
    output_filename: str
    verbosity: str
    k_quant: str
    v_quant: str
    note: str
    timestamp: str = ""


@dataclass
class ExecutionConfig:
    repeat: int = 1
    stop_on_error: bool = False


@dataclass
class Config:
    api: ApiConfig
    haystack: HaystackConfig
    output: OutputConfig
    execution: ExecutionConfig


@dataclass
class DebugContext:
    messages: list[Message] | None = None
    raw_response: ApiResponseBody | None = None
    response_text: str = ""
    model_name: str = ""


@dataclass
class ModelResult:
    rows: list[ResultRow]
    model_name: str
    stats: RunStats
    pairs: list[HaystackPair] = field(default_factory=list)
    needles: list[Needle] = field(default_factory=list)
    parsed: dict[str, str] = field(default_factory=dict)
    debug: DebugContext = field(default_factory=DebugContext)


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
    failed_runs: list[int] = field(default_factory=list)

    def add_model_result(self, run_idx:int, model_result: ModelResult) -> GlobalResultType:
        self.all_rows.extend(model_result.rows)
        self.all_stats.append(model_result.stats)
        error = model_result.stats["error"]
        truncated = model_result.stats["truncated"]
        if error:
            raise HaystackQueryError(error)
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
class ExperimentSummary:
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    total_pairs: int
    needles_num: int
    distractors_num: int
    needle_success_pct: float
    distractor_success_pct: float
    tokens_per_sec: float
    total_time_sec: float
    ctx_pos_buckets: list[int]
    distractor_failures: int


@dataclass
class QueryResult:
    parsed: dict[str, str]
    usage: ApiUsage | None
    latency_ms: float
    truncated: bool


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
        pairs.append(HaystackPair(key, value))
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
    The jitter is capped at fuzz_pct < MAX_FUZZ to guarantee no collisions:
    int() truncation ensures fuzz_abs < step/2, so two adjacent positions can never
    land on the same value or swap order. If fuzz_pct <= 0 or fewer than 2 positions,
    the original list is returned unchanged.

    Args:
        positions: Sorted list of positions to jitter.
        fuzz_pct: Fraction of the step size to use as jitter range (0.0 to MAX_FUZZ exclusive).

    Returns:
        New list of jittered positions. Order is preserved (no sorting needed)
        because fuzz_abs < step/2 guarantees no position swaps or collisions.

    Raises:
        ValueError: If fuzz_pct is outside [0, MAX_FUZZ). Must be validated via
            validate_params() before calling this function.
    """
    if fuzz_pct < 0 or fuzz_pct >= MAX_FUZZ:
        raise ValueError(
            f"fuzz_pct must be in [0, {MAX_FUZZ}), got {fuzz_pct}. "
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
    haystack_keys = set(k for k, _ in pairs)

    distractor_keys: set[str] = set()
    tries = 0
    while len(distractor_keys) < n_distractor and tries < n_distractor * MAX_DISTRACTOR_KEY_TRIES:
        k = generate_key(key_length)
        if k not in haystack_keys:
            distractor_keys.add(k)
        tries += 1

    distractor_needles: list[Needle] = []
    idx = starting_index
    for key in distractor_keys:
        distractor_needles.append(Needle(
            index=idx,
            key=key,
            expected=None,
            is_distractor=True,
        ))
        idx += 1

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
              Values >= MAX_FUZZ are silently skipped to avoid position collisions.

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
        Message(role="system", content=SYSTEM_PROMPT),
        Message(role="user", content="\n".join(lines)),
    ]


# ---------------------------------------------------------------------------
# API query
# ---------------------------------------------------------------------------

def _calc_latency(start: float) -> float:
    """Calculate latency in milliseconds from a monotonic start time."""
    return (time.monotonic() - start) * 1000

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
            return json.loads(raw), _calc_latency(start)
    except socket.timeout as e:
        raise HaystackQueryError(f"Timed out after {_calc_latency(start):.0f}ms") from e
    except urllib.error.HTTPError as e:
        body = e.read().decode("utf-8", errors="replace")
        raise HaystackQueryError(f"HTTP {e.code}: {body}") from e
    except urllib.error.URLError as e:
        raise HaystackQueryError(f"Connection failed: {e.reason}") from e
    except (ConnectionResetError, BrokenPipeError) as e:
        raise HaystackQueryError(f"Connection reset: {e}") from e
    except OSError as e:
        raise HaystackQueryError(f"Connection failed: {e}") from e


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



def _pad_right(value: str | int, width: int) -> str:
    """Right-align a value in a field of given width."""
    return str(value).rjust(width)

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
        config.api.endpoint,
        messages,
        temperature=config.api.temperature,
        max_tokens=config.api.max_tokens,
        timeout=config.api.timeout,
    )

    model_name = raw_response.get("model", "unknown") or "unknown"
    response_text = raw_response.get("choices", [{}])[0].get("message", {}).get("content", "")
    debug = DebugContext(messages, raw_response, response_text, model_name)

    # Extract usage info if available (may be missing in error responses or older API versions)
    usage = raw_response.get("usage")

    # Consider the output truncated if the completion_tokens equals max_tokens,
    truncated: bool = False
    if usage is not None:
        truncated = usage.get("completion_tokens") == config.api.max_tokens

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
            is_distractor=needle.is_distractor,
            prompt_tokens=usage.get("prompt_tokens") if usage else None,
            completion_tokens=usage.get("completion_tokens") if usage else None,
            total_tokens=usage.get("total_tokens") if usage else None,
            latency_ms=latency_ms,
        ))
    return rows


def _print_debug_context(debug: DebugContext) -> None:
    """Print debug output: messages and raw response.

    Args:
        debug: Context containing messages, raw response, response text, and model name.
    """
    messages = debug.messages
    raw_response = debug.raw_response
    response_text = debug.response_text

    if messages is not None:
        print(f"\n--- ALL MESSAGES ({len(messages)} messages) ---")
        for msg in messages:
            _print_message(msg, is_full=True)

    if raw_response is not None:
        print("\n--- RAW RESPONSE (JSON) ---")
        print(json.dumps(raw_response, indent=2, default=str))
    else:
        print("\n--- RAW RESPONSE (JSON) ---")
        print("(no response received)")
    print("\n--- CONTENT FIELD ---")
    print(repr(response_text))


def _print_message(msg: Message, is_full: bool) -> None:
    """Print a single message with optional truncation.

    Args:
        msg: The message to print (has role and content keys).
        is_full: If True, print full content; otherwise truncate to 5 lines.
    """
    role = msg.role
    content = msg.content
    print(f"\n--- MESSAGE ({role}, {len(content)} chars) ---")
    if is_full:
        print(content)
    else:
        print(_truncate(content, max_lines=5))


def run_single_experiment(
    run_index: int,
    config: Config,
) -> ModelResult:
    """Run one complete experiment: generate, query, score, return rows.

    Returns data only — printing is handled by the caller (main).
    """
    # each run gets its own seed
    random.seed(config.haystack.seed + run_index)

    haystack_text, pairs, needles = generate_haystack_and_needles(
        haystack_num=config.haystack.haystack_num,
        needles_num=config.haystack.needles_num,
        distractors_num=config.haystack.distractors_num,
        key_len=config.haystack.key_len,
        val_min=config.haystack.val_min,
        val_max=config.haystack.val_max,
        fuzz=config.haystack.fuzz,
    )

    result, debug = query_model(config, needles, haystack_text)

    rows: list[ResultRow] = _build_rows(
        run_index, needles, pairs, result.parsed, result.usage, result.latency_ms
    )
    total_tokens = result.usage["total_tokens"] if result.usage else 0
    total_latency = result.latency_ms
    total_completion = result.usage["completion_tokens"] if result.usage else 0

    tokens_per_sec = round(total_completion / (total_latency / 1000), 1) if total_latency > 0 else 0.0
    stats: RunStats = {
        "total_tokens": total_tokens,
        "total_latency_ms": total_latency,
        "total_completion": total_completion,
        "error": None,
        "truncated": result.truncated,
        "model_name": debug.model_name,
        "tokens_per_sec": tokens_per_sec,
        "avg_latency_ms": round(total_latency / len(needles), 1) if needles else 0.0,
    }

    return ModelResult(
        rows=rows,
        model_name=debug.model_name,
        stats=stats,
        pairs=pairs,
        needles=needles,
        parsed=result.parsed,
        debug=debug,
    )


def _validate_params(config: Config) -> None:
    """Validate parameters for haystack and needle generation."""
    if config.haystack.haystack_num < 1:
        raise ValueError(f"haystack_num must be >= 1, got {config.haystack.haystack_num}")
    if config.haystack.needles_num < 1:
        raise ValueError(f"needles_num must be >= 1, got {config.haystack.needles_num}")
    if config.haystack.distractors_num < 0:
        raise ValueError(
            f"distractors_num must be >= 0, got {config.haystack.distractors_num}"
        )
    if config.haystack.key_len < 1:
        raise ValueError(f"key_len must be >= 1, got {config.haystack.key_len}")
    if config.haystack.val_min > config.haystack.val_max:
        raise ValueError(f"val_min ({config.haystack.val_min}) > val_max ({config.haystack.val_max})")
    if config.haystack.fuzz < 0 or config.haystack.fuzz >= MAX_FUZZ:
        raise ValueError(
            f"fuzz must be in [0, {MAX_FUZZ}), got {config.haystack.fuzz}. "
            f"Values >= {MAX_FUZZ} risk needle position collisions."
        )


def compute_experiment_summary(
    global_result: GlobalResult,
) -> list[ExperimentSummary]:
    """Compute aggregated summary rows from all experiment runs.

    Groups per-needle ResultRows by run index and computes success rates,
    failure distribution across context intervals, and speed metrics.
    """
    # Group rows by run
    run_rows: dict[int, list[ResultRow]] = {}
    for row in global_result.all_rows:
        run_rows.setdefault(row.run, []).append(row)

    summaries: list[ExperimentSummary] = []
    for run_idx in range(len(global_result.all_stats)):
        rows = run_rows.get(run_idx, [])
        if not rows:
            continue

        stats = global_result.all_stats[run_idx]
        total_tokens = stats["total_tokens"] or 0
        total_latency_ms = stats["total_latency_ms"] or 0
        total_completion = stats["total_completion"] or 0
        total_prompt = 0
        total_completion_api = 0

        real_needles = []
        distractor_needles = []
        for row in rows:
            if row.is_distractor:
                distractor_needles.append(row)
            else:
                real_needles.append(row)

        # Collect needle keys from the experiment to match with needles list
        # We use the scored results to determine correctness
        total_pairs = rows[0].haystack_size if rows else 0

        # Compute needle success rate
        needle_correct = sum(1 for r in real_needles if r.correct == 1)
        needle_total = len(real_needles)
        needle_success_pct = (needle_correct / needle_total * 100) if needle_total > 0 else 0.0

        # Compute distractor success rate
        distractor_correct = sum(1 for r in distractor_needles if r.correct == 1)
        distractor_total = len(distractor_needles)
        distractor_success_pct = (distractor_correct / distractor_total * 100) if distractor_total > 0 else 0.0

        # Count distractor failures
        distractor_failures = distractor_total - distractor_correct

        # Compute tokens/sec and total time
        tokens_per_sec = stats["tokens_per_sec"]
        total_time_sec = total_latency_ms / 1000.0

        # Collect usage from first row
        if rows:
            total_prompt = rows[0].prompt_tokens or 0
            total_completion_api = rows[0].completion_tokens or 0

        # Bucket failed needles by depth percentage into 10% intervals
        buckets = [0] * 10  # 0-10, 10-20, ..., 90-100
        for row in real_needles:
            if row.correct == 0:
                depth = row.depth_pct
                bucket_idx = int(depth // 10)
                if bucket_idx >= 10:
                    bucket_idx = 9
                buckets[bucket_idx] += 1

        summaries.append(ExperimentSummary(
            prompt_tokens=total_prompt,
            completion_tokens=total_completion_api,
            total_tokens=total_tokens,
            total_pairs=total_pairs,
            needles_num=needle_total,
            distractors_num=distractor_total,
            needle_success_pct=round(needle_success_pct, 2),
            distractor_success_pct=round(distractor_success_pct, 2),
            tokens_per_sec=round(tokens_per_sec, 1) if tokens_per_sec else 0.0,
            total_time_sec=round(total_time_sec, 3),
            ctx_pos_buckets=buckets,
            distractor_failures=distractor_failures,
        ))

    return summaries


# ---------------------------------------------------------------------------
# Per-run summary helpers
# ---------------------------------------------------------------------------


def _print_run_minimal(
    run_number: int,
    run_total: int,
    summary: ExperimentSummary,
    stats: RunStats,
    run_rows: list[ResultRow],
    timestamp: str,
) -> None:
    """Print a single-line summary for minimal verbosity.

    Format: timestamp Run N/M: Needles X% | Distractors Y%
    """
    needle_rows = [r for r in run_rows if not r.is_distractor]
    distractor_rows = [r for r in run_rows if r.is_distractor]

    error = stats["error"]
    truncated = stats["truncated"]

    if error:
        print(f"{timestamp} Run {run_number}/{run_total}: ERROR ({error})")
    elif truncated:
        needle_pct = f"{100*sum(1 for r in needle_rows if r.correct==1)/len(needle_rows):.1f}%" if needle_rows else "N/A"
        print(f"{timestamp} Run {run_number}/{run_total}: TRUNCATED | Needles {needle_pct}")
    else:
        needle_pct = f"{100*sum(1 for r in needle_rows if r.correct==1)/len(needle_rows):.1f}%" if needle_rows else "N/A"
        distractor_pct = f"{100*sum(1 for r in distractor_rows if r.correct==1)/len(distractor_rows):.1f}%" if distractor_rows else "N/A"
        print(f"{timestamp} Run {run_number}/{run_total}: Needles {needle_pct} | Distractors {distractor_pct}")


def _print_run_detail(
    run_number: int,
    summary: ExperimentSummary,
    stats: RunStats,
    run_rows: list[ResultRow],
    timestamp: str | None = None,
) -> None:
    """Print the detailed per-run result block (needle/distractor accuracy, tokens, latency, failure buckets).

    Args:
        run_number: 1-based run number for display.
        summary: ExperimentSummary for this run.
        stats: RunStats for this run.
        run_rows: All ResultRow objects for this run.
        timestamp: Optional timestamp string for this run. If None, no timestamp line is printed.
    """
    needle_rows = [r for r in run_rows if not r.is_distractor]
    distractor_rows = [r for r in run_rows if r.is_distractor]
    needle_correct = sum(1 for r in needle_rows if r.correct == 1)
    distractor_correct = sum(1 for r in distractor_rows if r.correct == 1)

    total_tokens = stats["total_tokens"] or 0
    avg_lat = stats["avg_latency_ms"] or 0
    tps = stats["tokens_per_sec"]
    error = stats["error"]
    truncated = stats["truncated"]

    if timestamp is not None:
        print(f"  Timestamp:       {timestamp}")
    if error:
        print(f"  Status:          ERROR ({error})")
    elif truncated:
        print("  Status:          OUTPUT TRUNCATED")
    else:
        if needle_rows:
            print(f"  Needle accuracy:     {needle_correct}/{len(needle_rows)} "
                  f"({100*needle_correct/len(needle_rows):.1f}%)")
        else:
            print("  Needle accuracy:     N/A")
        if distractor_rows:
            print(f"  Distractor accuracy: {distractor_correct}/{len(distractor_rows)} "
                  f"({100*distractor_correct/len(distractor_rows):.1f}%)")
        else:
            print("  Distractor accuracy: N/A (no distractors)")
        print(f"  Tokens:          {total_tokens}")
        print(f"  Avg Lat:         {avg_lat:.0f}ms")
        print(f"  TPS:             {tps:.1f}")
        print(f"  Time:            {summary.total_time_sec:.1f}s")

    # Print failure buckets in a 2-row table (only if there are failures)
    if any(summary.ctx_pos_buckets) or summary.distractor_failures > 0:
        labels = ["0-10%", "10-20%", "20-30%", "30-40%", "40-50%",
                  "50-60%", "60-70%", "70-80%", "80-90%", "90-100%"]
        max_val = max(summary.ctx_pos_buckets) if summary.ctx_pos_buckets else 0
        col_width = max(len(l) for l in labels) + 1
        col_width = max(col_width, len(str(max_val)) + 1)
        print("  Failures by 10% interval:")
        print("    " + " ".join(_pad_right(l, col_width) for l in labels))
        print("    " + " ".join(_pad_right(b, col_width) for b in summary.ctx_pos_buckets))


def _print_needle_results(
    run_index: int,
    needles: list[Needle],
    pairs: list[HaystackPair],
    parsed: dict[str, str],
    *,
    verbosity: str,
    debug: DebugContext,
) -> None:
    """Print per-needle detail: failed only for medium, all for full/debug.

    Args:
        run_index: The experiment run number.
        needles: List of needles that were queried.
        pairs: Full list of haystack pairs.
        parsed: Parsed key -> value results from the model.
        verbosity: Current verbosity level.
        debug: Context containing model name.
    """
    scored = list(score_needles(needles, parsed))
    real_needles = []
    distractor_needles = []
    for needle, score in zip(needles, scored):
        if needle.is_distractor:
            distractor_needles.append((needle, score))
        else:
            real_needles.append((needle, score))

    if verbosity in ("medium", "full", "debug"):
        real_needles_sorted = sorted(real_needles, key=lambda x: x[0].index)
        distractor_needles_sorted = sorted(distractor_needles, key=lambda x: x[0].key)

        if verbosity == "medium":
            real_needles_sorted = [(n, s) for n, s in real_needles_sorted if not s.correct]
            distractor_needles_sorted = [(n, s) for n, s in distractor_needles_sorted if not s.correct]

        if real_needles_sorted:
            for needle, score in real_needles_sorted:
                depth = round(
                    needle.index / (len(pairs) - 1) * 100, 1
                ) if len(pairs) > 1 else 0.0
                status = "OK" if score.correct else "FAIL"
                print(f"  [{status}] {needle.key} (depth {depth}%) "
                      f"expected={score.expected!r} actual={score.actual!r}")

        if distractor_needles_sorted:
            print()
            for needle, score in distractor_needles_sorted:
                status = "OK" if score.correct else "FAIL"
                print(f"  [{status}] {needle.key} "
                      f"(distractor, expected=not_found actual={score.actual!r})")


def _build_summary_row(config: Config, model_name: str, summary: ExperimentSummary) -> dict:
    """Build a single experiment summary row as a dictionary."""
    row = {
        "timestamp": config.output.timestamp,
        "model_name": model_name,
        "k_quant": config.output.k_quant,
        "v_quant": config.output.v_quant,
        "prompt_tokens": summary.prompt_tokens,
        "completion_tokens": summary.completion_tokens,
        "total_tokens": summary.total_tokens,
        "total_pairs": summary.total_pairs,
        "needles_num": summary.needles_num,
        "distractors_num": summary.distractors_num,
        "needle_success_pct": f"{summary.needle_success_pct:.2f}",
        "distractor_success_pct": f"{summary.distractor_success_pct:.2f}",
        "tokens_per_sec": summary.tokens_per_sec,
        "total_time_sec": summary.total_time_sec,
        "distractor_failures": summary.distractor_failures,
        "note": config.output.note,
    }
    for i, col in enumerate(_SUMMARY_BUCKET_COLS):
        row[col] = summary.ctx_pos_buckets[i]
    return row


def _write_summary_csv(config: Config, model_name: str, summaries: list[ExperimentSummary]) -> None:
    """Write experiment summaries to CSV in append mode.

    Opens the file in append mode. Writes the header row only if the
    file does not exist or is empty. Writes all rows in a single open block.
    """
    write_header = False
    if not os.path.exists(config.output.output_filename) or os.path.getsize(config.output.output_filename) == 0:
        write_header = True

    with open(config.output.output_filename, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=SUMMARY_CSV_COLUMNS)
        if write_header:
            writer.writeheader()
        for summary in summaries:
            row = _build_summary_row(config, model_name, summary)
            writer.writerow(row)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def _parse_arguments() -> argparse.Namespace :
    """Parse CLI arguments and return an argparse.Namespace."""
    parser = argparse.ArgumentParser(
        description="Homogeneous Needle-in-a-Haystack benchmark"
    )
    parser.add_argument("--endpoint",
                        default="http://localhost:8080/v1/chat/completions",
                        help="llama-server endpoint (default: http://localhost:8080/v1/chat/completions)")
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
    parser.add_argument("--verbosity", "-v", choices=["minimal", "medium", "full", "debug"],
                        default="medium",
                        help="Output verbosity level one of 'minimal', 'medium', 'full', 'debug' (default: medium)")
    parser.add_argument("--repeat", type=int, default=1,
                        help="Number of independent runs (default: 1)")
    parser.add_argument("--stop-on-error", action="store_true",
                        help="Stop after the first query error (instead of continuing through all repeats)")
    parser.add_argument("--seed", type=int, default=None,
                        help="Base random seed (random if omitted)")
    parser.add_argument("--fuzz", type=float, default=0.49,
                        help="Jitter needle positions as fraction of step size (default: 0.49)")
    parser.add_argument("--output", default="results.csv",
                        help="Output CSV path (default: results.csv)")
    parser.add_argument("--k-quant", default="",
                        help="KV-cache key quantization type (e.g. Q8_0, Q4_0, turbo4)")
    parser.add_argument("--v-quant", default="",
                        help="KV-cache value quantization type (e.g. Q8_0, Q4_0, turbo4)")
    parser.add_argument("--note", default="",
                        help="Freeform note about this experiment")

    args: argparse.Namespace = parser.parse_args()
    return args


def _create_configuration() -> Config:
    args: argparse.Namespace = _parse_arguments()

    actual_real_needles = min(args.haystack_num, args.needles_num)
    actual_distractors = resolve_distractors_num(
        actual_real_needles, args.distractors_num, args.distractor_pct
    )
    actual_seed = args.seed if args.seed is not None else secrets.randbits(31)

    config = Config(
        api=ApiConfig(
            endpoint=args.endpoint,
            temperature=args.temperature,
            max_tokens=args.max_tokens,
            timeout=args.timeout,
        ),
        haystack=HaystackConfig(
            haystack_num=args.haystack_num,
            needles_num=actual_real_needles,
            distractors_num=actual_distractors,
            key_len=args.key_len,
            val_min=args.val_min,
            val_max=args.val_max,
            fuzz=args.fuzz,
            seed=actual_seed,
        ),
        output=OutputConfig(
            output_filename=args.output,
            verbosity=args.verbosity,
            k_quant=args.k_quant,
            v_quant=args.v_quant,
            note=args.note,
            timestamp=datetime.datetime.now().strftime("%Y-%m-%dT%H:%M:%S"),
        ),
        execution=ExecutionConfig(
            repeat=args.repeat,
            stop_on_error=args.stop_on_error,
        ),
    )

    _validate_params(config)

    return config


def _print_config(config: Config) -> None:
    """Print benchmark configuration parameters before starting runs."""
    print(f"Seed: {config.haystack.seed}")
    print(f"Endpoint: {config.api.endpoint}")
    total_needles = config.haystack.needles_num + config.haystack.distractors_num
    distractors_pct = int(config.haystack.distractors_num * 100.0 / total_needles)
    print(f"Haystack: {config.haystack.haystack_num} pairs, {total_needles} needles "
          f"({config.haystack.needles_num} real + {config.haystack.distractors_num} distractors, "
          f"{distractors_pct:.0f}% distractors)")
    print(f"Output: {config.output.output_filename}")
    print()


def main() -> None:
    config: Config = _create_configuration()
    start_time = datetime.datetime.now()
    print(f"Start time: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    _print_config(config)

    global_result = GlobalResult()
    is_minimal = config.output.verbosity == "minimal"
    is_debug = config.output.verbosity == "debug"

    run_experiment_loop(config, global_result, is_minimal, is_debug)

    summaries = compute_experiment_summary(global_result)
    model_name = global_result.all_stats[0].get("model_name", "unknown") if global_result.all_stats else "unknown"
    _write_summary_csv(config, model_name, summaries)

    _print_summary(
        config=config,
        result=global_result,
        summaries=summaries,
        start_time=start_time,
        model_name=model_name,
    )


def run_experiment_loop(
    config: Config,
    global_result: GlobalResult,
    is_minimal: bool,
    is_debug: bool,
) -> None:
    """Run all experiment repeats, handling results and errors."""
    for run_idx in range(config.execution.repeat):
        try:
            model_result: ModelResult = run_single_experiment(
                run_idx, config
            )

            status = global_result.add_model_result(run_idx, model_result)
            summaries = compute_experiment_summary(global_result)
            summary = summaries[-1] if summaries else None
            stats = global_result.all_stats[run_idx] if global_result.all_stats else {}
            run_rows_for_this = [r for r in global_result.all_rows if r.run == run_idx]

            if summary is not None and stats:
                run_timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                # Print per-run output based on verbosity
                print(f"Run {run_idx + 1}/{config.execution.repeat}: Done!")
                if is_minimal:
                    _print_run_minimal(
                        run_idx + 1, config.execution.repeat,
                        summary, stats, run_rows_for_this, run_timestamp
                    )
                else:
                    _print_run_detail(run_idx + 1, summary, stats, run_rows_for_this, run_timestamp)
                    _print_needle_results(
                        run_idx + 1, model_result.needles, model_result.pairs,
                        parsed=model_result.parsed,
                        verbosity=config.output.verbosity,
                        debug=model_result.debug,
                    )
                if is_debug:
                    _print_debug_context(model_result.debug)
            elif status == GlobalResultType.NO_RESULTS:
                print(f"model={model_result.model_name}, NO RESULTS")

        except HaystackQueryError as e:
            global_result.failed_runs.append(run_idx + 1)
            print()
            print("=" * 60)
            print(f"QUERY ERROR: Run {run_idx + 1} — {e}")
            print("=" * 60)
            print()
            if config.execution.stop_on_error:
                print("Stopping early due to --stop-on-error")
                break


def _print_summary(
    config: Config,
    result: GlobalResult,
    summaries: list[ExperimentSummary],
    start_time: datetime.datetime,
    model_name: str,
) -> None:
    """Print a comprehensive summary of the experiment."""
    is_minimal = config.output.verbosity == "minimal"

    print()
    print("=" * 60)
    print("SUMMARY")
    print("=" * 60)

    if is_minimal:
        print(f"  Model:           {model_name}")
        print(f"  Haystack size:   {config.haystack.haystack_num}")
        print(f"  Seeds:           {config.haystack.seed}")
        print(f"  Repeat:          {config.execution.repeat}")
    else:
        # Configuration
        print("\nConfiguration:")
        print(f"  Timestamp:       {config.output.timestamp}")
        if model_name == "unknown":
            print(f"  Model:           <no successful runs>")
        else:
            print(f"  Model:           {model_name}")
        print(f"  Haystack size:   {config.haystack.haystack_num}")
        print(f"  Num needles:     {config.haystack.needles_num + config.haystack.distractors_num} ({config.haystack.needles_num} real + {config.haystack.distractors_num} distractors)")
        distractor_pct = config.haystack.distractors_num / (config.haystack.needles_num + config.haystack.distractors_num)
        print(f"  Distractors:     {config.haystack.distractors_num} (exact count) ({distractor_pct * 100:.1f}%)")
        print(f"  Key length:      {config.haystack.key_len}")
        print(f"  Value range:     {config.haystack.val_min} - {config.haystack.val_max}")
        print(f"  Temperature:     {config.api.temperature}")
        print(f"  Max tokens:      {config.api.max_tokens}")
        print(f"  Timeout:         {config.api.timeout}s")
        print(f"  Seed:            {config.haystack.seed}")
        print(f"  Repeat:          {config.execution.repeat}")
        print(f"  Endpoint:        {config.api.endpoint}")
        if config.output.k_quant:
            print(f"  K quant:         {config.output.k_quant}")
        if config.output.v_quant:
            print(f"  V quant:         {config.output.v_quant}")
        if config.output.note:
            print(f"  Note:            {config.output.note}")

    # Per-run results
    if summaries:
        print("\nPer-run results:")
        run_rows_by_run: dict[int, list[ResultRow]] = {}
        for r in result.all_rows:
            run_rows_by_run.setdefault(r.run, []).append(r)
        if is_minimal:
            for i, s in enumerate(summaries):
                stats = result.all_stats[i] if i < len(result.all_stats) else {}
                run_rows_for_this = run_rows_by_run.get(i, [])
                needle_rows = [r for r in run_rows_for_this if not r.is_distractor]
                distractor_rows = [r for r in run_rows_for_this if r.is_distractor]
                needle_pct = f"{100*sum(1 for r in needle_rows if r.correct==1)/len(needle_rows):.1f}%" if needle_rows else "N/A"
                distractor_pct_run = f"{100*sum(1 for r in distractor_rows if r.correct==1)/len(distractor_rows):.1f}%" if distractor_rows else "N/A"
                print(f"  Run {i+1}: Needles {needle_pct} | Distractors {distractor_pct_run}")
        else:
            for i, s in enumerate(summaries):
                stats = result.all_stats[i] if i < len(result.all_stats) else {}
                run_rows_for_this = run_rows_by_run.get(i, [])
                _print_run_detail(i + 1, s, stats, run_rows_for_this, timestamp=None)

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
    if result.failed_runs or result.truncated_runs:
        print("\nIssues:")
        if result.failed_runs:
            print(f"  Failed runs:       {result.failed_runs}")
            if result.failed_runs and not summaries:
                print("  (no successful runs — all query attempts failed)")
        if result.truncated_runs:
            print(f"  Truncated runs:    {result.truncated_runs}")
            print("  (completion_tokens reached max_tokens — model was cut off)")

    print(f"\nOutput: {config.output.output_filename}")

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
