#!/usr/bin/env python3
"""
Homogeneous Needle-in-a-Haystack benchmark for llama-server (OpenAI-compatible API).

Generates random key=value pairs, embeds them as haystack, queries the model for
values at various depths, and writes results to CSV.

Usage:
  python haystack_test.py --haystack-n 5000 --num-needles 100 --repeat 3
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import secrets
import socket
import string
import sys
import time
import urllib.error
import urllib.request
from dataclasses import asdict, dataclass
from typing import TypedDict

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


class ApiSuccessResponse(TypedDict):
    model: str
    choices: list[ChatCompletionChoice]


ApiResponseBody = ApiSuccessResponse | ApiErrorResponse


class Payload(TypedDict):
    model: str
    messages: list[Message]
    temperature: float
    max_tokens: int
    stream: bool


@dataclass(frozen=True)
class Interaction:
    messages: list[Message]
    response: ApiResponseBody | None
    content: str
    needle: Needle


@dataclass(frozen=True)
class ResultRow:
    run: int
    haystack_size: int
    depth_pct: float
    correct: int
    expected: int | str
    actual: str


@dataclass
class Needle:
    index: int
    key: str
    expected: int | None
    is_distractor: bool


@dataclass
class Config:
    endpoint: str
    model: str | None
    key_len: int
    val_min: int
    val_max: int
    haystack_n: int
    num_needles: int
    distractor_pct: float
    temperature: float
    max_tokens: int
    timeout: int
    single: bool
    full: bool


@dataclass
class ScoredNeedle:
    needle_key: str
    expected: int | str
    actual: str
    correct: int


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
    n: int, key_len: int = 8, val_min: int = 1000, val_max: int = 9999
) -> tuple[str, list[HaystackPair]]:
    """Build a list of `n` random KEY = VALUE lines.

    Args:
    - n: number of pairs to generate
    - key_len: length of the random keys (default: 8)
    - val_min: minimum value (default: 1000)
    - val_max: maximum value (default: 9999)

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

def pick_needle_positions(n_total: int, n_needles: int) -> list[int]:
    """Pick `n_needles` indices uniformly spaced across [0, n_total)."""
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
    # a set that contains the all the keys of the haystack
    haystack_keys = set(k for k, _ in pairs)

    # contains the set of disctractors keys: keys not in the haystack
    distractor_keys: set[str] = set()
    tries = 0
    while len(distractor_keys) < n_distractor and tries < n_distractor * 20:
        k = generate_key(key_length)
        if k not in haystack_keys:
            distractor_keys.add(k)
        tries += 1

    distractor_needles: list[Needle] = []

    # crete the distractors using the remaining indexes of the positions
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
    distractor_pct: float = 0.08
) -> list[Needle]:
    """Select real needles from haystack at uniformly spaced intervals.

    Args:
        pairs: List of all haystack key-value pairs.
        n_needles: Total number of needles (real + distractors) to select.
        distractor_pct: Fraction of total that will be distractors.

    Returns:
        List of Needle objects extracted from the haystack positions.
    """
    # list n_needles indexes to the pairs list taken at fixed intervals
    positions: list[int] = pick_needle_positions(len(pairs), n_needles)
    # the number of real needles (without distractors) to include based on the specified percentage
    n_real = int(n_needles * (1 - distractor_pct))
    # the number of distractor needles to generate
    n_distractor = n_needles - n_real

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


def build_single_needle_prompt(haystack_text: str, needle_key: str) -> list[Message]:
    """Build a prompt querying for a single needle's value.

    Args:
        haystack_text: The full haystack text containing key=value pairs.
        needle_key: The key to look up in the haystack.

    Returns:
        System and user messages for the single query.
    """
    return [
        {
            "role": "system",
            "content": "These are random key=value pairs. Find the value for the given key by looking it up in the list. Answer with JUST the number, nothing else.",
        },
        {"role": "user", "content": f"{haystack_text}\n\n{needle_key}="},
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


def extract_model_name(response: ApiResponseBody) -> str:
    """Extract model name from OpenAI-compatible API response.

    Args:
        response: The API response body.

    Returns:
        The model name, or "unknown" if not present.
    """
    return response.get("model", "unknown")


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


def _query_single_needles(
    config: Config,
    haystack_text: str,
    needles: list[Needle],
) -> tuple[dict[str, str], list[Interaction], ApiResponseBody | None, str]:
    """Query the model one needle at a time.

    Args:
        config: Benchmark configuration.
        haystack_text: The full haystack text.
        needles: List of needles to query.

    Returns:
        Tuple of (parsed results, interaction history, last raw response,
        concatenated response text).
    """
    interactions: list[Interaction] = []
    raw_response: ApiResponseBody | None = None

    for needle in needles:
        messages = build_single_needle_prompt(haystack_text, needle.key)
        try:
            response, _ = query_llama(
                config.endpoint,
                messages,
                model=config.model,
                temperature=config.temperature,
                max_tokens=config.max_tokens,
                timeout=config.timeout,
            )
            raw_response = response
            content = response["choices"][0]["message"].get("content", "")  # type: ignore[typeddict-item]
        except HaystackQueryError as e:
            response = None  # type: ignore[assignment]
            content = str(e)

        interactions.append(Interaction(
            messages=messages,
            response=response,
            content=content,
            needle=needle,
        ))

    parsed = {n.key: inter.content for n, inter in zip(needles, interactions)}
    response_text = "\n".join(inter.content for inter in interactions)
    return parsed, interactions, raw_response, response_text


def _query_batch(
    config: Config,
    haystack_text: str,
    needles: list[Needle],
) -> tuple[dict[str, str], list[Message], str, ApiResponseBody | None, str]:
    """Query the model with all needles in a single prompt.

    Args:
        config: Benchmark configuration.
        haystack_text: The full haystack text.
        needles: List of needles to query.

    Returns:
        Tuple of (parsed results, prompt messages, response text,
        raw response body, model name).
    """
    messages = build_prompt(haystack_text, needles)
    needle_keys: set[str] = {n.key for n in needles}

    try:
        response, _ = query_llama(
            config.endpoint,
            messages,
            model=config.model,
            temperature=config.temperature,
            max_tokens=config.max_tokens,
            timeout=config.timeout,
        )
        raw_response = response
        model_name = extract_model_name(response)
        response_text = response["choices"][0]["message"].get("content", "")  # type: ignore[typeddict-item]
    except HaystackQueryError as e:
        raw_response = None
        model_name = "error"
        response_text = str(e)

    parsed = parse_response(response_text, needle_keys)
    return parsed, messages, response_text, raw_response, model_name


def _build_rows(
    run_index: int,
    needles: list[Needle],
    pairs: list[HaystackPair],
    parsed: dict[str, str],
) -> list[ResultRow]:
    """Build result rows from scored needles.

    Args:
        run_index: The experiment run number.
        needles: List of needles that were queried.
        pairs: Full list of haystack pairs (for depth calculation).
        parsed: Parsed key -> value results from the model.

    Returns:
        List of ResultRow objects with correctness and depth information.
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
        ))
    return rows


def _print_results(
    run_index: int,
    model_name: str,
    needles: list[Needle],
    pairs: list[HaystackPair],
    parsed: dict[str, str],
    *,
    show: str | None = None,
    is_full: bool,
    is_single: bool = False,
    interactions: list[Interaction] | None = None,
    messages: list[Message] | None = None,
    raw_response: ApiResponseBody | None = None,
    response_text: str = "",
) -> None:
    """Print experiment results and optional debug output.

    Args:
        run_index: The experiment run number.
        model_name: Name of the model that was queried.
        needles: List of needles that were queried.
        pairs: Full list of haystack pairs.
        parsed: Parsed key -> value results from the model.
        show: Which debug output to show ("prompt", "response", "all", or None).
        is_full: Whether to print full untruncated content.
        is_single: Whether single-needle mode was used.
        interactions: Interaction history for single-mode (optional).
        messages: Prompt messages for batch mode (optional).
        raw_response: Raw API response body (optional).
        response_text: Extracted response content text.
    """
    print()
    print("=" * 60)
    print(f"Run {run_index} — model={model_name}")
    print("=" * 60)

    # --- Single-mode interactions ---
    if is_single and interactions is not None and (show in ("prompt", "all") or is_full):
        for idx, inter in enumerate(interactions):
            print(f"\n--- INTERACTION {idx + 1}/{len(interactions)} ---")
            for i, msg in enumerate(inter.messages):
                _print_message(msg, is_full)
            if show in ("response", "all") or is_full:
                print(f"\n--- RESPONSE (needle={inter.needle.key}) ---")
                if inter.response is not None:
                    print(json.dumps(inter.response, indent=2, default=str))
                else:
                    print("(no response)")
                print(f"--- Parsed answer ---")
                print(repr(inter.content))

    # --- Batch-mode messages ---
    elif messages is not None and (show in ("prompt", "all") or is_full):
        print(f"\n--- ALL MESSAGES ({len(messages)} messages) ---")
        for msg in messages:
            _print_message(msg, is_full)

    # --- Batch-mode response ---
    if not is_single and (show in ("response", "all") or is_full):
        print(f"\n--- RAW RESPONSE (JSON) ---")
        if raw_response is not None:
            print(json.dumps(raw_response, indent=2, default=str))
        else:
            print("(no response received)")
        print(f"\n--- CONTENT FIELD ---")
        print(repr(response_text))

    # --- Parsed results summary ---
    print(f"\n--- PARSED RESULTS ---")
    for needle, score in zip(needles, score_needles(needles, parsed)):
        depth = round(
            needle.index / (len(pairs) - 1) * 100, 1
        ) if len(pairs) > 1 else 0.0
        status = "OK" if score.correct else "FAIL"
        print(f"  [{status}] {needle.key} (depth {depth}%) "
              f"expected={score.expected!r} actual={score.actual!r}")
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
    run_index: int, config: Config, seed: int, show: str | None = None
) -> tuple[list[ResultRow], str]:
    """Run one complete experiment: generate, query, score, return rows.

    Args:
        run_index: The experiment run number (for CSV output).
        config: Benchmark configuration.
        seed: Random seed for reproducibility.
        show: Which debug output to show ("prompt", "response", "all", or None).

    Returns:
        Tuple of (list of result rows, model name string).
    """
    random.seed(seed)
    is_full = config.full
    is_single = config.single

    haystack_text, pairs, needles = generate_haystack_and_needles(
        haystack_n=config.haystack_n,
        num_needles=config.num_needles,
        distractor_pct=config.distractor_pct,
        key_len=config.key_len,
        val_min=config.val_min,
        val_max=config.val_max,
    )

    interactions: list[Interaction] | None = None
    messages: list[Message] | None = None

    if is_single:
        parsed, interactions, raw_response, response_text = _query_single_needles(
            config, haystack_text, needles
        )
        model_name = extract_model_name(raw_response) if raw_response else "error"
    else:
        parsed, messages, response_text, raw_response, model_name = _query_batch(
            config, haystack_text, needles
        )

    rows = _build_rows(run_index, needles, pairs, parsed)

    if show or is_full:
        _print_results(
            run_index, model_name, needles, pairs, parsed,
            show=show, is_full=is_full, is_single=is_single,
            interactions=interactions,
            messages=messages,
            raw_response=raw_response,
            response_text=response_text,
        )

    return rows, model_name


def validate_generation_params(config: Config) -> None:
    """Validate parameters for haystack and needle generation.

    Args:
        config: Benchmark configuration to validate.

    Raises:
        ValueError: If any parameter is invalid (haystack_n < 1,
            num_needles < 1, distractor_pct out of range, key_len < 1,
            or val_min > val_max).
    """
    if config.haystack_n < 1:
        raise ValueError(f"haystack_n must be >= 1, got {config.haystack_n}")
    if config.num_needles < 1:
        raise ValueError(f"num_needles must be >= 1, got {config.num_needles}")
    if not (0.0 <= config.distractor_pct < 1.0):
        raise ValueError(
            f"distractor_pct must be in [0.0, 1.0), got {config.distractor_pct}"
        )
    if config.key_len < 1:
        raise ValueError(f"key_len must be >= 1, got {config.key_len}")
    if config.val_min > config.val_max:
        raise ValueError(f"val_min ({config.val_min}) > val_max ({config.val_max})")


def generate_haystack_and_needles(
    haystack_n: int,
    num_needles: int,
    distractor_pct: float,
    key_len: int,
    val_min: int,
    val_max: int,
) -> tuple[str, list[HaystackPair], list[Needle]]:
    """Generate haystack text, pair list, and shuffled needles.

    Args:
        haystack_n: Total number of key-value pairs in the haystack.
        num_needles: Total number of needles (real + distractors).
        distractor_pct: Fraction of needles that are distractors (0.0-1.0).
        key_len: Length of random keys in characters.
        val_min: Minimum value for generated values.
        val_max: Maximum value for generated values.

    Returns:
        A tuple of (haystack_text, pairs, shuffled_needles).

    """
    # Build haystack
    haystack_text, pairs = build_haystack(
        haystack_n, key_len, val_min, val_max
    )

    # Select real needles from haystack at fixed intervals
    real_needles = select_needles(pairs, num_needles, distractor_pct)

    # Compute number of distractors needed
    n_distractor = num_needles - len(real_needles)

    # Create distractor needles (keys not in haystack)
    distractor_needles = create_distractor_keys(
        pairs,
        n_distractor,
        len(real_needles),
        key_len,
    )

    # Combine real and distractor needles, then shuffle
    needles = real_needles + distractor_needles
    random.shuffle(needles)

    return haystack_text, pairs, needles


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

CSV_COLUMNS: list[str] = [
    "run", "haystack_size", "depth_pct", "correct", "expected", "actual",
]


def main() -> None:
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
    parser.add_argument("--val-min", type=int, default=1000,
                        help="Minimum value (default: 1000)")
    parser.add_argument("--val-max", type=int, default=9999,
                        help="Maximum value (default: 9999)")
    parser.add_argument("--haystack-n", type=int, default=5000,
                        help="Total number of pairs in haystack (default: 5000)")
    parser.add_argument("--num-needles", type=int, default=100,
                        help="Number of needles to query (default: 100)")
    parser.add_argument("--distractor-pct", type=float, default=0.08,
                        help="Fraction of needles that are distractors (default: 0.08)")
    parser.add_argument("--temperature", type=float, default=0.0,
                        help="Sampling temperature (default: 0.0)")
    parser.add_argument("--max-tokens", type=int, default=8192,
                        help="Max tokens per response (default: 8192)")
    parser.add_argument("--timeout", type=int, default=300,
                        help="Request timeout in seconds (default: 300)")
    parser.add_argument("--show", choices=["prompt", "response", "all"],
                        help="Print prompt/response for debugging (prompt=response/all)")
    parser.add_argument("--single", action="store_true",
                        help="Query one needle at a time (for debugging)")
    parser.add_argument("--full", action="store_true",
                        help="Print full untruncated prompt and response")
    parser.add_argument("--repeat", type=int, default=1,
                        help="Number of independent runs (default: 1)")
    parser.add_argument("--seed", type=int, default=None,
                        help="Base random seed (random if omitted)")
    parser.add_argument("--output", default="results.csv",
                        help="Output CSV path (default: results.csv)")

    args = parser.parse_args()

    config = Config(
        endpoint=args.endpoint,
        model=args.model,
        key_len=args.key_len,
        val_min=args.val_min,
        val_max=args.val_max,
        haystack_n=args.haystack_n,
        num_needles=args.num_needles,
        distractor_pct=args.distractor_pct,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        timeout=args.timeout,
        single=args.single,
        full=args.full,
    )

    validate_generation_params(config)

    seed = args.seed if args.seed is not None else secrets.randbits(31)
    print(f"Seed: {seed}")
    print(f"Endpoint: {args.endpoint}")
    print(f"Haystack: {args.haystack_n} pairs, {args.num_needles} needles "
          f"({args.distractor_pct*100:.0f}% distractors)")
    print(f"Output: {args.output}")
    print()

    all_rows: list[ResultRow] = []
    for run_idx in range(args.repeat):
        print(f"Run {run_idx + 1}/{args.repeat}...", end=" ", flush=True)
        try:
            run_seed = seed + run_idx
            rows, model_name = run_single_experiment(run_idx, config, run_seed, show=args.show)
            all_rows.extend(rows)
            correct_count = sum(1 for r in rows if r.correct == 1)
            print(f"model={model_name}, "
                  f"accuracy={correct_count}/{len(rows)} "
                  f"({100*correct_count/len(rows):.1f}%)")
        except Exception as e:
            print(f"FAILED: {e}", file=sys.stderr)

    with open(args.output, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        writer.writerows(asdict(r) for r in all_rows)

    print(f"\nWrote {len(all_rows)} rows to {args.output}")

    total = len(all_rows)
    correct = sum(1 for r in all_rows if r.correct == 1)
    if total > 0:
        print(f"Overall accuracy: {correct}/{total} ({100*correct/total:.1f}%)")
    else:
        print("No results collected.")


if __name__ == "__main__":
    main()
