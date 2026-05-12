#!/usr/bin/env python3
"""
Homogeneous Needle-in-a-Haystack benchmark for llama-server (OpenAI-compatible API).

Generates random key=value pairs, embeds them as haystack, queries the model for
values at various depths, and writes results to CSV.

Usage:
  python haystack_test.py --haystack-n 5000 --num-needles 100 --repeat 3
"""

import argparse
import csv
import json
import random
import socket
import string
import sys
import time
import urllib.error
import urllib.request


# ---------------------------------------------------------------------------
# Data generation
# ---------------------------------------------------------------------------

def generate_key(length=8):
    """Generate a random uppercase string of fixed length."""
    return "".join(random.choices(string.ascii_uppercase, k=length))


def generate_value(val_min=1000, val_max=9999):
    """Generate a random integer, avoiding obviously round numbers."""
    while True:
        v = random.randint(val_min, val_max)
        # Avoid multiples of 100 — they feel "round" and could bias the model
        if v % 100 != 0:
            return v


def build_haystack(n, key_len=8, val_min=1000, val_max=9999):
    """Build a list of `n` random KEY = VALUE lines.

    Returns (text, list_of (key, value) tuples).
    """
    pairs = []
    for _ in range(n):
        key = generate_key(key_len)
        value = generate_value(val_min, val_max)
        pairs.append((key, value))
    text = "\n".join(f"{k} = {v}" for k, v in pairs)
    return text, pairs


# ---------------------------------------------------------------------------
# Needle selection
# ---------------------------------------------------------------------------

def pick_needle_positions(n_total, n_needles):
    """Pick `n_needles` indices uniformly spaced across [0, n_total)."""
    if n_needles >= n_total:
        return list(range(n_total))
    step = n_total / n_needles
    return [int(i * step) for i in range(n_needles)]


def select_needles(pairs, n_needles, distractor_pct=0.08, haystack_keys=None):
    """Select needles from the haystack, mixing real and distractor keys.

    Returns:
        needles: list of dicts with keys:
            - index: position in haystack
            - key: the key string
            - expected: the expected value (None for distractors)
    """
    positions = pick_needle_positions(len(pairs), n_needles)
    n_real = int(n_needles * (1 - distractor_pct))
    n_distractor = n_needles - n_real

    # Real needles — pick from actual haystack positions
    real_needles = []
    for idx in positions[:n_real]:
        key, value = pairs[idx]
        real_needles.append({
            "index": idx,
            "key": key,
            "expected": value,
            "is_distractor": False,
        })

    # Distractor needles — keys NOT in the haystack
    if haystack_keys is None:
        haystack_keys = set(k for k, _ in pairs)

    distractor_keys = set()
    tries = 0
    while len(distractor_keys) < n_distractor and tries < n_distractor * 20:
        k = generate_key(len(real_needles[0]["key"]) if real_needles else 8)
        if k not in haystack_keys:
            distractor_keys.add(k)
        tries += 1

    distractor_needles = []
    for idx, key in zip(positions[n_real:], distractor_keys):
        distractor_needles.append({
            "index": idx,
            "key": key,
            "expected": None,
            "is_distractor": True,
        })

    # Interleave real and distractor by their original position
    all_needles = real_needles + distractor_needles
    all_needles.sort(key=lambda n: n["index"])
    return all_needles


# ---------------------------------------------------------------------------
# Prompt building
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = (
    "You will be given a list of key-value pairs and asked to retrieve "
    "the value for each key. Read the pairs carefully."
)


def build_prompt(haystack_text, needles):
    """Build the full prompt message.

    Format: list of pairs, then one question per needle, answer format KEY=VALUE.
    """
    lines = [
        SYSTEM_PROMPT,
        "",
        haystack_text,
        "",
        "Now answer for each key. Use the format KEY=VALUE on each line:",
    ]
    for needle in needles:
        lines.append(f"What is the value for {needle['key']}?")
    return [{"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": "\n".join(lines)}]


# ---------------------------------------------------------------------------
# API query
# ---------------------------------------------------------------------------

def query_llama(endpoint, messages, model=None, temperature=0.0,
                max_tokens=10, timeout=300):
    """Send a request to the llama-server OpenAI-compatible endpoint.

    Returns (response_json, latency_ms).
    Raises on HTTP errors or timeouts (errors are kept as data, not retried).
    """
    payload = {
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


def extract_model_name(response):
    """Extract model name from OpenAI-compatible API response."""
    return response.get("model", "unknown")


# ---------------------------------------------------------------------------
# Response parsing
# ---------------------------------------------------------------------------

def parse_response(response_text, needle_keys):
    """Parse the model's response into a dict mapping key -> value string.

    Skips lines that don't match any queried key.
    Non-numeric values are ignored.
    Missing keys are not included.
    """
    if not response_text:
        return {}

    found = {}
    for line in response_text.strip().split("\n"):
        line = line.strip()
        if "=" not in line:
            continue
        parts = line.split("=", 1)
        key = parts[0].strip()
        val = parts[1].strip()
        if key in needle_keys and val:
            # Only keep if value looks numeric
            try:
                int(val)
                found[key] = val
            except ValueError:
                pass
    return found


def score_needles(needles, parsed):
    """Score each needle against parsed responses.

    Returns list of dicts with:
        depth_pct, needle_key, expected, actual, correct
    """
    results = []
    for needle in needles:
        key = needle["key"]
        expected = needle["expected"]
        actual = parsed.get(key, "")
        if expected is not None:
            # Real needle
            correct = 1 if actual == str(expected) else 0
        else:
            # Distractor — should be empty or wrong
            correct = 0
        results.append({
            "needle_key": key,
            "expected": expected if expected is not None else "",
            "actual": actual,
            "correct": correct,
        })
    return results


# ---------------------------------------------------------------------------
# Experiment runner
# ---------------------------------------------------------------------------

def _truncate(text, max_lines=3, prefix="..."):
    """Show first and last N lines of a long text."""
    lines = text.split("\n")
    if len(lines) <= max_lines * 2:
        return text
    first = "\n".join(lines[:max_lines])
    last = "\n".join(lines[-max_lines:])
    return f"{first}\n{prefix}\n{last}"


def build_single_needle_prompt(haystack_text, needle_key):
    """Build a prompt for a single needle query."""
    return [
        {"role": "system", "content": "You will be given a list of key=value pairs. Answer with JUST the number, nothing else."},
        {"role": "user", "content": f"{haystack_text}\n\n{needle_key}="},
    ]


def run_single_experiment(run_index, config, seed, show=None):
    """Run one complete experiment: generate, query, score, return rows."""
    random.seed(seed)

    # Generate haystack
    haystack_text, pairs = build_haystack(
        config["haystack_n"],
        config["key_len"],
        config["val_min"],
        config["val_max"],
    )
    haystack_keys = set(k for k, _ in pairs)

    # Select needles
    needles = select_needles(
        pairs,
        config["num_needles"],
        config["distractor_pct"],
        haystack_keys,
    )

  # Query
    is_single = config.get("single", False)
    latency_ms = 0
    raw_response = None
    model_name = "unknown"

    def extract_content(choice, needle_key=None):
        """Extract response text, checking content and reasoning_content."""
        content = choice.get("message", {}).get("content", "")
        if not content:
            # Some models put output in reasoning_content
            reasoning = choice.get("message", {}).get("reasoning_content", "")
            if reasoning:
                # If we have a needle_key, search reasoning for KEY = VALUE
                if needle_key:
                    for line in reasoning.split("\n"):
                        line = line.strip()
                        if line.startswith(needle_key + " = "):
                            val = line.split("=", 1)[1].strip()
                            try:
                                int(val)
                                return val
                            except ValueError:
                                pass
                # Try to extract answer from reasoning (look for KEY = VALUE)
                lines = reasoning.strip().split("\n")
                for line in lines:
                    line = line.strip()
                    if "=" in line:
                        parts = line.split("=", 1)
                        val = parts[1].strip()
                        try:
                            int(val)
                            return val
                        except ValueError:
                            pass
                # If no KEY=VALUE found, try last numeric token
                tokens = reasoning.split()
                for token in reversed(tokens):
                    token = token.strip(".,;:!?\"'`)")
                    try:
                        int(token)
                        return token
                    except ValueError:
                        pass
        return content

    if is_single:
        # Query one needle at a time
        all_results = []
        for needle in needles:
            messages = build_single_needle_prompt(haystack_text, needle["key"])
            try:
                response, latency_ms = query_llama(
                    config["endpoint"],
                    messages,
                    model=config.get("model"),
                    temperature=config["temperature"],
                    max_tokens=config["max_tokens"],
                    timeout=config.get("timeout", 300),
                )
                raw_response = response
                if model_name == "unknown":
                    model_name = extract_model_name(response)
                content = extract_content(response["choices"][0], needle["key"])
                # Parse single value
                val = ""
                try:
                    val = content.strip()
                    int(val)
                except ValueError:
                    val = ""
                all_results.append({
                    "needle": needle,
                    "actual": val,
                    "content": content,
                })
            except HaystackQueryError as e:
                all_results.append({
                    "needle": needle,
                    "actual": "",
                    "content": str(e),
                })
        parsed = {r["needle"]["key"]: r["actual"] for r in all_results}
        response_text = "\n".join(r["content"] for r in all_results)
    else:
        # Build prompt
        messages = build_prompt(haystack_text, needles)
        needle_keys = {n["key"] for n in needles}
        try:
            response, latency_ms = query_llama(
                config["endpoint"],
                messages,
                model=config.get("model"),
                temperature=config["temperature"],
                max_tokens=config["max_tokens"],
                timeout=config.get("timeout", 300),
            )
            raw_response = response
            model_name = extract_model_name(response)
            response_text = extract_content(response["choices"][0])
        except HaystackQueryError as e:
            model_name = "error"
            response_text = str(e)

        # Parse and score
        parsed = parse_response(response_text, needle_keys)

    # Collect rows
    rows = []
    for needle, score in zip(needles, score_needles(needles, parsed)):
        depth_pct = round(needle["index"] / (len(pairs) - 1) * 100, 2) if len(pairs) > 1 else 0.0
        rows.append({
            "run": run_index,
            "haystack_size": config["haystack_n"],
            "depth_pct": depth_pct,
            "correct": score["correct"],
            "expected": score["expected"],
            "actual": score["actual"],
        })

     # Debug output
    if show:
        print()
        print("=" * 60)
        print(f"Run {run_index} — model={model_name}")
        print("=" * 60)
        if show in ("prompt", "all"):
            prompt_text = messages[1]["content"]
            print(f"\n--- PROMPT ({len(prompt_text)} chars, {len(prompt_text.split())} tokens est.) ---")
            print(_truncate(prompt_text, max_lines=5))
        if show in ("response", "all"):
            print(f"\n--- RAW RESPONSE (JSON) ---")
            if raw_response is not None:
                print(json.dumps(raw_response, indent=2, default=str))
            else:
                print("(no response received)")
            print(f"\n--- CONTENT FIELD ---")
            print(repr(response_text))
            print(f"\n--- PARSED RESULTS ---")
            for needle, score in zip(needles, score_needles(needles, parsed)):
                depth = round(needle["index"] / (len(pairs) - 1) * 100, 1) if len(pairs) > 1 else 0.0
                status = "OK" if score["correct"] else "FAIL"
                print(f"  [{status}] {needle['key']} (depth {depth}%) "
                      f"expected={score['expected']!r} actual={score['actual']!r}")
        print("=" * 60)
        print()

    return rows, model_name


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

CSV_COLUMNS = ["run", "haystack_size", "depth_pct", "correct", "expected", "actual"]


def main():
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
    parser.add_argument("--max-tokens", type=int, default=100,
                        help="Max tokens per response (default: 100)")
    parser.add_argument("--timeout", type=int, default=300,
                        help="Request timeout in seconds (default: 300)")
    parser.add_argument("--show", choices=["prompt", "response", "all"],
                        help="Print prompt/response for debugging (prompt=response/all)")
    parser.add_argument("--single", action="store_true",
                        help="Query one needle at a time (for debugging)")
    parser.add_argument("--repeat", type=int, default=1,
                        help="Number of independent runs (default: 1)")
    parser.add_argument("--seed", type=int, default=None,
                        help="Base random seed (random if omitted)")
    parser.add_argument("--output", default="results.csv",
                        help="Output CSV path (default: results.csv)")

    args = parser.parse_args()
    config = {
        "endpoint": args.endpoint,
        "model": args.model,
        "key_len": args.key_len,
        "val_min": args.val_min,
        "val_max": args.val_max,
        "haystack_n": args.haystack_n,
        "num_needles": args.num_needles,
        "distractor_pct": args.distractor_pct,
        "temperature": args.temperature,
        "max_tokens": args.max_tokens,
        "timeout": args.timeout,
        "single": args.single,
    }

    seed = args.seed if args.seed is not None else random.randint(0, 2**31)
    print(f"Seed: {seed}")
    print(f"Endpoint: {args.endpoint}")
    print(f"Haystack: {args.haystack_n} pairs, {args.num_needles} needles "
          f"({args.distractor_pct*100:.0f}% distractors)")
    print(f"Output: {args.output}")
    print()

    all_rows = []
    for run_idx in range(args.repeat):
        print(f"Run {run_idx + 1}/{args.repeat}...", end=" ", flush=True)
        try:
            run_seed = seed + run_idx
            rows, model_name = run_single_experiment(run_idx, config, run_seed, show=args.show)
            all_rows.extend(rows)
            correct_count = sum(1 for r in rows if r["correct"] == 1)
            print(f"model={model_name}, "
                  f"accuracy={correct_count}/{len(rows)} "
                  f"({100*correct_count/len(rows):.1f}%)")
        except Exception as e:
            print(f"FAILED: {e}", file=sys.stderr)

    # Write CSV
    with open(args.output, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        writer.writerows(all_rows)

    print(f"\nWrote {len(all_rows)} rows to {args.output}")

    # Summary
    total = len(all_rows)
    correct = sum(1 for r in all_rows if r["correct"] == 1)
    if total > 0:
        print(f"Overall accuracy: {correct}/{total} ({100*correct/total:.1f}%)")
    else:
        print("No results collected.")


if __name__ == "__main__":
    main()
