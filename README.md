# Homogeneous Haystack Test

A needle-in-a-haystack benchmark for long-context retrieval evaluation, designed to measure pure positional retrieval capability free from semantic contrast bias.

## Motivation

Standard needle-in-a-haystack tests embed semantically distinctive facts — sentences like *"The special magic number in the city of Paris is 7342"* — into natural prose. The needle stands out sharply against its surroundings. This creates a confound: attention mechanisms may weight the needle more heavily simply because it breaks the pattern, not because the model has genuine long-range retrieval capability. The result conflates true positional retrieval with sensitivity to semantic contrast and anomaly detection. More than that the entire paragraph lasting several tokens points to that magic association creating a target much larger that just the number itself.

This benchmark eliminates that confound by making every element in the context structurally and semantically identical. The haystack consists entirely of random `KEY = VALUE` pairs. There are no semantic outliers, no structural anomalies, no contrast signals. The only way for a model to correctly retrieve a queried value is to have genuinely attended to and retained that specific position in the context. To avoid letting the model optimize for keys at fixed positions it is even possible to shake the needle positions randomly (see `--fuzz`).

For a detailed explanation of the methodology, design rationale, and comparison with standard NIAH, see [homogeneous_haystack_test.md](homogeneous_haystack_test.md).

## How It Works

1. **Generate** a list of `N` random `KEY = VALUE` pairs (keys are random uppercase strings, values are random integers).
2. **Select** `M` pairs as query targets at evenly distributed positions, with optional jitter (see `--fuzz`).
3. **Optionally insert** distractor needles — keys that do not appear in the haystack — to test for hallucination.
4. **Query** the model with the full haystack and all queries in a single prompt.
5. **Score** by exact match: correct if the returned value matches the expected value, incorrect otherwise. Distractors are correct only if the model returns nothing for them.

The benchmark runs across multiple independent repetitions and writes all results to a CSV file with per-needle accuracy, token usage, and latency.

## Requirements

- **Python 3.13+**
- **An OpenAI-compatible API endpoint** (e.g. `llama-server` on `http://localhost:8080/v1/chat/completions`)

This benchmark uses only the standard library for HTTP communication and has no external Python dependencies beyond optional test execution. It works exclusively with OpenAI-compatible chat completion APIs — any endpoint that accepts the `POST /v1/chat/completions` format.

## Installation

No installation required. The project uses a `pyproject.toml` with `uv` for environment management, but the benchmark itself runs with just Python and the standard library.

```bash
# Optional: set up the dev environment
uv sync
```

## Usage

```bash
python haystack_test.py --haystack-num 5000 --needles-num 100 --repeat 3
```

### Command-line options

| Argument            | Default                                     | Description                                                              |
| ------------------- | ------------------------------------------- | ------------------------------------------------------------------------ |
| `--endpoint`        | `http://localhost:8080/v1/chat/completions` | OpenAI-compatible API endpoint                                           |
| `--model`           | Auto-detected                               | Model name (omit to detect from API)                                     |
| `--key-len`         | `8`                                         | Length of random keys (5–12)                                             |
| `--val-min`         | `10000`                                     | Minimum value range                                                      |
| `--val-max`         | `99999`                                     | Maximum value range                                                      |
| `--haystack-num`    | `5000`                                      | Number of key-value pairs in the haystack                                |
| `--needles-num`     | `100`                                       | Number of query targets                                                  |
| `--distractor-pct`  | `0.08`                                      | Fraction of needles that are distractors                                 |
| `--distractors-num` | —                                           | Exact number of distractors (mutually exclusive with `--distractor-pct`) |
| `--temperature`     | `0.0`                                       | Sampling temperature (use 0 for deterministic output)                    |
| `--max-tokens`      | `240000`                                    | Maximum tokens per response                                              |
| `--timeout`         | `7200`                                      | Request timeout in seconds                                               |
| `--repeat`          | `1`                                         | Number of independent runs                                               |
| `--seed`            | Random                                      | Base random seed for reproducibility                                     |
| `--fuzz`            | `0.49`                                      | Jitter needle positions as fraction of step size (0 to disable)          |
| `--output`          | `results.csv`                               | Output CSV path                                                          |
| `--show`            | —                                           | Print prompt/response for debugging (`prompt`, `response`, or `all`)     |
| `--full`            | —                                           | Print full untruncated prompt and response                               |

### Example: large-context evaluation

```bash
python haystack_test.py \
    --endpoint http://localhost:8080/v1/chat/completions \
    --haystack-num 20000 \
    --needles-num 200 \
    --repeat 5 \
    --output results_large.csv
```

### Needle position jitter (`--fuzz`)

By default, needles are placed at evenly spaced positions with 49% random jitter (`--fuzz 0.49`). This breaks the artificial periodicity while preserving even coverage across the context window. The 49% value (combined with integer truncation) guarantees no position collisions: two adjacent needles can never jitter to the same slot. Set `--fuzz 0` to use fixed equally-spaced positions, or tune the value to control the jitter magnitude. Fuzz is automatically disabled when `--needles-num` exceeds twice `--haystack-num` (positions are already fully packed). Using this technique, while making the test even more resistant to spurious optimizations, can have a minor effect on repeatability so it's better to repeat the test several times for true consistent results.

## Output

Results are written to a CSV file (default: `results.csv`) with columns: `run`, `haystack_size`, `depth_pct`, `correct`, `expected`, `actual`, `prompt_tokens`, `completion_tokens`, `total_tokens`, `latency_ms`.

A summary is printed to stdout after each run, including per-needle accuracy, token usage, and throughput.

## Tests

```bash
uv run pytest
```
