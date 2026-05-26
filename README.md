# Homogeneous Haystack Test

A needle-in-a-haystack benchmark for long-context retrieval evaluation, designed to measure pure positional retrieval capability free from semantic contrast bias.

## Motivation

This benchmark was built to evaluate Google's TurboQuant quantization on a local model (see [TurboQuant: Redefining AI efficiency with extreme compression](https://research.google/blog/turboquant-redefining-ai-efficiency-with-extreme-compression/)) compared to other quantization methods. Since it uses pure synthetic data — a list of random key-value pairs from which needles are randomly extracted — results are easy to verify and give a clear picture of a model's raw data retrieval capabilities, with no semantic or structural patterns involved.

Standard needle-in-a-haystack tests embed semantically distinctive facts — sentences like *"The special magic number in the city of Paris is 7342"* — into natural prose. The needle stands out sharply against its surroundings. This creates a **confound**: attention mechanisms may weight the needle more heavily simply because it *breaks the pattern*, not because the model has genuine long-range retrieval capability. The result conflates true positional retrieval with sensitivity to semantic contrast and anomaly detection. More than that, the entire surrounding passage lasting several tokens points to that association, creating a target much larger than just the value itself.

The present benchmark eliminates that confound by making *every* element in the context structurally and semantically **identical**. The haystack consists entirely of random `KEY = VALUE` pairs — no semantic outliers, no structural anomalies, no contrast signals. The only way for a model to correctly retrieve a queried value is to have genuinely attended to and retained that specific position in the context. To avoid letting the model optimize for keys at fixed positions, needle positions can also be randomized with jitter (see `--fuzz`).

- For a detailed explanation of the methodology, design rationale, and comparison with standard NIAH, see [homogeneous_haystack_test.md](homogeneous_haystack_test.md).
- For a review of the literature on how the NIAH problem has been tackled, see [NIAH Research](NIAH_Research.md).

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

This benchmark uses only the standard library for HTTP communication and has no external Python dependencies. It works exclusively with OpenAI-compatible chat completion APIs — any endpoint that accepts the `POST /v1/chat/completions` format.

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
| `--key-len`         | `8`                                         | Length of random keys, 5–12 (default: 8)                                 |
| `--val-min`         | `10000`                                     | Minimum value (default: 10000)                                           |
| `--val-max`         | `99999`                                     | Maximum value (default: 99999)                                           |
| `--haystack-num`    | `5000`                                      | Number of key-value pairs in the haystack                                |
| `--needles-num`     | `100`                                       | Number of needles to query (default: 100)                                |
| `--distractors-pct` | —                                           | Fraction of needles that are distractors (0.0–1.0; default is 20 when neither --distractors-num nor --distractors-pct is set) |
| `--distractors-num` | —                                           | Exact number of distractors (mutually exclusive with `--distractors-pct`) |
| `--temperature`     | `0.0`                                       | Sampling temperature (default: 0.0)                                      |
| `--max-tokens`      | `240000`                                    | Maximum tokens per response (default: 240000)                            |
| `--timeout`         | `7200`                                      | Request timeout in seconds (default: 7200)                               |
| `--verbosity` / `-v` | `medium`                                   | Output verbosity: `minimal`, `medium`, `full`, `debug` (default: medium) |
| `--repeat`          | `1`                                         | Number of independent runs (default: 1)                                  |
| `--stop-on-error`   | —                                           | Stop after the first query error                                         |
| `--seed`            | Random                                      | Base random seed for reproducibility                                     |
| `--fuzz`            | `0.49`                                      | Jitter needle positions as fraction of step size (default: 0.49)         |
| `--output`          | `results.csv`                               | Output CSV path (default: results.csv)                                   |
| `--k-quant`         | —                                           | KV-cache key quantization type (e.g. Q8_0, Q4_0, turbo4)                 |
| `--v-quant`         | —                                           | KV-cache value quantization type (e.g. Q8_0, Q4_0, turbo4)               |
| `--note`            | —                                           | Freeform note about this experiment                                      |

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

By default, needles are placed at evenly spaced positions with 49% random jitter (`--fuzz 0.49`). This breaks artificial periodicity while preserving even coverage across the context window. The 49% value (combined with integer truncation) guarantees no position collisions: two adjacent needles can never jitter to the same slot. Set `--fuzz 0` to use fixed equally-spaced positions, or tune the value to control jitter magnitude. Fuzz is automatically disabled when `--needles-num` exceeds twice `--haystack-num` (positions are already fully packed). Because jitter reduces repeatability, running multiple repetitions is recommended for reliable results.

## Output

Results are written to a CSV file (default: `results.csv`) with columns: `run`, `haystack_size`, `depth_pct`, `correct`, `expected`, `actual`, `prompt_tokens`, `completion_tokens`, `total_tokens`, `latency_ms`.

A summary is printed to stdout after each run, including per-needle accuracy, token usage, and throughput.

## Tests

```bash
uv run pytest
```

## AI Involvement

This project was developed with the assistance of Qwen 3.6 35b A3B (Q4_K_LM quantization from unsloth), running locally via llama.cpp with various KV cache quantization settings (Q8, turbo4, turbo3). The hardware is an NVIDIA 4090 laptop (16 GB VRAM) with an i9-13900HX CPU and 64 GB RAM, reaching around 53 tokens/s at context lengths exceeding 256K tokens — enough for a fluid vibe coding experience.

TurboQuant performed well: it compresses the KV cache using GPU capacity that would otherwise sit idle between layer transfers, delivering memory savings comparable to 4-bit quantization with retrieval quality that matched or exceeded Q8 in benchmark results.

The model generated a working ~1300-line Python implementation from the outset, though the initial code required significant refactoring to reach a clean architecture — improving abstraction layering, argument validation, type usage, and overall readability. Iterative prompting with explicit structural guidance would likely have improved the first-pass output.
