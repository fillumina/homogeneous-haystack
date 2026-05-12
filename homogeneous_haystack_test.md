# Homogeneous Needle-in-a-Haystack Test
## A More Rigorous Approach to Long-Context Retrieval Evaluation

---

## Overview

This document describes a variant of the standard Needle-in-a-Haystack (NIAH) benchmark designed to measure pure positional retrieval in large language models, free from the semantic contrast bias that affects conventional implementations. It is intended as both a design rationale and a specification for implementation.

---

## The Standard Test and Its Limitation

The canonical NIAH test, popularized by gkamradt's widely used implementation, works as follows: a semantically distinctive fact — the "needle" — is embedded at a specific position within a long passage of natural prose, typically essays or Wikipedia articles. The model is then asked to retrieve that fact. The test is repeated across varying context lengths and insertion depths to produce a heatmap of retrieval accuracy.

This approach has a fundamental flaw: **the needle is semantically alien to the haystack**.

A sentence like *"The special magic number in the city of Paris is 7342"* stands out sharply against surrounding prose about history or nature. It is structurally different, lexically unusual, and contextually incongruous. Attention mechanisms, which are sensitive to contrast and surprise, are likely to weight such a token sequence more heavily simply because it breaks the surrounding pattern — not because the model has genuinely developed robust long-range retrieval capability.

The result is that the standard test measures a combination of:

- true positional retrieval ability
- sensitivity to semantic contrast
- the model's ability to detect structural anomalies

These are conflated into a single score, making it impossible to isolate what is actually being measured.

---

## The Homogeneous Haystack Approach

The solution is to eliminate contrast entirely by making the needle **structurally identical** to every other element in the context. Instead of embedding a distinctive sentence in prose, the entire context consists of a large flat list of random key→value pairs, generated uniformly from start to finish. The "needles" are not inserted or seeded — they are simply some of those pairs that the test harness designates as query targets after the context is already fully formed.

From the model's perspective there is no needle. Every line is indistinguishable from every other line. There are no semantic outliers, no structural anomalies, no contrast signals of any kind. The designation of a pair as a target exists only in the test harness, never in the context. The only way for a model to correctly retrieve the value associated with a queried key is to have genuinely attended to and retained that specific position in the context.

### Example structure

```
KXTQM = 8471
RLVNP = 3209
BWSDF = 6714
ZQPLR = 1053       ← designated as query target by the harness, invisible to the model
FNTCK = 9382
MHVWB = 4467
...
DKRJT = 7291       ← designated as query target by the harness, invisible to the model
YSBQM = 2834
...
```

The query is simply: *"What is the value associated with ZQPLR?"*

The model either retrieved it or it did not. No inference, no convention, no prior knowledge can help. The keys are random strings with no semantic content. The values are arbitrary numbers with no conventional expectation.

---

## Why This Design Is Preferable

### 1. It isolates a single variable

The homogeneous haystack measures exactly one thing: the ability to retrieve a specific token sequence from a specific position in a long context. All confounds — semantic contrast, structural anomaly detection, language model priors — are neutralized by construction.

### 2. It is harder and more honest

Because the model receives no contrast signal to guide attention, it must rely on genuine positional encoding and KV cache retention. This makes the test more demanding and more truthful about actual long-context capability. A model that scores well here has genuinely retained the information, not merely detected an anomaly.

### 3. It is more relevant to real workloads

In practice, models working over long documents — codebases, legal texts, technical manuals — must retrieve facts that are unremarkable relative to their surroundings. A constant value among many constants, a parameter name among many parameters. The homogeneous haystack better approximates this reality than a distinctive sentence embedded in prose.

### 4. It eliminates the quantization confound

Models running under quantization (e.g. with techniques like TurboQuant) have reduced precision in their KV cache entries. A high-contrast needle may still be retrievable under quantization because the signal strength is large enough to survive precision loss. A homogeneous needle, by contrast, has a signal strength that is indistinguishable from its neighbors, making the test more sensitive to quantization-induced degradation. This is particularly important when evaluating locally-hosted quantized models.

### 5. Scoring is unambiguous

Exact match scoring is trivially applied. The ground truth is known, the answer space is constrained, and there is no need for an LLM-based evaluator to judge response quality. This eliminates an additional source of noise present in some standard implementations.

---

## Relationship to the Standard Test

The homogeneous haystack test is not a replacement for the standard NIAH test — it is a complementary instrument that measures a different point on the capability spectrum.

| Dimension | Standard NIAH | Homogeneous NIAH |
|---|---|---|
| Haystack content | Natural prose | Random key→value pairs |
| Needle visibility | High contrast | Zero contrast |
| Confounds | Semantic contrast, priors | None by design |
| Difficulty | Lower | Higher |
| Real-world relevance | Moderate | Higher for dense technical content |
| Scoring | Often LLM-evaluated | Exact match |
| Quantization sensitivity | Low | High |

Used together, the two tests provide a richer picture: the standard test gives a ceiling that includes contrast assistance, the homogeneous test gives a floor that excludes it. The gap between the two scores is itself informative about how much a given model relies on contrast to perform retrieval.

---

## Implementation Notes

### Key and value generation

Keys should be random uppercase strings of fixed length (5–8 characters) with no resemblance to real words or identifiers. Values should be random integers in a range that excludes obviously "round" numbers to prevent any prior-based guessing (e.g. 1000–9999, avoiding multiples of 100).

### Haystack size and query count

The haystack should be large enough to reach the target context length (e.g. 100k–300k tokens). The number of query targets should be sufficient to sample multiple depth positions per run — 10–20 targets across 5–10 depth percentages is a reasonable baseline. All pairs are generated uniformly; query targets are selected by the harness after generation, not inserted.

### Depth positions

Query targets should be selected from evenly distributed depth positions across the already-generated context: for example 5%, 15%, 25%, ... 95%. This produces a one-dimensional accuracy-vs-depth profile that reveals primacy and recency bias.

### Format choice

The format of pairs is a meaningful design decision. A token-like format (`KEY = VALUE`) minimizes linguistic framing and further suppresses model priors. A more prose-like format (`The value of KEY is VALUE`) reintroduces mild linguistic structure. The token-like format is recommended for maximum rigor; the prose format may be useful as an intermediate variant between homogeneous and standard NIAH.

### Evaluation

All queries should be submitted in a single prompt together with the full context. Since the target values are random and uncorrelated, there is no cross-contamination risk between answers — one correct or incorrect retrieval cannot influence another. A single-pass approach is more efficient, reflects real usage more accurately, and slightly increases difficulty by requiring the model to track multiple targets simultaneously. Temperature should be set to zero for deterministic output. Scoring is exact match after stripping whitespace. Results should be logged with the associated depth position and context length to enable heatmap visualization identical to standard NIAH tooling.

---

## Relationship to Current Criticism of NIAH

The concern that standard NIAH benchmarks overestimate true long-context capability is not new, and this design connects to an active line of research. NeedleChain (Moon & Lim, 2025) makes the overestimation argument explicitly, demonstrating that even state-of-the-art models like GPT-4o struggle when the haystack consists entirely of query-relevant content rather than irrelevant filler. EverMemBench (2025) replaces the standard haystack with semantically similar hard negatives, exposing a "realism gap" in conventional evaluation. Both works arrive at the same conclusion from different directions: the irrelevant, low-contrast haystack of standard NIAH makes retrieval artificially easier than it would be in practice.

The homogeneous approach described here is complementary to these efforts but attacks the problem from a different angle. Rather than making the haystack harder by increasing semantic similarity to the needle, it eliminates the contrast signal entirely by making every element structurally identical. This is a more radical intervention and likely a harder test, but it targets the same fundamental weakness.

A precise claim about the direction and magnitude of performance differences will be made once empirical results are available. What can be said in advance is that the delta between a model's score on standard NIAH and its score on the homogeneous test should be interpretable as a direct measure of how much the model relies on contrast assistance — and by extension, how much published benchmark scores would degrade under real workloads where content is dense and uniform.

---

## Limitations

The homogeneous haystack test measures retrieval of arbitrary facts with no semantic grounding. It does not measure — and is not intended to measure — the model's ability to reason over, synthesize, or apply retrieved information. For that, subjective evaluation over real workloads remains the appropriate instrument, with all the informality that implies. The two approaches are complementary, not competing.

---

## Summary

The homogeneous needle-in-a-haystack test is a stricter, more honest, and more practically relevant instrument for evaluating long-context retrieval in large language models. By ensuring that every element in the context is structurally identical, it eliminates the semantic contrast bias present in standard implementations and forces genuine positional retrieval. It is particularly well suited for evaluating quantized local models where KV cache precision loss is a first-order concern. The implementation is straightforward and requires no external dependencies beyond a basic API client.
