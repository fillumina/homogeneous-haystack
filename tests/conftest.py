import random

import pytest

from haystack_test import (
    build_haystack,
    select_needles,
    HaystackQueryError,
)


@pytest.fixture
def seeded_random():
    """Return a function that sets random.seed and returns the seed value."""
    def _set(seed):
        random.seed(seed)
        return seed
    return _set


@pytest.fixture
def haystack_pairs(seeded_random):
    """Generate a haystack with 100 pairs using a fixed seed."""
    seeded_random(42)
    text, pairs = build_haystack(100)
    return text, pairs


@pytest.fixture
def haystack_pairs_large(seeded_random):
    """Generate a haystack with 5000 pairs using a fixed seed."""
    seeded_random(42)
    text, pairs = build_haystack(5000)
    return text, pairs


@pytest.fixture
def needles(seeded_random, haystack_pairs):
    """Generate needles from the haystack fixture."""
    _, pairs = haystack_pairs
    seeded_random(123)
    result = select_needles(pairs, n_needles=20)
    return result


@pytest.fixture
def haystack_keys(haystack_pairs):
    _, pairs = haystack_pairs
    return set(k for k, _ in pairs)


@pytest.fixture
def mock_api_response(content="42", model="test-model"):
    """Return a mock OpenAI-compatible API response."""
    return {
        "model": model,
        "choices": [
            {
                "message": {
                    "role": "assistant",
                    "content": content,
                }
            }
        ],
    }


@pytest.fixture
def mock_api_error():
    """Return a function that raises HaystackQueryError."""
    def _raise(msg="API error"):
        raise HaystackQueryError(msg)
    return _raise
