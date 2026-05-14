import random

import pytest

from haystack_test import generate_value


class TestGenerateValue:
    @pytest.mark.parametrize("val_min,val_max", [
        (1000, 9999),
        (100, 500),
        (5000, 6000),
        (1, 100),
    ])
    def test_returns_value_in_range(self, val_min, val_max, seeded_random):
        seeded_random(42)
        result = generate_value(val_min, val_max)
        assert val_min <= result <= val_max

    @pytest.mark.parametrize("val_min,val_max", [
        (1000, 9999),
        (100, 500),
        (5000, 6000),
        (1, 100),
    ])
    def test_never_divisible_by_100(self, val_min, val_max, seeded_random):
        seeded_random(42)
        result = generate_value(val_min, val_max)
        assert result % 100 != 0

    def test_different_seeds_give_different_results(self):
        random.seed(1)
        result1 = generate_value()
        random.seed(2)
        result2 = generate_value()
        assert result1 != result2

    def test_same_seed_gives_same_result(self):
        random.seed(1)
        result1 = generate_value()
        random.seed(1)
        result2 = generate_value()
        assert result1 == result2

    def test_returns_integer(self, seeded_random):
        seeded_random(42)
        result = generate_value()
        assert isinstance(result, int)

    def test_avoids_round_numbers(self):
        """Test that values ending in 00 are excluded."""
        random.seed(42)
        for _ in range(100):
            val = generate_value()
            assert val % 100 != 0, f"Generated round number: {val}"
