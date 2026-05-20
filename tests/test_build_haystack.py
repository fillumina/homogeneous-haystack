import pytest

from haystack_test import build_haystack


class TestBuildHaystack:
    def test_returns_text_and_pairs(self, seeded_random):
        seeded_random(42)
        result = build_haystack(10)
        assert isinstance(result, tuple)
        assert len(result) == 2
        text, pairs = result
        assert isinstance(text, str)
        assert isinstance(pairs, list)

    @pytest.mark.parametrize("n", [1, 10, 100, 5000])
    def test_correct_number_of_pairs(self, n, seeded_random):
        seeded_random(42)
        _, pairs = build_haystack(n)
        assert len(pairs) == n

    @pytest.mark.parametrize("n", [1, 10, 100])
    def test_correct_number_of_lines(self, n, seeded_random):
        seeded_random(42)
        text, _ = build_haystack(n)
        lines = [line for line in text.strip().split("\n") if line]
        assert len(lines) == n

    def test_format_is_key_equals_value(self, seeded_random):
        seeded_random(42)
        text, _ = build_haystack(10)
        for line in text.strip().split("\n"):
            assert " = " in line
            parts = line.split(" = ", 1)
            assert len(parts) == 2

    def test_keys_are_uppercase_strings(self, seeded_random):
        seeded_random(42)
        _, pairs = build_haystack(50)
        for key, _ in pairs:
            assert key.isupper()
            assert key.isalpha()

    @pytest.mark.parametrize("val_min,val_max", [(1000, 9999), (100, 500)])
    def test_values_in_range(self, val_min, val_max, seeded_random):
        seeded_random(42)
        _, pairs = build_haystack(50, val_min=val_min, val_max=val_max)
        for _, val in pairs:
            assert val_min <= val <= val_max
            assert val % 100 != 0

    def test_default_key_length(self, seeded_random):
        seeded_random(42)
        _, pairs = build_haystack(10)
        for key, _ in pairs:
            assert len(key) == 8

    def test_custom_key_length(self, seeded_random):
        seeded_random(42)
        _, pairs = build_haystack(10, key_len=6)
        for key, _ in pairs:
            assert len(key) == 6

    def test_all_keys_unique(self, seeded_random):
        seeded_random(42)
        _, pairs = build_haystack(100)
        keys = [k for k, _ in pairs]
        assert len(keys) == len(set(keys))

    def test_text_matches_pairs(self, seeded_random):
        seeded_random(42)
        text, pairs = build_haystack(20)
        for line in text.strip().split("\n"):
            key, value_str = line.split(" = ", 1)
            value = int(value_str)
            assert (key, value) in pairs
