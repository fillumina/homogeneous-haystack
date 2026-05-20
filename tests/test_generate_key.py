
import pytest

from haystack_test import generate_key


class TestGenerateKey:
    @pytest.mark.parametrize("length", [5, 8, 10, 12])
    def test_returns_correct_length(self, length, seeded_random):
        seeded_random(42)
        result = generate_key(length)
        assert len(result) == length

    def test_contains_only_uppercase(self, seeded_random):
        seeded_random(42)
        for _ in range(10):
            result = generate_key(8)
            assert result.isupper()
            assert result.isalpha()

    def test_contains_only_ascii_letters(self, seeded_random):
        seeded_random(42)
        result = generate_key(100)
        for ch in result:
            assert ch in "ABCDEFGHIJKLMNOPQRSTUVWXYZ"

    def test_no_repeated_keys_in_batch(self, seeded_random):
        seeded_random(42)
        keys = set()
        for _ in range(50):
            keys.add(generate_key(8))
        assert len(keys) == 50
