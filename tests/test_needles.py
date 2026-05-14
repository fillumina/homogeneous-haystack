import pytest

from haystack_test import pick_needle_positions, select_needles


class TestPickNeedlePositions:
    @pytest.mark.parametrize("n_total,n_needles", [
        (100, 10),
        (5000, 100),
        (1000, 50),
        (50, 5),
    ])
    def test_returns_correct_count(self, n_total, n_needles):
        result = pick_needle_positions(n_total, n_needles)
        assert len(result) == n_needles

    def test_positions_in_range(self):
        result = pick_needle_positions(100, 10)
        for pos in result:
            assert 0 <= pos < 100

    def test_positions_are_sorted(self):
        result = pick_needle_positions(100, 10)
        assert result == sorted(result)

    def test_positions_are_uniformly_spaced(self):
        result = pick_needle_positions(100, 5)
        diffs = [result[i+1] - result[i] for i in range(len(result)-1)]
        # All diffs should be approximately equal
        assert max(diffs) - min(diffs) <= 1

    def test_n_needles_equals_n_total(self):
        result = pick_needle_positions(10, 10)
        assert result == list(range(10))

    def test_n_needles_greater_than_n_total(self):
        result = pick_needle_positions(5, 10)
        assert result == list(range(5))

    def test_n_needles_is_one(self):
        result = pick_needle_positions(100, 1)
        assert len(result) == 1
        assert 0 <= result[0] < 100

    def test_n_total_is_one(self):
        result = pick_needle_positions(1, 1)
        assert result == [0]


class TestSelectNeedles:
    def test_returns_list(self, haystack_pairs):
        _, pairs = haystack_pairs
        result = select_needles(pairs, 10)
        assert isinstance(result, list)

    def test_correct_total_count(self, haystack_pairs):
        _, pairs = haystack_pairs
        result = select_needles(pairs, 20)
        assert len(result) == 20

    def test_real_needles_count(self, haystack_pairs):
        """Default distractor_pct=0.08 means 92% are real."""
        _, pairs = haystack_pairs
        result = select_needles(pairs, 25, distractor_pct=0.08)
        real = [n for n in result if not n["is_distractor"]]
        expected_real = int(25 * (1 - 0.08))
        assert len(real) == expected_real

    def test_distractor_needles_count(self, haystack_pairs):
        _, pairs = haystack_pairs
        result = select_needles(pairs, 25, distractor_pct=0.2)
        distractors = [n for n in result if n["is_distractor"]]
        expected_distractors = 25 - int(25 * (1 - 0.2))
        assert len(distractors) == expected_distractors

    def test_real_needles_have_expected_values(self, haystack_pairs):
        _, pairs = haystack_pairs
        result = select_needles(pairs, 10, distractor_pct=0.0)
        for needle in result:
            assert needle["expected"] is not None
            assert isinstance(needle["expected"], int)
            assert needle["is_distractor"] is False

    def test_distractor_needles_have_none_expected(self, haystack_pairs):
        _, pairs = haystack_pairs
        result = select_needles(pairs, 10, distractor_pct=0.3)
        distractors = [n for n in result if n["is_distractor"]]
        for needle in distractors:
            assert needle["expected"] is None
            assert needle["is_distractor"] is True

    def test_needles_sorted_by_index(self, haystack_pairs):
        _, pairs = haystack_pairs
        result = select_needles(pairs, 20)
        indices = [n["index"] for n in result]
        assert indices == sorted(indices)

    def test_indices_in_valid_range(self, haystack_pairs):
        _, pairs = haystack_pairs
        result = select_needles(pairs, 20)
        for needle in result:
            assert 0 <= needle["index"] < len(pairs)

    def test_distractor_keys_not_in_haystack(self, haystack_pairs, haystack_keys):
        _, pairs = haystack_pairs
        result = select_needles(pairs, 20, distractor_pct=0.3, haystack_keys=haystack_keys)
        distractors = [n for n in result if n["is_distractor"]]
        for needle in distractors:
            assert needle["key"] not in haystack_keys

    def test_real_needles_keys_in_haystack(self, haystack_pairs, haystack_keys):
        _, pairs = haystack_pairs
        result = select_needles(pairs, 20, distractor_pct=0.0, haystack_keys=haystack_keys)
        for needle in result:
            assert needle["key"] in haystack_keys

    def test_all_needles_have_required_fields(self, haystack_pairs):
        _, pairs = haystack_pairs
        result = select_needles(pairs, 10)
        for needle in result:
            assert "index" in needle
            assert "key" in needle
            assert "expected" in needle
            assert "is_distractor" in needle
