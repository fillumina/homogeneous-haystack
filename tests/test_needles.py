import pytest

from haystack_test import (
    Config,
    Needle,
    create_distractor_keys,
    generate_haystack_and_needles,
    pick_needle_positions,
    select_needles,
    shake_positions,
    validate_generation_params,
)


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
        """select_needles returns all needles as real needles now."""
        _, pairs = haystack_pairs
        result = select_needles(pairs, 25)
        assert len(result) == 25
        for n in result:
            assert n.is_distractor is False

    def test_distractor_needles_separate(self, haystack_pairs):
        """Distractors are created by create_distractor_keys, not select_needles."""
        from haystack_test import create_distractor_keys
        _, pairs = haystack_pairs
        real = select_needles(pairs, 20)
        distractors = create_distractor_keys(pairs, 5, 20, 8)
        assert len(real) == 20
        assert len(distractors) == 5
        for n in real:
            assert n.is_distractor is False
        for n in distractors:
            assert n.is_distractor is True

    def test_real_needles_have_expected_values(self, haystack_pairs):
        _, pairs = haystack_pairs
        result = select_needles(pairs, 10)
        for needle in result:
            assert needle.expected is not None
            assert isinstance(needle.expected, int)
            assert needle.is_distractor is False

    def test_distractor_needles_have_none_expected(self, haystack_pairs):
        from haystack_test import create_distractor_keys
        _, pairs = haystack_pairs
        distractors = create_distractor_keys(pairs, 10, 20, 8)
        for needle in distractors:
            assert needle.expected is None
            assert needle.is_distractor is True

    def test_needles_sorted_by_index(self, haystack_pairs):
        _, pairs = haystack_pairs
        result = select_needles(pairs, 20)
        indices = [n.index for n in result]
        assert indices == sorted(indices)

    def test_indices_in_valid_range(self, haystack_pairs):
        _, pairs = haystack_pairs
        result = select_needles(pairs, 20)
        for needle in result:
            assert 0 <= needle.index < len(pairs)

    def test_distractor_keys_not_in_haystack(self, haystack_pairs, haystack_keys):
        from haystack_test import create_distractor_keys
        _, pairs = haystack_pairs
        distractors = create_distractor_keys(pairs, 5, 20, 8)
        for needle in distractors:
            assert needle.key not in haystack_keys

    def test_real_needles_keys_in_haystack(self, haystack_pairs, haystack_keys):
        _, pairs = haystack_pairs
        result = select_needles(pairs, 20)
        for needle in result:
            assert needle.key in haystack_keys

    def test_all_needles_have_required_fields(self, haystack_pairs):
        _, pairs = haystack_pairs
        result = select_needles(pairs, 10)
        for needle in result:
            assert isinstance(needle, Needle)
            assert hasattr(needle, "index")
            assert hasattr(needle, "key")
            assert hasattr(needle, "expected")
            assert hasattr(needle, "is_distractor")



class TestNumNeedlesGreaterThanHaystackN:
    def test_only_haystack_num_real_needles_when_requested_exceeds_haystack(self, seeded_random):
        seeded_random(42)
        haystack_num = 17
        needles_num = 100
        _, pairs, needles = generate_haystack_and_needles(
            haystack_num=haystack_num,
            needles_num=needles_num,
            distractor_pct=0.08,
            distractors_num=None,
            key_len=8,
            val_min=10000,
            val_max=99999,
            fuzz=0,
        )
        real = [n for n in needles if not n.is_distractor]
        assert len(real) == haystack_num

    def test_distractor_count_based_on_actual_real_needles(self, seeded_random):
        seeded_random(42)
        haystack_num = 17
        needles_num = 100
        distractor_pct = 0.08
        _, _, needles = generate_haystack_and_needles(
            haystack_num=haystack_num,
            needles_num=needles_num,
            distractor_pct=distractor_pct,
            distractors_num=None,
            key_len=8,
            val_min=10000,
            val_max=99999,
            fuzz=0,
        )
        real = [n for n in needles if not n.is_distractor]
        distractors = [n for n in needles if n.is_distractor]
        expected_distractors = int(len(real) * distractor_pct)
        assert len(distractors) == expected_distractors

    def test_total_needles_is_real_plus_distractors(self, seeded_random):
        seeded_random(42)
        haystack_num = 17
        needles_num = 100
        distractor_pct = 0.08
        _, _, needles = generate_haystack_and_needles(
            haystack_num=haystack_num,
            needles_num=needles_num,
            distractor_pct=distractor_pct,
            distractors_num=None,
            key_len=8,
            val_min=10000,
            val_max=99999,
            fuzz=0,
        )
        real = [n for n in needles if not n.is_distractor]
        distractors = [n for n in needles if n.is_distractor]
        assert len(needles) == len(real) + len(distractors)

    def test_default_case_still_works(self, seeded_random):
        seeded_random(42)
        haystack_num = 100
        needles_num = 50
        distractor_pct = 0.1
        _, _, needles = generate_haystack_and_needles(
            haystack_num=haystack_num,
            needles_num=needles_num,
            distractor_pct=distractor_pct,
            distractors_num=None,
            key_len=8,
            val_min=10000,
            val_max=99999,
            fuzz=0,
        )
        real = [n for n in needles if not n.is_distractor]
        distractors = [n for n in needles if n.is_distractor]
        assert len(real) == needles_num
        assert len(distractors) == int(needles_num * distractor_pct)

    def test_no_distractors_when_pct_is_zero(self, seeded_random):
        seeded_random(42)
        haystack_num = 17
        needles_num = 100
        _, _, needles = generate_haystack_and_needles(
            haystack_num=haystack_num,
            needles_num=needles_num,
            distractor_pct=0.0,
            distractors_num=None,
            key_len=8,
            val_min=10000,
            val_max=99999,
            fuzz=0,
        )
        distractors = [n for n in needles if n.is_distractor]
        assert len(distractors) == 0


class TestDistractorsNum:
    def test_exact_distractor_count(self, seeded_random):
        seeded_random(42)
        haystack_num = 50
        needles_num = 20
        exact_distractors = 5
        _, _, needles = generate_haystack_and_needles(
            haystack_num=haystack_num,
            needles_num=needles_num,
            distractor_pct=None,
            distractors_num=exact_distractors,
            key_len=8,
            val_min=10000,
            val_max=99999,
            fuzz=0,
        )
        real = [n for n in needles if not n.is_distractor]
        distractors = [n for n in needles if n.is_distractor]
        assert len(real) == needles_num
        assert len(distractors) == exact_distractors

    def test_zero_distractors_with_num(self, seeded_random):
        seeded_random(42)
        haystack_num = 17
        needles_num = 100
        _, _, needles = generate_haystack_and_needles(
            haystack_num=haystack_num,
            needles_num=needles_num,
            distractor_pct=None,
            distractors_num=0,
            key_len=8,
            val_min=10000,
            val_max=99999,
            fuzz=0,
        )
        distractors = [n for n in needles if n.is_distractor]
        assert len(distractors) == 0

    def test_default_8_pct_when_neither_set(self, seeded_random):
        seeded_random(42)
        haystack_num = 50
        needles_num = 20
        _, _, needles = generate_haystack_and_needles(
            haystack_num=haystack_num,
            needles_num=needles_num,
            distractor_pct=None,
            distractors_num=None,
            key_len=8,
            val_min=10000,
            val_max=99999,
            fuzz=0,
        )
        real = [n for n in needles if not n.is_distractor]
        distractors = [n for n in needles if n.is_distractor]
        expected = int(len(real) * 0.08)
        assert len(distractors) == expected

    def test_distractors_num_overrides_pct(self, seeded_random):
        seeded_random(42)
        haystack_num = 50
        needles_num = 20
        _, _, needles = generate_haystack_and_needles(
            haystack_num=haystack_num,
            needles_num=needles_num,
            distractor_pct=0.5,
            distractors_num=2,
            key_len=8,
            val_min=10000,
            val_max=99999,
            fuzz=0,
        )
        distractors = [n for n in needles if n.is_distractor]
        assert len(distractors) == 2


class TestValidation:
    def test_both_distractor_options_set_raises(self):
        config = Config(
            endpoint="http://localhost:8080/v1/chat/completions",
            model=None,
            key_len=8,
            val_min=10000,
            val_max=99999,
            haystack_num=100,
            needles_num=50,
            distractor_pct=0.1,
            distractors_num=5,
            temperature=0.0,
            max_tokens=240000,
            timeout=7200,
            full=False,
        )
        with pytest.raises(ValueError, match="cannot both be set"):
            validate_generation_params(config)

    def test_negative_distractors_num_raises(self):
        config = Config(
            endpoint="http://localhost:8080/v1/chat/completions",
            model=None,
            key_len=8,
            val_min=10000,
            val_max=99999,
            haystack_num=100,
            needles_num=50,
            distractor_pct=None,
            distractors_num=-1,
            temperature=0.0,
            max_tokens=240000,
            timeout=7200,
            full=False,
        )
        with pytest.raises(ValueError, match="distractors_num must be >= 0"):
            validate_generation_params(config)

    def test_valid_no_distractors(self):
        config = Config(
            endpoint="http://localhost:8080/v1/chat/completions",
            model=None,
            key_len=8,
            val_min=10000,
            val_max=99999,
            haystack_num=100,
            needles_num=50,
            distractor_pct=None,
            distractors_num=0,
            temperature=0.0,
            max_tokens=240000,
            timeout=7200,
            full=False,
        )
        validate_generation_params(config)

    def test_valid_pct_only(self):
        config = Config(
            endpoint="http://localhost:8080/v1/chat/completions",
            model=None,
            key_len=8,
            val_min=10000,
            val_max=99999,
            haystack_num=100,
            needles_num=50,
            distractor_pct=0.1,
            distractors_num=None,
            temperature=0.0,
            max_tokens=240000,
            timeout=7200,
            full=False,
        )
        validate_generation_params(config)

    def test_valid_num_only(self):
        config = Config(
            endpoint="http://localhost:8080/v1/chat/completions",
            model=None,
            key_len=8,
            val_min=10000,
            val_max=99999,
            haystack_num=100,
            needles_num=50,
            distractor_pct=None,
            distractors_num=10,
            temperature=0.0,
            max_tokens=240000,
            timeout=7200,
            full=False,
        )
        validate_generation_params(config)


class TestShakePositions:
    def test_fuzz_zero_is_deterministic(self):
        base = pick_needle_positions(100, 10)
        r1 = shake_positions(base, 0)
        r2 = shake_positions(base, 0)
        assert r1 == r2 == base

    def test_fuzz_zero_preserves_uniform_spacing(self):
        base = pick_needle_positions(100, 10)
        result = shake_positions(base, 0)
        diffs = [result[i+1] - result[i] for i in range(len(result)-1)]
        assert max(diffs) - min(diffs) <= 1

    def test_fuzz_breaks_uniform_spacing(self, seeded_random):
        seeded_random(42)
        base = pick_needle_positions(100, 10)
        result = shake_positions(base, 0.3)
        diffs = [result[i+1] - result[i] for i in range(len(result)-1)]
        assert max(diffs) - min(diffs) > 1

    def test_fuzz_positions_in_range(self, seeded_random):
        seeded_random(42)
        base = pick_needle_positions(100, 10)
        result = shake_positions(base, 0.5)
        for pos in result:
            assert 0 <= pos < 100

    def test_fuzz_positions_still_sorted(self, seeded_random):
        seeded_random(42)
        base = pick_needle_positions(100, 10)
        result = shake_positions(base, 0.5)
        assert result == sorted(result)

    def test_fuzz_positions_unique(self, seeded_random):
        seeded_random(42)
        base = pick_needle_positions(100, 10)
        result = shake_positions(base, 0.5)
        assert len(result) == len(set(result))

    def test_fuzz_large_does_not_deduplicate_all(self, seeded_random):
        seeded_random(42)
        base = pick_needle_positions(1000, 10)
        result = shake_positions(base, 0.5)
        assert len(result) == 10

    def test_fuzz_positions_are_randomized(self, seeded_random):
        seeded_random(42)
        base1 = pick_needle_positions(100, 10)
        result1 = shake_positions(base1, 0.5)
        seeded_random(99)
        base2 = pick_needle_positions(100, 10)
        result2 = shake_positions(base2, 0.5)
        assert result1 != result2

    def test_fuzz_boundary_clamp_low(self):
        # With 100% fuzz, all positions should be clamped to valid range
        base = pick_needle_positions(10, 2)
        result = shake_positions(base, 1.0)
        assert all(0 <= p < 10 for p in result)
        assert len(result) == 2

    def test_fuzz_boundary_clamp_high(self):
        base = pick_needle_positions(10, 2)
        result = shake_positions(base, 1.0)
        assert all(p < 10 for p in result)

    def test_fuzz_with_single_needle(self, seeded_random):
        seeded_random(42)
        base = pick_needle_positions(100, 1)
        result = shake_positions(base, 0.5)
        assert len(result) == 1
        assert 0 <= result[0] < 100

    def test_fuzz_returns_copy_not_mutated_base(self):
        base = pick_needle_positions(100, 10)
        original = base.copy()
        import random
        random.seed(42)
        result = shake_positions(base, 0.5)
        # Base should not be mutated
        assert base == original
        assert result is not base
