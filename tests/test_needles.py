import pytest

from haystack_test import (
    ApiConfig,
    Config,
    ExecutionConfig,
    HaystackConfig,
    Needle,
    OutputConfig,
    create_distractor_keys,
    generate_haystack_and_needles,
    pick_needle_positions,
    resolve_distractors_num,
    select_needles,
    shake_positions,
    _validate_params,
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
        result = select_needles(pairs, HaystackConfig(needles_num=10))
        assert isinstance(result, list)

    def test_correct_total_count(self, haystack_pairs):
        _, pairs = haystack_pairs
        result = select_needles(pairs, HaystackConfig(needles_num=20))
        assert len(result) == 20

    def test_real_needles_count(self, haystack_pairs):
        """select_needles returns all needles as real needles now."""
        _, pairs = haystack_pairs
        result = select_needles(pairs, HaystackConfig(needles_num=25))
        assert len(result) == 25
        for n in result:
            assert n.is_distractor is False

    def test_distractor_needles_separate(self, haystack_pairs):
        """Distractors are created by create_distractor_keys, not select_needles."""
        _, pairs = haystack_pairs
        real = select_needles(pairs, HaystackConfig(needles_num=20))
        distractors = create_distractor_keys(HaystackConfig(distractors_num=5, key_len=8), pairs, 20)
        assert len(real) == 20
        assert len(distractors) == 5
        for n in real:
            assert n.is_distractor is False
        for n in distractors:
            assert n.is_distractor is True

    def test_real_needles_have_expected_values(self, haystack_pairs):
        _, pairs = haystack_pairs
        result = select_needles(pairs, HaystackConfig(needles_num=10))
        for needle in result:
            assert needle.expected is not None
            assert isinstance(needle.expected, int)
            assert needle.is_distractor is False

    def test_distractor_needles_have_none_expected(self, haystack_pairs):
        _, pairs = haystack_pairs
        distractors = create_distractor_keys(HaystackConfig(distractors_num=10, key_len=8), pairs, 20)
        for needle in distractors:
            assert needle.expected is None
            assert needle.is_distractor is True

    def test_needles_sorted_by_index(self, haystack_pairs):
        _, pairs = haystack_pairs
        result = select_needles(pairs, HaystackConfig(needles_num=20))
        indices = [n.index for n in result]
        assert indices == sorted(indices)

    def test_indices_in_valid_range(self, haystack_pairs):
        _, pairs = haystack_pairs
        result = select_needles(pairs, HaystackConfig(needles_num=20))
        for needle in result:
            assert 0 <= needle.index < len(pairs)

    def test_distractor_keys_not_in_haystack(self, haystack_pairs, haystack_keys):
        _, pairs = haystack_pairs
        distractors = create_distractor_keys(HaystackConfig(distractors_num=5, key_len=8), pairs, 20)
        for needle in distractors:
            assert needle.key not in haystack_keys

    def test_real_needles_keys_in_haystack(self, haystack_pairs, haystack_keys):
        _, pairs = haystack_pairs
        result = select_needles(pairs, HaystackConfig(needles_num=20))
        for needle in result:
            assert needle.key in haystack_keys

    def test_all_needles_have_required_fields(self, haystack_pairs):
        _, pairs = haystack_pairs
        result = select_needles(pairs, HaystackConfig(needles_num=10))
        for needle in result:
            assert isinstance(needle, Needle)
            assert hasattr(needle, "index")
            assert hasattr(needle, "key")
            assert hasattr(needle, "expected")
            assert hasattr(needle, "is_distractor")


VALIDATION_CASES = [
    ({"haystack.distractors_num": -1}, "distractors_num must be >= 0"),
    ({"haystack.haystack_num": 0}, "haystack_num must be >= 1"),
    ({"haystack.haystack_num": -1}, "haystack_num must be >= 1"),
    ({"haystack.needles_num": 0}, "needles_num must be >= 1"),
    ({"haystack.needles_num": -5}, "needles_num must be >= 1"),
    ({"haystack.key_len": 0}, "key_len must be >= 1"),
    ({"haystack.key_len": -3}, "key_len must be >= 1"),
    ({"haystack.val_min": 99999, "haystack.val_max": 10000}, "val_min.*> val_max"),
    ({"haystack.fuzz": -0.1}, "fuzz must be in"),
    ({"haystack.fuzz": 0.5}, "fuzz must be in"),
    ({"haystack.fuzz": 0.9}, "fuzz must be in"),
]


class TestValidation:
    @pytest.mark.parametrize("overrides, expected_msg", VALIDATION_CASES)
    def test_invalid_params_raise(self, overrides, expected_msg):
        config = Config(
            api=ApiConfig(
                endpoint="http://localhost:8080/v1/chat/completions",
                temperature=0.0,
                max_tokens=240000,
                timeout=7200,
            ),
            haystack=HaystackConfig(
                haystack_num=100,
                needles_num=50,
                distractors_num=0,
                key_len=8,
                val_min=10000,
                val_max=99999,
            ),
            output=OutputConfig(
                output_filename="results.csv",
                verbosity="medium",
                k_quant="Q4_0",
                v_quant="Q8_0",
                note="test",
            ),
            execution=ExecutionConfig(
                repeat=1,
                stop_on_error=False,
            ),
        )
        for k, v in overrides.items():
            if "." in k:
                sub, attr = k.split(".", 1)
                setattr(getattr(config, sub), attr, v)
            else:
                setattr(config, k, v)
        with pytest.raises(ValueError, match=expected_msg):
            _validate_params(config)



class TestDistractors:
    def test_exceeds_haystack_capped_to_haystack(self, seeded_random):
        seeded_random(42)
        _, _, needles = generate_haystack_and_needles(HaystackConfig(haystack_num=17, needles_num=100, distractors_num=2, key_len=8, val_min=10000, val_max=99999, fuzz=0))
        real = [n for n in needles if not n.is_distractor]
        assert len(real) == 17

    def test_default_case(self, seeded_random):
        seeded_random(42)
        _, _, needles = generate_haystack_and_needles(HaystackConfig(haystack_num=100, needles_num=50, distractors_num=5, key_len=8, val_min=10000, val_max=99999, fuzz=0))
        real = [n for n in needles if not n.is_distractor]
        distractors = [n for n in needles if n.is_distractor]
        assert len(real) == 50
        assert len(distractors) == 5

    @pytest.mark.parametrize("distractors_num", [0, 2, 5])
    def test_exact_distractor_count(self, seeded_random, distractors_num):
        seeded_random(42)
        _, _, needles = generate_haystack_and_needles(HaystackConfig(haystack_num=50, needles_num=20, distractors_num=distractors_num, key_len=8, val_min=10000, val_max=99999, fuzz=0))
        distractors = [n for n in needles if n.is_distractor]
        assert len(distractors) == distractors_num


class TestResolveDistractorsNum:
    def test_explicit_num_takes_priority(self):
        assert resolve_distractors_num(100, 5, 0.5) == 5

    def test_pct_used_when_num_none(self):
        assert resolve_distractors_num(100, None, 0.25) == 25

    def test_default_20_when_both_none(self):
        result = resolve_distractors_num(100, None, None)
        assert result == 20

    def test_default_capped_at_needles(self):
        result = resolve_distractors_num(13, None, None)
        assert result == 13

    def test_zero_real_needles(self):
        assert resolve_distractors_num(0, None, None) == 0
        assert resolve_distractors_num(0, 5, 0.5) == 5

    def test_pct_zero(self):
        assert resolve_distractors_num(100, None, 0.0) == 0


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
        result = shake_positions(base, 0.49)
        for pos in result:
            assert 0 <= pos < 100

    def test_fuzz_positions_still_sorted(self, seeded_random):
        seeded_random(42)
        base = pick_needle_positions(100, 10)
        result = shake_positions(base, 0.49)
        assert result == sorted(result)

    def test_fuzz_no_collisions(self, seeded_random):
        # fuzz < 0.5 + int() truncation guarantees no duplicates
        for _ in range(1000):
            base = pick_needle_positions(1000, 50)
            result = shake_positions(base, 0.49)
            assert len(result) == len(set(result)), (
                f"Collision detected: {base} -> {result}"
            )

    def test_fuzz_positions_are_randomized(self, seeded_random):
        seeded_random(42)
        base1 = pick_needle_positions(100, 10)
        result1 = shake_positions(base1, 0.49)
        seeded_random(99)
        base2 = pick_needle_positions(100, 10)
        result2 = shake_positions(base2, 0.49)
        assert result1 != result2

    @pytest.mark.parametrize("bad_fuzz", [-0.1, 0.5, 1.0])
    def test_fuzz_out_of_range_raises(self, bad_fuzz):
        base = pick_needle_positions(100, 10)
        with pytest.raises(ValueError, match="fuzz_pct must be in"):
            shake_positions(base, bad_fuzz)

    def test_fuzz_with_single_needle(self, seeded_random):
        seeded_random(42)
        base = pick_needle_positions(100, 1)
        result = shake_positions(base, 0.49)
        assert len(result) == 1
        assert 0 <= result[0] < 100

    def test_fuzz_returns_copy_not_mutated_base(self):
        base = pick_needle_positions(100, 10)
        original = base.copy()
        import random
        random.seed(42)
        result = shake_positions(base, 0.49)
        assert base == original
        assert result is not base


class TestDenseNeedlesSkipFuzz:
    @pytest.mark.parametrize("haystack_num,needles_num,expected_skip_fuzz", [
        (10, 25, True),   # > 2x — skip fuzz, use all positions
        (100, 100, False), # 1x — apply fuzz
        (50, 100, False),  # exactly 2x — apply fuzz
    ])
    def test_fuzz_behavior_by_needle_density(self, seeded_random, haystack_num, needles_num, expected_skip_fuzz):
        seeded_random(42)
        _, _, needles = generate_haystack_and_needles(HaystackConfig(haystack_num=haystack_num, needles_num=needles_num, distractors_num=0, key_len=8, val_min=10000, val_max=99999, fuzz=0.49))
        real = [n for n in needles if not n.is_distractor]
        assert len(real) == haystack_num

        if expected_skip_fuzz:
            indices = sorted(n.index for n in real)
            assert indices == list(range(haystack_num))
        else:
            indices = [n.index for n in real]
            diffs = [indices[i+1] - indices[i] for i in range(len(indices)-1)]
            assert max(diffs) - min(diffs) > 1
