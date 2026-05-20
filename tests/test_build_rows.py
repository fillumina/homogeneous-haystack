from haystack_test import Needle, ResultRow, _build_rows


class TestBuildRows:
    def _make_needles(self):
        return [
            Needle(index=0, key="ABCD1234", expected=1234, is_distractor=False),
            Needle(index=50, key="EFGH5678", expected=5678, is_distractor=False),
        ]

    def _make_pairs(self):
        return [("ABCD1234", 1234), ("EFGH5678", 5678)]

    def _make_usage(self):
        return {
            "prompt_tokens": 100,
            "completion_tokens": 50,
            "total_tokens": 150,
        }

    def test_returns_result_rows_with_all_fields(self):
        needles = self._make_needles()
        pairs = self._make_pairs()
        parsed = {"ABCD1234": "1234", "EFGH5678": "5678"}
        usage = self._make_usage()

        result = _build_rows(42, needles, pairs, parsed, usage, latency_ms=100.0)

        assert len(result) == 2
        assert all(isinstance(r, ResultRow) for r in result)
        assert result[0].run == 42
        assert result[0].haystack_size == 2
        assert result[0].prompt_tokens == 100
        assert result[0].completion_tokens == 50
        assert result[0].total_tokens == 150
        assert result[0].latency_ms == 100.0

    def test_correctness_and_values(self):
        needles = self._make_needles()
        pairs = self._make_pairs()
        parsed = {"ABCD1234": "1234", "EFGH5678": "9999"}
        usage = self._make_usage()

        result = _build_rows(1, needles, pairs, parsed, usage)

        assert result[0].correct == 1  # correct match
        assert result[1].correct == 0  # wrong value
        assert result[0].expected == 1234
        assert result[1].expected == 5678
        assert result[0].actual == "1234"
        assert result[1].actual == "9999"

    def test_missing_values_and_none_usage(self):
        needles = self._make_needles()
        pairs = self._make_pairs()
        parsed = {"ABCD1234": "1234"}  # EFGH5678 missing
        usage = self._make_usage()

        result = _build_rows(1, needles, pairs, parsed, usage, latency_ms=None)

        assert result[0].actual == "1234"
        assert result[1].actual == ""
        assert result[0].latency_ms is None

        result_none = _build_rows(1, needles, pairs, parsed, usage=None)
        assert result_none[0].prompt_tokens is None
        assert result_none[0].completion_tokens is None
        assert result_none[0].total_tokens is None

    def test_distractor_needles(self):
        needles = [
            Needle(index=0, key="ABCD1234", expected=1234, is_distractor=False),
            Needle(index=1, key="FAKEKEY", expected=None, is_distractor=True),
        ]
        pairs = [("ABCD1234", 1234)]
        parsed = {"ABCD1234": "1234", "FAKEKEY": "9999"}

        result = _build_rows(1, needles, pairs, parsed, self._make_usage())

        assert result[0].correct == 1  # real needle correct
        assert result[1].correct == 0  # distractor hallucinated

    def test_depth_calculation(self):
        needles = [Needle(index=7, key="KEY", expected=1234, is_distractor=False)]
        pairs = [("K" + str(i), i) for i in range(10)]
        parsed = {"KEY": "1234"}

        result = _build_rows(1, needles, pairs, parsed, self._make_usage())
        assert result[0].depth_pct == 77.78  # 7/(10-1)*100

        # Single pair edge case
        single_needles = [Needle(index=0, key="K", expected=1, is_distractor=False)]
        single_pairs = [("K", 1)]
        result = _build_rows(1, single_needles, single_pairs, parsed, self._make_usage())
        assert result[0].depth_pct == 0.0
