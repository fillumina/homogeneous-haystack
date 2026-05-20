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

    def test_returns_list_of_result_rows(self):
        needles = self._make_needles()
        pairs = self._make_pairs()
        parsed = {"ABCD1234": "1234", "EFGH5678": "5678"}
        usage = self._make_usage()

        result = _build_rows(1, needles, pairs, parsed, usage, latency_ms=100.0)

        assert isinstance(result, list)
        assert len(result) == 2
        assert all(isinstance(r, ResultRow) for r in result)

    def test_result_row_has_correct_run(self):
        needles = self._make_needles()
        pairs = self._make_pairs()
        parsed = {"ABCD1234": "1234"}
        usage = self._make_usage()

        result = _build_rows(42, needles, pairs, parsed, usage, latency_ms=100.0)

        assert result[0].run == 42
        assert result[1].run == 42

    def test_result_row_has_haystack_size(self):
        needles = self._make_needles()
        pairs = self._make_pairs()
        parsed = {"ABCD1234": "1234"}
        usage = self._make_usage()

        result = _build_rows(1, needles, pairs, parsed, usage, latency_ms=100.0)

        assert result[0].haystack_size == 2
        assert result[1].haystack_size == 2

    def test_result_row_has_depth_pct(self):
        needles = self._make_needles()
        pairs = self._make_pairs()
        parsed = {"ABCD1234": "1234"}
        usage = self._make_usage()

        result = _build_rows(1, needles, pairs, parsed, usage, latency_ms=100.0)

        # First needle at index 0: depth = 0/(2-1)*100 = 0.0
        assert result[0].depth_pct == 0.0
        # Second needle at index 50: depth = 50/(2-1)*100 = 5000.0
        assert result[1].depth_pct == 5000.0

    def test_depth_pct_with_single_pair(self):
        needles = [Needle(index=0, key="ABCD1234", expected=1234, is_distractor=False)]
        pairs = [("ABCD1234", 1234)]
        parsed = {"ABCD1234": "1234"}
        usage = self._make_usage()

        result = _build_rows(1, needles, pairs, parsed, usage, latency_ms=100.0)

        # When len(pairs) == 1, depth_pct should be 0.0
        assert result[0].depth_pct == 0.0

    def test_result_row_has_correct_field(self):
        needles = self._make_needles()
        pairs = self._make_pairs()
        parsed = {"ABCD1234": "1234", "EFGH5678": "9999"}
        usage = self._make_usage()

        result = _build_rows(1, needles, pairs, parsed, usage, latency_ms=100.0)

        assert result[0].correct == 1  # ABCD1234 = 1234 (correct)
        assert result[1].correct == 0  # EFGH5678 expected 5678, got 9999

    def test_result_row_has_expected_field(self):
        needles = self._make_needles()
        pairs = self._make_pairs()
        parsed = {"ABCD1234": "1234"}
        usage = self._make_usage()

        result = _build_rows(1, needles, pairs, parsed, usage, latency_ms=100.0)

        assert result[0].expected == 1234
        assert result[1].expected == 5678

    def test_result_row_has_actual_field(self):
        needles = self._make_needles()
        pairs = self._make_pairs()
        parsed = {"ABCD1234": "1234"}
        usage = self._make_usage()

        result = _build_rows(1, needles, pairs, parsed, usage, latency_ms=100.0)

        assert result[0].actual == "1234"
        assert result[1].actual == ""  # not in parsed

    def test_result_row_has_prompt_tokens(self):
        needles = self._make_needles()
        pairs = self._make_pairs()
        parsed = {"ABCD1234": "1234"}
        usage = self._make_usage()

        result = _build_rows(1, needles, pairs, parsed, usage, latency_ms=100.0)

        assert result[0].prompt_tokens == 100

    def test_result_row_has_completion_tokens(self):
        needles = self._make_needles()
        pairs = self._make_pairs()
        parsed = {"ABCD1234": "1234"}
        usage = self._make_usage()

        result = _build_rows(1, needles, pairs, parsed, usage, latency_ms=100.0)

        assert result[0].completion_tokens == 50

    def test_result_row_has_total_tokens(self):
        needles = self._make_needles()
        pairs = self._make_pairs()
        parsed = {"ABCD1234": "1234"}
        usage = self._make_usage()

        result = _build_rows(1, needles, pairs, parsed, usage, latency_ms=100.0)

        assert result[0].total_tokens == 150

    def test_result_row_has_latency_ms(self):
        needles = self._make_needles()
        pairs = self._make_pairs()
        parsed = {"ABCD1234": "1234"}
        usage = self._make_usage()

        result = _build_rows(1, needles, pairs, parsed, usage, latency_ms=100.0)

        assert result[0].latency_ms == 100.0

    def test_usage_none_sets_tokens_to_none(self):
        needles = self._make_needles()
        pairs = self._make_pairs()
        parsed = {"ABCD1234": "1234"}

        result = _build_rows(1, needles, pairs, parsed, usage=None, latency_ms=100.0)

        assert result[0].prompt_tokens is None
        assert result[0].completion_tokens is None
        assert result[0].total_tokens is None

    def test_usage_missing_keys(self):
        needles = self._make_needles()
        pairs = self._make_pairs()
        parsed = {"ABCD1234": "1234"}
        usage = {"prompt_tokens": 100}  # missing completion_tokens and total_tokens

        result = _build_rows(1, needles, pairs, parsed, usage, latency_ms=100.0)

        assert result[0].prompt_tokens == 100
        assert result[0].completion_tokens is None
        assert result[0].total_tokens is None

    def test_no_latency_sets_none(self):
        needles = self._make_needles()
        pairs = self._make_pairs()
        parsed = {"ABCD1234": "1234"}
        usage = self._make_usage()

        result = _build_rows(1, needles, pairs, parsed, usage, latency_ms=None)

        assert result[0].latency_ms is None

    def test_single_needle(self):
        needles = [Needle(index=0, key="ABCD1234", expected=1234, is_distractor=False)]
        pairs = [("ABCD1234", 1234)]
        parsed = {"ABCD1234": "1234"}
        usage = self._make_usage()

        result = _build_rows(1, needles, pairs, parsed, usage, latency_ms=100.0)

        assert len(result) == 1
        assert result[0].correct == 1

    def test_distractor_needle_in_result(self):
        needles = [
            Needle(index=0, key="ABCD1234", expected=1234, is_distractor=False),
            Needle(index=1, key="FAKEKEY", expected=None, is_distractor=True),
        ]
        pairs = [("ABCD1234", 1234)]
        parsed = {"ABCD1234": "1234"}
        usage = self._make_usage()

        result = _build_rows(1, needles, pairs, parsed, usage, latency_ms=100.0)

        assert len(result) == 2
        assert result[0].correct == 1
        # Distractor: expected None, no response, so correct = 1
        assert result[1].correct == 1

    def test_distractor_hallucination_marked_wrong(self):
        needles = [
            Needle(index=0, key="ABCD1234", expected=1234, is_distractor=False),
            Needle(index=1, key="FAKEKEY", expected=None, is_distractor=True),
        ]
        pairs = [("ABCD1234", 1234)]
        parsed = {"ABCD1234": "1234", "FAKEKEY": "9999"}
        usage = self._make_usage()

        result = _build_rows(1, needles, pairs, parsed, usage, latency_ms=100.0)

        assert result[1].correct == 0
        assert result[1].actual == "9999"

    def test_all_fields_present_in_result_row(self):
        needles = self._make_needles()
        pairs = self._make_pairs()
        parsed = {"ABCD1234": "1234"}
        usage = self._make_usage()

        result = _build_rows(1, needles, pairs, parsed, usage, latency_ms=100.0)

        row = result[0]
        assert hasattr(row, "run")
        assert hasattr(row, "haystack_size")
        assert hasattr(row, "depth_pct")
        assert hasattr(row, "correct")
        assert hasattr(row, "expected")
        assert hasattr(row, "actual")
        assert hasattr(row, "prompt_tokens")
        assert hasattr(row, "completion_tokens")
        assert hasattr(row, "total_tokens")
        assert hasattr(row, "latency_ms")

    def test_depth_pct_rounding(self):
        needles = [Needle(index=7, key="ABCD1234", expected=1234, is_distractor=False)]
        pairs = [("K" + str(i), i) for i in range(10)]
        parsed = {"ABCD1234": "1234"}
        usage = self._make_usage()

        result = _build_rows(1, needles, pairs, parsed, usage, latency_ms=100.0)

        # depth = 7/(10-1)*100 = 77.777... -> rounded to 77.78
        assert result[0].depth_pct == 77.78
