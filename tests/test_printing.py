import datetime

from haystack_test import (
    Config,
    DebugContext,
    ExperimentSummary,
    GlobalResult,
    Needle,
    ResultRow,
    _print_debug_context,
    _print_message,
    _print_needle_results,
    _print_run_minimal,
    _print_summary,
    _truncate,
)


class TestTruncate:
    def test_short_text_unchanged(self):
        text = "line1\nline2\nline3"
        assert _truncate(text) == text

    def test_long_text_truncated(self):
        lines = [f"line{i}" for i in range(20)]
        text = "\n".join(lines)
        result = _truncate(text, max_lines=2)
        assert "..." in result
        assert "line0" in result
        assert "line19" in result

    def test_custom_prefix(self):
        lines = [f"line{i}" for i in range(10)]
        text = "\n".join(lines)
        result = _truncate(text, max_lines=2, prefix="***")
        assert "***" in result
        assert "..." not in result


class TestPrintMessage:
    def test_prints_role_and_length(self, capsys):
        msg = {"role": "system", "content": "test content"}
        _print_message(msg, is_full=False)
        captured = capsys.readouterr()
        assert "system" in captured.out
        assert "12 chars" in captured.out

    def test_full_vs_truncated(self, capsys):
        long_content = "\n".join(f"line{i}" for i in range(20))
        msg = {"role": "user", "content": long_content}
        _print_message(msg, is_full=False)
        captured = capsys.readouterr()
        assert "..." in captured.out
        assert "line10" not in captured.out
        _print_message(msg, is_full=True)
        captured = capsys.readouterr()
        assert "line10" in captured.out


class TestPrintSummary:
    def _make_config(self):
        return Config(
            endpoint="http://localhost:8080/v1/chat/completions",
            k_quant="Q4_0",
            v_quant="Q8_0",
            note="test",
            key_len=8,
            val_min=10000,
            val_max=99999,
            haystack_num=100,
            needles_num=10,
            distractors_num=2,
            temperature=0.0,
            max_tokens=240000,
            timeout=7200,
            seed=42,
            stop_on_error=False,
            repeat=1,
            output_filename="results.csv",
            verbosity="medium",
            timestamp="2024-01-01T10:00:00",
        )

    def _make_rows(self):
        return [
            ResultRow(
                run=1, haystack_size=100, depth_pct=10.0, correct=1,
                expected=12345, actual="12345",
                prompt_tokens=100, completion_tokens=50, total_tokens=150,
                latency_ms=100.0,
            ),
            ResultRow(
                run=1, haystack_size=100, depth_pct=20.0, correct=0,
                expected=54321, actual="wrong",
                prompt_tokens=100, completion_tokens=50, total_tokens=150,
                latency_ms=100.0,
            ),
        ]

    def _make_summary(self, model_name="test-model", needle_success=50.0,
                      distractor_success=50.0, truncated=False, error=None,
                      tokens_per_sec=500.0, avg_latency_ms=20.0, total_tokens=150,
                      total_latency_ms=200.0, total_completion=100):
        return ExperimentSummary(
            prompt_tokens=100,
            completion_tokens=50,
            total_tokens=total_tokens,
            total_pairs=100,
            needles_num=2,
            distractors_num=2,
            needle_success_pct=needle_success,
            distractor_success_pct=distractor_success,
            tokens_per_sec=tokens_per_sec,
            total_time_sec=0.2,
            ctx_pos_buckets=[0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
            distractor_failures=1,
        )

    def test_prints_full_summary(self, capsys):
        config = self._make_config()
        result = GlobalResult()
        result.all_rows = self._make_rows()
        result.all_stats = [
            {
                "total_tokens": 150,
                "total_latency_ms": 200.0,
                "total_completion": 100,
                "error": None,
                "truncated": False,
                "model_name": "test-model",
                "tokens_per_sec": 500.0,
                "avg_latency_ms": 20.0,
            }
        ]
        start = datetime.datetime(2024, 1, 1, 10, 0, 0)

        _print_summary(config=config, result=result, summaries=[self._make_summary()], start_time=start, model_name="test-model")

        captured = capsys.readouterr()
        assert "SUMMARY" in captured.out
        assert "Configuration" in captured.out
        assert "Haystack size" in captured.out
        assert "100" in captured.out
        assert "Overall results" in captured.out
        assert "Needle accuracy" in captured.out
        assert "1/2" in captured.out
        assert "Per-run results" in captured.out
        assert "test-model" in captured.out
        assert "Elapsed" in captured.out

    def test_prints_errors_and_truncation(self, capsys):
        config = self._make_config()
        result = GlobalResult()
        result.all_rows = []
        result.all_stats = [{
            "total_tokens": 0, "total_latency_ms": 0, "total_completion": 0,
            "error": "Connection refused", "truncated": False,
            "model_name": "error", "tokens_per_sec": 0, "avg_latency_ms": 0,
        }]
        result.failed_runs = [1]
        start = datetime.datetime.now()

        _print_summary(config=config, result=result, summaries=[self._make_summary()], start_time=start, model_name="test-model")

        captured = capsys.readouterr()
        assert "ERROR" in captured.out
        assert "Connection refused" in captured.out
        assert "Issues" in captured.out
        assert "Failed runs" in captured.out

    def test_prints_truncated_runs(self, capsys):
        config = self._make_config()
        result = GlobalResult()
        result.all_rows = []
        result.all_stats = [{
            "total_tokens": 100, "total_latency_ms": 5000, "total_completion": 240000,
            "error": None, "truncated": True,
            "model_name": "test-model", "tokens_per_sec": 48000.0, "avg_latency_ms": 500.0,
        }]
        result.truncated_runs = [1]
        start = datetime.datetime.now()

        _print_summary(config=config, result=result, summaries=[self._make_summary()], start_time=start, model_name="test-model")

        captured = capsys.readouterr()
        assert "TRUNCATED" in captured.out
        assert "Truncated runs" in captured.out

    def test_prints_latency_in_minutes(self, capsys):
        config = self._make_config()
        result = GlobalResult()
        result.all_rows = self._make_rows()
        result.all_stats = [{
            "total_tokens": 100000, "total_latency_ms": 180000, "total_completion": 50000,
            "error": None, "truncated": False,
            "model_name": "test-model", "tokens_per_sec": 277.8, "avg_latency_ms": 10.0,
        }]
        start = datetime.datetime.now()

        _print_summary(config=config, result=result, summaries=[self._make_summary()], start_time=start, model_name="test-model")

        captured = capsys.readouterr()
        assert "min" in captured.out

    def test_prints_needle_and_distractor_accuracy(self, capsys):
        config = self._make_config()
        result = GlobalResult()
        result.all_rows = [
            ResultRow(
                run=0, haystack_size=100, depth_pct=10.0, correct=1,
                expected=12345, actual="12345",
                prompt_tokens=100, completion_tokens=50, total_tokens=150,
                latency_ms=100.0,
            ),
            ResultRow(
                run=0, haystack_size=100, depth_pct=20.0, correct=0,
                expected=54321, actual="wrong",
                prompt_tokens=100, completion_tokens=50, total_tokens=150,
                latency_ms=100.0,
            ),
            ResultRow(
                run=0, haystack_size=100, depth_pct=30.0, correct=0,
                expected="", actual="HALLUCINATED",
                prompt_tokens=100, completion_tokens=50, total_tokens=150,
                latency_ms=100.0,
            ),
            ResultRow(
                run=0, haystack_size=100, depth_pct=40.0, correct=1,
                expected="", actual="",
                prompt_tokens=100, completion_tokens=50, total_tokens=150,
                latency_ms=100.0,
            ),
        ]
        result.all_stats = [{
            "total_tokens": 150, "total_latency_ms": 200.0, "total_completion": 100,
            "error": None, "truncated": False,
            "model_name": "test-model", "tokens_per_sec": 500.0, "avg_latency_ms": 20.0,
        }]
        start = datetime.datetime.now()

        _print_summary(config=config, result=result, summaries=[self._make_summary()], start_time=start, model_name="test-model")

        captured = capsys.readouterr()
        assert "Needle accuracy" in captured.out
        assert "Distractor accuracy" in captured.out
        assert "1/2" in captured.out  # 1 of 2 needles correct
        assert "1/2" in captured.out  # 1 of 2 distractors correct

    def test_prints_no_distractors_message(self, capsys):
        config = self._make_config()
        result = GlobalResult()
        result.all_rows = [
            ResultRow(
                run=1, haystack_size=100, depth_pct=10.0, correct=1,
                expected=12345, actual="12345",
                prompt_tokens=100, completion_tokens=50, total_tokens=150,
                latency_ms=100.0,
            ),
        ]
        result.all_stats = [{
            "total_tokens": 150, "total_latency_ms": 200.0, "total_completion": 100,
            "error": None, "truncated": False,
            "model_name": "test-model", "tokens_per_sec": 500.0, "avg_latency_ms": 20.0,
        }]
        start = datetime.datetime.now()

        _print_summary(config=config, result=result, summaries=[self._make_summary()], start_time=start, model_name="test-model")

        captured = capsys.readouterr()
        assert "Needle accuracy" in captured.out
        assert "Distractor accuracy: N/A (no distractors)" in captured.out


class TestPrintNeedleResults:
    def _make_needles(self):
        return [
            Needle(index=10, key="AAAA", expected=12345, is_distractor=False),
            Needle(index=20, key="BBBB", expected=54321, is_distractor=False),
            Needle(index=30, key="CCCC", expected=None, is_distractor=True),
        ]

    def _make_debug(self):
        return DebugContext(model_name="test-model")

    def test_prints_all_for_full_verbosity(self, capsys):
        needles = self._make_needles()
        parsed = {"AAAA": "12345", "BBBB": "wrong", "CCCC": "HALLUCINATED"}
        pairs = [("X", 1)] * 100

        _print_needle_results(
            1, needles, pairs, parsed,
            verbosity="full",
            debug=self._make_debug(),
        )

        captured = capsys.readouterr()
        assert "AAAA" in captured.out
        assert "BBBB" in captured.out
        assert "CCCC" in captured.out
        assert "[OK] AAAA" in captured.out
        assert "[FAIL] BBBB" in captured.out
        assert "[FAIL] CCCC" in captured.out

    def test_prints_summary_only_for_medium_verbosity(self, capsys):
        needles = self._make_needles()
        parsed = {"AAAA": "12345", "BBBB": "wrong", "CCCC": "HALLUCINATED"}
        pairs = [("X", 1)] * 100

        _print_needle_results(
            1, needles, pairs, parsed,
            verbosity="medium",
            debug=self._make_debug(),
        )

        captured = capsys.readouterr()
        assert "Model: test-model" in captured.out
        assert "Needles accuracy:" in captured.out
        assert "Distractors accuracy:" in captured.out
        assert "FAIL" in captured.out
        assert "AAAA" not in captured.out
        assert "BBBB" not in captured.out

    def test_prints_ok_status_when_all_correct(self, capsys):
        needles = self._make_needles()
        parsed = {"AAAA": "12345", "BBBB": "54321"}
        pairs = [("X", 1)] * 100

        _print_needle_results(
            1, needles, pairs, parsed,
            verbosity="medium",
            debug=self._make_debug(),
        )

        captured = capsys.readouterr()
        assert "Model: test-model" in captured.out
        assert "Needles accuracy:" in captured.out
        assert "Distractors accuracy:" in captured.out
        assert "OK" in captured.out
        assert "FAIL" not in captured.out


class TestPrintDebugContext:
    def test_prints_messages_and_response(self, capsys):
        debug = DebugContext(
            messages=[
                {"role": "system", "content": "system message"},
                {"role": "user", "content": "user prompt"},
            ],
            raw_response={
                "model": "test-model",
                "choices": [{"message": {"content": "AAAA = 12345"}}],
                "usage": {"prompt_tokens": 100, "completion_tokens": 10, "total_tokens": 110},
            },
            response_text="AAAA = 12345",
            model_name="test-model",
        )

        _print_debug_context(debug)

        captured = capsys.readouterr()
        assert "ALL MESSAGES" in captured.out
        assert "system message" in captured.out
        assert "user prompt" in captured.out
        assert "RAW RESPONSE" in captured.out
        assert "test-model" in captured.out
        assert "CONTENT FIELD" in captured.out

    def test_handles_empty_response(self, capsys):
        debug = DebugContext(
            messages=None,
            raw_response=None,
            response_text="",
            model_name="test-model",
        )

        _print_debug_context(debug)

        captured = capsys.readouterr()
        assert "(no response received)" in captured.out


class TestPrintRunMinimal:
    def test_prints_one_liner_with_accuracy(self, capsys):
        rows = [
            ResultRow(run=1, haystack_size=100, depth_pct=10.0, correct=1,
                      expected=12345, actual="12345"),
            ResultRow(run=1, haystack_size=100, depth_pct=20.0, correct=0,
                      expected=54321, actual="wrong"),
            ResultRow(run=1, haystack_size=100, depth_pct=30.0, correct=1,
                      expected="", actual=""),
        ]
        stats = {
            "total_tokens": 150, "total_latency_ms": 200.0,
            "total_completion": 100, "error": None, "truncated": False,
            "model_name": "test", "tokens_per_sec": 500.0, "avg_latency_ms": 20.0,
        }
        summary = ExperimentSummary(
            prompt_tokens=100, completion_tokens=50, total_tokens=150,
            total_pairs=100, needles_num=2, distractors_num=1,
            needle_success_pct=50.0, distractor_success_pct=100.0,
            tokens_per_sec=500.0, total_time_sec=0.2,
            ctx_pos_buckets=[0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
            distractor_failures=0,
        )

        _print_run_minimal(1, 3, summary, stats, rows, "2024-01-01 10:00:00")

        captured = capsys.readouterr()
        assert "2024-01-01 10:00:00" in captured.out
        assert "Run 1/3:" in captured.out
        assert "50.0%" in captured.out
        assert "100.0%" in captured.out

    def test_prints_error_status(self, capsys):
        rows: list[ResultRow] = []
        stats = {
            "total_tokens": 0, "total_latency_ms": 0,
            "total_completion": 0, "error": "Connection refused",
            "truncated": False, "model_name": "test",
            "tokens_per_sec": 0, "avg_latency_ms": 0,
        }
        summary = ExperimentSummary(
            prompt_tokens=0, completion_tokens=0, total_tokens=0,
            total_pairs=100, needles_num=0, distractors_num=0,
            needle_success_pct=0.0, distractor_success_pct=0.0,
            tokens_per_sec=0, total_time_sec=0.0,
            ctx_pos_buckets=[0] * 10, distractor_failures=0,
        )

        _print_run_minimal(2, 3, summary, stats, rows, "2024-01-01 10:05:00")

        captured = capsys.readouterr()
        assert "ERROR (Connection refused)" in captured.out

    def test_prints_truncated_status(self, capsys):
        rows = [
            ResultRow(run=2, haystack_size=100, depth_pct=10.0, correct=1,
                      expected=12345, actual="12345"),
            ResultRow(run=2, haystack_size=100, depth_pct=20.0, correct=0,
                      expected=54321, actual="wrong"),
        ]
        stats = {
            "total_tokens": 150, "total_latency_ms": 200.0,
            "total_completion": 100, "error": None, "truncated": True,
            "model_name": "test", "tokens_per_sec": 500.0, "avg_latency_ms": 20.0,
        }
        summary = ExperimentSummary(
            prompt_tokens=100, completion_tokens=50, total_tokens=150,
            total_pairs=100, needles_num=2, distractors_num=0,
            needle_success_pct=50.0, distractor_success_pct=0.0,
            tokens_per_sec=500.0, total_time_sec=0.2,
            ctx_pos_buckets=[0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
            distractor_failures=0,
        )

        _print_run_minimal(2, 3, summary, stats, rows, "2024-01-01 10:05:00")

        captured = capsys.readouterr()
        assert "TRUNCATED" in captured.out
        assert "50.0%" in captured.out
