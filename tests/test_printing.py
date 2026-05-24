import datetime

from haystack_test import (
    Config,
    GlobalResult,
    ResultRow,
    _print_message,
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
            model=None,
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
            full=False,
            seed=42,
            repeat=1,
            output_filename="results.csv",
            show="all",
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

        _print_summary(config=config, result=result, summaries=[], start_time=start)

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
        result.timed_out_runs = [1]
        start = datetime.datetime.now()

        _print_summary(config=config, result=result, summaries=[], start_time=start)

        captured = capsys.readouterr()
        assert "ERROR" in captured.out
        assert "Connection refused" in captured.out
        assert "Issues" in captured.out
        assert "Timed out runs" in captured.out

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

        _print_summary(config=config, result=result, summaries=[], start_time=start)

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

        _print_summary(config=config, result=result, summaries=[], start_time=start)

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

        _print_summary(config=config, result=result, summaries=[], start_time=start)

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

        _print_summary(config=config, result=result, summaries=[], start_time=start)

        captured = capsys.readouterr()
        assert "Needle accuracy" in captured.out
        assert "Distractor accuracy: N/A (no distractors)" in captured.out
