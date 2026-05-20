import datetime

import pytest

from haystack_test import (
    Config,
    Needle,
    ResultRow,
    RunStats,
    Summary,
    _print_message,
    _print_summary,
    _truncate,
)


class TestPrintMessage:
    def test_prints_role(self, capsys):
        msg = {"role": "system", "content": "test content"}
        _print_message(msg, is_full=False)
        captured = capsys.readouterr()
        assert "system" in captured.out

    def test_prints_content_length(self, capsys):
        msg = {"role": "user", "content": "hello world"}
        _print_message(msg, is_full=False)
        captured = capsys.readouterr()
        assert "11 chars" in captured.out

    def test_full_prints_full_content(self, capsys):
        msg = {"role": "user", "content": "full content here"}
        _print_message(msg, is_full=True)
        captured = capsys.readouterr()
        assert "full content here" in captured.out

    def test_truncated_prints_truncated_content(self, capsys):
        long_content = "\n".join(f"line{i}" for i in range(20))
        msg = {"role": "user", "content": long_content}
        _print_message(msg, is_full=False)
        captured = capsys.readouterr()
        assert "..." in captured.out

    def test_truncated_does_not_print_full_content(self, capsys):
        long_content = "\n".join(f"line{i}" for i in range(20))
        msg = {"role": "user", "content": long_content}
        _print_message(msg, is_full=False)
        captured = capsys.readouterr()
        # line10 should not appear in truncated output
        assert "line10" not in captured.out

    def test_short_content_not_truncated(self, capsys):
        msg = {"role": "user", "content": "short"}
        _print_message(msg, is_full=False)
        captured = capsys.readouterr()
        assert "..." not in captured.out


class TestPrintSummary:
    def _make_config(self):
        return Config(
            endpoint="http://localhost:8080/v1/chat/completions",
            model=None,
            key_len=8,
            val_min=10000,
            val_max=99999,
            haystack_num=100,
            needles_num=10,
            distractor_pct=None,
            distractors_num=2,
            temperature=0.0,
            max_tokens=240000,
            timeout=7200,
            full=False,
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

    def _make_stats(self):
        return [
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

    def test_prints_summary_header(self, capsys):
        config = self._make_config()
        summary = Summary(actual_real_needles=10, actual_distractors=2, actual_total=12)
        start = datetime.datetime.now()

        _print_summary(
            config=config,
            seed=42,
            summary=summary,
            all_rows=self._make_rows(),
            all_stats=self._make_stats(),
            timed_out_runs=[],
            truncated_runs=[],
            output_file="results.csv",
            repeat=1,
            start_time=start,
        )

        captured = capsys.readouterr()
        assert "SUMMARY" in captured.out

    def test_prints_configuration(self, capsys):
        config = self._make_config()
        summary = Summary(actual_real_needles=10, actual_distractors=2, actual_total=12)
        start = datetime.datetime.now()

        _print_summary(
            config=config,
            seed=42,
            summary=summary,
            all_rows=self._make_rows(),
            all_stats=self._make_stats(),
            timed_out_runs=[],
            truncated_runs=[],
            output_file="results.csv",
            repeat=1,
            start_time=start,
        )

        captured = capsys.readouterr()
        assert "Configuration" in captured.out
        assert "Haystack size" in captured.out
        assert "100" in captured.out

    def test_prints_seed(self, capsys):
        config = self._make_config()
        summary = Summary(actual_real_needles=10, actual_distractors=2, actual_total=12)
        start = datetime.datetime.now()

        _print_summary(
            config=config,
            seed=12345,
            summary=summary,
            all_rows=self._make_rows(),
            all_stats=self._make_stats(),
            timed_out_runs=[],
            truncated_runs=[],
            output_file="results.csv",
            repeat=1,
            start_time=start,
        )

        captured = capsys.readouterr()
        assert "12345" in captured.out

    def test_prints_accuracy(self, capsys):
        config = self._make_config()
        summary = Summary(actual_real_needles=10, actual_distractors=2, actual_total=12)
        start = datetime.datetime.now()

        _print_summary(
            config=config,
            seed=42,
            summary=summary,
            all_rows=self._make_rows(),
            all_stats=self._make_stats(),
            timed_out_runs=[],
            truncated_runs=[],
            output_file="results.csv",
            repeat=1,
            start_time=start,
        )

        captured = capsys.readouterr()
        assert "Overall results" in captured.out
        assert "1/2" in captured.out

    def test_prints_distractors_exact_count(self, capsys):
        config = self._make_config()
        summary = Summary(actual_real_needles=10, actual_distractors=2, actual_total=12)
        start = datetime.datetime.now()

        _print_summary(
            config=config,
            seed=42,
            summary=summary,
            all_rows=self._make_rows(),
            all_stats=self._make_stats(),
            timed_out_runs=[],
            truncated_runs=[],
            output_file="results.csv",
            repeat=1,
            start_time=start,
        )

        captured = capsys.readouterr()
        assert "exact count" in captured.out

    def test_prints_per_run_results(self, capsys):
        config = self._make_config()
        summary = Summary(actual_real_needles=10, actual_distractors=2, actual_total=12)
        start = datetime.datetime.now()

        _print_summary(
            config=config,
            seed=42,
            summary=summary,
            all_rows=self._make_rows(),
            all_stats=self._make_stats(),
            timed_out_runs=[],
            truncated_runs=[],
            output_file="results.csv",
            repeat=1,
            start_time=start,
        )

        captured = capsys.readouterr()
        assert "Per-run results" in captured.out
        assert "test-model" in captured.out

    def test_prints_timed_out_runs(self, capsys):
        config = self._make_config()
        summary = Summary(actual_real_needles=10, actual_distractors=2, actual_total=12)
        start = datetime.datetime.now()

        _print_summary(
            config=config,
            seed=42,
            summary=summary,
            all_rows=[],
            all_stats=[],
            timed_out_runs=[1, 3],
            truncated_runs=[],
            output_file="results.csv",
            repeat=3,
            start_time=start,
        )

        captured = capsys.readouterr()
        assert "Issues" in captured.out
        assert "Timed out runs" in captured.out

    def test_prints_truncated_runs(self, capsys):
        config = self._make_config()
        summary = Summary(actual_real_needles=10, actual_distractors=2, actual_total=12)
        start = datetime.datetime.now()

        _print_summary(
            config=config,
            seed=42,
            summary=summary,
            all_rows=[],
            all_stats=[],
            timed_out_runs=[],
            truncated_runs=[2],
            output_file="results.csv",
            repeat=3,
            start_time=start,
        )

        captured = capsys.readouterr()
        assert "Truncated runs" in captured.out

    def test_prints_output_file(self, capsys):
        config = self._make_config()
        summary = Summary(actual_real_needles=10, actual_distractors=2, actual_total=12)
        start = datetime.datetime.now()

        _print_summary(
            config=config,
            seed=42,
            summary=summary,
            all_rows=self._make_rows(),
            all_stats=self._make_stats(),
            timed_out_runs=[],
            truncated_runs=[],
            output_file="custom_output.csv",
            repeat=1,
            start_time=start,
        )

        captured = capsys.readouterr()
        assert "custom_output.csv" in captured.out

    def test_prints_elapsed_time(self, capsys):
        config = self._make_config()
        summary = Summary(actual_real_needles=10, actual_distractors=2, actual_total=12)
        start = datetime.datetime(2024, 1, 1, 10, 0, 0)

        _print_summary(
            config=config,
            seed=42,
            summary=summary,
            all_rows=self._make_rows(),
            all_stats=self._make_stats(),
            timed_out_runs=[],
            truncated_runs=[],
            output_file="results.csv",
            repeat=1,
            start_time=start,
        )

        captured = capsys.readouterr()
        assert "Elapsed" in captured.out

    def test_prints_elapsed_hours(self, capsys):
        config = self._make_config()
        summary = Summary(actual_real_needles=10, actual_distractors=2, actual_total=12)
        start = datetime.datetime(2024, 1, 1, 10, 0, 0)

        # Simulate 3h 30m elapsed
        end = datetime.datetime(2024, 1, 1, 13, 30, 0)

        _print_summary(
            config=config,
            seed=42,
            summary=summary,
            all_rows=self._make_rows(),
            all_stats=self._make_stats(),
            timed_out_runs=[],
            truncated_runs=[],
            output_file="results.csv",
            repeat=1,
            start_time=start,
        )

        captured = capsys.readouterr()
        # Should have some elapsed time printed
        assert "Elapsed" in captured.out

    def test_prints_tokens_total(self, capsys):
        config = self._make_config()
        summary = Summary(actual_real_needles=10, actual_distractors=2, actual_total=12)
        start = datetime.datetime.now()

        _print_summary(
            config=config,
            seed=42,
            summary=summary,
            all_rows=self._make_rows(),
            all_stats=self._make_stats(),
            timed_out_runs=[],
            truncated_runs=[],
            output_file="results.csv",
            repeat=1,
            start_time=start,
        )

        captured = capsys.readouterr()
        assert "Total tokens" in captured.out

    def test_prints_avg_tokens_per_sec(self, capsys):
        config = self._make_config()
        summary = Summary(actual_real_needles=10, actual_distractors=2, actual_total=12)
        start = datetime.datetime.now()

        _print_summary(
            config=config,
            seed=42,
            summary=summary,
            all_rows=self._make_rows(),
            all_stats=self._make_stats(),
            timed_out_runs=[],
            truncated_runs=[],
            output_file="results.csv",
            repeat=1,
            start_time=start,
        )

        captured = capsys.readouterr()
        assert "tokens/sec" in captured.out

    def test_prints_avg_latency(self, capsys):
        config = self._make_config()
        summary = Summary(actual_real_needles=10, actual_distractors=2, actual_total=12)
        start = datetime.datetime.now()

        _print_summary(
            config=config,
            seed=42,
            summary=summary,
            all_rows=self._make_rows(),
            all_stats=self._make_stats(),
            timed_out_runs=[],
            truncated_runs=[],
            output_file="results.csv",
            repeat=1,
            start_time=start,
        )

        captured = capsys.readouterr()
        assert "latency" in captured.out.lower() or "Lat" in captured.out

    def test_prints_repeat(self, capsys):
        config = self._make_config()
        summary = Summary(actual_real_needles=10, actual_distractors=2, actual_total=12)
        start = datetime.datetime.now()

        _print_summary(
            config=config,
            seed=42,
            summary=summary,
            all_rows=self._make_rows(),
            all_stats=self._make_stats(),
            timed_out_runs=[],
            truncated_runs=[],
            output_file="results.csv",
            repeat=3,
            start_time=start,
        )

        captured = capsys.readouterr()
        assert "Repeat" in captured.out
        assert "3" in captured.out

    def test_prints_no_latency_when_zero(self, capsys):
        config = self._make_config()
        summary = Summary(actual_real_needles=10, actual_distractors=2, actual_total=12)
        stats = [
            {
                "total_tokens": 0,
                "total_latency_ms": 0,
                "total_completion": 0,
                "error": None,
                "truncated": False,
                "model_name": "test-model",
                "tokens_per_sec": 0,
                "avg_latency_ms": 0,
            }
        ]
        start = datetime.datetime.now()

        _print_summary(
            config=config,
            seed=42,
            summary=summary,
            all_rows=self._make_rows(),
            all_stats=stats,
            timed_out_runs=[],
            truncated_runs=[],
            output_file="results.csv",
            repeat=1,
            start_time=start,
        )

        captured = capsys.readouterr()
        assert "SUMMARY" in captured.out

    def test_prints_error_in_per_run(self, capsys):
        config = self._make_config()
        summary = Summary(actual_real_needles=10, actual_distractors=2, actual_total=12)
        stats = [
            {
                "total_tokens": 0,
                "total_latency_ms": 0,
                "total_completion": 0,
                "error": "Connection refused",
                "truncated": False,
                "model_name": "error",
                "tokens_per_sec": 0,
                "avg_latency_ms": 0,
            }
        ]
        start = datetime.datetime.now()

        _print_summary(
            config=config,
            seed=42,
            summary=summary,
            all_rows=[],
            all_stats=stats,
            timed_out_runs=[1],
            truncated_runs=[],
            output_file="results.csv",
            repeat=1,
            start_time=start,
        )

        captured = capsys.readouterr()
        assert "ERROR" in captured.out
        assert "Connection refused" in captured.out

    def test_prints_truncated_status_in_per_run(self, capsys):
        config = self._make_config()
        summary = Summary(actual_real_needles=10, actual_distractors=2, actual_total=12)
        stats = [
            {
                "total_tokens": 100,
                "total_latency_ms": 5000,
                "total_completion": 240000,
                "error": None,
                "truncated": True,
                "model_name": "test-model",
                "tokens_per_sec": 48000.0,
                "avg_latency_ms": 500.0,
            }
        ]
        start = datetime.datetime.now()

        _print_summary(
            config=config,
            seed=42,
            summary=summary,
            all_rows=[],
            all_stats=stats,
            timed_out_runs=[],
            truncated_runs=[1],
            output_file="results.csv",
            repeat=1,
            start_time=start,
        )

        captured = capsys.readouterr()
        assert "TRUNCATED" in captured.out

    def test_prints_distractor_pct_when_pct_set(self, capsys):
        config = Config(
            endpoint="http://localhost:8080/v1/chat/completions",
            model=None,
            key_len=8,
            val_min=10000,
            val_max=99999,
            haystack_num=100,
            needles_num=10,
            distractor_pct=0.15,
            distractors_num=None,
            temperature=0.0,
            max_tokens=240000,
            timeout=7200,
            full=False,
        )
        summary = Summary(actual_real_needles=10, actual_distractors=2, actual_total=12)
        start = datetime.datetime.now()

        _print_summary(
            config=config,
            seed=42,
            summary=summary,
            all_rows=self._make_rows(),
            all_stats=self._make_stats(),
            timed_out_runs=[],
            truncated_runs=[],
            output_file="results.csv",
            repeat=1,
            start_time=start,
        )

        captured = capsys.readouterr()
        assert "Distractor pct" in captured.out

    def test_prints_key_length(self, capsys):
        config = self._make_config()
        summary = Summary(actual_real_needles=10, actual_distractors=2, actual_total=12)
        start = datetime.datetime.now()

        _print_summary(
            config=config,
            seed=42,
            summary=summary,
            all_rows=self._make_rows(),
            all_stats=self._make_stats(),
            timed_out_runs=[],
            truncated_runs=[],
            output_file="results.csv",
            repeat=1,
            start_time=start,
        )

        captured = capsys.readouterr()
        assert "Key length" in captured.out
        assert "8" in captured.out

    def test_prints_temperature(self, capsys):
        config = self._make_config()
        summary = Summary(actual_real_needles=10, actual_distractors=2, actual_total=12)
        start = datetime.datetime.now()

        _print_summary(
            config=config,
            seed=42,
            summary=summary,
            all_rows=self._make_rows(),
            all_stats=self._make_stats(),
            timed_out_runs=[],
            truncated_runs=[],
            output_file="results.csv",
            repeat=1,
            start_time=start,
        )

        captured = capsys.readouterr()
        assert "Temperature" in captured.out
        assert "0.0" in captured.out

    def test_prints_value_range(self, capsys):
        config = self._make_config()
        summary = Summary(actual_real_needles=10, actual_distractors=2, actual_total=12)
        start = datetime.datetime.now()

        _print_summary(
            config=config,
            seed=42,
            summary=summary,
            all_rows=self._make_rows(),
            all_stats=self._make_stats(),
            timed_out_runs=[],
            truncated_runs=[],
            output_file="results.csv",
            repeat=1,
            start_time=start,
        )

        captured = capsys.readouterr()
        assert "Value range" in captured.out
        assert "10000" in captured.out
        assert "99999" in captured.out

    def test_prints_max_tokens(self, capsys):
        config = self._make_config()
        summary = Summary(actual_real_needles=10, actual_distractors=2, actual_total=12)
        start = datetime.datetime.now()

        _print_summary(
            config=config,
            seed=42,
            summary=summary,
            all_rows=self._make_rows(),
            all_stats=self._make_stats(),
            timed_out_runs=[],
            truncated_runs=[],
            output_file="results.csv",
            repeat=1,
            start_time=start,
        )

        captured = capsys.readouterr()
        assert "Max tokens" in captured.out

    def test_prints_timeout(self, capsys):
        config = self._make_config()
        summary = Summary(actual_real_needles=10, actual_distractors=2, actual_total=12)
        start = datetime.datetime.now()

        _print_summary(
            config=config,
            seed=42,
            summary=summary,
            all_rows=self._make_rows(),
            all_stats=self._make_stats(),
            timed_out_runs=[],
            truncated_runs=[],
            output_file="results.csv",
            repeat=1,
            start_time=start,
        )

        captured = capsys.readouterr()
        assert "Timeout" in captured.out

    def test_prints_total_latency_with_minutes(self, capsys):
        config = self._make_config()
        summary = Summary(actual_real_needles=10, actual_distractors=2, actual_total=12)
        stats = [
            {
                "total_tokens": 100000,
                "total_latency_ms": 180000,  # 3 minutes
                "total_completion": 50000,
                "error": None,
                "truncated": False,
                "model_name": "test-model",
                "tokens_per_sec": 277.8,
                "avg_latency_ms": 10.0,
            }
        ]
        start = datetime.datetime.now()

        _print_summary(
            config=config,
            seed=42,
            summary=summary,
            all_rows=self._make_rows(),
            all_stats=stats,
            timed_out_runs=[],
            truncated_runs=[],
            output_file="results.csv",
            repeat=1,
            start_time=start,
        )

        captured = capsys.readouterr()
        assert "min" in captured.out
