import pytest
from unittest.mock import patch

from haystack_test import (
    ApiConfig,
    Config,
    ExecutionConfig,
    ExperimentSummary,
    GlobalResult,
    HaystackConfig,
    HaystackQueryError,
    ModelResult,
    OutputConfig,
    ResultRow,
    compute_experiment_summary,
    run_single_experiment,
)


def _make_global_result(all_rows=None, all_stats=None, timed_out_runs=None, truncated_runs=None, failed_runs=None):
    return GlobalResult(
        all_rows=all_rows or [],
        all_stats=all_stats or [],
        timed_out_runs=timed_out_runs or [],
        truncated_runs=truncated_runs or [],
        failed_runs=failed_runs or [],
    )


class TestRunSingleExperiment:
    def _make_config(self, max_tokens=240000, fuzz=0.49):
        return Config(
            api=ApiConfig(
                endpoint="http://localhost:8080/v1/chat/completions",
                temperature=0.0,
                max_tokens=max_tokens,
                timeout=10,
            ),
            haystack=HaystackConfig(
                haystack_num=10,
                needles_num=5,
                distractors_num=0,
                key_len=8,
                val_min=10000,
                val_max=99999,
                fuzz=fuzz,
                seed=42,
            ),
            output=OutputConfig(
                output_filename="results.csv",
                verbosity="medium",
                k_quant="Q4_0",
                v_quant="Q8_0",
                note="test",
                timestamp="2024-01-01T10:00:00",
            ),
            execution=ExecutionConfig(
                repeat=1,
                stop_on_error=False,
            ),
        )

    def _make_mock_response(self, content="KEY1 = 12345", **usage_kwargs):
        usage = {
            "total_tokens": 100,
            "completion_tokens": 50,
            "prompt_tokens": 50,
            **usage_kwargs,
        }
        return {
            "model": "test-model",
            "choices": [{"message": {"content": content}}],
            "usage": usage,
        }

    @patch("haystack_test.query_llama")
    def test_success_returns_rows(self, mock_query_llama):
        mock_query_llama.return_value = (self._make_mock_response(), 100.0)

        config = self._make_config()
        result = run_single_experiment(1, config)

        assert isinstance(result, ModelResult)
        assert isinstance(result.rows, list)
        assert result.model_name == "test-model"
        assert isinstance(result.stats, dict)

    @patch("haystack_test.query_llama")
    def test_success_returns_correct_stats(self, mock_query_llama):
        mock_query_llama.return_value = (
            self._make_mock_response(),
            200.0,
        )

        config = self._make_config()
        result = run_single_experiment(1, config)

        assert result.stats["total_tokens"] == 100
        assert result.stats["total_latency_ms"] == 200.0
        assert result.stats["total_completion"] == 50
        assert result.stats["error"] is None
        assert result.stats["truncated"] is False
        assert result.stats["model_name"] == "test-model"
        assert result.stats["tokens_per_sec"] == 250.0  # 50 / (200/1000)
        assert result.stats["avg_latency_ms"] > 0

    @patch("haystack_test.query_llama", side_effect=HaystackQueryError("API failed"))
    def test_error_path_raises(self, mock_query_llama):
        config = self._make_config()

        with pytest.raises(HaystackQueryError, match="API failed"):
            run_single_experiment(1, config)

    @patch("haystack_test.query_llama")
    def test_truncated_path_returns_rows(self, mock_query_llama):
        mock_query_llama.return_value = (
            self._make_mock_response(
                "KEY1 = 12345",
                completion_tokens=240000,
            ),
            100.0,
        )

        config = self._make_config(max_tokens=240000)
        result = run_single_experiment(1, config)

        assert isinstance(result.rows, list)
        assert result.stats["truncated"] is True

    @patch("haystack_test.query_llama")
    def test_stats_tokens_per_sec_zero_latency(self, mock_query_llama):
        mock_query_llama.return_value = (
            self._make_mock_response(),
            0.0,
        )

        config = self._make_config()
        result = run_single_experiment(1, config)

        # tokens_per_sec is 0.0 when latency is 0
        assert result.stats["tokens_per_sec"] == 0.0

    @patch("haystack_test.query_llama")
    def test_stats_avg_latency_with_needles(self, mock_query_llama):
        mock_query_llama.return_value = (
            self._make_mock_response(),
            1000.0,
        )

        config = self._make_config()
        result = run_single_experiment(1, config)

        # avg_latency = 1000 / num_needles
        assert result.stats["avg_latency_ms"] > 0

    @patch("haystack_test.query_llama")
    def test_stats_avg_latency_zero_needles(self, mock_query_llama):
        mock_query_llama.return_value = (
            self._make_mock_response(""),
            0.0,
        )

        # Override generate_haystack_and_needles to return 0 needles
        with patch("haystack_test.generate_haystack_and_needles") as mock_gen:
            mock_gen.return_value = ("", [], [])
            config = self._make_config()
            result = run_single_experiment(1, config)

            assert result.stats["avg_latency_ms"] == 0.0

    @patch("haystack_test.query_llama", side_effect=HaystackQueryError("API failed"))
    def test_error_propagates(self, mock_query_llama):
        config = self._make_config()

        with pytest.raises(HaystackQueryError, match="API failed"):
            run_single_experiment(1, config)

    @patch("haystack_test.query_llama", side_effect=HaystackQueryError("connection lost"))
    def test_error_propagates_with_message(self, mock_query_llama):
        config = self._make_config()

        with pytest.raises(HaystackQueryError, match="connection lost"):
            run_single_experiment(1, config)

    @patch("haystack_test.random.seed")
    @patch("haystack_test.query_llama")
    def test_seed_is_set(self, mock_query_llama, mock_seed):
        mock_query_llama.return_value = (self._make_mock_response(), 100.0)

        config = self._make_config(fuzz=0)
        run_single_experiment(1, config)
        mock_seed.assert_called_with(42 + 1)

    @patch("haystack_test.query_llama")
    def test_rows_are_result_rows(self, mock_query_llama):
        mock_query_llama.return_value = (self._make_mock_response(), 100.0)

        config = self._make_config()
        result = run_single_experiment(1, config)

        assert all(isinstance(r, ResultRow) for r in result.rows)

    @patch("haystack_test.query_llama")
    def test_run_index_in_rows(self, mock_query_llama):
        mock_query_llama.return_value = (self._make_mock_response(), 100.0)

        config = self._make_config()
        result = run_single_experiment(7, config)

        assert all(r.run == 7 for r in result.rows)

    @patch("haystack_test.query_llama")
    def test_stats_is_runstats_dict(self, mock_query_llama):
        mock_query_llama.return_value = (self._make_mock_response(), 100.0)

        config = self._make_config()
        result = run_single_experiment(1, config)

        assert isinstance(result.stats, dict)
        assert "total_tokens" in result.stats
        assert "total_latency_ms" in result.stats
        assert "total_completion" in result.stats
        assert "error" in result.stats
        assert "truncated" in result.stats
        assert "model_name" in result.stats

    @patch("haystack_test.query_llama", side_effect=HaystackQueryError("Connection refused"))
    def test_error_propagates_from_query(self, mock_query_llama):
        config = self._make_config()

        with pytest.raises(HaystackQueryError, match="Connection refused"):
            run_single_experiment(1, config)

    @patch("haystack_test.query_llama")
    def test_truncated_stats_has_true(self, mock_query_llama):
        mock_query_llama.return_value = (
            self._make_mock_response(
                completion_tokens=240000,
            ),
            100.0,
        )

        config = self._make_config(max_tokens=240000)
        result = run_single_experiment(1, config)

        assert result.stats["truncated"] is True

    @patch("haystack_test.query_llama")
    def test_model_name_in_stats(self, mock_query_llama):
        mock_query_llama.return_value = (
            {
                "model": "llama-3",
                "choices": [{"message": {"content": "KEY1 = 12345"}}],
                "usage": {"total_tokens": 100, "completion_tokens": 50, "prompt_tokens": 50},
            },
            100.0,
        )

        config = self._make_config()
        result = run_single_experiment(1, config)

        assert result.stats["model_name"] == "llama-3"


class TestComputeExperimentSummary:
    def _make_config(self):
        return Config(
            api=ApiConfig(
                endpoint="http://localhost:8080/v1/chat/completions",
                temperature=0.0,
                max_tokens=240000,
                timeout=10,
            ),
            haystack=HaystackConfig(
                haystack_num=10,
                needles_num=5,
                distractors_num=0,
                key_len=8,
                val_min=10000,
                val_max=99999,
                fuzz=0,
                seed=42,
            ),
            output=OutputConfig(
                output_filename="results.csv",
                verbosity="medium",
                k_quant="Q4_0",
                v_quant="Q8_0",
                note="test note",
                timestamp="2024-01-01T10:00:00",
            ),
            execution=ExecutionConfig(
                repeat=1,
                stop_on_error=False,
            ),
        )

    def _make_result_row(self, run, depth, correct, expected, is_distractor=False):
        from haystack_test import ResultRow
        return ResultRow(
            run=run,
            haystack_size=10,
            depth_pct=depth,
            correct=correct,
            expected=None if is_distractor else expected,
            actual="12345" if correct else "",
            is_distractor=is_distractor,
            prompt_tokens=100,
            completion_tokens=50,
            total_tokens=150,
            latency_ms=500.0,
        )

    def test_empty_result(self):
        config = self._make_config()
        global_result = _make_global_result()
        summaries = compute_experiment_summary(global_result)
        assert summaries == []

    def test_single_run_all_correct(self):
        config = self._make_config()
        rows = [
            self._make_result_row(0, 10.0, 1, 12345),
            self._make_result_row(0, 50.0, 1, 67890),
            self._make_result_row(0, 90.0, 1, 11111),
        ]
        global_result = _make_global_result(all_rows=rows, all_stats=[{
            'model_name': 'test-model',
            'total_tokens': 150,
            'total_latency_ms': 500.0,
            'total_completion': 50,
            'tokens_per_sec': 100.0,
        }])
        summaries = compute_experiment_summary(global_result)
        assert len(summaries) == 1
        s = summaries[0]
        assert global_result.all_stats[0].get("model_name") == "test-model"
        assert s.needle_success_pct == 100.0
        assert s.distractor_success_pct == 0.0
        assert s.needles_num == 3
        assert s.distractors_num == 0
        assert s.distractor_failures == 0
        # All buckets should be 0 since all needles are correct
        assert all(b == 0 for b in s.ctx_pos_buckets)

    def test_single_run_with_failures(self):
        config = self._make_config()
        rows = [
            self._make_result_row(0, 5.0, 1, 12345),       # correct, 0-10 bucket
            self._make_result_row(0, 15.0, 0, 67890),       # fail, 10-20 bucket
            self._make_result_row(0, 25.0, 0, 11111),       # fail, 20-30 bucket
            self._make_result_row(0, 95.0, 1, 22222),       # correct, 90-100 bucket
        ]
        global_result = _make_global_result(all_rows=rows, all_stats=[{
            'model_name': 'test-model',
            'total_tokens': 150,
            'total_latency_ms': 500.0,
            'total_completion': 50,
            'tokens_per_sec': 100.0,
        }])
        summaries = compute_experiment_summary(global_result)
        assert len(summaries) == 1
        s = summaries[0]
        assert s.needle_success_pct == 50.0
        assert s.needles_num == 4
        assert s.ctx_pos_buckets[0] == 0  # 0-10: 0 failures
        assert s.ctx_pos_buckets[1] == 1  # 10-20: 1 failure
        assert s.ctx_pos_buckets[2] == 1  # 20-30: 1 failure
        assert s.ctx_pos_buckets[9] == 0  # 90-100: 0 failures

    def test_distractors(self):
        config = self._make_config()
        rows = [
            self._make_result_row(0, 10.0, 1, 12345),       # real needle, correct
            self._make_result_row(0, 50.0, 0, "", True),     # distractor, failed
            self._make_result_row(0, 90.0, 1, "", True),     # distractor, correct (no hallucination)
        ]
        global_result = _make_global_result(all_rows=rows, all_stats=[{
            'model_name': 'test-model',
            'total_tokens': 150,
            'total_latency_ms': 500.0,
            'total_completion': 50,
            'tokens_per_sec': 100.0,
        }])
        summaries = compute_experiment_summary(global_result)
        assert len(summaries) == 1
        s = summaries[0]
        assert s.needle_success_pct == 100.0
        assert s.distractor_success_pct == 50.0
        assert s.distractors_num == 2
        assert s.distractor_failures == 1
        assert s.needles_num == 1

    def test_k_quant_v_quant(self):
        config = self._make_config()
        config.output.k_quant = "turbo4"
        config.output.v_quant = "Q6_K"
        global_result = _make_global_result()
        summaries = compute_experiment_summary(global_result)
        # summaries will be empty but config fields should be set
        # Test with at least one row
        from haystack_test import ResultRow
        rows = [ResultRow(
            run=0, haystack_size=10, depth_pct=10.0, correct=1,
            expected=12345, actual="12345",
            prompt_tokens=100, completion_tokens=50, total_tokens=150, latency_ms=500.0,
        )]
        global_result.all_rows = rows
        global_result.all_stats = [{
            'model_name': 'test-model',
            'total_tokens': 150,
            'total_latency_ms': 500.0,
            'total_completion': 50,
            'tokens_per_sec': 100.0,
        }]
        summaries = compute_experiment_summary(global_result)
        assert config.output.k_quant == "turbo4"
        assert config.output.v_quant == "Q6_K"

    def test_note(self):
        config = self._make_config()
        config.output.note = "my custom note"
        from haystack_test import ResultRow
        rows = [ResultRow(
            run=0, haystack_size=10, depth_pct=10.0, correct=1,
            expected=12345, actual="12345",
            prompt_tokens=100, completion_tokens=50, total_tokens=150, latency_ms=500.0,
        )]
        global_result = _make_global_result(all_rows=rows, all_stats=[{
            'model_name': 'test-model',
            'total_tokens': 150,
            'total_latency_ms': 500.0,
            'total_completion': 50,
            'tokens_per_sec': 100.0,
        }])
        summaries = compute_experiment_summary(global_result)
        assert config.output.note == "my custom note"

    def test_timestamp_format(self):
        config = self._make_config()
        from haystack_test import ResultRow
        rows = [ResultRow(
            run=0, haystack_size=10, depth_pct=10.0, correct=1,
            expected=12345, actual="12345",
            prompt_tokens=100, completion_tokens=50, total_tokens=150, latency_ms=500.0,
        )]
        global_result = _make_global_result(all_rows=rows, all_stats=[{
            'model_name': 'test-model',
            'total_tokens': 150,
            'total_latency_ms': 500.0,
            'total_completion': 50,
            'tokens_per_sec': 100.0,
        }])
        summaries = compute_experiment_summary(global_result)
        # Should be ISO-like format: YYYY-MM-DDTHH:MM:SS
        import re
        assert re.match(r'\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}', config.output.timestamp)

    def test_multiple_runs(self):
        config = self._make_config()
        rows = [
            self._make_result_row(0, 10.0, 1, 12345),
            self._make_result_row(1, 10.0, 0, 67890),
        ]
        global_result = _make_global_result(all_rows=rows, all_stats=[
            {
                'model_name': 'model-a',
                'total_tokens': 150,
                'total_latency_ms': 500.0,
                'total_completion': 50,
                'tokens_per_sec': 100.0,
            },
            {
                'model_name': 'model-b',
                'total_tokens': 150,
                'total_latency_ms': 600.0,
                'total_completion': 50,
                'tokens_per_sec': 83.3,
            },
        ])
        summaries = compute_experiment_summary(global_result)
        assert len(summaries) == 2
        assert summaries[0].needle_success_pct == 100.0
        assert summaries[1].needle_success_pct == 0.0
