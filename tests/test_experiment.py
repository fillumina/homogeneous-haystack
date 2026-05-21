import pytest
from unittest.mock import patch

from haystack_test import Config, HaystackQueryError, ResultRow, run_single_experiment


class TestRunSingleExperiment:
    def _make_config(self, max_tokens=240000, fuzz=0.49):
        return Config(
            endpoint="http://localhost:8080/v1/chat/completions",
            model=None,
            key_len=8,
            val_min=10000,
            val_max=99999,
            haystack_num=10,
            needles_num=5,
            distractor_pct=None,
            distractors_num=None,
            temperature=0.0,
            max_tokens=max_tokens,
            timeout=10,
            full=False,
            fuzz=fuzz,
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
        rows, model_name, stats = run_single_experiment(1, config, 42)

        assert isinstance(rows, list)
        assert model_name == "test-model"
        assert isinstance(stats, dict)

    @patch("haystack_test.query_llama")
    def test_success_returns_correct_stats(self, mock_query_llama):
        mock_query_llama.return_value = (
            self._make_mock_response(),
            200.0,
        )

        config = self._make_config()
        rows, model_name, stats = run_single_experiment(1, config, 42)

        assert stats["total_tokens"] == 100
        assert stats["total_latency_ms"] == 200.0
        assert stats["total_completion"] == 50
        assert stats["error"] is None
        assert stats["truncated"] is False
        assert stats["model_name"] == "test-model"
        assert stats["tokens_per_sec"] == 250.0  # 50 / (200/1000)
        assert stats["avg_latency_ms"] > 0

    @patch("haystack_test.query_llama", side_effect=HaystackQueryError("API failed"))
    def test_error_path_raises(self, mock_query_llama):
        config = self._make_config()

        with pytest.raises(HaystackQueryError, match="API failed"):
            run_single_experiment(1, config, 42)

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
        rows, model_name, stats = run_single_experiment(1, config, 42)

        assert isinstance(rows, list)
        assert stats["truncated"] is True

    @patch("haystack_test.query_llama")
    def test_stats_tokens_per_sec_zero_latency(self, mock_query_llama):
        mock_query_llama.return_value = (
            self._make_mock_response(),
            0.0,
        )

        config = self._make_config()
        rows, model_name, stats = run_single_experiment(1, config, 42)

        # tokens_per_sec should not be calculated when latency is 0
        assert "tokens_per_sec" not in stats

    @patch("haystack_test.query_llama")
    def test_stats_avg_latency_with_needles(self, mock_query_llama):
        mock_query_llama.return_value = (
            self._make_mock_response(),
            1000.0,
        )

        config = self._make_config()
        rows, model_name, stats = run_single_experiment(1, config, 42)

        # avg_latency = 1000 / num_needles
        assert stats["avg_latency_ms"] > 0

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
            rows, model_name, stats = run_single_experiment(1, config, 42)

            assert stats["avg_latency_ms"] == 0.0

    @patch("haystack_test.query_llama", side_effect=HaystackQueryError("API failed"))
    def test_error_propagates(self, mock_query_llama):
        config = self._make_config()

        with pytest.raises(HaystackQueryError, match="API failed"):
            run_single_experiment(1, config, 42)

    @patch("haystack_test.query_llama", side_effect=HaystackQueryError("connection lost"))
    def test_error_propagates_with_message(self, mock_query_llama):
        config = self._make_config()

        with pytest.raises(HaystackQueryError, match="connection lost"):
            run_single_experiment(1, config, 42)

    @patch("haystack_test.random.seed")
    @patch("haystack_test.query_llama")
    def test_seed_is_set(self, mock_query_llama, mock_seed):
        mock_query_llama.return_value = (self._make_mock_response(), 100.0)

        config = self._make_config(fuzz=0)
        run_single_experiment(1, config, 999)
        mock_seed.assert_called_with(999)

    @patch("haystack_test.query_llama")
    def test_rows_are_result_rows(self, mock_query_llama):
        mock_query_llama.return_value = (self._make_mock_response(), 100.0)

        config = self._make_config()
        rows, model_name, stats = run_single_experiment(1, config, 42)

        assert all(isinstance(r, ResultRow) for r in rows)

    @patch("haystack_test.query_llama")
    def test_run_index_in_rows(self, mock_query_llama):
        mock_query_llama.return_value = (self._make_mock_response(), 100.0)

        config = self._make_config()
        rows, model_name, stats = run_single_experiment(7, config, 42)

        assert all(r.run == 7 for r in rows)

    @patch("haystack_test.query_llama")
    def test_stats_is_runstats_dict(self, mock_query_llama):
        mock_query_llama.return_value = (self._make_mock_response(), 100.0)

        config = self._make_config()
        rows, model_name, stats = run_single_experiment(1, config, 42)

        assert isinstance(stats, dict)
        assert "total_tokens" in stats
        assert "total_latency_ms" in stats
        assert "total_completion" in stats
        assert "error" in stats
        assert "truncated" in stats
        assert "model_name" in stats

    @patch("haystack_test.query_llama", side_effect=HaystackQueryError("Connection refused"))
    def test_error_propagates_from_query(self, mock_query_llama):
        config = self._make_config()

        with pytest.raises(HaystackQueryError, match="Connection refused"):
            run_single_experiment(1, config, 42)

    @patch("haystack_test.query_llama")
    def test_truncated_stats_has_true(self, mock_query_llama):
        mock_query_llama.return_value = (
            self._make_mock_response(
                completion_tokens=240000,
            ),
            100.0,
        )

        config = self._make_config(max_tokens=240000)
        rows, model_name, stats = run_single_experiment(1, config, 42)

        assert stats["truncated"] is True

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
        rows, model_name, stats = run_single_experiment(1, config, 42)

        assert stats["model_name"] == "llama-3"
