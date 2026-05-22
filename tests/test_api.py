import json
import socket
import urllib.error
from unittest.mock import MagicMock, Mock, patch

import pytest

from haystack_test import (
    Config,
    DebugContext,
    HaystackQueryError,
    Needle,
    QueryResult,
    query_model,
    query_llama,
)


class TestQueryLlama:
    def _make_success_response(self, content="test answer", model="test-model"):
        return {
            "model": model,
            "choices": [
                {
                    "message": {
                        "role": "assistant",
                        "content": content,
                    }
                }
            ],
            "usage": {
                "prompt_tokens": 100,
                "completion_tokens": 50,
                "total_tokens": 150,
            },
        }

    @patch("haystack_test.urllib.request.urlopen")
    def test_success_returns_response_and_latency(self, mock_urlopen):
        mock_resp = MagicMock()
        mock_resp.read.return_value = json.dumps(self._make_success_response()).encode("utf-8")
        mock_resp.__enter__ = Mock(return_value=mock_resp)
        mock_resp.__exit__ = Mock(return_value=None)
        mock_urlopen.return_value = mock_resp

        response, latency = query_llama(
            "http://localhost:8080/v1/chat/completions",
            [{"role": "user", "content": "test"}],
        )

        assert response["model"] == "test-model"
        assert response["choices"][0]["message"]["content"] == "test answer"
        assert isinstance(latency, float)
        assert latency > 0

    @patch("haystack_test.urllib.request.urlopen")
    def test_success_with_model_param(self, mock_urlopen):
        mock_resp = MagicMock()
        mock_resp.read.return_value = json.dumps(self._make_success_response()).encode("utf-8")
        mock_resp.__enter__ = Mock(return_value=mock_resp)
        mock_resp.__exit__ = Mock(return_value=None)
        mock_urlopen.return_value = mock_resp

        response, latency = query_llama(
            "http://localhost:8080/v1/chat/completions",
            [{"role": "user", "content": "test"}],
            model="my-model",
        )

        assert response["model"] == "test-model"

    @patch("haystack_test.urllib.request.urlopen")
    def test_success_with_custom_params(self, mock_urlopen):
        mock_resp = MagicMock()
        mock_resp.read.return_value = json.dumps(self._make_success_response()).encode("utf-8")
        mock_resp.__enter__ = Mock(return_value=mock_resp)
        mock_resp.__exit__ = Mock(return_value=None)
        mock_urlopen.return_value = mock_resp

        response, latency = query_llama(
            "http://localhost:8080/v1/chat/completions",
            [{"role": "user", "content": "test"}],
            model="custom",
            temperature=0.7,
            max_tokens=500,
            timeout=30,
        )

        assert response["model"] == "test-model"

    @patch("haystack_test.urllib.request.urlopen")
    def test_default_model_is_local(self, mock_urlopen):
        mock_resp = MagicMock()
        response_body = {
            "choices": [
                {
                    "message": {
                        "role": "assistant",
                        "content": "answer",
                    }
                }
            ],
        }
        mock_resp.read.return_value = json.dumps(response_body).encode("utf-8")
        mock_resp.__enter__ = Mock(return_value=mock_resp)
        mock_resp.__exit__ = Mock(return_value=None)
        mock_urlopen.return_value = mock_resp

        query_llama(
            "http://localhost:8080/v1/chat/completions",
            [{"role": "user", "content": "test"}],
        )

        # Verify the payload sent had model="local"
        call_args = mock_urlopen.call_args
        data = json.loads(call_args[0][0].data)
        assert data["model"] == "local"

    @patch("haystack_test.urllib.request.urlopen")
    def test_stream_is_false(self, mock_urlopen):
        mock_resp = MagicMock()
        mock_resp.read.return_value = json.dumps(self._make_success_response()).encode("utf-8")
        mock_resp.__enter__ = Mock(return_value=mock_resp)
        mock_resp.__exit__ = Mock(return_value=None)
        mock_urlopen.return_value = mock_resp

        query_llama(
            "http://localhost:8080/v1/chat/completions",
            [{"role": "user", "content": "test"}],
        )

        call_args = mock_urlopen.call_args
        data = json.loads(call_args[0][0].data)
        assert data["stream"] is False

    @patch("haystack_test.urllib.request.urlopen")
    def test_content_type_header(self, mock_urlopen):
        mock_resp = MagicMock()
        mock_resp.read.return_value = json.dumps(self._make_success_response()).encode("utf-8")
        mock_resp.__enter__ = Mock(return_value=mock_resp)
        mock_resp.__exit__ = Mock(return_value=None)
        mock_urlopen.return_value = mock_resp

        query_llama(
            "http://localhost:8080/v1/chat/completions",
            [{"role": "user", "content": "test"}],
        )

        call_args = mock_urlopen.call_args
        req = call_args[0][0]
        # urllib uses lowercase 'Content-type' key
        assert "application/json" in req.headers.get("Content-type") or req.headers.get("Content-Type") == "application/json"
        assert req.method == "POST"

    def test_socket_timeout_raises_haystack_query_error(self):
        with patch("haystack_test.urllib.request.urlopen") as mock_urlopen:
            mock_urlopen.side_effect = socket.timeout("timed out")

            with pytest.raises(HaystackQueryError, match="Timed out"):
                query_llama(
                    "http://localhost:8080/v1/chat/completions",
                    [{"role": "user", "content": "test"}],
                    timeout=1,
                )

    def test_http_error_raises_haystack_query_error(self):
        with patch("haystack_test.urllib.request.urlopen") as mock_urlopen:
            http_error = urllib.error.HTTPError(
                "http://localhost:8080/v1/chat/completions",
                400,
                "Bad Request",
                {},
                None,
            )
            mock_urlopen.side_effect = http_error

            with pytest.raises(HaystackQueryError, match="HTTP 400"):
                query_llama(
                    "http://localhost:8080/v1/chat/completions",
                    [{"role": "user", "content": "test"}],
                )

    def test_http_error_500_raises(self):
        with patch("haystack_test.urllib.request.urlopen") as mock_urlopen:
            http_error = urllib.error.HTTPError(
                "http://localhost:8080/v1/chat/completions",
                500,
                "Internal Server Error",
                {},
                MagicMock(),
            )
            mock_urlopen.side_effect = http_error

            with pytest.raises(HaystackQueryError, match="HTTP 500"):
                query_llama(
                    "http://localhost:8080/v1/chat/completions",
                    [{"role": "user", "content": "test"}],
                )

    def test_url_error_raises_haystack_query_error(self):
        with patch("haystack_test.urllib.request.urlopen") as mock_urlopen:
            url_error = urllib.error.URLError(ConnectionRefusedError("Connection refused"))
            mock_urlopen.side_effect = url_error

            with pytest.raises(HaystackQueryError, match="Connection failed"):
                query_llama(
                    "http://localhost:8080/v1/chat/completions",
                    [{"role": "user", "content": "test"}],
                )


class TestQueryModel:
    def _make_mock_success_response(self, content="test answer"):
        return {
            "model": "test-model",
            "choices": [
                {
                    "message": {
                        "role": "assistant",
                        "content": content,
                    }
                }
            ],
            "usage": {
                "prompt_tokens": 100,
                "completion_tokens": 50,
                "total_tokens": 150,
            },
        }

    def _make_config(self, max_tokens=240000):
        return Config(
            endpoint="http://localhost:8080/v1/chat/completions",
            model=None,
            key_len=8,
            val_min=10000,
            val_max=99999,
            haystack_num=100,
            needles_num=5,
            distractors_num=0,
            temperature=0.0,
            max_tokens=max_tokens,
            timeout=10,
            full=False,
            seed=42,
            repeat=1,
            output_filename="results.csv",
            show="all",
        )

    def _make_needles(self):
        return [
            Needle(index=0, key="ABCD1234", expected=1234, is_distractor=False),
            Needle(index=1, key="EFGH5678", expected=5678, is_distractor=False),
        ]

    @patch("haystack_test.query_llama")
    def test_success_path_returns_parsed_results(self, mock_query_llama):
        mock_query_llama.return_value = (
            self._make_mock_success_response("ABCD1234 = 1234\nEFGH5678 = 5678"),
            100.0,
        )

        config = self._make_config()
        needles = self._make_needles()
        haystack_text = "ABCD1234 = 1234\nEFGH5678 = 5678"

        result, debug = query_model(config, needles, haystack_text)

        assert isinstance(result, QueryResult)
        assert result.parsed == {"ABCD1234": "1234", "EFGH5678": "5678"}
        assert not result.truncated
        assert result.latency_ms == 100.0
        assert isinstance(debug, DebugContext)
        assert debug.model_name == "test-model"

    @patch("haystack_test.query_llama")
    def test_error_propagates_haystack_query_error(self, mock_query_llama):
        mock_query_llama.side_effect = HaystackQueryError("API failed")

        config = self._make_config()
        needles = self._make_needles()
        haystack_text = "ABCD1234 = 1234"

        with pytest.raises(HaystackQueryError, match="API failed"):
            query_model(config, needles, haystack_text)

    @patch("haystack_test.query_llama")
    def test_error_propagates_without_debug(self, mock_query_llama):
        mock_query_llama.side_effect = HaystackQueryError("API failed")

        config = self._make_config()
        needles = self._make_needles()
        haystack_text = "ABCD1234 = 1234"

        with pytest.raises(HaystackQueryError):
            query_model(config, needles, haystack_text)

    @patch("haystack_test.query_llama")
    def test_truncated_detection(self, mock_query_llama):
        response = self._make_mock_success_response("ABCD1234 = 1234")
        response["usage"]["completion_tokens"] = 240000
        mock_query_llama.return_value = (response, 100.0)

        config = self._make_config(max_tokens=240000)
        needles = self._make_needles()
        haystack_text = "ABCD1234 = 1234"

        result, debug = query_model(config, needles, haystack_text)

        assert result.truncated is True

    @patch("haystack_test.query_llama")
    def test_not_truncated_when_below_max(self, mock_query_llama):
        response = self._make_mock_success_response("ABCD1234 = 1234")
        response["usage"]["completion_tokens"] = 20
        mock_query_llama.return_value = (response, 100.0)

        config = self._make_config(max_tokens=240000)
        needles = self._make_needles()
        haystack_text = "ABCD1234 = 1234"

        result, debug = query_model(config, needles, haystack_text)

        assert result.truncated is False

    @patch("haystack_test.query_llama")
    def test_debug_is_returned(self, mock_query_llama):
        mock_query_llama.return_value = (
            self._make_mock_success_response("ABCD1234 = 1234"),
            100.0,
        )

        config = self._make_config()
        needles = self._make_needles()
        haystack_text = "ABCD1234 = 1234"

        result, debug = query_model(config, needles, haystack_text)

        assert isinstance(debug, DebugContext)
        assert isinstance(debug.messages, list)
        assert len(debug.messages) == 2
        assert debug.messages[0]["role"] == "system"

    @patch("haystack_test.query_llama")
    def test_usage_is_returned(self, mock_query_llama):
        mock_query_llama.return_value = (
            self._make_mock_success_response("ABCD1234 = 1234"),
            100.0,
        )

        config = self._make_config()
        needles = self._make_needles()
        haystack_text = "ABCD1234 = 1234"

        result, debug = query_model(config, needles, haystack_text)

        assert result.usage is not None
        assert result.usage["total_tokens"] == 150

    @patch("haystack_test.query_llama")
    def test_response_text_in_debug(self, mock_query_llama):
        mock_query_llama.return_value = (
            self._make_mock_success_response("my custom response"),
            100.0,
        )

        config = self._make_config()
        needles = self._make_needles()
        haystack_text = "ABCD1234 = 1234"

        result, debug = query_model(config, needles, haystack_text)

        assert debug.response_text == "my custom response"

    @patch("haystack_test.query_llama")
    def test_raw_response_in_debug(self, mock_query_llama):
        mock_query_llama.return_value = (
            self._make_mock_success_response("ABCD1234 = 1234"),
            100.0,
        )

        config = self._make_config()
        needles = self._make_needles()
        haystack_text = "ABCD1234 = 1234"

        result, debug = query_model(config, needles, haystack_text)

        assert debug.raw_response is not None
        assert debug.raw_response["model"] == "test-model"
