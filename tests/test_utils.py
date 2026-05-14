import pytest

from haystack_test import _truncate


class TestTruncate:
    def test_short_text_unchanged(self):
        text = "line1\nline2\nline3"
        assert _truncate(text) == text

    def test_text_at_max_lines(self):
        text = "\n".join(f"line{i}" for i in range(10))
        result = _truncate(text, max_lines=5)
        assert result == text

    def test_long_text_truncated(self):
        lines = [f"line{i}" for i in range(20)]
        text = "\n".join(lines)
        result = _truncate(text, max_lines=2)
        assert "..." in result

    def test_truncated_shows_first_and_last_lines(self):
        lines = [f"line{i}" for i in range(10)]
        text = "\n".join(lines)
        result = _truncate(text, max_lines=3)
        assert "line0" in result
        assert "line9" in result

    def test_prefix_appears_between_parts(self):
        lines = [f"line{i}" for i in range(10)]
        text = "\n".join(lines)
        result = _truncate(text, max_lines=2)
        parts = result.split("\n")
        assert "..." in parts

    def test_custom_prefix(self):
        lines = [f"line{i}" for i in range(10)]
        text = "\n".join(lines)
        result = _truncate(text, max_lines=2, prefix="***")
        assert "***" in result
        assert "..." not in result

    def test_max_lines_zero(self):
        lines = [f"line{i}" for i in range(5)]
        text = "\n".join(lines)
        result = _truncate(text, max_lines=0)
        assert "..." in result

    def test_single_line(self):
        assert _truncate("single") == "single"

    def test_empty_string(self):
        assert _truncate("") == ""

    def test_preserves_line_count_in_first_part(self):
        lines = [f"line{i}" for i in range(20)]
        text = "\n".join(lines)
        result = _truncate(text, max_lines=5)
        first_part = result.split("\n")[0]
        # First 5 lines should be present
        for i in range(5):
            assert f"line{i}" in result

    def test_preserves_line_count_in_last_part(self):
        lines = [f"line{i}" for i in range(20)]
        text = "\n".join(lines)
        result = _truncate(text, max_lines=5)
        # Last 5 lines should be present
        for i in range(15, 20):
            assert f"line{i}" in result
