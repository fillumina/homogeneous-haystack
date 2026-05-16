import pytest

from haystack_test import Needle, ScoredNeedle, parse_response, score_needles


class TestParseResponse:
    def test_empty_response(self):
        result = parse_response("", set())
        assert result == {}

    def test_whitespace_only(self):
        result = parse_response("   \n  \n  ", set())
        assert result == {}

    def test_single_key_value(self):
        result = parse_response("ABC12345 = 5678", {"ABC12345"})
        assert result == {"ABC12345": "5678"}

    def test_multiple_key_values(self):
        text = "ABC12345 = 5678\nXYZ99999 = 1234"
        result = parse_response(text, {"ABC12345", "XYZ99999"})
        assert result == {"ABC12345": "5678", "XYZ99999": "1234"}

    def test_skips_non_integer_values(self):
        text = "ABC12345 = notanumber\nXYZ99999 = 1234"
        result = parse_response(text, {"ABC12345", "XYZ99999"})
        assert "ABC12345" not in result
        assert result == {"XYZ99999": "1234"}

    def test_skips_empty_values(self):
        text = "ABC12345 = \nXYZ99999 = 1234"
        result = parse_response(text, {"ABC12345", "XYZ99999"})
        assert "ABC12345" not in result
        assert result == {"XYZ99999": "1234"}

    def test_handles_spaces_around_equals(self):
        text = "ABC12345=5678"
        result = parse_response(text, {"ABC12345"})
        assert result == {"ABC12345": "5678"}

    def test_handles_spaces_around_equals_v2(self):
        text = "ABC12345  =  5678"
        result = parse_response(text, {"ABC12345"})
        assert result == {"ABC12345": "5678"}

    def test_partial_match(self):
        text = "ABC12345 = 5678"
        result = parse_response(text, {"ABC12345", "XYZ99999"})
        assert result == {"ABC12345": "5678"}
        assert "XYZ99999" not in result

    def test_multiple_lines_with_blanks(self):
        text = "ABC12345 = 5678\n\n\nXYZ99999 = 1234"
        result = parse_response(text, {"ABC12345", "XYZ99999"})
        assert result == {"ABC12345": "5678", "XYZ99999": "1234"}


class TestScoreNeedles:
    def test_correct_needle(self):
        needles = [Needle(index=0, key="ABC12345", expected=5678, is_distractor=False)]
        result = score_needles(needles, {"ABC12345": "5678"})
        assert isinstance(result[0], ScoredNeedle)
        assert result[0].correct == 1
        assert result[0].expected == 5678
        assert result[0].actual == "5678"

    def test_incorrect_needle(self):
        needles = [Needle(index=0, key="ABC12345", expected=5678, is_distractor=False)]
        result = score_needles(needles, {"ABC12345": "9999"})
        assert isinstance(result[0], ScoredNeedle)
        assert result[0].correct == 0
        assert result[0].expected == 5678
        assert result[0].actual == "9999"

    def test_missing_needle(self):
        needles = [Needle(index=0, key="ABC12345", expected=5678, is_distractor=False)]
        result = score_needles(needles, {})
        assert isinstance(result[0], ScoredNeedle)
        assert result[0].correct == 0
        assert result[0].expected == 5678
        assert result[0].actual == ""

    def test_distractor_no_response(self):
        needles = [Needle(index=0, key="FAKEKEY", expected=None, is_distractor=True)]
        result = score_needles(needles, {})
        assert isinstance(result[0], ScoredNeedle)
        assert result[0].correct == 1
        assert result[0].expected == ""
        assert result[0].actual == ""

    def test_distractor_with_response(self):
        needles = [Needle(index=0, key="FAKEKEY", expected=None, is_distractor=True)]
        result = score_needles(needles, {"FAKEKEY": "1234"})
        assert isinstance(result[0], ScoredNeedle)
        assert result[0].correct == 0
        assert result[0].expected == ""
        assert result[0].actual == "1234"

    def test_multiple_needles(self):
        needles = [
            Needle(index=0, key="A", expected=100, is_distractor=False),
            Needle(index=1, key="B", expected=200, is_distractor=False),
            Needle(index=2, key="C", expected=None, is_distractor=True),
        ]
        result = score_needles(needles, {"A": "100", "B": "999"})
        assert result[0].correct == 1
        assert result[1].correct == 0
        assert result[2].correct == 1

    def test_returns_dataclass_with_all_fields(self):
        needles = [Needle(index=0, key="ABC12345", expected=5678, is_distractor=False)]
        result = score_needles(needles, {"ABC12345": "5678"})
        assert isinstance(result[0], ScoredNeedle)
        assert result[0].needle_key == "ABC12345"
        assert result[0].expected == 5678
        assert result[0].actual == "5678"
        assert result[0].correct == 1

    def test_expected_preserved_for_distractor(self):
        needles = [Needle(index=0, key="FAKE", expected=None, is_distractor=True)]
        result = score_needles(needles, {})
        assert result[0].expected == ""
        assert result[0].correct == 1

    def test_expected_preserved_for_real(self):
        needles = [Needle(index=0, key="ABC12345", expected=5678, is_distractor=False)]
        result = score_needles(needles, {})
        assert result[0].expected == 5678

    def test_empty_needles_list(self):
        result = score_needles([], {})
        assert result == []
