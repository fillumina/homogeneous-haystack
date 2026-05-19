import pytest

from haystack_test import (
    Needle,
    build_prompt,
    SYSTEM_PROMPT,
)


class TestBuildPrompt:
    def test_returns_two_messages(self, needles):
        haystack_text = "\n".join(f"{n.key} = {n.expected}" for n in needles[:5])
        result = build_prompt(haystack_text, needles)
        assert isinstance(result, list)
        assert len(result) == 2

    def test_first_message_is_system(self, needles):
        haystack_text = "\n".join(f"{n.key} = {n.expected}" for n in needles[:5])
        result = build_prompt(haystack_text, needles)
        assert result[0]["role"] == "system"

    def test_second_message_is_user(self, needles):
        haystack_text = "\n".join(f"{n.key} = {n.expected}" for n in needles[:5])
        result = build_prompt(haystack_text, needles)
        assert result[1]["role"] == "user"

    def test_system_message_content(self, needles):
        haystack_text = "\n".join(f"{n.key} = {n.expected}" for n in needles[:5])
        result = build_prompt(haystack_text, needles)
        assert result[0]["content"] == SYSTEM_PROMPT

    def test_haystack_text_in_user_message(self, needles):
        haystack_text = "KEY1 = 1234\nKEY2 = 5678"
        result = build_prompt(haystack_text, needles)
        assert haystack_text in result[1]["content"]

    def test_one_query_per_needle(self, needles):
        haystack_text = "KEY1 = 1234\nKEY2 = 5678"
        result = build_prompt(haystack_text, needles)
        user_content = result[1]["content"]
        for needle in needles:
            assert f"What is the value for {needle.key}?" in user_content

    def test_query_format(self, needles):
        haystack_text = "KEY1 = 1234"
        result = build_prompt(haystack_text, needles)
        user_content = result[1]["content"]
        assert "What is the value for" in user_content
        assert "?" in user_content

    def test_all_needles_queries_present(self, haystack_pairs, seeded_random):
        seeded_random(99)
        _, pairs = haystack_pairs
        from haystack_test import select_needles, build_haystack
        needles = select_needles(pairs, 10)
        text, _ = build_haystack(10)
        result = build_prompt(text, needles)
        user_content = result[1]["content"]
        for needle in needles:
            assert needle.key in user_content

    def test_empty_needles_list(self, haystack_pairs):
        haystack_text, _ = haystack_pairs
        result = build_prompt(haystack_text, [])
        assert len(result) == 2
        assert "What is the value for" not in result[1]["content"]



