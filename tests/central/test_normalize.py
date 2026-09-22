"""nlp/normalize.py — the two normalisation levels every utterance passes through."""

import pytest

from central_module.nlp.normalize import (
    command_normalize, light_normalize, normalize, words_to_digits,
)


class TestWordsToDigits:
    @pytest.mark.parametrize("tokens, expected", [
        (["fifty", "five"], ["55"]),
        (["twenty"], ["20"]),
        (["seven"], ["7"]),
        (["thirteen"], ["13"]),
        (["zero"], ["0"]),
        (["one", "hundred"], ["100"]),
        (["two", "hundred"], ["200"]),
        (["twice"], ["2"]),
        (["thrice"], ["3"]),
        (["half"], ["50"]),
    ])
    def test_number_words_become_digits(self, tokens, expected):
        assert words_to_digits(tokens) == expected

    def test_non_number_tokens_pass_through_in_order(self):
        assert words_to_digits(["scroll", "down", "three", "times"]) == \
            ["scroll", "down", "3", "times"]

    def test_tens_followed_by_non_unit_do_not_merge(self):
        assert words_to_digits(["twenty", "apples"]) == ["20", "apples"]

    def test_separate_numbers_stay_separate(self):
        assert words_to_digits(["five", "six"]) == ["5", "6"]

    def test_empty(self):
        assert words_to_digits([]) == []


class TestLightNormalize:
    def test_lowercases_and_strips_punctuation(self):
        assert light_normalize("  Open, CHROME!  ") == "open chrome"

    @pytest.mark.parametrize("raw", [
        "hey open chrome", "ok open chrome", "okay open chrome",
        "hey computer open chrome", "could you open chrome",
        "can you open chrome", "i want to open chrome",
        "hey system would you open chrome",
    ])
    def test_polite_prefixes_removed(self, raw):
        assert light_normalize(raw) == "open chrome"

    def test_keeps_payload_words_including_determiners(self):
        # The light form is what text slots are cut from — nothing may vanish.
        assert light_normalize("Type The report is due on Monday.") == \
            "type the report is due on monday"

    def test_inner_dots_and_hyphens_survive(self):
        assert light_normalize("edit chatter.txt") == "edit chatter.txt"
        assert light_normalize("switch to en-in") == "switch to en-in"

    def test_sentence_dots_and_hyphens_are_stripped(self):
        assert light_normalize("done. well -") == "done well"

    def test_apostrophes_survive(self):
        assert light_normalize("don't") == "don't"

    def test_whitespace_collapses(self):
        assert light_normalize("open \t\n  chrome") == "open chrome"

    def test_empty_and_punctuation_only(self):
        assert light_normalize("") == ""
        assert light_normalize("?!,") == ""


class TestCommandNormalize:
    def test_drops_determiners_and_fillers(self):
        assert command_normalize("open the calculator please") == "open calculator"

    def test_converts_numbers(self):
        assert command_normalize("set volume to fifty five") == "set volume to 55"

    def test_full_pipeline(self):
        light, cmd = normalize("Hey, could you open the calculator please?")
        assert light == "open the calculator please"
        assert cmd == "open calculator"

    def test_normalize_returns_both_forms(self):
        assert normalize("scroll down three times") == \
            ("scroll down three times", "scroll down 3 times")
