"""
normalize.py
Text normalization for intent matching.

Two levels, because commands and payloads want different treatment:

  light_normalize   lowercase, punctuation stripped, spaces collapsed,
                    polite prefixes removed. Safe for slot payloads —
                    "type The report is due Monday" keeps every word.

  command_normalize light + determiners ("the/a/an") and courtesy fillers
                    dropped, number words → digits. "could you open the
                    calculator please" → "open calculator";
                    "set volume to fifty five" → "set volume to 55".

The engine matches text-payload patterns against the light form and
everything else against the command form.
"""

from __future__ import annotations

import re

# Courtesy words safe to drop anywhere in a command.
_FILLERS = {"please", "kindly"}
_DETERMINERS = {"the", "a", "an"}

# Leading politeness that carries no meaning: "hey can you open chrome".
_PREFIX_RE = re.compile(
    r"^(?:hey |ok |okay )?(?:computer |system )?"
    r"(?:can you |could you |would you |will you |i want to |i want you to )?",
)

_PUNCT_RE = re.compile(r"[^\w\s'.\-]")   # keep apostrophes: "don't"
# Dots/hyphens survive only *inside* a word ("chatter.txt", "en-in");
# sentence punctuation ("done.", "well -") is stripped like the rest.
_STRAY_RE = re.compile(r"(?<!\w)[.\-]|[.\-](?!\w)")
_SPACE_RE = re.compile(r"\s+")

# ── Number words ───────────────────────────────────────────────────────────────

_UNITS = {"zero": 0, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
          "six": 6, "seven": 7, "eight": 8, "nine": 9}
_TEENS = {"ten": 10, "eleven": 11, "twelve": 12, "thirteen": 13, "fourteen": 14,
          "fifteen": 15, "sixteen": 16, "seventeen": 17, "eighteen": 18,
          "nineteen": 19}
_TENS = {"twenty": 20, "thirty": 30, "forty": 40, "fifty": 50, "sixty": 60,
         "seventy": 70, "eighty": 80, "ninety": 90}
_MULTIPLIERS = {"once": 1, "twice": 2, "thrice": 3, "double": 2, "half": 50}


def words_to_digits(tokens: list[str]) -> list[str]:
    """
    Replace number-word runs with digit tokens: ["fifty", "five"] → ["55"],
    ["one", "hundred"] → ["100"]. Non-number tokens pass through untouched.
    """
    out: list[str] = []
    i = 0
    while i < len(tokens):
        tok = tokens[i]
        if tok in _TENS:
            value = _TENS[tok]
            if i + 1 < len(tokens) and tokens[i + 1] in _UNITS:
                value += _UNITS[tokens[i + 1]]
                i += 1
        elif tok in _TEENS:
            value = _TEENS[tok]
        elif tok in _UNITS:
            value = _UNITS[tok]
        elif tok in _MULTIPLIERS:
            value = _MULTIPLIERS[tok]
        else:
            out.append(tok)
            i += 1
            continue
        # "<n> hundred" composes; "a hundred" was reduced to "hundred" upstream.
        if i + 1 < len(tokens) and tokens[i + 1] == "hundred":
            value *= 100
            i += 1
        out.append(str(value))
        i += 1
    return out


# ── Public API ─────────────────────────────────────────────────────────────────

def light_normalize(text: str) -> str:
    text = text.lower().strip()
    text = _PUNCT_RE.sub(" ", text)
    text = _STRAY_RE.sub(" ", text)
    text = _SPACE_RE.sub(" ", text).strip()
    text = _PREFIX_RE.sub("", text, count=1).strip()
    return text


def command_normalize(light: str) -> str:
    """light form → command form. Pass output of light_normalize()."""
    tokens = [t for t in light.split()
              if t not in _FILLERS and t not in _DETERMINERS]
    tokens = words_to_digits(tokens)
    return " ".join(tokens)


def normalize(text: str) -> tuple[str, str]:
    """Convenience: raw utterance → (light, command) forms."""
    light = light_normalize(text)
    return light, command_normalize(light)


# ── Smoke-test ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    cases = [
        "Hey, could you open the calculator please?",
        "set volume to fifty five",
        "scroll down three times",
        "Type The report is due on Monday.",
        "turn the volume up",
    ]
    for c in cases:
        light, cmd = normalize(c)
        print(f"  {c!r}\n    light: {light!r}\n    cmd:   {cmd!r}")
