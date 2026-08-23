"""
dataset.py
Synthetic training data for the joint intent + slot model.

Every example starts from a template — the catalog patterns plus a bank of
natural paraphrases the rule engine could never cover ("fire up {app}",
"make it louder") — then gets slot values filled from realistic banks and
passes through an augmentation gauntlet:

    politeness      "hey can you … please"
    determiners     "open the calculator"
    number forms    67 ⇄ "sixty seven"
    typos           swap / drop / double / neighbour-key, word joins
    ASR noise       lowercase, no punctuation — like VOSK/Whisper output

A "none" class (questions, prose, chatter, and *near-miss* commands like
"delete chatter.txt") teaches the model to refuse instead of guess — the
exact failure mode rule-based fuzzy matching had.

Output: Example(words, tags, intent) with word-level BIO tags, ready for
wordpiece alignment in train.py. Pure stdlib — generation needs no torch.
"""

from __future__ import annotations

import random
import re
from dataclasses import dataclass

from central_module.nlp.intents import CATALOG, KEY_ALIASES

# ── Slot value banks ───────────────────────────────────────────────────────────

APPS = [
    "chrome", "google chrome", "firefox", "edge", "brave", "safari",
    "notepad", "word", "excel", "powerpoint", "outlook", "onenote",
    "spotify", "vlc", "discord", "slack", "zoom", "teams", "whatsapp",
    "telegram", "calculator", "terminal", "powershell", "command prompt",
    "file explorer", "explorer", "settings", "control panel", "task manager",
    "paint", "photoshop", "blender", "steam", "epic games", "valorant",
    "minecraft", "elden ring", "visual studio", "visual studio code",
    "vs code", "code", "pycharm", "intellij", "android studio", "obs",
    "obsidian", "notion", "figma", "postman", "docker", "youtube", "google",
    "gmail", "github", "netflix", "camera", "clock", "calendar", "maps",
]
FILES = [
    "report.docx", "notes.txt", "chatter.txt", "meow.tex", "resume.pdf",
    "main.py", "index.html", "budget.xlsx", "my-notes.md", "todo.md",
    "song.mp3", "video.mp4", "photo.png", "slides.pptx", "data.csv",
    "readme.md", "config.json", "assignment.pdf", "thesis.tex", "demo.mp4",
]
SEARCH_QUERIES = [
    "weather in bangalore", "how to make pasta", "python list comprehension",
    "best laptops under 50000", "capstone report format", "news today",
    "distilbert paper", "flight to delhi", "movie showtimes near me",
    "何 is this",  # unicode survives the pipeline
    "speech recognition tutorial", "cricket score", "bus timings",
    "cheap flights to goa", "resume templates", "gesture recognition datasets",
]
TYPE_SENTENCES = [
    "the report is due on monday", "hello there", "i will be late today",
    "meeting moved to 3 pm", "thanks for the update", "see you tomorrow",
    "please review the attached file", "lunch at noon works for me",
    "the demo went well", "call me when you are free",
]
URLS = [
    "youtube.com", "github.com", "google.com", "wikipedia.org",
    "stackoverflow.com", "reddit.com", "amazon.in", "gmail.com",
    "canvas", "college portal", "my drive",
]
DIRECTIONS = ["up", "down", "left", "right", "top", "bottom"]
KEYS = [k for k in KEY_ALIASES if len(k) > 1]     # spoken forms, not letters

# ── Extra natural paraphrases the strict catalog does not cover ────────────────
# {slot} placeholders use the same kinds as the catalog intent declares.

PARAPHRASES: dict[str, list[str]] = {
    "open_app": [
        "open up {app}", "fire up {app}", "bring up {app}", "pull up {app}",
        "can you open {app}", "get {app} open", "open {app} for me",
        "launch the {app} app", "i need {app}", "switch on {app}",
    ],
    "search_web": [
        "search the web for {text}", "find {text}", "look for {text}",
        "search google for {text}", "what does the internet say about {text}",
        "google it {text}",
    ],
    "open_url": ["take me to {text}", "browse to {text}", "open the site {text}"],
    "type_text": ["type out {text}", "enter {text}", "write down {text}",
                  "type the words {text}"],
    "press_key": ["tap {key}", "press the {key} key", "hit the {key} key"],
    "copy": ["copy it", "copy the selection", "copy selected text"],
    "paste": ["paste that", "paste the clipboard", "paste it here"],
    "cut": ["cut it", "cut the selection"],
    "select_all": ["select the whole thing", "highlight everything"],
    "undo": ["undo the last change", "go back one step", "revert that"],
    "redo": ["redo the last change", "do that again"],
    "save": ["save it", "save the document", "save my work", "save changes"],
    "delete": ["delete it", "remove the selection", "erase that"],
    "scroll": ["scroll {direction} a bit", "move {direction}",
               "go {direction} a little", "scroll {direction} {count} steps",
               "keep scrolling {direction}"],
    "click": ["single click", "click the mouse", "click there"],
    "double_click": ["double click there", "open it with double click"],
    "right_click": ["right click there", "open the right click menu"],
    "volume_up": ["make it louder", "crank up the volume", "turn it up",
                  "increase the sound", "raise the sound a bit",
                  "volume higher", "pump up the volume", "a bit louder please"],
    "volume_down": ["make it quieter", "turn it down", "lower the sound",
                    "decrease the sound", "bring the volume down",
                    "volume lower", "too loud turn it down"],
    "mute": ["mute it", "mute the sound", "kill the sound", "shut the sound off"],
    "unmute": ["unmute it", "turn the sound back on", "bring the sound back"],
    "set_volume": ["put the volume at {number}", "volume at {number} percent",
                   "make the volume {number}", "set the sound to {number}",
                   "change volume to {number}"],
    "play_pause": ["play the music", "pause the video", "resume playback",
                   "pause it", "play it", "toggle playback"],
    "next_track": ["play the next one", "skip this song", "next please",
                   "change the song"],
    "prev_track": ["play the previous one", "go back a song",
                   "play that song again"],
    "minimize": ["minimize it", "put this window away", "hide this window"],
    "maximize": ["maximize it", "make the window bigger", "make it full screen"],
    "close_window": ["close it", "get rid of this window", "shut this window"],
    "switch_window": ["go to the other window", "alt tab", "cycle windows",
                      "show the next window"],
    "show_desktop": ["take me to the desktop", "minimize everything"],
    "new_tab": ["open a new tab", "give me a fresh tab", "another tab"],
    "close_tab": ["close the current tab", "get rid of this tab"],
    "next_tab": ["go to the next tab", "move to the next tab"],
    "prev_tab": ["go to the previous tab", "back one tab"],
    "browser_back": ["go back a page", "previous page", "take me back"],
    "browser_forward": ["go forward a page", "next page"],
    "refresh": ["refresh it", "reload this page", "refresh the browser"],
    "screenshot": ["grab a screenshot", "capture the screen for me",
                   "take a picture of the screen", "screen capture"],
    "lock": ["lock it", "lock the screen please", "lock this machine"],
    "brightness_up": ["make the screen brighter", "turn up the brightness",
                      "screen too dark make it brighter"],
    "brightness_down": ["make the screen dimmer", "turn down the brightness",
                        "the screen is too bright"],
    "sleep": ["put the pc to sleep", "send the computer to sleep"],
    "shutdown": ["turn the computer off", "power down the machine",
                 "switch off the computer"],
    "restart": ["restart the machine", "reboot the system",
                "restart my computer please"],
    "set_language": ["switch to {text}", "talk in {text}",
                     "understand {text} now", "listen in {text}"],
    "start_dictation": ["let me dictate", "take dictation", "typing mode on",
                        "write what i say"],
    "stop_dictation": ["stop taking dictation", "typing mode off",
                       "done dictating"],
    "confirm": ["yes please", "yeah do it", "confirmed", "go for it"],
    "cancel": ["no don't", "forget it", "leave it", "no thanks"],
}

LANG_NAMES = ["hindi", "english", "american english", "indian english"]

# Prose / chatter / questions → the "none" class.
NONE_SENTENCES = [
    "what is the meaning of life", "how are you doing today",
    "the weather is really nice outside", "i think we should leave early",
    "did you watch the match yesterday", "my professor extended the deadline",
    "this is taking longer than expected", "what time is the meeting",
    "the pasta was delicious", "he said he would call back",
    "when does the semester end", "that movie was pretty good",
    "i am not sure about this", "the traffic today was terrible",
    "where did i keep my keys", "why is the sky blue",
    "tell me a joke", "sing me a song", "who won the world cup",
    "the presentation went okay i guess", "we need more coffee",
    "remind me about nothing in particular", "it might rain tomorrow",
    "this room is too cold", "the neighbours are loud again",
    "let me think about it", "maybe later", "not right now",
    "hmm interesting", "okay cool", "wait what", "never mind that thing",
    "hello hello testing", "one two three testing", "caps", "meow",
    "the quick brown fox jumps over the lazy dog",
]
# Near-miss commands that MUST be refused: command-like verbs with objects we
# have no capability for. Teaches "delete <file>" ≠ the delete key.
NONE_TEMPLATES = [
    "delete {app}", "remove {app}", "uninstall {app}", "install {app}",
    "rename {app}", "move {app} to desktop", "compress {app}",
    "email {app} to me", "print {app}", "download {app}",
    "what is {app}", "is {app} open", "update {app}",
]

_POLITE_PRE = ["please ", "can you ", "could you ", "hey ", "okay ",
               "can you please ", "i want to ", "would you "]
_POLITE_POST = [" please", " now", " for me", " thanks", " quickly"]

_KEYBOARD_NEIGHBOURS = {
    "a": "sq", "b": "vn", "c": "xv", "d": "sf", "e": "wr", "f": "dg",
    "g": "fh", "h": "gj", "i": "uo", "j": "hk", "k": "jl", "l": "k",
    "m": "n", "n": "bm", "o": "ip", "p": "o", "q": "wa", "r": "et",
    "s": "ad", "t": "ry", "u": "yi", "v": "cb", "w": "qe", "x": "zc",
    "y": "tu", "z": "x",
}

_UNITS = ["zero", "one", "two", "three", "four", "five", "six", "seven",
          "eight", "nine"]
_TEENS = ["ten", "eleven", "twelve", "thirteen", "fourteen", "fifteen",
          "sixteen", "seventeen", "eighteen", "nineteen"]
_TENS = {2: "twenty", 3: "thirty", 4: "forty", 5: "fifty", 6: "sixty",
         7: "seventy", 8: "eighty", 9: "ninety"}


def _num_to_words(n: int) -> str:
    if n < 10:
        return _UNITS[n]
    if n < 20:
        return _TEENS[n - 10]
    if n < 100:
        tens, unit = divmod(n, 10)
        return _TENS[tens] + (f" {_UNITS[unit]}" if unit else "")
    return str(n)


@dataclass
class Example:
    words: list[str]
    tags: list[str]                 # BIO, aligned 1:1 with words
    intent: str


# ── Generator ──────────────────────────────────────────────────────────────────

class DatasetGenerator:

    def __init__(self, seed: int = 13) -> None:
        self._rng = random.Random(seed)
        self._templates: dict[str, list[tuple[str, dict]]] = {}
        for intent in CATALOG:
            entries = [(p, intent.slots) for p in intent.patterns]
            entries += [(p, intent.slots)
                        for p in PARAPHRASES.get(intent.name, [])]
            self._templates[intent.name] = entries

    # ── Slot filling ───────────────────────────────────────────────────────────

    def _slot_value(self, intent: str, kind: str) -> str:
        rng = self._rng
        if kind == "app":
            return rng.choice(APPS if rng.random() < 0.7 else FILES)
        if kind == "text":
            if intent == "type_text":
                return rng.choice(TYPE_SENTENCES)
            if intent == "open_url":
                return rng.choice(URLS)
            if intent == "set_language":
                return rng.choice(LANG_NAMES)
            return rng.choice(SEARCH_QUERIES)
        if kind in ("number", "count"):
            n = rng.randint(1, 100) if kind == "number" else rng.randint(1, 10)
            return str(n) if rng.random() < 0.5 else _num_to_words(n)
        if kind == "direction":
            return rng.choice(DIRECTIONS)
        if kind == "key":
            return rng.choice(KEYS)
        raise ValueError(kind)

    def _fill(self, intent: str, template: str, slots: dict) -> Example:
        words: list[str] = []
        tags: list[str] = []
        for token in template.split():
            m = re.fullmatch(r"\{(\w+)\}", token)
            if not m:
                words.append(token)
                tags.append("O")
                continue
            kind = slots[m.group(1)]
            value_words = self._slot_value(intent, kind).split()
            for i, w in enumerate(value_words):
                words.append(w)
                tags.append(("B-" if i == 0 else "I-") + kind)
        return Example(words, tags, intent)

    # ── Augmentation ───────────────────────────────────────────────────────────

    def _typo(self, word: str) -> str:
        rng = self._rng
        if len(word) < 3:
            return word
        i = rng.randrange(len(word) - 1)
        op = rng.random()
        if op < 0.3:                                     # swap
            return word[:i] + word[i + 1] + word[i] + word[i + 2:]
        if op < 0.55:                                    # drop
            return word[:i] + word[i + 1:]
        if op < 0.8:                                     # double
            return word[:i] + word[i] + word[i:]
        near = _KEYBOARD_NEIGHBOURS.get(word[i])         # neighbour key
        return word[:i] + rng.choice(near) + word[i + 1:] if near else word

    def _augment(self, ex: Example) -> Example:
        rng = self._rng
        words, tags = list(ex.words), list(ex.tags)

        if rng.random() < 0.35:                          # politeness
            pre = rng.choice(_POLITE_PRE).split()
            words = pre + words
            tags = ["O"] * len(pre) + tags
        if rng.random() < 0.20:
            post = rng.choice(_POLITE_POST).split()
            words = words + post
            tags = tags + ["O"] * len(post)

        if rng.random() < 0.15:                          # determiner
            for i, (w, t) in enumerate(zip(words, tags)):
                if t == "B-app":
                    words.insert(i, "the")
                    tags.insert(i, "O")
                    break

        if rng.random() < 0.35:                          # typos, 1-2 words
            for _ in range(rng.choice((1, 1, 2))):
                i = rng.randrange(len(words))
                # Corrupt literals freely, slot words occasionally: the model
                # should still find a misspelled span, and the resolution
                # layer downstream handles matching it to a real app.
                if tags[i] == "O" or rng.random() < 0.3:
                    words[i] = self._typo(words[i])

        if rng.random() < 0.08 and len(words) >= 2:      # join two O-words
            for i in range(len(words) - 1):
                if tags[i] == "O" and tags[i + 1] == "O":
                    words[i:i + 2] = [words[i] + words[i + 1]]
                    tags[i:i + 2] = ["O"]
                    break

        return Example(words, tags, ex.intent)

    # ── None class ─────────────────────────────────────────────────────────────

    def _none_example(self) -> Example:
        rng = self._rng
        if rng.random() < 0.35:
            template = rng.choice(NONE_TEMPLATES)
            ex = self._fill("none", template, {"app": "app"})
            # Refusals carry no slots — the span must not leak out.
            return Example(ex.words, ["O"] * len(ex.words), "none")
        words = rng.choice(NONE_SENTENCES).split()
        return Example(words, ["O"] * len(words), "none")

    # ── Public API ─────────────────────────────────────────────────────────────

    def generate(self, per_intent: int = 600,
                 none_fraction: float = 0.18) -> list[Example]:
        examples: list[Example] = []
        for intent, templates in self._templates.items():
            for _ in range(per_intent):
                template, slots = self._rng.choice(templates)
                ex = self._fill(intent, template, slots)
                examples.append(self._augment(ex))
        n_none = int(len(examples) * none_fraction / (1 - none_fraction))
        for _ in range(n_none):
            examples.append(self._augment(self._none_example()))
        self._rng.shuffle(examples)
        return examples


# ── Smoke-test ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    gen = DatasetGenerator()
    data = gen.generate(per_intent=3)
    intents = {e.intent for e in data}
    print(f"{len(data)} examples, {len(intents)} intent classes")
    for ex in data[:12]:
        pairs = " ".join(f"{w}[{t}]" if t != "O" else w
                         for w, t in zip(ex.words, ex.tags))
        print(f"  {ex.intent:16s} {pairs}")
    assert all(len(e.words) == len(e.tags) for e in data)
    assert "none" in intents
    print("dataset smoke-test passed.")
