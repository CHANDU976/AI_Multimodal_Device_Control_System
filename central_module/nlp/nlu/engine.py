"""
engine.py
Inference wrapper around the fine-tuned joint model.

    nlu = NeuralNLU()               # loads Models/NLU/joint-distilbert
    nlu.parse("make it louder")     # → ("volume_up", {}, 0.97)
    nlu.parse("what time is it")    # → None   (refused: none class / low conf)

Loading takes a few seconds (done once at startup); a parse is ~15-40 ms on
CPU thanks to dynamic int8 quantization of the linear layers. Everything is
read from the model directory — no internet, matching the rest of the stack.
"""

from __future__ import annotations

import json
import logging
import os

from central_module.config import NLP
from central_module.nlp.intents import DIRECTIONS, KEY_ALIASES
from central_module.nlp.normalize import light_normalize, words_to_digits
from central_module.nlp.nlu.labels import INTENT_SLOT_KINDS, REQUIRED_SLOT_KINDS

logger = logging.getLogger(__name__)


class NeuralNLU:

    def __init__(self, model_dir: str | None = None,
                 confidence: float | None = None) -> None:
        self._dir = model_dir or NLP["model_dir"]
        self._threshold = (NLP["intent_confidence"]
                           if confidence is None else confidence)
        if not os.path.isfile(os.path.join(self._dir, "joint_nlu.pt")):
            raise FileNotFoundError(
                f"no trained NLU model in {self._dir} — run "
                f"`python -m central_module.nlp.nlu.train` first")

        import torch
        from transformers import AutoTokenizer, DistilBertConfig, DistilBertModel
        from central_module.nlp.nlu.model import JointNLU

        with open(os.path.join(self._dir, "nlu_config.json"),
                  encoding="utf-8") as fh:
            cfg = json.load(fh)
        self._intent_labels: list[str] = cfg["intent_labels"]
        self._slot_labels: list[str] = cfg["slot_labels"]
        self._max_len: int = cfg["max_len"]

        # A model is "trusted" only if it was properly trained. An untrusted
        # one (a --quick sanity artifact, or a run that never converged)
        # still answers, but its *refusals* are not authoritative: the ladder
        # keeps the fuzzy fallback alive behind it, so a smoke-test model
        # can never leave the system less capable than the rules alone.
        self.dev_acc: float = float(cfg.get("dev_intent_acc", 0.0))
        self.trusted: bool = (not cfg.get("quick", False)
                              and self.dev_acc >= NLP["trusted_dev_acc"])

        self._tokenizer = AutoTokenizer.from_pretrained(self._dir)
        backbone = DistilBertModel(DistilBertConfig.from_pretrained(self._dir))
        model = JointNLU(backbone, len(self._intent_labels),
                         len(self._slot_labels))
        state = torch.load(os.path.join(self._dir, "joint_nlu.pt"),
                           map_location="cpu")
        model.load_state_dict(state)
        model.eval()
        # int8 linears ≈ 2-3× faster on CPU, indistinguishable accuracy here.
        self._model = torch.ao.quantization.quantize_dynamic(
            model, {torch.nn.Linear}, dtype=torch.qint8)
        self._torch = torch
        logger.info("NeuralNLU loaded from %s (%d intents, dev acc %.3f%s)",
                    self._dir, len(self._intent_labels), self.dev_acc,
                    "" if self.trusted else " — UNTRUSTED, rules stay in front")
        if not self.trusted:
            logger.warning(
                "the NLU model in %s is a sanity artifact or undertrained "
                "(dev acc %.3f). Run `python -m central_module.nlp.nlu.train` "
                "for the real one.", self._dir, self.dev_acc)

    # ── Inference ──────────────────────────────────────────────────────────────

    def parse(self, text: str) -> tuple[str, dict, float] | None:
        """→ (intent_name, slots, confidence), or None to refuse."""
        words = light_normalize(text).split()
        if not words:
            return None

        enc = self._tokenizer([words], is_split_into_words=True,
                              truncation=True, max_length=self._max_len,
                              return_tensors="pt")
        with self._torch.no_grad():
            intent_logits, slot_logits = self._model(
                enc["input_ids"], enc["attention_mask"])

        probs = intent_logits.softmax(-1)[0]
        conf, intent_id = probs.max(0)
        conf = float(conf)
        intent = self._intent_labels[int(intent_id)]
        if intent == "none" or conf < self._threshold:
            return None

        slots = self._extract_slots(intent, words, enc,
                                    slot_logits[0].argmax(-1).tolist())
        if slots is None:               # a mandatory slot is missing/invalid
            return None
        return intent, slots, conf

    def _extract_slots(self, intent: str, words: list[str],
                       enc, tag_ids: list[int]) -> dict | None:
        # First-subword predictions → word-level tags.
        word_tags: list[str] = ["O"] * len(words)
        prev = None
        for pos, word_id in enumerate(enc.word_ids(batch_index=0)):
            if word_id is None or word_id == prev:
                continue
            word_tags[word_id] = self._slot_labels[tag_ids[pos]]
            prev = word_id

        # BIO decode: first span per kind wins.
        spans: dict[str, list[str]] = {}
        current: str | None = None
        for word, tag in zip(words, word_tags):
            if tag.startswith("B-"):
                kind = tag[2:]
                if kind not in spans:
                    spans[kind] = [word]
                    current = kind
                else:
                    current = None
            elif tag.startswith("I-") and current == tag[2:]:
                spans[current].append(word)
            else:
                current = None

        allowed = INTENT_SLOT_KINDS.get(intent, set())
        slots: dict = {}
        for kind, span in spans.items():
            if kind not in allowed:
                continue                # stray span for this intent — drop
            value = self._clean(kind, " ".join(span))
            if value is not None:
                slots[kind] = value

        required = REQUIRED_SLOT_KINDS.get(intent, set())
        if not required <= set(slots):
            logger.debug("refusing %s: required slots %s, got %s",
                         intent, required, set(slots))
            return None
        return slots

    @staticmethod
    def _clean(kind: str, value: str):
        """Canonicalize a span; None = unusable."""
        if kind in ("number", "count"):
            digits = " ".join(words_to_digits(value.split()))
            if not digits.isdigit():
                return None
            n = int(digits)
            return n if 0 <= n <= 999 else None
        if kind == "direction":
            return value if value in DIRECTIONS else None
        if kind == "key":
            return KEY_ALIASES.get(value)
        return value                     # app / text


# ── Smoke-test (needs a trained model) ────────────────────────────────────────

if __name__ == "__main__":
    import sys
    logging.basicConfig(level=logging.INFO, stream=sys.stderr)
    nlu = NeuralNLU()
    for line in ("open chrome", "make it louder", "increase volume by 5",
                 "opun vs code", "edit note.txt", "delete chatter.txt",
                 "caps", "type hello there how are you",
                 "what is the meaning of life"):
        print(f"  {line!r:40s} → {nlu.parse(line)}")
