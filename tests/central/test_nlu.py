"""nlp/nlu/ — label spaces, the synthetic dataset, and the inference
post-processing. None of this needs torch; the model itself is covered by
test_nlu_ml.py (marker `ml`)."""

import json
import re

import pytest

from central_module import config
from central_module.nlp.intents import BY_NAME, CATALOG, KEY_ALIASES
from central_module.nlp.normalize import words_to_digits
from central_module.nlp.nlu import dataset as ds
from central_module.nlp.nlu.engine import NeuralNLU
from central_module.nlp.nlu.labels import (
    ID2INTENT, ID2SLOT, INTENT2ID, INTENT_LABELS, INTENT_SLOT_KINDS,
    REQUIRED_SLOT_KINDS, SLOT2ID, SLOT_KINDS, SLOT_LABELS,
)
from helpers.paths import NLU_DIR

_SLOT_RE = re.compile(r"\{(\w+)\}")


# ── Label spaces ───────────────────────────────────────────────────────────────

def test_intent_labels_are_catalog_plus_refusal():
    assert INTENT_LABELS == [i.name for i in CATALOG] + ["none"]


def test_slot_labels_are_bio_over_every_kind():
    kinds_in_catalog = {k for i in CATALOG for k in i.slots.values()}
    assert kinds_in_catalog <= set(SLOT_KINDS)
    assert SLOT_LABELS[0] == "O"
    assert len(SLOT_LABELS) == 1 + 2 * len(SLOT_KINDS)
    for kind in SLOT_KINDS:
        assert f"B-{kind}" in SLOT_LABELS and f"I-{kind}" in SLOT_LABELS


def test_id_maps_are_inverse():
    assert all(ID2INTENT[i] == name for name, i in INTENT2ID.items())
    assert all(ID2SLOT[i] == name for name, i in SLOT2ID.items())


@pytest.mark.parametrize("intent, required", [
    ("open_app", {"app"}), ("scroll", {"direction"}), ("volume_up", set()),
    ("set_volume", {"number"}), ("press_key", {"key"}), ("copy", set()),
    ("type_text", {"text"}), ("set_language", {"text"}),
])
def test_required_slots(intent, required):
    assert REQUIRED_SLOT_KINDS[intent] == required
    assert required <= INTENT_SLOT_KINDS[intent]


# ── Contract with the shipped model (these files are plain git, not LFS) ──────

@pytest.fixture(scope="module")
def shipped():
    with open(NLU_DIR / "nlu_config.json", encoding="utf-8") as fh:
        return json.load(fh)


def test_shipped_model_label_space_matches_the_catalog(shipped):
    """Adding, removing or renaming an intent without retraining would make the
    model's output indices mean different intents. Retrain, then commit the new
    Models/NLU artefacts together with the catalog change."""
    assert shipped["intent_labels"] == INTENT_LABELS
    assert shipped["slot_labels"] == SLOT_LABELS


def test_shipped_model_is_a_trusted_full_run(shipped):
    assert shipped["quick"] is False and shipped.get("partial") is False
    assert shipped["dev_intent_acc"] >= config.NLP["trusted_dev_acc"]


def test_shipped_model_directory_is_complete(shipped):
    for name in ("config.json", "tokenizer.json", "tokenizer_config.json",
                 "vocab.txt", "special_tokens_map.json", "joint_nlu.pt"):
        assert (NLU_DIR / name).exists(), name
    with open(NLU_DIR / "config.json", encoding="utf-8") as fh:
        assert json.load(fh)["model_type"] == "distilbert"
    assert shipped["max_len"] >= 16


# ── Dataset generator ──────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def sample():
    return ds.DatasetGenerator(seed=13).generate(per_intent=12)


def test_generation_is_deterministic():
    a = ds.DatasetGenerator(seed=7).generate(per_intent=3)
    b = ds.DatasetGenerator(seed=7).generate(per_intent=3)
    c = ds.DatasetGenerator(seed=8).generate(per_intent=3)
    assert [(e.words, e.tags, e.intent) for e in a] == [(e.words, e.tags, e.intent) for e in b]
    assert [e.words for e in a] != [e.words for e in c]


def test_every_intent_and_the_refusal_class_are_present(sample):
    intents = {e.intent for e in sample}
    assert intents == set(INTENT_LABELS)


def test_none_fraction(sample):
    n_none = sum(e.intent == "none" for e in sample)
    assert n_none == int(len(CATALOG) * 12 * 0.18 / 0.82)


def test_tags_align_and_are_valid(sample):
    for ex in sample:
        assert len(ex.words) == len(ex.tags), ex
        assert all(tag in SLOT2ID for tag in ex.tags), ex
        assert all(w for w in ex.words), ex


def test_tags_are_well_formed_bio(sample):
    for ex in sample:
        prev = "O"
        for tag in ex.tags:
            if tag.startswith("I-"):
                assert prev in (f"B-{tag[2:]}", tag), ex
            prev = tag


def test_refusals_carry_no_slots(sample):
    assert all(set(e.tags) == {"O"} for e in sample if e.intent == "none")


def test_slot_spans_only_use_kinds_the_intent_declares(sample):
    for ex in sample:
        if ex.intent == "none":
            continue
        kinds = {t[2:] for t in ex.tags if t != "O"}
        assert kinds <= INTENT_SLOT_KINDS[ex.intent], ex


def test_paraphrases_reference_real_intents_and_slots():
    for name, templates in ds.PARAPHRASES.items():
        assert name in BY_NAME, f"paraphrases for unknown intent {name!r}"
        for template in templates:
            for slot in _SLOT_RE.findall(template):
                assert slot in BY_NAME[name].slots, (name, template)


def test_refusal_templates_only_use_app_slots():
    for template in ds.NONE_TEMPLATES:
        assert set(_SLOT_RE.findall(template)) <= {"app"}, template


def test_key_bank_is_speakable():
    assert ds.KEYS and all(k in KEY_ALIASES and len(k) > 1 for k in ds.KEYS)


@pytest.mark.parametrize("n", range(0, 100))
def test_number_words_round_trip_through_the_normaliser(n):
    # The generator spells numbers one way; the runtime reads them back
    # another. They must agree or "sixty seven" trains as something the
    # engine can never parse.
    assert words_to_digits(ds._num_to_words(n).split()) == [str(n)]


def test_typo_never_empties_a_word():
    gen = ds.DatasetGenerator(seed=1)
    for word in ("open", "chrome", "volume", "ab", "x"):
        for _ in range(50):
            assert gen._typo(word)


# ── Inference post-processing (no model needed) ────────────────────────────────

class _Enc:
    def __init__(self, word_ids):
        self._ids = word_ids

    def word_ids(self, batch_index=0):
        return self._ids


@pytest.fixture
def nlu():
    obj = object.__new__(NeuralNLU)        # skip __init__: no weights needed
    obj._slot_labels = list(SLOT_LABELS)
    return obj


def tags(*names):
    return [SLOT2ID[n] for n in names]


def test_multiword_app_span(nlu):
    words = ["open", "vs", "code"]
    enc = _Enc([None, 0, 1, 2, None])
    got = nlu._extract_slots("open_app", words, enc, tags("O", "O", "B-app", "I-app", "O"))
    assert got == {"app": "vs code"}


def test_only_first_subword_votes(nlu):
    words = ["open", "photoshop"]
    enc = _Enc([None, 0, 1, 1, 1, None])      # photo ##sh ##op
    got = nlu._extract_slots("open_app", words, enc,
                             tags("O", "O", "B-app", "O", "O", "O"))
    assert got == {"app": "photoshop"}


def test_missing_required_slot_refuses(nlu):
    words = ["open", "it"]
    assert nlu._extract_slots("open_app", words, _Enc([None, 0, 1, None]),
                              tags("O", "O", "O", "O")) is None


def test_stray_span_for_a_slotless_intent_is_dropped(nlu):
    words = ["close", "window"]
    got = nlu._extract_slots("close_window", words, _Enc([None, 0, 1, None]),
                             tags("O", "O", "B-app", "O"))
    assert got == {}


def test_number_words_become_ints(nlu):
    words = ["volume", "at", "fifty", "five"]
    got = nlu._extract_slots("set_volume", words, _Enc([None, 0, 1, 2, 3, None]),
                             tags("O", "O", "O", "B-number", "I-number", "O"))
    assert got == {"number": 55}


def test_invalid_optional_slot_is_dropped_required_is_kept(nlu):
    words = ["scroll", "down", "lots"]
    got = nlu._extract_slots("scroll", words, _Enc([None, 0, 1, 2, None]),
                             tags("O", "O", "B-direction", "B-count", "O"))
    assert got == {"direction": "down"}


def test_first_span_per_kind_wins(nlu):
    words = ["open", "chrome", "and", "edge"]
    got = nlu._extract_slots("open_app", words, _Enc([None, 0, 1, 2, 3, None]),
                             tags("O", "O", "B-app", "O", "B-app", "O"))
    assert got == {"app": "chrome"}


def test_orphan_inside_tag_is_ignored(nlu):
    words = ["open", "chrome"]
    assert nlu._extract_slots("open_app", words, _Enc([None, 0, 1, None]),
                              tags("O", "O", "I-app", "O")) is None


@pytest.mark.parametrize("kind, value, expected", [
    ("number", "sixty seven", 67), ("count", "3", 3), ("number", "1000", None),
    ("number", "loud", None), ("direction", "up", "up"), ("direction", "north", None),
    ("key", "return", "enter"), ("key", "banana", None), ("app", "vs code", "vs code"),
])
def test_clean(kind, value, expected):
    assert NeuralNLU._clean(kind, value) == expected


def test_missing_weights_raise_file_not_found(tmp_path):
    with pytest.raises(FileNotFoundError, match="nlu.train"):
        NeuralNLU(model_dir=str(tmp_path))


def test_word_level_input_matches_training():
    # parse() tokenises light_normalize(text).split(); training used the
    # generator's word lists. Both must be lowercase, punctuation-free words.
    gen = ds.DatasetGenerator(seed=3).generate(per_intent=2)
    for ex in gen:
        for w in ex.words:
            assert w == w.lower() and w.strip() == w and w
