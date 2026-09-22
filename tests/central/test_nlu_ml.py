"""nlp/nlu/ model, training helpers and the full NeuralNLU load path.

Marker `ml`: needs torch + transformers (the CI `ml-tests` job installs the
CPU wheels). The 255 MB fine-tuned weights are NOT needed — a tiny, randomly
initialised DistilBERT is built on the fly with the real tokenizer (plain
git, not LFS) so load → quantise → parse is exercised end to end.

Marker `models` (last section): the real shipped weights, skipped when they
are Git-LFS pointer files.
"""

import json
import shutil

import pytest

from central_module import config
from central_module.nlp.nlu.dataset import Example
from central_module.nlp.nlu.labels import (
    INTENT2ID, INTENT_LABELS, SLOT2ID, SLOT_LABELS,
)
from helpers.paths import NLU_DIR, require_real_file

pytestmark = pytest.mark.ml
torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")

from central_module.nlp.nlu.model import JointNLU  # noqa: E402

TINY = dict(vocab_size=30522, dim=32, n_layers=2, n_heads=2, hidden_dim=64,
            max_position_embeddings=64)
TOKENIZER_FILES = ("tokenizer.json", "tokenizer_config.json", "vocab.txt",
                   "special_tokens_map.json")


def tiny_backbone():
    return transformers.DistilBertModel(transformers.DistilBertConfig(**TINY))


@pytest.fixture(scope="module")
def tokenizer():
    return transformers.AutoTokenizer.from_pretrained(str(NLU_DIR))


# ── JointNLU ───────────────────────────────────────────────────────────────────

def test_forward_shapes():
    model = JointNLU(tiny_backbone(), n_intents=len(INTENT_LABELS), n_slots=len(SLOT_LABELS))
    ids = torch.randint(0, 1000, (3, 7))
    intent, slots = model(ids, torch.ones_like(ids))
    assert intent.shape == (3, len(INTENT_LABELS))
    assert slots.shape == (3, 7, len(SLOT_LABELS))


def test_freeze_lower_layers():
    model = JointNLU(tiny_backbone(), 5, 3)
    model.freeze_lower(1)
    assert not any(p.requires_grad for p in model.backbone.embeddings.parameters())
    assert not any(p.requires_grad for p in model.backbone.transformer.layer[0].parameters())
    assert all(p.requires_grad for p in model.backbone.transformer.layer[1].parameters())
    assert all(p.requires_grad for p in model.intent_head.parameters())


# ── train.encode / evaluate ────────────────────────────────────────────────────

def test_encode_aligns_one_label_per_word(tokenizer):
    from central_module.nlp.nlu.train import encode
    ex = Example(["open", "photoshop", "please"], ["O", "B-app", "O"], "open_app")
    data = encode([ex], tokenizer, max_len=16)
    ids, mask, intent, slots = data[0]
    assert int(intent) == INTENT2ID["open_app"]
    labelled = [(tokenizer.convert_ids_to_tokens(int(i)), int(s))
                for i, s in zip(ids, slots) if int(s) != -100]
    # Exactly one supervised position per word: its first sub-token.
    assert [lab for _, lab in labelled] == [SLOT2ID["O"], SLOT2ID["B-app"], SLOT2ID["O"]]
    assert labelled[1][0] == tokenizer.tokenize("photoshop")[0]
    assert int(mask.sum()) == len(tokenizer(["open", "photoshop", "please"],
                                            is_split_into_words=True)["input_ids"])


def test_evaluate_returns_fractions(tokenizer):
    from torch.utils.data import DataLoader
    from central_module.nlp.nlu.train import encode, evaluate
    examples = [Example(["copy"], ["O"], "copy"),
                Example(["open", "chrome"], ["O", "B-app"], "open_app")]
    model = JointNLU(tiny_backbone(), len(INTENT_LABELS), len(SLOT_LABELS))
    acc, slot_acc = evaluate(model, DataLoader(encode(examples, tokenizer, 8), batch_size=2), "cpu")
    assert 0.0 <= acc <= 1.0 and 0.0 <= slot_acc <= 1.0
    assert model.training                     # evaluate() restores train mode


# ── NeuralNLU: full load path on a tiny artefact ───────────────────────────────

def build_artifact(path, *, force_intent=None, strength=10.0, quick=False, dev_acc=0.99):
    """Write a loadable model dir whose intent head always favours force_intent.

    Weights are zeroed and only the biases carry signal, so the output is
    fully determined: softmax gives force_intent e^s / (e^s + 49).
    """
    path.mkdir(parents=True, exist_ok=True)
    for name in TOKENIZER_FILES:
        shutil.copy(NLU_DIR / name, path / name)
    backbone = tiny_backbone()
    backbone.config.save_pretrained(str(path))
    model = JointNLU(backbone, len(INTENT_LABELS), len(SLOT_LABELS))
    with torch.no_grad():
        model.intent_head.weight.zero_()
        model.intent_head.bias.zero_()
        model.slot_head.weight.zero_()
        model.slot_head.bias.fill_(-10.0)
        model.slot_head.bias[SLOT2ID["O"]] = 10.0        # every token tagged O
        if force_intent:
            model.intent_head.bias[INTENT2ID[force_intent]] = strength
    torch.save(model.state_dict(), path / "joint_nlu.pt")
    (path / "nlu_config.json").write_text(json.dumps({
        "intent_labels": INTENT_LABELS, "slot_labels": SLOT_LABELS, "max_len": 32,
        "quick": quick, "dev_intent_acc": dev_acc,
    }), encoding="utf-8")
    return path


def test_loaded_model_answers_with_its_intent(tmp_path):
    from central_module.nlp.nlu.engine import NeuralNLU
    nlu = NeuralNLU(str(build_artifact(tmp_path / "m", force_intent="copy")))
    name, slots, conf = nlu.parse("please copy that")
    assert (name, slots) == ("copy", {})
    assert conf > 0.99


def test_refusal_class_returns_none(tmp_path):
    from central_module.nlp.nlu.engine import NeuralNLU
    nlu = NeuralNLU(str(build_artifact(tmp_path / "m", force_intent="none")))
    assert nlu.parse("what is the meaning of life") is None


def test_missing_required_slot_is_refused(tmp_path):
    from central_module.nlp.nlu.engine import NeuralNLU
    nlu = NeuralNLU(str(build_artifact(tmp_path / "m", force_intent="open_app")))
    assert nlu.parse("open chrome") is None          # all tags O → no {app}


def test_low_confidence_is_refused(tmp_path):
    from central_module.nlp.nlu.engine import NeuralNLU
    unsure = build_artifact(tmp_path / "m", force_intent="copy", strength=2.0)   # p ≈ 0.13
    assert NeuralNLU(str(unsure)).parse("copy") is None
    assert NeuralNLU(str(unsure), confidence=0.1).parse("copy")[0] == "copy"


def test_empty_input_is_refused(tmp_path):
    from central_module.nlp.nlu.engine import NeuralNLU
    nlu = NeuralNLU(str(build_artifact(tmp_path / "m", force_intent="copy")))
    assert nlu.parse("  ?! ") is None


@pytest.mark.parametrize("quick, dev_acc, trusted", [
    (False, 0.99, True), (True, 0.99, False), (False, 0.5, False),
])
def test_trust_follows_provenance(tmp_path, quick, dev_acc, trusted):
    from central_module.nlp.nlu.engine import NeuralNLU
    nlu = NeuralNLU(str(build_artifact(tmp_path / "m", quick=quick, dev_acc=dev_acc)))
    assert nlu.trusted is trusted


def test_intent_engine_uses_a_loaded_model(tmp_path, monkeypatch):
    from central_module.nlp.intent_engine import IntentEngine
    monkeypatch.setitem(config.NLP, "engine", "neural")
    model_dir = build_artifact(tmp_path / "m", force_intent="mute")
    monkeypatch.setitem(config.NLP, "model_dir", str(model_dir))
    engine = IntentEngine()
    assert engine.mode == "neural"
    assert engine.parse("open chrome").source == "exact"          # patterns first
    match = engine.parse("kill the sound")
    assert (match.intent.name, match.source) == ("mute", "model")


# ── The real shipped model (Git LFS) ───────────────────────────────────────────

# Multi-word app names come back truncated to one word. Two causes, measured:
# in fp32 "google chrome" is tagged B-app B-app (both words are standalone
# entries in the dataset's APPS bank, so the model treats them as two apps),
# and int8 quantisation additionally flips "file explorer"'s I-app, which fp32
# gets right. The exact patterns catch the literal phrasings first, so this
# bites on paraphrases only. Strict: a retrained model that fixes it fails
# here until the marker is removed.
_MULTIWORD_APPS = pytest.mark.xfail(
    strict=True, reason="shipped model truncates multi-word app spans (see comment)")


@pytest.mark.models
@pytest.mark.parametrize("text, intent, slots", [
    ("make it louder", "volume_up", {}),
    ("fire up spotify", "open_app", {"app": "spotify"}),
    ("put the volume at forty", "set_volume", {"number": 40}),
    ("take me to github.com", "open_url", {"text": "github.com"}),
    ("type out hello there", "type_text", {"text": "hello there"}),
    ("press the enter key", "press_key", {"key": "enter"}),
    ("double click", "double_click", {}),       # the rule ladder alone gets this wrong
    ("scroll down a bit", "scroll", {"direction": "down"}),
    pytest.param("launch google chrome", "open_app", {"app": "google chrome"},
                 marks=_MULTIWORD_APPS),
    pytest.param("bring up file explorer", "open_app", {"app": "file explorer"},
                 marks=_MULTIWORD_APPS),
])
def test_shipped_model_understands_paraphrases(real_nlu, text, intent, slots):
    name, got_slots, conf = real_nlu.parse(text)
    assert (name, got_slots) == (intent, slots)
    assert conf >= config.NLP["intent_confidence"]


@pytest.mark.models
@pytest.mark.parametrize("text", [
    "what is the meaning of life", "delete chatter.txt", "the weather is nice today",
])
def test_shipped_model_refuses_non_commands(real_nlu, text):
    assert real_nlu.parse(text) is None


@pytest.fixture(scope="module")
def real_nlu():
    require_real_file(NLU_DIR / "joint_nlu.pt")
    from central_module.nlp.nlu.engine import NeuralNLU
    nlu = NeuralNLU(str(NLU_DIR))
    assert nlu.trusted
    return nlu
