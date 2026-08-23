"""
labels.py
Label spaces for the joint model, derived from the intent catalog so the
model and the rule engine can never disagree about what exists.

    intent labels : every catalog intent + "none" (the refusal class)
    slot labels   : BIO tags over the catalog's slot kinds
"""

from __future__ import annotations

from central_module.nlp.intents import CATALOG

SLOT_KINDS = ("app", "count", "direction", "key", "number", "text")

INTENT_LABELS: list[str] = [i.name for i in CATALOG] + ["none"]
SLOT_LABELS: list[str] = ["O"] + [f"{p}-{k}" for k in SLOT_KINDS
                                  for p in ("B", "I")]

INTENT2ID = {name: i for i, name in enumerate(INTENT_LABELS)}
SLOT2ID = {name: i for i, name in enumerate(SLOT_LABELS)}
ID2INTENT = dict(enumerate(INTENT_LABELS))
ID2SLOT = dict(enumerate(SLOT_LABELS))

# Which slot kinds each intent may legally carry — used to discard stray
# spans at inference ("close window" must never yield an {app}).
INTENT_SLOT_KINDS: dict[str, set] = {
    i.name: set(i.slots.values()) for i in CATALOG
}


def _required_kinds(intent) -> set:
    """Slot kinds present in *every* pattern = mandatory at inference.
    (scroll's {count} is optional — it only appears in some patterns —
    but open_app without an {app} span is not a usable prediction.)"""
    import re
    per_pattern = []
    for pattern in intent.patterns:
        names = re.findall(r"\{(\w+)\}", pattern)
        per_pattern.append({intent.slots[n] for n in names})
    return set.intersection(*per_pattern) if per_pattern else set()


REQUIRED_SLOT_KINDS: dict[str, set] = {
    i.name: _required_kinds(i) for i in CATALOG
}
