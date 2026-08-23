"""
train.py
Fine-tunes DistilBERT into the joint intent + slot model.

    # full training (CPU ~30-60 min, downloads distilbert-base-uncased once)
    python -m central_module.nlp.nlu.train

    # 3-minute sanity run (tiny data, 1 epoch) — checks the pipeline works
    python -m central_module.nlp.nlu.train --quick

Everything is saved to Models/NLU/joint-distilbert: tokenizer, backbone
config and fine-tuned weights — inference (engine.py) needs no internet.

Loss = CE(intent) + CE(slot tags); sub-word continuation positions are
masked out (-100) so each *word* votes exactly once.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from central_module.nlp.nlu.dataset import DatasetGenerator, Example
from central_module.nlp.nlu.labels import (
    ID2INTENT, INTENT2ID, INTENT_LABELS, SLOT2ID, SLOT_LABELS,
)

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__)))))
DEFAULT_OUT = os.path.join(_PROJECT_ROOT, "Models", "NLU", "joint-distilbert")
BASE_MODEL = "distilbert-base-uncased"


def encode(examples: list[Example], tokenizer, max_len: int):
    enc = tokenizer([e.words for e in examples], is_split_into_words=True,
                    truncation=True, max_length=max_len,
                    padding="max_length", return_tensors="pt")
    slot_labels = torch.full(enc["input_ids"].shape, -100, dtype=torch.long)
    for i, ex in enumerate(examples):
        prev = None
        for pos, word_id in enumerate(enc.word_ids(batch_index=i)):
            if word_id is None or word_id == prev:
                continue                    # special token / continuation piece
            slot_labels[i, pos] = SLOT2ID[ex.tags[word_id]]
            prev = word_id
    intents = torch.tensor([INTENT2ID[e.intent] for e in examples])
    return TensorDataset(enc["input_ids"], enc["attention_mask"],
                         intents, slot_labels)


def evaluate(model, loader, device) -> tuple[float, float]:
    model.eval()
    intent_hits = intent_total = slot_hits = slot_total = 0
    with torch.no_grad():
        for ids, mask, intents, slots in loader:
            intent_logits, slot_logits = model(ids.to(device), mask.to(device))
            intent_hits += (intent_logits.argmax(-1).cpu() == intents).sum().item()
            intent_total += len(intents)
            pred = slot_logits.argmax(-1).cpu()
            keep = slots != -100
            slot_hits += (pred[keep] == slots[keep]).sum().item()
            slot_total += keep.sum().item()
    model.train()
    return intent_hits / intent_total, slot_hits / max(slot_total, 1)


def save(model, tokenizer, args, acc: float, slot_acc: float,
         partial: bool = False) -> None:
    """Write a self-contained, offline-loadable model directory."""
    os.makedirs(args.out, exist_ok=True)
    tokenizer.save_pretrained(args.out)
    model.backbone.config.save_pretrained(args.out)
    torch.save(model.state_dict(), os.path.join(args.out, "joint_nlu.pt"))
    with open(os.path.join(args.out, "nlu_config.json"), "w",
              encoding="utf-8") as fh:
        json.dump({"intent_labels": INTENT_LABELS, "slot_labels": SLOT_LABELS,
                   "max_len": args.max_len, "base": args.base,
                   # Quality provenance. engine.py will not let an
                   # undertrained model (a --quick sanity artifact) veto the
                   # rule fallback, so a 90-second smoke run can never
                   # silently degrade the whole system.
                   "quick": bool(args.quick) or partial,
                   "dev_intent_acc": round(acc, 4),
                   "dev_slot_acc": round(slot_acc, 4),
                   "per_intent": args.per_intent,
                   "epochs": args.epochs,
                   "partial": partial}, fh, indent=2)


def main() -> int:
    ap = argparse.ArgumentParser(description="Fine-tune the joint NLU model.")
    ap.add_argument("--per-intent", type=int, default=600)
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--lr", type=float, default=3e-5)
    ap.add_argument("--max-len", type=int, default=32)
    ap.add_argument("--freeze", type=int, default=2,
                    help="lower transformer layers to freeze (0-6)")
    ap.add_argument("--out", default=DEFAULT_OUT)
    ap.add_argument("--base", default=BASE_MODEL)
    ap.add_argument("--seed", type=int, default=13)
    ap.add_argument("--checkpoint-every", type=int, default=250,
                    help="steps between mid-epoch snapshots (0 = off). "
                         "Protects a long CPU run from an unexpected reboot.")
    ap.add_argument("--quick", action="store_true",
                    help="tiny run to sanity-check the pipeline")
    args = ap.parse_args()
    if args.quick:
        args.per_intent, args.epochs = 40, 1

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.set_num_threads(max(1, os.cpu_count() - 1))

    from transformers import AutoTokenizer, DistilBertModel

    print(f"loading base model '{args.base}' …", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(args.base)
    backbone = DistilBertModel.from_pretrained(args.base)

    from central_module.nlp.nlu.model import JointNLU
    model = JointNLU(backbone, len(INTENT_LABELS), len(SLOT_LABELS)).to(device)
    if args.freeze:
        model.freeze_lower(args.freeze)

    print(f"generating dataset ({args.per_intent}/intent) …", flush=True)
    data = DatasetGenerator(seed=args.seed).generate(per_intent=args.per_intent)
    n_dev = max(200, len(data) // 20)
    dev, train = data[:n_dev], data[n_dev:]
    print(f"  train {len(train)}  dev {n_dev}  "
          f"intents {len(INTENT_LABELS)}  slot tags {len(SLOT_LABELS)}")

    train_loader = DataLoader(encode(train, tokenizer, args.max_len),
                              batch_size=args.batch, shuffle=True)
    dev_loader = DataLoader(encode(dev, tokenizer, args.max_len),
                            batch_size=64)

    optim = torch.optim.AdamW(
        (p for p in model.parameters() if p.requires_grad), lr=args.lr)
    ce = nn.CrossEntropyLoss()
    ce_slots = nn.CrossEntropyLoss(ignore_index=-100)

    model.train()
    t0 = time.time()
    for epoch in range(1, args.epochs + 1):
        running, seen = 0.0, 0
        for step, (ids, mask, intents, slots) in enumerate(train_loader, 1):
            ids, mask = ids.to(device), mask.to(device)
            intent_logits, slot_logits = model(ids, mask)
            loss = (ce(intent_logits, intents.to(device))
                    + ce_slots(slot_logits.view(-1, len(SLOT_LABELS)),
                               slots.to(device).view(-1)))
            optim.zero_grad()
            loss.backward()
            optim.step()
            running += loss.item() * len(ids)
            seen += len(ids)
            if step % 50 == 0 or step == len(train_loader):
                rate = seen / (time.time() - t0)
                print(f"  epoch {epoch}  step {step}/{len(train_loader)}  "
                      f"loss {running / seen:.4f}  ({rate:.0f} ex/s)",
                      flush=True)
            if (args.checkpoint_every and step % args.checkpoint_every == 0
                    and step != len(train_loader)):
                # Unscored snapshot — saved as partial so the engine keeps
                # the rules in front of it until a full epoch validates.
                save(model, tokenizer, args, 0.0, 0.0, partial=True)
                print(f"    · snapshot at step {step}", flush=True)
        acc, slot_acc = evaluate(model, dev_loader, device)
        print(f"  epoch {epoch} done — dev intent acc {acc:.3f}  "
              f"slot acc {slot_acc:.3f}", flush=True)
        # Checkpoint every epoch: a long CPU run that gets interrupted
        # should leave a usable model behind rather than nothing.
        save(model, tokenizer, args, acc, slot_acc)
        print(f"  checkpoint written to {args.out}", flush=True)

    print(f"done in {time.time() - t0:.0f}s")
    # Qualitative sanity — the exact utterances the rules got wrong.
    from central_module.nlp.nlu.engine import NeuralNLU
    nlu = NeuralNLU(args.out)
    for text in ("increase volume by 5", "opun chrome", "make it louder",
                 "edit note.txt", "delete chatter.txt", "caps",
                 "what is the meaning of life"):
        print(f"  {text!r:35s} → {nlu.parse(text)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
