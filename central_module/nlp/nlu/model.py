"""
model.py
JointNLU — DistilBERT with two heads (the JointBERT architecture):

    intent head : CLS vector  → one of the ~50 intent classes
    slot head   : each token  → BIO slot tag

One forward pass answers both "what does the user want" and "which words
are the arguments", so intent and slots can never come from two different
readings of the utterance.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class JointNLU(nn.Module):

    def __init__(self, backbone, n_intents: int, n_slots: int,
                 dropout: float = 0.2) -> None:
        super().__init__()
        self.backbone = backbone                    # DistilBertModel
        hidden = backbone.config.dim
        self.dropout = nn.Dropout(dropout)
        self.intent_head = nn.Linear(hidden, n_intents)
        self.slot_head = nn.Linear(hidden, n_slots)

    def forward(self, input_ids: torch.Tensor,
                attention_mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        hidden = self.backbone(
            input_ids=input_ids, attention_mask=attention_mask,
        ).last_hidden_state                          # (B, T, H)
        hidden = self.dropout(hidden)
        intent_logits = self.intent_head(hidden[:, 0])   # CLS position
        slot_logits = self.slot_head(hidden)             # every token
        return intent_logits, slot_logits

    def freeze_lower(self, n_layers: int) -> None:
        """Freeze embeddings + the first n transformer layers: DistilBERT's
        lower layers hold general language features that a 50-class command
        task has no need to rewrite — training gets ~40% faster on CPU."""
        for p in self.backbone.embeddings.parameters():
            p.requires_grad = False
        for layer in self.backbone.transformer.layer[:n_layers]:
            for p in layer.parameters():
                p.requires_grad = False
