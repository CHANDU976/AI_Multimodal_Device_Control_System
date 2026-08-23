"""
Neural NLU: a fine-tuned DistilBERT that jointly classifies the intent and
tags the slot tokens of an utterance (JointBERT architecture).

    dataset.py   synthetic training data from the catalog + paraphrases
    model.py     DistilBERT backbone + intent head + slot head
    train.py     fine-tuning CLI  (python -m central_module.nlp.nlu.train)
    engine.py    inference wrapper used by IntentEngine

The trained model lives in Models/NLU/ next to the other offline models.
transformers/torch are imported lazily so the rest of the central module
never pays for them.
"""
