## What and why

<!-- One or two sentences: what this PR changes and the problem it solves. -->

## Modules touched

- [ ] Central module (orchestrator / NLU / OS adapters)
- [ ] Speech module
- [ ] Gesture module
- [ ] Models (Git LFS)
- [ ] CI / tooling / docs only

## How it was verified

<!-- CI covers the logic with fakes. Say what you checked by hand that CI cannot:
     a real microphone, a webcam, a specific OS, a retrained model... -->

- [ ] `python -m pytest` passes locally
- [ ] Tried on real hardware: <!-- mic / webcam / Windows / Linux-X11 / macOS -->

## Checklist

- [ ] Tests added or updated for the behaviour I changed
- [ ] Changed a calibrated constant? The comment next to it carries the new measurement
- [ ] Changed `central_module/nlp/intents.py`? Retrained the NLU model and committed `Models/NLU/` with it (or explained why not)
- [ ] New or changed model weights are tracked by Git LFS
- [ ] No `.venv/` or `__pycache__/` changes in this PR

<!-- The CI report comment below lists anything that still needs a reviewer's attention. -->
