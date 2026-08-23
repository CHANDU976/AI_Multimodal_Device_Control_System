"""
central_module — orchestrator for the Multimodal Device Control System.

Runs the speech module (and the external gesture app) as supervised child
processes, turns recognised speech into intents, executes those intents
through a per-OS action adapter, and adapts to the user over time.

    Speech child ──┐                        ┌─→ ActionExecutor → OSAdapter
                   ├─→ ProcessSupervisor ──→│      (win / mac / linux)
    Gesture child ─┘        (JSON lines)    └─→ BehaviorModel + PreferenceStore

Run it:
    python -m central_module                # full system
    python -m central_module --simulate     # type commands instead of speaking
    python -m central_module --dry-run      # log actions instead of executing

This package is imported inside child processes too (the runners), so it
must stay side-effect free at import time.
"""

__version__ = "0.1.0"
