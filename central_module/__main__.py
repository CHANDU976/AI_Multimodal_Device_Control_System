"""
__main__.py — `python -m central_module`

    python -m central_module                 run the full system: speech child
                                             (microphone) + gesture child
                                             (Gesture Studio window)
    python -m central_module --simulate      type commands instead of speaking
    python -m central_module --dry-run       log actions instead of executing
    python -m central_module --once "open chrome"
                                             run one command through the pipeline
    python -m central_module --no-speech     without the speech child
    python -m central_module --no-gesture    without the gesture child
    python -m central_module --gesture-headless
                                             gesture engine without its window
    python -m central_module --gesture-stub  the gesture stub instead (no camera)
"""

from __future__ import annotations

import argparse
import logging
import sys

from central_module.manager import CentralManager


def main() -> int:
    parser = argparse.ArgumentParser(
        prog="central_module",
        description="Central orchestrator for the multimodal control system.")
    parser.add_argument("--simulate", action="store_true",
                        help="no child processes; type commands interactively")
    parser.add_argument("--dry-run", action="store_true",
                        help="report actions without executing them")
    parser.add_argument("--once", metavar="TEXT",
                        help="process one utterance, then exit")
    parser.add_argument("--no-speech", action="store_true",
                        help="do not start the speech child")
    parser.add_argument("--no-gesture", action="store_true",
                        help="do not start the gesture child")
    parser.add_argument("--gesture-headless", action="store_true",
                        help="run the gesture engine without its window "
                             "(camera from preferences gesture.camera)")
    parser.add_argument("--gesture-stub", action="store_true",
                        help="run the bundled gesture stub child (no camera)")
    parser.add_argument("-v", "--verbose", action="store_true",
                        help="debug logging + live speech partials")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
        stream=sys.stderr,
    )

    manager = CentralManager(
        dry_run=args.dry_run,
        verbose=args.verbose,
        enable_speech=not args.no_speech,
        force_gesture_stub=args.gesture_stub,
        enable_gesture=not args.no_gesture,
        gesture_mode="headless" if args.gesture_headless else None,
    )

    if args.once:
        manager.handle_text(args.once, origin="cli")
        manager.behavior.save()
        return 0
    if args.simulate:
        manager.run_simulated()
        return 0
    manager.run_forever()
    return 0


if __name__ == "__main__":
    sys.exit(main())
