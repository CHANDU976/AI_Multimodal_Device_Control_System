"""
path_setup.py
Adds the project root (MyProject/) to sys.path so that sibling packages
like `utils` are importable from anywhere inside speech_module/.

Import this at the very top of any file that needs cross-package imports:

    import path_setup   # noqa: F401  (must be first)
"""

import os
import sys

# MyProject/speech_module/path_setup.py → MyProject/
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)
