"""
Root conftest — ensures vendored isolation is active for all pytest runs.

Adds src/ to sys.path minimally so `import htdml` works, then lets
htdml.paths.bootstrap_paths() install the full vendored path ordering.
"""

import sys
from pathlib import Path

# Minimal bootstrap: make src/ importable so htdml itself can be imported.
_src = str(Path(__file__).resolve().parent / "src")
if _src not in sys.path:
    sys.path.insert(0, _src)

import htdml.paths  # noqa: E402  (import after sys.path setup)

htdml.paths.bootstrap_paths()
