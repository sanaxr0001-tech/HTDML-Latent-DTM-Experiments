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

# Also expose repo root so `from scripts.dataset_gate import ...` works in tests.
_repo_root = str(Path(__file__).resolve().parent)
if _repo_root not in sys.path:
    sys.path.insert(0, _repo_root)

import htdml.paths  # noqa: E402  (import after sys.path setup)

htdml.paths.bootstrap_paths()
