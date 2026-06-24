"""
Isolation bootstrap for the htdml-latent-dtm companion repo.

Call bootstrap_paths() before importing thrml or thrmlDenoising to ensure
they resolve to the vendored copies rather than conda site-packages.
"""

import sys
from pathlib import Path


def bootstrap_paths() -> None:
    """Prepend vendored paths to sys.path (idempotent).

    Order (so overlay wins over conda site-packages):
      1. vendor/thrml_overlay  → import thrml resolves to patched overlay copy
      2. vendor/dtm-replication → import thrmlDenoising resolves to clean vendored tree
      3. src                   → import htdml resolves to this package

    Safe to call multiple times — entries are not duplicated.
    """
    repo_root = Path(__file__).resolve().parents[2]

    entries = [
        str(repo_root / "vendor" / "thrml_overlay"),
        str(repo_root / "vendor" / "dtm-replication"),
        str(repo_root / "src"),
    ]

    for entry in reversed(entries):
        if entry not in sys.path:
            sys.path.insert(0, entry)
