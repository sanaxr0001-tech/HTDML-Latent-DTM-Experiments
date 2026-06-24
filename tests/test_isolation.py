"""
Isolation harness tests — Task 1.

Verifies that the vendored path bootstrap correctly shadows conda site-packages
and that the pinned source blobs are intact.
"""

import hashlib
import importlib
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
OVERLAY_ROOT = str(REPO_ROOT / "vendor" / "thrml_overlay")
DTM_REPL_ROOT = str(REPO_ROOT / "vendor" / "dtm-replication")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_thrml_resolves_to_overlay():
    """import thrml must resolve to the vendored overlay, NOT conda site-packages."""
    # Import htdml first to trigger bootstrap (conftest does this at collection,
    # but be explicit here for clarity).
    import htdml  # noqa: F401
    import thrml

    # Reload to pick up the bootstrapped path ordering
    importlib.reload(thrml)

    thrml_file = Path(thrml.__file__).resolve()
    assert str(thrml_file).startswith(
        str(REPO_ROOT / "vendor" / "thrml_overlay")
    ), (
        f"thrml resolved to {thrml_file!s}, expected a path under "
        f"{REPO_ROOT / 'vendor' / 'thrml_overlay'}"
    )


def test_thrmlDenoising_resolves_to_vendor():
    """import thrmlDenoising must resolve to vendor/dtm-replication."""
    import htdml  # noqa: F401
    import thrmlDenoising

    importlib.reload(thrmlDenoising)

    td_file = Path(thrmlDenoising.__file__).resolve()
    assert str(td_file).startswith(
        str(REPO_ROOT / "vendor" / "dtm-replication")
    ), (
        f"thrmlDenoising resolved to {td_file!s}, expected a path under "
        f"{REPO_ROOT / 'vendor' / 'dtm-replication'}"
    )


def test_dtm_py_sha256():
    """vendor/dtm-replication/thrmlDenoising/DTM.py must match the clean 7c22d19 blob sha256."""
    dtm_path = REPO_ROOT / "vendor" / "dtm-replication" / "thrmlDenoising" / "DTM.py"
    assert dtm_path.exists(), f"DTM.py not found at {dtm_path}"
    expected = "e7d48e2304e7667c55a0862a1155e08ef340b7869fc1814f4b0b7ef27913f472"
    actual = _sha256(dtm_path)
    assert actual == expected, (
        f"DTM.py sha256 mismatch: got {actual}, expected {expected}\n"
        "This means the vendored copy is NOT the clean 7c22d19 blob."
    )


def test_fid_ref_exists_and_sha256():
    """bw_fashion_mnist_train.npz must exist and match the pinned sha256."""
    fid_path = (
        REPO_ROOT
        / "vendor"
        / "dtm-replication"
        / "thrmlDenoising"
        / "fid"
        / "precomputed_stats"
        / "bw_fashion_mnist_train.npz"
    )
    assert fid_path.exists(), f"FID ref not found at {fid_path}"
    expected = "66003004dc99115b20c146bd3c2a7d9d85fb85a3c0c9e991f11951933f97c5d8"
    actual = _sha256(fid_path)
    assert actual == expected, (
        f"bw_fashion_mnist_train.npz sha256 mismatch: got {actual}, expected {expected}"
    )


def test_thrml_version():
    """The vendored thrml overlay must report version 0.1.3."""
    import htdml  # noqa: F401
    import thrml

    importlib.reload(thrml)

    assert thrml.__version__ == "0.1.3", (
        f"thrml.__version__ == {thrml.__version__!r}, expected '0.1.3'"
    )
