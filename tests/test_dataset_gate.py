"""
Task 10 — tests for scripts/dataset_gate.py.

Test plan:
  1. patch_correctness           — inception.py no longer calls utils.download in setup()
  2. fid_bw_npz_intact           — bw_fashion_mnist_train.npz sha256 still matches PINS
  3. inception_pickle_present    — cache/inception_v3_weights_fid.pickle exists
  4. inception_pickle_sha_pins   — pickle sha256 matches PINS.md entry
  5. dataset_cache_present       — data/fashion_mnist/ npz files exist
  6. dataset_sha_pins            — dataset sha256 matches PINS.md entry
  7. no_network_fid_load         — FID weights load with requests.get + utils.download monkeypatched to raise
  8. idempotency                 — running gate twice does NOT re-download (offline second run)

Tests 3–8 are skipped with SKIP_IF_NOT_DOWNLOADED if the cache hasn't been populated yet
(i.e., the gate hasn't been run). This lets TDD work before network access.
"""

import ast
import hashlib
import importlib
import re
import sys
import textwrap
import types
import unittest.mock as mock
from pathlib import Path

import pytest

# ---------------------------------------------------------------------------
# Bootstrap (mirrors conftest.py — harmless if already done)
# ---------------------------------------------------------------------------

REPO_ROOT = Path(__file__).resolve().parents[1]
_src = str(REPO_ROOT / "src")
if _src not in sys.path:
    sys.path.insert(0, _src)

import htdml.paths  # noqa: E402

htdml.paths.bootstrap_paths()

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

INCEPTION_PY = (
    REPO_ROOT
    / "vendor"
    / "dtm-replication"
    / "thrmlDenoising"
    / "fid"
    / "inception.py"
)
BW_NPZ = (
    REPO_ROOT
    / "vendor"
    / "dtm-replication"
    / "thrmlDenoising"
    / "fid"
    / "precomputed_stats"
    / "bw_fashion_mnist_train.npz"
)
INCEPTION_PICKLE = REPO_ROOT / "cache" / "inception_v3_weights_fid.pickle"
DATASET_CACHE_DIR = REPO_ROOT / "data" / "fashion_mnist"
PINS_MD = REPO_ROOT / "PINS.md"

# Skip marker for tests that require the cache to be populated.
SKIP_IF_NOT_DOWNLOADED = pytest.mark.skipif(
    not INCEPTION_PICKLE.exists(),
    reason=(
        "Inception pickle not cached yet — run `python scripts/dataset_gate.py` first. "
        "Script + test file are still the deliverable."
    ),
)

SKIP_IF_NO_DATASET = pytest.mark.skipif(
    not DATASET_CACHE_DIR.exists() or not any(DATASET_CACHE_DIR.glob("*.npz")),
    reason=(
        "Dataset cache not populated yet — run `python scripts/dataset_gate.py` first."
    ),
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def _read_pin(key: str) -> str | None:
    """Extract a sha256 value from PINS.md by key substring.

    Handles both bare hex and backtick-wrapped hex (e.g. `abc123…`).
    """
    text = PINS_MD.read_text()
    for line in text.splitlines():
        if key in line:
            # Strip backticks and pipes from tokens, then look for 64-char hex
            tokens = line.split()
            for tok in reversed(tokens):
                tok_clean = tok.strip("`|")
                if re.fullmatch(r"[0-9a-f]{64}", tok_clean):
                    return tok_clean
    return None


# ---------------------------------------------------------------------------
# Test 1 — patch correctness (AST / grep check)
# ---------------------------------------------------------------------------


def test_patch_correctness_no_utils_download_in_setup():
    """
    The vendored inception.py setup() must NOT call utils.download.
    We verify this via AST: look for a Call node whose func resolves
    to 'utils.download' inside the 'setup' method of 'InceptionV3'.
    """
    source = INCEPTION_PY.read_text()
    tree = ast.parse(source)

    # Find InceptionV3 class
    inception_class = None
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef) and node.name == "InceptionV3":
            inception_class = node
            break
    assert inception_class is not None, "Could not find InceptionV3 class in inception.py"

    # Find setup method
    setup_method = None
    for node in inception_class.body:
        if isinstance(node, ast.FunctionDef) and node.name == "setup":
            setup_method = node
            break
    assert setup_method is not None, "Could not find setup() in InceptionV3"

    # Check that utils.download is NOT called
    for node in ast.walk(setup_method):
        if isinstance(node, ast.Call):
            func = node.func
            # utils.download(...) appears as Attribute(value=Name(id='utils'), attr='download')
            if (
                isinstance(func, ast.Attribute)
                and func.attr == "download"
                and isinstance(func.value, ast.Name)
                and func.value.id == "utils"
            ):
                pytest.fail(
                    "setup() still calls utils.download — P1 patch was NOT applied correctly."
                )

    # Also verify via raw text (belt + suspenders)
    setup_lines = source.splitlines()
    in_setup = False
    for line in setup_lines:
        stripped = line.strip()
        if stripped.startswith("def setup("):
            in_setup = True
        elif in_setup and stripped.startswith("def ") and not stripped.startswith("def setup("):
            break
        if in_setup and "utils.download" in line:
            pytest.fail(
                f"setup() contains 'utils.download' on line: {line!r}\n"
                "P1 patch was NOT applied correctly."
            )


# ---------------------------------------------------------------------------
# Test 2 — bw_fashion_mnist_train.npz still intact
# ---------------------------------------------------------------------------


def test_fid_bw_npz_intact():
    """bw_fashion_mnist_train.npz sha256 must still match PINS (unchanged by Task 10)."""
    assert BW_NPZ.exists(), f"BW FID ref not found at {BW_NPZ}"
    expected = "66003004dc99115b20c146bd3c2a7d9d85fb85a3c0c9e991f11951933f97c5d8"
    actual = _sha256(BW_NPZ)
    assert actual == expected, (
        f"bw_fashion_mnist_train.npz sha256 mismatch: got {actual}, expected {expected}"
    )


# ---------------------------------------------------------------------------
# Test 3 — inception pickle present
# ---------------------------------------------------------------------------


@SKIP_IF_NOT_DOWNLOADED
def test_inception_pickle_present():
    """cache/inception_v3_weights_fid.pickle must exist after the gate runs."""
    assert INCEPTION_PICKLE.exists(), (
        f"Inception pickle not found at {INCEPTION_PICKLE}. "
        "Run `python scripts/dataset_gate.py` to download it."
    )


# ---------------------------------------------------------------------------
# Test 4 — inception pickle sha matches PINS
# ---------------------------------------------------------------------------


@SKIP_IF_NOT_DOWNLOADED
def test_inception_pickle_sha_matches_pins():
    """sha256 of the cached pickle must match the value recorded in PINS.md."""
    pin = _read_pin("InceptionV3-FID-weights sha256")
    assert pin is not None, (
        "Could not find 'InceptionV3-FID-weights sha256' pin in PINS.md. "
        "Did dataset_gate.py record it?"
    )
    actual = _sha256(INCEPTION_PICKLE)
    assert actual == pin, (
        f"Inception pickle sha256 mismatch: got {actual}, PINS says {pin}"
    )


# ---------------------------------------------------------------------------
# Test 5 — dataset cache present
# ---------------------------------------------------------------------------


@SKIP_IF_NO_DATASET
def test_dataset_cache_present():
    """data/fashion_mnist/ must contain at least one .npz file after the gate."""
    npz_files = list(DATASET_CACHE_DIR.glob("*.npz"))
    assert len(npz_files) > 0, (
        f"No .npz files found under {DATASET_CACHE_DIR}. "
        "Run `python scripts/dataset_gate.py` to populate the cache."
    )


# ---------------------------------------------------------------------------
# Test 6 — dataset sha matches PINS
# ---------------------------------------------------------------------------


@SKIP_IF_NO_DATASET
def test_dataset_sha_matches_pins():
    """sha256 of the cached dataset split must match the value recorded in PINS.md."""
    pin = _read_pin("dataset-split sha256")
    assert pin is not None, (
        "Could not find 'dataset-split sha256' pin in PINS.md. "
        "Did dataset_gate.py record it?"
    )
    # The gate serialises train images + labels into a canonical .npz
    npz_path = DATASET_CACHE_DIR / "fashion_mnist_raw.npz"
    assert npz_path.exists(), (
        f"Expected canonical npz at {npz_path}. "
        "Run `python scripts/dataset_gate.py` to populate the cache."
    )
    actual = _sha256(npz_path)
    assert actual == pin, (
        f"Dataset split sha256 mismatch: got {actual}, PINS says {pin}"
    )


# ---------------------------------------------------------------------------
# Test 7 — no-network FID load (load-bearing)
# ---------------------------------------------------------------------------


@SKIP_IF_NOT_DOWNLOADED
def test_no_network_fid_load():
    """
    With requests.get AND thrmlDenoising.fid.utils.download both monkeypatched
    to raise RuntimeError, InceptionV3.setup() must load the pickle from the
    repo-local cache WITHOUT making any network call.

    This is the load-bearing test proving the P1 patch is correct.
    """
    from scripts.dataset_gate import assert_fid_offline

    assert_fid_offline()  # raises AssertionError / RuntimeError on failure


# ---------------------------------------------------------------------------
# Test 8 — idempotency
# ---------------------------------------------------------------------------


@SKIP_IF_NOT_DOWNLOADED
def test_idempotency(tmp_path, capsys):
    """
    Running the gate a second time (cache already populated) must NOT
    re-download anything. We verify by monkeypatching requests.get to raise
    and confirming the gate still completes (reads from cache).
    """
    import requests

    def _raise(*args, **kwargs):
        raise RuntimeError("Network call attempted on second run — NOT idempotent!")

    with mock.patch.object(requests, "get", side_effect=_raise):
        # Import fresh to avoid module-level caching issues
        import importlib
        import scripts.dataset_gate as dg

        importlib.reload(dg)
        dg.run_gate(offline_ok=True)

    # If we reach here without RuntimeError, idempotency is confirmed.
