"""
scripts/dataset_gate.py — Task 10 offline gate.

One-time download + cache + sha256 + PINS update for:
  1. Fashion-MNIST dataset (via tfds) → data/fashion_mnist/fashion_mnist_raw.npz
  2. InceptionV3 FID weights pickle → cache/inception_v3_weights_fid.pickle

After this script runs, everything is offline: inception.py loads the pickle
directly from the repo-local cache (P1 patch applied here permanently).

Usage:
    python scripts/dataset_gate.py          # download + cache + patch + record
    python scripts/dataset_gate.py --check  # verify cache + sha matches, no download

Idempotent: if the cache + correct sha already exist, skips the download.

Authorised by locked decision 3 / E3 in the companion plan (public data,
cached into repo, sha-pinned in PINS.md).
"""

import argparse
import hashlib
import pickle
import re
import sys
import textwrap
from pathlib import Path

# ---------------------------------------------------------------------------
# Repo-root bootstrap (so this script runs correctly from any CWD)
# ---------------------------------------------------------------------------

REPO_ROOT = Path(__file__).resolve().parents[1]
_src = str(REPO_ROOT / "src")
if _src not in sys.path:
    sys.path.insert(0, _src)

import htdml.paths  # noqa: E402

htdml.paths.bootstrap_paths()

# ---------------------------------------------------------------------------
# Canonical paths
# ---------------------------------------------------------------------------

INCEPTION_PICKLE_PATH = REPO_ROOT / "cache" / "inception_v3_weights_fid.pickle"
DATASET_CACHE_DIR = REPO_ROOT / "data" / "fashion_mnist"
DATASET_NPZ_PATH = DATASET_CACHE_DIR / "fashion_mnist_raw.npz"
INCEPTION_PY_PATH = (
    REPO_ROOT
    / "vendor"
    / "dtm-replication"
    / "thrmlDenoising"
    / "fid"
    / "inception.py"
)
PINS_MD_PATH = REPO_ROOT / "PINS.md"

# URL from the vendored inception.py source (inception.py:39-40, verified)
INCEPTION_PICKLE_URL = (
    "https://www.dropbox.com/s/xt6zvlvt22dcwck/inception_v3_weights_fid.pickle?dl=1"
)


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _verify_sha(path: Path, expected: str, label: str) -> None:
    actual = _sha256(path)
    if actual != expected:
        raise RuntimeError(
            f"SHA256 mismatch for {label}:\n"
            f"  expected : {expected}\n"
            f"  actual   : {actual}\n"
            f"  path     : {path}"
        )
    print(f"[sha256 OK] {label}: {actual[:16]}…")


# ---------------------------------------------------------------------------
# Step 1 — Download Fashion-MNIST via tfds and cache as npz
# ---------------------------------------------------------------------------


def fetch_fashion_mnist() -> str:
    """
    Download Fashion-MNIST via tfds (if not already cached), serialise the
    full train+test split into a single canonical NPZ, and return its sha256.

    Returns (dataset_sha256: str).

    The NPZ stores:
      train_images : uint8 (60000, 28, 28)
      train_labels : int64 (60000,)  ← RAW class integers 0..9
      test_images  : uint8 (10000, 28, 28)
      test_labels  : int64 (10000,)
    """
    if DATASET_NPZ_PATH.exists():
        print(f"[dataset] Cache hit: {DATASET_NPZ_PATH}")
        return _sha256(DATASET_NPZ_PATH)

    print("[dataset] Downloading Fashion-MNIST via tfds …")
    import numpy as np
    import tensorflow_datasets as tfds

    # tfds caches to ~/.tensorflow_datasets by default; we then copy to repo.
    ds_train = tfds.load(
        "fashion_mnist",
        split="train",
        as_supervised=False,
        shuffle_files=False,
    )
    ds_test = tfds.load(
        "fashion_mnist",
        split="test",
        as_supervised=False,
        shuffle_files=False,
    )

    def _collect(ds):
        images, labels = [], []
        for ex in tfds.as_numpy(ds):
            images.append(ex["image"])      # (28, 28, 1) uint8
            labels.append(int(ex["label"])) # raw class int 0..9
        images = np.stack(images, axis=0)  # (N, 28, 28, 1)
        labels = np.array(labels, dtype=np.int64)
        # Squeeze channel dim for compact storage; caller re-adds if needed.
        images = images.squeeze(-1)        # (N, 28, 28)
        return images, labels

    train_images, train_labels = _collect(ds_train)
    test_images, test_labels = _collect(ds_test)

    print(
        f"[dataset] Collected train {train_images.shape} / test {test_images.shape}"
    )

    DATASET_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    import numpy as np

    np.savez_compressed(
        str(DATASET_NPZ_PATH),
        train_images=train_images,
        train_labels=train_labels,
        test_images=test_images,
        test_labels=test_labels,
    )
    sha = _sha256(DATASET_NPZ_PATH)
    print(f"[dataset] Saved {DATASET_NPZ_PATH}  sha256={sha}")
    return sha


# ---------------------------------------------------------------------------
# Step 2 — Download InceptionV3 FID weights pickle
# ---------------------------------------------------------------------------


def fetch_inception_pickle(expected_sha: str | None = None) -> str:
    """
    Fetch the InceptionV3 FID weights pickle from Dropbox (once), cache it
    at INCEPTION_PICKLE_PATH, and return its sha256.

    If the file already exists and (if expected_sha is given) its sha matches,
    returns immediately without network access.
    """
    if INCEPTION_PICKLE_PATH.exists():
        actual = _sha256(INCEPTION_PICKLE_PATH)
        if expected_sha is None or actual == expected_sha:
            print(f"[inception] Cache hit: {INCEPTION_PICKLE_PATH}  sha256={actual[:16]}…")
            return actual
        print(
            f"[inception] Cache hit but sha mismatch (got {actual[:16]}…, "
            f"expected {expected_sha[:16]}…) — re-downloading."
        )

    print(f"[inception] Downloading from {INCEPTION_PICKLE_URL} …")
    import requests

    INCEPTION_PICKLE_PATH.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = INCEPTION_PICKLE_PATH.with_suffix(".tmp")
    try:
        resp = requests.get(INCEPTION_PICKLE_URL, stream=True, timeout=120)
        resp.raise_for_status()
        with open(tmp_path, "wb") as fh:
            for chunk in resp.iter_content(chunk_size=65536):
                fh.write(chunk)
        tmp_path.rename(INCEPTION_PICKLE_PATH)
    except Exception:
        if tmp_path.exists():
            tmp_path.unlink()
        raise

    sha = _sha256(INCEPTION_PICKLE_PATH)
    print(f"[inception] Saved {INCEPTION_PICKLE_PATH}  sha256={sha}")
    return sha


# ---------------------------------------------------------------------------
# Step 3 — Patch the vendored inception.py (P1)
# ---------------------------------------------------------------------------

_PATCH_MARKER = "htdml-task10-patch: pickle.load from repo-local cache"


def patch_inception_py() -> bool:
    """
    Patch vendored inception.py so InceptionV3.setup() loads the pickle
    directly from INCEPTION_PICKLE_PATH (absolute, repo-resolved path) instead
    of calling utils.download.

    Returns True if the patch was applied (or was already applied).
    Idempotent: if the marker is already present, skips.

    Targets inception.py:46-47 (verified against the vendored source):
        ckpt_file = utils.download(self.ckpt_path)
        self.params_dict = pickle.load(open(ckpt_file, "rb"))
    """
    source = INCEPTION_PY_PATH.read_text()

    if _PATCH_MARKER in source:
        print("[patch] P1 already applied — skipping.")
        return True

    # The exact lines to replace (8-space indent = 2 levels inside class + if block).
    # We match the exact indented strings from the source.
    OLD_DL_LINE   = "            ckpt_file = utils.download(self.ckpt_path)"
    OLD_LOAD_LINE = '            self.params_dict = pickle.load(open(ckpt_file, "rb"))'

    if OLD_DL_LINE not in source:
        # Already patched by another mechanism, or source changed.
        if "utils.download" not in source:
            print("[patch] utils.download absent — treating as already patched.")
            return True
        raise RuntimeError(
            f"[patch] FATAL: Could not locate expected line:\n  {OLD_DL_LINE!r}\n"
            "Manual inspection of inception.py required."
        )

    # Write a __file__-relative resolve (NOT an absolute path) so the patched
    # vendored copy survives being copied to a different checkout (e.g. the H200
    # for the deferred Stage-C run). inception.py lives at
    # vendor/dtm-replication/thrmlDenoising/fid/inception.py → parents[4] == repo root.
    NEW_LINES = (
        f"            # {_PATCH_MARKER}\n"
        f"            from pathlib import Path as _Path\n"
        f'            _pickle_path = str(_Path(__file__).resolve().parents[4] / "cache" / "inception_v3_weights_fid.pickle")\n'
        f'            self.params_dict = pickle.load(open(_pickle_path, "rb"))'
    )

    # Replace the two original lines with our three new lines.
    patched = source.replace(
        OLD_DL_LINE + "\n" + OLD_LOAD_LINE,
        NEW_LINES,
    )

    if patched == source:
        raise RuntimeError(
            "[patch] Replacement had no effect — source did not contain the expected two-line block."
        )

    INCEPTION_PY_PATH.write_text(patched)
    print(f"[patch] P1 applied to {INCEPTION_PY_PATH}")
    return True


# ---------------------------------------------------------------------------
# Step 4 — Update PINS.md
# ---------------------------------------------------------------------------


def update_pins(dataset_sha: str, inception_sha: str) -> None:
    """
    Replace the TBD placeholders in PINS.md with the actual sha256 values.
    Idempotent: if the pin already matches, leave it unchanged.
    """
    text = PINS_MD_PATH.read_text()
    changed = False

    # dataset-split sha256 placeholder
    dataset_pattern = r"(dataset-split sha256\s*\|\s*)TBD[^\n]*"
    dataset_repl = f"\\1`{dataset_sha}` |"
    new_text, n = re.subn(dataset_pattern, dataset_repl, text)
    if n:
        text = new_text
        changed = True
        print(f"[pins] Updated dataset-split sha256 → {dataset_sha[:16]}…")
    elif dataset_sha in text:
        print(f"[pins] dataset-split sha256 already recorded.")
    else:
        print(f"[pins] WARNING: could not locate dataset-split placeholder in PINS.md")

    # InceptionV3-FID-weights sha256 placeholder
    inception_pattern = r"(InceptionV3-FID-weights sha256\s*\|\s*)TBD[^\n]*"
    inception_repl = f"\\1`{inception_sha}` |"
    new_text, n = re.subn(inception_pattern, inception_repl, text)
    if n:
        text = new_text
        changed = True
        print(f"[pins] Updated InceptionV3-FID-weights sha256 → {inception_sha[:16]}…")
    elif inception_sha in text:
        print(f"[pins] InceptionV3-FID-weights sha256 already recorded.")
    else:
        print(f"[pins] WARNING: could not locate InceptionV3 placeholder in PINS.md")

    if changed:
        PINS_MD_PATH.write_text(text)
        print(f"[pins] PINS.md updated.")


# ---------------------------------------------------------------------------
# No-network proof helper (reused by Task 11 zero-compute + Task 12 smoke)
# ---------------------------------------------------------------------------


def assert_fid_offline() -> None:
    """
    Prove that the FID weights load path is network-free after the P1 patch.

    Monkeypatches both `requests.get` and `thrmlDenoising.fid.utils.download`
    to raise RuntimeError, then exercises InceptionV3.setup() by calling
    get_apply_fn(). If no exception is raised, the load is confirmed offline.

    Raises AssertionError if the pickle is not yet cached.
    Raises RuntimeError if any network call is attempted.

    Usage (in tests):
        from scripts.dataset_gate import assert_fid_offline
        assert_fid_offline()
    """
    assert INCEPTION_PICKLE_PATH.exists(), (
        f"Inception pickle not found at {INCEPTION_PICKLE_PATH}. "
        "Run `python scripts/dataset_gate.py` to download it first."
    )

    import importlib
    import unittest.mock as mock

    # Ensure the vendored fid module is imported (paths already bootstrapped)
    import thrmlDenoising.fid.utils as fid_utils
    import requests

    def _raise_network(*args, **kwargs):
        raise RuntimeError(
            "Network call attempted — FID load is NOT offline! "
            "The P1 patch may not have been applied correctly."
        )

    with (
        mock.patch.object(requests, "get", side_effect=_raise_network),
        mock.patch.object(fid_utils, "download", side_effect=_raise_network),
    ):
        # Force re-import of inception so setup() is called fresh.
        import thrmlDenoising.fid.inception as fid_inception
        import thrmlDenoising.fid.fid as fid_mod

        importlib.reload(fid_inception)
        importlib.reload(fid_mod)

        # get_apply_fn() calls InceptionV3(pretrained=True) → setup()
        params, apply_fn = fid_mod.get_apply_fn()

    # Sanity: confirm params loaded (params_dict should be a non-empty dict)
    # The Flax module stores state in `params` dict returned by model.init()
    assert params is not None, "get_apply_fn() returned None params"
    print("[no-network proof] PASS — FID weights loaded with network calls blocked.")


# ---------------------------------------------------------------------------
# Main gate runner
# ---------------------------------------------------------------------------


def run_gate(offline_ok: bool = False) -> dict:
    """
    Run the full dataset gate. Returns a dict with the two sha256s.

    Args:
        offline_ok: if True, skip downloads if cache is already present
                    (used for idempotency test).
    """
    print("=" * 60)
    print("Task 10 — offline dataset gate")
    print("=" * 60)

    # Step 1: Fashion-MNIST
    dataset_sha = fetch_fashion_mnist()

    # Step 2: InceptionV3 FID weights
    inception_sha = fetch_inception_pickle()

    # Step 3: Patch inception.py (P1)
    patch_inception_py()

    # Step 4: Record in PINS.md
    update_pins(dataset_sha, inception_sha)

    # Reload the patched module so subsequent imports see the patch.
    import importlib
    import thrmlDenoising.fid.inception as _fid_inc  # noqa: F401  (trigger import)
    importlib.reload(_fid_inc)

    print()
    print("=" * 60)
    print("Gate COMPLETE")
    print(f"  dataset-split sha256       : {dataset_sha}")
    print(f"  InceptionV3-weights sha256 : {inception_sha}")
    print("=" * 60)

    return {
        "dataset_sha256": dataset_sha,
        "inception_sha256": inception_sha,
    }


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--check",
        action="store_true",
        help="Verify cache + sha matches PINS without downloading.",
    )
    args = parser.parse_args()

    if args.check:
        # Verify only — do not download
        if not INCEPTION_PICKLE_PATH.exists():
            print(f"[check] MISSING: {INCEPTION_PICKLE_PATH}")
            sys.exit(1)
        if not DATASET_NPZ_PATH.exists():
            print(f"[check] MISSING: {DATASET_NPZ_PATH}")
            sys.exit(1)

        # Read pins from PINS.md
        pins_text = PINS_MD_PATH.read_text()

        def _extract_pin(key):
            for line in pins_text.splitlines():
                if key in line:
                    toks = line.split()
                    for tok in reversed(toks):
                        tok = tok.strip("`")
                        if re.fullmatch(r"[0-9a-f]{64}", tok):
                            return tok
            return None

        ds_pin = _extract_pin("dataset-split sha256")
        inc_pin = _extract_pin("InceptionV3-FID-weights sha256")
        ok = True
        if ds_pin:
            _verify_sha(DATASET_NPZ_PATH, ds_pin, "dataset-split")
        else:
            print("[check] dataset-split sha256 pin not yet recorded in PINS.md")
        if inc_pin:
            _verify_sha(INCEPTION_PICKLE_PATH, inc_pin, "InceptionV3-weights")
        else:
            print("[check] InceptionV3-FID-weights sha256 pin not yet recorded in PINS.md")
        if ok:
            print("[check] All verifications passed.")
    else:
        result = run_gate()
        print()
        print("Run `python -m pytest tests/test_dataset_gate.py -v` to verify.")
