"""Seam-A latent dataset adapter — Task 6.

Bypasses `DTM.py:112-118`'s `load_dataset` call by injecting a pre-encoded latent
dataset directly into the DTM constructor's expected dict format.

bool ↔ spin convention
-----------------------
The thrml Ising model converts bool → spin via  `2 * x.astype(int8) - 1`
(vendor/thrml_overlay/thrml/models/ising.py:204).  Therefore:
    bool True  (1) → spin +1
    bool False (0) → spin −1

The encoder's `hard_latent` is ∈ {−1, +1}.  We map:
    hard_latent == +1  →  bool True   (bit 1)
    hard_latent == −1  →  bool False  (bit 0)

i.e.  image_bool = (hard_latent > 0)

Round-trip verification: hard_latent → bool → (DTM internal spin = 2*bool - 1)
   = 2*(hard_latent > 0).astype(int) - 1 = hard_latent   ✓  (holds at ±1)

DTM expected dict structure (DTM.py:112-121, step.py:393)
---------------------------------------------------------
  train_dataset       = {"image": (N_train, 196) bool, "label": (N_train, n_label_nodes) bool}
  test_dataset        = {"image": (1000, 196)    bool, "label": (1000, n_label_nodes)    bool}
  one_hot_target_labels = (num_classes, n_label_nodes) bool

  n_label_nodes = len(target_classes) * num_label_spots  (default: 10 * 5 = 50)

Label encoding (matching utils.one_hot)
-----------------------------------------
  utils.one_hot(x, digits, num_label_spots) returns a bool array of shape
  (N, len(digits) * num_label_spots) where each row is the class one-hot repeated
  num_label_spots times along the feature axis.

  For image class c ∈ target_classes:
    one_hot_row[i] = True  iff  target_classes[i % len(target_classes)] == c

  one_hot_target_labels: row k = the label encoding for class target_classes[k].
"""

from __future__ import annotations

import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
_SRC = str(_REPO_ROOT / "src")
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

from htdml.paths import bootstrap_paths  # noqa: E402
bootstrap_paths()

from typing import Sequence, Tuple

import jax
import jax.numpy as jnp
import numpy as np

# Default config values from PINS.md
_DEFAULT_TARGET_CLASSES = tuple(range(10))   # all 10 Fashion-MNIST classes
_DEFAULT_NUM_LABEL_SPOTS = 5                 # upstream training_script.py:32
_TEST_SIZE = 1000                            # hard requirement: step.py:393 / DTM.py:675-680


# ------------------------------------------------------------------ one-hot label construction
def _one_hot_labels(
    class_ids: np.ndarray,
    target_classes: Sequence[int],
    num_label_spots: int,
) -> np.ndarray:
    """Build the repeated one-hot label array matching utils.one_hot(class_ids, digits, spots).

    utils.one_hot(x, digits, num_label_spots) computes:
        one_hot = (x[:, None] == digits)        → (N, K) bool
        one_hot_repeated = tile/concat * spots  → (N, K*spots) bool

    Args:
        class_ids       : (N,) integer class indices ∈ target_classes.
        target_classes  : ordered sequence of class ids (the "digits" argument).
        num_label_spots : repetitions.

    Returns:
        (N, len(target_classes)*num_label_spots) bool array.
    """
    digits = np.array(target_classes, dtype=np.int32)
    base = (class_ids[:, None] == digits[None, :])              # (N, K) bool
    return np.tile(base, (1, num_label_spots))                   # (N, K*spots) bool


def _one_hot_target_labels(
    target_classes: Sequence[int],
    num_label_spots: int,
) -> np.ndarray:
    """Build `one_hot_target_labels` matching utils.one_hot(jnp.array(target_classes), ...).

    Shape: (len(target_classes), len(target_classes)*num_label_spots) bool.
    Row k is the label encoding for class target_classes[k].
    """
    digits = np.array(target_classes, dtype=np.int32)
    return _one_hot_labels(digits, target_classes, num_label_spots)


# ------------------------------------------------------------------ spin ↔ bool
def hard_latent_to_bool(hard_latent: np.ndarray) -> np.ndarray:
    """Convert encoder hard spins {−1,+1} to bool image bits.

    Convention (from thrml ising.py:204: `2*x.astype(int8) - 1`):
        bool True  → spin +1   ∴  spin == +1  ↔  bool True
        bool False → spin −1   ∴  spin == −1  ↔  bool False

    Round-trip: hard_latent → bool → (2*bool - 1) = hard_latent  (at ±1).

    Args:
        hard_latent: (..., 196) array with values ∈ {−1, +1}.
    Returns:
        (..., 196) bool array.
    """
    return np.asarray(hard_latent > 0, dtype=bool)


def bool_to_hard_latent(image_bool: np.ndarray) -> np.ndarray:
    """Inverse: bool image bits → {−1,+1} spins.

    Args:
        image_bool: (..., 196) bool array.
    Returns:
        (..., 196) int8 array with values ∈ {−1, +1}.
    """
    return (2 * np.asarray(image_bool, dtype=np.int8) - 1)


# ------------------------------------------------------------------ encoding a split
def encode_split(
    encode_fn,
    images: np.ndarray,
    class_ids: np.ndarray,
    target_classes: Sequence[int] = _DEFAULT_TARGET_CLASSES,
    num_label_spots: int = _DEFAULT_NUM_LABEL_SPOTS,
) -> dict:
    """Encode one dataset split (train or test) into the DTM dict format.

    Args:
        encode_fn   : callable(x) → (hard_latent, logits) where x is (B, 28, 28, 1) float32.
                      The encoder must already have params applied (e.g. a partial or a closure).
        images      : (N, 28, 28, 1) float32 ∈ [0,1] — raw pixel images.
        class_ids   : (N,) int — integer class labels ∈ target_classes.
        target_classes : ordered class ids (default: 0..9).
        num_label_spots: repetitions for the one-hot label (default: 5).

    Returns:
        dict with keys "image" (N, 196) bool and "label" (N, K*spots) bool.
    """
    hard_latent, _ = encode_fn(images)
    hard_latent = np.asarray(hard_latent)
    image_bool = hard_latent_to_bool(hard_latent)                # (N, 196) bool

    class_ids_np = np.asarray(class_ids, dtype=np.int32)
    label_bool = _one_hot_labels(class_ids_np, target_classes, num_label_spots)

    return {"image": image_bool, "label": label_bool}


# ------------------------------------------------------------------ main adapter
def build_latent_dataset(
    encode_fn,
    train_images: np.ndarray,
    train_class_ids: np.ndarray,
    test_images: np.ndarray,
    test_class_ids: np.ndarray,
    target_classes: Sequence[int] = _DEFAULT_TARGET_CLASSES,
    num_label_spots: int = _DEFAULT_NUM_LABEL_SPOTS,
) -> Tuple[dict, dict, np.ndarray]:
    """Encode train + test splits and emit the triple DTM.py:112-118 expects.

    Seam-A bypass of `load_dataset`: after calling this function the caller
    sets `dtm.train_dataset`, `dtm.test_dataset`, `dtm.one_hot_target_labels`
    directly, skipping the tfds path entirely.

    bool ↔ spin convention (documented in module docstring):
        image_bool[i, b] = True  ↔  encoder hard_latent = +1  ↔  DTM spin = +1
        image_bool[i, b] = False ↔  encoder hard_latent = −1  ↔  DTM spin = −1

    Assertion: test split is sliced to EXACTLY 1000 rows, satisfying
        DTM.py:675-680 (`assert test["image"].shape == (1000, n_image_pixels)`)
        and step.py:393 (`jr.randint(..., 0, 1000)`).

    Args:
        encode_fn        : callable(images: ndarray) → (hard_latent, logits).
                           Already has params bound (e.g. `partial(encode, params)` or closure).
        train_images     : (N_train, 28, 28, 1) float32.
        train_class_ids  : (N_train,) int.
        test_images      : (N_test, 28, 28, 1) float32.  N_test >= 1000 required.
        test_class_ids   : (N_test,) int.
        target_classes   : ordered class ids (default all 10).
        num_label_spots  : label repetitions (default 5).

    Returns:
        (train_dataset, test_dataset, one_hot_target_labels)
        where each dict has "image" bool + "label" bool as described above,
        test_dataset["image"].shape == (1000, 196),
        one_hot_target_labels.shape == (len(target_classes), len(target_classes)*num_label_spots).
    """
    if len(test_images) < _TEST_SIZE:
        raise ValueError(
            f"test_images must have at least {_TEST_SIZE} rows; got {len(test_images)}"
        )

    train_ds = encode_split(
        encode_fn, train_images, train_class_ids, target_classes, num_label_spots
    )
    # Encode the full test set, then slice to exactly 1000
    test_ds_full = encode_split(
        encode_fn, test_images, test_class_ids, target_classes, num_label_spots
    )
    test_ds = {k: v[:_TEST_SIZE] for k, v in test_ds_full.items()}

    ohtl = _one_hot_target_labels(target_classes, num_label_spots)  # (num_classes, K*spots) bool

    # --- shape assertions (DTM.py:120-121 + DTM.py:675-680) ---
    n_image_pixels = train_ds["image"].shape[1]
    n_label_nodes = train_ds["label"].shape[1]
    assert n_image_pixels == 196, f"expected 196-wide image; got {n_image_pixels}"
    assert n_label_nodes == len(target_classes) * num_label_spots, (
        f"n_label_nodes {n_label_nodes} != {len(target_classes)}*{num_label_spots}"
    )
    assert test_ds["image"].shape == (_TEST_SIZE, n_image_pixels), (
        f"test image shape {test_ds['image'].shape} != ({_TEST_SIZE}, {n_image_pixels})"
    )
    assert test_ds["label"].shape == (_TEST_SIZE, n_label_nodes), (
        f"test label shape {test_ds['label'].shape} != ({_TEST_SIZE}, {n_label_nodes})"
    )
    assert ohtl.shape == (len(target_classes), n_label_nodes), (
        f"one_hot_target_labels shape {ohtl.shape} != ({len(target_classes)}, {n_label_nodes})"
    )
    assert train_ds["image"].dtype == bool, f"train image dtype {train_ds['image'].dtype} != bool"
    assert test_ds["image"].dtype == bool, f"test image dtype {test_ds['image'].dtype} != bool"
    assert train_ds["label"].dtype == bool, f"train label dtype {train_ds['label'].dtype} != bool"
    assert test_ds["label"].dtype == bool, f"test label dtype {test_ds['label'].dtype} != bool"

    return train_ds, test_ds, ohtl
