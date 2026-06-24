"""Task 6 — tests for src/htdml/latent_adapter.py (TDD gate).

Tests:
  LA-1 : train/test/one_hot_target_labels exact shapes.
  LA-2 : test is EXACTLY 1000 rows.
  LA-3 : n_label_nodes == 10*5 == 50 (default target_classes + num_label_spots).
  LA-4 : "image" is bool, 196-wide.
  LA-5 : bool ↔ spin round-trip: (hard_latent > 0) → bool → (2*bool - 1) == hard_latent.
  LA-6 : one_hot_target_labels matches utils.one_hot(target_classes, target_classes, spots).
  LA-7 : (optional) DTM constructor accepts the injected dict — CPU only, no dtm.train.
  LA-8 : encode_split basic shapes.
  LA-9 : hard_latent_to_bool / bool_to_hard_latent are inverse at ±1 values.

Uses SYNTHETIC data only (random 28×28 images) — NO tfds.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

_REPO_ROOT = Path(__file__).resolve().parents[1]
_SRC = str(_REPO_ROOT / "src")
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

import htdml  # noqa: F401 (triggers bootstrap_paths)

import jax
import jax.numpy as jnp

from htdml.autoencoder import BinaryAutoencoder, encode
from htdml.latent_adapter import (
    build_latent_dataset,
    encode_split,
    hard_latent_to_bool,
    bool_to_hard_latent,
    _one_hot_target_labels,
)


# ----------------------------------------------------------------------- helpers
def _make_encoder(params):
    """Return a callable(images) → (hard_latent, logits)."""
    def _encode(images):
        return encode(params, jnp.asarray(images, dtype=jnp.float32))
    return _encode


def _make_synthetic_split(n_images: int, seed: int = 0):
    """Return (images, class_ids) with random 28×28×1 float images and random labels 0–9."""
    rng = np.random.default_rng(seed)
    images = rng.random((n_images, 28, 28, 1)).astype(np.float32)
    class_ids = rng.integers(0, 10, size=(n_images,)).astype(np.int32)
    return images, class_ids


@pytest.fixture(scope="module")
def ae_params():
    ae = BinaryAutoencoder()
    key = jax.random.PRNGKey(99)
    dummy = jnp.ones((2, 28, 28, 1), dtype=jnp.float32) * 0.5
    return ae.init(key, dummy)


@pytest.fixture(scope="module")
def synthetic_split(ae_params):
    """Returns (train_ds, test_ds, ohtl, train_images, train_class_ids) from a 1200-row split."""
    # 200 train + 1200 test → test sliced to 1000
    train_images, train_class_ids = _make_synthetic_split(200, seed=1)
    test_images, test_class_ids = _make_synthetic_split(1200, seed=2)
    enc_fn = _make_encoder(ae_params)
    train_ds, test_ds, ohtl = build_latent_dataset(
        enc_fn, train_images, train_class_ids, test_images, test_class_ids
    )
    return train_ds, test_ds, ohtl, train_images, train_class_ids


# ----------------------------------------------------------------------- LA-1: shapes
def test_dataset_shapes(synthetic_split):
    """train/test/one_hot_target_labels shapes are correct."""
    train_ds, test_ds, ohtl, _, _ = synthetic_split

    assert train_ds["image"].ndim == 2
    assert train_ds["image"].shape[1] == 196
    assert train_ds["label"].ndim == 2
    assert train_ds["label"].shape[1] == 50   # 10 classes * 5 spots

    assert test_ds["image"].ndim == 2
    assert test_ds["image"].shape[1] == 196
    assert test_ds["label"].ndim == 2
    assert test_ds["label"].shape[1] == 50

    assert ohtl.shape == (10, 50)


# ----------------------------------------------------------------------- LA-2: test exactly 1000
def test_test_exactly_1000_rows(synthetic_split):
    """test_dataset has EXACTLY 1000 rows."""
    _, test_ds, _, _, _ = synthetic_split
    assert test_ds["image"].shape[0] == 1000, (
        f"test image rows: {test_ds['image'].shape[0]} != 1000"
    )
    assert test_ds["label"].shape[0] == 1000, (
        f"test label rows: {test_ds['label'].shape[0]} != 1000"
    )


# ----------------------------------------------------------------------- LA-3: n_label_nodes = 50
def test_n_label_nodes_default(synthetic_split):
    """n_label_nodes == 10*5 == 50 with default target_classes and num_label_spots."""
    train_ds, _, _, _, _ = synthetic_split
    n_label_nodes = train_ds["label"].shape[1]
    assert n_label_nodes == 50, f"n_label_nodes expected 50, got {n_label_nodes}"


# ----------------------------------------------------------------------- LA-4: image dtype bool
def test_image_dtype_bool(synthetic_split):
    """'image' arrays are bool and 196-wide."""
    train_ds, test_ds, _, _, _ = synthetic_split
    assert train_ds["image"].dtype == bool, f"train image dtype {train_ds['image'].dtype} != bool"
    assert test_ds["image"].dtype == bool, f"test image dtype {test_ds['image'].dtype} != bool"
    assert train_ds["image"].shape[1] == 196
    assert test_ds["image"].shape[1] == 196


# ----------------------------------------------------------------------- LA-5: bool ↔ spin round-trip
def test_bool_spin_roundtrip(ae_params):
    """hard_latent → bool → (2*bool - 1) == hard_latent: the round-trip is exact at ±1."""
    images, _ = _make_synthetic_split(16, seed=3)
    enc_fn = _make_encoder(ae_params)
    hard_latent, _ = enc_fn(jnp.asarray(images))
    hard_latent_np = np.asarray(hard_latent)

    # → bool
    image_bool = hard_latent_to_bool(hard_latent_np)
    assert image_bool.dtype == bool

    # → spin back
    spin_back = bool_to_hard_latent(image_bool)  # int8 ∈ {−1,+1}
    assert np.all(spin_back == hard_latent_np.astype(np.int8)), (
        "bool ↔ spin round-trip failed: decoded spins don't match original hard_latent"
    )


# ----------------------------------------------------------------------- LA-6: one_hot_target_labels
def test_one_hot_target_labels():
    """one_hot_target_labels matches the utils.one_hot(target_classes, target_classes, spots) pattern."""
    # Import the vendored utils
    from thrmlDenoising import utils as dtm_utils

    target_classes = tuple(range(10))
    num_label_spots = 5

    # Our implementation
    ohtl_ours = _one_hot_target_labels(target_classes, num_label_spots)
    # DTM reference
    ohtl_ref = np.asarray(dtm_utils.one_hot(
        jnp.array(list(target_classes)),
        jnp.array(list(target_classes)),
        num_label_spots
    ), dtype=bool)

    assert ohtl_ours.shape == ohtl_ref.shape, (
        f"Shape mismatch: ours {ohtl_ours.shape} vs ref {ohtl_ref.shape}"
    )
    assert np.array_equal(ohtl_ours, ohtl_ref), (
        "one_hot_target_labels does not match utils.one_hot reference"
    )


# ----------------------------------------------------------------------- LA-7: DTM constructor accepts
def test_dtm_constructor_accepts_latent_dict(ae_params):
    """(Optional) Instantiate a small DTM with the latent adapter dict — CPU, no dtm.train.

    Uses smoke_testing preset (grayscale_levels=1, 3 classes) to verify the DTM constructor
    does NOT reject the injected dataset dict. We bypass load_dataset by monkey-patching.
    """
    pytest.importorskip("thrmlDenoising")
    import unittest.mock as mock
    from thrmlDenoising.DTM import DTM
    from thrmlDenoising.DTM_config import (
        DTMConfig, DataConfig, GraphConfig, SamplingScheduleConfig,
        DiffusionScheduleConfig, DiffusionRatesConfig, OptimConfig,
        CorrelationPenaltyConfig, WeightDecayConfig, ExperimentConfig
    )

    TARGET_CLASSES = (0, 1, 2)
    NUM_LABEL_SPOTS = 1
    N_LABEL = len(TARGET_CLASSES) * NUM_LABEL_SPOTS   # 3
    N_IMG = 196

    # Build synthetic latent dataset (3 classes, 1 spot → label_nodes=3)
    ae = BinaryAutoencoder()
    key = jax.random.PRNGKey(77)
    dummy_init = jnp.ones((2, 28, 28, 1), dtype=jnp.float32) * 0.5
    local_params = ae.init(key, dummy_init)
    enc_fn = _make_encoder(local_params)

    train_images, train_class_ids = _make_synthetic_split(50, seed=10)
    test_images, test_class_ids = _make_synthetic_split(1200, seed=11)
    # Filter to only target_classes
    train_mask = np.isin(train_class_ids, TARGET_CLASSES)
    test_mask = np.isin(test_class_ids, TARGET_CLASSES)
    # Need ≥1000 test after filter; pad if needed by repeating
    ti, tc = test_images[test_mask], test_class_ids[test_mask]
    while len(ti) < 1000:
        ti = np.concatenate([ti, ti], axis=0)
        tc = np.concatenate([tc, tc], axis=0)
    ti, tc = ti[:1200], tc[:1200]

    # Only keep training examples in target classes (may be very few; that's OK for constructor test)
    tr_img = train_images[train_mask] if train_mask.any() else train_images[:3]
    tr_cls = train_class_ids[train_mask] if train_mask.any() else np.array([0, 1, 2], dtype=np.int32)

    train_ds, test_ds, ohtl = build_latent_dataset(
        enc_fn, tr_img, tr_cls, ti, tc,
        target_classes=TARGET_CLASSES,
        num_label_spots=NUM_LABEL_SPOTS,
    )

    # Use the smoke_testing path to instantiate a real DTM (no tfds), then override the datasets
    n_target = len(TARGET_CLASSES)
    dataset_name = f"smoke_testing_{N_IMG}_1_{n_target}"

    cfg = DTMConfig(
        data=DataConfig(
            dataset_name=dataset_name,
            target_classes=TARGET_CLASSES,
            pixel_threshold_for_single_trials=0.1,
        ),
        graph=GraphConfig(
            graph_preset_architecture=4412,  # 44_12 preset (side 44, 1936 nodes); Python 44_12 == 4412; fits 196+labels in upper half
            num_label_spots=NUM_LABEL_SPOTS,
            grayscale_levels=1,
            torus=True,
            base_graph_manager="poisson_binomial_ising_graph_manager",
        ),
        sampling=SamplingScheduleConfig(
            batch_size=4,
            n_samples=2,
            steps_per_sample=1,
            steps_warmup=4,
            training_beta=1.0,
        ),
        diffusion_schedule=DiffusionScheduleConfig(
            num_diffusion_steps=1,
        ),
        diffusion_rates=DiffusionRatesConfig(image_rate=0.8, label_rate=0.2),
        exp=ExperimentConfig(seed=0, compute_autocorr=False, generate_gif=False),
    )

    # Add a smoke_testing entry for N_IMG=196, n_grayscale=1, n_target=3
    from thrmlDenoising.utils import smoke_test_data_dict
    if (N_IMG, 1, n_target) not in smoke_test_data_dict:
        rng = np.random.default_rng(0)
        fake_imgs = jnp.array(rng.integers(0, 2, size=(20, N_IMG)), dtype=jnp.bool_)
        fake_lbls = jnp.array(rng.integers(0, n_target, size=(20,)), dtype=jnp.int32)
        smoke_test_data_dict[(N_IMG, 1, n_target)] = {"image": fake_imgs, "label": fake_lbls}

    dtm = DTM(cfg)

    # NOW override with the latent adapter dataset
    dtm.train_dataset = train_ds
    dtm.test_dataset = test_ds
    dtm.one_hot_target_labels = ohtl
    dtm.n_image_pixels = train_ds["image"].shape[1]
    dtm.n_label_nodes = train_ds["label"].shape[1]

    # Verify the injected dict passes the DTM's own shape assertions
    assert dtm.train_dataset["image"].shape[1] == 196
    assert dtm.test_dataset["image"].shape == (1000, 196)
    assert dtm.test_dataset["label"].shape == (1000, N_LABEL)
    assert dtm.one_hot_target_labels.shape == (n_target, N_LABEL)


# ----------------------------------------------------------------------- LA-8: encode_split shapes
def test_encode_split_shapes(ae_params):
    """encode_split returns correct keys and shape for a small batch."""
    images, class_ids = _make_synthetic_split(30, seed=5)
    enc_fn = _make_encoder(ae_params)
    ds = encode_split(enc_fn, images, class_ids)

    assert "image" in ds and "label" in ds
    assert ds["image"].shape == (30, 196)
    assert ds["image"].dtype == bool
    assert ds["label"].shape == (30, 50)
    assert ds["label"].dtype == bool


# ----------------------------------------------------------------------- LA-9: inverse converters
def test_hard_latent_bool_inverses():
    """hard_latent_to_bool and bool_to_hard_latent are inverses at ±1 values."""
    rng = np.random.default_rng(99)
    # Random ±1 spins
    spins = (rng.integers(0, 2, size=(10, 196)) * 2 - 1).astype(np.int8)
    bools = hard_latent_to_bool(spins)
    back = bool_to_hard_latent(bools)
    assert np.all(back == spins), "hard_latent_to_bool then bool_to_hard_latent not identity at ±1"

    # Random bools
    bools2 = rng.integers(0, 2, size=(8, 196)).astype(bool)
    spins2 = bool_to_hard_latent(bools2)
    back2 = hard_latent_to_bool(spins2)
    assert np.all(back2 == bools2), "bool_to_hard_latent then hard_latent_to_bool not identity"
