"""smoke_common — shared GPU-path builders for the Task-12 local-4060 smoke + calibration.

Builds the REAL 44_12 companion DTM on REAL Fashion-MNIST-encoded latents and exercises the full
companion path (encode → fit → probe → generate → decode → FID).  Used by BOTH
``scripts/smoke.py`` (plumbing verification) and ``scripts/calibrate_epoch_cost.py`` (epoch cost +
trajectory-adequacy freeze), so a single 1hr GPU session can run smoke + calibration on ONE trained
DTM (the build-notes §"TASK-12 SCOPE" cap).

Design (build-notes-faithful):
  * The DTM is constructed via a ``smoke_testing_196_1_C`` dataset name → ``is_smoke_test=True`` so
    the constructor routes ``eval_epoch`` to the gen-based ``_smoke_eval_epoch`` (NOT the 28-hardcoded
    ``do_draw_and_fid``, which would crash on the 196-wide latent — build-notes §"Fork Seam A").  We
    pass ``evaluate_every=0`` to ``.fit`` so NO eval epoch runs at all (the smoke does generate/decode/
    FID itself as explicit stages).
  * The ``is_smoke_test`` constructor forces ``batch_size = len(registered smoke dataset)``
    (DTM.py:138-139), so we register a SMALL smoke dataset → small train batch → fast real-graph step.
  * AFTER construction we ``inject_latents`` the REAL encoded Fashion-MNIST latent dataset (seam A),
    so training + the ACP ``compute_autocorr`` run on real latents through the real 44_12 graph.
  * The reversible ½(P_AB+P_BA) kernel must be LIVE (asserted in ``LatentDTM.fit``).

NOTHING here trains the autoencoder to convergence — Stage A is a tiny pre-train (the smoke verifies
the encode→decode plumbing, not encoder quality).  NO Stage-C / joint / control / two-seed (the
researcher-conferred Task-12 scope excludes them).
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
for _p in (str(_REPO_ROOT), str(_REPO_ROOT / "src")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import htdml.paths  # noqa: E402

htdml.paths.bootstrap_paths()

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import jax.random as jr  # noqa: E402
import numpy as np  # noqa: E402

from htdml.autoencoder import BinaryAutoencoder, decode as ae_decode, stage_a_loss  # noqa: E402
from htdml.latent_adapter import build_latent_dataset  # noqa: E402
from htdml.latent_dtm import LatentDTM, make_companion_cfg  # noqa: E402

DATASET_NPZ_PATH = _REPO_ROOT / "data" / "fashion_mnist" / "fashion_mnist_raw.npz"
BW_NPZ_PATH = (
    _REPO_ROOT / "vendor" / "dtm-replication" / "thrmlDenoising" / "fid"
    / "precomputed_stats" / "bw_fashion_mnist_train.npz"
)

# Smoke registered-dataset key (n_image_pixels, n_grayscale_levels, n_target_classes).
SMOKE_N_IMG = 196
SMOKE_NGRAY = 1
SMOKE_NCLS = 3           # 3 classes keeps the registered-smoke key small but >1


# The smoke restricts to the first SMOKE_NCLS classes with num_label_spots=1 so the injected latent
# dataset's label width (= n_target_classes · num_label_spots) MATCHES the n_label_nodes the DTM graph
# was constructed with from the registered smoke key (n3 · num_label_spots).  Both = SMOKE_NCLS · 1.
SMOKE_TARGET_CLASSES = tuple(range(SMOKE_NCLS))   # (0, 1, 2)
SMOKE_NUM_LABEL_SPOTS = 1


def load_fashion_mnist(n_train: int, n_test: int = 1000, *, target_classes=SMOKE_TARGET_CLASSES):
    """Load the cached Fashion-MNIST split (Task-10 gate), FILTERED to ``target_classes`` →
    float32 [0,1] (N,28,28,1) + int labels.  Filtering keeps the label space = the graph's
    n_label_nodes (consistency with the registered smoke key)."""
    assert DATASET_NPZ_PATH.exists(), (
        f"Fashion-MNIST cache missing at {DATASET_NPZ_PATH}; run scripts/dataset_gate.py first."
    )
    z = np.load(DATASET_NPZ_PATH)
    tc = np.asarray(target_classes)

    def _filter(imgs, labs, n_keep):
        mask = np.isin(labs, tc)
        imgs, labs = imgs[mask], labs[mask]
        return imgs[:n_keep], labs[:n_keep]

    tr_img, tr_lab = _filter(z["train_images"], z["train_labels"].astype(np.int32), n_train)
    te_img, te_lab = _filter(z["test_images"], z["test_labels"].astype(np.int32), max(n_test, 1000))
    tr_img = (tr_img.astype(np.float32) / 255.0)[..., None]
    te_img = (te_img.astype(np.float32) / 255.0)[..., None]
    return tr_img, tr_lab, te_img, te_lab


def pretrain_autoencoder(train_images, *, key, n_steps, batch_size, lr):
    """Tiny Stage-A pre-train of the BinaryAutoencoder (plumbing, NOT convergence)."""
    import optax

    ae = BinaryAutoencoder()
    k_init, k_train = jr.split(key)
    params = ae.init(k_init, jnp.ones((2, 28, 28, 1), jnp.float32) * 0.5)
    optim = optax.adam(lr)
    opt_state = optim.init(params)

    @jax.jit
    def update(params, opt_state, x):
        (loss, aux), grads = jax.value_and_grad(stage_a_loss, has_aux=True)(params, x)
        updates, opt_state = optim.update(grads, opt_state, params)
        params = optax.apply_updates(params, updates)
        return params, opt_state, loss

    n = train_images.shape[0]
    losses = []
    for i in range(int(n_steps)):
        k_train, kb = jr.split(k_train)
        idx = jr.randint(kb, (int(batch_size),), 0, n)
        x = jnp.asarray(train_images[np.asarray(idx)])
        params, opt_state, loss = update(params, opt_state, x)
        losses.append(float(loss))
    return params, losses


def build_companion_dtm(latent_ds, *, seed=0):
    """Construct the REAL 44_12 companion DTM (is_smoke_test routing) then inject the real latent ds.

    Returns the constructed (pre-train) vendored DTM with the real latent dataset injected and the
    correct ``n_image_pixels``/``n_label_nodes`` set."""
    from thrmlDenoising.utils import smoke_test_data_dict
    from thrmlDenoising.DTM import DTM

    train_ds, test_ds, ohtl = latent_ds
    n_label_nodes = int(train_ds["label"].shape[1])

    # register a SMALL smoke dataset so is_smoke_test=True (eval→_smoke_eval_epoch, no 28-FID) AND the
    # forced batch_size (= registered length) is small → fast real-graph train step.
    rng = np.random.default_rng(seed)
    smoke_batch = 50
    smoke_test_data_dict[(SMOKE_N_IMG, SMOKE_NGRAY, SMOKE_NCLS)] = {
        "image": jnp.asarray(rng.integers(0, 2, (smoke_batch, SMOKE_N_IMG)), dtype=jnp.bool_),
        "label": jnp.asarray(rng.integers(0, SMOKE_NCLS, (smoke_batch,)), dtype=jnp.int32),
    }
    cfg = make_companion_cfg(
        data=dict(dataset_name=f"smoke_testing_{SMOKE_N_IMG}_{SMOKE_NGRAY}_{SMOKE_NCLS}",
                  target_classes=tuple(range(SMOKE_NCLS)),
                  pixel_threshold_for_single_trials=0.1),
        exp=dict(seed=int(seed), compute_autocorr=False, generate_gif=False, n_cores=1),
        # keep the registered smoke label space tiny (the constructor uses it before injection)
        graph=dict(num_label_spots=1),
        # small generation warmup so .generate is cheap (28-FID bypassed regardless)
        generation=dict(steps_warmup=40, fid_images_per_digit=8),
    )
    dtm = DTM(cfg)

    # seam-A: inject the REAL encoded latent dataset (overrides the empty smoke test dict).
    # The adapter returns NUMPY bool arrays; the ACP compute_autocorr path (step.py:387-395) closes
    # over test_images/test_labels inside a jitted inner_fn and indexes them by a TRACER (rand_idx), so
    # numpy arrays trigger TracerArrayConversionError.  train_step_model similarly traces the data.
    # → inject as JAX arrays (matches upstream load_dataset / the smoke_test_data_dict fixtures).
    dtm.train_dataset = {k: jnp.asarray(v) for k, v in train_ds.items()}
    dtm.test_dataset = {k: jnp.asarray(v) for k, v in test_ds.items()}
    dtm.one_hot_target_labels = jnp.asarray(ohtl)
    dtm.n_image_pixels = int(train_ds["image"].shape[1])
    dtm.n_label_nodes = n_label_nodes
    # With evaluate_every=0 the train() logging-dir setup (DTM.py:262-274) is skipped, leaving
    # self.log_file == '' (the class default).  The per-epoch ACP/cp/wd write(..., self.log_file)
    # calls (DTM.py:303/351/364) then do open('') → FileNotFoundError.  Setting log_file=None makes
    # utils.write print-only (it only opens a file when log_path is not None).  (Surfaced by the smoke.)
    dtm.log_file = None
    return dtm


def fid_on_decoded(decoded_28x28):
    """Compute FID on decoded 28×28×1 ∈ [0,1] via the vendored bootstrap_fid_fn (network-free after
    the Task-10 P1 patch).  Returns (fid, term1, term2)."""
    from thrmlDenoising.fid.fid import bootstrap_fid_fn

    imgs = np.asarray(decoded_28x28, dtype=np.float32).reshape(-1, 28, 28, 1)
    return bootstrap_fid_fn(imgs, str(BW_NPZ_PATH))


def make_decode_fn(ae_params):
    def decode_fn(latent_spins):
        return ae_decode(ae_params, jnp.asarray(latent_spins, jnp.float32))
    return decode_fn
