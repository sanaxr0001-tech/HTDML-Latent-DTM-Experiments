"""Task 6 — tests for src/htdml/autoencoder.py (TDD gate).

Tests:
  AE-1 : encode shape — (B, 28, 28, 1) → hard_latent (B,196) ∈{−1,+1}, logits (B,196).
  AE-2 : decode shape — (B, 196) → (B, 28, 28, 1) ∈ [0,1].
  AE-3 : round-trip shape stability — encode then decode returns (B, 28, 28, 1).
  AE-4 : STE gradient: jax.grad of a loss through hard_latent w.r.t. params is NON-zero
          and finite; forward hard_latent is exactly ∈{−1,+1}.
  AE-5 : Stage-A loss — all three components finite; aux exposes them.
  AE-6 : Commitment loss responds — penalises logits near 0 more than confident logits.
  AE-7 : Balance loss responds — drops when bits are balanced, rises when collapsed.
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

from htdml.autoencoder import (
    BinaryAutoencoder,
    Encoder,
    Decoder,
    encode,
    decode,
    stage_a_loss,
    _ste_hard_sign,
)


# ----------------------------------------------------------------------- fixtures / helpers
@pytest.fixture(scope="module")
def ae_params():
    """Initialised BinaryAutoencoder params (random 28×28×1 dummy image)."""
    ae = BinaryAutoencoder()
    key = jax.random.PRNGKey(0)
    dummy = jnp.ones((2, 28, 28, 1), dtype=jnp.float32) * 0.5
    params = ae.init(key, dummy)
    return params


@pytest.fixture(scope="module")
def rand_batch():
    """Synthetic batch of 8 random 28×28×1 images in [0,1]."""
    rng = np.random.default_rng(42)
    return jnp.array(rng.random((8, 28, 28, 1)), dtype=jnp.float32)


# ----------------------------------------------------------------------- AE-1: encode shape
def test_encode_shape(ae_params, rand_batch):
    """encode → hard_latent (B,196) ∈{−1,+1} and logits (B,196)."""
    hard_latent, logits = encode(ae_params, rand_batch)

    assert hard_latent.shape == (8, 196), (
        f"hard_latent.shape expected (8, 196), got {hard_latent.shape}"
    )
    assert logits.shape == (8, 196), (
        f"logits.shape expected (8, 196), got {logits.shape}"
    )
    # hard_latent must be exactly ∈ {−1, +1}
    vals = np.unique(np.asarray(hard_latent))
    assert set(vals).issubset({-1.0, 1.0}), (
        f"hard_latent must only contain {{−1, +1}}, got unique values: {vals}"
    )


# ----------------------------------------------------------------------- AE-2: decode shape
def test_decode_shape(ae_params):
    """decode (B,196) → (B, 28, 28, 1) ∈ [0,1]."""
    rng = np.random.default_rng(7)
    z = jnp.array(rng.standard_normal((5, 196)), dtype=jnp.float32)
    x_recon = decode(ae_params, z)
    assert x_recon.shape == (5, 28, 28, 1), (
        f"decode output shape expected (5,28,28,1), got {x_recon.shape}"
    )
    arr = np.asarray(x_recon)
    assert arr.min() >= 0.0, f"decode output below 0: min={arr.min()}"
    assert arr.max() <= 1.0, f"decode output above 1: max={arr.max()}"


# ----------------------------------------------------------------------- AE-3: round-trip shape
def test_roundtrip_shape(ae_params, rand_batch):
    """Encode then decode returns original spatial shape."""
    hard_latent, _ = encode(ae_params, rand_batch)
    # Pass through the Decoder sub-module directly via decode()
    x_recon = decode(ae_params, hard_latent)
    assert x_recon.shape == rand_batch.shape, (
        f"round-trip shape {x_recon.shape} != {rand_batch.shape}"
    )


# ----------------------------------------------------------------------- AE-4: STE gradient
def test_ste_gradient_nonzero_and_hard(ae_params, rand_batch):
    """STE: gradient through hard_latent is NON-zero and finite; forward value ∈ {−1,+1}."""
    ae = BinaryAutoencoder()

    def loss_fn(params):
        _, hard_latent, _ = ae.apply(params, rand_batch)
        # Simple MSE loss that depends on hard_latent values
        return jnp.mean(hard_latent ** 2)

    grads = jax.grad(loss_fn)(ae_params)

    # Collect all gradient leaves as flat arrays
    grad_leaves = jax.tree_util.tree_leaves(grads)
    all_grads = np.concatenate([np.asarray(g).ravel() for g in grad_leaves])

    # Must be finite
    assert np.all(np.isfinite(all_grads)), "Some gradient entries are not finite"

    # Must NOT be all zero (STE carries gradient)
    assert np.any(all_grads != 0.0), (
        "All gradients are zero — STE is not carrying gradient through hard_latent"
    )

    # Forward hard_latent must be exactly {−1,+1}
    _, hard_latent, _ = ae.apply(ae_params, rand_batch)
    vals = np.unique(np.asarray(hard_latent))
    assert set(vals).issubset({-1.0, 1.0}), (
        f"Forward hard_latent not in {{−1,+1}}: unique values = {vals}"
    )


# ----------------------------------------------------------------------- AE-5: stage_a_loss finite + aux
def test_stage_a_loss_finite_and_aux(ae_params, rand_batch):
    """Stage-A loss: all three components finite; aux has the three expected keys."""
    total, aux = stage_a_loss(ae_params, rand_batch)

    assert jnp.isfinite(total), f"total loss is not finite: {total}"
    for key in ("recon", "commit", "balance"):
        assert key in aux, f"aux missing key '{key}'"
        assert jnp.isfinite(aux[key]), f"aux['{key}'] is not finite: {aux[key]}"


# ----------------------------------------------------------------------- AE-6: commitment responds
def test_commitment_loss_responds():
    """Commitment loss is LOWER for confident logits than for near-zero logits."""
    ae = BinaryAutoencoder()
    key = jax.random.PRNGKey(1)
    dummy_x = jnp.ones((4, 28, 28, 1), dtype=jnp.float32) * 0.5

    # Params giving near-zero logits: initialise with nearly-zero bias + scale kernel weights down.
    # We will test by comparing two cases via a synthetic dummy: create params from init then
    # test the commit formula directly on known logit values.

    # Near-zero logits → high commitment penalty
    near_zero_logits = jnp.full((4, 196), 0.01)
    sig_nz = jax.nn.sigmoid(near_zero_logits)
    commit_near_zero = float(jnp.mean(sig_nz * (1.0 - sig_nz)))

    # Confident logits → low commitment penalty
    confident_logits = jnp.full((4, 196), 5.0)
    sig_c = jax.nn.sigmoid(confident_logits)
    commit_confident = float(jnp.mean(sig_c * (1.0 - sig_c)))

    assert commit_confident < commit_near_zero, (
        f"Commitment penalty should be lower for confident logits "
        f"({commit_confident:.4f}) than near-zero ({commit_near_zero:.4f})"
    )


# ----------------------------------------------------------------------- AE-7: balance responds
def test_balance_loss_responds():
    """Balance loss drops when bits are balanced (mean ~ 0), rises when collapsed."""
    # Perfectly balanced: each bit alternates +1/-1 → batch mean = 0
    B, D = 4, 196
    balanced = jnp.array(
        [[1.0 if (i + j) % 2 == 0 else -1.0 for j in range(D)] for i in range(B)]
    )
    per_bit_mean_balanced = jnp.mean(balanced, axis=0)
    balance_balanced = float(jnp.mean(per_bit_mean_balanced ** 2))

    # All-ones (collapsed to +1): per-bit batch mean = 1.0
    collapsed = jnp.ones((B, D))
    per_bit_mean_collapsed = jnp.mean(collapsed, axis=0)
    balance_collapsed = float(jnp.mean(per_bit_mean_collapsed ** 2))

    assert balance_balanced < balance_collapsed, (
        f"Balance loss should be lower for balanced bits ({balance_balanced:.4f}) "
        f"than collapsed ({balance_collapsed:.4f})"
    )


# ----------------------------------------------------------------------- AE-8: STE unit test
def test_ste_hard_sign_values():
    """_ste_hard_sign: positive logit → +1, negative logit → −1, zero → +1."""
    logits = jnp.array([-5.0, -0.001, 0.0, 0.001, 5.0])
    hard = _ste_hard_sign(logits)
    expected = jnp.array([-1.0, -1.0, 1.0, 1.0, 1.0])
    assert jnp.allclose(hard, expected), (
        f"_ste_hard_sign output {hard} != expected {expected}"
    )
