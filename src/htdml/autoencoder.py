"""Spatial binary autoencoder with straight-through estimator (STE) — Task 6.

Architecture
------------
  Encoder : 28×28×1 conv → 14×14×1 → flatten → 196 logits → STE hard-sign ∈ {−1,+1}
  Decoder : 196 → 14×14×1 → 28×28×1 → sigmoid ∈ [0,1]

STE convention (forward = hard, backward = smooth surrogate)
------------------------------------------------------------
  hard_latent = tanh(logit) + stop_gradient(sign(logit) − tanh(logit))
  forward:  hard_latent = sign(logit) ∈ {−1, +1}  (since tanh(...) + (sign − tanh) = sign)
  backward: ∂hard_latent/∂logit = ∂tanh(logit)/∂logit = 1 − tanh²(logit)  (smooth surrogate)

Stage-A loss components
-----------------------
  1. Reconstruction BCE : BCE(decode(hard_latent), x)  where x ∈ [0,1] (float input).
  2. Binary commitment  : mean_over_bits[ σ(logit)·(1−σ(logit)) ]
                         → 0 when logits are confident (pushed away from 0).
  3. Bit balance        : mean_over_bits[ (mean_batch(hard_latent_b))² ]
                         → 0 when each bit has mean ~0 across the batch (balanced ±1).

All three components are exposed in the `aux` dict returned by `stage_a_loss`.
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

from typing import Tuple

import jax
import jax.numpy as jnp
import flax.linen as nn


# --------------------------------------------------------------------------- architecture
class Encoder(nn.Module):
    """Conv encoder: 28×28×1 → 14×14×1 → flatten → 196 logits.

    Uses a single 2D conv with stride=2 to halve spatial dimensions, producing
    14×14=196 features per image.  A bias term is included so the network can
    express balanced logits from the start.
    """

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        """Return latent logits (196,) for a single 28×28×1 image (or (B,28,28,1) batch).

        The caller is responsible for STE; this module returns the RAW logits only.
        """
        # x: (..., 28, 28, 1)
        x = nn.Conv(features=1, kernel_size=(4, 4), strides=(2, 2), padding="SAME")(x)
        # x: (..., 14, 14, 1)
        x = nn.relu(x)
        # Flatten spatial dims → 14*14*1 = 196
        batch_shape = x.shape[:-3]
        x = x.reshape(batch_shape + (196,))
        # One dense layer so the latent is a learned linear combination of conv features.
        x = nn.Dense(196)(x)
        return x  # logits: (..., 196)


class Decoder(nn.Module):
    """Decoder: 196 → 14×14×1 → 28×28×1, output ∈ [0,1].

    Uses a dense layer to reshape to 14×14×1, then a transposed-conv (or
    upsampled conv) to recover 28×28, followed by sigmoid.
    """

    @nn.compact
    def __call__(self, z: jnp.ndarray) -> jnp.ndarray:
        """Return reconstructed image (..., 28, 28, 1) ∈ [0,1] from latent (..., 196)."""
        batch_shape = z.shape[:-1]
        # Project to 14×14×4 feature map
        x = nn.Dense(14 * 14 * 4)(z)
        x = nn.relu(x)
        x = x.reshape(batch_shape + (14, 14, 4))
        # ConvTranspose to 28×28×1
        x = nn.ConvTranspose(features=1, kernel_size=(4, 4), strides=(2, 2), padding="SAME")(x)
        x = jax.nn.sigmoid(x)
        return x  # (..., 28, 28, 1)


class BinaryAutoencoder(nn.Module):
    """Spatial binary autoencoder (encoder + decoder) as a single Flax module.

    The module exposes:
      - `encode(x)` → (hard_latent, latent_logits)
      - `decode(z)` → x_recon
      - forward `__call__(x)` → (x_recon, hard_latent, latent_logits)
    """

    @nn.compact
    def __call__(
        self, x: jnp.ndarray
    ) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        """Forward pass: x → (x_recon, hard_latent, latent_logits).

        Args:
            x: Input image (..., 28, 28, 1) as float32 ∈ [0,1].
        Returns:
            x_recon     : (..., 28, 28, 1) ∈ [0,1]
            hard_latent : (..., 196) ∈ {−1, +1}  — the STE hard binary code
            latent_logits: (..., 196) ∈ ℝ        — pre-STE logits
        """
        logits = Encoder()(x)
        hard_latent = _ste_hard_sign(logits)
        x_recon = Decoder()(hard_latent)
        return x_recon, hard_latent, logits


# --------------------------------------------------------------------------- STE
def _ste_hard_sign(logits: jnp.ndarray) -> jnp.ndarray:
    """Straight-through estimator: forward = hard sign ∈ {−1,+1}, backward = tanh surrogate.

    Implementation:
        hard = tanh(logit) + stop_gradient(sign(logit) − tanh(logit))
    Forward evaluation: tanh(logit) + (sign(logit) − tanh(logit)) = sign(logit) ∈ {−1,+1}
    Backward gradient: ∂/∂logit = (1 − tanh²(logit))  (the smooth surrogate).
    sign(0) = 0 in jnp; to avoid a zero hard spin at exactly logit=0 we treat 0 as +1 by
    using jnp.where(logits >= 0, 1.0, -1.0) for the stop_gradient offset.
    """
    t = jnp.tanh(logits)
    hard = jax.lax.stop_gradient(jnp.where(logits >= 0, 1.0, -1.0))
    return t + (hard - jax.lax.stop_gradient(t))


# --------------------------------------------------------------------------- encode / decode helpers
def encode(params, x: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Encode a batch of images.

    Args:
        params: Flax param dict (from `BinaryAutoencoder().init(...)`).
        x:      (B, 28, 28, 1) float32 ∈ [0,1].
    Returns:
        hard_latent   : (B, 196) ∈ {−1, +1}
        latent_logits : (B, 196) ∈ ℝ
    """
    _, hard_latent, logits = BinaryAutoencoder().apply(params, x)
    return hard_latent, logits


def decode(params, z: jnp.ndarray) -> jnp.ndarray:
    """Decode a batch of latent vectors.

    Args:
        params: Flax param dict.
        z:      (B, 196) latent vectors (any real values; the decoder is a plain Dense+ConvTranspose).
    Returns:
        x_recon: (B, 28, 28, 1) ∈ [0,1]
    """
    return Decoder().apply({"params": params["params"]["Decoder_0"]}, z)


# --------------------------------------------------------------------------- Stage-A loss
def stage_a_loss(
    params,
    x: jnp.ndarray,
    *,
    w_recon: float = 1.0,
    w_commit: float = 0.1,
    w_balance: float = 0.1,
):
    """Stage-A self-supervised pre-training loss.

    Three components (all scalars):
      1. recon_loss   : Binary cross-entropy between decode(hard_latent) and x.
      2. commit_loss  : σ(logit)·(1−σ(logit)) per bit, mean across batch + bits.
                        Penalizes logits near 0 → pushes toward confident ±∞.
      3. balance_loss : (mean_batch(hard_bit_b))² per bit, mean across bits.
                        Penalizes any bit that is consistently +1 or consistently −1.
                        In the {−1,+1} convention the ideal per-bit batch-mean is 0.

    Args:
        params   : BinaryAutoencoder param dict.
        x        : (B, 28, 28, 1) float32 ∈ [0,1] — target images.
        w_recon  : weight for recon_loss.
        w_commit : weight for commit_loss.
        w_balance: weight for balance_loss.

    Returns:
        (total_loss, aux)  where aux = {"recon": ..., "commit": ..., "balance": ...}
    """
    ae = BinaryAutoencoder()
    x_recon, hard_latent, logits = ae.apply(params, x)

    # 1. Reconstruction BCE: x ∈ [0,1], x_recon ∈ [0,1].
    #    BCE(y_hat, y) = −[y·log(ŷ) + (1−y)·log(1−ŷ)]  (clipped for stability)
    eps = 1e-7
    x_recon_c = jnp.clip(x_recon, eps, 1.0 - eps)
    recon_loss = -jnp.mean(
        x * jnp.log(x_recon_c) + (1.0 - x) * jnp.log(1.0 - x_recon_c)
    )

    # 2. Binary commitment: σ(logit)·(1−σ(logit)) — maximised at logit=0, zero at ±∞.
    sig = jax.nn.sigmoid(logits)
    commit_loss = jnp.mean(sig * (1.0 - sig))

    # 3. Bit balance: per-bit batch mean of hard latent, squared, then mean over bits.
    #    hard_latent: (B, 196) ∈ {−1,+1}; ideal per-bit mean = 0.
    per_bit_mean = jnp.mean(hard_latent, axis=0)  # (196,)
    balance_loss = jnp.mean(per_bit_mean ** 2)

    total = w_recon * recon_loss + w_commit * commit_loss + w_balance * balance_loss
    aux = {
        "recon": recon_loss,
        "commit": commit_loss,
        "balance": balance_loss,
    }
    return total, aux
