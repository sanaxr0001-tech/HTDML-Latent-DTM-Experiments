"""LatentDTM — Task 7 wrapper around the vendored DTM for the 196-bit latent companion.

This wraps the vendored ``thrmlDenoising.DTM`` with the FROZEN 44_12 companion config
(see PINS.md / build-notes.md §"Verified config constants") and provides two operations:

  * ``.fit(latent_dataset)`` — inject the Task-6 latent-adapter dict (bypassing
    ``load_dataset``) and run ``dtm.train``.  **``dtm.train`` HARD-REQUIRES a GPU**, so
    ``.fit`` is WIRED here and exercised only at the smoke (Task 11); it is NEVER unit-tested
    on CPU (the CPU constraint forbids ``dtm.train``).

  * ``.generate(labels)`` — conditional annealing generation through the live reversible
    kernel → 196-bit latents → ``autoencoder.decode`` → 28×28 ∈ [0,1].  The SAMPLING path
    runs on CPU jax, so ``.generate`` is CPU-testable on a perturbed DTM (no ``dtm.train``).
    The 28-hard-coded FID / draw paths (``do_draw_and_fid`` fid.py:120, ``draw_image_batch``
    utils.py:207, ``generate_gif``) are BYPASSED — we reuse only ``dtm.gen_images`` (which is
    sized by ``self.n_image_pixels`` throughout and so survives a 196-wide latent), then take
    its ``images_for_fid`` return (the raw 196-bit per-condition latents) and decode in pixel
    space.  FID is computed on the DECODED 28×28 (Task 9 path), never on the 196-bit latent.

The reversible ½(P_AB+P_BA) kernel must be LIVE for training (assert
``harness.reversible_scan.is_patch_live()``).  The per-step trained-weight refresh
(``harness.probe_primitives.refresh_program_weights`` / the exp15/16 stale-factors bug fix)
is a Task-8 probe concern and is NOT applied here; ``.generate`` reads the generation
program's own ``per_block_interactions`` (which are refreshed by ``dtm.train``'s write-back,
and by the CPU perturbation helper in the tests).
"""

from __future__ import annotations

import sys
from pathlib import Path

# --- self-bootstrap: make `import htdml` work, then install the vendored path ordering ---------
_REPO_ROOT = Path(__file__).resolve().parents[2]
_SRC = str(_REPO_ROOT / "src")
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

from htdml.paths import bootstrap_paths  # noqa: E402

bootstrap_paths()

from typing import Optional, Sequence  # noqa: E402

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import jax.random as jr  # noqa: E402
import numpy as np  # noqa: E402


# ============================================================================== companion config
# Frozen companion config (PINS.md / build-notes.md §"Verified config constants").
# Companion divergences from upstream are flagged in PINS.md.
COMPANION_CFG = dict(
    graph=dict(
        graph_preset_architecture=44_12,  # == int 4412 (side 44, 1936 nodes); embeds 196+labels
        num_label_spots=5,
        grayscale_levels=1,
        torus=True,
        base_graph_manager="poisson_binomial_ising_graph_manager",
    ),
    sampling=dict(
        batch_size=400,
        n_samples=50,
        steps_per_sample=8,
        steps_warmup=400,
        training_beta=1.0,
    ),
    diffusion_schedule=dict(
        num_diffusion_steps=4,
        kind="log",
        diffusion_offset=0.1,
    ),
    diffusion_rates=dict(
        image_rate=0.8,
        label_rate=0.2,
    ),
    optim=dict(
        step_learning_rates=(0.05,),
    ),
    cp=dict(
        correlation_penalty=(0.001,) * 4,  # c0 = cp_min (ACP re-floors each adaptive epoch)
        adaptive_cp=True,
        cp_min=0.001,
        adaptive_threshold=0.016,
    ),
)


def make_companion_cfg(**extra):
    """Build the FROZEN companion ``DTMConfig`` via the vendored ``make_cfg``.

    ``extra`` is merged section-wise on TOP of ``COMPANION_CFG`` (e.g. to inject the
    ``data``/``exp`` sections the constructor needs to reach a 196-wide smoke dataset on CPU,
    or to override the graph preset for a CPU-cheap test).  The companion-pinned values are
    the defaults; ``extra`` only adds/overrides what the caller explicitly passes.
    """
    from thrmlDenoising.utils import make_cfg

    merged: dict[str, dict] = {k: dict(v) for k, v in COMPANION_CFG.items()}
    for section, vals in extra.items():
        merged.setdefault(section, {})
        merged[section].update(vals)
    return make_cfg(**merged)


# ============================================================================== LatentDTM wrapper
class LatentDTM:
    """Thin wrapper around the vendored ``DTM`` for the 196-bit latent companion study.

    Parameters
    ----------
    dtm:
        A pre-constructed vendored ``thrmlDenoising.DTM.DTM`` instance (built with the
        companion cfg).  Construction is the caller's job because the constructor's
        ``load_dataset`` call must be bypassed via the seam-A latent dataset (CPU) or the real
        dataset gate (GPU) — both happen OUTSIDE this wrapper so the wrapper stays config-pure
        and never re-runs ``load_dataset``.
    decode_fn:
        ``callable(latent_spins (B,196) {−1,+1}) -> (B,28,28,1) ∈ [0,1]`` — the trained
        autoencoder decoder (``functools.partial(autoencoder.decode, ae_params)`` composed with
        the bool→spin lift, or any closure with that signature).  Used by ``.generate`` to map
        sampled 196-bit latents back to pixel space.  May be ``None`` if only ``.fit`` is used.
    assert_kernel_live:
        If ``True`` (default), ``.fit`` asserts the reversible kernel patch is LIVE before
        training (build-notes §"Reversible kernel").
    """

    def __init__(self, dtm, decode_fn=None, *, assert_kernel_live: bool = True):
        self.dtm = dtm
        self.decode_fn = decode_fn
        self.assert_kernel_live = assert_kernel_live

    # ----------------------------------------------------------------------------- construction
    @classmethod
    def from_cfg(cls, cfg, decode_fn=None, *, assert_kernel_live: bool = True):
        """Construct a ``LatentDTM`` from an explicit ``DTMConfig`` (constructs the DTM here).

        NOTE: the constructor calls ``load_dataset`` — the caller must have arranged a 196-wide
        dataset (real gate or smoke entry) so ``self.n_image_pixels == 196``.  Use ``.fit`` to
        inject the seam-A latent dict afterward.
        """
        from thrmlDenoising.DTM import DTM

        dtm = DTM(cfg)
        return cls(dtm, decode_fn=decode_fn, assert_kernel_live=assert_kernel_live)

    # ----------------------------------------------------------------------------------- .fit
    def inject_latents(self, latent_dataset) -> None:
        """Inject the Task-6 latent-adapter triple, bypassing ``load_dataset`` (seam A).

        ``latent_dataset`` is ``(train_ds, test_ds, one_hot_target_labels)`` from
        ``latent_adapter.build_latent_dataset`` (build-notes §"Fork Seam A").  Sets the DTM's
        dataset attributes + the derived ``n_image_pixels``/``n_label_nodes`` so the test-dict
        assertions (DTM.py:675-680, step.py:393) and ACP autocorr path are satisfied.
        """
        train_ds, test_ds, ohtl = latent_dataset
        self.dtm.train_dataset = train_ds
        self.dtm.test_dataset = test_ds
        self.dtm.one_hot_target_labels = ohtl
        self.dtm.n_image_pixels = int(train_ds["image"].shape[1])
        self.dtm.n_label_nodes = int(train_ds["label"].shape[1])

    def fit(self, latent_dataset, *, n_epochs: int, evaluate_every: int):
        """Inject the latent dict and run ``dtm.train`` (GPU-only).

        WARNING: ``dtm.train`` HARD-REQUIRES a GPU (build-notes §"CPU vs GPU").  This method is
        wired here and exercised at the smoke (Task 11) — it is NEVER called in the CPU unit
        tests.  ``.fit`` asserts the reversible kernel is LIVE before training so the negative
        phase uses the ½(P_AB+P_BA) kernel.

        Returns whatever ``dtm.train`` returns.
        """
        if self.assert_kernel_live:
            from harness import reversible_scan

            live, detail = reversible_scan.is_patch_live()
            assert live, f"reversible kernel NOT live — refusing to train: {detail}"

        self.inject_latents(latent_dataset)
        return self.dtm.train(n_epochs=n_epochs, evaluate_every=evaluate_every)

    # ---------------------------------------------------------------------------- .generate
    def generate(
        self,
        key,
        labels: Optional[Sequence[int]] = None,
        *,
        samples_per_label: int = 1,
        free: bool = False,
        decode: bool = True,
    ):
        """Conditional (or free) annealing generation → 196-bit latents → decoded 28×28.

        Reuses ``dtm.gen_images`` (the FID/draw-free annealing sampler; sized by
        ``self.n_image_pixels`` throughout → survives a 196-wide latent) and takes its
        ``images_for_fid`` return: the FINAL-step per-condition latents, shape
        ``(n_target_classes, samples_per_label, n_image_pixels)`` with values in {0, 1}
        (grayscale_levels=1).  These are mapped bit→spin ({−1,+1}) and (optionally) decoded to
        28×28 ∈ [0,1] via ``decode_fn``.

        The 28-hard-coded paths (``do_draw_and_fid``/``draw_image_batch``/``generate_gif``) are
        NOT touched.

        Parameters
        ----------
        key:
            jax PRNG key.
        labels:
            Optional sequence of class indices (positions into ``one_hot_target_labels``,
            i.e. into the companion's ``target_classes`` ordering) to RETURN.  If ``None``,
            returns every class.  Only meaningful when ``free=False``.
        samples_per_label:
            Number of independent chains per condition (``batch_size`` of ``gen_images``).
        free:
            Free (unconditional) generation if ``True``; conditional on the target labels if
            ``False`` (default).
        decode:
            If ``True`` (default) and ``decode_fn`` is set, returns decoded 28×28 images.
            If ``False``, returns the raw 196-bit latents (spins) — useful for tests/inspection.

        Returns
        -------
        If ``decode`` and ``decode_fn`` set:
            ``x_recon`` array, shape ``(n_returned, samples_per_label, 28, 28, 1)`` ∈ [0,1].
        Else:
            ``latent_spins`` array, shape ``(n_returned, samples_per_label, n_image_pixels)``
            ∈ {−1, +1}.
        """
        n_image_pixels = int(self.dtm.n_image_pixels)

        # gen_images returns (to_draw, to_draw_labels, images_for_fid).  images_for_fid is the
        # FINAL-step latent batch: (n_target_classes, batch_size, n_image_pixels), normalized
        # to /n_grayscale_levels (== /1 here → values in {0,1}).  drawn_images_per_digit only
        # affects to_draw (a slice we discard), so pass samples_per_label to keep the slice valid.
        _, _, images_for_fid = self.dtm.gen_images(
            key,
            batch_size=samples_per_label,
            free=free,
            drawn_images_per_digit=samples_per_label,
        )
        latent = np.asarray(images_for_fid)  # (n_classes, samples_per_label, n_image_pixels)
        assert latent.shape[-1] == n_image_pixels, (
            f"gen_images latent width {latent.shape[-1]} != n_image_pixels {n_image_pixels}"
        )

        # Select the requested class rows.
        if labels is not None and not free:
            idx = np.asarray(list(labels), dtype=np.int64)
            latent = latent[idx]

        # bit {0,1} → spin {−1,+1}  (production-faithful: ising.py:204 `2x−1`).
        latent_spins = (2.0 * latent - 1.0).astype(np.float32)

        if not decode or self.decode_fn is None:
            return latent_spins

        # Decode each (class, chain) latent through the autoencoder decoder → 28×28.
        n_ret, n_chain = latent_spins.shape[0], latent_spins.shape[1]
        flat = jnp.asarray(latent_spins).reshape(n_ret * n_chain, n_image_pixels)
        x_recon = self.decode_fn(flat)  # (n_ret*n_chain, 28, 28, 1)
        x_recon = jnp.asarray(x_recon)
        return x_recon.reshape(n_ret, n_chain, *x_recon.shape[1:])
