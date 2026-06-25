"""Task 7 — tests for src/htdml/latent_dtm.py (LatentDTM) + the shared-coin caller threading.

CPU ONLY — NO ``dtm.train`` (it HARD-REQUIRES a GPU; build-notes §"CPU vs GPU").  The
``.generate`` SAMPLING path is exercised on a CPU-built + perturbed DTM.

Tests
-----
  LD-1  : the COMPANION cfg (built via make_companion_cfg) sets every PINS value exactly.
  LD-2  : .generate on a small CPU perturbed DTM: sample → 196-bit latent ∈ {−1,+1}.
  LD-3  : .generate decode path → 28×28×1 ∈ [0,1] of the right shape.
  LD-4  : .generate label selection returns only the requested class rows.
  LD-5  : shared-coin threading — estimate_moments accepts + propagates order_key;
          a SHARED order_key actually reaches sample_blocks (traced), order_key=None →
          per-chain default path unchanged.
  LD-6  : T1/T2/T3 — symmetric_kl_grad splits + passes shared order keys to pos/neg vmaps.
  LD-7  : DB cert STILL passes after the threading (reversibility not broken).
  LD-8  : .fit asserts the reversible kernel is live (without running dtm.train).
"""

from __future__ import annotations

import inspect
import sys
from pathlib import Path

import numpy as np
import pytest

_REPO_ROOT = Path(__file__).resolve().parents[1]
_SRC = str(_REPO_ROOT / "src")
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

import htdml  # noqa: E402,F401  (triggers bootstrap_paths)

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import jax.random as jr  # noqa: E402
import equinox as eqx  # noqa: E402

from htdml.latent_dtm import (  # noqa: E402
    LatentDTM,
    make_companion_cfg,
    COMPANION_CFG,
)
from htdml.autoencoder import BinaryAutoencoder, decode as ae_decode  # noqa: E402

_CPU = jax.devices("cpu")[0]


# ============================================================================== LD-1: cfg values
def test_companion_cfg_matches_pins():
    """The built companion cfg sets every FROZEN PINS value exactly (build-notes §config table)."""
    cfg = make_companion_cfg()

    # --- graph ---
    assert cfg.graph.graph_preset_architecture == 4412   # 44_12 == int 4412
    assert cfg.graph.num_label_spots == 5
    assert cfg.graph.grayscale_levels == 1
    assert cfg.graph.torus is True
    assert cfg.graph.base_graph_manager == "poisson_binomial_ising_graph_manager"

    # --- sampling (K=50 / stride 8 / B=400 convention) ---
    assert cfg.sampling.batch_size == 400
    assert cfg.sampling.n_samples == 50
    assert cfg.sampling.steps_per_sample == 8
    assert cfg.sampling.steps_warmup == 400
    assert cfg.sampling.steps_warmup == cfg.sampling.n_samples * cfg.sampling.steps_per_sample
    assert cfg.sampling.training_beta == 1.0

    # --- diffusion schedule (companion: 4 steps, log) ---
    assert cfg.diffusion_schedule.num_diffusion_steps == 4
    assert cfg.diffusion_schedule.kind == "log"
    assert cfg.diffusion_schedule.diffusion_offset == 0.1

    # --- diffusion rates ---
    assert cfg.diffusion_rates.image_rate == 0.8
    assert cfg.diffusion_rates.label_rate == 0.2

    # --- optim ---
    assert cfg.optim.step_learning_rates == (0.05,)

    # --- correlation penalty (ACP on, seeded at cp_min, padded to 4) ---
    assert cfg.cp.adaptive_cp is True
    assert cfg.cp.cp_min == 0.001
    assert cfg.cp.adaptive_threshold == 0.016
    assert cfg.cp.correlation_penalty == (0.001,) * 4
    assert all(c == cfg.cp.cp_min for c in cfg.cp.correlation_penalty)


def test_companion_cfg_constant_table():
    """The COMPANION_CFG module constant carries the frozen graph + cp divergences."""
    assert COMPANION_CFG["graph"]["graph_preset_architecture"] == 4412
    assert COMPANION_CFG["diffusion_schedule"]["num_diffusion_steps"] == 4
    assert COMPANION_CFG["cp"]["adaptive_cp"] is True
    assert COMPANION_CFG["cp"]["correlation_penalty"] == (0.001,) * 4


# ============================================================================== CPU DTM fixtures
_N_IMG = 196
_NGRAY = 1
_NCLS = 2


def _perturb(step, scale=0.5, seed=123):
    """Perturb step weights EXACTLY as DTM.train's write-back does (refreshes the generation
    programs so .generate reads trained-≠-init weights; leaves model.factors stale — the
    faithful exp15/16 reproduction)."""
    from thrmlDenoising.sampling_specs import get_new_per_block_interactions

    k = jr.PRNGKey(seed)
    w1 = step.model.weights + scale * jr.normal(k, step.model.weights.shape)
    b1 = step.model.biases + scale * jr.normal(jr.fold_in(k, 1), step.model.biases.shape)
    npos = get_new_per_block_interactions(step.training_spec.program_positive, w1, b1)
    nneg = get_new_per_block_interactions(step.training_spec.program_negative, w1, b1)
    nfree = get_new_per_block_interactions(step.generation_spec.program_free, w1, b1)
    ncond = get_new_per_block_interactions(step.generation_spec.program_conditioned, w1, b1)
    return eqx.tree_at(
        lambda s: (s.model.weights, s.model.biases,
                   s.training_spec.program_positive.per_block_interactions,
                   s.training_spec.program_negative.per_block_interactions,
                   s.generation_spec.program_free.per_block_interactions,
                   s.generation_spec.program_conditioned.per_block_interactions),
        step, (w1, b1, npos, nneg, nfree, ncond))


@pytest.fixture(scope="module")
def cpu_perturbed_ldtm():
    """A small CPU-built, perturbed LatentDTM at the 44_12 preset (tiny sampling) with an AE decoder.

    Built entirely under ``jax.default_device(cpu)`` so weights live on CPU and the generate
    sampling path NEVER touches the GPU (no dtm.train; CPU constraint honored)."""
    from thrmlDenoising.utils import smoke_test_data_dict
    from thrmlDenoising.DTM import DTM

    with jax.default_device(_CPU):
        # register a 196-wide smoke dataset entry (bypass tfds)
        rng = np.random.default_rng(0)
        smoke_test_data_dict[(_N_IMG, _NGRAY, _NCLS)] = {
            "image": jnp.array(rng.integers(0, 2, (20, _N_IMG)), dtype=jnp.bool_),
            "label": jnp.array(rng.integers(0, _NCLS, (20,)), dtype=jnp.int32),
        }
        cfg = make_companion_cfg(
            data=dict(dataset_name=f"smoke_testing_{_N_IMG}_{_NGRAY}_{_NCLS}",
                      target_classes=tuple(range(_NCLS)),
                      pixel_threshold_for_single_trials=0.1),
            exp=dict(seed=0, compute_autocorr=False, generate_gif=False, n_cores=1),
            graph=dict(num_label_spots=1),                       # smaller label space for the test
            sampling=dict(batch_size=4, n_samples=2, steps_per_sample=1, steps_warmup=2),
            generation=dict(steps_warmup=3),
        )
        dtm = DTM(cfg)
        dtm.steps = [_perturb(s) for s in dtm.steps]

        ae = BinaryAutoencoder()
        ae_params = ae.init(jax.random.PRNGKey(7), jnp.ones((2, 28, 28, 1), jnp.float32) * 0.5)

        def decode_fn(latent_spins):
            return ae_decode(ae_params, jnp.asarray(latent_spins, jnp.float32))

        ldtm = LatentDTM(dtm, decode_fn=decode_fn)
    return ldtm


# ============================================================================== LD-2/3/4: generate
def test_generate_raw_latent_is_196_bit_spins(cpu_perturbed_ldtm):
    """LD-2: .generate(decode=False) returns 196-wide latents ∈ {−1,+1}."""
    with jax.default_device(_CPU):
        raw = cpu_perturbed_ldtm.generate(
            jax.random.PRNGKey(1), labels=None, samples_per_label=2, free=False, decode=False
        )
    raw = np.asarray(raw)
    assert raw.shape == (_NCLS, 2, _N_IMG), f"raw latent shape {raw.shape}"
    assert set(np.unique(raw)).issubset({-1.0, 1.0}), (
        f"latent values not ±1 spins: {np.unique(raw)}"
    )


def test_generate_decoded_is_28x28_in_unit_range(cpu_perturbed_ldtm):
    """LD-3: .generate(decode=True) → decoded 28×28×1 ∈ [0,1] of the right shape."""
    with jax.default_device(_CPU):
        imgs = cpu_perturbed_ldtm.generate(
            jax.random.PRNGKey(2), labels=[0], samples_per_label=2, free=False, decode=True
        )
    imgs = np.asarray(imgs)
    assert imgs.shape == (1, 2, 28, 28, 1), f"decoded shape {imgs.shape}"
    assert imgs.min() >= 0.0 and imgs.max() <= 1.0, (
        f"decoded pixels outside [0,1]: [{imgs.min()}, {imgs.max()}]"
    )


def test_generate_label_selection(cpu_perturbed_ldtm):
    """LD-4: label selection returns exactly the requested class rows."""
    with jax.default_device(_CPU):
        all_raw = cpu_perturbed_ldtm.generate(
            jax.random.PRNGKey(3), labels=None, samples_per_label=2, free=False, decode=False
        )
        one_raw = cpu_perturbed_ldtm.generate(
            jax.random.PRNGKey(3), labels=[1], samples_per_label=2, free=False, decode=False
        )
    all_raw = np.asarray(all_raw)
    one_raw = np.asarray(one_raw)
    assert all_raw.shape[0] == _NCLS
    assert one_raw.shape[0] == 1
    # same key → row 1 of the full batch equals the single selected row
    assert np.array_equal(one_raw[0], all_raw[1]), "label selection did not slice the right row"


# ============================================================================== LD-5/6: shared-coin
def test_estimate_moments_accepts_and_threads_order_key():
    """LD-5: M1/M2 — estimate_moments accepts order_key AND passes it to sample_with_observation."""
    import thrml.models.ising as ising

    sig = inspect.signature(ising.estimate_moments)
    assert "order_key" in sig.parameters, "estimate_moments missing order_key param (M1)"
    assert sig.parameters["order_key"].default is None, "order_key default must be None (per-chain)"

    src = inspect.getsource(ising.estimate_moments)
    assert "order_key=order_key" in src, (
        "estimate_moments does not thread order_key into sample_with_observation (M2)"
    )


def _trace_order_subkey(cpu_perturbed_ldtm, monkeypatch, order_key):
    """Run one real single-chain sample_with_observation through the tiny DTM's NEGATIVE program
    with the given ``order_key`` and capture every ``order_subkey`` ``sample_blocks`` receives."""
    import thrml.block_sampling as bs
    from thrml.block_sampling import SamplingSchedule, sample_with_observation
    from thrml.observers import StateObserver
    from thrmlDenoising.annealing_graph_ising import hinton_init_from_graph

    step = cpu_perturbed_ldtm.dtm.steps[0]
    prog = step.training_spec.program_negative
    spec = prog.gibbs_spec

    seen = []
    orig = bs.sample_blocks

    def _spy(key, state_free, clamp_state, program, sampler_state, order_subkey=None):
        seen.append(order_subkey)
        return orig(key, state_free, clamp_state, program, sampler_state, order_subkey=order_subkey)

    monkeypatch.setattr(bs, "sample_blocks", _spy)

    with jax.default_device(_CPU):
        k = jax.random.PRNGKey(0)
        # single-chain free init via hinton; clamped blocks = zeros of the right width
        init_state = hinton_init_from_graph(k, step.model, spec.free_blocks, 1, 1.0)
        init_state = [s[0] for s in init_state]  # drop the batch axis → single chain
        clamped_data = [jnp.zeros((len(b),), dtype=jnp.bool_) for b in spec.clamped_blocks]
        observer = StateObserver(spec.free_blocks)
        init_mem = observer.init()
        schedule = SamplingSchedule(1, 2, 1)  # 1 warmup, 2 samples, stride 1 → several sweeps
        sample_with_observation(
            k, prog, schedule, init_state, clamped_data, init_mem, observer, order_key=order_key
        )
    return seen


def test_shared_order_key_reaches_sample_blocks(cpu_perturbed_ldtm, monkeypatch):
    """LD-5: a SHARED order_key actually reaches sample_blocks as a NON-None order_subkey; an
    order_key=None call leaves every order_subkey None (per-chain default path unchanged)."""
    from harness import reversible_scan

    # _run_blocks must carry the per-sweep order_subkey when a shared order_key is given.
    import thrml.block_sampling as bs
    src_rb = inspect.getsource(bs._run_blocks)
    assert "order_key" in src_rb, "_run_blocks lost its order_key param"
    assert "order_subkey=_ok" in src_rb, "_run_blocks does not pass a per-sweep order_subkey when shared"

    shared_key = reversible_scan.make_order_key(jax.random.PRNGKey(7), reversible_scan.SHARED)
    seen_shared = _trace_order_subkey(cpu_perturbed_ldtm, monkeypatch, shared_key)
    assert len(seen_shared) > 0, "sample_blocks was never called"
    assert all(ok is not None for ok in seen_shared), (
        f"SHARED order_key did NOT reach sample_blocks (got order_subkey={seen_shared})"
    )

    seen_none = _trace_order_subkey(cpu_perturbed_ldtm, monkeypatch, None)
    assert len(seen_none) > 0, "sample_blocks was never called (None path)"
    assert all(ok is None for ok in seen_none), (
        f"order_key=None must leave order_subkey None (per-chain); got {seen_none}"
    )


def test_symmetric_kl_grad_threads_shared_order_keys():
    """LD-6: T1/T2/T3 — symmetric_kl_grad splits shared order keys + passes them to the vmaps."""
    import thrmlDenoising.ising_training as it

    src = inspect.getsource(it.symmetric_kl_grad)
    # T1: 4-way split producing key_order_pos / key_order_neg
    assert "key_order_pos" in src and "key_order_neg" in src, "T1 shared order-key split missing"
    assert "jax.random.split(key, 4)" in src, "T1 must split the key 4-ways (pos/neg/order_pos/order_neg)"
    # T2 / T3: the order keys are passed into the positive/negative vmaps
    assert "order_key=key_order_pos" in src, "T2: positive vmap missing order_key=key_order_pos"
    assert "order_key=key_order_neg" in src, "T3: negative vmap missing order_key=key_order_neg"


def test_default_order_key_is_per_chain():
    """LD-5: order_key=None (the diagnostic default) still gives the per-chain path."""
    from harness import reversible_scan
    assert reversible_scan.make_order_key(None, reversible_scan.PER_CHAIN) is None


# ============================================================================== LD-7: DB cert
def test_db_cert_still_passes_after_threading():
    """LD-7: the ½(P_AB+P_BA) DB certificate STILL passes (the shared-coin threading is a caller
    optimization that must NOT break reversibility)."""
    from harness import selfadjoint_cert

    res = selfadjoint_cert.certify(np.random.default_rng(0), sizes=(1, 1, 1, 1), verbose=False)
    assert res["passed"] is True
    assert res["max_asym"] < selfadjoint_cert.TOL_SYM, (
        f"DB residual {res['max_asym']:.2e} not < {selfadjoint_cert.TOL_SYM:.0e}"
    )


# ============================================================================== LD-8: .fit gate
def test_fit_asserts_kernel_live_without_training(cpu_perturbed_ldtm, monkeypatch):
    """LD-8: .fit checks the reversible kernel is live BEFORE training. We stub dtm.train so the
    CPU constraint (no real train) is honored, and confirm the kernel-live assert runs + the
    latents are injected."""
    from harness import reversible_scan

    # The kernel must genuinely be live in this build.
    live, detail = reversible_scan.is_patch_live()
    assert live, f"reversible kernel not live: {detail}"

    ldtm = cpu_perturbed_ldtm
    called = {"train": False, "args": None}

    def _fake_train(*, n_epochs, evaluate_every):
        called["train"] = True
        called["args"] = (n_epochs, evaluate_every)
        return "stub-train-return"

    monkeypatch.setattr(ldtm.dtm, "train", _fake_train)

    # a minimal latent triple (shapes don't need to be production; inject_latents just sets attrs)
    train_ds = {"image": np.zeros((3, 196), dtype=bool), "label": np.zeros((3, 2), dtype=bool)}
    test_ds = {"image": np.zeros((1000, 196), dtype=bool), "label": np.zeros((1000, 2), dtype=bool)}
    ohtl = np.zeros((2, 2), dtype=bool)

    out = ldtm.fit((train_ds, test_ds, ohtl), n_epochs=1, evaluate_every=1)
    assert out == "stub-train-return"
    assert called["train"] is True and called["args"] == (1, 1)
    assert ldtm.dtm.n_image_pixels == 196 and ldtm.dtm.n_label_nodes == 2
    # inject_latents now stores the dataset as JAX arrays (the Task-12 smoke surfaced that the ACP
    # compute_autocorr path indexes test_images by a tracer → numpy raises TracerArrayConversionError).
    # So check VALUE-equality + jax-array type, not object identity.
    assert isinstance(ldtm.dtm.train_dataset["image"], jnp.ndarray)
    assert np.array_equal(np.asarray(ldtm.dtm.train_dataset["image"]), train_ds["image"])
    assert np.array_equal(np.asarray(ldtm.dtm.train_dataset["label"]), train_ds["label"])
    assert np.array_equal(np.asarray(ldtm.dtm.test_dataset["image"]), test_ds["image"])
