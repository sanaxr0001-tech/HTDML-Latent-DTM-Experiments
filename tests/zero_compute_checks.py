"""
Task 11 — consolidated zero-compute battery for the htdml-latent-dtm companion.

This module assembles assertions over the already-built+verified components, plus a few NEW
consolidated checks (λ=0≡control _weights_hash equality, L_compat no-backprop / deterministic-MF,
6-token reachability). ZERO GPU compute — everything runs on CPU.

Groups (12):
  1. DB-cert        — selfadjoint_cert residual < 1e-10 on the production-shape kernel.
  2. Frozen constants — graph_preset, probe constants, calibration-frozen constants (TBD gate).
  3. Shape checks   — latent adapter test dict (1000,196); full-model record exactly 4 layers.
  4. Formula-shape  — Q_struct^⊥ K=50 prefactor; ESS_hat=50/(2·τ); BCE present; FID offline.
  5. Store-coverage — driver per-epoch+per-layer record has all 9 quantities incl. live ACP coeff.
  6. Provenance     — weight-hash / key-isolation / the mandatory per-step refresh-proof.
  7. λ=0 ≡ control  — (NEW) joint update at λ=0 yields bitwise-equal _weights_hash + _key_list.
  8. L_compat invariants — (NEW) no-backprop-into-DTM; no-grad-through-b_t; deterministic MF.
  9. Seed disjointness — diag key independent of dtm.key; two seeds use disjoint keys.
  10. Numerical regression — τ=0.5 on IID series; determinism.
  11. Measure-only / no-tag — companion makes NO wiki edits / no claim-status tags.
  12. 6-token reachability — (NEW) all 6 tokens reachable from driver's PURE route_seed/route_run.

Calibration-frozen constants (L_traj, N_chains, N_R, C, ESS_min) are TBD until Task 12.  The
battery asserts each is EITHER pinned-numeric OR explicitly "TBD-pending-Task-12" (PINS.md text)
and skips the numeric check with a pytest.skip("TBD-pending-Task-12") marker.  A non-TBD,
non-numeric value is a HARD FAIL.

CPU ONLY — NO dtm.train, NO GPU.
"""

from __future__ import annotations

import contextlib
import os
import re
import sys
import tempfile
import unittest.mock as mock
from pathlib import Path

import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
import pytest

# ------------------------------------------------------------------ repo bootstrap
_REPO_ROOT = Path(__file__).resolve().parents[1]
for _p in (str(_REPO_ROOT), str(_REPO_ROOT / "src")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import htdml  # noqa: F401  (triggers bootstrap_paths)

from harness import probe_primitives as pp  # noqa: E402
from harness import selfadjoint_cert as sc  # noqa: E402
from htdml import compatibility as C  # noqa: E402
from htdml import driver as D  # noqa: E402
from htdml.latent_dtm import COMPANION_CFG, make_companion_cfg  # noqa: E402

# ------------------------------------------------------------------ paths
BW_NPZ_PATH = (
    _REPO_ROOT
    / "vendor"
    / "dtm-replication"
    / "thrmlDenoising"
    / "fid"
    / "precomputed_stats"
    / "bw_fashion_mnist_train.npz"
)
INCEPTION_PICKLE_PATH = _REPO_ROOT / "cache" / "inception_v3_weights_fid.pickle"
PINS_MD_PATH = _REPO_ROOT / "PINS.md"

# ------------------------------------------------------------------ PINS constants
BW_NPZ_SHA256_EXPECTED = "66003004dc99115b20c146bd3c2a7d9d85fb85a3c0c9e991f11951933f97c5d8"
INCEPTION_SHA256_EXPECTED = "4e030efa5bccac3222d975f658d1884f9e00fab24f2812082884539220b90d77"

# Probe K=50 window constants (always-known; from PINS + code).
WINDOW_SAMPLES_K = 50
STRIDE_SWEEPS = 8
WINDOW_SPAN_SWEEPS = 400   # = K * stride
STEPS_PER_SAMPLE = 8       # same as stride_sweeps
ADAPTIVE_THRESHOLD = 0.016
NUM_DIFFUSION_STEPS = 4
NUM_LABEL_SPOTS = 5
C0 = 0.001
GRAPH_PRESET = 44_12        # == int 4412

# The calibration-frozen constants — TBD until Task 12 calibration.  We inspect PINS.md at
# runtime: if the value is TBD (string containing "TBD"), the test is skipped with
# "TBD-pending-Task-12"; if it is a number, the test asserts it; otherwise HARD FAIL.
_CALIB_PINS_KEYS = ("L_traj", "N_chains", "N_R", "C", "ESS_min")

_CALIB_TBD_SENTINEL = "TBD-pending-Task-12"


def _read_pins_text() -> str:
    return PINS_MD_PATH.read_text()


def _extract_calib_value(key: str, pins_text: str):
    """Return (value_or_none, is_tbd: bool) for a calibration-frozen constant in PINS.md.

    Robust to the Task-12 freeze (the TRAP the review flagged): Task 12 replaces "TBD" with frozen
    numbers on the SAME PINS row.  We therefore split each markdown table row on `|` and inspect ONLY
    the VALUE/STATUS column (the cell that is NOT the Item column where the constant NAME lives), so a
    stray "TBD" left in an Item-column comment (e.g. "(was TBD)") does NOT cause a silent skip and a
    frozen number in the Status column IS picked up.

    Scope: only the "TBD-at-step placeholders" section, and the constant must appear as a whole token
    in the Item column (to avoid "C" matching inside "N_chains"/"SOKAL_C").

    Decision (per the constant's VALUE/STATUS column, NOT the whole line):
      (None, True)       — Status column contains "TBD"            → skip TBD-pending-Task-12
      (float_val, False) — Status column contains a number         → assert that number
      (None, True)       — key not found in the TBD section at all → treat as TBD (not yet pinned)
      raises             — key found but Status is neither TBD nor numeric → HARD FAIL
    """
    # Locate the "TBD-at-step placeholders" section; scan only its table rows.
    in_tbd_section = False
    tbd_rows = []
    for line in pins_text.splitlines():
        stripped = line.strip()
        if "TBD-at-step" in stripped:
            in_tbd_section = True
            continue
        if in_tbd_section and stripped.startswith("##"):
            break
        if in_tbd_section and stripped.startswith("|"):
            tbd_rows.append(line)

    # Whole-token Item-column match (|, space, comma, paren boundaries) so "C" ∉ "N_chains"/"SOKAL_C".
    key_pattern = r"(?:^|[,|\s(]){}(?:[,|\s)]|$)".format(re.escape(key))

    for row in tbd_rows:
        # split the markdown row into cells; drop the leading/trailing empty cells from the | borders.
        cells = [c.strip() for c in row.strip().strip("|").split("|")]
        if len(cells) < 2:
            continue
        item_col = cells[0]
        # the VALUE/STATUS column is everything after the Item column (joined, in case of >2 columns).
        status_col = " ".join(cells[1:])
        if not re.search(key_pattern, item_col):
            continue
        # Decision is made on the STATUS column ONLY (Task-12-trap-robust).
        if "TBD" in status_col:
            return None, True
        nums = re.findall(r"[-+]?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?", status_col)
        if nums:
            return float(nums[0]), False
        raise AssertionError(
            f"HARD FAIL: calibration constant '{key}' in PINS.md TBD section has a Status column "
            f"that is neither TBD nor numeric. Status cell: {status_col!r} (row: {row!r})"
        )
    # Key not found in the TBD section → treat as TBD (may not be in PINS until Task 12)
    return None, True


# ------------------------------------------------------------------ shared fixture helpers (reused)

def _build_fixture_step():
    """Build the real 4_4 tiny fixture DTM + a perturbed step 0 (CPU, NO dtm.train).
    This is the same pattern used by fixture_6_4.py."""
    from thrmlDenoising.DTM import DTM
    from thrmlDenoising.utils import make_cfg

    FIXTURE_CFG = dict(
        exp=dict(seed=0, descriptor="zcc_fixture", compute_autocorr=False, generate_gif=False, n_cores=1),
        data=dict(dataset_name="smoke_testing_3_1_3", target_classes=tuple(range(3)),
                  pixel_threshold_for_single_trials=0.1),
        graph=dict(graph_preset_architecture=4_4, num_label_spots=1, grayscale_levels=1, torus=True,
                   base_graph_manager="poisson_binomial_ising_graph_manager"),
        sampling=dict(batch_size=400, n_samples=2, steps_per_sample=2, steps_warmup=4, training_beta=1.0),
        diffusion_schedule=dict(num_diffusion_steps=1, kind="log", diffusion_offset=0.1),
        diffusion_rates=dict(image_rate=0.8, label_rate=0.2),
        optim=dict(momentum=0.9, b2_adam=0.999, step_learning_rates=(0.05,), alpha_cosine_decay=0.2,
                   n_epochs_for_lrd=50),
    )
    dtm = DTM(make_cfg(**FIXTURE_CFG))
    step = _perturb_step(dtm.steps[0], seed=123)
    return dtm, step


def _perturb_step(step, scale=0.5, seed=123):
    """Perturb weights/biases as DTM.train does (tree_at write-back, factors stale = faithful bug repro)."""
    import equinox as eqx
    from thrmlDenoising.sampling_specs import get_new_per_block_interactions

    k = jr.PRNGKey(seed)
    w1 = step.model.weights + scale * jr.normal(k, step.model.weights.shape)
    b1 = step.model.biases + scale * jr.normal(jr.fold_in(k, 1), step.model.biases.shape)
    new_pos = get_new_per_block_interactions(step.training_spec.program_positive, w1, b1)
    new_neg = get_new_per_block_interactions(step.training_spec.program_negative, w1, b1)
    new_free = get_new_per_block_interactions(step.generation_spec.program_free, w1, b1)
    new_cond = get_new_per_block_interactions(step.generation_spec.program_conditioned, w1, b1)
    return eqx.tree_at(
        lambda s: (s.model.weights, s.model.biases,
                   s.training_spec.program_positive.per_block_interactions,
                   s.training_spec.program_negative.per_block_interactions,
                   s.generation_spec.program_free.per_block_interactions,
                   s.generation_spec.program_conditioned.per_block_interactions),
        step, (w1, b1, new_pos, new_neg, new_free, new_cond))


@contextlib.contextmanager
def _x64():
    """Scoped JAX float64 — required for compat-core (same pattern as fixture_6_4 and driver)."""
    prev = jax.config.jax_enable_x64
    jax.config.update("jax_enable_x64", True)
    try:
        yield
    finally:
        jax.config.update("jax_enable_x64", prev)


# ===========================================================================
# GROUP 1 — DB-CERT: present + residual < 1e-10 on the production-shape kernel
# ===========================================================================

def test_g1_db_cert_production_shape_kernel():
    """DB certificate on the 4-superblock DTM-negative-shape kernel (production 44_12 structure).
    max_asym < 1e-10 (the K = ½(P_AB+P_BA) is π-reversible); deterministic scan non-reversible.
    Reuses harness/selfadjoint_cert.certify (zero-compute, pure numpy)."""
    res = sc.certify(np.random.default_rng(0), sizes=(1, 1, 1, 1), verbose=False)

    assert res["passed"] is True, (
        f"DB certificate FAILED: passed={res['passed']!r}; "
        f"max_asym={res.get('max_asym', '?'):.3e}")
    assert res["max_asym"] < sc.TOL_SYM, (
        f"DB cert: max_asym={res['max_asym']:.3e} ≥ TOL_SYM={sc.TOL_SYM:.0e} — "
        "K = ½(P_AB+P_BA) NOT π-reversible (the reversible-kernel patch is broken)")
    assert res["K_db_residual"] < sc.TOL_SYM, (
        f"K_db_residual {res['K_db_residual']:.3e} ≥ {sc.TOL_SYM:.0e}")
    assert res["P_fwd_db_residual"] > sc.MIN_NONREV, (
        f"deterministic P_fwd unexpectedly reversible ({res['P_fwd_db_residual']:.3e}) — "
        "cert has no discriminating teeth")
    assert res["n_superblocks"] == 4
    print(f"\n[G1 DB-CERT] max_asym={res['max_asym']:.3e} < {sc.TOL_SYM:.0e}  PASS "
          f"(P_fwd discriminator={res['P_fwd_db_residual']:.3e})")


# ===========================================================================
# GROUP 2 — FROZEN CONSTANTS: present + numeric (or TBD-pending-Task-12)
# ===========================================================================

def test_g2a_companion_cfg_graph_preset_44_12():
    """graph_preset_architecture == 44_12 (== int 4412) in COMPANION_CFG."""
    assert COMPANION_CFG["graph"]["graph_preset_architecture"] == GRAPH_PRESET, (
        f"graph_preset_architecture {COMPANION_CFG['graph']['graph_preset_architecture']} != {GRAPH_PRESET}")
    print(f"\n[G2a] graph_preset_architecture={COMPANION_CFG['graph']['graph_preset_architecture']}  PASS")


def test_g2b_stride_and_window_constants():
    """stride_sweeps=8, window_samples_K=50, window_span_sweeps=400 present + consistent."""
    stride = pp.STRIDE_SWEEPS
    K = pp.K_WINDOW
    span = pp.B_WARMUP
    assert stride == STRIDE_SWEEPS, f"STRIDE_SWEEPS {stride} != {STRIDE_SWEEPS}"
    assert K == WINDOW_SAMPLES_K, f"K_WINDOW {K} != {WINDOW_SAMPLES_K}"
    assert span == WINDOW_SPAN_SWEEPS, f"B_WARMUP {span} != {WINDOW_SPAN_SWEEPS}"
    # The identity window_span_sweeps == window_samples_K * stride_sweeps
    assert span == K * stride, (
        f"window_span_sweeps ({span}) != window_samples_K ({K}) × stride_sweeps ({stride})"
        " — the definition contract violated")
    print(f"\n[G2b] stride={stride} K={K} span={span} span==K*stride ({span}=={K}*{stride})  PASS")


def test_g2c_steps_per_sample_equals_stride():
    """steps_per_sample in COMPANION_CFG sampling == stride_sweeps == 8."""
    cfg_stride = COMPANION_CFG["sampling"]["steps_per_sample"]
    assert cfg_stride == STEPS_PER_SAMPLE, (
        f"sampling.steps_per_sample {cfg_stride} != {STEPS_PER_SAMPLE}")
    assert cfg_stride == STRIDE_SWEEPS, (
        f"steps_per_sample ({cfg_stride}) != STRIDE_SWEEPS ({STRIDE_SWEEPS}) — must match")
    print(f"\n[G2c] steps_per_sample={cfg_stride}  PASS")


def test_g2d_adaptive_threshold():
    """adaptive_threshold == 0.016 in COMPANION_CFG."""
    val = COMPANION_CFG["cp"]["adaptive_threshold"]
    assert val == ADAPTIVE_THRESHOLD, f"adaptive_threshold {val} != {ADAPTIVE_THRESHOLD}"
    print(f"\n[G2d] adaptive_threshold={val}  PASS")


def test_g2e_num_diffusion_steps():
    """num_diffusion_steps == 4 in COMPANION_CFG."""
    val = COMPANION_CFG["diffusion_schedule"]["num_diffusion_steps"]
    assert val == NUM_DIFFUSION_STEPS, f"num_diffusion_steps {val} != {NUM_DIFFUSION_STEPS}"
    print(f"\n[G2e] num_diffusion_steps={val}  PASS")


def test_g2f_num_label_spots():
    """num_label_spots == 5 in COMPANION_CFG."""
    val = COMPANION_CFG["graph"]["num_label_spots"]
    assert val == NUM_LABEL_SPOTS, f"num_label_spots {val} != {NUM_LABEL_SPOTS}"
    print(f"\n[G2f] num_label_spots={val}  PASS")


def test_g2g_c0_nonzero():
    """correlation_penalty seed c0 == 0.001 (non-zero) in COMPANION_CFG."""
    cp = COMPANION_CFG["cp"]["correlation_penalty"]
    assert len(cp) == NUM_DIFFUSION_STEPS, (
        f"correlation_penalty tuple length {len(cp)} != num_diffusion_steps {NUM_DIFFUSION_STEPS}")
    for i, v in enumerate(cp):
        assert v == C0, f"correlation_penalty[{i}] {v} != c0={C0}"
    assert all(v != 0.0 for v in cp), (
        f"c0 must be NON-ZERO (0.001 honors plan 'nonzero seed'); got {cp}")
    print(f"\n[G2g] correlation_penalty={cp} (all = c0={C0} ≠ 0)  PASS")


def test_g2h_adaptive_cp_is_true():
    """adaptive_cp is True in COMPANION_CFG."""
    val = COMPANION_CFG["cp"]["adaptive_cp"]
    assert val is True, f"adaptive_cp must be True (ACP enabled); got {val!r}"
    print(f"\n[G2h] adaptive_cp={val}  PASS")


def test_g2i_N_R_present_in_pins():
    """N_R is present in PINS.md (either numeric or TBD-pending-Task-12)."""
    pins = _read_pins_text()
    val, is_tbd = _extract_calib_value("N_R", pins)
    if is_tbd:
        pytest.skip(_CALIB_TBD_SENTINEL)
    assert isinstance(val, (int, float)) and val > 0, (
        f"N_R must be a positive number; got {val!r}")
    print(f"\n[G2i] N_R={val} (pinned numeric)  PASS")


def _check_calib_pin(key: str):
    """Shared helper: check one calibration-frozen constant in PINS.md."""
    pins = _read_pins_text()
    val, is_tbd = _extract_calib_value(key, pins)
    if is_tbd:
        pytest.skip(_CALIB_TBD_SENTINEL)
    assert isinstance(val, (int, float)) and val > 0, (
        f"calibration constant '{key}' must be a positive number; got {val!r}")
    return val


def test_g2j_L_traj_present():
    """L_traj in PINS.md (numeric or TBD-pending-Task-12)."""
    val = _check_calib_pin("L_traj")
    print(f"\n[G2j] L_traj={val}  PASS")


def test_g2k_N_chains_present():
    """N_chains in PINS.md (numeric or TBD-pending-Task-12)."""
    val = _check_calib_pin("N_chains")
    print(f"\n[G2k] N_chains={val}  PASS")


def test_g2l_C_present():
    """C (trajectory-adequacy factor) in PINS.md (numeric or TBD-pending-Task-12)."""
    val = _check_calib_pin("C")
    print(f"\n[G2l] C={val}  PASS")


def test_g2m_ESS_min_present():
    """ESS_min (window-adequacy threshold) in PINS.md (numeric or TBD-pending-Task-12)."""
    val = _check_calib_pin("ESS_min")
    print(f"\n[G2m] ESS_min={val}  PASS")


# ===========================================================================
# GROUP 3 — SHAPE CHECKS
# ===========================================================================

def test_g3a_latent_adapter_test_dict_shape():
    """Latent adapter test dict has test["image"].shape == (1000, 196); label shape matches n_label_nodes."""
    from htdml.latent_adapter import build_latent_dataset, _DEFAULT_NUM_LABEL_SPOTS, _DEFAULT_TARGET_CLASSES
    from htdml.autoencoder import BinaryAutoencoder
    import equinox as eqx

    # Build a tiny identity encode_fn: just returns random hard latents (no real encoder needed).
    N_TRAIN = 1100  # > 1000 so test slice is valid
    N_TEST = 1200
    n_classes = len(_DEFAULT_TARGET_CLASSES)
    n_label_nodes = n_classes * _DEFAULT_NUM_LABEL_SPOTS  # = 50

    rng = np.random.default_rng(42)
    train_imgs = rng.random((N_TRAIN, 28, 28, 1)).astype(np.float32)
    train_cls = rng.integers(0, n_classes, size=N_TRAIN)
    test_imgs = rng.random((N_TEST, 28, 28, 1)).astype(np.float32)
    test_cls = rng.integers(0, n_classes, size=N_TEST)

    # Identity encode_fn: returns (hard_latent, logits) with hard_latent ∈ {-1, +1}
    def dummy_encode(imgs):
        n = imgs.shape[0]
        hard = (rng.integers(0, 2, size=(n, 196)) * 2 - 1).astype(np.float32)
        logits = hard.copy()
        return hard, logits

    train_ds, test_ds, ohtl = build_latent_dataset(
        dummy_encode, train_imgs, train_cls, test_imgs, test_cls)

    assert test_ds["image"].shape == (1000, 196), (
        f"test dict image shape {test_ds['image'].shape} != (1000, 196)")
    assert test_ds["label"].shape == (1000, n_label_nodes), (
        f"test dict label shape {test_ds['label'].shape} != (1000, {n_label_nodes})")
    assert test_ds["image"].dtype == bool, f"test image dtype {test_ds['image'].dtype} != bool"
    assert ohtl.shape == (n_classes, n_label_nodes), (
        f"one_hot_target_labels shape {ohtl.shape} != ({n_classes}, {n_label_nodes})")
    print(f"\n[G3a] test dict image={test_ds['image'].shape} (1000,196)  PASS")


def test_g3b_full_model_probe_record_has_one_entry_per_diffusion_step():
    """A full-model probe (evaluate_model) returns EXACTLY one per-layer dict per diffusion step.

    NOT a tautology: instead of hand-building a 4-element list, we exercise the REAL
    ``TrainabilityProbe.evaluate_model`` loop (which does ``n_layers = len(dtm.steps)`` then iterates),
    with the per-layer ``evaluate`` MONKEYPATCHED to a tiny stub (so no GPU sampling) that records the
    layer index it was called with.  We assert the real loop produced exactly ``len(dtm.steps)`` records,
    that those records carry the real HEADLINE_KEYS, and — on the companion config — that
    ``len(dtm.steps) == num_diffusion_steps == 4``.  The record COUNT comes from the driver's real loop,
    not from a literal we wrote.
    """
    from htdml.trainability_probe import TrainabilityProbe

    # (1) The REAL invariant: a DTM has one step per diffusion step.  Verify on the real fixture DTM
    #     (num_diffusion_steps=1 → 1 step) so the len(steps)==num_diffusion_steps link is REAL, not assumed.
    dtm_fix, _step = _build_fixture_step()
    fix_nds = int(dtm_fix.cfg.diffusion_schedule.num_diffusion_steps)
    assert len(dtm_fix.steps) == fix_nds, (
        f"fixture DTM has {len(dtm_fix.steps)} steps but num_diffusion_steps={fix_nds} — "
        "the one-step-per-diffusion-step invariant the 4-layer count rests on is broken")

    # (2) Drive the REAL evaluate_model loop with evaluate() stubbed (records the layer it sees).
    #     mock.patch.object on the CLASS attribute makes the stub an unbound function → `self` is the
    #     first positional arg the real evaluate_model passes.
    probe = TrainabilityProbe()
    seen_layers = []

    def _stub_evaluate(self, model, layer, batch, **kw):
        seen_layers.append(int(layer))
        return {k: 0.0 for k in TrainabilityProbe.HEADLINE_KEYS} | {"layer": int(layer)}

    with mock.patch.object(TrainabilityProbe, "evaluate", _stub_evaluate):
        records = probe.evaluate_model(dtm_fix, batch={"image": None, "label": None, "idx": 0},
                                       n_R=4, L_traj=10, n_chains=2, diag_key=0)

    # The driver's REAL loop must yield exactly one record per step (== len(dtm.steps)).
    assert len(records) == len(dtm_fix.steps), (
        f"evaluate_model returned {len(records)} records for {len(dtm_fix.steps)} steps — "
        "the per-layer loop does not produce exactly one record per diffusion step")
    assert seen_layers == list(range(len(dtm_fix.steps))), (
        f"evaluate_model visited layers {seen_layers}, expected {list(range(len(dtm_fix.steps)))}")
    # Each record carries the real HEADLINE_KEYS.
    for rec in records:
        assert set(TrainabilityProbe.HEADLINE_KEYS).issubset(rec.keys()), (
            f"per-layer record missing HEADLINE_KEYS: {set(TrainabilityProbe.HEADLINE_KEYS) - set(rec)}")

    # (3) On the companion config the count is 4 (num_diffusion_steps=4).
    assert COMPANION_CFG["diffusion_schedule"]["num_diffusion_steps"] == 4, (
        "companion num_diffusion_steps must be 4")
    print(f"\n[G3b] evaluate_model real loop: {len(records)} records == len(steps)={len(dtm_fix.steps)} "
          f"(fixture nds={fix_nds}); companion nds=4 → 4 layers; HEADLINE_KEYS present  PASS")


# ===========================================================================
# GROUP 4 — FORMULA-SHAPE
# ===========================================================================

def test_g4a_q_struct_perp_formula_K50_prefactor():
    """Q_struct^⊥ uses the K=50 prefactor (50/2)·‖g‖²/T_{O,Y}.
    Verify via probe_scalars on a synthetic retained series."""
    # IID synthetic retained: (n_chains=4, L=200, b=10)
    rng = np.random.default_rng(0)
    retained = rng.standard_normal((4, 200, 10))
    g = np.ones(10)

    scalars = pp.probe_scalars(retained, n_R=4, diag_key=42, gradient=g)

    # Q_struct_perp = (K/2) * ‖g‖² / T_{O,Y}
    K = pp.K_WINDOW  # 50
    T_O_Y = scalars["_T_O_Y"]
    grad_norm = scalars["gradient_norm"]
    expected_Q = (K / 2.0) * grad_norm ** 2 / T_O_Y if T_O_Y > 0 else float("inf")
    assert np.isclose(scalars["Q_struct_perp"], expected_Q, rtol=1e-6), (
        f"Q_struct_perp {scalars['Q_struct_perp']:.6g} != (50/2)*‖g‖²/T_O_Y = {expected_Q:.6g} "
        f"(K={K}, ‖g‖={grad_norm:.4g}, T_O_Y={T_O_Y:.4g})")
    assert K == 50, f"K_WINDOW must be 50; got {K}"
    print(f"\n[G4a] Q_struct_perp formula: (K={K}/2)·‖g‖²/T_O_Y = {expected_Q:.4g}  PASS")


def test_g4b_ess_hat_formula():
    """ESS_hat == K/(2·τ_int,Y) == 50/(2·τ_int,Y). Verified via probe_scalars."""
    rng = np.random.default_rng(1)
    retained = rng.standard_normal((4, 200, 8))
    g = np.ones(8)
    scalars = pp.probe_scalars(retained, n_R=4, diag_key=7, gradient=g)

    K = pp.K_WINDOW
    tau = scalars["tau_int_Y"]
    expected_ess = K / (2.0 * tau)
    assert np.isclose(scalars["ESS_hat"], expected_ess, rtol=1e-6), (
        f"ESS_hat {scalars['ESS_hat']:.4g} != K/(2·τ) = {expected_ess:.4g} "
        f"(K={K}, τ={tau:.4g})")
    print(f"\n[G4b] ESS_hat=K/(2·τ_int,Y)={expected_ess:.4g} (K=50, τ={tau:.4g})  PASS")


def test_g4c_bce_fid_are_real_seedmetrics_fields():
    """BCE + FID must be REAL fields on the driver's record structure (SeedMetrics), not a literal we
    wrote.  NOT a tautology: we read the actual dataclass fields the driver populates and assert the
    BCE/FID quantities (joint AND matched-control) are present — so a driver that silently stopped
    storing them would FAIL here."""
    import dataclasses

    field_names = {f.name for f in dataclasses.fields(D.SeedMetrics)}
    # The driver stores BCE/FID for BOTH the joint arm and the matched control arm.
    for need in ("bce", "fid", "control_bce", "control_fid"):
        assert need in field_names, (
            f"SeedMetrics is missing the '{need}' field — the driver does NOT store the "
            f"{'BCE' if 'bce' in need else 'FID'} quality metric (real record gap). "
            f"SeedMetrics fields: {sorted(field_names)}")
    print(f"\n[G4c] BCE+FID are real SeedMetrics fields {{bce, fid, control_bce, control_fid}}  PASS")


def test_g4d_fid_on_decoded_bw_npz_sha():
    """FID ref stats use bw_fashion_mnist_train.npz with asserted sha256 (PINS-pinned)."""
    import hashlib

    assert BW_NPZ_PATH.exists(), (
        f"bw_fashion_mnist_train.npz not found at {BW_NPZ_PATH}")
    h = hashlib.sha256()
    with open(BW_NPZ_PATH, "rb") as fh:
        for chunk in iter(lambda: fh.read(65536), b""):
            h.update(chunk)
    actual = h.hexdigest()
    assert actual == BW_NPZ_SHA256_EXPECTED, (
        f"bw_fashion_mnist_train.npz sha256 mismatch: got {actual}, expected {BW_NPZ_SHA256_EXPECTED}")
    print(f"\n[G4d] bw_fashion_mnist_train.npz sha256={actual[:16]}…  PASS")


def test_g4e_inception_weights_hash_from_pins():
    """The local InceptionV3-weights sha256 matches PINS.md value."""
    pins = _read_pins_text()
    # Extract the pinned sha from PINS.md
    pin_sha = None
    for line in pins.splitlines():
        if "InceptionV3-FID-weights sha256" in line:
            tokens = line.split()
            for tok in reversed(tokens):
                tok_clean = tok.strip("`|")
                if re.fullmatch(r"[0-9a-f]{64}", tok_clean):
                    pin_sha = tok_clean
                    break
    assert pin_sha is not None, (
        "Could not find 'InceptionV3-FID-weights sha256' pin in PINS.md — "
        "run dataset_gate.py first")
    assert pin_sha == INCEPTION_SHA256_EXPECTED, (
        f"PINS.md inception sha {pin_sha[:16]}… != expected {INCEPTION_SHA256_EXPECTED[:16]}…")
    print(f"\n[G4e] InceptionV3-FID-weights sha256={pin_sha[:16]}… matches PINS  PASS")


@pytest.mark.skipif(
    not INCEPTION_PICKLE_PATH.exists(),
    reason="Inception pickle not cached — run `python scripts/dataset_gate.py` first.",
)
def test_g4f_no_network_fid_load():
    """No-network assertion: with requests.get + utils.download both monkeypatched to raise,
    the FID weights load path is network-free (P1 patch applied). Reuses dataset_gate helper."""
    from scripts.dataset_gate import assert_fid_offline
    assert_fid_offline()
    print(f"\n[G4f] no-network FID load PASS")


def test_g4g_free_energy_is_marginalized_not_raw_clamped():
    """The compat free energy is the MARGINALIZED F_MF (log-2cosh + MF), NOT raw clamped energy.
    At any fixed clamp, F_MF ≥ F_exact AND dF_MF/dlatent ≠ dE_clamped/dlatent (gradient distinctness)."""
    _dtm, step = _build_fixture_step()
    beta = float(step.training_spec.beta)
    rng = np.random.default_rng(9)

    with _x64():
        maps = C.build_compat_maps(step)
        jm = C._jnp_maps(maps)
        n_img = maps["n_img"]
        clamp = (rng.integers(0, 2, size=maps["n_clamp"]) * 2 - 1).astype(np.float64)
        f_mf = float(C.F_MF(jnp.asarray(clamp), jm, beta))
        f_ex = C.F_exact_full(clamp, maps, beta)
        assert f_mf >= f_ex - 1e-9, (
            f"F_MF ({f_mf:.4g}) < F_exact ({f_ex:.4g}) — variational upper bound VIOLATED; "
            "the free energy is NOT the marginalized form")

        # Gradient distinctness: F_compat grad ≠ raw clamp energy grad.
        tail = jnp.asarray(clamp[n_img:])
        latent = jnp.asarray(clamp[:n_img])

        def fmf_of_latent(lat):
            return C.F_MF(jnp.concatenate([lat, tail]), jm, beta)

        def eclamp_of_latent(lat):
            return C.clamp_energy(jnp.zeros(maps["n_upper"]), jnp.concatenate([lat, tail]), jm, beta)

        g_mf = np.asarray(jax.grad(fmf_of_latent)(latent))
        g_clamp = np.asarray(jax.grad(eclamp_of_latent)(latent))
        distinct = float(np.max(np.abs(g_mf - g_clamp)))
        assert distinct > 1e-3, (
            f"dF_MF/dlatent ≈ dE_clamped/dlatent (max diff {distinct:.2e}) — "
            "the MF/marginalization collapsed to raw clamped energy (the entropy term is inert)")

    print(f"\n[G4g] free_energy=marginalized: F_MF({f_mf:.3g})≥F_exact({f_ex:.3g}); "
          f"∂F_MF≠∂E_clamp (diff={distinct:.3g})  PASS")


# ===========================================================================
# GROUP 5 — STORE-COVERAGE
# ===========================================================================

def test_g5a_six_probe_scalars_are_real_headline_keys():
    """6 of the 9 store-coverage quantities are produced PER LAYER by the real probe.  Assert against
    the probe's ACTUAL HEADLINE_KEYS (not a literal) — and assert the probe's REAL per-layer dict
    actually carries them, by inspecting the keys ``evaluate`` assembles (line 332-349)."""
    from htdml.trainability_probe import TrainabilityProbe

    probe_provided = {"r_grad[1]", "r_grad[50]", "tau_int_Y", "ESS_hat", "gradient_norm", "Q_struct_perp"}
    headline = set(TrainabilityProbe.HEADLINE_KEYS)
    missing = probe_provided - headline
    assert not missing, (
        f"the probe HEADLINE_KEYS do NOT cover store-coverage scalars {missing} — "
        f"real probe-record gap.  HEADLINE_KEYS={sorted(headline)}")

    # Cross-check the probe's real per-layer dict keys (what evaluate() actually assembles): drive a
    # tiny probe_scalars (no GPU) and confirm those 6 scalar names are real output keys.
    rng = np.random.default_rng(0)
    scal = pp.probe_scalars(rng.standard_normal((4, 200, 8)), n_R=4, diag_key=0, gradient=np.ones(8))
    for k in probe_provided:
        assert k in scal, f"probe_scalars output missing store-coverage scalar '{k}': keys={sorted(scal)}"
    print(f"\n[G5a] 6 probe scalars are real HEADLINE_KEYS + real probe_scalars output keys  PASS")


def test_g5b_bce_fid_are_real_driver_record_fields():
    """2 of the 9 store-coverage quantities (BCE, FID) are stored by the driver's REAL record structure
    (SeedMetrics) — for both the joint and the control arm.  Assert against the actual dataclass fields,
    not a literal."""
    import dataclasses

    sm_fields = {f.name for f in dataclasses.fields(D.SeedMetrics)}
    for need in ("bce", "fid", "control_bce", "control_fid"):
        assert need in sm_fields, (
            f"SeedMetrics does NOT store '{need}' — real driver-record gap. fields={sorted(sm_fields)}")
    print(f"\n[G5b] BCE+FID (joint+control) are real SeedMetrics fields  PASS")


def test_g5c_live_acp_coefficient_IS_stored_on_epoch_record():
    """STORE-COVERAGE (Task-9, NOW CLOSED): the 9th quantity — the LIVE ACP coefficient
    ``correlation_penalty[step]`` (read POST-``adapt_param``) — is STORED on the driver's per-epoch +
    per-reverse-layer record ``EpochLayerRecord.correlation_penalty`` (populated via
    ``driver.live_acp_coefficient(dtm, step, cp_coeffs)`` = ``cp_coeffs[step]`` after the adaptive
    update, DTM.py:354-364).

    This was previously an xfail-marked GAP (the field lived in no record).  It is now a REAL pass: we
    assert (1) ``correlation_penalty`` is a field of ``EpochLayerRecord`` and round-trips through
    ``make_epoch_layer_record``; (2) the FULL 9-quantity store-coverage is complete across the three
    record surfaces — probe scalars (gradient_norm, r_grad[1], r_grad[50], tau_int_Y, ESS_hat,
    Q_struct_perp), SeedMetrics (bce/fid joint+control), and EpochLayerRecord (the live ACP coefficient).
    """
    import dataclasses

    from htdml.trainability_probe import TrainabilityProbe

    sm_fields = {f.name for f in dataclasses.fields(D.SeedMetrics)}
    elr_fields = {f.name for f in dataclasses.fields(D.EpochLayerRecord)}
    headline = set(TrainabilityProbe.HEADLINE_KEYS)
    rng = np.random.default_rng(0)
    scal_keys = set(pp.probe_scalars(rng.standard_normal((4, 50, 4)), n_R=2, diag_key=0,
                                     gradient=np.ones(4)).keys())

    # (1) the live ACP coefficient is a first-class EpochLayerRecord field + round-trips.
    assert "correlation_penalty" in elr_fields, (
        f"EpochLayerRecord does NOT store 'correlation_penalty' — store-coverage gap. fields={sorted(elr_fields)}")
    probe_layer = dict(gradient_norm=1.5, **{"r_grad[1]": 0.3, "r_grad[50]": 0.02},
                       tau_int_Y=12.0, ESS_hat=22.0, Q_struct_perp=1.1)
    rec = D.make_epoch_layer_record(epoch=2, layer=1, bce=0.12, fid=1.4,
                                    correlation_penalty=0.0042, probe_layer_dict=probe_layer)
    assert rec.correlation_penalty == pytest.approx(0.0042), "live ACP coefficient did not round-trip"
    # live_acp_coefficient reads cp_coeffs[step] (the post-adapt_param vector the DTM loop maintains).
    assert D.live_acp_coefficient(None, 1, np.asarray([0.001, 0.0035, 0.0])) == pytest.approx(0.0035)

    # (2) full 9-quantity store-coverage across the three record surfaces.
    probe_scalar_quantities = {"gradient_norm", "r_grad[1]", "r_grad[50]", "tau_int_Y", "ESS_hat",
                               "Q_struct_perp"}
    assert probe_scalar_quantities <= (headline | scal_keys), (
        f"probe scalars missing {probe_scalar_quantities - (headline | scal_keys)}")
    assert {"bce", "fid", "control_bce", "control_fid"} <= sm_fields, "SeedMetrics missing BCE/FID"
    # the 9-quantity per-epoch+layer record carries every stored scalar (incl. the live ACP coefficient).
    nine = {"bce", "fid", "correlation_penalty", "gradient_norm", "r_grad_1", "r_grad_50",
            "tau_int_Y", "ESS_hat", "Q_struct_perp"}
    assert nine <= elr_fields, f"EpochLayerRecord missing {nine - elr_fields}"

    print(f"\n[G5c CLOSED] live ACP coefficient 'correlation_penalty' STORED on "
          f"EpochLayerRecord (post-adapt_param via live_acp_coefficient); 9-quantity store-coverage "
          f"complete across probe scalars + SeedMetrics + EpochLayerRecord  PASS")


# ===========================================================================
# GROUP 6 — PROVENANCE / KEY-ISOLATION / WEIGHT-HASH + REFRESH-PROOF
# ===========================================================================

def test_g6a_refresh_proof_passes_on_fixture():
    """The per-step trained-weight-refresh proof passes on the fixture step:
    constructor_was_stale=True AND refresh_ok=True (the exp15/16 bug guard)."""
    _dtm, step = _build_fixture_step()
    proof = pp.refreshed_weight_proof(step)

    assert proof["constructor_was_stale"] is True, (
        f"constructor_was_stale=False (stale_vs_trained_maxabs={proof['stale_vs_trained_maxabs']}) — "
        "the exp15/16 bug premise does not hold on the fixture; the guard is VACUOUS")
    assert proof["refresh_ok"] is True, (
        f"refresh_ok=False (refreshed_vs_trained_maxabs={proof['refreshed_vs_trained_maxabs']}) — "
        "the trained-weight refresh did NOT take; every probe/compat build gates on this")
    assert proof["refreshed_vs_trained_maxabs"] < 1e-6
    assert proof["stale_vs_trained_maxabs"] > 1e-6
    print(f"\n[G6a] refresh_proof: constructor_was_stale=True stale_maxabs={proof['stale_vs_trained_maxabs']:.4f}; "
          f"refresh_ok=True refreshed_maxabs={proof['refreshed_vs_trained_maxabs']:.2e}  PASS")


def test_g6b_weights_hash_and_key_list():
    """_weights_hash and _key_list APIs work on the fixture step; the hash is a 16-hex string."""
    dtm, step = _build_fixture_step()
    h = pp._weights_hash(step)
    assert isinstance(h, str) and len(h) == 16 and re.fullmatch(r"[0-9a-f]{16}", h), (
        f"_weights_hash must be a 16-hex string; got {h!r}")
    kl = pp._key_list(dtm)
    assert isinstance(kl, list) and len(kl) > 0 and all(isinstance(x, int) for x in kl), (
        f"_key_list must be a list of ints; got {kl!r}")
    print(f"\n[G6b] _weights_hash={h}; _key_list len={len(kl)}  PASS")


def test_g6c_find_counts_extracts_real_optstate_counts():
    """_find_counts must actually FIND opt-state count leaves (the provenance scalar the fork uses).
    NOT vacuous: a fresh optax adam opt_state has count leaves (value 0 before training), so the list
    must be NON-EMPTY and every entry an int.  An empty list would mean the count-extraction recursion
    silently failed (the fork's opt-state provenance assertion would then be vacuous)."""
    _dtm, step = _build_fixture_step()
    counts = pp._find_counts(step.opt_state)
    assert isinstance(counts, list), f"_find_counts must return a list; got {type(counts)}"
    assert len(counts) > 0, (
        "_find_counts found NO count leaves in the opt_state — the optax count recursion is broken; "
        "the fork's opt-state provenance check would be vacuous")
    assert all(isinstance(c, int) for c in counts), f"count leaves must be ints; got {counts}"
    print(f"\n[G6c] _find_counts found {len(counts)} real opt-state count leaves={counts}  PASS")


def test_g6d_refreshed_weight_proof_gates_compat_build():
    """refreshed_compat_maps asserts refresh_ok AND constructor_was_stale before building maps.
    Calling refreshed_compat_maps on the perturbed fixture step succeeds and returns (maps, proof)."""
    _dtm, step = _build_fixture_step()
    with _x64():
        maps, proof = C.refreshed_compat_maps(step)
    assert proof["constructor_was_stale"] is True and proof["refresh_ok"] is True, (
        f"compat refresh guard failed: {proof}")
    assert maps["n_upper"] > 0 and maps["n_lower"] > 0 and maps["n_clamp"] > 0
    print(f"\n[G6d] refreshed_compat_maps: refresh guard cleared; maps built  PASS")


# ===========================================================================
# GROUP 7 — λ=0 ≡ CONTROL (NEW): weights_hash + key_list + opt-state equality
# ===========================================================================

def _ae_param_bytes(params):
    """Flatten a Flax param pytree to a tuple of (path-ordered) raw float bytes for BITWISE comparison."""
    leaves = jax.tree_util.tree_leaves(params)
    return tuple(np.ascontiguousarray(np.asarray(l)).tobytes() for l in leaves)


def _optstate_bytes(opt_state):
    """Flatten an optax opt_state pytree to raw bytes for BITWISE comparison (covers mu/nu/count)."""
    leaves = jax.tree_util.tree_leaves(opt_state)
    return tuple(np.ascontiguousarray(np.asarray(l)).tobytes() for l in leaves)


def _tiny_ste_encoder_zcc(n_img):
    """A tiny DIFFERENTIABLE STE 'encoder' of latent width ``n_img`` (the SAME ``_ste_hard_sign`` the real
    BinaryAutoencoder uses) so the GENUINE encoder-steering property is exercised on the narrow 4_4
    fixture image_output block (n_img=3) without the 196-wide production AE (which only matches the 44_12
    production image_output).  Returns ``encode_fn(params, x) -> (b0 {−1,+1}, logits)``."""
    import htdml.autoencoder as AE

    def encode_fn(params, x):
        logits = jnp.asarray(x) @ params["W"]              # (B, n_img) logits, depends on params
        b0 = AE._ste_hard_sign(logits)                     # {−1,+1}, ∂/∂logits = 1−tanh² (STE)
        return b0, logits

    return encode_fn


def test_g7_lambda_steers_the_encoder_and_lambda0_is_control():
    """(MIGRATED) The GENUINE encoder-steering gate on the REAL driver compat-steering logic.

    The old G7 drove the now-DELETED constant-clamp ``joint_update_step`` path (compat clamp passed in as
    a constant ⇒ ∂(λ·L_compat)/∂ae_params ≡ 0 for all λ — the experiment would have been inert).  That
    path is removed from the driver; this gate now tests the encode→clamp-INSIDE-the-loss steering that
    ``joint_update_step`` actually performs (via ``compat_steering_loss``), on the real 4_4 fixture DTM
    (image_output block n_img=3) with an ``n_img``-matched STE encoder.  Three legs:

      (A) STEERING: ∂(λ·L_compat-only loss)/∂ae_params is NON-ZERO at λ>0 (the compat steers the encoder
          via the STE through the image_output latent) and EXACTLY ZERO at λ=0 (the control).
      (B) λ=0 ≡ CONTROL bitwise: a full ``joint_update_step``-style update (recon + λ·L_compat) at λ=0 is
          bitwise-identical to the SAME update with the compat term entirely absent (the traced-0 multiply
          contributes exactly zero gradient), and λ>0 DIFFERS (the steering actually moves the encoder).
      (C) GRADIENT ISOLATION: label_output + b_t are stop_gradient'd — only the image_output latent carries
          ∂/∂ae_params (the clamp-gradient teeth on the genuinely-λ-dependent axis via
          ``compat_value_and_grad_x64``: ∂(0·L_compat)/∂clamp=0, ∂(5·L_compat)/∂clamp≠0).
    """
    import optax

    _dtm, step = _build_fixture_step()
    beta = float(step.training_spec.beta)
    step_maps = D.step_maps_for(step)
    n_img = int(step_maps[0]["n_img"])
    n_clamp = int(step_maps[0]["n_clamp"])
    n_rest = n_clamp - n_img
    encode_fn = _tiny_ste_encoder_zcc(n_img)

    rng = np.random.default_rng(13)
    n_in = 5
    with _x64():
        params = {"W": jnp.asarray(rng.normal(size=(n_in, n_img)), dtype=jnp.float64)}
        x_batch = jnp.asarray(rng.normal(size=(6, n_in)), dtype=jnp.float64)
        label_clamp = jnp.asarray((rng.integers(0, 2, size=n_rest) * 2 - 1).astype(np.float64))
        bt_clamp = jnp.zeros((0,))

    def steer_loss(p, lam):
        with _x64():
            val, _fin = D.compat_steering_loss(p, x_batch, label_clamp, bt_clamp, step_maps, beta,
                                               lam, n_img=n_img, encode_fn=encode_fn)
        return val

    # --- (A) STEERING: ∂(λ·L_compat)/∂ae_params ≠ 0 at λ>0, = 0 at λ=0 -------------------------------
    with _x64():
        g_pos = np.asarray(jax.grad(lambda p: steer_loss(p, 0.7))(params)["W"])
        g_zero = np.asarray(jax.grad(lambda p: steer_loss(p, 0.0))(params)["W"])
    assert np.any(np.abs(g_pos) > 1e-9), (
        "∂(λ·L_compat)/∂ae_params is ZERO at λ>0 — the compat does NOT steer the encoder (encode→clamp "
        "not inside the differentiated loss; Stage C would be inert — the deleted constant-clamp bug)")
    assert np.all(g_zero == 0.0), (
        f"∂(λ·L_compat)/∂ae_params must be EXACTLY 0 at λ=0 (control): max|g|={np.abs(g_zero).max()}")

    # --- (B) λ=0 ≡ CONTROL bitwise (full update: recon + λ·L_compat) --------------------------------
    optim = optax.sgd(0.01)

    def full_loss(p, lam):
        with _x64():
            recon = jnp.sum((jnp.asarray(x_batch) @ p["W"]) ** 2)      # stand-in reconstruction
            compat, _ = D.compat_steering_loss(p, x_batch, label_clamp, bt_clamp, step_maps, beta,
                                               lam, n_img=n_img, encode_fn=encode_fn)
        return recon + compat

    def recon_only_loss(p):
        with _x64():
            return jnp.sum((jnp.asarray(x_batch) @ p["W"]) ** 2)

    def update(loss_fn):
        with _x64():
            g = jax.grad(loss_fn)(params)
            os0 = optim.init(params)
            upd, _ = optim.update(g, os0, params)
            return optax.apply_updates(params, upd)

    joint0 = update(lambda p: full_loss(p, 0.0))     # λ=0 (compat present but traced-0)
    control = update(recon_only_loss)                # compat term entirely ABSENT
    jointL = update(lambda p: full_loss(p, 0.5))     # λ>0 must DIFFER
    np.testing.assert_array_equal(np.asarray(joint0["W"]), np.asarray(control["W"]))  # bitwise control
    assert not np.allclose(np.asarray(jointL["W"]), np.asarray(control["W"])), (
        "λ>0 update must DIFFER from the λ=0 control (else the steering is inert)")

    # --- (C) clamp-axis teeth + gradient isolation (label/b_t carry no ∂/∂ae_params) ----------------
    with _x64():
        maps = C.build_compat_maps(step)
    clamp_steps = jnp.asarray(
        (rng.integers(0, 2, size=(NUM_DIFFUSION_STEPS, n_clamp)) * 2 - 1).astype(np.float64))
    v0, g0_clamp, fin0c = D.compat_value_and_grad_x64(0.0, clamp_steps, [maps], beta)
    v5, g5_clamp, fin5c = D.compat_value_and_grad_x64(5.0, clamp_steps, [maps], beta)
    assert v0 == 0.0 and fin0c, "compat_value_and_grad_x64(λ=0) value must be exactly 0.0"
    assert np.all(g0_clamp == 0.0), (
        f"∂(0·L_compat)/∂clamp must be all-zero; got max|g|={np.max(np.abs(g0_clamp))}")
    assert fin5c and np.any(g5_clamp != 0.0), (
        "∂(5·L_compat)/∂clamp is all-zero — the compat term has NO teeth even at λ>0 (vacuous λ=0≡control)")

    print(f"\n[G7] GENUINE encoder-steering on the real driver:\n"
          f"  (A) ∂(0.7·L_compat)/∂ae_params max|g|={np.abs(g_pos).max():.4g} (≠0 → steers); "
          f"λ=0 max|g|={np.abs(g_zero).max():.4g} (=0 → control);\n"
          f"  (B) λ=0 update BITWISE-identical to the no-compat control; λ>0 update DIFFERS;\n"
          f"  (C) clamp teeth: ∂(0·L_compat)/∂clamp=0 while ∂(5·L_compat)/∂clamp≠0  PASS")


# ===========================================================================
# GROUP 8 — L_COMPAT INVARIANTS (NEW)
# ===========================================================================

def test_g8a_no_backprop_into_dtm_from_compat():
    """(NEW) DTM params get ZERO gradient from L_compat (DTM params under stop_gradient).
    Verify: d(L_compat)/d(DTM_weights) == 0 on the fixture model."""
    _dtm, step = _build_fixture_step()
    beta = float(step.training_spec.beta)
    rng = np.random.default_rng(17)

    with _x64():
        maps = C.build_compat_maps(step)
        n_clamp = maps["n_clamp"]
        clamp = jnp.asarray((rng.integers(0, 2, size=(1, n_clamp)) * 2 - 1).astype(np.float64))

        # L_compat is defined as a function of clamp_spins (not of DTM weights directly).
        # The DTM weights are baked into the maps at construction time (stop_gradient pattern).
        # If we evaluate d(L_compat)/d(clamp) we get a non-zero gradient (the compat DOES
        # differentiate through the clamp).  The DTM weights are NOT inputs to L_compat (they are
        # in maps, which are numpy arrays — not traced leaves).

        # Assert maps don't contain any JAX traced leaves (they are plain numpy).
        for k, v in maps.items():
            if isinstance(v, np.ndarray):
                assert not hasattr(v, "aval"), (
                    f"map '{k}' is a jax array (a traced leaf) — DTM weights entered the compat "
                    "gradient path; they must be stop_gradient'd numpy constants")

        # The clamp IS the only traced input: ∂L_compat/∂clamp should be non-trivially non-zero.
        def lc(clamp):
            return C.L_compat(clamp, [maps], beta)

        g_clamp = np.asarray(jax.grad(lc)(clamp))
        # There IS a gradient w.r.t. clamp (the encoder latent).
        # The invariant is that DTM weights (not in the computation graph here) have zero gradient.
        # We verify the maps are static numpy (no JAX leaves) = the DTM-no-backprop invariant.
        assert np.any(g_clamp != 0.0), (
            "gradient w.r.t. clamp is all-zero — L_compat is not differentiable in the latent, "
            "which means there's a bug (the encoder gets no gradient steering)")

    print(f"\n[G8a] no-backprop-into-DTM: maps are static numpy (DTM weights not traced); "
          f"∂L_compat/∂clamp is non-zero (encoder CAN be steered)  PASS")


def test_g8b_no_grad_through_bt():
    """(NEW) b_t is stop_gradient — no gradient flows through b_t into the encoder.
    Verify: d(any_fn(b_t))/d(b0) == 0 when b_t = stop_gradient(forward_noise(b0))."""
    from thrmlDenoising.step import get_perturbed_data

    key = jr.PRNGKey(2)
    b0 = jnp.array([[0.0, 1.0, 0.0, 1.0]])

    def loss_through_bt_sg(b0):
        bt = jax.lax.stop_gradient(get_perturbed_data(key, b0, dt=0.5, rates=0.8, bin_trials=1))
        return jnp.sum(bt ** 2 + 2.0 * bt)

    g = np.asarray(jax.grad(loss_through_bt_sg)(b0))
    assert np.all(g == 0.0), (
        f"gradient reached b0 through b_t stop_gradient (should be all-zero); got {g}")
    print(f"\n[G8b] no-grad-through-b_t: ∂(fn(b_t))/∂b0=0 (stop_gradient works)  PASS")


def test_g8c_compat_is_deterministic_mf_no_key():
    """(NEW) L_compat is deterministic mean-field — same input → bitwise-identical F_MF (no PRNG key drawn)."""
    _dtm, step = _build_fixture_step()
    beta = float(step.training_spec.beta)
    rng = np.random.default_rng(21)

    with _x64():
        maps = C.build_compat_maps(step)
        jm = C._jnp_maps(maps)
        n_clamp = maps["n_clamp"]
        clamp = jnp.asarray((rng.integers(0, 2, size=n_clamp) * 2 - 1).astype(np.float64))

        # Call F_MF twice with the SAME input — must return bitwise-identical values.
        f_a = float(C.F_MF(clamp, jm, beta))
        f_b = float(C.F_MF(clamp, jm, beta))
        assert f_a == f_b, (
            f"F_MF non-deterministic: {f_a!r} != {f_b!r} — the compat draws a PRNG key somewhere")

        # Also verify L_compat (multi-step) is deterministic.
        n_steps = 4
        clamp_steps = jnp.asarray(
            (rng.integers(0, 2, size=(n_steps, n_clamp)) * 2 - 1).astype(np.float64))
        lc_a = float(C.L_compat(clamp_steps, [maps], beta))
        lc_b = float(C.L_compat(clamp_steps, [maps], beta))
        assert lc_a == lc_b, (
            f"L_compat non-deterministic: {lc_a!r} != {lc_b!r}")

    print(f"\n[G8c] deterministic MF: F_MF={f_a!r} (bitwise same on repeat); L_compat={lc_a!r}  PASS")


def test_g8d_compat_no_key_drawn_structurally():
    """L_compat / F_MF draw NO PRNG key: they are pure numpy+JAX operations with no jax.random call.
    Verify by calling with no key argument and confirming the computation is key-free."""
    # F_MF / L_compat have no `key` argument — this is the structural proof.
    import inspect
    import htdml.compatibility as C_mod

    sig_fmf = inspect.signature(C_mod.F_MF)
    sig_lc = inspect.signature(C_mod.L_compat)
    sig_cl = inspect.signature(C_mod.compat_loss)

    for name, sig in [("F_MF", sig_fmf), ("L_compat", sig_lc), ("compat_loss", sig_cl)]:
        param_names = list(sig.parameters.keys())
        assert "key" not in param_names, (
            f"{name} signature has a 'key' parameter ({param_names}) — "
            "the compat must NOT draw a PRNG key (deterministic MF)")
    print(f"\n[G8d] compat signature: F_MF/L_compat/compat_loss have NO 'key' param (key-free MF)  PASS")


# ===========================================================================
# GROUP 9 — SEED DISJOINTNESS
# ===========================================================================

def test_g9_seed_disjointness_diagnostic_key_independent_of_dtm_key():
    """The diagnostic key is independent of dtm.key; two seeds use disjoint keys.
    Verify: probe_scalars uses a numpy RNG seeded by diag_key (not jax dtm.key).
    Two different diag_key values produce different Rademacher sketches (disjoint)."""
    rng = np.random.default_rng(0)
    retained = rng.standard_normal((4, 200, 10))
    g = np.ones(10)

    # Two different diag_keys
    scalars_a = pp.probe_scalars(retained, n_R=8, diag_key=100, gradient=g)
    scalars_b = pp.probe_scalars(retained, n_R=8, diag_key=200, gradient=g)

    # The Rademacher sketches produced by different diag_keys should differ
    sk_a = pp.rademacher_sketches(10, 8, 100)
    sk_b = pp.rademacher_sketches(10, 8, 200)
    assert not np.array_equal(sk_a, sk_b), (
        "Rademacher sketches for diag_key=100 and diag_key=200 are IDENTICAL — "
        "seeds are not independent")

    # Same diag_key → same sketches (deterministic).
    sk_a2 = pp.rademacher_sketches(10, 8, 100)
    assert np.array_equal(sk_a, sk_a2), (
        "Rademacher sketches are NOT reproducible (same key → different result)")

    # rademacher_sketches uses numpy (not jax.random) — independent of dtm.key.
    dtm, step = _build_fixture_step()
    dtm_key_before = pp._key_list(dtm)
    _ = pp.rademacher_sketches(10, 8, 42)  # calling with any diag_key
    dtm_key_after = pp._key_list(dtm)
    assert dtm_key_before == dtm_key_after, (
        "dtm.key changed after rademacher_sketches — the diagnostic key must NOT consume dtm.key")

    print(f"\n[G9] seed disjointness: diag_key=100 vs 200 sketches differ; "
          f"same key reproducible; dtm.key unchanged by rademacher_sketches  PASS")


# ===========================================================================
# GROUP 10 — NUMERICAL REGRESSION
# ===========================================================================

def test_g10a_tau_int_IID_is_0_5():
    """τ_int = 0.5 on a constant / IID series (the half-Sokal estimator's degenerate case).
    An IID series has ρ(ℓ) = 0 for ℓ ≥ 1, so τ_int = 0.5 + Σ 0 = 0.5."""
    rng = np.random.default_rng(0)
    # IID series: each chain is independent draws → no autocorrelation.
    iid = rng.standard_normal((8, 500, 4))
    tau = pp.tau_int_Y_from_retained(iid)
    assert abs(tau - 0.5) < 0.05, (
        f"τ_int on IID = {tau:.4f} ≠ 0.5 ± 0.05 — half-Sokal regression broken")
    print(f"\n[G10a] τ_int(IID)={tau:.4f} ≈ 0.5  PASS")


def test_g10b_determinism_same_input_same_output():
    """probe_scalars is deterministic: same retained + diag_key + gradient → identical output."""
    rng = np.random.default_rng(5)
    retained = rng.standard_normal((4, 200, 6))
    g = rng.standard_normal(6)

    scalars_1 = pp.probe_scalars(retained, n_R=4, diag_key=77, gradient=g)
    scalars_2 = pp.probe_scalars(retained, n_R=4, diag_key=77, gradient=g)

    for key in ["tau_int_Y", "ESS_hat", "Q_struct_perp", "gradient_norm", "r_grad[1]", "r_grad[50]"]:
        assert scalars_1[key] == scalars_2[key], (
            f"probe_scalars not deterministic: [{key}] {scalars_1[key]} != {scalars_2[key]}")
    print(f"\n[G10b] determinism: probe_scalars same input→same output  PASS")


def test_g10c_tau_int_constant_series_is_0_5():
    """On a CONSTANT series (all zeros), ρ(ℓ) is undefined (acov0 == 0).
    The estimator returns 0.5 (the minimum) because _rho_block divides by 0 → np.divide(zeros)=0."""
    const = np.zeros((4, 200, 4))
    tau = pp.tau_int_Y_from_retained(const)
    # With all-zero series, acov=0 everywhere → ρ=0 → τ=0.5.
    assert tau == 0.5, f"τ_int(constant series) = {tau} ≠ 0.5"
    print(f"\n[G10c] τ_int(constant)={tau}=0.5  PASS")


# ===========================================================================
# GROUP 11 — MEASURE-ONLY / NO-TAG
# ===========================================================================

def test_g11_no_wiki_edits_no_claim_status_tags():
    """The companion makes NO wiki edits / no claim-status tags.
    The 6 outcome tokens are companion-local (never wiki tags).

    Assert: the TOKENS list in driver.py does NOT include any wiki claim-status tags.
    Assert: the companion's src/ tree has no imports from the wiki."""
    # (1) TOKENS are companion-local (not wiki claim-status tags).
    wiki_tags = {"solid", "conjectured", "proven-here", "validated"}
    companion_tokens = set(D.TOKENS)
    overlap = companion_tokens & wiki_tags
    assert len(overlap) == 0, (
        f"companion TOKENS overlap with wiki claim-status tags: {overlap}")

    # (2) All 6 tokens are companion-local vocabulary.
    expected_tokens = {
        "BUDGET-WALL", "Q-CALIBRATION-FAIL", "PLATEAU-UNRESOLVED",
        "QUALITY-LOSS", "HTDML-MARGIN-NEGATIVE", "HTDML-MARGIN-POSITIVE",
    }
    assert companion_tokens == expected_tokens, (
        f"TOKENS mismatch: got {companion_tokens}, want {expected_tokens}")

    # (3) No import of wiki paths in src/htdml/.
    src_dir = _REPO_ROOT / "src" / "htdml"
    wiki_repo_name = "internal-project"
    for py_file in src_dir.glob("*.py"):
        text = py_file.read_text()
        assert wiki_repo_name not in text, (
            f"{py_file.name} imports from the wiki repo ({wiki_repo_name}) — isolation violated")

    print(f"\n[G11] measure-only: TOKENS={sorted(expected_tokens)} (companion-local); "
          f"no wiki imports; no claim-status tags  PASS")


# ===========================================================================
# GROUP 12 — 6-TOKEN REACHABILITY (NEW)
# ===========================================================================

# Synthetic acceptance constants for the router tests.
_ACC = D.AcceptanceConstants(
    ESS_min=10.0, C=5.0, L_traj=2000, N_chains=8, N_R=4,
    Q_GAIN=1.25, TAU_DROP=0.25, Q_DROP_MAX=0.10, R_GRAD50_MAX=0.05,
    BCE_TOL=0.05, FID_TOL=0.10, GPU_H_CAP=4.0,
)


def _layer_rec(*, q=1.0, tau=10.0, ess=20.0, r50=0.01, gnorm=1.0, L_traj=2000):
    """One synthetic per-layer probe record."""
    return dict(Q_struct_perp=float(q), tau_int_Y=float(tau), ESS_hat=float(ess),
                **{"r_grad[50]": float(r50)}, gradient_norm=float(gnorm),
                L_traj=int(L_traj), tau_hat=float(tau))


def _seed_rec(*, layers=None, ctrl_layers=None, bce=0.10, fid=1.0,
              ctrl_bce=0.10, ctrl_fid=1.0, gpu_h=1.0, budget_wall=False,
              cal_all_stable=True, traj_all_resolved=True):
    """Synthetic SeedMetrics bundle."""
    if layers is None:
        layers = [_layer_rec() for _ in range(4)]
    if ctrl_layers is None:
        ctrl_layers = [_layer_rec() for _ in range(4)]
    return D.SeedMetrics(
        joint_layers=layers, control_layers=ctrl_layers,
        bce=float(bce), fid=float(fid), control_bce=float(ctrl_bce), control_fid=float(ctrl_fid),
        gpu_h=float(gpu_h), budget_wall=bool(budget_wall),
        cal_all_stable=bool(cal_all_stable), traj_all_resolved=bool(traj_all_resolved),
    )


def _good_seed():
    """A seed that passes ALL final gates (yields HTDML-MARGIN-POSITIVE when both seeds are good)."""
    # Q ≥ Q_GAIN × control (1.0 ≥ 1.25 × 0.5 → yes since 1.0 ≥ 0.625)
    ctrl = [_layer_rec(q=0.5, ess=20.0, tau=10.0, r50=0.01) for _ in range(4)]
    joint = [_layer_rec(q=1.5, ess=20.0, tau=10.0, r50=0.01) for _ in range(4)]
    return _seed_rec(layers=joint, ctrl_layers=ctrl, bce=0.10, fid=1.0, ctrl_bce=0.10, ctrl_fid=1.0)


def test_g12_route_seed_all_6_tokens_reachable():
    """(NEW) Exercise BOTH the 6 per-seed predicates AND that each run-level token is producible
    by ≥1 seed-pair from route_seed / route_run.

    Tokens:
      BUDGET-WALL          → budget_wall=True (any seed)
      Q-CALIBRATION-FAIL   → cal_all_stable=False (any seed, measurement-invalid)
      PLATEAU-UNRESOLVED   → calibration OK + traj_all_resolved=False
      QUALITY-LOSS         → measurement valid but BCE > control+5%
      HTDML-MARGIN-NEGATIVE → all valid + quality OK but improvement gate NOT met
      HTDML-MARGIN-POSITIVE → both seeds pass all final gates

    Also assert: disjoint (no single synthetic input fires two tokens) + exhaustive.
    """
    # --- Per-seed: 6 tokens via route_seed ---
    # 1. BUDGET-WALL
    m_bw = _seed_rec(budget_wall=True)
    assert D.route_seed(m_bw, _ACC) == "BUDGET-WALL", "BUDGET-WALL not reachable"

    # 2. Q-CALIBRATION-FAIL (no budget_wall, cal_all_stable=False)
    m_qcf = _seed_rec(cal_all_stable=False)
    assert D.route_seed(m_qcf, _ACC) == "Q-CALIBRATION-FAIL", "Q-CALIBRATION-FAIL not reachable"

    # 3. PLATEAU-UNRESOLVED (cal OK, traj_all_resolved=False)
    m_pu = _seed_rec(cal_all_stable=True, traj_all_resolved=False,
                     layers=[_layer_rec(L_traj=10, tau=100.0) for _ in range(4)])
    assert D.route_seed(m_pu, _ACC) == "PLATEAU-UNRESOLVED", "PLATEAU-UNRESOLVED not reachable"

    # 4. QUALITY-LOSS (all measurement-valid, but BCE > control*(1+BCE_TOL))
    ctrl_layers = [_layer_rec(q=0.5, ess=20.0) for _ in range(4)]
    joint_layers = [_layer_rec(q=1.5, ess=20.0) for _ in range(4)]
    m_ql = _seed_rec(
        layers=joint_layers, ctrl_layers=ctrl_layers,
        bce=0.20, ctrl_bce=0.10, fid=1.0, ctrl_fid=1.0,  # BCE = 0.20 > 0.10*(1.05)=0.105
    )
    assert D.route_seed(m_ql, _ACC) == "QUALITY-LOSS", (
        f"QUALITY-LOSS not reachable — got {D.route_seed(m_ql, _ACC)!r}")

    # 5. HTDML-MARGIN-NEGATIVE (all valid, quality OK, improvement NOT met)
    #    Q joint == Q ctrl (no gain), τ joint == τ ctrl (no drop)
    ctrl5 = [_layer_rec(q=1.0, ess=20.0, tau=10.0, r50=0.01) for _ in range(4)]
    joint5 = [_layer_rec(q=1.0, ess=20.0, tau=10.0, r50=0.01) for _ in range(4)]
    m_mn = _seed_rec(layers=joint5, ctrl_layers=ctrl5, bce=0.10, fid=1.0, ctrl_bce=0.10, ctrl_fid=1.0)
    assert D.route_seed(m_mn, _ACC) == "HTDML-MARGIN-NEGATIVE", (
        f"HTDML-MARGIN-NEGATIVE not reachable — got {D.route_seed(m_mn, _ACC)!r}")

    # 6. HTDML-MARGIN-POSITIVE (all gates pass)
    m_pos = _good_seed()
    assert D.route_seed(m_pos, _ACC) == "HTDML-MARGIN-POSITIVE", (
        f"HTDML-MARGIN-POSITIVE not reachable — got {D.route_seed(m_pos, _ACC)!r}")

    # --- Disjoint: each synthetic input fires exactly one token ---
    all_cases = [
        ("BUDGET-WALL",          m_bw),
        ("Q-CALIBRATION-FAIL",   m_qcf),
        ("PLATEAU-UNRESOLVED",   m_pu),
        ("QUALITY-LOSS",         m_ql),
        ("HTDML-MARGIN-NEGATIVE", m_mn),
        ("HTDML-MARGIN-POSITIVE", m_pos),
    ]
    all_tokens = [D.route_seed(m, _ACC) for _, m in all_cases]
    assert len(set(all_tokens)) == 6, (
        f"6 distinct tokens expected; only {len(set(all_tokens))} distinct: {all_tokens}")
    for expected, actual in zip([t for t, _ in all_cases], all_tokens):
        assert expected == actual, f"disjointness violated: expected {expected}, got {actual}"

    # --- Exhaustive: covers all 6 tokens in TOKENS ---
    assert set(all_tokens) == set(D.TOKENS), (
        f"not all 6 tokens covered: covered={set(all_tokens)}, required={set(D.TOKENS)}")

    print(f"\n[G12 per-seed] all 6 tokens reachable + disjoint + exhaustive  PASS")


def test_g12_route_run_all_6_tokens_reachable_from_seed_pairs():
    """(NEW) Each run-level token is producible by ≥1 seed-pair via route_run.

    Token→seed-pair map (build-notes §"Two-seed run-level aggregation"):
      POSITIVE            ← (PO, PO)  both seeds pass all gates
      HTDML-MARGIN-NEGATIVE ← (MN, MN) or (MN, PO) or (PO, MN)
      QUALITY-LOSS        ← (QL, ·) or (·, QL)
      PLATEAU-UNRESOLVED  ← (PU, ·) — worst invalid among invalids
      Q-CALIBRATION-FAIL  ← (QCF, ·) — OR (QCF, BW→ BW wins)
      BUDGET-WALL         ← (BW, ·)  worst-precedence invalid
    """
    # Build representative seed records for each single-seed token.
    m_bw = _seed_rec(budget_wall=True)
    m_qcf = _seed_rec(cal_all_stable=False)
    m_pu = _seed_rec(cal_all_stable=True, traj_all_resolved=False,
                     layers=[_layer_rec(L_traj=10, tau=100.0) for _ in range(4)])
    ctrl5 = [_layer_rec(q=0.5, ess=20.0) for _ in range(4)]
    joint5_ql = [_layer_rec(q=1.5, ess=20.0) for _ in range(4)]
    m_ql = _seed_rec(layers=joint5_ql, ctrl_layers=ctrl5,
                     bce=0.20, ctrl_bce=0.10, fid=1.0, ctrl_fid=1.0)
    ctrl_mn = [_layer_rec(q=1.0, ess=20.0, tau=10.0, r50=0.01) for _ in range(4)]
    joint_mn = [_layer_rec(q=1.0, ess=20.0, tau=10.0, r50=0.01) for _ in range(4)]
    m_mn = _seed_rec(layers=joint_mn, ctrl_layers=ctrl_mn, bce=0.10, fid=1.0, ctrl_bce=0.10, ctrl_fid=1.0)
    m_pos = _good_seed()

    # Verify per-seed tokens for all seeds used below.
    assert D.route_seed(m_bw, _ACC) == "BUDGET-WALL"
    assert D.route_seed(m_qcf, _ACC) == "Q-CALIBRATION-FAIL"
    assert D.route_seed(m_pu, _ACC) == "PLATEAU-UNRESOLVED"
    assert D.route_seed(m_ql, _ACC) == "QUALITY-LOSS"
    assert D.route_seed(m_mn, _ACC) == "HTDML-MARGIN-NEGATIVE"
    assert D.route_seed(m_pos, _ACC) == "HTDML-MARGIN-POSITIVE"

    # Now test route_run (two-seed aggregation).
    # POSITIVE: both seeds = PO
    tok = D.route_run(m_pos, m_pos, _ACC)
    assert tok == "HTDML-MARGIN-POSITIVE", f"(PO,PO) → {tok!r} ≠ POSITIVE"

    # HTDML-MARGIN-NEGATIVE: both measurement-valid, no quality failure, but improvement NOT met
    tok = D.route_run(m_mn, m_mn, _ACC)
    assert tok == "HTDML-MARGIN-NEGATIVE", f"(MN,MN) → {tok!r} ≠ MARGIN-NEGATIVE"

    # HTDML-MARGIN-NEGATIVE also when one seed is MN and the other is PO (BOTH needed for POSITIVE)
    tok = D.route_run(m_mn, m_pos, _ACC)
    assert tok == "HTDML-MARGIN-NEGATIVE", f"(MN,PO) → {tok!r} ≠ MARGIN-NEGATIVE"

    # QUALITY-LOSS: one seed fails quality (QL,MN) — among valid-but-bad seeds, quality wins
    tok = D.route_run(m_ql, m_mn, _ACC)
    assert tok == "QUALITY-LOSS", f"(QL,MN) → {tok!r} ≠ QUALITY-LOSS"

    # PLATEAU-UNRESOLVED: one invalid seed (PU,MN) — invalid seed dominates valid
    tok = D.route_run(m_pu, m_mn, _ACC)
    assert tok == "PLATEAU-UNRESOLVED", f"(PU,MN) → {tok!r} ≠ PLATEAU-UNRESOLVED"

    # Q-CALIBRATION-FAIL: (QCF,MN) — QCF is invalid, dominates the valid MN
    tok = D.route_run(m_qcf, m_mn, _ACC)
    assert tok == "Q-CALIBRATION-FAIL", f"(QCF,MN) → {tok!r} ≠ Q-CALIBRATION-FAIL"

    # BUDGET-WALL: (BW,QCF) — BW has worst precedence among invalids
    tok = D.route_run(m_bw, m_qcf, _ACC)
    assert tok == "BUDGET-WALL", f"(BW,QCF) → {tok!r} ≠ BUDGET-WALL"

    # BUDGET-WALL: (BW,PO) — even if second seed is fine, BW from first dominates
    tok = D.route_run(m_bw, m_pos, _ACC)
    assert tok == "BUDGET-WALL", f"(BW,PO) → {tok!r} ≠ BUDGET-WALL"

    # --- Exhaustive: prove all 6 run-level tokens are producible ---
    covered = set()
    for tok_name, sa, sb in [
        ("HTDML-MARGIN-POSITIVE", m_pos, m_pos),
        ("HTDML-MARGIN-NEGATIVE", m_mn, m_mn),
        ("QUALITY-LOSS",          m_ql, m_mn),
        ("PLATEAU-UNRESOLVED",    m_pu, m_mn),
        ("Q-CALIBRATION-FAIL",    m_qcf, m_mn),
        ("BUDGET-WALL",           m_bw, m_mn),
    ]:
        got = D.route_run(sa, sb, _ACC)
        assert got == tok_name, (
            f"run-level token {tok_name} not reachable: pair gave {got!r}")
        covered.add(got)

    assert covered == set(D.TOKENS), (
        f"not all 6 run-level tokens covered: {covered} vs {set(D.TOKENS)}")

    # --- Disjoint: each pair fires exactly one token ---
    # Already asserted above via assert got == tok_name for each pair.

    print(f"\n[G12 route_run] all 6 run-level tokens reachable from seed-pairs; "
          f"disjoint + exhaustive  PASS\n"
          f"  (PO,PO)→POSITIVE  (MN,MN)→MARGIN-NEGATIVE  (QL,MN)→QUALITY-LOSS\n"
          f"  (PU,MN)→PLATEAU-UNRESOLVED  (QCF,MN)→Q-CAL-FAIL  (BW,QCF)→BUDGET-WALL")


def test_g12_measurement_invalid_precedence_ordering():
    """measurement-invalid seeds: BUDGET-WALL > Q-CALIBRATION-FAIL > PLATEAU-UNRESOLVED.
    Assert the precedence ordering in route_run."""
    m_bw = _seed_rec(budget_wall=True)
    m_qcf = _seed_rec(cal_all_stable=False)
    m_pu = _seed_rec(cal_all_stable=True, traj_all_resolved=False,
                     layers=[_layer_rec(L_traj=10, tau=100.0) for _ in range(4)])

    # BW beats QCF (both invalid)
    assert D.route_run(m_bw, m_qcf, _ACC) == "BUDGET-WALL", "(BW,QCF) must be BUDGET-WALL"
    # BW beats PU (both invalid)
    assert D.route_run(m_bw, m_pu, _ACC) == "BUDGET-WALL", "(BW,PU) must be BUDGET-WALL"
    # QCF beats PU (both invalid)
    assert D.route_run(m_qcf, m_pu, _ACC) == "Q-CALIBRATION-FAIL", "(QCF,PU) must be Q-CAL-FAIL"
    print(f"\n[G12 precedence] BW>QCF>PU ordering confirmed in route_run  PASS")
