"""Task 3 — harness/probe_primitives.py (the shared primitive layer) — TDD gate.

Two parts under test:

  PART A — VERBATIM-faithful ports of the internal reference sokal / energy / refresh
  primitives (the math is already validated upstream; these tests pin the math, not re-derive it):
    * half-Sokal τ_int floor (τ=0.5 on a constant / IID series),
    * the three-term `energy_free` conditional (the coupling-to-clamped-input term is LOAD-BEARING),
    * the MANDATORY trained-weight refresh proof (`refreshed_weight_proof` returns
      refresh_ok=True ∧ constructor_was_stale=True on a perturbed tiny DTM) — the single most
      important guard in the whole build.

  PART B — the companion's K=50 Y-process + Rademacher sketch:
    * white-noise SE on the retained process,
    * determinism (same input + key → identical output),
    * the ESS / Q_struct^⊥ formula shapes (ESS=50/(2τ); prefactor K/2=25),
    * the Rademacher worst-of-N_R reduction.

conftest.py installs the vendored isolation; probe_primitives self-bootstraps on import too.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

# conftest installs src/ + vendored paths; be explicit so the file is runnable directly too.
_REPO_ROOT = Path(__file__).resolve().parents[1]
_SRC = str(_REPO_ROOT / "src")
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

import htdml  # noqa: E402,F401  (triggers bootstrap_paths)

from harness import probe_primitives as pp  # noqa: E402


# ============================================================ PART A — sokal floor (τ = 0.5)
def test_tau_half_floor_constant_series():
    """half-Sokal τ_int = 0.5 on a zero-variance series — the floor mechanism in the verbatim
    estimator: a0 == 0 → `np.divide(..., where=a0>0)` leaves ρ == 0 → no positive ρ-pair → τ = 0.5.
    Uses an exactly-representable constant (2.0) so block.mean is exact and a0 is exactly 0 (a
    non-representable constant like 1.7 leaves float-rounding residual that autocorrelates to ~1 — that
    is the genuine estimator behaviour, NOT the floor; the IID test below covers the practical floor)."""
    block = np.full((4, 256, 3), 2.0, dtype=np.float64)
    rho = pp._rho_block(block)
    assert np.max(np.abs(rho)) == 0.0, "exact-constant block must give ρ == 0 (a0 == 0 branch)"
    tau = pp._tau_half_from_rho(rho)
    assert tau.shape == (3,)
    assert np.allclose(tau, 0.5), f"zero-variance-series tau should be the 0.5 floor, got {tau}"


def test_tau_half_floor_iid_series():
    """half-Sokal τ_int ≈ 0.5 on an IID (white-noise) series — the no-autocorrelation floor."""
    rng = np.random.default_rng(0)
    # long IID block: rho(ℓ≥1) ~ 0 within SE → the positive-pair truncation lands ~immediately → τ ≈ 0.5
    block = rng.standard_normal((16, 4096, 5))
    tau = pp._tau_half_from_rho(pp._rho_block(block))
    # IID floor: τ should be very close to 0.5 (a small positive bias from the first lucky positive pair)
    assert np.all(tau >= 0.5 - 1e-9)
    assert np.all(tau < 0.65), f"IID series tau drifted from the 0.5 floor: {tau}"


def test_sokal_profile_returns_TO_half_sum_Sa():
    """sokal_profile_from_spins returns (tau_max, T_O, S_a) with T_O == ½·Σ_a S_a and S_a = 2·τ_a·Var_a."""
    rng = np.random.default_rng(1)
    n_chains, L, n_free = 8, 512, 6
    spins = (rng.integers(0, 2, size=(n_chains, L, n_free)) * 2 - 1).astype(np.float32)
    maps = _toy_maps(n_free)
    tau_max, T_O, S_a = pp.sokal_profile_from_spins(spins, maps)
    P = maps["n_edge"] + maps["n_bias"]
    assert S_a.shape == (P,)
    assert np.isclose(T_O, 0.5 * S_a.sum())
    assert tau_max >= 0.5


# ============================================================ PART A — energy_free three-term
def test_energy_free_three_term_coupling_is_load_bearing():
    """energy_free with a NONZERO clamp differs from a ZERO clamp — the coupling-to-clamped-input term
    is non-vanishing (the load-bearing third term)."""
    rng = np.random.default_rng(2)
    n_chains, n_free, n_clamp = 5, 6, 4
    maps = _toy_maps(n_free, n_clamp=n_clamp, n_coupling=3)
    spins = (rng.integers(0, 2, size=(n_chains, n_free)) * 2 - 1).astype(np.float64)
    clamp_nonzero = (rng.integers(0, 2, size=(n_chains, n_clamp)) * 2 - 1).astype(np.float64)
    clamp_zero = np.zeros((n_chains, n_clamp), dtype=np.float64)

    e_nonzero = pp.energy_free(spins, clamp_nonzero, maps)
    e_zero = pp.energy_free(spins, clamp_zero, maps)
    # base-edge + free-bias terms identical; only the coupling term differs → energies must differ
    assert not np.allclose(e_nonzero, e_zero), (
        "coupling-to-clamped-input term vanished — the three-term energy_free is broken")
    # and the difference equals exactly the coupling term evaluated against the nonzero clamp
    e_coup = -(spins[:, maps["coup_out_pos"]] * clamp_nonzero[:, maps["coup_in_pos"]]) @ maps["coup_w"]
    assert np.allclose(e_nonzero - e_zero, e_coup)


# ============================================================ PART A — trained-weight refresh proof
def test_refreshed_weight_proof_on_tiny_dtm():
    """THE single most important guard: on a tiny perturbed DTM the refresh proof must return
    refresh_ok=True ∧ constructor_was_stale=True (a rebuilt program reads stale INIT factors; the
    refresh injects the trained weights). NEEDS_CONTEXT (not skip) if a tiny DTM can't instantiate."""
    step = _tiny_perturbed_step()
    proof = pp.refreshed_weight_proof(step)
    assert set(["refresh_ok", "constructor_was_stale", "refreshed_vs_trained_maxabs"]).issubset(proof)
    assert proof["constructor_was_stale"] is True, (
        f"constructor was NOT stale (stale_vs_trained_maxabs="
        f"{proof['stale_vs_trained_maxabs']}) — the stale-factors bug premise does not hold here")
    assert proof["refresh_ok"] is True, (
        f"refresh did NOT take (refreshed_vs_trained_maxabs="
        f"{proof['refreshed_vs_trained_maxabs']}) — the mandatory weight refresh is broken")
    assert proof["refreshed_vs_trained_maxabs"] < 1e-6
    assert proof["stale_vs_trained_maxabs"] > 1e-6


def test_refresh_program_weights_injects_trained_weights():
    """refresh_program_weights returns a program whose per-block interactions equal the trained weights."""
    step = _tiny_perturbed_step()
    import jax.numpy as jnp

    from thrmlDenoising.annealing_graph_ising import AnnealingIsingSamplingProgram

    ts = step.training_spec
    prog = AnnealingIsingSamplingProgram(
        step.model, list(ts.program_negative.gibbs_spec.free_blocks),
        list(ts.program_negative.gibbs_spec.clamped_blocks), jnp.asarray(1.0), ts.schedule_negative)
    refreshed = pp.refresh_program_weights(prog, step)
    # first weight interaction in the refreshed program must equal trained step.model.weights[gi]
    fresh = []
    pp._collect_weight_interactions(refreshed.per_block_interactions, fresh)
    gi, wv = fresh[0]
    wt = np.asarray(step.model.weights)
    assert np.max(np.abs(wv - wt[gi])) < 1e-6


# ============================================================ PART A — provenance helpers
def test_weights_hash_changes_with_perturbation():
    """_weights_hash differs between init and a perturbed step (rollback / provenance helper)."""
    init_step = _tiny_step()
    pert_step = _perturb(init_step)
    assert pp._weights_hash(init_step) != pp._weights_hash(pert_step)


# ============================================================ PART B — K=50 Y-process + Rademacher
def test_white_noise_se_retained_process():
    """On an IID retained process, ρ_Y(ℓ) for ℓ≥1 is ~0 within the white-noise SE (≈1/√(N·L)).
    Assert |ρ_Y(1)| < a few SE."""
    rng = np.random.default_rng(3)
    n_chains, L, b = 32, 4000, 4
    retained = rng.standard_normal((n_chains, L, b))
    rho = pp.rho_Y(retained)
    se = 1.0 / np.sqrt(n_chains * L)
    assert abs(rho[1].max()) < 5 * se, f"rho_Y(1)={rho[1]} exceeds 5·SE={5*se:.4g} on white noise"


def test_determinism_same_input_key():
    """Same retained observables + same diag_key → identical probe_scalars output (reproducible)."""
    rng = np.random.default_rng(4)
    retained = rng.standard_normal((16, 300, 8))
    g = rng.standard_normal(8)
    out1 = pp.probe_scalars(retained, n_R=16, diag_key=7, gradient=g)
    out2 = pp.probe_scalars(retained, n_R=16, diag_key=7, gradient=g)
    for k in out1:
        assert out1[k] == out2[k] or np.allclose(out1[k], out2[k]), f"non-deterministic field {k}"


def test_ess_and_q_formula_shapes():
    """ESS_hat == 50/(2·τ_int,Y); Q_struct^⊥ prefactor is K/2 = 25."""
    rng = np.random.default_rng(5)
    retained = rng.standard_normal((16, 300, 6))
    g = rng.standard_normal(6)
    out = pp.probe_scalars(retained, n_R=8, diag_key=11, gradient=g)
    tau = out["tau_int_Y"]
    assert np.isclose(out["ESS_hat"], 50.0 / (2.0 * tau))
    # Q_struct^⊥ = (K/2)·‖g‖² / T_{O,Y}; reconstruct T_{O,Y} and check the prefactor is exactly 25
    assert np.isclose(out["Q_struct_perp"], 25.0 * out["gradient_norm"] ** 2 / out["_T_O_Y"])
    assert np.isclose(pp.K_WINDOW / 2.0, 25.0)


def test_rademacher_worst_of_NR():
    """The reported margin equals the WORST (max τ_int,Y) across the N_R Rademacher sketches."""
    rng = np.random.default_rng(6)
    retained = rng.standard_normal((16, 400, 12))
    n_R = 16
    res = pp.rademacher_sketch_scalars(retained, n_R=n_R, diag_key=21)
    # the per-sketch taus are exposed; the reported tau must be their max (worst-case screening)
    assert len(res["per_sketch_tau"]) == n_R
    assert np.isclose(res["tau_int_Y"], max(res["per_sketch_tau"]))
    # and T_{O,Y} is reported for the same (worst) sketch
    worst_idx = int(np.argmax(res["per_sketch_tau"]))
    assert np.isclose(res["_T_O_Y"], res["per_sketch_T_O"][worst_idx])


def test_rademacher_sketches_deterministic_given_key():
    """The Rademacher sketch vectors are deterministic given a fixed diag_key (reproducible)."""
    s1 = pp.rademacher_sketches(p=20, n_R=16, diag_key=99)
    s2 = pp.rademacher_sketches(p=20, n_R=16, diag_key=99)
    s3 = pp.rademacher_sketches(p=20, n_R=16, diag_key=100)
    assert np.array_equal(s1, s2)
    assert s1.shape == (16, 20)
    assert set(np.unique(s1)).issubset({-1.0, 1.0})
    assert not np.array_equal(s1, s3), "different key must give different sketches"


# ============================================================ test fixtures (toy maps + tiny DTM)
def _toy_maps(n_free, n_clamp=0, n_coupling=0, seed=0):
    """A minimal `maps` dict sufficient for energy_free / sokal_profile_from_spins (no DTM needed)."""
    rng = np.random.default_rng(seed)
    n_edge = max(1, n_free - 1)
    n_bias = n_free
    e0 = np.arange(n_edge, dtype=np.int32)
    e1 = (np.arange(n_edge, dtype=np.int32) + 1) % n_free
    bp = np.arange(n_bias, dtype=np.int32)
    maps = dict(
        edge_pos0=e0, edge_pos1=e1, bias_pos=bp,
        n_edge=int(n_edge), n_bias=int(n_bias), n_free=int(n_free),
        W_e=rng.standard_normal(n_edge), b_n=rng.standard_normal(n_bias),
        coup_out_pos=np.zeros(n_coupling, dtype=np.int32),
        coup_in_pos=np.zeros(n_coupling, dtype=np.int32),
        coup_w=np.zeros(n_coupling, dtype=np.float64),
        n_clamp=int(n_clamp), n_coupling=int(n_coupling),
    )
    if n_coupling > 0:
        maps["coup_out_pos"] = rng.integers(0, n_free, size=n_coupling).astype(np.int32)
        maps["coup_in_pos"] = rng.integers(0, max(1, n_clamp), size=n_coupling).astype(np.int32)
        maps["coup_w"] = rng.standard_normal(n_coupling)
    return maps


_TINY_CFG = dict(
    exp=dict(seed=0, descriptor="probe_test", compute_autocorr=False, generate_gif=False, n_cores=1),
    data=dict(dataset_name="smoke_testing_3_1_3", target_classes=tuple(range(3)),
              pixel_threshold_for_single_trials=0.1),
    graph=dict(graph_preset_architecture=6_4, num_label_spots=1, grayscale_levels=1, torus=True,
               base_graph_manager="poisson_binomial_ising_graph_manager"),
    sampling=dict(batch_size=400, n_samples=2, steps_per_sample=2, steps_warmup=4, training_beta=1.0),
    diffusion_schedule=dict(num_diffusion_steps=1, kind="log", diffusion_offset=0.1),
    diffusion_rates=dict(image_rate=0.8, label_rate=0.2),
    optim=dict(momentum=0.9, b2_adam=0.999, step_learning_rates=(0.05,), alpha_cosine_decay=0.2,
               n_epochs_for_lrd=50),
)


def _tiny_step():
    """Build the smallest instantiable DTM (3-pixel smoke dataset, 6_4 preset) and return step 0.
    NEEDS_CONTEXT (fail loudly) if it cannot be instantiated on CPU."""
    from thrmlDenoising.DTM import DTM
    from thrmlDenoising.utils import make_cfg

    try:
        dtm = DTM(make_cfg(**_TINY_CFG))
    except Exception as e:  # pragma: no cover
        pytest.fail(f"NEEDS_CONTEXT: tiny DTM could not be instantiated on CPU for the refresh-proof: {e}")
    return dtm.steps[0]


def _perturb(step, scale=0.5, seed=123):
    """Perturb step.model.weights/biases EXACTLY as DTM.train's write-back does — tree_at updates
    weights/biases + the program per_block_interactions but DELIBERATELY leaves model.factors stale
    (DTM.py:337-340 omits model.factors). This is a faithful, GPU-free reproduction of the trained
    state that triggers the stale-factors bug."""
    import equinox as eqx
    import jax.random as jr

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


def _tiny_perturbed_step():
    return _perturb(_tiny_step())
