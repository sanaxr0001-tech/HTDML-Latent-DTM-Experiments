"""Task 4 — the enumerable correctness ARBITER fixture (MECHANISM checks a, b, c, e).

This module builds a small DTM model isomorphic to the production superblock structure and
validates the build's MECHANISMS by EXACT enumeration. It is the arbiter the rest of the companion
trusts: every check below HARD-FAILS (pytest assertion) on violation.

  (a) hard-bit forward noising + b_t stop-gradient  — the REAL forward noiser `get_perturbed_data`
      (step.py:454, used at step.py:203-204) draws HARD {0,1} bits → hard {−1,+1} spins, and NO
      gradient flows from b_t into b0 / the encoder (b_t is `stop_gradient(forward_noise(b0))`).
  (b) reversible-sampler detailed balance               — re-certify DB (reuse `selfadjoint_cert`) on
      the fixture's 4-superblock negative-phase structure (max_asym < 1e-10).
  (c) Rademacher / Sokal trainability estimator vs EXACT — full 2^N enumeration of the Boltzmann law
      gives the EXACT Var_π[f_a] and the EXACT T_O via the reversible kernel's enumerable transition
      matrix (exp1-style); the real `sokal_profile_from_spins` estimator on CPU trajectories from that
      same kernel must AGREE within a stated, SE-aware tolerance.
  (e) checkpoint rollback reproducibility               — perturb (eqx.tree_at write-back, NO
      dtm.train), save (the DTM.save_epoch eqx-partition), mutate, restore (the DTM.load eqx path);
      `_weights_hash` / `_key_list` / `_find_counts` + the autocorrelations dict all reproduce.

THE SIZING DECISION (resolved by inspecting source — documented in task-4-report.md):
  * The grid is `side_len²` nodes (poisson_binomial_ising_graph_manager.py:122 `size = side_len**2`),
    and the 4 free superblocks {upper_hidden, lower_hidden, image_output, label_output} together cover
    EVERY grid node (visibles ⊂ upper; hidden = rest of upper + all lower) → N_total = side_len².
  * The literal `6_4` preset = side 6 = 36 free nodes → 2^36 NOT enumerable.
  * The SMALLEST real preset that (i) instantiates with a real smoke dataset, (ii) keeps N_total ≤ ~20,
    (iii) is bipartite under the PoissonBinomial manager, is `4_4` (side 4 → N_total = 16, 2^16 = 65536).
    With `smoke_testing_3_1_3` + num_label_spots=1: upper_hidden=2, lower_hidden=8, image_output=3,
    label_output=3 → N_total = 16; clamp b_t = 6 input nodes; 6 coupling edges; 32 base edges.
    This is the REAL DTM (real build_maps / energy_free / get_perturbed_data / DTM.load exercised).
  * The EXACT integrated-autocorrelation T_O needs the kernel's 2^N × 2^N transition matrix; a dense
    65536² matrix is ~34 GB → INFEASIBLE. So check (c)'s EXACT-T_O comparison uses a tiny (N=10)
    enumerable cell that is ISOMORPHIC to the 5-superblock layout, built with the cert's exp1-style
    exact machinery (`selfadjoint_cert.make_dtm_negative_cell` → `block_gibbs_matrix` →
    ½(P_fwd+P_rev)), and the REAL estimator `sokal_profile_from_spins`. The REAL 4_4 model's
    `energy_free` exact-π enumeration is ALSO exercised in (c) to confirm the real energy path
    enumerates. (Brief: "If NO real preset is small enough for full 2^N, hand-build a tiny Ising
    structure isomorphic to the 5-superblock layout … but PREFER a real DTM step.")

conftest.py installs the vendored isolation; the module self-bootstraps on import too.
"""

from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path

import numpy as np
import pytest

# conftest installs src/ + vendored paths; be explicit so the file is runnable directly too.
_REPO_ROOT = Path(__file__).resolve().parents[1]
for _p in (str(_REPO_ROOT), str(_REPO_ROOT / "src")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import htdml  # noqa: E402,F401  (triggers bootstrap_paths)

from harness import probe_primitives as pp  # noqa: E402
from harness import selfadjoint_cert as sc  # noqa: E402


# ====================================================================== the REAL fixture model (4_4)
# N_total = 16 free nodes (2^16 = 65536 fully enumerable). The exact superblock node counts are
# asserted in test_fixture_model_is_production_shape below (so the sizing decision is itself a gate).
FIXTURE_CFG = dict(
    exp=dict(seed=0, descriptor="fixture_6_4", compute_autocorr=False, generate_gif=False, n_cores=1),
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

# The production-shape invariants the fixture must satisfy (build-notes §"Training-negative free set").
SUPERBLOCK_NAMES = ("upper_hidden", "lower_hidden", "image_output", "label_output")
EXPECTED_BLOCK_LENS = [2, 8, 3, 3]            # 4_4 + smoke_testing_3_1_3 + num_label_spots=1
EXPECTED_N_TOTAL = 16                         # 2^16 = 65536 enumerable
EXPECTED_N_CLAMP = 6                          # b_t = 3 image_input + 3 label_input
EXPECTED_N_COUPLING = 6                       # 3 image + 3 label coupling edges (1-to-1 to outputs)

# Check (c) tolerances — JUSTIFIED in the report:
#   * Var_π is unbiased → tight band.
#   * The verbatim FFT half-Sokal T_O estimator is SYSTEMATICALLY biased LOW (chain-mean subtraction +
#     the `/L` divisor), ~0.86× exact and STABLE in L (cross-seed CV ≈ 0.3% at 512 chains × 4000) — so
#     the band is a BIAS band, not an SE band, and is one-sided-leaning (est ≤ exact).
VAR_RELERR_MEDIAN_TOL = 0.05                  # median rel-err of Var_π (estimator vs exact)
VAR_RELERR_MAX_TOL = 0.10                     # max rel-err of Var_π
TO_RATIO_LO, TO_RATIO_HI = 0.70, 1.10         # T_O_est / T_O_exact band (FFT half-Sokal bias)
TO_CROSS_SEED_CV_TOL = 0.05                   # cross-seed CV of T_O_est (proves the gap is bias, not SE)


# --------------------------------------------------------------------------- model builders (reused)
def _build_fixture_step():
    """Instantiate the REAL 4_4 tiny DTM and return a PERTURBED step 0 (trained-≠-init weights via the
    exact `eqx.tree_at` write-back DTM.train uses, deliberately leaving `model.factors` stale — the
    faithful, GPU-free exp15/16-bug reproduction). Does NOT call `dtm.train` (CPU constraint)."""
    from thrmlDenoising.DTM import DTM
    from thrmlDenoising.utils import make_cfg

    try:
        dtm = DTM(make_cfg(**FIXTURE_CFG))
    except Exception as e:  # pragma: no cover
        pytest.fail(f"NEEDS_CONTEXT: 4_4 fixture DTM could not be instantiated on CPU: {e}")
    return dtm, _perturb_step(dtm.steps[0], seed=123)


def _perturb_step(step, scale=0.5, seed=123):
    """Perturb weights/biases EXACTLY as DTM.train's write-back does (tree_at updates weights/biases +
    the program per_block_interactions but DELIBERATELY leaves `model.factors` stale, DTM.py:337-340).
    Faithful, GPU-free reproduction of the trained state that triggers the exp15/16 stale-factors bug."""
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


# ====================================================================== sizing gate (the model itself)
def test_fixture_model_is_production_shape():
    """The sizing decision is itself a gate: the REAL 4_4 fixture must have the production 4-superblock
    negative-phase structure with N_total = 16 (2^16 enumerable), the b_t clamp, the 1-to-1 output
    coupling, and the PoissonBinomial manager (the convolved manager breaks the bipartite premise)."""
    dtm, step = _build_fixture_step()

    # PoissonBinomial manager invariant (build-notes §"MUST assert base_graph_manager ...").
    assert dtm.cfg.graph.base_graph_manager == "poisson_binomial_ising_graph_manager", (
        "fixture must use the PoissonBinomial manager (convolved breaks 'all visibles in upper')")

    free_blocks = list(step.training_spec.program_negative.gibbs_spec.free_blocks)
    clamped_blocks = list(step.training_spec.program_negative.gibbs_spec.clamped_blocks)
    block_lens = [len(b) for b in free_blocks]
    n_total = sum(block_lens)
    n_clamp = sum(len(b) for b in clamped_blocks)

    # 4 free superblocks = hidden_blocks + data_blocks = [upper_hidden, lower_hidden, image, label].
    assert len(free_blocks) == 4, f"expected 4 free superblocks, got {len(free_blocks)}"
    assert block_lens == EXPECTED_BLOCK_LENS, (
        f"superblock node counts {block_lens} != expected {EXPECTED_BLOCK_LENS} "
        f"({list(SUPERBLOCK_NAMES)})")
    assert n_total == EXPECTED_N_TOTAL, f"N_total {n_total} != {EXPECTED_N_TOTAL} (2^N must be enumerable)"
    assert n_total <= 20, "N_total must be ≤ ~20 for full 2^N enumeration"
    assert len(clamped_blocks) == 1 and n_clamp == EXPECTED_N_CLAMP, (
        f"clamp must be the single b_t block of {EXPECTED_N_CLAMP} input nodes, got {n_clamp}")

    # maps (the REAL build_maps path) — coupling 1-to-1 to outputs, real base edges.
    maps = pp.build_maps(step)
    assert maps["n_free"] == EXPECTED_N_TOTAL
    assert maps["n_coupling"] == EXPECTED_N_COUPLING, (
        f"b_t coupling edges {maps['n_coupling']} != {EXPECTED_N_COUPLING} (1-to-1 to outputs)")
    assert maps["n_clamp"] == EXPECTED_N_CLAMP
    assert maps["n_edge"] > 0 and maps["n_bias"] == EXPECTED_N_TOTAL
    print(f"\n[FIXTURE] real 4_4 DTM: superblocks {list(SUPERBLOCK_NAMES)} sizes {block_lens} "
          f"N_total={n_total} (2^N={2**n_total}); clamp b_t={n_clamp}; coupling={maps['n_coupling']}; "
          f"base_edges={maps['n_edge']}; manager=PoissonBinomial")


# ====================================================================== (a) hard-bit + b_t stop-gradient
def test_a_forward_noise_is_hard_bit_draw():
    """(a-i) The REAL forward noiser `get_perturbed_data` (step.py:454) produces HARD {0,1} bit draws
    (grayscale-1) → hard {−1,+1} ising spins, NOT soft probabilities."""
    import jax.numpy as jnp
    import jax.random as jr

    from thrmlDenoising.step import get_perturbed_data

    b0 = jnp.array([[0.0, 1.0, 0.0], [1.0, 0.0, 1.0]])       # 2 examples × 3 pixels, grayscale-1 ∈ {0,1}
    bt = get_perturbed_data(jr.PRNGKey(0), b0, dt=0.5, rates=0.8, bin_trials=1)
    vals = np.unique(np.asarray(bt))
    assert set(vals.tolist()).issubset({0.0, 1.0}), f"forward noise not a hard 0/1 draw: {vals}"
    spins = 2 * np.asarray(bt) - 1                            # ising convention 2b−1
    assert set(np.unique(spins).tolist()).issubset({-1.0, 1.0}), (
        f"b_t spins not hard ±1: {np.unique(spins)}")


def test_a_bt_stop_gradient_detaches_encoder():
    """(a-ii) b_t = stop_gradient(forward_noise(b0)) carries NO gradient back into b0 / the encoder:
    jax.grad of any function of b_t w.r.t. b0 is EXACTLY zero (only b0 carries ∂L/∂latent)."""
    import jax
    import jax.numpy as jnp
    import jax.random as jr

    from thrmlDenoising.step import get_perturbed_data

    key = jr.PRNGKey(1)
    b0 = jnp.array([[0.0, 1.0, 0.0], [1.0, 0.0, 1.0]])

    def loss_through_bt(b0):
        bt = jax.lax.stop_gradient(get_perturbed_data(key, b0, dt=0.5, rates=0.8, bin_trials=1))
        return jnp.sum(bt ** 2 + 3.0 * bt)                   # arbitrary non-trivial fn of b_t

    g = np.asarray(jax.grad(loss_through_bt)(b0))
    assert np.all(g == 0.0), f"gradient leaked through b_t into b0 (must be all-zero): {g}"

    # and even WITHOUT stop_gradient the discrete bernoulli draw is non-differentiable (also zero) —
    # belt-and-suspenders: the detachment is structural, not only the explicit stop_gradient.
    def loss_no_sg(b0):
        bt = get_perturbed_data(key, b0, dt=0.5, rates=0.8, bin_trials=1)
        return jnp.sum(bt ** 2 + 3.0 * bt)

    g2 = np.asarray(jax.grad(loss_no_sg)(b0))
    assert np.all(g2 == 0.0), f"unexpected nonzero grad through a discrete draw: {g2}"


# ====================================================================== (b) reversible-sampler DB
def test_b_detailed_balance_on_fixture_structure():
    """(b) Re-certify detailed balance (reuse selfadjoint_cert) on the fixture's negative-phase
    4-superblock structure {upper_hidden, lower_hidden, image_output, label_output} + clamped b_t:
    the symmetrized kernel K = ½(P_AB+P_BA) is π-reversible (max_asym < 1e-10) AND π-stationary, while
    the deterministic scan is genuinely non-reversible (the cert's discriminator)."""
    # sizes (2,1,1,1) mirror the fixture's superblock COUNT (4 blocks) with the upper-hidden being the
    # largest; the cert exact-shadow keeps the structure tiny+enumerable (the DB residual is structural,
    # independent of exact sizes — robustness across sizes is the Task-2 suite). Use the production
    # forward [0,1,2,3] / reverse [3,2,1,0] order.
    res = sc.certify(np.random.default_rng(0), sizes=(1, 1, 1, 1), verbose=False)
    assert res["n_superblocks"] == 4
    assert res["superblock_names"] == list(SUPERBLOCK_NAMES)
    assert res["fwd_order"] == [0, 1, 2, 3] and res["rev_order"] == [3, 2, 1, 0]
    assert res["passed"] is True
    assert res["max_asym"] < sc.TOL_SYM, (
        f"DB residual max_asym={res['max_asym']:.3e} NOT < {sc.TOL_SYM:.0e} — reversible kernel REJECTED")
    assert res["K_inv_residual"] < sc.TOL_INV, "K not π-stationary"
    assert res["P_fwd_db_residual"] > sc.MIN_NONREV, (
        "deterministic scan must be demonstrably non-reversible (the cert's discriminator)")
    print(f"\n[DB-CERT] fixture 4-superblock K=½(P_AB+P_BA): max_asym={res['max_asym']:.3e} "
          f"< {sc.TOL_SYM:.0e}  PASS  (deterministic P_fwd residual {res['P_fwd_db_residual']:.3e} "
          f">> 0: discriminator live)")


# ====================================================================== (c) estimator vs EXACT
def _build_exact_cell(seed=0, sizes=(2, 4, 2, 2), n_clamp=4, beta=0.9):
    """Build the tiny (N=10) 5-superblock-isomorphic enumerable cell (exp1-style EXACT machinery,
    reusing selfadjoint_cert), returning everything check (c) needs: the exact Boltzmann π, the
    reversible kernel K = ½(P_fwd+P_rev), and the gradient-observable maps (edge products + node spins).
    """
    rng = np.random.default_rng(seed)
    blocks, J, h, coupling, s_clamp, beta = sc.make_dtm_negative_cell(rng, sizes=sizes,
                                                                      n_clamp=n_clamp, beta=beta)
    N = sum(len(b) for b in blocks)
    S = sc.spin_table(N)                                       # (2^N, N) in {−1,+1}
    pi = sc.boltzmann_clamped(S, J, h, coupling, s_clamp, beta)
    block_mats = [sc.block_gibbs_matrix(pi, S, b) for b in blocks]
    fwd = list(range(len(blocks)))
    P_fwd = sc.ordered_product(block_mats, fwd)
    P_rev = sc.ordered_product(block_mats, list(reversed(fwd)))
    K = 0.5 * (P_fwd + P_rev)                                  # the reversible kernel (DB-certified math)

    # gradient observables f_a = {edge products s_e0·s_e1, node spins s_n} (exp4 ordering: edges then bias)
    iu, ju = np.where(np.triu(J, 1) != 0)
    edge0, edge1 = iu.astype(np.int32), ju.astype(np.int32)
    n_edge = len(edge0)
    obs_maps = dict(n_edge=n_edge, n_bias=N,
                    edge_pos0=edge0, edge_pos1=edge1, bias_pos=np.arange(N, dtype=np.int32))
    return dict(N=N, S=S, pi=pi, K=K, edge0=edge0, edge1=edge1, n_edge=n_edge,
                obs_maps=obs_maps, P_fwd=P_fwd)


def _exact_var_and_TO(cell):
    """EXACT Var_π[f_a] and EXACT T_O (½ Σ_a S_a, S_a = 2·τ_a·Var_a) under the reversible kernel K, by
    full 2^N enumeration (exp1 exact Q_op). τ_a is the half-Sokal integrated autocorrelation computed
    from the EXACT ρ(ℓ) = ⟨f, K^ℓ f⟩_π/⟨f,f⟩_π via K's π-symmetrized eigendecomposition — then the SAME
    `_tau_half_from_rho` truncation rule the estimator uses is applied to the exact ρ (matched
    definition, so any residual gap is the estimator's FINITE-SAMPLE / FFT behaviour, not a def mismatch)."""
    S, pi, K = cell["S"], cell["pi"], cell["K"]
    Sf = S.astype(np.float64)
    F = np.concatenate([Sf[:, cell["edge0"]] * Sf[:, cell["edge1"]], Sf], axis=1)  # (2^N, P)
    mu = pi @ F
    Fc = F - mu[None, :]
    var_exact = pi @ (Fc ** 2)                                # (P,) exact Var_π

    # π-symmetrize K (it is π-reversible to ~1e-18) → real symmetric → eigendecompose.
    D = np.sqrt(pi)
    Kp = (D[:, None] * K) / D[None, :]
    Ks = 0.5 * (Kp + Kp.T)
    evals, evecs = np.linalg.eigh(Ks)
    # autocov(ℓ)_a = Σ_k λ_k^ℓ c_{k,a}², with c = evecs^T (√π · centered f).
    W = D[:, None] * Fc
    C = evecs.T @ W
    c2 = C ** 2
    ac0 = c2.sum(axis=0)                                       # == var_exact
    Lmax = 2000                                                # |λ|<1 ⇒ λ^Lmax underflows: τ converged
    acov = np.vstack([evals ** lag for lag in range(Lmax)]) @ c2  # (Lmax, P)
    rho_exact = np.zeros_like(acov)
    nz = ac0 > 0
    rho_exact[:, nz] = acov[:, nz] / ac0[nz][None, :]
    tau_exact = pp._tau_half_from_rho(rho_exact)              # SAME truncation rule as the estimator
    S_a_exact = 2.0 * tau_exact * var_exact
    T_O_exact = 0.5 * float(S_a_exact.sum())
    return var_exact, T_O_exact, float(tau_exact.max())


def _sample_kernel_trajectory(cell, n_chains, L, seed, burn=80):
    """Sample a CPU trajectory of the FREE spins from the reversible kernel K (inverse-CDF transition
    sampling on the enumerable state space — exactly the marginal kernel the overlay realizes). Returns
    the gradient-observable trajectory (n_chains, L, P) = [edge products, node spins], the input the
    REAL estimator `sokal_profile_from_spins` consumes."""
    S, K, N = cell["S"], cell["K"], cell["N"]
    r = np.random.default_rng(seed)
    cdf = np.cumsum(K, axis=1)
    states = r.integers(0, 2 ** N, size=n_chains)
    for _ in range(burn):
        u = r.random(n_chains)
        states = (cdf[states] < u[:, None]).sum(axis=1)
    spins = np.empty((n_chains, L, N), dtype=np.float32)
    for t in range(L):
        u = r.random(n_chains)
        states = (cdf[states] < u[:, None]).sum(axis=1)
        spins[:, t, :] = S[states]
    return np.concatenate([spins[:, :, cell["edge0"]] * spins[:, :, cell["edge1"]], spins], axis=-1)


def test_c_real_energy_free_enumerates_on_fixture():
    """(c-prereq) The REAL 4_4 DTM energy path enumerates: `build_maps` + the THREE-term `energy_free`
    give a valid Boltzmann law over the full 2^16 free-state space (exact π sums to 1, normalised). This
    exercises the REAL energy primitive that the production probe's Q_op rests on."""
    _dtm, step = _build_fixture_step()
    maps = pp.build_maps(step)
    N = maps["n_free"]
    assert N == EXPECTED_N_TOTAL and 2 ** N == 65536
    S = sc.spin_table(N)                                      # (65536, 16) — fast (~0.1 s)
    # a HARD b_t clamp (±1), broadcast across states (the negative-phase clamp is fixed).
    rng = np.random.default_rng(0)
    clamp = (rng.integers(0, 2, size=maps["n_clamp"]) * 2 - 1).astype(np.float64)
    clamp2d = np.broadcast_to(clamp, (S.shape[0], maps["n_clamp"]))
    beta = float(step.training_spec.beta)
    E = pp.energy_free(S.astype(np.float64), clamp2d, maps)   # REAL three-term conditional energy
    w = np.exp(-beta * (E - E.min()))
    pi = w / w.sum()
    assert np.isclose(pi.sum(), 1.0)
    assert np.all(np.isfinite(pi)) and pi.max() < 1.0, "degenerate exact π over the real 2^16 space"
    print(f"\n[EXACT-π real 4_4] N={N} 2^N={2**N}: real energy_free enumerated; max π={pi.max():.4f}")


def test_c_estimator_var_matches_exact():
    """(c-1) UNBIASED gate: the estimator's sample Var_π[f_a] matches the EXACT enumerated Var_π within
    a tight band (median rel-err < 5%, max < 10%). Var is the unbiased half of T_O / Q_op."""
    cell = _build_exact_cell()
    var_exact, _T_O_exact, _ = _exact_var_and_TO(cell)
    obs = _sample_kernel_trajectory(cell, n_chains=512, L=4000, seed=11)
    var_est = obs.reshape(-1, obs.shape[-1]).var(axis=0)
    relerr = np.abs(var_est - var_exact) / np.abs(var_exact)
    med, mx = float(np.median(relerr)), float(np.max(relerr))
    assert med < VAR_RELERR_MEDIAN_TOL, f"Var_π median rel-err {med:.4f} ≥ {VAR_RELERR_MEDIAN_TOL}"
    assert mx < VAR_RELERR_MAX_TOL, f"Var_π max rel-err {mx:.4f} ≥ {VAR_RELERR_MAX_TOL}"
    print(f"\n[ESTIMATOR-vs-EXACT Var_π] median rel-err={med:.4f} max rel-err={mx:.4f}  "
          f"(< {VAR_RELERR_MEDIAN_TOL}/{VAR_RELERR_MAX_TOL})  PASS")


def test_c_estimator_TO_matches_exact_within_bias_band():
    """(c-2) The core estimator gate: the REAL `sokal_profile_from_spins` T_O on reversible-kernel
    trajectories AGREES with the EXACT enumerated T_O within the verbatim-estimator's bias band, AND is
    stable across seeds (cross-seed CV ≪ tolerance) — proving the residual gap is the estimator's
    SYSTEMATIC FFT half-Sokal bias (est ≲ exact), not Monte-Carlo SE. Also reports Q_op agreement
    (Q_op = ‖g‖²/T_O, same ‖g‖ ⇒ Q_op ratio = T_O_exact/T_O_est)."""
    cell = _build_exact_cell()
    _var_exact, T_O_exact, tau_max_exact = _exact_var_and_TO(cell)

    TOs = []
    for seed in (11, 22, 33, 44, 55):
        obs = _sample_kernel_trajectory(cell, n_chains=512, L=4000, seed=seed)
        _tau_max_est, T_O_est, _S_a = pp.sokal_profile_from_spins(obs, cell["obs_maps"])
        TOs.append(T_O_est)
    TOs = np.asarray(TOs)
    mean_est = float(TOs.mean())
    cv = float(TOs.std() / TOs.mean())
    ratio = mean_est / T_O_exact

    # (i) cross-seed CV small ⇒ the gap is BIAS not SE.
    assert cv < TO_CROSS_SEED_CV_TOL, (
        f"cross-seed CV {cv:.4f} ≥ {TO_CROSS_SEED_CV_TOL} — T_O_est is not stable; the comparison is "
        "SE-dominated (need more chains/length)")
    # (ii) the (stable) estimator agrees with exact within the FFT half-Sokal bias band.
    assert TO_RATIO_LO <= ratio <= TO_RATIO_HI, (
        f"T_O_est/T_O_exact = {ratio:.4f} outside the bias band [{TO_RATIO_LO},{TO_RATIO_HI}]; "
        f"T_O_est={mean_est:.4f} T_O_exact={T_O_exact:.4f}")

    # Q_op consistency (same fixed ‖g‖ on both sides → the ratio is exactly the inverse T_O ratio).
    g = np.ones(cell["obs_maps"]["n_edge"] + cell["obs_maps"]["n_bias"])  # fixed nonzero gradient
    K = pp.K_WINDOW
    q_op_exact = (K / 2.0) * np.linalg.norm(g) ** 2 / T_O_exact
    q_op_est = (K / 2.0) * np.linalg.norm(g) ** 2 / mean_est
    q_ratio = q_op_est / q_op_exact
    assert np.isclose(q_ratio, T_O_exact / mean_est), "Q_op ratio must be the inverse T_O ratio"
    print(f"\n[ESTIMATOR-vs-EXACT T_O] exact T_O={T_O_exact:.4f} (τ_max={tau_max_exact:.3f}); "
          f"est T_O mean={mean_est:.4f} std={TOs.std():.4f} cross-seed CV={cv:.4f}; "
          f"ratio est/exact={ratio:.4f} ∈ [{TO_RATIO_LO},{TO_RATIO_HI}]  PASS\n"
          f"[ESTIMATOR-vs-EXACT Q_op] Q_op_exact={q_op_exact:.4f} Q_op_est={q_op_est:.4f} "
          f"ratio={q_ratio:.4f} (= T_O_exact/T_O_est, the inverse FFT bias)")


def test_c_rademacher_screen_brackets_full_profile():
    """(c-3) The Rademacher worst-of-N_R screening estimator (the actual `probe_scalars` path) is a
    higher-variance single-projection screen; verify it is CONSISTENT with the exact T_O — its per-sketch
    T_{O,Y} values should BRACKET the exact T_O (some above, some below) rather than collapse, confirming
    the screen is not pathologically biased away from the truth."""
    cell = _build_exact_cell()
    _v, T_O_exact, _t = _exact_var_and_TO(cell)
    obs = _sample_kernel_trajectory(cell, n_chains=512, L=4000, seed=7)
    sk = pp.rademacher_sketch_scalars(obs, n_R=24, diag_key=7)
    per = np.asarray(sk["per_sketch_T_O"])
    assert len(per) == 24
    assert per.min() < T_O_exact < per.max(), (
        f"Rademacher per-sketch T_O range [{per.min():.3f},{per.max():.3f}] does not bracket exact "
        f"T_O={T_O_exact:.3f} — the screening projection is pathologically biased")
    print(f"\n[RADEMACHER screen] per-sketch T_O range [{per.min():.3f},{per.max():.3f}] brackets "
          f"exact T_O={T_O_exact:.3f} (median sketch {np.median(per):.3f})  PASS")


# ====================================================================== (e) checkpoint rollback
def test_e_checkpoint_rollback_reproducibility():
    """(e) Perturb the fixture DTM weights (eqx.tree_at write-back; NO dtm.train), inject a non-trivial
    autocorrelations dict + opt count, save via the DTM.save_epoch eqx-partition, MUTATE, then RESTORE
    via the DTM.load eqx-deserialise path; assert bitwise reproducibility through the probe_primitives
    rollback helpers (`_weights_hash`, `_key_list`, `_find_counts`) AND the autocorrelations dict.
    This is the rollback the Task-8 driver fork/restore depends on."""
    import equinox as eqx
    import jax

    dtm, step_ckpt = _build_fixture_step()                    # perturbed (trained-like) checkpoint state

    # Inject non-default provenance state into the checkpoint so the restore carries REAL (non-empty)
    # state — autocorrelations dict (NOT in the save-mask: reconstructed from the like-step template,
    # exactly as DTM.load does) + a bumped opt count (IS serialised).
    autocorr_payload = {"epoch_0": np.asarray([0.42, 0.17, 0.05], dtype=np.float64)}
    step_ckpt = eqx.tree_at(lambda s: s.autocorrelations, step_ckpt, autocorr_payload,
                            is_leaf=lambda x: isinstance(x, dict))
    hash_ckpt = pp._weights_hash(step_ckpt)
    counts_ckpt = pp._find_counts(step_ckpt.opt_state)
    key_ckpt = pp._key_list(dtm)

    # --- SAVE: the exact DTM.save_epoch partition (weights, biases, opt_state) + eqx serialise. ------
    save_mask = jax.tree_util.tree_map(lambda _: False, step_ckpt)
    save_mask = eqx.tree_at(lambda s: (s.model.weights, s.model.biases, s.opt_state),
                            save_mask, (True, True, True))
    params, _static = eqx.partition(step_ckpt, save_mask)
    fp = os.path.join(tempfile.mkdtemp(prefix="htdml_roll_"), "step_00.eqx")
    eqx.tree_serialise_leaves(fp, params)

    # --- MUTATE in place (simulate further training: weights AND opt state change). ------------------
    w_mut = step_ckpt.model.weights + 1.0
    b_mut = step_ckpt.model.biases + 1.0
    step_mut = eqx.tree_at(lambda s: (s.model.weights, s.model.biases), step_ckpt, (w_mut, b_mut))
    assert pp._weights_hash(step_mut) != hash_ckpt, "mutation did not change the weights hash"

    # --- RESTORE: the DTM.load deserialise-into-like-step path (the saved leaves win). ---------------
    restored_params = eqx.tree_deserialise_leaves(fp, params)
    step_restored = eqx.combine(restored_params, eqx.filter(step_ckpt, save_mask, inverse=True))

    # weights restored bitwise.
    assert pp._weights_hash(step_restored) == hash_ckpt, (
        f"weights hash mismatch after rollback: {pp._weights_hash(step_restored)} != {hash_ckpt}")
    # optax/step counts restored.
    assert pp._find_counts(step_restored.opt_state) == counts_ckpt, "opt-state counts not restored"
    # RNG keys (dtm-level) unchanged across the round-trip.
    assert pp._key_list(dtm) == key_ckpt, "dtm RNG key drifted across rollback"
    # the autocorrelations dict (reconstructed from the like-step template) reproduces.
    assert set(step_restored.autocorrelations.keys()) == set(autocorr_payload.keys()), (
        "autocorrelations dict keys not restored")
    np.testing.assert_array_equal(step_restored.autocorrelations["epoch_0"],
                                  autocorr_payload["epoch_0"])
    print(f"\n[ROLLBACK] weights-hash {hash_ckpt} restored bitwise; opt counts {counts_ckpt} restored; "
          f"dtm key {key_ckpt} stable; autocorrelations dict reproduced  PASS")
