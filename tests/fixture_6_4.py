"""Task 4 — the enumerable correctness ARBITER fixture (MECHANISM checks a, b, c, e).

This module builds a small DTM model isomorphic to the production superblock structure and
validates the build's MECHANISMS by EXACT enumeration. It is the arbiter the rest of the companion
trusts: every check below HARD-FAILS (pytest assertion) on violation.

  (a) hard-bit forward noising + b_t detachment      — the REAL forward noiser `get_perturbed_data`
      (step.py:454, used at step.py:203-204) draws HARD {0,1} bits → hard {−1,+1} spins, and NO
      gradient reaches b0 / the encoder through b_t (b_t = `stop_gradient(forward_noise(b0))`; the
      discrete draw is non-differentiable too — see the scope note on the gradient sub-test).
  (b) reversible-sampler detailed balance               — re-certify DB (reuse `selfadjoint_cert`) on
      the fixture's 4-superblock negative-phase structure (max_asym < 1e-10).
  (c) Rademacher / Sokal trainability estimator vs EXACT — full 2^N enumeration of the Boltzmann law
      gives the EXACT Var_π[f_a] and the EXACT T_O via the reversible kernel's enumerable transition
      matrix (exp1-style); the real `sokal_profile_from_spins` estimator on CPU trajectories from that
      same kernel must AGREE within a stated, SE-aware tolerance.
  (e) checkpoint rollback via the REAL DTM.load          — perturb (eqx.tree_at write-back, NO
      dtm.train), save (`DTM.save_epoch`), mutate, restore (`DTM.load`); weights (`_weights_hash`) /
      opt counts (`_find_counts`) / RNG keys (`_key_list`) restored bitwise, while `autocorrelations`
      come back EMPTY (DTM.load drops the unsaved static) → recovered by the OUT-OF-BAND re-inject the
      Task-8 driver performs.
  Plus a MANDATORY `refreshed_weight_proof` gate on the fixture DTM (the stale-factors guard).

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
    enumerable cell that is GENUINELY STRICTLY-BIPARTITE-ISOMORPHIC to the production 5-superblock
    layout (base edges ONLY upper_hidden↔lower_hidden, lower_hidden↔image_output, lower_hidden↔
    label_output — NO upper↔output, NO intra-superblock, NO output↔output edges; verified against the
    REAL graph in test_fixture_model_is_production_shape and asserted on the cell by
    `_assert_strictly_bipartite_cell`). It is built by a FIXTURE-LOCAL `_make_bipartite_cell` (NOT the
    Task-2 `selfadjoint_cert.make_dtm_negative_cell`, which wires extra upper↔output edges that break
    isomorphism — those do not affect the DB-cert, which we leave UNCHANGED), reusing the cert's
    exp1-style enumeration helpers (`spin_table` / `boltzmann_clamped` / `block_gibbs_matrix` /
    `ordered_product` → ½(P_fwd+P_rev)) and the REAL estimator `sokal_profile_from_spins`. The REAL 4_4
    model's `energy_free` exact-π enumeration is ALSO exercised in (c) to confirm the real energy path
    enumerates. (Brief: "If NO real preset is small enough for full 2^N, hand-build a tiny Ising
    structure isomorphic to the 5-superblock layout … but PREFER a real DTM step.")

conftest.py installs the vendored isolation; the module self-bootstraps on import too.
"""

from __future__ import annotations

import contextlib
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

# Stage-C compat free-energy checks (d-i, d-ii). The d-i marginalization-EXACTNESS gate (<1e-10) needs
# float64, so the compat numerics run inside a SCOPED `_x64()` toggle (below). We do NOT enable x64 for
# the whole module: the Task-4 DTM.load round-trip saved its checkpoint in float32, and a global x64
# flip makes the fresh `like`-step float64 → an eqx deserialise dtype-mismatch. So the toggle is local.


@contextlib.contextmanager
def _x64():
    """Scoped JAX float64 — REQUIRED for the d-i 1e-10 marginalization gate (float32 cannot resolve it).
    Restores the prior flag on exit so the rest of the module (and the float32 DTM.load round-trip) is
    unaffected. The real-DTM constructor emits a benign vendored f64→f32 scatter FutureWarning under x64
    (thrml step.py:165 `_set_coupling_weights`) — not from our code; pytest does not escalate it."""
    import jax

    prev = jax.config.jax_enable_x64
    jax.config.update("jax_enable_x64", True)
    try:
        yield
    finally:
        jax.config.update("jax_enable_x64", prev)


# Compat-check tolerances (JUSTIFIED in task-5-report.md):
#   d-i: the analytic log-2cosh lower-marginalization is ALGEBRAICALLY exact for the strict-bipartite
#        real structure; residual is pure float64 round-off (≈ 1e-15) → a tight 1e-10 SAFETY-NET gate.
DI_MARGINALIZATION_TOL = 1e-10        # the load-bearing exactness gate (BLOCKER if missed)
DII_BOUND_TOL = 1e-9                  # F_MF ≥ F_exact − tol (numerical slack on the variational bound)
DII_FD_GRAD_TOL = 1e-6                # autodiff vs central-FD on ∂F_MF/∂(latent clamp)
DII_GRAD_DISTINCT_MIN = 1e-3          # min ‖∂F_compat − ∂E_clamped‖∞ (MF/marginalization did NOT collapse)


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

# The production-shape invariants the fixture must satisfy (the training-negative free set).
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
    faithful, GPU-free stale-factors-bug reproduction). Does NOT call `dtm.train` (CPU constraint)."""
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
    Faithful, GPU-free reproduction of the trained state that triggers the stale-factors bug."""
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

    # PoissonBinomial manager invariant (MUST assert base_graph_manager ...).
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

    # --- STRICT-BIPARTITE premise (load-bearing for Stage-C marginalization, Task 5) -----------------
    # Classify EVERY real base_graph_edge by superblock. Production must be: upper half ↔ lower half
    # ONLY; NO intra-half edges; upper_hidden couples ONLY to lower_hidden (NO upper_hidden↔output);
    # NO output↔output. This is the premise the closed-form lower_hidden marginalization rests on.
    g = step.model.graph
    nm = g.node_mapping
    uh = set(nm[n] for n in free_blocks[0])
    lh = set(nm[n] for n in free_blocks[1])
    io = set(nm[n] for n in free_blocks[2])
    lo = set(nm[n] for n in free_blocks[3])
    upper, lower = uh | io | lo, lh

    def _cls(x):
        return "uh" if x in uh else ("out" if (x in io or x in lo) else "lh")

    n_uh_lh = n_out_lh = 0
    for e in g.base_graph_edges:
        a, b = nm[e.connected_nodes[0]], nm[e.connected_nodes[1]]
        A = "upper" if a in upper else "lower"
        B = "upper" if b in upper else "lower"
        assert A != B, f"intra-half base edge {a}-{b} — production must be strictly bipartite"
        ca, cb = _cls(a), _cls(b)
        pair = tuple(sorted((ca, cb)))
        assert pair != ("out", "uh"), (
            f"upper_hidden↔output base edge {a}-{b} — upper_hidden must couple ONLY to lower_hidden")
        assert pair != ("out", "out"), f"output↔output base edge {a}-{b} — outputs couple only to lower"
        if pair == ("lh", "uh"):
            n_uh_lh += 1
        elif "lh" in pair and "out" in pair:
            n_out_lh += 1
    assert n_uh_lh > 0 and n_out_lh > 0, (
        f"degenerate coupling (uh↔lh={n_uh_lh}, out↔lh={n_out_lh}) — both must be present")

    print(f"\n[FIXTURE] real 4_4 DTM: superblocks {list(SUPERBLOCK_NAMES)} sizes {block_lens} "
          f"N_total={n_total} (2^N={2**n_total}); clamp b_t={n_clamp}; coupling={maps['n_coupling']}; "
          f"base_edges={maps['n_edge']}; manager=PoissonBinomial\n"
          f"[FIXTURE] STRICT-BIPARTITE verified: upper↔lower only; uh↔lh={n_uh_lh}, out↔lh={n_out_lh}; "
          f"NO upper_hidden↔output, NO output↔output, NO intra-half edges (Stage-C premise holds)")


# ====================================================================== MANDATORY refresh-proof gate
def test_mandatory_refreshed_weight_proof_on_fixture():
    """MANDATORY (brief: "must pass for the fixture DTM before any probe. Call it."): the
    stale-factors guard. The fixture's perturbation (eqx.tree_at write-back leaving `model.factors`
    stale, the faithful stale-factors-bug repro) sets up the stale substrate; a freshly-built
    AnnealingIsingSamplingProgram reads those stale INIT factors, and the refresh re-injects the trained
    weights. `refreshed_weight_proof` must return constructor_was_stale=True AND refresh_ok=True
    (refreshed_vs_trained_maxabs ≈ 0). This is the whole reason the companion exists — every probe and
    every L_compat build must clear it."""
    _dtm, step = _build_fixture_step()
    proof = pp.refreshed_weight_proof(step)
    assert set(["refresh_ok", "constructor_was_stale", "refreshed_vs_trained_maxabs",
                "stale_vs_trained_maxabs"]).issubset(proof)
    assert proof["constructor_was_stale"] is True, (
        f"constructor was NOT stale (stale_vs_trained_maxabs={proof['stale_vs_trained_maxabs']}) — the "
        "stale-factors bug premise does not hold on the fixture; the guard would be vacuous")
    assert proof["refresh_ok"] is True, (
        f"refresh did NOT take (refreshed_vs_trained_maxabs={proof['refreshed_vs_trained_maxabs']}) — "
        "the mandatory trained-weight refresh is broken")
    assert proof["refreshed_vs_trained_maxabs"] < 1e-6
    assert proof["stale_vs_trained_maxabs"] > 1e-6
    print(f"\n[REFRESH-PROOF] fixture DTM: constructor_was_stale=True "
          f"(stale_vs_trained_maxabs={proof['stale_vs_trained_maxabs']:.4f}); refresh_ok=True "
          f"(refreshed_vs_trained_maxabs={proof['refreshed_vs_trained_maxabs']:.2e})  PASS")


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


def test_a_no_gradient_reaches_b0_through_bt():
    """(a-ii) NO gradient reaches b0 / the encoder through b_t = stop_gradient(forward_noise(b0)):
    jax.grad of any function of b_t w.r.t. b0 is EXACTLY zero (only b0 carries ∂L/∂latent).

    SCOPE NOTE (honest labelling): the REAL forward noiser is a discrete bernoulli draw and is therefore
    already non-differentiable, so jax.grad is zero WITH OR WITHOUT the explicit `stop_gradient` — this
    test validates the *property the build needs* ("no gradient reaches b0 through b_t"), but it CANNOT
    distinguish stop_gradient-present from stop_gradient-absent (both give exactly zero). The companion
    relies on BOTH facts (the build wraps b_t in `stop_gradient` as a declarative guard; the draw is
    structurally non-differentiable anyway). We assert both branches to document this, NOT to claim the
    test exercises stop_gradient specifically."""
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
    assert np.all(g == 0.0), f"gradient reached b0 through b_t (must be all-zero): {g}"

    # WITHOUT stop_gradient the discrete bernoulli draw is STILL non-differentiable (also exactly zero):
    # this is why the test can't discriminate present/absent — the detachment is structural.
    def loss_no_sg(b0):
        bt = get_perturbed_data(key, b0, dt=0.5, rates=0.8, bin_trials=1)
        return jnp.sum(bt ** 2 + 3.0 * bt)

    g2 = np.asarray(jax.grad(loss_no_sg)(b0))
    assert np.all(g2 == 0.0), (
        "discrete draw unexpectedly differentiable — the 'no gradient reaches b0' property would then "
        f"rely solely on stop_gradient: {g2}")


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
def _make_bipartite_cell(rng, sizes=(2, 4, 2, 2), n_clamp=4, beta=0.9):
    """Build a STRICTLY-BIPARTITE 5-superblock cell that is genuinely ISOMORPHIC to the production
    DTM negative-phase graph (verified in test_fixture_model_is_production_shape): base edges are
    ONLY upper_hidden↔lower_hidden, lower_hidden↔image_output, lower_hidden↔label_output — i.e. the
    upper half {upper_hidden, image_output, label_output} couples ONLY to the lower half
    {lower_hidden}; there are NO upper↔output edges, NO intra-superblock edges, NO output↔output
    edges. b_t (the clamp) couples 1-to-1 to the OUTPUT sites with fixed forward-diffusion weights.

    NOTE: this is a corrected, FIXTURE-LOCAL builder. The Task-2 `selfadjoint_cert.make_dtm_negative_cell`
    additionally wires upper↔output edges (selfadjoint_cert.py:152-154) that do NOT exist in production
    — those extra edges do not affect the DB certificate (self-adjointness holds for ANY edge set), but
    they break structural isomorphism, so we do NOT reuse it for the structurally-faithful exact-T_O
    comparison. We DO reuse its enumeration helpers (spin_table / boltzmann_clamped / block_gibbs_matrix
    / ordered_product / max_asym) — the exp1-style EXACT machinery. We do NOT change the DB-cert.

    Returns (blocks, J, h, coupling, s_clamp, beta) matching the selfadjoint_cert signature."""
    blocks = []
    idx = 0
    for sz in sizes:
        blocks.append(list(range(idx, idx + sz)))
        idx += sz
    n = idx
    upper, lower, img_out, lab_out = blocks
    J = np.zeros((n, n))

    def add_edge(a, b, w):
        J[a, b] = J[b, a] = w

    pairs = []
    for u in upper:                       # upper_hidden ↔ lower_hidden ONLY
        for l in lower:
            pairs.append((u, l))
    for l in lower:                       # lower_hidden ↔ {image_output, label_output} ONLY
        for o in img_out + lab_out:
            pairs.append((l, o))
    for a, b in pairs:
        add_edge(a, b, float(rng.normal(0.0, 0.7)))

    h = rng.normal(0.0, 0.5, size=n)
    out_sites = img_out + lab_out         # b_t couples 1-to-1 to the OUTPUT sites (fixed diffusion wts)
    cf = np.array(out_sites, dtype=np.int64)
    cc = rng.integers(0, n_clamp, size=len(out_sites)).astype(np.int64)
    cw = rng.normal(0.0, 0.6, size=len(out_sites))
    s_clamp = rng.choice([-1.0, 1.0], size=n_clamp).astype(float)
    return blocks, J, h, (cf, cc, cw), s_clamp, beta


def _assert_strictly_bipartite_cell(blocks, J):
    """Assert the cell's base graph is strictly bipartite and matches production: no intra-half edges,
    no upper_hidden↔output edges, no output↔output edges (upper half ↔ lower half only)."""
    uh, lh, io, lo = (set(b) for b in blocks)
    upper, lower = uh | io | lo, lh
    n = J.shape[0]
    for a in range(n):
        for b in range(a + 1, n):
            if J[a, b] == 0:
                continue
            A = "upper" if a in upper else "lower"
            B = "upper" if b in upper else "lower"
            assert A != B, f"intra-half edge {a}-{b} (cell not bipartite)"
            uh_out = (a in uh and (b in io or b in lo)) or (b in uh and (a in io or a in lo))
            assert not uh_out, f"upper_hidden↔output edge {a}-{b} (not production-isomorphic)"
            out_out = (a in io or a in lo) and (b in io or b in lo)
            assert not out_out, f"output↔output edge {a}-{b} (not production-isomorphic)"


def _build_exact_cell(seed=0, sizes=(2, 4, 2, 2), n_clamp=4, beta=0.9):
    """Build the tiny (N=10) STRICTLY-BIPARTITE 5-superblock-ISOMORPHIC enumerable cell (exp1-style
    EXACT machinery, reusing selfadjoint_cert's enumeration helpers on the corrected bipartite graph),
    returning everything check (c) needs: the exact Boltzmann π, the reversible kernel K = ½(P_fwd+P_rev),
    and the gradient-observable maps (edge products + node spins)."""
    rng = np.random.default_rng(seed)
    blocks, J, h, coupling, s_clamp, beta = _make_bipartite_cell(rng, sizes=sizes,
                                                                 n_clamp=n_clamp, beta=beta)
    _assert_strictly_bipartite_cell(blocks, J)                # the cell IS production-isomorphic
    N = sum(len(b) for b in blocks)
    S = sc.spin_table(N)                                       # (2^N, N) in {−1,+1}
    pi = sc.boltzmann_clamped(S, J, h, coupling, s_clamp, beta)
    block_mats = [sc.block_gibbs_matrix(pi, S, b) for b in blocks]
    fwd = list(range(len(blocks)))
    P_fwd = sc.ordered_product(block_mats, fwd)
    P_rev = sc.ordered_product(block_mats, list(reversed(fwd)))
    K = 0.5 * (P_fwd + P_rev)                                  # the reversible kernel (DB-certified math)

    # gradient observables f_a = {edge products s_e0·s_e1, node spins s_n} (ordering: edges then bias)
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
def test_e_checkpoint_rollback_real_dtm_load_round_trip():
    """(e) The REAL `DTM.save_epoch` + `DTM.load` round-trip (both run on CPU — verified; only
    `dtm.train` needs a GPU). Perturb the fixture DTM (eqx.tree_at write-back; NO dtm.train), inject a
    non-trivial autocorrelations dict, save, then load with `DTM.load`. Validates the TRUE behaviour the
    Task-8 driver must handle:

      * weights restored BITWISE (`_weights_hash`), opt counts (`_find_counts`) and RNG keys
        (`_key_list`) restored — these ARE in the save-mask (genuine);
      * the `autocorrelations` dict is NOT restored by `DTM.load` — it is UNSAVED static (not in the
        save-mask), so `DTM.load` (DTM.py:1002-1019) recombines the saved params with a FRESH like-step
        whose `autocorrelations == {}` → it comes back EMPTY (the dict is LOST on load). We assert this
        explicitly (the plan requires restoring autocorrelations OUT-OF-BAND for exactly this reason);
      * the OUT-OF-BAND restore (re-inject the saved dict, as the Task-8 driver will) recovers them."""
    import equinox as eqx
    import jax.random as jr

    from thrmlDenoising.DTM import DTM
    from thrmlDenoising.utils import make_cfg

    # Build a real DTM and perturb step 0 (trained-like; faithful stale-factors write-back, NO dtm.train).
    try:
        dtm = DTM(make_cfg(**FIXTURE_CFG))
    except Exception as e:  # pragma: no cover
        pytest.fail(f"NEEDS_CONTEXT: 4_4 fixture DTM could not be instantiated on CPU: {e}")
    step_ckpt = _perturb_step(dtm.steps[0], seed=7)

    # Inject a non-trivial autocorrelations dict (the UNSAVED static the driver must carry out-of-band).
    autocorr_payload = {"epoch_0": np.asarray([0.42, 0.17, 0.05], dtype=np.float64)}
    step_ckpt = eqx.tree_at(lambda s: s.autocorrelations, step_ckpt, autocorr_payload)
    hash_ckpt = pp._weights_hash(step_ckpt)
    counts_ckpt = pp._find_counts(step_ckpt.opt_state)
    key_ckpt = pp._key_list(dtm)

    # DTM is a plain class (NOT an eqx.Module): set fields by direct assignment.
    dtm.steps[0] = step_ckpt
    workdir = tempfile.mkdtemp(prefix="htdml_roll_")
    dtm.logging_and_saving_dir = workdir

    # --- SAVE via the REAL DTM.save_epoch (eqx partition weights/biases/opt_state + serialise). -------
    dtm.save_epoch(0)
    base = os.path.join(workdir, "model_saving")
    assert os.path.isdir(os.path.join(base, "epoch_000")), "save_epoch did not write the epoch dir"

    # --- MUTATE the live object (simulate further training drifting away from the checkpoint). --------
    w_mut = step_ckpt.model.weights + 1.0
    b_mut = step_ckpt.model.biases + 1.0
    dtm.steps[0] = eqx.tree_at(lambda s: (s.model.weights, s.model.biases), step_ckpt, (w_mut, b_mut))
    assert pp._weights_hash(dtm.steps[0]) != hash_ckpt, "mutation did not change the weights hash"

    # --- RESTORE via the REAL DTM.load (constructs a fresh DTM + deserialises into a like-step). ------
    try:
        dtm_loaded = DTM.load(base, epoch=0)
    except Exception as e:  # pragma: no cover
        pytest.fail(f"NEEDS_CONTEXT: DTM.load failed on CPU (does it require a GPU like dtm.train?): {e}")
    step_loaded = dtm_loaded.steps[0]

    # (1) weights restored BITWISE (in the save-mask).
    assert pp._weights_hash(step_loaded) == hash_ckpt, (
        f"weights hash mismatch after DTM.load: {pp._weights_hash(step_loaded)} != {hash_ckpt}")
    # (2) opt counts restored (opt_state is in the save-mask).
    assert pp._find_counts(step_loaded.opt_state) == counts_ckpt, "opt-state counts not restored by DTM.load"
    # (3) dtm-level RNG keys reconstructed identically (fresh DTM from the same config).
    assert pp._key_list(dtm_loaded) == key_ckpt, "DTM.load RNG key differs from the checkpoint DTM"

    # (4) THE REAL BEHAVIOUR: autocorrelations are NOT restored — DTM.load drops the unsaved static and
    #     the fresh like-step's autocorrelations is {} (the Task-8 bug the driver must work around).
    assert step_loaded.autocorrelations == {}, (
        f"DTM.load unexpectedly RESTORED autocorrelations ({step_loaded.autocorrelations}) — if this "
        "ever becomes true the out-of-band restore in the Task-8 driver is redundant; update the plan")

    # (5) OUT-OF-BAND restore (the Task-8 driver pattern): re-inject the saved dict → recovered.
    step_loaded.autocorrelations.update(autocorr_payload)
    assert set(step_loaded.autocorrelations.keys()) == set(autocorr_payload.keys())
    np.testing.assert_array_equal(step_loaded.autocorrelations["epoch_0"], autocorr_payload["epoch_0"])

    print(f"\n[ROLLBACK real DTM.load] weights-hash {hash_ckpt} restored bitwise; opt counts "
          f"{counts_ckpt} restored; dtm key {key_ckpt} reconstructed; autocorrelations DROPPED by "
          f"DTM.load (=={{}}) then OUT-OF-BAND restored (Task-8 driver pattern)  PASS")


# ====================================================================== (d) Stage-C compat free energy
# The compat-CORE (`src/htdml/compatibility.py`) validated against brute-force enumeration on the REAL
# 4_4 model. ALL compat numerics run inside `with _x64():` (float64) — the d-i exactness gate (<1e-10)
# cannot be resolved in float32. The compat maps are built from the POSITIVE-phase partition
# (program_positive): FREE = {upper_hidden, lower_hidden}, CLAMPED = {image_output, label_output, b_t}.

def _rng_pm1(rng, n):
    return (rng.integers(0, 2, size=n) * 2 - 1).astype(np.float64)


def test_d_i_lower_marginalization_is_exact_on_real_model():
    """(d-i) THE SAFETY NET. The analytic log-2cosh marginalization of lower_hidden — `F_low(u)` — must
    EQUAL the brute-force 2^{N_lower} enumeration `−log Σ_lower exp(−β E(u, lower, clamps))` to < 1e-10
    on the REAL 4_4 model, at multiple FIXED upper_hidden configs u and clamps. This is the gate that
    proves the marginalization formula is correct for the real structure — the WHOLE Stage-C approach
    rests on it. A miss is a BLOCKER (the formula is wrong); do NOT loosen the tolerance."""
    import jax.numpy as jnp

    import htdml.compatibility as C

    _dtm, step = _build_fixture_step()
    beta = float(step.training_spec.beta)
    with _x64():
        maps = C.build_compat_maps(step)
        jm = C._jnp_maps(maps)
        # structural sanity: the positive partition has the expected layer sizes.
        assert maps["n_upper"] == EXPECTED_BLOCK_LENS[0], "upper_hidden (MF layer) size mismatch"
        assert maps["n_lower"] == EXPECTED_BLOCK_LENS[1], "lower_hidden (marginalized layer) size mismatch"
        assert maps["n_img"] == EXPECTED_BLOCK_LENS[2] and maps["n_lab"] == EXPECTED_BLOCK_LENS[3]
        assert maps["n_clamp"] == maps["n_img"] + maps["n_lab"] + maps["n_bt"]
        assert maps["n_bt"] == EXPECTED_N_CLAMP

        rng = np.random.default_rng(0)
        residuals = []
        for _ in range(8):
            u = _rng_pm1(rng, maps["n_upper"])
            clamp = _rng_pm1(rng, maps["n_clamp"])
            f_analytic = float(C.F_low(jnp.asarray(u), jnp.asarray(clamp), jm, beta))
            f_brute = C.F_low_bruteforce(u, clamp, maps, beta)       # EXACT 2^{N_lower} enumeration
            residuals.append(abs(f_analytic - f_brute))
        max_res = float(np.max(residuals))

    assert max_res < DI_MARGINALIZATION_TOL, (
        f"BLOCKER: lower-marginalization residual {max_res:.3e} ≥ {DI_MARGINALIZATION_TOL:.0e} — the "
        "log-2cosh marginalization formula is WRONG for the real structure; Stage-C is unsafe. Do NOT "
        "loosen the tolerance.")
    print(f"\n[COMPAT d-i] lower_hidden log-2cosh marginalization EXACT on real 4_4: max |F_low_analytic "
          f"− F_low_brute| = {max_res:.3e} < {DI_MARGINALIZATION_TOL:.0e} over 8 (u, clamp) configs "
          f"(N_lower={maps['n_lower']}, brute = −log Σ_{{2^{maps['n_lower']}}} exp(−βE))  PASS")


def test_d_ii_mf_bound_gradient_and_determinism_on_real_model():
    """(d-ii) MF surrogate boundness + gradient correctness + determinism on the REAL 4_4 model:
      * F_MF ≥ F_exact over the FULL 2^N free space (variational UPPER bound); record the NON-ZERO gap
        F_MF − F_exact (the honesty metric). F_exact = −log Σ_{upper,lower} exp(−βE).
      * FD-gradient agreement: autodiff ∂F_MF/∂(image_output latent clamp) matches central finite
        differences (gradients flow through the unrolled MF).
      * dF_compat/dlatent ≠ dE_clamped/dlatent: the compat gradient DIFFERS from the raw clamped-energy
        gradient (they differ by the entropy/log-cosh term; equality ⇒ the MF/marginalization collapsed).
      * Deterministic MF: same input → bitwise-identical F_MF (draws NO PRNG key)."""
    import jax
    import jax.numpy as jnp

    import htdml.compatibility as C

    _dtm, step = _build_fixture_step()
    beta = float(step.training_spec.beta)
    rng = np.random.default_rng(7)

    with _x64():
        maps = C.build_compat_maps(step)
        jm = C._jnp_maps(maps)
        n_img = maps["n_img"]

        # --- (1) variational bound F_MF ≥ F_exact, over several clamps; report the gap. --------------
        gaps = []
        for _ in range(4):
            clamp = _rng_pm1(rng, maps["n_clamp"])
            f_mf = float(C.F_MF(jnp.asarray(clamp), jm, beta))
            f_ex = C.F_exact_full(clamp, maps, beta)                  # EXACT 2^{N_upper+N_lower}
            gaps.append(f_mf - f_ex)
            assert f_mf >= f_ex - DII_BOUND_TOL, (
                f"F_MF ({f_mf:.6f}) < F_exact ({f_ex:.6f}) − tol — the variational upper bound is "
                "VIOLATED; the mean-field surrogate is mis-derived")
        gap_min, gap_med, gap_max = float(np.min(gaps)), float(np.median(gaps)), float(np.max(gaps))
        assert gap_min > 0.0, (
            f"the F_MF−F_exact gap collapsed to 0 (min {gap_min:.2e}) — the surrogate equals the exact "
            "free energy, which the mean-field heuristic should NOT on a coupled model (check the MF)")

        # --- (2) FD-gradient agreement on the image_output (latent) clamp columns. -------------------
        clamp_base = _rng_pm1(rng, maps["n_clamp"])
        tail = jnp.asarray(clamp_base[n_img:])

        def fmf_of_latent(latent):
            return C.F_MF(jnp.concatenate([latent, tail]), jm, beta)

        latent0 = jnp.asarray(clamp_base[:n_img])
        g_ad = np.asarray(jax.grad(fmf_of_latent)(latent0))
        eps = 1e-5
        g_fd = np.empty(n_img)
        for i in range(n_img):
            lp = np.array(clamp_base[:n_img]); lp[i] += eps
            lm = np.array(clamp_base[:n_img]); lm[i] -= eps
            g_fd[i] = (float(fmf_of_latent(jnp.asarray(lp))) - float(fmf_of_latent(jnp.asarray(lm)))) / (2 * eps)
        fd_err = float(np.max(np.abs(g_ad - g_fd)))
        assert fd_err < DII_FD_GRAD_TOL, (
            f"autodiff ∂F_MF/∂latent disagrees with finite differences (max abs {fd_err:.2e} ≥ "
            f"{DII_FD_GRAD_TOL:.0e}) — gradients do NOT flow correctly through the unrolled MF")

        # --- (3) gradient distinctness: F_compat grad ≠ raw clamped-energy grad. ----------------------
        def e_clamped_of_latent(latent):       # raw clamp energy (u=0, no MF / no marginalization)
            return C.clamp_energy(jnp.zeros(maps["n_upper"]), jnp.concatenate([latent, tail]), jm, beta)

        g_eclamp = np.asarray(jax.grad(e_clamped_of_latent)(latent0))
        distinct = float(np.max(np.abs(g_ad - g_eclamp)))
        assert distinct > DII_GRAD_DISTINCT_MIN, (
            f"dF_compat/dlatent == dE_clamped/dlatent (max abs diff {distinct:.2e} ≤ "
            f"{DII_GRAD_DISTINCT_MIN:.0e}) — the MF/marginalization collapsed to the raw clamped energy "
            "(the entropy/log-cosh term is inert; this is a bug)")

        # --- (4) determinism: same input → bitwise-identical F_MF (no PRNG key). ----------------------
        clamp_det = _rng_pm1(rng, maps["n_clamp"])
        f_a = float(C.F_MF(jnp.asarray(clamp_det), jm, beta))
        f_b = float(C.F_MF(jnp.asarray(clamp_det), jm, beta))
        assert f_a == f_b, f"F_MF non-deterministic ({f_a!r} != {f_b!r}) — the compat must draw NO PRNG key"

        # --- (5) λ=0 ≡ control + multi-step L_compat sanity. ------------------------------------------
        clamp_steps = jnp.asarray(np.stack([_rng_pm1(rng, maps["n_clamp"]) for _ in range(4)]))
        l_compat = float(C.L_compat(clamp_steps, [maps], beta))
        v0, fin0 = C.compat_loss(0.0, clamp_steps, [maps], beta)
        v_half, fin_h = C.compat_loss(0.5, clamp_steps, [maps], beta)
        assert float(v0) == 0.0 and bool(fin0), "λ=0 compat term must be exactly 0.0 (the control)"
        assert bool(fin_h) and np.isclose(float(v_half), 0.5 * l_compat), "λ·L_compat must scale linearly"

    print(f"\n[COMPAT d-ii] real 4_4: F_MF ≥ F_exact (gap min/med/max = {gap_min:.4f}/{gap_med:.4f}/"
          f"{gap_max:.4f} > 0, the honesty metric); FD-grad max|AD−FD|={fd_err:.2e} < {DII_FD_GRAD_TOL:.0e}; "
          f"∂F_compat≠∂E_clamped (Δ={distinct:.4f}); F_MF deterministic (bitwise); "
          f"L_compat(4 steps)={l_compat:.4f}, λ=0 ≡ control (0.0)  PASS")
