"""Task 8 — tests for src/htdml/trainability_probe.py (TrainabilityProbe).

CPU ONLY — NO ``dtm.train`` (GPU-only; build-notes §"CPU vs GPU").  The frozen-θ negative
phase is exercised on the small REAL 4_4 fixture DTM (Task 4), perturbed via the exact
``eqx.tree_at`` write-back (trained-≠-init weights, ``model.factors`` left stale = the faithful
exp15/16 stale-factors reproduction → the refresh-proof is genuinely exercised).

Tests
-----
  TP-refresh   : evaluate() calls + HARD-HALTS on the per-layer trained-weight refresh
                 (constructor_was_stale ∧ refresh_ok asserted for the layer's step); a stub
                 that makes refresh fail makes evaluate() raise.
  TP-4layers   : evaluate_model() yields EXACTLY 4 per-layer records (diffusion steps 0..3).
  TP-keys      : the returned dict has all 7 headline keys; ESS_hat == 50/(2·τ_int,Y);
                 Q_struct_perp prefactor K/2 == 25; r_grad[1]=ρ_Y(1), r_grad[50]=ρ_Y(50).
  TP-rademacher: the reported margin is the MAX τ_int,Y over the N_R sketches (worst-of-N_R,
                 not mean) AND is reproducible given diag_key.
  TP-g-sanity  : on the tiny enumerable fixture, g is finite/non-zero AND responds to a weight
                 change (the gradient is wired to the real moments, not a constant).
  TP-kernel    : the negative-phase sampling uses the LIVE reversible kernel with order_key=None
                 (per-chain diagnostics).
  TP-calib     : the per-layer calibration API returns (tau_hat, T_O, cal_stable).
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
_TESTS = str(_REPO_ROOT / "tests")
if _TESTS not in sys.path:
    sys.path.insert(0, _TESTS)

import htdml  # noqa: E402,F401  (triggers bootstrap_paths)

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402,F401
import jax.random as jr  # noqa: E402

from htdml import trainability_probe as tpmod  # noqa: E402
from htdml.trainability_probe import TrainabilityProbe  # noqa: E402
from harness import probe_primitives as pp  # noqa: E402
import fixture_6_4 as fx  # noqa: E402

_CPU = jax.devices("cpu")[0]

# Small CPU-cheap probe sizes (L_traj ≫ K is honored at calibration; the fixture is for plumbing).
_NR = 8
_LTRAJ = 60          # ≫ K=50 (so r_grad[50] = ρ_Y(50) is defined)
_NCHAINS = 8
_DIAGKEY = 20240624


@pytest.fixture(scope="module")
def fixture_dtm():
    """The REAL 4_4 fixture DTM (Task 4), perturbed step (trained-≠-init, model.factors stale)."""
    with jax.default_device(_CPU):
        dtm, _step0 = fx._build_fixture_step()
        # perturb EVERY step (so multi-step / 4-layer paths each have a trained-≠-init step).
        dtm.steps = [fx._perturb_step(s) for s in dtm.steps]
    return dtm


@pytest.fixture(scope="module")
def probe():
    return TrainabilityProbe()


def _batch(dtm):
    """A probe batch = (image, label, idx) for the single-input clamp (exp16 phase_data_1 pattern)."""
    return dict(image=dtm.train_dataset["image"], label=dtm.train_dataset["label"], idx=0)


# ============================================================ TP-refresh: HARD-HALT refresh proof
def test_evaluate_hard_halts_on_per_layer_refresh(fixture_dtm, probe, monkeypatch):
    """The MANDATORY exp15/16 guard: evaluate() asserts the layer's step refresh took
    (constructor_was_stale ∧ refresh_ok) and HARD-HALTS otherwise.

    (a) a clean evaluate() PASSES the proof (constructor_was_stale=True, refresh_ok=True);
    (b) if the proof is stubbed to report a FAILED refresh, evaluate() raises."""
    with jax.default_device(_CPU):
        # (a) the proof is recorded and both legs hold on a real perturbed step.
        out = probe.evaluate(fixture_dtm, layer=0, batch=_batch(fixture_dtm),
                             n_R=_NR, L_traj=_LTRAJ, n_chains=_NCHAINS, diag_key=_DIAGKEY)
        proof = out["_refresh_proof"]
        assert proof["constructor_was_stale"] is True and proof["refresh_ok"] is True

        # (b) make refresh look broken → evaluate() must HARD-HALT (AssertionError).
        def _broken_proof(step):
            return dict(refresh_ok=False, constructor_was_stale=True,
                        refreshed_vs_trained_maxabs=1.0, stale_vs_trained_maxabs=1.0)

        monkeypatch.setattr(tpmod.pp, "refreshed_weight_proof", _broken_proof)
        with pytest.raises(AssertionError):
            probe.evaluate(fixture_dtm, layer=0, batch=_batch(fixture_dtm),
                           n_R=_NR, L_traj=_LTRAJ, n_chains=_NCHAINS, diag_key=_DIAGKEY)


def test_evaluate_hard_halts_when_constructor_not_stale(fixture_dtm, probe, monkeypatch):
    """The bug-PREMISE leg: if constructor_was_stale=False the guard would be vacuous → HARD-HALT."""
    with jax.default_device(_CPU):
        def _not_stale(step):
            return dict(refresh_ok=True, constructor_was_stale=False,
                        refreshed_vs_trained_maxabs=0.0, stale_vs_trained_maxabs=0.0)

        monkeypatch.setattr(tpmod.pp, "refreshed_weight_proof", _not_stale)
        with pytest.raises(AssertionError):
            probe.evaluate(fixture_dtm, layer=0, batch=_batch(fixture_dtm),
                           n_R=_NR, L_traj=_LTRAJ, n_chains=_NCHAINS, diag_key=_DIAGKEY)


def test_refresh_is_called_per_layer(fixture_dtm, probe, monkeypatch):
    """The refresh + its proof are called for THIS layer's step before any sampling (spy)."""
    with jax.default_device(_CPU):
        seen = {"refresh": 0, "proof": 0}
        orig_refresh = tpmod.pp.refresh_program_weights
        orig_proof = tpmod.pp.refreshed_weight_proof

        def _spy_refresh(prog, step):
            seen["refresh"] += 1
            return orig_refresh(prog, step)

        def _spy_proof(step):
            seen["proof"] += 1
            return orig_proof(step)

        monkeypatch.setattr(tpmod.pp, "refresh_program_weights", _spy_refresh)
        monkeypatch.setattr(tpmod.pp, "refreshed_weight_proof", _spy_proof)
        probe.evaluate(fixture_dtm, layer=0, batch=_batch(fixture_dtm),
                       n_R=_NR, L_traj=_LTRAJ, n_chains=_NCHAINS, diag_key=_DIAGKEY)
        assert seen["proof"] >= 1, "refreshed_weight_proof was never called for the layer"
        assert seen["refresh"] >= 1, "refresh_program_weights was never called for the layer"


# ============================================================ TP-4layers: exactly 4 per-layer records
def test_evaluate_model_yields_exactly_4_layers(fixture_dtm):
    """evaluate_model returns EXACTLY one record per diffusion step (0..3) — the companion's 4 layers."""
    # The fixture is a 1-step DTM; tile it to 4 steps so the structural 4-layer contract is exercised
    # without a GPU (the production DTM has num_diffusion_steps=4).  `steps` is a plain list attribute
    # (not an eqx pytree leaf), so we set it directly on a shallow-copied DTM (test_latent_dtm pattern).
    import copy

    with jax.default_device(_CPU):
        dtm4 = copy.copy(fixture_dtm)
        dtm4.steps = list(fixture_dtm.steps) * 4
        assert len(dtm4.steps) == 4
        probe = TrainabilityProbe()
        records = probe.evaluate_model(dtm4, batch=_batch(dtm4),
                                       n_R=_NR, L_traj=_LTRAJ, n_chains=_NCHAINS, diag_key=_DIAGKEY)
    assert isinstance(records, list)
    assert len(records) == 4, f"expected exactly 4 per-layer records, got {len(records)}"
    assert [r["layer"] for r in records] == [0, 1, 2, 3]
    for r in records:
        assert set(TrainabilityProbe.HEADLINE_KEYS).issubset(r)


# ============================================================ TP-keys: dict + formulas
def test_evaluate_returns_all_keys_and_formulas(fixture_dtm, probe):
    """The dict has all 7 headline keys; ESS_hat == 50/(2τ); Q prefactor K/2 == 25; r_grad lags."""
    with jax.default_device(_CPU):
        out = probe.evaluate(fixture_dtm, layer=0, batch=_batch(fixture_dtm),
                             n_R=_NR, L_traj=_LTRAJ, n_chains=_NCHAINS, diag_key=_DIAGKEY)
    for k in TrainabilityProbe.HEADLINE_KEYS:
        assert k in out, f"missing headline key {k}"

    tau = out["tau_int_Y"]
    assert np.isclose(out["ESS_hat"], 50.0 / (2.0 * tau)), "ESS_hat must be K/(2τ), K=50"

    # Q_struct_perp = (K/2)·‖g‖²/T_{O,Y}, K/2 = 25.  Recover T_{O,Y} from the exposed underscore key.
    T_O_Y = out["_T_O_Y"]
    g = out["gradient_norm"]
    if T_O_Y > 0:
        expected_Q = 25.0 * g ** 2 / T_O_Y
        assert np.isclose(out["Q_struct_perp"], expected_Q, rtol=1e-9), "Q prefactor K/2 must be 25"

    # r_grad[1] / r_grad[50] are the retained-process autocorrelations at lags 1 and 50.
    assert "r_grad[1]" in out and "r_grad[50]" in out
    assert np.isfinite(out["r_grad[1]"])
    assert out["r_grad[1]"] <= 1.0 + 1e-9
    # L_traj=60 > K=50 ⇒ r_grad[50] is defined (not NaN).
    assert np.isfinite(out["r_grad[50]"]), "r_grad[50] should be defined for L_traj > 50"


# ============================================================ TP-rademacher: worst-of-N_R + reproducible
def test_rademacher_worst_of_NR_and_reproducible(fixture_dtm, probe):
    """The reported tau_int_Y is the MAX over the N_R sketches (not the mean), and reproducible
    given diag_key (same key → identical scalars)."""
    with jax.default_device(_CPU):
        out1 = probe.evaluate(fixture_dtm, layer=0, batch=_batch(fixture_dtm),
                              n_R=_NR, L_traj=_LTRAJ, n_chains=_NCHAINS, diag_key=_DIAGKEY)
        out2 = probe.evaluate(fixture_dtm, layer=0, batch=_batch(fixture_dtm),
                              n_R=_NR, L_traj=_LTRAJ, n_chains=_NCHAINS, diag_key=_DIAGKEY)

    # reproducible (same diag_key + same probe RNG): identical headline scalars
    for k in TrainabilityProbe.HEADLINE_KEYS:
        assert np.isclose(out1[k], out2[k], rtol=1e-12, atol=0.0, equal_nan=True), (
            f"key {k} not reproducible: {out1[k]} != {out2[k]}")

    # the reported margin is the WORST (max) over N_R, not the mean.
    per_tau = out1["_per_sketch_tau"]
    assert len(per_tau) == _NR
    assert np.isclose(out1["tau_int_Y"], max(per_tau)), "reported tau_int_Y must be the worst-of-N_R (max)"
    assert out1["tau_int_Y"] >= float(np.mean(per_tau)) - 1e-12, "worst ≥ mean"
    assert out1["worst_sketch_idx"] == int(np.argmax(per_tau))


# ============================================================ TP-g-sanity: g finite, nonzero, responsive
def test_gradient_finite_nonzero_and_responds_to_weight_change(fixture_dtm, probe):
    """On the tiny enumerable fixture, g = E_data[f] − E_model[f] is finite + non-zero AND responds to a
    weight change (so the gradient is wired to the real moments, not a constant)."""
    with jax.default_device(_CPU):
        out = probe.evaluate(fixture_dtm, layer=0, batch=_batch(fixture_dtm),
                             n_R=_NR, L_traj=_LTRAJ, n_chains=_NCHAINS, diag_key=_DIAGKEY)
        g = np.asarray(out["_g"])
        assert g.size > 0 and np.all(np.isfinite(g)), "g must be finite"
        assert out["gradient_norm"] > 0.0, "‖g‖ must be non-zero on the perturbed fixture"
        assert np.isclose(out["gradient_norm"], float(np.linalg.norm(g)))

        # responds to a weight change: re-perturb step 0 with a DIFFERENT seed → different g.
        import copy

        s2 = fx._perturb_step(fixture_dtm.steps[0], scale=0.9, seed=999)
        dtm2 = copy.copy(fixture_dtm)
        dtm2.steps = [s2] + list(fixture_dtm.steps[1:])
        out2 = probe.evaluate(dtm2, layer=0, batch=_batch(dtm2),
                              n_R=_NR, L_traj=_LTRAJ, n_chains=_NCHAINS, diag_key=_DIAGKEY)
        g2 = np.asarray(out2["_g"])
    assert g.shape == g2.shape
    assert not np.allclose(g, g2, atol=1e-6), "g did NOT respond to a weight change (gradient not wired)"


def test_gradient_matches_exact_within_SE(fixture_dtm, probe):
    """Stronger g-vs-exact check on a tiny ENUMERABLE structure: the per-chain estimated negative-phase
    moment E_model[f] matches the EXACT enumerated moment (real energy_free over 2^16) within MC SE,
    for the bias (node-spin) observables.  Confirms the negative-phase estimate_moments path is faithful."""
    with jax.default_device(_CPU):
        step = fixture_dtm.steps[0]
        maps = pp.build_maps(step)
        N = maps["n_free"]
        assert 2 ** N == 65536
        # EXACT E_model[node spins] over the real 2^16 free space at a FIXED b_t clamp.
        from thrml.models.ising import estimate_moments  # noqa: F401  (path live check)
        S = _spin_table(N)
        rng = np.random.default_rng(0)
        clamp_bits = rng.integers(0, 2, size=maps["n_clamp"])
        clamp_spins = (clamp_bits * 2 - 1).astype(np.float64)
        clamp2d = np.broadcast_to(clamp_spins, (S.shape[0], maps["n_clamp"]))
        beta = float(step.training_spec.beta)
        E = pp.energy_free(S.astype(np.float64), clamp2d, maps)
        w = np.exp(-beta * (E - E.min()))
        pi = w / w.sum()
        # pi @ S gives node means in FREE-POSITION order (column j of the spin table = free position j).
        # The probe returns node-spin means in BIAS-NODE order (maps["bias_pos"]), so reorder the exact
        # means the same way before comparing — the two orderings differ by a block permutation.
        exact_node_means = (pi @ S)[maps["bias_pos"]]   # (n_bias,) exact E_model[s_n], bias-node order

        # ESTIMATED E_model[node spins] via the probe's negative-phase sampler at this SAME clamp.
        est = probe._negative_node_means_for_clamp(
            step, maps, clamp_bits, n_chains=64, K=400, B=200, stride=4, key=jr.PRNGKey(7))
    err = np.abs(est - exact_node_means)
    # MC band (plumbing-grade: 64 chains × 400 retained, finite τ; observed median ~0.006, max ~0.015).
    assert np.median(err) < 0.05, f"estimated node means off exact (median |err|={np.median(err):.3f})"
    assert np.max(err) < 0.10, f"estimated node means off exact (max |err|={np.max(err):.3f})"


def _spin_table(N):
    idx = np.arange(2 ** N, dtype=np.int64)
    bits = ((idx[:, None] >> np.arange(N)[None, :]) & 1).astype(np.float64)
    return 2.0 * bits - 1.0


# ============================================================ TP-kernel: reversible kernel + order_key
def test_negative_phase_uses_live_reversible_kernel_per_chain(fixture_dtm, probe):
    """The negative-phase sampling routes through the LIVE reversible kernel; the probe uses the
    per-chain (order_key=None) diagnostic mode."""
    from harness import reversible_scan

    live, detail = reversible_scan.is_patch_live()
    assert live, f"reversible kernel not live: {detail}"
    # the probe declares the per-chain order mode (order_key=None default in the overlay).
    assert probe.ORDER_KEY is None, "diagnostics must use the per-chain kernel (order_key=None)"


def test_negative_sampling_threads_order_key_none(fixture_dtm, probe, monkeypatch):
    """Functional: the probe's negative-phase sampler reaches sample_blocks with order_subkey=None
    (per-chain), confirming order_key=None is threaded end-to-end."""
    import thrml.block_sampling as bs

    seen = []
    orig = bs.sample_blocks

    def _spy(key, state_free, clamp_state, program, sampler_state, order_subkey=None):
        seen.append(order_subkey)
        return orig(key, state_free, clamp_state, program, sampler_state, order_subkey=order_subkey)

    monkeypatch.setattr(bs, "sample_blocks", _spy)
    with jax.default_device(_CPU):
        probe.evaluate(fixture_dtm, layer=0, batch=_batch(fixture_dtm),
                       n_R=_NR, L_traj=_LTRAJ, n_chains=4, diag_key=_DIAGKEY)
    assert len(seen) > 0, "sample_blocks never called — negative-phase sampler did not run"
    assert all(ok is None for ok in seen), (
        f"per-chain diagnostics must leave order_subkey None; got {set(map(type, seen))}")


# ============================================================ TP-calib: per-layer calibration API
def test_calibration_returns_tau_TO_calstable(fixture_dtm, probe):
    """The per-layer calibration API returns (tau_hat, T_O, cal_stable) for the driver's
    Q-CALIBRATION-FAIL gate."""
    with jax.default_device(_CPU):
        calib = probe.calibrate(fixture_dtm, layer=0, batch=_batch(fixture_dtm),
                                n_chains=8, L0=24, warm=8, n_rungs=3,
                                diag_key=_DIAGKEY, key=jr.PRNGKey(3))
    assert "tau_hat" in calib and "T_O" in calib and "cal_stable" in calib
    assert isinstance(calib["cal_stable"], bool)
    if calib["tau_hat"] is not None:
        assert calib["tau_hat"] >= 0.5
    assert "curve" in calib and len(calib["curve"]) >= 1


# ============================================================ TP-calib: Cal-STABLE classifier (exp16-faithful)
def test_cal_stable_classifier_thresholds_are_exp16_constants():
    """The Cal-STABLE thresholds are the VERBATIM exp15/exp16 constants (NOT laxer).  The small-τ
    ABSOLUTE floor is an ADDED a-priori estimator constant; the three relative thresholds are unchanged."""
    assert pp.SOKAL_C == 5.0
    assert pp.TAU_TOL == 0.15
    assert pp.STAB_TOL == 0.15
    assert pp.TAU_ABS_FLOOR == 1.0                       # small-τ absolute floor (NEW, regime-aware)
    assert pp.TAU_ABS_FLOOR / pp.TAU_TOL == 1.0 / 0.15   # derived small-τ cut ≈ 6.67


def test_cal_stable_requires_two_consecutive_stable_rungs():
    """exp16's `consec_stable >= 2`: a SINGLE stable rung (all three axes) is NOT Cal-STABLE; two
    CONSECUTIVE stable rungs ARE."""
    base = dict(tau_max=10.0, T_O=100.0, self_consistent=True, dS_l1=0.01)
    rung0 = dict(base, L=1000, dS_l1=None)               # rung 0 has no prev → not a step
    rung1 = dict(base, L=2000)                            # STABLE vs rung0 (all axes tiny)
    rung2_drift = dict(tau_max=10.0, T_O=140.0, self_consistent=True, dS_l1=0.01, L=4000)  # dT=0.29 → NOT stable
    cs1, _ann, _fa = pp.classify_calibration_stable([rung0, rung1, rung2_drift])
    assert cs1 is False, "ONE stable rung must NOT be Cal-STABLE (needs 2 consecutive)"

    rung2_stable = dict(base, L=4000)                    # STABLE vs rung1 → 2 consecutive
    cs2, ann2, _fa2 = pp.classify_calibration_stable([rung0, rung1, rung2_stable])
    assert cs2 is True, "two CONSECUTIVE stable rungs must be Cal-STABLE"
    assert [r.get("step_class") for r in ann2[1:]] == ["STABLE", "STABLE"]


def test_cal_stable_can_fail_on_drifting_TO_with_stable_Sa_shape():
    """THE failure mode the laxer (dS_l1-only) criterion missed: a chain whose AGGREGATE T_O is still
    DRIFTING (large dT) but whose L1-normalized S_a *shape* momentarily stabilizes (small dS_l1) on a
    rung must be reported cal_stable=False — the dT axis must independently veto.  exp16 = UNRESOLVED."""
    # Every rung: tau stable + self-consistent + tiny dS_l1 (shape frozen) BUT T_O grows ~40%/doubling.
    curve = [
        dict(L=1000, tau_max=10.0, T_O=100.0, self_consistent=True, dS_l1=None),
        dict(L=2000, tau_max=10.2, T_O=140.0, self_consistent=True, dS_l1=0.01),  # dT≈0.286 ≥ 0.15 → NOT stable
        dict(L=4000, tau_max=10.3, T_O=196.0, self_consistent=True, dS_l1=0.01),  # dT≈0.286 ≥ 0.15 → NOT stable
    ]
    cal_stable, annotated, failed_axis = pp.classify_calibration_stable(curve)
    assert cal_stable is False, (
        "drifting-T_O chain (large dT) with a momentarily-stable S_a shape MUST be cal_stable=False — "
        "the dT axis must independently veto (the exact failure the dS_l1-only criterion let pass)")
    assert "aggregate_T_O" in failed_axis, "the failed axis must name the drifting aggregate T_O"
    assert all(r.get("step_class") == "NOT-STABLE" for r in annotated[1:])


def test_cal_stable_can_fail_on_tau_instability():
    """The tau_max axis independently vetoes: a chain with stable T_O + S_a shape but a JUMPING τ_max
    (rel_tau ≥ TAU_TOL) is NOT Cal-STABLE."""
    curve = [
        dict(L=1000, tau_max=10.0, T_O=100.0, self_consistent=True, dS_l1=None),
        dict(L=2000, tau_max=14.0, T_O=101.0, self_consistent=True, dS_l1=0.01),  # rel_tau=0.4 ≥ 0.15
        dict(L=4000, tau_max=20.0, T_O=101.0, self_consistent=True, dS_l1=0.01),  # rel_tau≈0.43 ≥ 0.15
    ]
    cal_stable, _ann, failed_axis = pp.classify_calibration_stable(curve)
    assert cal_stable is False, "a jumping τ_max (rel_tau ≥ TAU_TOL) must veto Cal-STABLE"
    assert "tau_hat" in failed_axis


def test_cal_stable_vetoed_by_non_self_consistent_rung():
    """A rung that is NOT self-consistent (L < SOKAL_C·τ_max) can NOT be STABLE even with tiny dT/dS_l1."""
    curve = [
        dict(L=1000, tau_max=10.0, T_O=100.0, self_consistent=True, dS_l1=None),
        dict(L=2000, tau_max=10.0, T_O=101.0, self_consistent=False, dS_l1=0.01),  # not self-consistent
        dict(L=4000, tau_max=10.0, T_O=101.0, self_consistent=False, dS_l1=0.01),
    ]
    cal_stable, _ann, failed_axis = pp.classify_calibration_stable(curve)
    assert cal_stable is False
    assert "tau_hat" in failed_axis  # self_consistent is folded into the tau axis (exp16)


# ====================================================== TP-calib: Cal-STABLE small-τ regime floor (exp2)
# The 5b9cbbc paid H200 run returned Q-CALIBRATION-FAIL on both seeds at τ̂≈2 — a cal-GATE artifact, NOT
# bad mixing (chains self-consistent ~16× over L_traj≥C·τ̂).  At tiny τ̂ the PURE relative TAU_TOL/STAB_TOL
# tests are dominated by estimation noise (a 2.0→2.5 jitter = 25% > 15%) → never 2 consecutive STABLE
# rungs → false fail.  The regime-aware fix adds an ABSOLUTE floor (numpy.isclose rtol+atol style):
# TAU_ABS_FLOOR = the half-Sokal τ_int resolution floor (sub-1-sweep |Δτ| is unresolvable, an a-priori
# property NOT tuned to seed data).  self-consistency (L≥SOKAL_C·τ) + the L1-normalized SHAPE guard
# (dS_l1<STAB_TOL) are RETAINED ALWAYS so the gate stays non-vacuous.
def test_cal_stable_small_tau_noise_regime_now_passes():
    """REGIME FIX (the exp1→exp2 repair): a well-mixed small-τ̂ chain whose rung-to-rung Δτ (~0.5) is
    sub-resolution and whose L1 shape is stable MUST be cal_stable=True — even though the PURE relative
    rel_tau (~0.2–0.32) and dT (~0.17–0.19) both EXCEED 0.15.  Doubling rungs L=100/200/400/800 (the full
    cal config).  Without the floor this is the FALSE Q-CALIBRATION-FAIL of run 5b9cbbc."""
    curve = [
        dict(L=100, tau_max=1.9, T_O=5200.0, self_consistent=True, dS_l1=None),
        dict(L=200, tau_max=2.5, T_O=6300.0, self_consistent=True, dS_l1=0.04),  # rel_tau=0.316, dT=0.175, |Δτ|=0.6
        dict(L=400, tau_max=2.0, T_O=5300.0, self_consistent=True, dS_l1=0.04),  # rel_tau=0.20,  dT=0.189, |Δτ|=0.5
        dict(L=800, tau_max=2.4, T_O=6400.0, self_consistent=True, dS_l1=0.04),
    ]
    cal_stable, annotated, _failed = pp.classify_calibration_stable(curve)
    assert cal_stable is True, (
        "small-τ noise-regime chain (sub-Sokal Δτ, stable L1 shape, self-consistent) must be cal_stable=True")
    # both relaxations are load-bearing here: rel_tau ≥ TAU_TOL but |Δτ| < TAU_ABS_FLOOR, and dT ≥ STAB_TOL
    # but small_tau forgives the magnitude — yet two consecutive STABLE rungs are reached.
    assert annotated[1]["small_tau"] is True and annotated[1]["abs_dtau"] < pp.TAU_ABS_FLOOR
    assert [r.get("step_class") for r in annotated[1:3]] == ["STABLE", "STABLE"]


def test_cal_stable_small_tau_drifting_shape_still_fails():
    """GATE-NOT-VACUOUS guard: a small-τ chain whose L1-normalized S_a SHAPE is still drifting
    (dS_l1 ≥ STAB_TOL) MUST stay cal_stable=False — the shape guard is RETAINED ALWAYS even when the
    magnitude (dT) axis is relaxed in the small-τ regime.  Proves the floor did not make the gate vacuous."""
    curve = [
        dict(L=100, tau_max=2.0, T_O=5200.0, self_consistent=True, dS_l1=None),
        dict(L=200, tau_max=2.3, T_O=5400.0, self_consistent=True, dS_l1=0.30),  # shape drifting ≥ 0.15
        dict(L=400, tau_max=2.1, T_O=5300.0, self_consistent=True, dS_l1=0.25),
        dict(L=800, tau_max=2.2, T_O=5350.0, self_consistent=True, dS_l1=0.28),
    ]
    cal_stable, annotated, failed_axis = pp.classify_calibration_stable(curve)
    assert cal_stable is False, "drifting L1 shape (dS_l1 ≥ STAB_TOL) must veto even in the small-τ regime"
    assert "aggregate_T_O" in failed_axis
    assert all(r.get("step_class") == "NOT-STABLE" for r in annotated[1:])


def test_cal_stable_small_tau_floor_does_not_forgive_real_tau_drift():
    """The τ ABSOLUTE floor forgives only SUB-resolution (|Δτ| < TAU_ABS_FLOOR=1 sweep) jitter, NOT a
    genuine τ trend: a small-ish chain whose rung-to-rung |Δτ| ≥ TAU_ABS_FLOOR still fails the τ axis."""
    curve = [
        dict(L=100, tau_max=2.0, T_O=5200.0, self_consistent=True, dS_l1=None),
        dict(L=200, tau_max=3.5, T_O=5400.0, self_consistent=True, dS_l1=0.02),  # |Δτ|=1.5 ≥ floor
        dict(L=400, tau_max=5.2, T_O=5300.0, self_consistent=True, dS_l1=0.02),  # |Δτ|=1.7 ≥ floor
        dict(L=800, tau_max=2.0, T_O=5350.0, self_consistent=True, dS_l1=0.02),
    ]
    cal_stable, _annotated, failed_axis = pp.classify_calibration_stable(curve)
    assert cal_stable is False, "rung-to-rung |Δτ| ≥ TAU_ABS_FLOOR must still veto the τ axis"
    assert "tau_hat" in failed_axis


def test_cal_stable_small_tau_vetoed_by_non_self_consistent():
    """The small-τ floor relaxes only the RELATIVE-change axes — it must NOT bypass self-consistency.
    A small-τ chain that is NOT self-consistent (L < SOKAL_C·τ_max) MUST still fail: self_consistent is a
    hard AND on the τ axis AND a required conjunct of the small-τ T_O relaxation."""
    curve = [
        dict(L=8, tau_max=2.0, T_O=5200.0, self_consistent=False, dS_l1=None),   # L=8 < 5·2=10
        dict(L=16, tau_max=2.2, T_O=5300.0, self_consistent=False, dS_l1=0.02),
        dict(L=32, tau_max=2.1, T_O=5250.0, self_consistent=False, dS_l1=0.02),
    ]
    cal_stable, _annotated, failed_axis = pp.classify_calibration_stable(curve)
    assert cal_stable is False, "a non-self-consistent (L < C·τ) chain must fail even in the small-τ regime"
    assert "tau_hat" in failed_axis  # self_consistent folds into the tau axis
