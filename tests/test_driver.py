"""Task 9 — tests for src/htdml/driver.py (Stage A/B/C driver + fork/restore + gates + router).

CPU ONLY — NO ``dtm.train`` (it HARD-REQUIRES a GPU; the CPU vs GPU split).  The Stage A/B/C
TRAINING is GPU-wired (smoke-deferred, Task 12).  What is CPU-unit-tested here:

  * the 6-token outcome ROUTER (``route_seed`` / ``route_run``) — PURE functions, all 6 tokens +
    per-seed predicate + two-seed aggregation reachable (the Task-11 zero-compute battery proves the
    full reachability matrix; these tests prove the logic);
  * the per-update REJECT gate (``reject_gate``) — PURE logic for the 3 reject conditions, halve-LR,
    stop-after-2-consecutive;
  * the FORK + OUT-OF-BAND restore mechanism — save→load×2→re-inject autocorrelations/key/opt-state,
    on a small REAL perturbed 4_4 DTM (the Task-4-confirmed DTM.save/load round-trip);
  * λ=0 ≡ control bitwise (the traced-0.0 multiply through compat_loss; no python branch on λ);
  * float64 scoping (the compat grad is x64-scoped — no global leak; the float32 DTM.load test stays
    green in the same suite).

Disjointness + exhaustiveness of the router are asserted directly (no input fires two tokens; every
input maps to exactly one of the 6).
"""

from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path

import numpy as np
import pytest

_REPO_ROOT = Path(__file__).resolve().parents[1]
for _p in (str(_REPO_ROOT), str(_REPO_ROOT / "src")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import htdml  # noqa: E402,F401  (triggers bootstrap_paths)

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import jax.random as jr  # noqa: E402

from htdml import driver as D  # noqa: E402
from harness import probe_primitives as pp  # noqa: E402


# ============================================================================== acceptance constants
# Synthetic acceptance constants (the PINS values are TBD-at-Task-12; the router/gate take them as
# params so the logic is testable with synthetic values).  These are NOT production bars.
ACC = D.AcceptanceConstants(
    ESS_min=10.0, C=5.0, L_traj=2000, N_chains=8, N_R=4,
    Q_GAIN=1.25, TAU_DROP=0.25, Q_DROP_MAX=0.10, R_GRAD50_MAX=0.05,
    BCE_TOL=0.05, FID_TOL=0.10, GPU_H_CAP=4.0,
)


def _layer(*, q=1.0, tau=10.0, ess=20.0, r50=0.01, gnorm=1.0, cal_stable=True, L_traj=2000):
    """One synthetic per-layer probe record (4 of these = a seed's measurement)."""
    return dict(Q_struct_perp=float(q), tau_int_Y=float(tau), ESS_hat=float(ess),
                **{"r_grad[50]": float(r50)}, gradient_norm=float(gnorm),
                cal_stable=bool(cal_stable), L_traj=int(L_traj), tau_hat=float(tau))


def _seed(*, layers=None, control_layers=None, bce=0.10, fid=1.0,
          control_bce=0.10, control_fid=1.0, gpu_h=1.0, budget_wall=False,
          cal_all_stable=True, traj_all_resolved=True):
    """A synthetic per-seed metrics bundle (4 joint layers vs 4 control layers + quality + budget)."""
    if layers is None:
        layers = [_layer() for _ in range(4)]
    if control_layers is None:
        control_layers = [_layer() for _ in range(4)]
    return D.SeedMetrics(
        joint_layers=layers, control_layers=control_layers,
        bce=float(bce), fid=float(fid), control_bce=float(control_bce), control_fid=float(control_fid),
        gpu_h=float(gpu_h), budget_wall=bool(budget_wall),
        cal_all_stable=bool(cal_all_stable), traj_all_resolved=bool(traj_all_resolved),
    )


# ====================================================================== ROUTER — the 6 tokens reachable
def test_route_seed_budget_wall_first():
    """Priority 1 — BUDGET-WALL is checked FIRST (even if everything else would fail/pass)."""
    m = _seed(budget_wall=True, cal_all_stable=False, bce=99.0)  # other failures present
    assert D.route_seed(m, ACC) == "BUDGET-WALL"


def test_route_seed_budget_wall_via_gpu_h_cap():
    """BUDGET-WALL also fires when gpu_h exceeds the cap (the allocation gate)."""
    m = _seed(gpu_h=ACC.GPU_H_CAP + 0.01)
    assert D.route_seed(m, ACC) == "BUDGET-WALL"


def test_route_seed_q_calibration_fail():
    """Priority 2 — Q-CALIBRATION-FAIL when any layer's T_O doubling-stability fails."""
    m = _seed(cal_all_stable=False)
    assert D.route_seed(m, ACC) == "Q-CALIBRATION-FAIL"


def test_route_seed_plateau_unresolved():
    """Priority 3 — PLATEAU-UNRESOLVED: calibration OK but trajectory adequacy fails (L_traj<C·τ̂)."""
    m = _seed(cal_all_stable=True, traj_all_resolved=False)
    assert D.route_seed(m, ACC) == "PLATEAU-UNRESOLVED"


def test_route_seed_plateau_unresolved_from_layer_Ltraj():
    """PLATEAU-UNRESOLVED also fires when a layer's own L_traj < C·τ̂ (computed, not just the flag)."""
    bad = [_layer() for _ in range(4)]
    bad[2] = _layer(tau=500.0, L_traj=2000)  # C·τ̂ = 5·500 = 2500 > 2000 → unresolved
    m = _seed(layers=bad)
    assert D.route_seed(m, ACC) == "PLATEAU-UNRESOLVED"


def test_route_seed_quality_loss_bce():
    """Priority 4 — QUALITY-LOSS when BCE > control + 5% (measurement valid)."""
    m = _seed(bce=0.20, control_bce=0.10)  # 0.20 > 0.10*1.05
    assert D.route_seed(m, ACC) == "QUALITY-LOSS"


def test_route_seed_quality_loss_fid():
    """QUALITY-LOSS also when FID > control + 10%."""
    m = _seed(fid=2.0, control_fid=1.0)  # 2.0 > 1.0*1.10
    assert D.route_seed(m, ACC) == "QUALITY-LOSS"


def test_route_seed_htdml_margin_negative():
    """Priority 5 — HTDML-MARGIN-NEGATIVE: all valid + quality OK but improvement gate NOT met."""
    # joint Q == control Q (no gain), τ not lowered → margin negative.
    m = _seed(layers=[_layer(q=1.0, tau=10.0) for _ in range(4)],
              control_layers=[_layer(q=1.0, tau=10.0) for _ in range(4)])
    assert D.route_seed(m, ACC) == "HTDML-MARGIN-NEGATIVE"


def test_route_seed_htdml_margin_positive_via_Q():
    """Priority 6 — HTDML-MARGIN-POSITIVE: lower-quartile joint Q ≥ 1.25× control."""
    m = _seed(layers=[_layer(q=2.0) for _ in range(4)],
              control_layers=[_layer(q=1.0) for _ in range(4)])
    assert D.route_seed(m, ACC) == "HTDML-MARGIN-POSITIVE"


def test_route_seed_htdml_margin_positive_via_tau():
    """HTDML-MARGIN-POSITIVE via the τ leg: worst-layer τ_int,Y ≥ 25% lower than control."""
    m = _seed(layers=[_layer(q=1.0, tau=7.0) for _ in range(4)],     # 7 ≤ 0.75·10
              control_layers=[_layer(q=1.0, tau=10.0) for _ in range(4)])
    assert D.route_seed(m, ACC) == "HTDML-MARGIN-POSITIVE"


# ====================================================================== ROUTER — disjoint + exhaustive
def test_route_seed_is_exhaustive_and_disjoint():
    """Every input maps to EXACTLY ONE of the 6 tokens (route_seed is total + single-valued)."""
    cases = [
        _seed(budget_wall=True), _seed(cal_all_stable=False), _seed(traj_all_resolved=False),
        _seed(bce=0.5), _seed(fid=5.0),
        _seed(layers=[_layer(q=1.0) for _ in range(4)], control_layers=[_layer(q=1.0) for _ in range(4)]),
        _seed(layers=[_layer(q=3.0) for _ in range(4)], control_layers=[_layer(q=1.0) for _ in range(4)]),
    ]
    for m in cases:
        tok = D.route_seed(m, ACC)
        assert tok in D.TOKENS, f"route_seed returned a non-token: {tok}"


def test_all_six_tokens_reachable_from_route_seed():
    """The set of tokens route_seed can emit is exactly the 6-token vocab (battery prerequisite)."""
    reached = {
        D.route_seed(_seed(budget_wall=True), ACC),
        D.route_seed(_seed(cal_all_stable=False), ACC),
        D.route_seed(_seed(traj_all_resolved=False), ACC),
        D.route_seed(_seed(bce=0.5), ACC),
        D.route_seed(_seed(layers=[_layer(q=1.0) for _ in range(4)],
                           control_layers=[_layer(q=1.0) for _ in range(4)]), ACC),
        D.route_seed(_seed(layers=[_layer(q=3.0) for _ in range(4)],
                           control_layers=[_layer(q=1.0) for _ in range(4)]), ACC),
    }
    assert reached == set(D.TOKENS), f"not all 6 tokens reachable: got {reached}"


# ====================================================================== per-seed PASS predicate
def test_seed_passes_true_when_all_gates_met():
    """seed_passes True iff quality OK + all resolved + ESS-adequate + r50 OK + improvement + ESS non-deg."""
    m = _seed(layers=[_layer(q=2.0, ess=30.0) for _ in range(4)],
              control_layers=[_layer(q=1.0, ess=25.0) for _ in range(4)])
    assert D.seed_passes(m, ACC) is True


def test_seed_passes_false_on_ess_degradation():
    """ESS non-degradation is a CO-REQUIREMENT — a Q gain via ESS collapse must NOT pass."""
    # joint Q is 2× control BUT joint ESS collapsed below control (Q inflated via T_O shrink).
    m = _seed(layers=[_layer(q=2.0, ess=5.0) for _ in range(4)],
              control_layers=[_layer(q=1.0, ess=25.0) for _ in range(4)])
    assert D.seed_passes(m, ACC) is False


def test_seed_passes_false_when_margin_negative():
    m = _seed(layers=[_layer(q=1.0) for _ in range(4)],
              control_layers=[_layer(q=1.0) for _ in range(4)])
    assert D.seed_passes(m, ACC) is False


# ====================================================================== two-seed aggregation
def _PO():  # a passing (positive) seed
    return _seed(layers=[_layer(q=2.0, ess=30.0) for _ in range(4)],
                 control_layers=[_layer(q=1.0, ess=25.0) for _ in range(4)])


def _MN():  # a margin-negative seed
    return _seed(layers=[_layer(q=1.0) for _ in range(4)],
                 control_layers=[_layer(q=1.0) for _ in range(4)])


def _QL():  # a quality-loss seed
    return _seed(bce=0.5, control_bce=0.10)


def test_route_run_positive_iff_both_pass():
    """POSITIVE iff BOTH seeds pass all final gates."""
    assert D.route_run(_PO(), _PO(), ACC) == "HTDML-MARGIN-POSITIVE"


def test_route_run_one_margin_negative_is_negative():
    """(MN, PO) → HTDML-MARGIN-NEGATIVE (a run can't be POSITIVE unless BOTH pass)."""
    assert D.route_run(_MN(), _PO(), ACC) == "HTDML-MARGIN-NEGATIVE"
    assert D.route_run(_PO(), _MN(), ACC) == "HTDML-MARGIN-NEGATIVE"


def test_route_run_quality_loss_takes_precedence_over_negative():
    """(QL, ·) → QUALITY-LOSS (quality failure beats a margin-negative)."""
    assert D.route_run(_QL(), _PO(), ACC) == "QUALITY-LOSS"
    assert D.route_run(_QL(), _MN(), ACC) == "QUALITY-LOSS"


def test_route_run_measurement_invalid_takes_worst_precedence():
    """A run CANNOT be POSITIVE unless BOTH seeds are measurement-valid; worst-precedence wins."""
    budget = _seed(budget_wall=True)
    calfail = _seed(cal_all_stable=False)
    plateau = _seed(traj_all_resolved=False)
    # BUDGET-WALL > Q-CALIBRATION-FAIL > PLATEAU-UNRESOLVED
    assert D.route_run(budget, calfail, ACC) == "BUDGET-WALL"
    assert D.route_run(calfail, plateau, ACC) == "Q-CALIBRATION-FAIL"
    assert D.route_run(plateau, _PO(), ACC) == "PLATEAU-UNRESOLVED"


def test_route_run_is_exhaustive_and_disjoint():
    """route_run is total + single-valued over the cross product of representative seeds."""
    reps = [_PO(), _MN(), _QL(), _seed(budget_wall=True), _seed(cal_all_stable=False),
            _seed(traj_all_resolved=False)]
    for a in reps:
        for b in reps:
            tok = D.route_run(a, b, ACC)
            assert tok in D.TOKENS


# ====================================================================== per-update REJECT gate
def test_reject_gate_accepts_clean_candidate():
    joint = [_layer(q=1.0, ess=20.0, r50=0.01) for _ in range(4)]
    control = [_layer(q=1.0, ess=20.0) for _ in range(4)]
    dec = D.reject_gate(joint, control, ACC)
    assert dec.reject is False and dec.reason is None


def test_reject_gate_rejects_on_q_drop():
    """Reject if lower-quartile-over-4-layers Q_struct drops > 10% vs matched control."""
    joint = [_layer(q=0.80) for _ in range(4)]      # 20% drop > 10%
    control = [_layer(q=1.0) for _ in range(4)]
    dec = D.reject_gate(joint, control, ACC)
    assert dec.reject is True and "Q_drop" in dec.reason


def test_reject_gate_rejects_on_ess_below_min():
    """Reject if worst gradient-observable layer ESS_hat < ESS_min (window-adequacy gate)."""
    joint = [_layer(ess=5.0) for _ in range(4)]     # < ESS_min=10
    control = [_layer(ess=20.0) for _ in range(4)]
    dec = D.reject_gate(joint, control, ACC)
    assert dec.reject is True and "ESS" in dec.reason


def test_reject_gate_ess_gate_requires_trajectory_resolved():
    """The ESS gate fires only if τ̂ is trajectory-resolved (L_traj ≥ C·τ̂); else PLATEAU-UNRESOLVED."""
    # ESS below min BUT trajectory not resolved (τ huge) → the gate routes to plateau, not ESS-reject.
    joint = [_layer(ess=5.0, tau=500.0, L_traj=2000) for _ in range(4)]   # C·τ̂=2500>2000
    control = [_layer(ess=20.0) for _ in range(4)]
    dec = D.reject_gate(joint, control, ACC)
    assert dec.reject is True and dec.reason == "PLATEAU-UNRESOLVED"


def test_reject_gate_rejects_on_r_grad50():
    """Reject if r_grad[50] > 0.05 (full-window plateau sanity, absolute)."""
    joint = [_layer(r50=0.10) for _ in range(4)]
    control = [_layer() for _ in range(4)]
    dec = D.reject_gate(joint, control, ACC)
    assert dec.reject is True and "r_grad" in dec.reason


def test_reject_loop_halves_lr_then_stops_after_two_consecutive():
    """After a rejection: halve encoder LR.  After 2 CONSECUTIVE rejections: stop the seed."""
    st = D.RejectState(encoder_lr=0.001)
    # first reject → halve LR, not stopped
    st = D.apply_rejection(st)
    assert st.encoder_lr == pytest.approx(0.0005) and st.consecutive == 1 and st.stop is False
    # second consecutive reject → halve again, STOP
    st = D.apply_rejection(st)
    assert st.encoder_lr == pytest.approx(0.00025) and st.consecutive == 2 and st.stop is True


def test_acceptance_resets_consecutive_rejections():
    """An ACCEPTED update between rejections resets the consecutive counter (not 2-in-a-row)."""
    st = D.RejectState(encoder_lr=0.001)
    st = D.apply_rejection(st)          # consecutive=1
    st = D.apply_acceptance(st)         # reset
    assert st.consecutive == 0 and st.stop is False
    st = D.apply_rejection(st)          # consecutive=1 again, NOT 2 → no stop
    assert st.consecutive == 1 and st.stop is False


# ====================================================================== λ=0 ≡ control (traced multiply)
def test_lambda_zero_is_control_bitwise():
    """The joint compat term at λ=0 is bitwise 0.0 via a TRACED multiply (no python branch on λ)."""
    import htdml.compatibility as C

    # build a tiny finite L_compat input directly (no DTM needed): a 1-step clamp + a trivial map.
    # Reuse the REAL compat path on the 4_4 fixture to get a genuine nonzero L_compat, then check
    # that the λ-multiply zeroes it bitwise at λ=0 and scales linearly at λ=0.5.
    from tests.fixture_6_4 import _build_fixture_step, _rng_pm1, _x64

    _dtm, step = _build_fixture_step()
    beta = float(step.training_spec.beta)
    with _x64():
        maps = C.build_compat_maps(step)
        rng = np.random.default_rng(3)
        clamp = jnp.asarray(np.stack([_rng_pm1(rng, maps["n_clamp"]) for _ in range(4)]))
        l = float(C.L_compat(clamp, [maps], beta))
        v0, fin0 = D.compat_term(0.0, clamp, [maps], beta)
        vh, finh = D.compat_term(0.5, clamp, [maps], beta)
    assert float(v0) == 0.0 and bool(fin0), "λ=0 compat term must be bitwise 0.0 (the control)"
    assert bool(finh) and np.isclose(float(vh), 0.5 * l), "λ·L_compat must scale linearly (traced λ)"


def test_compat_term_does_not_branch_on_lambda():
    """compat_term has no python `if lam == 0` branch — it is the traced-multiply (source inspection)."""
    import inspect

    src = inspect.getsource(D.compat_term)
    assert "if lam" not in src and "if lam ==" not in src, (
        "compat_term must not python-branch on λ (the control is λ=0 through the SAME traced code path)")


# ====================================================================== λ STEERS the encoder (the real gate)
def _tiny_ste_encoder(n_img):
    """A tiny DIFFERENTIABLE STE 'encoder' of latent width ``n_img`` for the steering test: params is a
    weight matrix; b0 = STE-hard-sign(x @ W) ∈ {−1,+1} with a tanh backward surrogate (the SAME STE the
    real BinaryAutoencoder uses).  Returns encode_fn.  Lets the steering property be exercised on the
    narrow (n_img=3) 4_4 fixture image_output block without the 196-wide production AE."""
    import htdml.autoencoder as AE

    def encode_fn(params, x):
        logits = jnp.asarray(x) @ params["W"]              # (B, n_img) logits, depends on params
        b0 = AE._ste_hard_sign(logits)                     # {−1,+1}, ∂/∂logits = 1−tanh² (STE)
        return b0, logits

    return encode_fn


def test_lambda_steers_the_encoder_nonzero_at_lam_gt0_zero_at_lam0():
    """THE real 'λ steers the encoder' gate (was MISSING — the bug): ∂(λ·L_compat-only loss)/∂ae_params
    is NON-ZERO at λ>0 (the compat steers the encoder via the STE through the image_output latent) and
    EXACTLY ZERO at λ=0 (the control).  On the real 4_4 fixture DTM (image_output block n_img=3)."""
    from tests.fixture_6_4 import _build_fixture_step, _rng_pm1

    _dtm, step = _build_fixture_step()
    beta = float(step.training_spec.beta)
    step_maps = D.step_maps_for(step)
    n_img = int(step_maps[0]["n_img"])
    n_clamp = int(step_maps[0]["n_clamp"])
    n_rest = n_clamp - n_img

    encode_fn = _tiny_ste_encoder(n_img)
    rng = np.random.default_rng(0)
    n_in = 5
    # build the float64 arrays INSIDE the x64 scope (else JAX truncates to float32 with a UserWarning).
    with D._x64():
        params = {"W": jnp.asarray(rng.normal(size=(n_in, n_img)), dtype=jnp.float64)}
        x_batch = jnp.asarray(rng.normal(size=(4, n_in)), dtype=jnp.float64)
        label_clamp = jnp.asarray(_rng_pm1(rng, n_rest))           # label_output + b_t columns (hard)
        bt_clamp = jnp.zeros((0,))                                 # all 'rest' folded into label_clamp

    def steer_loss(p, lam):
        with D._x64():
            val, _fin = D.compat_steering_loss(p, x_batch, label_clamp, bt_clamp, step_maps, beta,
                                               lam, n_img=n_img, encode_fn=encode_fn)
        return val

    # λ>0 → gradient w.r.t. the encoder params is NON-ZERO (the compat steers the encoder).
    with D._x64():
        g_pos = jax.grad(lambda p: steer_loss(p, 0.7))(params)
    g_pos_W = np.asarray(g_pos["W"])
    assert np.any(np.abs(g_pos_W) > 1e-9), (
        "∂(λ·L_compat)/∂ae_params is ZERO at λ>0 — the compat does NOT steer the encoder (the encode→"
        "clamp wiring is not inside the differentiated loss; Stage C would be inert)")

    # λ=0 → gradient is EXACTLY zero (the control, via the traced-0.0 multiply).
    with D._x64():
        g_zero = jax.grad(lambda p: steer_loss(p, 0.0))(params)
    g_zero_W = np.asarray(g_zero["W"])
    assert np.all(g_zero_W == 0.0), (
        f"∂(λ·L_compat)/∂ae_params must be EXACTLY 0 at λ=0 (control): max|g|={np.abs(g_zero_W).max()}")

    print(f"\n[STEERING] λ=0.7: max|∂L_compat/∂W|={np.abs(g_pos_W).max():.4g} (≠0 → steers encoder); "
          f"λ=0: max|∂/∂W|={np.abs(g_zero_W).max():.4g} (=0 → control)")


def test_lambda_zero_joint_update_is_bitwise_control():
    """The λ=0 joint update yields ae_params bitwise-identical to the control (pure-reconstruction)
    update — the compat half contributes EXACTLY 0 to the gradient at λ=0 (same code path, traced 0.0);
    λ>0 must DIFFER (the steering actually moved the encoder)."""
    import optax

    from tests.fixture_6_4 import _build_fixture_step, _rng_pm1

    _dtm, step = _build_fixture_step()
    beta = float(step.training_spec.beta)
    step_maps = D.step_maps_for(step)
    n_img = int(step_maps[0]["n_img"])
    n_clamp = int(step_maps[0]["n_clamp"])
    n_rest = n_clamp - n_img
    encode_fn = _tiny_ste_encoder(n_img)
    rng = np.random.default_rng(1)
    n_in = 5
    # build the float64 arrays INSIDE the x64 scope (else JAX truncates to float32 with a UserWarning).
    with D._x64():
        params = {"W": jnp.asarray(rng.normal(size=(n_in, n_img)), dtype=jnp.float64)}
        x_batch = jnp.asarray(rng.normal(size=(4, n_in)), dtype=jnp.float64)
        label_clamp = jnp.asarray(_rng_pm1(rng, n_rest))
        bt_clamp = jnp.zeros((0,))

    # the differentiable half: a stand-in reconstruction (sum of squared logits) + λ·L_compat steering.
    def loss_at(p, lam):
        with D._x64():
            recon = jnp.sum((jnp.asarray(x_batch) @ p["W"]) ** 2)
            compat, _fin = D.compat_steering_loss(p, x_batch, label_clamp, bt_clamp, step_maps, beta,
                                                  lam, n_img=n_img, encode_fn=encode_fn)
        return recon + compat

    optim = optax.sgd(0.01)

    def one_update(lam):
        with D._x64():
            g = jax.grad(lambda p: loss_at(p, lam))(params)
            opt_state = optim.init(params)
            updates, _ = optim.update(g, opt_state, params)
            return optax.apply_updates(params, updates)

    joint0 = one_update(0.0)         # λ=0 (control via the same code path)
    control = one_update(0.0)        # the pure-reconstruction reference (λ=0 IS the control)
    jointL = one_update(0.5)         # λ>0 must DIFFER (the steering actually moved the encoder)

    np.testing.assert_array_equal(np.asarray(joint0["W"]), np.asarray(control["W"]))  # bitwise control
    assert not np.allclose(np.asarray(jointL["W"]), np.asarray(control["W"])), (
        "λ>0 update must DIFFER from the λ=0 control (else the steering is inert)")


# ====================================================================== live ACP coefficient stored
def test_epoch_record_stores_live_acp_coefficient():
    """The per-epoch + per-layer record stores the LIVE ACP coefficient (correlation_penalty post-
    adapt_param) — the field that was MISSING from every driver record."""
    probe_layer = dict(gradient_norm=1.5, **{"r_grad[1]": 0.3, "r_grad[50]": 0.02},
                       tau_int_Y=12.0, ESS_hat=22.0, Q_struct_perp=1.1)
    rec = D.make_epoch_layer_record(epoch=3, layer=2, bce=0.12, fid=1.4,
                                    correlation_penalty=0.0042, probe_layer_dict=probe_layer)
    assert rec.correlation_penalty == pytest.approx(0.0042), "live ACP coefficient not stored"
    # every plan-required scalar is present on the record.
    for f in ("epoch", "layer", "bce", "fid", "correlation_penalty", "gradient_norm",
              "r_grad_1", "r_grad_50", "tau_int_Y", "ESS_hat", "Q_struct_perp"):
        assert hasattr(rec, f), f"EpochLayerRecord missing field {f}"


def test_live_acp_coefficient_reads_post_adapt_param_vector():
    """live_acp_coefficient reads cp_coeffs[step] (the post-adapt_param value the DTM loop maintains)."""
    cp_coeffs = np.asarray([0.001, 0.0035, 0.0])     # one per reverse layer (post-adapt_param)
    assert D.live_acp_coefficient(None, 1, cp_coeffs) == pytest.approx(0.0035)
    assert D.live_acp_coefficient(None, 2, cp_coeffs) == pytest.approx(0.0)


# ====================================================================== float64 scoping (no global leak)
def test_compat_term_is_x64_scoped_no_global_leak():
    """The compat grad runs x64-scoped; after the call jax_enable_x64 is restored (no global leak)."""
    before = jax.config.jax_enable_x64
    from tests.fixture_6_4 import _build_fixture_step, _rng_pm1
    import htdml.compatibility as C

    _dtm, step = _build_fixture_step()
    beta = float(step.training_spec.beta)
    # build the map OUTSIDE x64 the way the driver does (driver scopes x64 internally around grad).
    maps = D.build_compat_maps_x64(step)
    rng = np.random.default_rng(9)
    clamp = np.stack([_rng_pm1(rng, maps["n_clamp"]) for _ in range(4)])
    val, grad, is_finite = D.compat_value_and_grad_x64(0.5, clamp, [maps], beta)
    assert jax.config.jax_enable_x64 == before, "compat_value_and_grad_x64 LEAKED the global x64 flag"
    assert bool(is_finite) and np.all(np.isfinite(np.asarray(grad)))
    # grad shape = (K_steps, n_clamp); only the image_output columns carry signal (the rest are zeroed
    # by the caller in production, but the raw grad is over all clamp columns here).
    assert np.asarray(grad).shape == (4, maps["n_clamp"])


# ====================================================================== FORK + out-of-band restore
def test_fork_and_out_of_band_restore_round_trip():
    """Fork = save_epoch + load×2 (control + joint); after each load OUT-OF-BAND restore the per-step
    autocorrelations (DTM.load returns {}), dtm.key, and the opt-state position.  Both arms share the
    restored key; the probe key is independent.  On a REAL perturbed 4_4 DTM (NO dtm.train)."""
    import equinox as eqx

    from thrmlDenoising.DTM import DTM
    from thrmlDenoising.utils import make_cfg
    from tests.fixture_6_4 import FIXTURE_CFG, _perturb_step

    dtm = DTM(make_cfg(**FIXTURE_CFG))
    dtm.steps[0] = _perturb_step(dtm.steps[0], seed=7)
    # inject a non-trivial autocorrelations dict (the unsaved static the driver carries out-of-band).
    autocorr_payload = {0: np.asarray(0.42, dtype=np.float64), 1: np.asarray(0.31, dtype=np.float64)}
    dtm.steps[0] = eqx.tree_at(lambda s: s.autocorrelations, dtm.steps[0], dict(autocorr_payload))

    parent_hash = pp._weights_hash(dtm.steps[0])
    parent_counts = pp._find_counts(dtm.steps[0].opt_state)
    parent_key = pp._key_list(dtm)

    workdir = tempfile.mkdtemp(prefix="htdml_fork_")
    control_dtm, joint_dtm = D.fork_checkpoint(dtm, workdir)

    for arm_name, arm in (("control", control_dtm), ("joint", joint_dtm)):
        st = arm.steps[0]
        # weights / opt-counts / key restored (in the save-mask or out-of-band reinjected)
        assert pp._weights_hash(st) == parent_hash, f"{arm_name}: weights not restored bitwise"
        assert pp._find_counts(st.opt_state) == parent_counts, f"{arm_name}: opt-state counts not restored"
        assert pp._key_list(arm) == parent_key, f"{arm_name}: dtm.key not restored out-of-band"
        # autocorrelations RE-INJECTED (NOT the {} DTM.load returns)
        assert set(st.autocorrelations.keys()) == set(autocorr_payload.keys()), (
            f"{arm_name}: autocorrelations not re-injected out-of-band (DTM.load drops them)")
        for k, v in autocorr_payload.items():
            np.testing.assert_array_equal(np.asarray(st.autocorrelations[k]), np.asarray(v))

    # both arms share the SAME restored key (control and joint start from the identical parent state).
    assert pp._key_list(control_dtm) == pp._key_list(joint_dtm), "control and joint must share dtm.key"


def test_fork_arms_are_independent_objects():
    """The control and joint arms must be DISTINCT objects — mutating one must not touch the other
    (else ACP adapt_param would couple the arms)."""
    import equinox as eqx

    from thrmlDenoising.DTM import DTM
    from thrmlDenoising.utils import make_cfg
    from tests.fixture_6_4 import FIXTURE_CFG, _perturb_step

    dtm = DTM(make_cfg(**FIXTURE_CFG))
    dtm.steps[0] = _perturb_step(dtm.steps[0], seed=11)
    dtm.steps[0] = eqx.tree_at(lambda s: s.autocorrelations, dtm.steps[0],
                               {0: np.asarray(0.5, dtype=np.float64)})

    workdir = tempfile.mkdtemp(prefix="htdml_fork_indep_")
    control_dtm, joint_dtm = D.fork_checkpoint(dtm, workdir)

    assert control_dtm is not joint_dtm
    assert control_dtm.steps is not joint_dtm.steps
    # mutate the joint arm's autocorrelations → the control arm must be untouched
    joint_dtm.steps[0].autocorrelations[99] = np.asarray(123.0)
    assert 99 not in control_dtm.steps[0].autocorrelations, "fork arms share autocorrelations state (BUG)"


def test_restore_out_of_band_reinjects_after_a_raw_load():
    """The OUT-OF-BAND restore primitive (driver) re-injects autocorrelations + key + sets opt-state on a
    freshly DTM.load'd arm (DTM.load itself returns autocorrelations == {} and a fresh key)."""
    import equinox as eqx

    from thrmlDenoising.DTM import DTM
    from thrmlDenoising.utils import make_cfg
    from tests.fixture_6_4 import FIXTURE_CFG, _perturb_step

    dtm = DTM(make_cfg(**FIXTURE_CFG))
    dtm.steps[0] = _perturb_step(dtm.steps[0], seed=5)
    payload = {0: np.asarray(0.7), 3: np.asarray(0.2)}
    dtm.steps[0] = eqx.tree_at(lambda s: s.autocorrelations, dtm.steps[0], dict(payload))

    workdir = tempfile.mkdtemp(prefix="htdml_oob_")
    dtm.logging_and_saving_dir = workdir
    dtm.save_epoch(0)
    base = os.path.join(workdir, "model_saving")

    loaded = DTM.load(base, epoch=0)
    # DTM.load drops autocorrelations + uses a fresh key (this is the bug we restore around)
    assert loaded.steps[0].autocorrelations == {}, "precondition: DTM.load must drop autocorrelations"

    captured = D.capture_arm_state(dtm)   # the parent's (key, per-step autocorrelations)
    D.restore_out_of_band(loaded, captured)
    assert loaded.steps[0].autocorrelations.keys() == payload.keys()
    assert pp._key_list(loaded) == pp._key_list(dtm)
