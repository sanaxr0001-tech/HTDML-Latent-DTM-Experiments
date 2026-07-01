"""calib_logic — PURE (CPU-verifiable, no-GPU) logic for the Task-12 calibration freeze.

Separated from ``scripts/calibrate_epoch_cost.py`` so the NON-GPU logic (the wall-time budget
guard, the trajectory-adequacy freeze rules for ``L_traj`` / ``N_chains`` / ``N_R`` / ``C``, and the
a-priori ``ESS_min`` rule) is unit-testable on CPU WITHOUT any ``dtm.train`` / GPU sampling.

The Task-12 split (pre-registered scope, see design notes):
  * ``L_traj`` / ``N_chains`` / ``N_R`` / ``C``  are FROZEN FROM MEASUREMENT (runtime/adequacy
    quantities — these functions turn a measured τ̂ + the budget into the frozen values);
  * ``ESS_min`` is NOT a calibration output.  It is set by :func:`ess_min_rule`, an a-priori
    scientific ACCEPTANCE threshold defined BEFORE any joint/control comparison (there is none in
    Task 12).  It must NOT be reverse-engineered from a favorable smoke result.

Bias caveat (design notes, "Half-Sokal T_O bias"): the half-Sokal τ̂ is ~0.86× the exact τ.  The
ABSOLUTE thresholds (ESS_min, C) are frozen against the SAME biased estimator → self-consistent; the
companion only ever uses τ̂/Q as a RELATIVE ruler, so the bias cancels in the gates.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass
from typing import Optional


# ============================================================================== wall-time guard
class BudgetWall(RuntimeError):
    """Raised (or signalled) when the 1hr local-4060 cap is hit.  A BUDGET-WALL STOP is NOT a
    failure — it means the cap bound the run; the caller reports what was frozen so far."""


@dataclass
class WallClock:
    """A monotonic wall-time budget guard for the 1hr local-4060 cap.

    Usage::

        wc = WallClock(cap_seconds=3600.0)
        ...
        wc.checkpoint("after smoke")          # raises/flags if over cap
        remaining = wc.remaining()

    The guard is CHECKED between stages (cooperative); it does not interrupt a running jax kernel.
    ``would_exceed(est)`` lets a stage decline to start if its estimated cost would blow the cap.
    """

    cap_seconds: float = 3600.0
    _start: float = None  # type: ignore[assignment]

    def __post_init__(self):
        if self._start is None:
            self._start = time.monotonic()

    def elapsed(self) -> float:
        return time.monotonic() - self._start

    def remaining(self) -> float:
        return self.cap_seconds - self.elapsed()

    def over_cap(self) -> bool:
        return self.elapsed() >= self.cap_seconds

    def would_exceed(self, est_seconds: float, *, margin: float = 1.0) -> bool:
        """True iff starting a stage estimated at ``est_seconds`` (× safety ``margin``) would push
        past the cap.  Lets a stage decline to start rather than overrun."""
        return self.elapsed() + margin * float(est_seconds) >= self.cap_seconds

    def checkpoint(self, label: str = "", *, raise_on_over: bool = False) -> bool:
        """Return True if still within budget; if over, optionally raise :class:`BudgetWall`."""
        if self.over_cap():
            if raise_on_over:
                raise BudgetWall(
                    f"1hr local-4060 cap hit at '{label}' "
                    f"(elapsed {self.elapsed():.1f}s ≥ cap {self.cap_seconds:.1f}s)"
                )
            return False
        return True


# ============================================================================== ESS_min a-priori RULE
# K=50 retained-window convention (see PINS).  ESS_hat = K / (2·τ_int,Y).
K_WINDOW = 50

# The a-priori minimum effective-sample count.  This is a SCIENTIFIC ACCEPTANCE threshold fixed
# BEFORE any joint/control comparison, NOT a calibration output (see design notes).
ESS_MIN_FLOOR = 10.0


def ess_min_rule(k_window: int = K_WINDOW, ess_floor: float = ESS_MIN_FLOOR) -> dict:
    """The a-priori ESS_min selection RULE (define-before-you-look).

    RULE
    ----
    The window-adequacy gate is ``worst-layer ESS_hat ≥ ESS_min`` with
    ``ESS_hat = K/(2·τ_int,Y)``, K=50.  ``ESS_min`` is the minimum number of *effective* (decorrelated)
    samples the K=50 window estimator must contain for its mean to have acceptable Monte-Carlo
    variance.  For an MCMC mean estimator the relative standard error scales as ``∝ 1/√ESS``; we
    require at least ``ESS_FLOOR = 10`` effective samples — the conventional statistical floor below
    which (i) the SE of the mean exceeds ~1/√10 ≈ 32 % of an i.i.d.-sd unit AND (ii) the
    autocorrelation-corrected variance ESTIMATE itself becomes unreliable (Sokal: need ≳10 effective
    samples to trust τ̂).  ``ESS_min`` is therefore ``ESS_FLOOR`` directly.

    This value is INDEPENDENT of any measured τ̂, of the smoke result, and of any joint-vs-control
    comparison.  It is pre-registered alongside PINS so the later H200 acceptance is reproducible.

    Equivalent τ ceiling (diagnostic only): ESS_hat ≥ ESS_min  ⟺  τ_int,Y ≤ K/(2·ESS_min).

    Returns
    -------
    dict with the frozen ``ESS_min``, the equivalent τ ceiling, and a one-line ``rule`` string.
    """
    ess_min = float(ess_floor)
    tau_ceiling = k_window / (2.0 * ess_min)
    return dict(
        ESS_min=ess_min,
        ess_floor=float(ess_floor),
        K_window=int(k_window),
        tau_ceiling=float(tau_ceiling),
        rule=(
            f"a-priori: ESS_min = ESS_FLOOR = {ess_min:g} effective samples "
            f"(min for SE≈1/√ESS acceptable AND τ̂ trustworthy, Sokal); "
            f"⟺ τ_int,Y ≤ K/(2·ESS_min) = {tau_ceiling:g}.  "
            f"Fixed before any joint/control comparison; NOT a calibration output."
        ),
    )


# ============================================================================== freeze-FROM-MEASUREMENT
# C — the trajectory-adequacy factor in the gate L_traj ≥ C·τ̂.  This is the SAME self-consistency
# constant the half-Sokal estimator uses (pp.SOKAL_C = 5.0): a retained process is "resolved" only
# when its length is ≥ C times the integrated autocorrelation time.  Frozen at the validated
# reference value (design notes, "Probe K=50 convention" gate (i): trajectory adequacy L_traj ≥ C·τ̂).
C_TRAJ_ADEQUACY = 5.0

# N_R — the fixed shared Rademacher sketch count.  Target ≈16 (design notes / brief): the worst-of-N_R
# screening reduction needs enough random projections that the projection standard error is small but
# few enough to stay cheap.  16 i.i.d. ±1 sketches give a worst-of-16 screen with the projection SE on
# any single sketch ~1/√(n_chains·L_traj) (the sketch itself is exact ±1, variance 1 per component);
# 16 is the fixed Rademacher screen count and the brief's target.
N_R_TARGET = 16


def freeze_from_measurement(
    tau_hat: float,
    *,
    c: float = C_TRAJ_ADEQUACY,
    n_r: int = N_R_TARGET,
    se_target: float = 0.05,
    k_window: int = K_WINDOW,
    stride: int = 8,
    l_traj_min: int = 200,
    l_traj_cap: Optional[int] = None,
) -> dict:
    """Turn a MEASURED τ̂ (half-Sokal, from the smoke calibration) into the frozen runtime/adequacy
    constants ``L_traj`` / ``N_chains`` / ``N_R`` / ``C`` (see design notes).

    Freeze rules (all FROM MEASUREMENT / adequacy):

    * ``C``  = trajectory-adequacy factor (gate (i): ``L_traj ≥ C·τ̂``).  Frozen at the validated
      self-consistency value ``C = pp.SOKAL_C = 5.0`` (NOT measured per-run; it is the estimator's
      own resolution constant — the MEASUREMENT confirms the chosen ``L_traj`` satisfies ``L_traj ≥ C·τ̂``).

    * ``L_traj`` ≫ τ̂ so ρ_Y(50) is estimable AND the white-noise autocorrelation SE ≪ 0.05.  The
      autocorr-at-lag SE for an (approximately) white retained process is ``1/√(N_eff)`` with
      ``N_eff = n_chains·L_traj``; but the binding constraint is *trajectory adequacy*: we require
      ``L_traj ≥ C·τ̂`` AND ``L_traj > K`` (so r_grad[50] is defined) AND a floor ``l_traj_min`` so the
      half-Sokal tail is resolved.  We round up to a power-of-two-friendly value ≥ all three.

    * ``N_chains`` so that the per-sketch projection SE ``≈ 1/√(n_chains·L_traj) ≤ se_target``.  Given
      the frozen ``L_traj`` we solve ``n_chains ≥ (1/se_target)² / L_traj`` and floor it at a small
      minimum (≥4) so the chain-axis vmap is non-degenerate.

    * ``N_R`` = ``n_r`` (target 16) — the fixed Rademacher screen count.

    Parameters
    ----------
    tau_hat : float
        The MEASURED half-Sokal τ̂ from the smoke calibration (the worst layer's tau_hat).
    se_target : float
        The white-noise autocorrelation SE we want ``L_traj`` (and ``N_chains·L_traj``) to beat.
    l_traj_min : int
        Absolute floor on ``L_traj`` (so the tail is resolved even at tiny τ̂).
    l_traj_cap : int, optional
        Optional ceiling (e.g. to fit the H200 budget); ``L_traj`` is min'd with it if set.

    Returns
    -------
    dict
        ``{L_traj, N_chains, N_R, C, tau_hat, justification}`` — each value with the measurement that
        justifies it in ``justification``.
    """
    tau_hat = float(tau_hat)
    if not math.isfinite(tau_hat) or tau_hat <= 0:
        raise ValueError(f"tau_hat must be a finite positive measurement; got {tau_hat!r}")

    # --- L_traj: ≥ C·τ̂  AND  > K  AND  ≥ l_traj_min;  then round UP to a multiple of K for clean ρ_Y(50)
    l_adequacy = c * tau_hat
    l_needed = max(l_adequacy, float(k_window) + 1.0, float(l_traj_min))
    # SE constraint on the retained length alone (n_chains adds more); 1/√L ≤ se_target ⇒ L ≥ 1/se²
    l_se = 1.0 / (se_target ** 2)
    l_needed = max(l_needed, l_se)
    # round up to a multiple of K (50) so the window tiles cleanly
    l_traj = int(math.ceil(l_needed / k_window) * k_window)
    if l_traj_cap is not None:
        l_traj = min(l_traj, int(l_traj_cap))

    # --- N_chains: per-sketch projection SE ≈ 1/√(n_chains·L_traj) ≤ se_target ⇒ n_chains ≥ (1/se²)/L
    n_eff_needed = 1.0 / (se_target ** 2)
    n_chains = int(math.ceil(n_eff_needed / l_traj))
    n_chains = max(n_chains, 4)  # non-degenerate chain-axis vmap

    return dict(
        L_traj=int(l_traj),
        N_chains=int(n_chains),
        N_R=int(n_r),
        C=float(c),
        tau_hat=float(tau_hat),
        justification=dict(
            L_traj=(
                f"L_traj={l_traj} ≥ C·τ̂={l_adequacy:.1f} (trajectory adequacy gate (i)) "
                f"AND > K={k_window} (ρ_Y(50) defined) AND ≥ 1/se²={l_se:.0f} "
                f"(white-noise autocorr SE ≤ {se_target}); rounded up to a multiple of K"
                + (f"; capped at {l_traj_cap}" if l_traj_cap is not None else "")
            ),
            N_chains=(
                f"N_chains={n_chains}: n_chains·L_traj={n_chains * l_traj} ≥ 1/se²={n_eff_needed:.0f} "
                f"(projection SE ≈ 1/√(n_chains·L_traj) ≤ {se_target}); floored at 4 for a "
                f"non-degenerate chain-axis vmap"
            ),
            N_R=f"N_R={n_r}: fixed Rademacher screen count (≈16)",
            C=(
                f"C={c}: trajectory-adequacy factor = the half-Sokal self-consistency constant "
                f"(pp.SOKAL_C); MEASUREMENT confirms L_traj ≥ C·τ̂"
            ),
            tau_hat=f"τ̂={tau_hat:.3f} (MEASURED half-Sokal, worst layer; ~0.86× exact, systematic)",
        ),
    )
