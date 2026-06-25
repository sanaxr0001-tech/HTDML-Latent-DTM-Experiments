"""scripts/calibrate_epoch_cost.py — Task-12 calibration: measure epoch cost + trajectory adequacy
on the local RTX 4060, then FREEZE L_traj / N_chains / N_R / C from MEASUREMENT and set ESS_min by
the a-priori rule, writing PINS.md + pre-commitment.md.

Researcher-conferred scope (build-notes §"TASK-12 SCOPE"):
  * Freeze L_traj / N_chains / N_R / C FROM MEASUREMENT (runtime/adequacy):
      - measure the per-epoch wall cost (for the later H200 budget estimate in p0_decision);
      - run the probe's T_O doubling-stability calibration (the exp16-faithful
        ``classify_calibration_stable``) on the smoke DTM to MEASURE τ̂;
      - freeze L_traj (≫ τ̂; white-noise autocorr SE ≪ 0.05), N_chains, N_R (≈16), C (L_traj ≥ C·τ̂).
  * ESS_min is NOT a calibration output — set by the a-priori RULE (``calib_logic.ess_min_rule``),
    fixed BEFORE any joint/control comparison (there is none here).
  * 1hr wall cap on the 4060 (shared with the smoke when run together).

This is the freeze step; it ALSO runs a minimal encode→fit so the calibration sees a TRAINED DTM
(the stale-factors refresh guard requires trained-≠-init weights).  NO Stage-C / joint / control /
two-seed.

Usage::

    python scripts/calibrate_epoch_cost.py            # build+train+calibrate+freeze
    python scripts/calibrate_epoch_cost.py --no-write # measure + print, do NOT touch PINS
"""

from __future__ import annotations

import argparse
import functools
import sys
import time
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
for _p in (str(_REPO_ROOT), str(_REPO_ROOT / "src")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import htdml.paths  # noqa: E402

htdml.paths.bootstrap_paths()

import jax  # noqa: E402
import jax.random as jr  # noqa: E402
import numpy as np  # noqa: E402

from scripts import calib_logic as cl  # noqa: E402
from scripts import smoke_common as sm  # noqa: E402

PINS_MD_PATH = _REPO_ROOT / "PINS.md"
PRECOMMIT_MD_PATH = _REPO_ROOT / "pre-commitment.md"


def _banner(msg):
    print("\n" + "=" * 70 + f"\n{msg}\n" + "=" * 70, flush=True)


# ============================================================================== build + train + measure
def build_and_train(*, epochs, n_train, ae_steps, seed, wc):
    """Encode + build the real 44_12 DTM + dtm.train (few epochs).  Returns (ldtm, dtm, per_epoch_s)."""
    _banner("CALIBRATE — encode + fit (trained DTM for the refresh-guarded calibration)")
    tr_img, tr_lab, te_img, te_lab = sm.load_fashion_mnist(n_train=int(n_train))
    ae_params, ae_losses = sm.pretrain_autoencoder(
        tr_img, key=jr.PRNGKey(seed + 1), n_steps=int(ae_steps), batch_size=64, lr=1e-3)
    from htdml.autoencoder import encode as ae_encode
    encode_fn = functools.partial(ae_encode, ae_params)
    latent_ds = sm.build_latent_dataset(
        encode_fn, tr_img, tr_lab, te_img, te_lab,
        target_classes=sm.SMOKE_TARGET_CLASSES, num_label_spots=sm.SMOKE_NUM_LABEL_SPOTS)
    dtm = sm.build_companion_dtm(latent_ds, seed=seed)
    ldtm = sm.LatentDTM(dtm, decode_fn=sm.make_decode_fn(ae_params))

    if wc.would_exceed(60.0):
        raise cl.BudgetWall("declining FIT in calibrate — would exceed cap")
    t0 = time.time()
    ldtm.fit(latent_ds, n_epochs=int(epochs), evaluate_every=0)
    per_epoch = (time.time() - t0) / max(int(epochs), 1)
    print(f"  measured per-epoch fit cost: {per_epoch:.2f}s/epoch ({epochs} epochs)")
    return ldtm, dtm, latent_ds, per_epoch


def measure_tau_hat(ldtm, latent_ds, *, wc, n_chains, L0, warm, n_rungs, diag_key=20240624):
    """Run the per-layer T_O doubling-stability calibration on EVERY layer; return the WORST-layer
    (max) measured τ̂ + the per-layer calibration records.  exp16-faithful classify_calibration_stable."""
    _banner("CALIBRATE — probe T_O doubling-stability (MEASURE τ̂ per layer)")
    from htdml.trainability_probe import TrainabilityProbe
    train_ds, _test_ds, _ohtl = latent_ds
    batch = dict(image=train_ds["image"], label=train_ds["label"], idx=0)
    probe = TrainabilityProbe()
    per_layer = []
    tau_hats = []
    for layer in range(len(ldtm.dtm.steps)):
        wc.checkpoint(f"calib layer {layer}", raise_on_over=True)
        calib = probe.calibrate(ldtm, layer=layer, batch=batch, n_chains=int(n_chains),
                                L0=int(L0), warm=int(warm), n_rungs=int(n_rungs), diag_key=diag_key)
        print(f"  layer {layer}: tau_hat={calib['tau_hat']:.3f} T_O={calib['T_O']:.3g} "
              f"cal_stable={calib['cal_stable']} failed_axis={calib.get('failed_axis')}")
        per_layer.append({"layer": layer, "tau_hat": float(calib["tau_hat"]),
                          "T_O": float(calib["T_O"]), "cal_stable": bool(calib["cal_stable"]),
                          "failed_axis": calib.get("failed_axis")})
        tau_hats.append(float(calib["tau_hat"]))
    tau_hat_worst = float(max(tau_hats))
    print(f"  worst-layer measured τ̂ = {tau_hat_worst:.3f}")
    return tau_hat_worst, per_layer


# ============================================================================== freeze + PINS write
def _pins_calib_rows(frozen, ess):
    """Build the 5 per-key PINS rows in the format the zero-compute battery parser accepts (each key a
    whole-token Item-column cell; numeric in the Status column; NO 'TBD' anywhere in the Status cell)."""
    j = frozen["justification"]
    rows = [
        f"| L_traj | {frozen['L_traj']} (frozen Task-12 from MEASUREMENT: {j['L_traj']}) |",
        f"| N_chains | {frozen['N_chains']} (frozen Task-12 from MEASUREMENT: {j['N_chains']}) |",
        f"| N_R | {frozen['N_R']} (frozen Task-12: {j['N_R']}) |",
        f"| C | {frozen['C']} (frozen Task-12: {j['C']}) |",
        f"| ESS_min | {ess['ESS_min']} (a-priori RULE, NOT measured: {ess['rule']}) |",
    ]
    return rows


_CALIB_KEYS = ("L_traj", "N_chains", "N_R", "C", "ESS_min")


def _row_item_key(line):
    """If ``line`` is a TBD-section table row whose Item column is exactly one calibration key, return
    that key; else None.  Whole-token match (so 'C' ∉ 'N_chains'/'SOKAL_C') — mirrors the battery parser."""
    import re as _re
    s = line.strip()
    if not s.startswith("|"):
        return None
    cells = [c.strip() for c in s.strip("|").split("|")]
    if not cells:
        return None
    item = cells[0]
    for k in _CALIB_KEYS:
        if _re.search(r"(?:^|[,|\s(]){}(?:[,|\s)]|$)".format(_re.escape(k)), item):
            return k
    return None


def write_pins(frozen, ess):
    """Replace the calibration-constant placeholder(s) in PINS.md with 5 per-key rows carrying frozen
    numerics.  Handles both the initial single COMBINED placeholder row AND an idempotent re-run (per-
    key rows already present).  FULLY removes 'TBD' from the calibration line(s) (brief warning)."""
    text = PINS_MD_PATH.read_text()
    rows = _pins_calib_rows(frozen, ess)
    row_for = dict(zip(_CALIB_KEYS, rows))

    out = []
    in_tbd = False
    seen = set()
    combined_replaced = False
    for line in text.splitlines():
        s = line.strip()
        if "TBD-at-step" in s:
            in_tbd = True
            out.append(line)
            continue
        if in_tbd and s.startswith("##"):
            # leaving the TBD section: emit any not-yet-written calibration rows first
            for k in _CALIB_KEYS:
                if k not in seen:
                    out.append(row_for[k])
                    seen.add(k)
            in_tbd = False
            out.append(line)
            continue
        if in_tbd and s.startswith("|"):
            # combined placeholder row (mentions all five names) → drop, rows emitted on section exit
            if "L_traj" in s and "ESS_min" in s and "N_chains" in s:
                combined_replaced = True
                continue
            k = _row_item_key(line)
            if k is not None:                      # idempotent: replace a per-key row in place
                if k not in seen:
                    out.append(row_for[k])
                    seen.add(k)
                continue
        out.append(line)

    # if the TBD section was the last section (no trailing '##'), flush remaining rows
    if in_tbd:
        for k in _CALIB_KEYS:
            if k not in seen:
                out.append(row_for[k])
                seen.add(k)

    PINS_MD_PATH.write_text("\n".join(out) + ("\n" if text.endswith("\n") else ""))
    print(f"  PINS.md updated (combined_replaced={combined_replaced}, rows_written={sorted(seen)}).")


def write_precommitment(frozen, ess, per_layer, per_epoch_s, *, run_args):
    """Write pre-commitment.md recording the frozen constants + the ESS_min RULE + the justifying
    measurements (so the freeze is reproducible; Task 13 finalizes the full pre-commitment)."""
    layers_md = "\n".join(
        f"  - layer {p['layer']}: τ̂={p['tau_hat']:.3f}, T_O={p['T_O']:.3g}, "
        f"cal_stable={p['cal_stable']}" for p in per_layer)
    content = f"""# pre-commitment.md — htdml-latent-dtm companion (Task-12 calibration freeze)

> Task 12 records the calibration freeze here so the later H200 acceptance is reproducible.
> **Task 13 finalizes the full pre-commitment** (predictions + the full go/no-go criteria); this file
> currently holds ONLY the Task-12-frozen acceptance constants + the a-priori ESS_min RULE.
>
> Researcher-conferred Task-12 scope (build-notes §"TASK-12 SCOPE"): NO Stage-C joint/control, NO
> two-seed, NO outcome token, NO H200; 1hr local-4060 cap; freeze L_traj/N_chains/N_R/C from
> MEASUREMENT; ESS_min by an a-priori RULE fixed BEFORE any joint/control comparison.

## Frozen-from-MEASUREMENT acceptance constants (local 4060)

| Constant | Value | Justifying measurement |
|----------|-------|------------------------|
| L_traj   | {frozen['L_traj']} | {frozen['justification']['L_traj']} |
| N_chains | {frozen['N_chains']} | {frozen['justification']['N_chains']} |
| N_R      | {frozen['N_R']} | {frozen['justification']['N_R']} |
| C        | {frozen['C']} | {frozen['justification']['C']} |

Measured worst-layer half-Sokal τ̂ = {frozen['tau_hat']:.3f}
({frozen['justification']['tau_hat']}).

Per-layer T_O doubling-stability calibration (exp16-faithful `classify_calibration_stable`):
{layers_md}

Measured per-epoch fit cost on the 4060: {per_epoch_s:.2f} s/epoch
(used for the later H200 budget estimate in p0_decision).

Calibration run config: {run_args}

## a-priori ESS_min RULE (NOT a calibration output)

**ESS_min = {ess['ESS_min']:g}.**

RULE — {ess['rule']}

ESS_min is a SCIENTIFIC ACCEPTANCE threshold, fixed BEFORE any joint/control comparison (there is no
joint/control arm in Task 12).  It is NOT reverse-engineered from any smoke / joint / control result.
The window-adequacy gate is `worst-layer ESS_hat ≥ ESS_min` with `ESS_hat = K/(2·τ_int,Y)`, K=50.

## Bias caveat (build-notes §"Half-Sokal T_O bias")

The half-Sokal τ̂/T_O is systematically ~0.86× the exact value.  The ABSOLUTE bars (ESS_min, C) are
frozen against the SAME biased estimator → self-consistent; the companion only ever uses τ̂/Q as a
RELATIVE ruler/guard, so the systematic bias cancels in the (joint-vs-control) gates.
"""
    PRECOMMIT_MD_PATH.write_text(content)
    print(f"  pre-commitment.md written ({PRECOMMIT_MD_PATH}).")


# ============================================================================== main
def run_calibration(*, epochs, n_train, ae_steps, seed, cap_seconds,
                    n_chains, L0, warm, n_rungs, write=True):
    wc = cl.WallClock(cap_seconds=float(cap_seconds))
    ldtm, dtm, latent_ds, per_epoch = build_and_train(
        epochs=epochs, n_train=n_train, ae_steps=ae_steps, seed=seed, wc=wc)
    tau_hat, per_layer = measure_tau_hat(
        ldtm, latent_ds, wc=wc, n_chains=n_chains, L0=L0, warm=warm, n_rungs=n_rungs)

    _banner("CALIBRATE — freeze L_traj/N_chains/N_R/C from MEASUREMENT + a-priori ESS_min")
    frozen = cl.freeze_from_measurement(tau_hat=tau_hat)
    ess = cl.ess_min_rule()
    print(f"  frozen: L_traj={frozen['L_traj']} N_chains={frozen['N_chains']} "
          f"N_R={frozen['N_R']} C={frozen['C']}")
    print(f"  a-priori ESS_min={ess['ESS_min']}  (τ ceiling {ess['tau_ceiling']})")

    if write:
        write_pins(frozen, ess)
        write_precommitment(frozen, ess, per_layer, per_epoch,
                            run_args=dict(epochs=epochs, n_train=n_train, n_chains=n_chains,
                                          L0=L0, warm=warm, n_rungs=n_rungs))
    else:
        print("  --no-write: NOT touching PINS.md / pre-commitment.md")

    return {"frozen": frozen, "ess_min": ess, "per_layer": per_layer,
            "per_epoch_seconds": per_epoch, "tau_hat": tau_hat, "wall_seconds": wc.elapsed()}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--n-train", type=int, default=600)
    ap.add_argument("--ae-steps", type=int, default=80)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--cap-seconds", type=float, default=3600.0)
    ap.add_argument("--n-chains", type=int, default=8, help="calibration chains per rung")
    ap.add_argument("--L0", type=int, default=100, help="initial doubling-curve length")
    ap.add_argument("--warm", type=int, default=200, help="calibration warmup")
    ap.add_argument("--n-rungs", type=int, default=4, help="doubling rungs")
    ap.add_argument("--no-write", action="store_true", help="measure + print only; do not touch PINS")
    args = ap.parse_args()

    _banner("Task 12 — local-4060 CALIBRATION (epoch cost + trajectory-adequacy freeze)")
    try:
        res = run_calibration(epochs=args.epochs, n_train=args.n_train, ae_steps=args.ae_steps,
                              seed=args.seed, cap_seconds=args.cap_seconds, n_chains=args.n_chains,
                              L0=args.L0, warm=args.warm, n_rungs=args.n_rungs, write=not args.no_write)
    except cl.BudgetWall as e:
        _banner(f"CALIBRATION — BUDGET-WALL STOP (not a failure): {e}")
        sys.exit(2)

    _banner("CALIBRATION COMPLETE")
    print(f"frozen: {res['frozen']['L_traj']=} {res['frozen']['N_chains']=} "
          f"{res['frozen']['N_R']=} {res['frozen']['C']=}  ESS_min={res['ess_min']['ESS_min']}")
    print(f"wall used: {res['wall_seconds']:.1f}s")
    sys.exit(0)


if __name__ == "__main__":
    main()
