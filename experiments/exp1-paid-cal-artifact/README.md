# exp1 — paid H200 Stage-C run → `Q-CALIBRATION-FAIL` (cal-gate artifact)

**Companion-local experiment numbering** (distinct from the parent research project's own experiment numbering).
This directory archives the **first** paid Stage-C run so its record is not overwritten by the exp2 re-run.

## Verdict

`Q-CALIBRATION-FAIL` on **both** seeds — a **calibration-GATE artifact, NOT bad mixing**. The
joint-steering hypothesis (Stage C) was **never reached**; this run produced **no result on the actual
hypothesis**.

## What happened

| | |
|---|---|
| Run params | `MODE=full BUDGET_H=16.0 SEEDS=1,2` |
| Code SHA (executed) | `57eac01` (`provenance.git_sha` in `run_stage_c.json`) |
| Results committed at | `5b9cbbc` (force-added past the `results/` gitignore) |
| Wall | 9.29 GPU-h (studio `htdml` / teamspace `TEAMSPACE/htdml`, since STOPPED) |
| Backend | `gpu`, `is_patch_live=true` |

Per-layer τ̂ (tiny → fast-mixing chains):
- seed 1: `[2.060, 2.459, 2.280, 5.071]`, `failed_layer=0`
- seed 2: `[1.971, 3.995, 3.195, 1.418]`, `failed_layer=0`

All layers were self-consistent ~16× over the `L_traj ≥ C·τ̂` adequacy (Stage-B autocorr ~0.00–0.05),
yet `cal_stable=False` at the **fastest** layer (τ̂≈2).

## Root cause

`harness/probe_primitives.py::classify_calibration_stable` (as of `57eac01`) used **pure relative**
doubling tolerances (`rel_tau=|Δτ|/τ < 0.15`, `dT=|ΔT_O|/T_O < 0.15`). At τ̂≈2 these are dominated by
estimation noise (a 2.0→2.5 jitter = 25% > 15%) → never 2 consecutive STABLE rungs → false fail.
Reproducible across both seeds ⇒ **systematic** (criterion × architecture), the **opposite** failure mode
to stickiness. Mirrors a related internal experiment (a gate-spec artifact). The per-rung doubling curve was **not
persisted** by this run (`RealOps.calibrate_tau` kept only `tau_hat`/`cal_stable`), so the failing axis
could not be pinpointed from the JSON.

## Files
- `run_stage_c.json` — the full run record (outcome, per-seed τ̂, provenance, budget).
- `report.md` — the one-line outcome report.
- `run.log` — Stage-B ACP autocorrelations across all 200 epochs (no calibration curve).

## → exp2
The repair + re-run is `../exp2-cal-gate-fix/` (regime-aware cal-gate floor `TAU_ABS_FLOOR`, pre-registered
before the re-run; the run now also **persists** the per-rung curve). MEASURE-ONLY — companion-local
tokens only.
