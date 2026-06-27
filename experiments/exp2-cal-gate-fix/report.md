# exp2 — cal-gate repair + re-run → `HTDML-MARGIN-NEGATIVE` (REPORT)

**RAN on H200 2026-06-27 → `HTDML-MARGIN-NEGATIVE` (both seeds; measurement-VALID).** Code `a26dbce`,
`MODE=full BUDGET_H=16.0 SEEDS=1,2 OUTDIR=experiments/exp2-cal-gate-fix/artifacts`, **11.78 GPU-h**, no
budget wall. MEASURE-ONLY — no wiki tag. Raw artifacts in `artifacts/` (`run_stage_c.json`, `report.md`,
`run.log`).

## Two results — keep them separate

**1. The measurement/engineering win (the point of exp2) — CONFIRMED.** The `TAU_ABS_FLOOR` cal-gate
fix worked, and robustly: **both** independently-trained seeds passed the calibration gate at the trained
τ̂ (seed-1 worst τ̂=2.42, seed-2 worst τ̂=3.67; `cal_stable=True` both), so **Stage C ran for both** — the
joint-steering test exp1 never reached (exp1 died at the broken gate on the *same* regime). The Stage-B
checkpoint persistence also worked end-to-end (both `checkpoints/seed{1,2}/stage_b/` written). So exp1's
`Q-CALIBRATION-FAIL` is confirmed to have been a gate artifact, now repaired; the probe's measurements are
admissible and were consumed by Stage C.

**2. The science verdict — NEGATIVE.** With a valid measurement, the guarded joint-training objective at
λ=1.0 **did not clear the mixing-margin gate** vs the matched (λ=0) control. The per-update reject gate
repeatedly fired **`Q_drop`** — the joint arm's lower-quartile `Q` fell *below* the control's (it never
reached the required ≥1.25× improvement; if anything the steering slightly *degraded* the margin). Image
quality was preserved (joint ≈ control on BCE and FID), so this is **not** `QUALITY-LOSS` — it is
specifically a *margin* failure.

## Per-seed

| | seed 1 | seed 2 |
|---|---|---|
| token | `HTDML-MARGIN-NEGATIVE` | `HTDML-MARGIN-NEGATIVE` |
| cal_stable | True (τ̂ layers ≈ [2.05, 1.99, 1.87, 2.42]) | True (τ̂ layers ≈ [2.22, 3.67, 3.13, 0.97]) |
| BCE joint / control | 0.3044 / 0.3023 | 0.3045 / 0.3049 |
| FID joint / control | 306.5 / 304.6 | 303.5 / 290.7 |
| Stage-C reject loop | 10 blocks, 2 rejects (blocks 0, 4 — `Q_drop`), 8 accepted | 4 blocks, 3 rejects (0, 2, 3 — `Q_drop`); blocks 2–3 consecutive → early stop |

Both seeds: every rejection was `Q_drop` (joint lower-quartile Q < 0.90× control). e.g. seed-1 block-0:
joint 1.987 < 0.90×2.486; seed-2 block-3: joint 1.358 < 0.90×2.22. The improvement bar (joint ≥ 1.25×
control Q, or τ_int,Y ≤ 0.75× control) was never met for either seed → run token `HTDML-MARGIN-NEGATIVE`
(POSITIVE requires BOTH seeds to pass; neither did).

## Prediction vs outcome (vs `pre-commitment.md`)

- **Pre-registered prediction (exp2):** "cal gate now passes at trained τ̂≈2 → Stage C runs → a real
  joint-steering verdict." → **CONFIRMED** (empirically measured): both seeds passed cal, Stage C ran,
  a measurement-valid verdict was produced. `Q-CALIBRATION-FAIL` did not recur.
- **The joint-steering hypothesis** (joint training retains quality *and* holds/improves a finite-budget
  mixing margin over the matched control): **NOT SUPPORTED here.** Quality retained ✓; margin held/improved
  ✗ (steering tends to drop Q).

## Interpretation + scope

A clean, informative negative — the experiment's actual question got a real answer for the first time
(exp1 couldn't even ask it). On this setup the `Q`-guarded free-energy-compat steering does **not** yield
a mixing-margin benefit; it slightly *degrades* the margin while leaving quality intact. This is **scoped**
— feasibility/signal study, **n=2 seeds, λ=1.0, 44_12 DTM, Fashion-MNIST, the deterministic mean-field
compat surrogate.** It is **not** a refutation of the broader HTDML idea, and **not** a wiki-theorem
result (companion-local token, MEASURE-ONLY).

## Next (cheap now — checkpoints persisted)

The iteration ladder is now ~1–3 GPU-h per attempt (both Stage-B checkpoints are saved → `RESUME_FROM`
skips Stage A+B): sweep **λ** (1.0 may over-steer → the `Q_drop`; smaller λ may find a regime that holds
the margin), revisit the **compat objective / guard threshold**, or accept the negative as the scoped
answer. Researcher's call.

## Provenance
Code `a26dbce`; studio `s_STUDIO` (htdml / TEAMSPACE/htdml), stopped after the run.
Constants: λ_joint=1.0, ESS_min=10, C=5.0, L_traj=400, N_chains=4, N_R=16. 11.78 GPU-h. MEASURE-ONLY,
no wiki tag move.
