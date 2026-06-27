# exp3 — lower-λ sweep → `HTDML-MARGIN-NEGATIVE` (no demonstrable steering effect; apparent per-seed pass is control-denominator noise)

**RAN on H200 2026-06-27.** Resume sweep over `λ ∈ {0.5, 0.3, 0.1}` on exp2's persisted Stage-B
checkpoints (`RESUME_FROM`, Stage A+B skipped), code `19cd46f`, `MODE=full SEEDS=1,2`; all three runs
finished clean, **no budget wall**. Sweep total **5.67 GPU-h** (2.181 + 2.200 + 1.287) of the conferred
9.0 backstop. Plotted against exp2's `λ=1.0` anchor (code `a26dbce`). **MEASURE-ONLY — companion-local
tokens, no wiki tag.** Raw: `artifacts/lam{0.5,0.3,0.1}/run_stage_c.json` + `.run.log`.

> **Interpretation note (publication-grade adversarial verification, 2026-06-27).** An earlier draft of
> this report read the per-seed gate passes as a "reproducible τ-route mixing improvement." A 4-lens
> adversarial verification against the raw JSONs **refuted that interpretation** (numbers and provenance
> reproduced exactly; the *mechanism narrative did not*). This version states the corrected, defensible
> conclusion: **run-level negative, and the apparent per-seed clearance is not distinguishable from
> control-denominator noise.** The retraction is recorded deliberately rather than silently revised.

---

## Headline

Pre-registered question (`pre-commitment.md`): **does any λ yield a robust (both-seed) margin
clearance — improvement *and* ESS-non-degradation — where λ=1.0 (exp2) did not?**

**Answer: no.** Run-level **`HTDML-MARGIN-NEGATIVE` at every λ** (`two_seed.both_pass=False` for all four).
The driver emitted a per-*seed* `HTDML-MARGIN-POSITIVE` token for **seed 1 at λ=0.3 and λ=0.1** (its τ-leg
gate passed), but **this is not evidence of a steering effect** — it is consistent with noise in the
worst-layer control denominator of the gate ratio (§"Why the per-seed pass is not a steering effect").
**Conclusion: no demonstrable effect of the mixing-aware steering on the finite-budget mixing margin, at
any λ in the ladder.** Both improvement routes (Q and τ) are null once the gate's estimator variance is
accounted for.

---

## The measured verdict — per-λ × per-seed × 7-criterion gate

Gate (inherited verbatim from `../../pre-commitment.md`, registered a-priori): a seed PASSES iff all 7
hold — quality (BCE ≤ control+5%, FID ≤ control+10%), trajectory resolved (`L_traj ≥ C·τ̂`, all
`cal_stable`), ESS adequate (worst ESS ≥ 10), plateau (`|r_grad[50]| ≤ 0.05`), **improvement**
(lower-quartile joint `Q_struct^⊥` ≥ **1.25×** control **OR** worst-layer `τ_int,Y` ≤ **0.75×** control),
**ESS-non-degradation** (worst joint ESS ≥ control). A *run* is POSITIVE iff BOTH seeds pass.

| λ | seed | token | lq-Q ratio (≥1.25?) | worst-τ ratio (≤0.75?) | worst-ESS j/c (nondeg?) | quality held? |
|---|---|---|---|---|---|---|
| **1.0** | 1 | NEGATIVE | 0.93× ✗ | 1.04× ✗ | 20.1/20.9 ✗ | ✓ |
| (exp2) | 2 | NEGATIVE | 1.08× ✗ | 0.94× ✗ | 25.0/23.5 ✓ | ✓ |
| **0.5** | 1 | NEGATIVE | 1.04× ✗ | 0.99× ✗ | 27.4/27.3 ✓ | ✓ |
|  | 2 | NEGATIVE | 1.33× ✓ | 1.16× ✗ | 18.3/21.2 ✗ | ✓ |
| **0.3** | 1 | POSITIVE¹ | 0.99× ✗ | 0.61× ✓ | 17.9/10.9 ✓ | ✓ |
|  | 2 | NEGATIVE | 1.07× ✗ | 0.88× ✗ | 16.6/14.7 ✓ | ✓ |
| **0.1** | 1 | POSITIVE¹ | 0.91× ✗ | 0.51× ✓ | 22.6/11.6 ✓ | ✓ |
|  | 2 | NEGATIVE | 1.01× ✗ | 0.85× ✗ | 23.7/20.3 ✓ | ✓ |

¹ **per-seed token only; both runs are run-level NEGATIVE** (seed 2 fails). The seed-1 passes are
analyzed — and discounted — below. (`τ_int,Y` is the half-Sokal estimator; all values are the final
accepted Stage-C state, lower-quartile / worst-layer per the gate definitions.)

---

## Why the per-seed pass is **not** a steering effect — control-denominator noise

The τ-leg is `max₄(joint τ) / max₄(control τ) ≤ 0.75×`. The λ=0 control arm is forked from the *same*
Stage-B checkpoint and is the *same condition* across all of a seed's runs, so its worst-layer `τ_int,Y`
should be ~one number per seed. It is not — and the "passes" ride entirely on its excursions:

| λ | seed-1 **joint** (steered) worst-τ | seed-1 **control** (λ=0) worst-τ | ratio | control worst-ESS |
|---|---|---|---|---|
| 1.0 | 1.242 | 1.198 | 1.04× | 20.9 |
| 0.5 | 0.911 | 0.917 | 0.99× | 27.3 |
| 0.3 | 1.394 | **2.293** | 0.61× | **10.9** |
| 0.1 | 1.107 | **2.152** | 0.52× | **11.6** |

Six independent reasons the seed-1 passes are noise, not signal:

1. **The control denominator is non-reproducible.** Seed-1 λ=0 control worst-τ swings **0.92 → 2.29 →
   2.15** (~2.5×) across identical-code runs — for a quantity that should be fixed. The two passes are
   exactly the two runs where it spiked high.
2. **The steered numerator shows no λ-trend.** Seed-1 *joint* worst-τ is **1.24 / 0.91 / 1.39 / 1.11** —
   flat and non-monotone; it is *worse* at λ=0.3 (1.39) than at λ=1.0 (1.24). The ratio fell because the
   denominator rose, not because the steered chain mixed faster.
3. **The passing runs have anomalously poorly-mixed control arms.** Control worst-ESS bottoms at **10.9
   / 11.6**, just above the `ESS_min=10` floor, precisely at λ=0.3/0.1 — the low-ESS ⇔ high-τ signature
   of a bad control draw. (This is the *same* signature used to disqualify the λ=0.5 seed-2 Q-route
   "lift", whose control ESS was degraded — applied symmetrically, it disqualifies the τ passes too.)
4. **`max₄` compares different physical layers.** At λ=0.3 the slow control layer is layer-3 (2.29; joint
   there 0.67) but joint layer-1 *rose* (1.16→1.39); at λ=0.1 the slow control layer is layer-1, with
   other joint layers rising. A different layer drives the max each run — a noisy-max artifact, not a
   reproducible per-layer acceleration.
5. **The reliable mixing measure inverts the story.** The deterministic reconfirm `τ̂` (bit-identical
   across all runs) is **seed-1 = 2.42, seed-2 = 3.67** → **seed 2 is the slower-mixing seed**, the
   opposite of any "seed 1 is slow / has headroom" reading.
6. **Multiple comparisons.** The OR-gate over (Q, τ) × 4 λ × 2 seeds is up to 16 leg-tests on a
   high-variance worst-of-4 ruler; two sub-0.75 τ ratios (both seed-1, both lowest λ) are well within
   what denominator noise produces, uncorrected for multiplicity.

**Both routes are null.** Q-route: relaxes to control (seed-2 lq-Q 1.33×→1.07×→1.01×); its single
apparent clearance (λ=0.5 seed-2, 1.33×) coincides with ESS degradation (18.3 < 21.2). τ-route:
control-denominator noise, above. **No λ produces a clearance attributable to the steering.**

---

## Methodological takeaway (the durable contribution)

At these probe lengths (`L_traj=400`, `K=50`, `B=400`), the **worst-layer-max `τ_int,Y` ratio is
noise-dominated**: the λ=0 control's `max₄ τ` varies ~2.5× run-to-run at fixed seed. The relative-ruler
discipline (`../../pre-commitment.md §Half-Sokal T_O bias`) correctly cancels the multiplicative *bias*
of the half-Sokal estimator — but a constant bias cancels in a ratio while **estimator variance does
not**, and the gate uses a *max over 4 short-chain estimates* (variance-amplifying). **Per-seed gate
passes are therefore uninterpretable as effects without a control-variance estimate.** The minimal fixes
for any future effect claim: (a) repeated λ=0 control draws at fixed seed to measure the worst-τ
run-to-run variance; (b) longer trajectories (`L_traj ≫ 400`) to stabilize `max₄`; (c) paired per-layer
τ tests instead of a worst-of-4 ratio; then (d) `n > 2` seeds. This reliability caveat is the
transferable result of exp3.

---

## Prediction vs outcome (vs `pre-commitment.md`)

- **Pre-registered modal prediction:** channel-negative ("no λ clears the margin"). → **CONFIRMED, and
  more strongly** — not only no robust positive, but no demonstrable per-seed effect: the apparent
  passes are control-denominator noise. The registered Q-centric reasoning under-specified the failure
  (it predicted Q→control; the τ-leg also yields only noise), but the **direction (negative) was correct.**
- **Decisive-either-way clause:** "any λ clears on BOTH seeds → POSITIVE; none → channel negative." →
  **Channel-negative**, with the per-seed positives explicitly attributed to noise after adversarial
  verification. No acceptance bar moved; frozen-five and the 7-criterion predicate unchanged.

---

## Provenance / reproducibility

| Item | Value |
|---|---|
| Code (exp3 runs) | `19cd46f` — `LAMBDA_JOINT` knob (`94046ba`) + two resume fixes found on the first GPU exercise of the resume path and validated on the local RTX 4060 before re-running: `RESUME_FROM` path semantics (`19cd46f`) and `np`-import in `load_stage_b` (`aa6c38f`) |
| Code (λ=1.0 anchor) | `a26dbce` (exp2) — **note: the λ=1.0 point is a different code version than the exp3 points** |
| Hardware / backend | NVIDIA H200 (143 GB) · jax 0.10.2 (`gpu`) · `is_patch_live=true` · env `env-h200-freeze.txt` |
| Stage-B source | exp2's checkpoints, **copied self-contained** into `artifacts/checkpoints/` (decoupled from exp2); `RESUME_FROM=experiments/exp3-lambda-sweep/artifacts` |
| Resume caveat | checkpoints trained under the **pre-fix** cal-gate (`a26dbce`); Stage-C **recomputes** `calibrate_tau` under the fixed gate (manifest `cal_gate_note`). All layers `cal_stable=True` at trained τ̂ (seed-1 worst 2.42, seed-2 worst 3.67); `L_traj=400 ≥ C·τ̂` |
| Dataset / FID | Fashion-MNIST split sha256 `9e7e9929…` · InceptionV3-FID pickle sha256 `4e030efa…` (offline, PINS-verified) |
| Config | `44_12` DTM, 4 reverse steps, ACP; frozen-five `L_traj=400, N_chains=4, N_R=16, C=5.0, ESS_min=10.0`; probe `K=50, B=400, s=8` |
| Budget | λ=0.5 **2.181**, λ=0.3 **2.200**, λ=0.1 **1.287** GPU-h, all `budget_wall=False`. λ=0.1 launched at `BUDGET_H=4.0` (pre-result amendment, `p0_decision.md §Pre-result amendment`) but finished at **1.29 GPU-h → clean under the original 3.0 cap** (4.0 backstop unused) |
| Pre-flight (on-box) | dataset sha ✓ · zero-compute battery 43/43 ✓ · reversible cert 10/10 ✓ · `LAMBDA_JOINT` knob test 8/8 ✓ |
| Verification | 4-lens adversarial pass (numbers / provenance / claims / scope), all numbers + provenance reproduced from raw JSONs; the claims+scope lenses drove this report's corrected interpretation |

---

## Scope & limitations (front and center)

- **No demonstrable effect.** The headline result is a **negative**; the per-seed gate passes are a
  measured curiosity attributable to control-denominator noise, not a steering signal.
- **Estimator-variance limit.** The gate's worst-layer-max τ ratio is noise-dominated at `L_traj=400`;
  this, not seed biology, governs the per-seed passes (see Methodological takeaway).
- **n = 2 seeds** — clearance *rate* unestimable regardless; but here even the per-seed signal is null.
- **Degenerate absolute-quality regime:** absolute FID ≈ 280–309 in *both* arms; "quality held" means
  joint ≈ control (BCE within 5%, FID within 10%), **not** that images are good. Relevant when judging
  whether τ differences are physically meaningful.
- **Companion-divergent setup:** `44_12` (not upstream `60_12`), Fashion-MNIST, the **deterministic
  mean-field compatibility surrogate**, 200-epoch Stage-B; λ=1.0 anchor is a different code version.
- **Two resume bugs** (`RESUME_FROM` path, `np` import) were found on the first GPU exercise and fixed +
  locally validated before the reported runs — disclosed for reproducibility.
- **MEASURE-ONLY:** companion-local tokens; **not** operational validation of the wiki theorem; **no
  hardware-energy claim**; no wiki tag move. Terminal verdict is **researcher-conferred**.

---

## Next (separate, conferred)

The result does not motivate an immediate follow-up *effect* experiment; it motivates a **measurement-
reliability** step first. Before any steering-effect claim: (1) quantify the λ=0 control worst-τ
run-to-run variance (repeated control draws at fixed seed); (2) lengthen trajectories (`L_traj ≫ 400`) to
stabilize the worst-of-4-layers estimator; (3) move to a paired per-layer τ statistic rather than a
worst-of-4 ratio; then (4) `n > 2` seeds with the stabilized ruler. Each is a new pre-commitment, not
part of exp3.
