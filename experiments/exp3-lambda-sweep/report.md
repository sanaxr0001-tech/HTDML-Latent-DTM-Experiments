# exp3 — lower-λ sweep → `HTDML-MARGIN-NEGATIVE` (no demonstrable steering effect; apparent per-seed pass not distinguishable from control-denominator noise)

**RAN on H200 2026-06-27.** Resume sweep over `λ ∈ {0.5, 0.3, 0.1}` on exp2's persisted Stage-B
checkpoints (`RESUME_FROM`, Stage A+B skipped), code `19cd46f`, `MODE=full SEEDS=1,2`; all three runs
finished clean, **no budget wall**. Sweep total **5.67 GPU-h** (2.181 + 2.200 + 1.287) of the
9.0 backstop. Plotted against exp2's `λ=1.0` anchor (code `a26dbce`). **MEASURE-ONLY — companion-local
tokens only.** Raw: `artifacts/lam{0.5,0.3,0.1}/run_stage_c.json` + `.run.log`.

> **Interpretation note (publication-grade adversarial verification, 2026-06-27).** An earlier draft of
> this report read the per-seed gate passes as a "reproducible τ-route mixing improvement." A 4-lens
> adversarial verification against the raw JSONs **refuted that interpretation** (numbers and provenance
> reproduced exactly; the *mechanism narrative did not*). This version states the corrected, defensible
> conclusion: **run-level negative, and the apparent per-seed clearance is not distinguishable from
> control-denominator noise.** The retraction is recorded deliberately rather than silently revised.

---

## Headline

Pre-registered question: **does any λ yield a robust (both-seed) margin
clearance — improvement *and* ESS-non-degradation — where λ=1.0 (exp2) did not?**

**Answer: no.** Run-level **`HTDML-MARGIN-NEGATIVE` at every λ** (`two_seed.both_pass=False` for all four).
The driver emitted a per-*seed* `HTDML-MARGIN-POSITIVE` token for **seed 1 at λ=0.3 and λ=0.1** (its τ-leg
gate passed), but **this is not evidence of a steering effect** — it is consistent with noise in the
worst-layer control denominator of the gate ratio (§"Why the per-seed pass is not a steering effect").
**Conclusion: no demonstrable effect of the mixing-aware steering on the finite-budget mixing margin, at
any λ in the ladder** — once the gate's estimator variance is accounted for, neither route shows a
clearance distinguishable from null. **Equally, exp3 cannot *refute* a steering effect:** at n=2 with this
noisy ruler, a true acceleration smaller than the ~2.5× run-to-run control swing would be invisible. The
result is **unidentified, not proven-zero** — the current ruler cannot adjudicate either way.

---

## The measured verdict — per-λ × per-seed × 7-criterion gate

Gate (registered a-priori): a seed PASSES iff all 7
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

## Why the per-seed pass is not distinguishable from control-denominator noise

The τ-leg is `max₄(joint τ) / max₄(control τ) ≤ 0.75×`. The λ=0 control arm is forked from the *same*
Stage-B checkpoint and is the *same condition* across all of a seed's runs, so its worst-layer `τ_int,Y`
should be ~one number per seed. It is not — and the "passes" ride entirely on its excursions:

| λ | seed-1 **joint** (steered) worst-τ | seed-1 **control** (λ=0) worst-τ | ratio | control worst-ESS |
|---|---|---|---|---|
| 1.0 | 1.242 | 1.198 | 1.04× | 20.9 |
| 0.5 | 0.911 | 0.917 | 0.99× | 27.3 |
| 0.3 | 1.394 | **2.293** | 0.61× | **10.9** |
| 0.1 | 1.107 | **2.152** | 0.51× | **11.6** |

Six independent reasons the noise explanation is far more parsimonious than a steering signal — the
passes are *not distinguishable* from control-denominator noise, not *proven* to be noise:

1. **The control denominator is non-reproducible.** Seed-1 λ=0 control worst-τ swings **0.92 → 2.29 →
   2.15** (~2.5×) across identical-code runs — for a quantity that should be fixed. The two passes are
   exactly the two runs where it spiked high.
2. **The steered numerator shows no λ-trend.** Seed-1 *joint* worst-τ is **1.24 / 0.91 / 1.39 / 1.11** —
   flat and non-monotone; it is *worse* at λ=0.3 (1.39) than at λ=1.0 (1.24). The ratio fell because the
   denominator rose, not because the steered chain mixed faster.
3. **The passing runs have anomalously poorly-mixed control arms.** Control worst-ESS bottoms at **10.9
   / 11.6**, just above the `ESS_min=10` floor, precisely at λ=0.3/0.1 — the low-ESS ⇔ high-τ signature
   of a badly-mixed control *draw*, which is what inflates the ratio. (Distinct from — but pointing the
   same way as — the λ=0.5 seed-2 Q-route "lift", which fails on a *different* leg: its **joint** ESS
   degrades below control, 18.3 < 21.2, failing ESS-non-degradation; the control ESS there, 21.2, is fine.)
4. **`max₄` compares different physical layers.** At λ=0.3 the slowest control layer is layer-3 (control
   τ 2.29; the steered chain there is 0.67) — but at layer-1 the steered τ (1.39) sits *above* its own
   control (1.16), so the steering did not accelerate that layer; the ratio fell only because the
   control's slowest layer spiked. At λ=0.1 the slowest control layer is instead layer-1. A different
   layer drives the max each run — a noisy-max artifact, not a reproducible per-layer acceleration.
5. **The reliable mixing measure inverts the story.** The deterministic reconfirm `τ̂` (bit-identical
   across all runs) is **seed-1 = 2.42, seed-2 = 3.67** → **seed 2 is the slower-mixing seed**, the
   opposite of any "seed 1 is slow / has headroom" reading.
6. **Multiple comparisons.** The OR-gate over (Q, τ) × 4 λ × 2 seeds is up to 16 leg-tests on a
   high-variance worst-of-4 ruler; two sub-0.75 τ ratios (both seed-1, both lowest λ) are well within
   what denominator noise produces, uncorrected for multiplicity.

**Quantitatively, the strongest pro-effect summary is null.** Metastable stalls (worst-layer `τ_int,Y >
2`) occur in the seed-1 control arm in **2/6** runs but the steered arm in **0/6** — Fisher exact **p ≈
0.46**; and at the two passes the steered chain is *slower* than its own control on **5 of 8** non-stall
layers (reshuffling, not acceleration), the lone "win" each run being the single max-picked control-stall
layer. The 2/2 paired stall-avoidance is a real but **underpowered** observation — a hypothesis fix (a)
below would adjudicate two-sided, **not** a refuted one.

**Both routes are consistent with null** (no clearance distinguishable from noise). Q-route: relaxes to
control (seed-2 lq-Q 1.33×→1.07×→1.01×); its single apparent clearance (λ=0.5 seed-2, 1.33×) fails the
ESS-non-degradation leg (joint ESS 18.3 < control 21.2). τ-route: control-denominator noise, above. **No
λ produces a clearance distinguishable from control-denominator noise.**

---

## Methodological takeaway (the durable contribution)

At these probe lengths (`L_traj=400`, `K=50`, `B=400`), the **worst-layer-max `τ_int,Y` ratio is
plausibly noise-dominated** — the estimator variance is not yet formally quantified (see fix (a)). Across
the n=3 shared-code runs the λ=0 control's `max₄ τ` varies **~2.5× (seed-1), ~1.45× (seed-2)** at fixed
seed, and the steered numerator also varies ~1.5× — it is a *noisy ratio of two short-chain max-of-4
estimators*, not merely a noisy denominator. The relative-ruler discipline
(the half-Sokal T_O bias-cancellation argument) cancels the multiplicative *bias* of the half-Sokal
estimator (a common factor in a joint/control ratio — exact only if both arms share the same bias,
approximately true in the same short-`L` regime) — but a constant bias cancels in a ratio while
**estimator variance does not**, and a *max over 4 short-chain estimates* is variance-amplifying.
**Per-seed gate passes are therefore uninterpretable as effects without a control-variance estimate.**
The minimal fixes for any future effect claim: (a) repeated λ=0 control draws at fixed seed to measure
the worst-τ run-to-run variance — and, **two-sided**, to test whether the steered arm has a
*systematically lower* stall rate (the 2/2 paired stall-avoidance above); (b) longer trajectories
(`L_traj ≫ 400`) to stabilize `max₄`; (c) paired per-layer τ tests instead of a worst-of-4 ratio; then
(d) `n > 2` seeds. This reliability caveat is the transferable result of exp3.

---

## Prediction vs outcome (vs the pre-registration)

- **Pre-registered modal prediction:** channel-negative ("no λ clears the margin"). → **CONFIRMED** — no
  robust positive, and no demonstrable per-seed effect either: the apparent passes are not distinguishable
  from control-denominator noise. The registered Q-centric reasoning under-specified the failure (it
  predicted Q→control; the τ-leg too yields only a noise-consistent signal), but the **direction
  (negative) was correct.**
- **Decisive-either-way clause:** "any λ clears on BOTH seeds → POSITIVE; none → channel negative." →
  **Channel-negative**, with the per-seed positives shown — most parsimoniously — to be control-denominator
  noise after adversarial verification. No acceptance bar moved; frozen-five and the 7-criterion predicate
  unchanged.

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
| Budget | λ=0.5 **2.181**, λ=0.3 **2.200**, λ=0.1 **1.287** GPU-h, all `budget_wall=False`. λ=0.1 launched at `BUDGET_H=4.0` (pre-result amendment) but finished at **1.29 GPU-h → clean under the original 3.0 cap** (4.0 backstop unused) |
| Pre-flight (on-box) | dataset sha ✓ · zero-compute battery 43/43 ✓ · reversible cert 10/10 ✓ · `LAMBDA_JOINT` knob test 8/8 ✓ |
| Verification | 4-lens adversarial pass (numbers / provenance / claims / scope), all numbers + provenance reproduced from raw JSONs; the claims+scope lenses drove this report's corrected interpretation |

---

## Scope & limitations (front and center)

- **No demonstrable effect.** The headline result is a **negative**; the per-seed gate passes are a
  measured curiosity most parsimoniously attributable to control-denominator noise rather than a steering
  signal.
- **Underpowered in both directions.** The same estimator variance that prevents reading the per-seed
  passes as a *positive* equally prevents exp3 from *refuting* a genuine steering effect — a true
  acceleration below the ~2.5× run-to-run control swing would be undetectable at L_traj=400, n=2. **exp3
  does not show the steering is ineffective; it shows the current ruler cannot adjudicate either way.**
- **Estimator-variance limit.** The gate's worst-layer-max τ ratio is *plausibly* noise-dominated at
  `L_traj=400` (the n=3 control draws span ~1.45–2.5×; variance not yet formally quantified); this, not
  seed biology, most parsimoniously governs the per-seed passes (see Methodological takeaway).
- **n = 2 seeds** — clearance *rate* unestimable regardless; and the per-seed signal here is not
  distinguishable from noise.
- **Degenerate absolute-quality regime:** absolute FID ≈ 280–326 in *both* arms (joint ~280–309, control
  ~287–326); "quality held" means
  joint ≈ control (BCE within 5%, FID within 10%), **not** that images are good. Relevant when judging
  whether τ differences are physically meaningful.
- **Companion-divergent setup:** `44_12` (not upstream `60_12`), Fashion-MNIST, the **deterministic
  mean-field compatibility surrogate**, 200-epoch Stage-B; λ=1.0 anchor is a different code version.
- **Two resume bugs** (`RESUME_FROM` path, `np` import) were found on the first GPU exercise and fixed +
  locally validated before the reported runs — disclosed for reproducibility.
- **MEASURE-ONLY:** companion-local tokens; **not** operational validation of the broader theorem; **no
  hardware-energy claim.** Terminal verdict pending separate review.

---

## Next (separate)

The result does not motivate an immediate follow-up *effect* experiment; it motivates a **measurement-
reliability** step first. Before any steering-effect claim: (1) quantify the λ=0 control worst-τ
run-to-run variance (repeated control draws at fixed seed); (2) lengthen trajectories (`L_traj ≫ 400`) to
stabilize the worst-of-4-layers estimator; (3) move to a paired per-layer τ statistic rather than a
worst-of-4 ratio; then (4) `n > 2` seeds with the stabilized ruler. Each is a new pre-commitment, not
part of exp3.
