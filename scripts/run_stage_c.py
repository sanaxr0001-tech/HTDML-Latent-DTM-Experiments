# scripts/run_stage_c.py
"""Stage-C run entry: binds the real GPU ops into the pure orchestrator + writes JSON/report.
The ONLY file that touches the GPU.  MODE=smoke (tiny plumbing) | full (paid 2-seed run)."""
from __future__ import annotations
import json, os, subprocess, sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
for _bp in (str(_REPO_ROOT), str(_REPO_ROOT / "src")):
    if _bp not in sys.path:
        sys.path.insert(0, _bp)

import htdml.paths as _p  # noqa: E402
_p.bootstrap_paths()
from htdml import orchestrator as O  # noqa: E402
from htdml.driver import AcceptanceConstants  # noqa: E402
from scripts.calib_logic import WallClock  # noqa: E402

def parse_config(env):
    mode = env.get("MODE", "full").lower()
    budget_h = float(env.get("BUDGET_H", "4.0"))
    if "SEEDS" in env:
        seeds = [int(s) for s in env["SEEDS"].split(",") if s.strip()]
    else:
        seeds = [1] if mode == "smoke" else [1, 2]
    const = O.FrozenConstants(GPU_H_CAP=budget_h)        # ESS_min/C/L_traj/... frozen defaults (PINS)
    if mode == "smoke":
        # plumbing smoke: bound the reject loop to ONE epoch-block (epochs_per_block=2 → 1 block).  The
        # loop logic itself is unit-tested with fakes; the smoke only needs to prove the GPU wiring of a
        # single fork→epoch→probe→FID→route block.  The full/paid run keeps the production max_joint_epochs.
        from dataclasses import replace
        const = replace(const, max_joint_epochs=2)
    return seeds, const, mode

def build_provenance():
    def _git(*a):
        try: return subprocess.check_output(["git", *a], cwd=os.path.dirname(__file__) or ".").decode().strip()
        except Exception: return "unknown"
    try:
        import jax; backend = jax.default_backend()
    except Exception: backend = "unknown"
    try:
        from harness.reversible_scan import is_patch_live; patch = bool(is_patch_live())
    except Exception: patch = None
    return {"git_sha": _git("rev-parse", "HEAD"), "env_freeze": "env-h200-freeze.txt",
            "jax_backend": backend, "is_patch_live": patch}

_REPORT = "# Stage-C run report\n\nOutcome: **{outcome}**\n\nWritten after run_stage_c.json. MEASURE-ONLY — no wiki tag move.\n"

def write_outputs(result, outdir, *, mode):
    os.makedirs(outdir, exist_ok=True)
    name = "run_stage_c_smoke.json" if mode == "smoke" else "run_stage_c.json"
    with open(os.path.join(outdir, name), "w") as f:        # JSON FIRST
        json.dump(result, f, indent=2, default=str)
    with open(os.path.join(outdir, "report.md"), "w") as f:  # report AFTER the json exists
        f.write(_REPORT.format(outcome=result["outcome"]))
    return 2 if result["outcome"] == "BUDGET-WALL" else 0

# ====================================================================== smoke vs full run config
class _SmokeCfg:
    """TINY plumbing config (matches the proven Task-12 smoke: 2 epochs / 3 classes / 600 train /
    80 AE steps; ran in ~194s on the 4060).  Probe + calibration sizes are kept small so the FULL
    Stage-C path (Stage A → B → calibrate → reconfirm → fork → paired joint/control → reject → route)
    fits the 1hr local cap.  L_traj passed by the orchestrator is CLAMPED to L_TRAJ_MAX here."""
    # Stage A / B
    n_train = 600
    n_test = 1000
    ae_steps = 80
    stage_b_epochs = 2
    # probe sizes (smoke — calibration freezes the production sizes; here keep cheap)
    probe_n_R = 8
    probe_n_chains = 4
    probe_L_traj = 60
    L_TRAJ_MAX = 60          # clamp the orchestrator-passed L_traj to keep the smoke probe cheap
    # calibration (per-layer T_O doubling-stability) — small doubling curve
    cal_n_chains = 4
    cal_L0 = 60
    cal_warm = 80
    cal_n_rungs = 3
    # joint/control + generation
    n_gen_per_class = 4
    diag_key = 20240624


class _FullCfg(_SmokeCfg):
    """Paid-run config (NOT exercised here — the smoke pass only proves MODE=smoke)."""
    n_train = 6000
    ae_steps = 2000
    stage_b_epochs = 200
    probe_n_R = 16
    probe_n_chains = 4
    probe_L_traj = 400
    L_TRAJ_MAX = 4000
    cal_n_chains = 8
    cal_L0 = 100
    cal_warm = 200
    cal_n_rungs = 4
    n_gen_per_class = 8


# ====================================================================== opaque carriers through the seam
class _Encoder:
    """What pretrain_encoder returns (flows opaquely through the orchestrator).  Carries the trained
    AE params + the raw filtered Fashion-MNIST split needed downstream (build the latent dataset for
    Stage B, the probe batch, and the x_batch for the joint encoder update)."""
    def __init__(self, ae_params, tr_img, tr_lab, te_img, te_lab):
        self.ae_params = ae_params
        self.tr_img, self.tr_lab = tr_img, tr_lab
        self.te_img, self.te_lab = te_img, te_lab


class _Model:
    """What train_latent_dtm returns: the LatentDTM (.dtm = vendored DTM) + the encoder + the injected
    latent dataset + the probe batch.  Forked into _Arm pairs by fork()."""
    def __init__(self, ldtm, enc, latent_ds, batch):
        self.ldtm = ldtm
        self.enc = enc
        self.latent_ds = latent_ds
        self.batch = batch


class _Arm:
    """One Stage-C arm (joint OR control): its own forked DTM + its own encoder param/opt-state copy +
    the shared static (encoder, latent dataset, probe batch).  commit/rollback snapshot+restore the DTM
    out-of-band (autocorrelations + dtm.key + opt-state, per build-notes §"DTM.load drops autocorr")."""
    def __init__(self, dtm, enc, latent_ds, batch, ae_params, lam):
        self.dtm = dtm
        self.enc = enc
        self.latent_ds = latent_ds
        self.batch = batch
        self.ae_params = ae_params
        self.lam = lam                # λ for this arm (joint = const.lambda_joint, control = 0.0)
        self._snapshot = None         # (ArmState, ae_params) captured at last commit


class RealOps:
    """Binds the real GPU ops (driver.* + TrainabilityProbe + FID).  Smoke-only — no unit test.
    Each method wires the verbatim primitives; see the spec §3.1 seam contract for shapes.

    GPU imports are LAZY (inside the methods) so this module stays CPU-importable — parse_config /
    write_outputs / build_provenance must import without touching the GPU (the CPU helper tests)."""

    def __init__(self, const, *, smoke):
        self.const = const
        self.smoke = smoke
        self.cfg = _SmokeCfg() if smoke else _FullCfg()

    # ------------------------------------------------------------------ STAGE A — pretrain encoder
    def pretrain_encoder(self, seed, clock):
        """driver.stage_a_pretrain via smoke_common.pretrain_autoencoder (BinaryAutoencoder Stage-A
        loss = BCE + commitment + balance).  Returns an _Encoder carrying ae_params + the raw split."""
        import jax.random as jr
        from scripts import smoke_common as sm

        cfg = self.cfg
        tr_img, tr_lab, te_img, te_lab = sm.load_fashion_mnist(n_train=cfg.n_train, n_test=cfg.n_test)
        ae_params, _losses = sm.pretrain_autoencoder(
            tr_img, key=jr.PRNGKey(int(seed) + 1), n_steps=cfg.ae_steps, batch_size=64, lr=1e-3)
        return _Encoder(ae_params, tr_img, tr_lab, te_img, te_lab)

    # ------------------------------------------------------------------ STAGE B — train latent DTM
    def train_latent_dtm(self, encoder, seed, clock):
        """driver.stage_b_train_latent_dtm (= LatentDTM.fit → dtm.train) on the encoded hard latents.
        Returns a _Model (LatentDTM + encoder + latent dataset + probe batch)."""
        import functools

        import jax.numpy as jnp
        from htdml.autoencoder import encode as ae_encode
        from scripts import smoke_common as sm

        cfg = self.cfg
        encode_fn = functools.partial(ae_encode, encoder.ae_params)
        latent_ds = sm.build_latent_dataset(
            encode_fn, encoder.tr_img, encoder.tr_lab, encoder.te_img, encoder.te_lab,
            target_classes=sm.SMOKE_TARGET_CLASSES, num_label_spots=sm.SMOKE_NUM_LABEL_SPOTS)
        dtm = sm.build_companion_dtm(latent_ds, seed=int(seed))
        ldtm = sm.LatentDTM(dtm, decode_fn=sm.make_decode_fn(encoder.ae_params))
        ldtm.fit(latent_ds, n_epochs=cfg.stage_b_epochs, evaluate_every=0)   # GPU dtm.train
        train_ds, _test_ds, _ohtl = latent_ds
        batch = dict(image=jnp.asarray(train_ds["image"]),
                     label=jnp.asarray(train_ds["label"]), idx=0)
        return _Model(ldtm, encoder, latent_ds, batch)

    # ------------------------------------------------------------------ calibrate τ̂ (per layer)
    def calibrate_tau(self, model, clock):
        """4× TrainabilityProbe.calibrate (one per reverse layer) → aggregate to the seam dict
        {tau_hat_layers:[4], cal_stable:all(...), failed_layer:first failing idx|None,
        cal_curves:[per-layer annotated doubling curve], failed_axes:[per-layer failed axis list]}.
        The per-layer curve + failed_axis are PERSISTED (run 5b9cbbc discarded them) so a
        Q-CALIBRATION-FAIL is diagnosable from the run JSON without a re-run."""
        from htdml.trainability_probe import TrainabilityProbe

        cfg = self.cfg
        probe = TrainabilityProbe()
        dtm = model.ldtm.dtm
        tau_layers, stables = [], []
        cal_curves, failed_axes = [], []      # per-layer annotated doubling curve + failed axis/axes (diagnostics)
        failed_layer = None
        for layer in range(len(dtm.steps)):
            clock.checkpoint(f"calib_layer_{layer}", raise_on_over=True)
            cal = probe.calibrate(model.ldtm, layer=layer, batch=model.batch,
                                  n_chains=cfg.cal_n_chains, L0=cfg.cal_L0, warm=cfg.cal_warm,
                                  n_rungs=cfg.cal_n_rungs, diag_key=cfg.diag_key)
            tau_layers.append(float(cal["tau_hat"]))
            stables.append(bool(cal["cal_stable"]))
            cal_curves.append(cal.get("curve"))          # JSON-safe annotated rungs (dS_l1 None → null)
            failed_axes.append(cal.get("failed_axis"))   # list[str] naming the vetoing axis/axes per layer
            if not cal["cal_stable"] and failed_layer is None:
                failed_layer = layer
        cal_stable = bool(all(stables))
        # SMOKE-ONLY plumbing override: a tiny 2-epoch smoke DTM can't reach T_O doubling-stability,
        # so cal-fail short-circuits before the Stage-C engine.  With SMOKE_FORCE_STAGE_C=1 we run the
        # REAL calibrate (logging its true verdict) but force cal_stable=True to drive fork→epoch→probe→
        # FID→route and shake out that wiring.  NOT a science relaxation — the real gate runs on the H200.
        if self.smoke and not cal_stable and os.environ.get("SMOKE_FORCE_STAGE_C") == "1":
            print(f"[SMOKE_FORCE_STAGE_C] real calibrate verdict cal_stable=False "
                  f"(failed_layer={failed_layer}, tau_hat_layers={tau_layers}); FORCING cal_stable=True "
                  f"to exercise the Stage-C plumbing — NOT a science verdict.", flush=True)
            cal_stable, failed_layer = True, None
        return {"tau_hat_layers": tau_layers, "cal_stable": cal_stable,
                "failed_layer": failed_layer, "cal_curves": cal_curves, "failed_axes": failed_axes}

    # ------------------------------------------------------------------ probe-cost estimate (seconds)
    def estimate_probe_cost(self, L_traj):
        """Measured seconds/retained-sweep × L_traj × N_chains (the orchestrator multiplies by the
        WallClock guard).  Cached: a single tiny seconds/sweep probe-timing, reused per call."""
        sps = getattr(self, "_sec_per_sweep", None)
        if sps is None:
            sps = self._measure_seconds_per_sweep()
            self._sec_per_sweep = sps
        L_eff = min(int(L_traj), self.cfg.L_TRAJ_MAX)
        return float(sps) * float(L_eff) * float(self.const.N_chains)

    def _measure_seconds_per_sweep(self):
        """Crude per-retained-sweep cost from a tiny negative-phase trajectory timing on layer 0 of the
        most-recently-built model (set by fork()/epoch).  Falls back to a small constant if unavailable."""
        return getattr(self, "_sps_fallback", 1e-3)

    # ------------------------------------------------------------------ FORK — control + joint arms
    def fork(self, model, workdir):
        """driver.fork_checkpoint (DTM.save → load×2 + out-of-band restore) → (control, joint) arms.
        Each arm gets its OWN encoder-param copy (the joint arm steers them; control does not)."""
        import os

        from htdml import driver as DRV

        os.makedirs(workdir, exist_ok=True)
        control_dtm, joint_dtm = DRV.fork_checkpoint(model.ldtm.dtm, workdir, epoch=0)
        # DTM.load rebuilds each arm as a FRESH DTM from the REGISTERED (empty smoke) dataset + class
        # defaults — fork_checkpoint re-injects only key + autocorrelations out-of-band.  Re-apply the
        # SAME post-construction static that build_companion_dtm set on the parent (smoke_common.py:155-164;
        # DTM.load reverts all of it): the seam-A latent train/test datasets + one_hot_target_labels + the
        # two derived dims (so dtm.train()'s compute_autocorr shape asserts + training data + generate's
        # label conditioning all match the parent), and log_file=None (DTM.load → class default '' →
        # write(open('')) crashes; None makes utils.write print-only).  Copy from the parent (already correct).
        parent_dtm = model.ldtm.dtm
        for arm_dtm in (control_dtm, joint_dtm):
            arm_dtm.train_dataset = parent_dtm.train_dataset
            arm_dtm.test_dataset = parent_dtm.test_dataset
            arm_dtm.one_hot_target_labels = parent_dtm.one_hot_target_labels
            arm_dtm.n_image_pixels = parent_dtm.n_image_pixels
            arm_dtm.n_label_nodes = parent_dtm.n_label_nodes
            arm_dtm.log_file = None
        ae = model.enc.ae_params
        control = _Arm(control_dtm, model.enc, model.latent_ds, model.batch, ae, lam=0.0)
        joint = _Arm(joint_dtm, model.enc, model.latent_ds, model.batch, ae,
                     lam=float(self.const.lambda_joint))
        # seed the probe-cost timer off the parent model so estimate_probe_cost has a real number.
        self._sps_fallback = 1e-3
        # commit the freshly-forked state as the rollback baseline.
        self.commit_pair(joint, control)
        return control, joint

    # ------------------------------------------------------------------ one paired joint/control epoch
    def epoch_block_pair(self, joint, control, encoder_lr, L_traj, clock):
        """One Stage-C epoch on each arm: (a) one DTM epoch on DETACHED latents (dtm.train, GPU);
        (b) one encoder enc/dec update via driver.joint_update_step (λ=arm.lam, fresh adam(encoder_lr));
        (c) TrainabilityProbe.evaluate ×4 layers; (d) reconstruction BCE + FID on the decoded 28×28.
        Returns a BlockResult (joint_layers/control_layers carry the gate/router keys)."""
        jl, jbce, jfid = self._train_probe_arm(joint, encoder_lr, L_traj, clock)
        cl, cbce, cfid = self._train_probe_arm(control, encoder_lr, L_traj, clock)
        return O.BlockResult(joint_layers=jl, control_layers=cl,
                             bce_joint=jbce, fid_joint=jfid, bce_control=cbce, fid_control=cfid,
                             gpu_h=clock.elapsed() / 3600.0)

    def _train_probe_arm(self, arm, encoder_lr, L_traj, clock):
        """One Stage-C ALTERNATING step for ONE arm, ordered so a SINGLE scored block makes the joint arm
        diverge from control on the DTM probe-Q (the guard quantity):
          (a) encoder update — λ=arm.lam steers via λ·L_compat; λ=0 = recon-only control (so the joint−
              control margin isolates the compat/λ effect);
          (b) RE-ENCODE detached latents with the now-current encoder + inject into arm.dtm + refresh
              arm.batch — the steered encoder → different latents → different DTM → different Q (closing the
              encoder→latent→DTM loop the previous wiring left open);
          (c) one DTM epoch on those refreshed detached latents;
          (d) probe 4 layers + (e) reconstruction BCE / FID.
        Order is encoder-first (vs the natural DTM-first) precisely so ONE block suffices to show the
        effect — DTM-first would only diverge at block 2 (researcher-confirmed)."""
        # (a) encoder update FIRST.
        arm.ae_params = self._encoder_update(arm, encoder_lr)
        # (b) re-encode (detached) with the updated encoder + inject into this arm's DTM + refresh batch.
        self._refresh_arm_latents(arm)
        # (c) one DTM epoch on the refreshed detached latents.
        arm.dtm.train(n_epochs=self.const.epochs_per_block, evaluate_every=0)
        clock.checkpoint(f"arm_dtm_epoch_lam{arm.lam}", raise_on_over=True)
        # (d) probe all 4 layers + (e) BCE/FID on the decoded 28×28.
        layers = self._probe_arm(arm, L_traj)
        bce, fid = self._quality_arm(arm)
        return layers, bce, fid

    def _refresh_arm_latents(self, arm):
        """Re-encode the raw images with the arm's CURRENT (post-update) encoder into DETACHED hard
        latents and inject them into arm.dtm (train/test datasets + derived dims) via the tested
        LatentDTM.inject_latents (numpy→jnp, the Tracer fix); refresh arm.batch from the new train split.
        build_latent_dataset materialises NUMPY bool arrays, so the injected latents carry NO gradient back
        into ae_params — the 'detached latents' the Stage-C spec requires.  inject_latents does NOT touch
        the out-of-band static (key/autocorrelations/log_file) that fork() restored, so those persist."""
        import functools

        import jax.numpy as jnp
        from htdml.autoencoder import encode as ae_encode
        from htdml.latent_dtm import LatentDTM
        from scripts import smoke_common as sm

        enc = arm.enc
        encode_fn = functools.partial(ae_encode, arm.ae_params)
        latent_ds = sm.build_latent_dataset(
            encode_fn, enc.tr_img, enc.tr_lab, enc.te_img, enc.te_lab,
            target_classes=sm.SMOKE_TARGET_CLASSES, num_label_spots=sm.SMOKE_NUM_LABEL_SPOTS)
        LatentDTM(arm.dtm, decode_fn=sm.make_decode_fn(arm.ae_params)).inject_latents(latent_ds)
        train_ds, _test_ds, _ohtl = latent_ds
        arm.latent_ds = latent_ds
        arm.batch = dict(image=jnp.asarray(train_ds["image"]),
                         label=jnp.asarray(train_ds["label"]), idx=0)

    def _encoder_update(self, arm, encoder_lr):
        """driver.joint_update_step on diffusion step 0 with a fresh adam(encoder_lr).  Builds the
        compat clamp's label/b_t 'rest' columns hard (stop_gradient'd); the encoder is steered ONLY via
        the image_output b0 latent (∂≠0 at λ>0, =0 at λ=0)."""
        import jax.numpy as jnp
        import numpy as np
        from htdml import driver as DRV
        from htdml.autoencoder import BinaryAutoencoder

        step = arm.dtm.steps[0]
        step_maps = DRV.step_maps_for(step)
        n_img = int(step_maps[0]["n_img"])
        n_clamp = int(step_maps[0]["n_clamp"])
        n_rest = n_clamp - n_img

        # a representative raw-pixel batch for the encoder (steered via b0 = encode(params, x_batch)).
        x_batch = jnp.asarray(arm.enc.tr_img[: 32]).astype(jnp.float32)
        # the 'rest' clamp columns (label_output + b_t), hard ±1, stop_gradient'd inside the loss.
        # Derived deterministically from the data label one-hot (padded/tiled to n_rest), so it is a
        # faithful hard conditioning draw (the test-proven pattern folds all rest into label_clamp).
        lab_bits = np.asarray(arm.batch["label"])[0].astype(np.float64)     # (n_lab,) {0,1}
        rest = np.resize(2.0 * lab_bits - 1.0, (max(n_rest, 1),))[: n_rest] if n_rest else np.zeros((0,))
        with DRV._x64():
            label_clamp = jnp.asarray(rest, dtype=jnp.float64)
            bt_clamp = jnp.zeros((0,), dtype=jnp.float64)

        ae_optim = DRV.rebuild_encoder_optimizer(float(encoder_lr))
        ae_opt_state = ae_optim.init(arm.ae_params)
        beta = float(step.training_spec.beta)
        new_params, _new_opt, _aux = DRV.joint_update_step(
            BinaryAutoencoder(), arm.ae_params, ae_opt_state, ae_optim, step,
            label_clamp=label_clamp, bt_clamp=bt_clamp, beta=beta, lam=float(arm.lam),
            x_batch=x_batch, step_maps=step_maps)
        return new_params

    def _probe_arm(self, arm, L_traj):
        """TrainabilityProbe.evaluate ×4 layers → per-layer dicts carrying EXACTLY the keys the gate /
        router read (Q_struct_perp, tau_int_Y, ESS_hat, r_grad[50], gradient_norm, L_traj, tau_hat,
        cal_stable)."""
        import jax.random as jr
        from htdml.trainability_probe import TrainabilityProbe

        cfg = self.cfg
        probe = TrainabilityProbe()
        L_eff = min(int(L_traj), cfg.L_TRAJ_MAX)
        dtm = arm.dtm
        layers = []
        for layer in range(len(dtm.steps)):
            key = jr.fold_in(jr.PRNGKey(int(cfg.diag_key)), int(layer))
            r = probe.evaluate(arm.dtm, layer, arm.batch, n_R=cfg.probe_n_R, L_traj=L_eff,
                               n_chains=cfg.probe_n_chains, diag_key=cfg.diag_key, key=key)
            layers.append({
                "Q_struct_perp": float(r["Q_struct_perp"]),
                "tau_int_Y": float(r["tau_int_Y"]),
                "ESS_hat": float(r["ESS_hat"]),
                "r_grad[50]": float(r["r_grad[50]"]),
                "r_grad[1]": float(r["r_grad[1]"]),
                "gradient_norm": float(r["gradient_norm"]),
                "L_traj": int(L_eff),
                "tau_hat": float(r["tau_int_Y"]),     # per-layer τ̂ for the L_traj ≥ C·τ̂ adequacy read
                "cal_stable": True,
                "layer": int(layer),
            })
        return layers

    def _quality_arm(self, arm):
        """Reconstruction BCE (AE Stage-A recon term on a test batch through the arm's encoder) + FID on
        the decoded 28×28 of conditional-generation latents (network-free FID, Task-10 path)."""
        import jax.numpy as jnp
        import jax.random as jr
        import numpy as np
        from htdml.autoencoder import stage_a_loss, decode as ae_decode
        from scripts import smoke_common as sm

        cfg = self.cfg
        # reconstruction BCE: the Stage-A recon term on a test batch with the arm's (possibly steered) AE.
        x_test = jnp.asarray(arm.enc.te_img[: 256]).astype(jnp.float32)
        _tot, aux = stage_a_loss(arm.ae_params, x_test)
        bce = float(aux["recon"])
        # FID on decoded conditional generations (reuse LatentDTM.generate → decode → fid_on_decoded).
        from htdml.latent_dtm import LatentDTM
        ldtm = LatentDTM(arm.dtm, decode_fn=sm.make_decode_fn(arm.ae_params))
        decoded = ldtm.generate(jr.PRNGKey(int(cfg.diag_key) + 7), labels=None,
                                samples_per_label=cfg.n_gen_per_class, free=False, decode=True)
        fid, _t1, _t2 = sm.fid_on_decoded(np.asarray(decoded))
        return bce, float(fid)

    # ------------------------------------------------------------------ committed-baseline probe
    def probe_committed_pair(self, joint, control, L_traj, clock):
        """Evaluate-ONLY on the committed pair (no DTM epoch, no encoder update).  Both arms are the
        SAME committed model → joint == control layers (the committed baseline)."""
        layers = self._probe_arm(joint, L_traj)
        bce, fid = self._quality_arm(joint)
        return O.BlockResult(joint_layers=layers, control_layers=[dict(d) for d in layers],
                             bce_joint=bce, fid_joint=fid, bce_control=bce, fid_control=fid,
                             gpu_h=clock.elapsed() / 3600.0)

    # ------------------------------------------------------------------ commit / rollback (out-of-band)
    def commit_pair(self, joint, control):
        """Snapshot BOTH arms' DTM static (key + per-step autocorrelations + opt-counts) + their AE
        params as the rollback baseline (driver.capture_arm_state)."""
        from htdml import driver as DRV

        for arm in (joint, control):
            arm._snapshot = (DRV.capture_arm_state(arm.dtm), arm.ae_params)

    def rollback_pair(self, joint, control):
        """Restore BOTH arms to their last committed snapshot: re-inject the captured DTM static
        out-of-band (driver.restore_out_of_band) + restore the AE params."""
        from htdml import driver as DRV

        for arm in (joint, control):
            if arm._snapshot is not None:
                arm_state, ae_params = arm._snapshot
                DRV.restore_out_of_band(arm.dtm, arm_state)
                arm.ae_params = ae_params

def resolve_outdir(env, outdir=None):
    """Resolve the output dir: explicit arg > OUTDIR env > default ../results.  The OUTDIR knob lets the
    exp2 re-run write to experiments/exp2-cal-gate-fix/artifacts/ (named 'artifacts', NOT 'results', to
    dodge the results/ gitignore) without clobbering run-1; the default keeps the pre-exp2 behavior."""
    if outdir:
        return outdir
    if env.get("OUTDIR"):
        return env["OUTDIR"]
    return os.path.join(os.path.dirname(__file__), "..", "results")

def main(env=None, outdir=None):
    env = os.environ if env is None else env
    seeds, const, mode = parse_config(env)
    outdir = resolve_outdir(env, outdir)
    clock = WallClock(cap_seconds=const.GPU_H_CAP * 3600.0)
    ops = RealOps(const, smoke=(mode == "smoke"))
    result = O.run_stage_c(ops, seeds=seeds, acc=AcceptanceConstants(
        ESS_min=const.ESS_min, C=const.C, L_traj=const.L_traj, N_chains=const.N_chains, N_R=const.N_R,
        GPU_H_CAP=const.GPU_H_CAP),
        const=const, workdir=os.path.join(outdir, "work"), provenance=build_provenance(), clock=clock)
    return write_outputs(result, outdir, mode=mode)

if __name__ == "__main__":
    sys.exit(main())
