"""scripts/smoke.py — full-path plumbing verification on the local RTX 4060 (Task 12).

Runs the WHOLE companion path end-to-end on the GPU with a TINY config, stopping + reporting at the
FIRST failing stage (the smoke is also the first real integration test — surface bugs, don't paper):

    encode (autoencoder) → fit (LatentDTM.fit = dtm.train, a few epochs) → probe
    (TrainabilityProbe.evaluate: per-step refresh HARD-HALT + the 4 layers) → generate → decode → FID
    (on the DECODED 28×28, INCLUDING the no-network ``assert_fid_offline`` assertion).

HARD constraints (pre-registered, see design notes):
  * 1hr wall-time cap on the 4060 (``--cap-seconds``, default 3600) — a BUDGET-WALL stop is not a
    failure; the script reports what completed.
  * NO Stage-C joint/control, NO two-seed, NO outcome token, NO H200.  Plumbing + calibration only.
  * The reversible ½(P_AB+P_BA) kernel must be LIVE (``LatentDTM.fit`` asserts ``is_patch_live``).

Usage::

    python scripts/smoke.py                 # default tiny config, 1hr cap
    python scripts/smoke.py --epochs 2 --n-train 600 --cap-seconds 3600

Exit code 0 = smoke PASS; non-zero = the stage that failed (printed).
"""

from __future__ import annotations

import argparse
import sys
import time
import traceback
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


def _banner(msg):
    print("\n" + "=" * 70 + f"\n{msg}\n" + "=" * 70, flush=True)


def _stage(wc, label):
    print(f"\n[stage] {label}  (elapsed {wc.elapsed():.1f}s / cap {wc.cap_seconds:.0f}s)", flush=True)


def run_smoke(*, epochs, n_train, n_gen_per_class, cap_seconds, ae_steps, seed=0):
    """Run the full plumbing path; return a result dict.  Raises on a stage failure (the caller maps
    the exception to the failing stage)."""
    wc = cl.WallClock(cap_seconds=float(cap_seconds))
    result = {"stages": {}, "wall_seconds": None, "passed": False, "failed_stage": None}

    # ---------------------------------------------------------------- assert GPU present
    _stage(wc, "device check")
    devs = jax.devices()
    has_gpu = any(d.platform == "gpu" for d in devs)
    print(f"  jax devices: {devs}  default_backend={jax.default_backend()}")
    assert has_gpu, f"no GPU device visible to jax (devices={devs}); the smoke REQUIRES the 4060"
    result["stages"]["device"] = {"devices": str(devs), "has_gpu": has_gpu}

    # ---------------------------------------------------------------- reversible kernel LIVE
    _stage(wc, "reversible kernel live check")
    from harness import reversible_scan
    live, detail = reversible_scan.is_patch_live()
    assert live, f"reversible kernel NOT live: {detail}"
    print(f"  is_patch_live: {live} ({detail})")
    result["stages"]["kernel_live"] = {"live": bool(live), "detail": str(detail)}

    # ---------------------------------------------------------------- (0) no-network FID assertion
    _stage(wc, "FID offline (no-network) assertion")
    from scripts.dataset_gate import assert_fid_offline
    assert_fid_offline()
    result["stages"]["fid_offline"] = {"ok": True}

    # ---------------------------------------------------------------- (1) ENCODE: AE pretrain + adapter
    _stage(wc, "ENCODE — Stage-A AE pretrain + latent-adapter encode")
    tr_img, tr_lab, te_img, te_lab = sm.load_fashion_mnist(n_train=int(n_train))
    print(f"  loaded train {tr_img.shape} / test {te_img.shape} "
          f"(classes {sm.SMOKE_TARGET_CLASSES}, spots {sm.SMOKE_NUM_LABEL_SPOTS})")
    ae_params, ae_losses = sm.pretrain_autoencoder(
        tr_img, key=jr.PRNGKey(seed + 1), n_steps=int(ae_steps), batch_size=64, lr=1e-3)
    print(f"  AE pretrain {len(ae_losses)} steps; loss {ae_losses[0]:.4f} → {ae_losses[-1]:.4f}")

    from htdml.autoencoder import encode as ae_encode
    import functools
    encode_fn = functools.partial(ae_encode, ae_params)
    latent_ds = sm.build_latent_dataset(
        encode_fn, tr_img, tr_lab, te_img, te_lab,
        target_classes=sm.SMOKE_TARGET_CLASSES, num_label_spots=sm.SMOKE_NUM_LABEL_SPOTS)
    train_ds, test_ds, ohtl = latent_ds
    print(f"  latent train image {train_ds['image'].shape} label {train_ds['label'].shape}; "
          f"test {test_ds['image'].shape}; ohtl {np.asarray(ohtl).shape}")
    result["stages"]["encode"] = {
        "ae_loss_first": float(ae_losses[0]), "ae_loss_last": float(ae_losses[-1]),
        "train_image_shape": list(train_ds["image"].shape),
        "label_width": int(train_ds["label"].shape[1])}

    # ---------------------------------------------------------------- (2) FIT: build DTM + dtm.train
    _stage(wc, "FIT — build real 44_12 DTM + dtm.train (few epochs)")
    dtm = sm.build_companion_dtm(latent_ds, seed=seed)
    print(f"  DTM built: n_image_pixels={dtm.n_image_pixels} n_label_nodes={dtm.n_label_nodes} "
          f"steps={len(dtm.steps)} is_smoke_test={dtm.is_smoke_test} "
          f"base_graph_manager={type(dtm.base_graph_manager).__name__}")
    decode_fn = sm.make_decode_fn(ae_params)
    ldtm = sm.LatentDTM(dtm, decode_fn=decode_fn)

    if wc.would_exceed(60.0):
        raise cl.BudgetWall("declining FIT — estimated to exceed the cap")
    t0 = time.time()
    ldtm.fit(latent_ds, n_epochs=int(epochs), evaluate_every=0)  # evaluate_every=0 → no eval epoch
    fit_seconds = time.time() - t0
    per_epoch = fit_seconds / max(int(epochs), 1)
    print(f"  dtm.train: {epochs} epochs in {fit_seconds:.1f}s ({per_epoch:.1f}s/epoch)")
    result["stages"]["fit"] = {"epochs": int(epochs), "fit_seconds": float(fit_seconds),
                               "per_epoch_seconds": float(per_epoch)}
    wc.checkpoint("after fit", raise_on_over=True)

    # ---------------------------------------------------------------- (3) PROBE: refresh + 4 layers
    _stage(wc, "PROBE — TrainabilityProbe.evaluate_model (per-step refresh HARD-HALT + 4 layers)")
    from htdml.trainability_probe import TrainabilityProbe
    probe = TrainabilityProbe()
    batch = dict(image=train_ds["image"], label=train_ds["label"], idx=0)
    # tiny probe sizes for the smoke (calibration freezes the production sizes separately)
    records = probe.evaluate_model(ldtm, batch=batch, n_R=8, L_traj=60, n_chains=4, diag_key=20240624)
    assert len(records) == len(dtm.steps), f"expected {len(dtm.steps)} layer records, got {len(records)}"
    for r in records:
        assert r["_refresh_proof"]["constructor_was_stale"] is True, "refresh-proof not stale (vacuous)"
        assert r["_refresh_proof"]["refresh_ok"] is True, "refresh did not take"
    probe_summary = [{"layer": r["layer"], "tau_int_Y": r["tau_int_Y"], "ESS_hat": r["ESS_hat"],
                      "Q_struct_perp": r["Q_struct_perp"], "gradient_norm": r["gradient_norm"],
                      "r_grad[50]": r["r_grad[50]"]} for r in records]
    for ps in probe_summary:
        print(f"  layer {ps['layer']}: τ_int,Y={ps['tau_int_Y']:.3f} ESS_hat={ps['ESS_hat']:.2f} "
              f"Q⊥={ps['Q_struct_perp']:.3g} |g|={ps['gradient_norm']:.3g} "
              f"r_grad[50]={ps['r_grad[50]']:.3g}")
    result["stages"]["probe"] = {"n_layers": len(records), "summary": probe_summary,
                                 "refresh_ok": True}
    wc.checkpoint("after probe", raise_on_over=True)

    # ---------------------------------------------------------------- (4) GENERATE → (5) DECODE
    _stage(wc, "GENERATE → DECODE — conditional annealing + AE decode to 28×28")
    decoded = ldtm.generate(jr.PRNGKey(seed + 2), labels=None,
                            samples_per_label=int(n_gen_per_class), free=False, decode=True)
    decoded = np.asarray(decoded)
    print(f"  decoded shape {decoded.shape}  range [{decoded.min():.3f}, {decoded.max():.3f}]")
    assert decoded.shape[-3:] == (28, 28, 1), f"decoded not 28x28x1: {decoded.shape}"
    assert decoded.min() >= 0.0 and decoded.max() <= 1.0, "decoded pixels outside [0,1]"
    result["stages"]["generate_decode"] = {"shape": list(decoded.shape),
                                           "min": float(decoded.min()), "max": float(decoded.max())}
    wc.checkpoint("after generate/decode", raise_on_over=True)

    # ---------------------------------------------------------------- (6) FID on decoded 28×28
    _stage(wc, "FID — on decoded 28×28 (network-free path)")
    fid, t1, t2 = sm.fid_on_decoded(decoded)
    print(f"  FID={fid:.3f} (term1={t1:.3f}, term2={t2:.3f}) on {decoded.reshape(-1,28,28,1).shape[0]} images")
    assert np.isfinite(fid), f"FID not finite: {fid}"
    result["stages"]["fid"] = {"fid": float(fid), "term1": float(t1), "term2": float(t2)}

    result["wall_seconds"] = wc.elapsed()
    result["passed"] = True
    return result


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--epochs", type=int, default=2, help="dtm.train epochs (few — plumbing)")
    ap.add_argument("--n-train", type=int, default=600, help="filtered Fashion-MNIST train rows")
    ap.add_argument("--n-gen-per-class", type=int, default=4, help="generate samples per class for FID")
    ap.add_argument("--ae-steps", type=int, default=80, help="Stage-A AE pretrain steps")
    ap.add_argument("--cap-seconds", type=float, default=3600.0, help="1hr local-4060 wall cap")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    _banner("Task 12 — local-4060 SMOKE (full-path plumbing verification)")
    t0 = time.time()
    try:
        res = run_smoke(epochs=args.epochs, n_train=args.n_train,
                        n_gen_per_class=args.n_gen_per_class, cap_seconds=args.cap_seconds,
                        ae_steps=args.ae_steps, seed=args.seed)
    except cl.BudgetWall as e:
        _banner(f"SMOKE — BUDGET-WALL STOP (not a failure): {e}")
        print(f"wall used: {time.time() - t0:.1f}s")
        sys.exit(2)
    except Exception as e:  # surface the failing stage
        _banner(f"SMOKE FAILED: {type(e).__name__}: {e}")
        traceback.print_exc()
        sys.exit(1)

    _banner("SMOKE PASS — full path encode→fit→probe→generate→decode→FID completed")
    print(f"wall used: {res['wall_seconds']:.1f}s")
    print(f"per-epoch fit cost: {res['stages']['fit']['per_epoch_seconds']:.1f}s")
    print(f"FID (decoded): {res['stages']['fid']['fid']:.3f}")
    sys.exit(0)


if __name__ == "__main__":
    main()
