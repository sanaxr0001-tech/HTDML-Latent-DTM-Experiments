"""
Import the companion's Stage-C results (exp1 / exp2 / exp3) into Weights & Biases.

Creates one W&B run per (experiment x lambda-point x seed), grouped into three
experiment groups (exp1, exp2, exp3) inside a single project (default: "Htdml").

Design:
  * `build_seed_payloads()` is a PURE transform: run_stage_c.json dict + spec
    -> list of per-seed payload dicts (config / summary / tables / step-series).
    No network, no wandb -> unit-testable (see tests/test_wandb_import.py).
  * `main()` is the thin W&B driver that materializes each payload as a run.

MEASURE-ONLY: this only *mirrors* the recorded outcome tokens and raw metrics into
W&B for visualization. It computes no new science and moves no claim status. The
authoritative per-seed verdict is the driver-emitted `seed_token`; the recomputed
gate legs (`quality_held` / `improvement_met` / ...) are descriptive reproductions
of the a-priori pre-registration predicate and are asserted to agree with that
token (`gate_consistent`) — they are not a new acceptance bar.

Usage:
    python scripts/wandb_import.py --dry-run        # build + print, no upload
    python scripts/wandb_import.py                  # upload to project "Htdml"
    python scripts/wandb_import.py --project Htdml --entity <ent>
    python scripts/wandb_import.py --mode offline   # local-only wandb dir
"""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent

# --- a-priori gate thresholds (verbatim from the exp3 report) ---
BCE_TOL = 1.05          # BCE_joint <= 1.05 * BCE_control
FID_TOL = 1.10          # FID_joint <= 1.10 * FID_control
Q_IMPROVE = 1.25        # lower-quartile Q_joint >= 1.25 * Q_control
TAU_IMPROVE = 0.75      # worst-layer tau_joint <= 0.75 * tau_control
PLATEAU_TOL = 0.05      # |r_grad[50]| <= 0.05
ESS_MIN = 10.0
C_ADQ = 5.0             # L_traj >= C * tau_hat

# probe constants (the "Q ruler") — recorded into config for provenance
PROBE_K, PROBE_B, PROBE_S = 50, 400, 8

# --- the eight source points (one run_stage_c.json each, 2 seeds inside) ---
RUN_SPECS = [
    dict(
        exp_short="exp1", exp_full="exp1-paid-cal-artifact", lam=1.0,
        json_path="experiments/exp1-paid-cal-artifact/run_stage_c.json",
        report_path="experiments/exp1-paid-cal-artifact/report.md",
        log_path="experiments/exp1-paid-cal-artifact/run.log",
        stage_b_acp=None, stage_b_source="exp1 (Stage A+B trained in-run; not persisted)",
    ),
    dict(
        exp_short="exp2", exp_full="exp2-cal-gate-fix", lam=1.0,
        json_path="experiments/exp2-cal-gate-fix/artifacts/run_stage_c.json",
        report_path="experiments/exp2-cal-gate-fix/report.md",
        log_path="experiments/exp2-cal-gate-fix/artifacts/run.log",
        stage_b_acp="experiments/exp2-cal-gate-fix/artifacts/checkpoints/seed{seed}/stage_b/autocorrelations.json",
        stage_b_source="exp2 (Stage A+B trained in-run; persisted)",
    ),
    dict(
        exp_short="exp3", exp_full="exp3-lambda-sweep", lam=0.5,
        json_path="experiments/exp3-lambda-sweep/artifacts/lam0.5/run_stage_c.json",
        report_path="experiments/exp3-lambda-sweep/report.md",
        log_path="experiments/exp3-lambda-sweep/artifacts/lam0.5.run.log",
        stage_b_acp=None, stage_b_source="exp2 checkpoints (RESUME_FROM; Stage A+B skipped)",
    ),
    dict(
        exp_short="exp3", exp_full="exp3-lambda-sweep", lam=0.3,
        json_path="experiments/exp3-lambda-sweep/artifacts/lam0.3/run_stage_c.json",
        report_path="experiments/exp3-lambda-sweep/report.md",
        log_path="experiments/exp3-lambda-sweep/artifacts/lam0.3.run.log",
        stage_b_acp=None, stage_b_source="exp2 checkpoints (RESUME_FROM; Stage A+B skipped)",
    ),
    dict(
        exp_short="exp3", exp_full="exp3-lambda-sweep", lam=0.1,
        json_path="experiments/exp3-lambda-sweep/artifacts/lam0.1/run_stage_c.json",
        report_path="experiments/exp3-lambda-sweep/report.md",
        log_path="experiments/exp3-lambda-sweep/artifacts/lam0.1.run.log",
        stage_b_acp=None, stage_b_source="exp2 checkpoints (RESUME_FROM; Stage A+B skipped)",
    ),
]


# ----------------------------- helpers -----------------------------

def _finite(x):
    """Sanitize a float for W&B summary: inf/nan -> None."""
    try:
        xf = float(x)
    except (TypeError, ValueError):
        return None
    return xf if math.isfinite(xf) else None


def _quality_metric(summary, key, raw):
    """Store a BCE/FID value W&B-safely, preserving the divergence signal.

    inf/nan can't be a W&B summary scalar, but collapsing them to None hides that
    exp1's joint arm *diverged* (Infinity) vs merely being absent. Keep the numeric
    None but flag it explicitly with `<key>_diverged` + a raw string.
    """
    summary[key] = _finite(raw)
    if summary[key] is None and raw is not None:
        summary[f"{key}_diverged"] = True
        summary[f"{key}_raw"] = str(raw)  # e.g. "inf"


def _lower_quartile(vals):
    return float(np.percentile(np.asarray(vals, dtype=float), 25))


def load_run_json(path):
    """json.load handles Infinity/NaN natively (exp1 has Infinity BCE/FID)."""
    with open(REPO / path) as fh:
        return json.load(fh)


def _lam_slug(lam):
    return str(lam).replace(".", "_")


# ----------------------------- pure transform -----------------------------

def build_seed_payloads(data, spec):
    """Map one run_stage_c.json dict (+ its spec) to a list of per-seed payloads."""
    const = data["constants"]
    prov = data["provenance"]
    budget = data["budget"]
    lam = spec["lam"]
    payloads = []

    for s in data["seeds"]:
        seed = s["seed"]
        joint = s.get("joint_layers") or []
        control = s.get("control_layers") or []
        has_stage_c = len(joint) > 0
        recon = s.get("l_traj_reconfirm", {})
        tau_hat_worst = recon.get("tau_hat_worst")
        cal_stable = bool(recon.get("cal_stable", False))

        name = f"{spec['exp_short']}-lam{lam}-seed{seed}"
        rid = f"{spec['exp_short']}-lam{_lam_slug(lam)}-s{seed}"

        config = {
            "experiment": spec["exp_full"],
            "experiment_short": spec["exp_short"],
            "seed": seed,
            "lambda_joint": lam,
            "run_outcome": data["outcome"],
            "seed_token": s["token"],
            "two_seed_both_pass": data["two_seed"]["both_pass"],
            "stage_c_reached": has_stage_c,
            "cal_stable": cal_stable,
            "git_sha": prov.get("git_sha"),
            "env_freeze": prov.get("env_freeze"),
            "jax_backend": prov.get("jax_backend"),
            "is_patch_live": prov.get("is_patch_live"),
            "gpu_h_total": budget.get("gpu_h_total"),
            "budget_wall": budget.get("budget_wall"),
            "GPU_H_CAP": const.get("GPU_H_CAP"),
            "ESS_min": const.get("ESS_min"),
            "C": const.get("C"),
            "L_traj": const.get("L_traj"),
            "N_chains": const.get("N_chains"),
            "N_R": const.get("N_R"),
            "probe_K": PROBE_K, "probe_B": PROBE_B, "probe_s": PROBE_S,
            "dtm_config": "44_12", "reverse_steps": 4,
            "stage_b_source": spec.get("stage_b_source"),
            "source_json": spec["json_path"],
            # --- l_traj_reconfirm provenance (else silently dropped, esp. for exp1) ---
            "tau_hat_layers": recon.get("tau_hat_layers"),
            "reconfirm_failed_layer": recon.get("failed_layer"),
            "reconfirm_failed_axes": recon.get("failed_axes"),
            "reconfirm_adjusted": recon.get("adjusted"),
            "reconfirm_affordable": recon.get("affordable"),
            "L_traj_adequate": recon.get("L_traj_adequate"),
            "L_traj_frozen": recon.get("L_traj_frozen"),
        }

        summary = {"tau_hat_worst": _finite(tau_hat_worst)}
        for key in ("bce_joint", "fid_joint", "bce_control", "fid_control"):
            _quality_metric(summary, key, s.get(key))

        # per-layer deterministic reconfirm tau (the only per-layer mixing diagnostic
        # for a cal-fail run like exp1, which has no Stage-C / cal-curve layers).
        reconfirm_rows = []
        thl = recon.get("tau_hat_layers") or []
        fax = recon.get("failed_axes") or []
        for li, tau in enumerate(thl):
            axes = ",".join(fax[li]) if li < len(fax) and fax[li] else ""
            reconfirm_rows.append([li, tau, axes])

        per_layer_rows, reject_rows, cal_curve_rows, layer_series = [], [], [], []

        if has_stage_c:
            qj = [L["Q_struct_perp"] for L in joint]
            qc = [L["Q_struct_perp"] for L in control]
            tj = [L["tau_int_Y"] for L in joint]
            tc = [L["tau_int_Y"] for L in control]
            ej = [L["ESS_hat"] for L in joint]
            ec = [L["ESS_hat"] for L in control]
            rj = [abs(L["r_grad[50]"]) for L in joint]
            rc = [abs(L["r_grad[50]"]) for L in control]

            lq_qj, lq_qc = _lower_quartile(qj), _lower_quartile(qc)
            worst_tau_j, worst_tau_c = max(tj), max(tc)
            worst_ess_j, worst_ess_c = min(ej), min(ec)
            max_rgrad = max(max(rj), max(rc))

            bce_j, bce_c = s["bce_joint"], s["bce_control"]
            fid_j, fid_c = s["fid_joint"], s["fid_control"]
            quality_held = (bce_j <= BCE_TOL * bce_c) and (fid_j <= FID_TOL * fid_c)
            improvement_met = (lq_qj >= Q_IMPROVE * lq_qc) or (worst_tau_j <= TAU_IMPROVE * worst_tau_c)
            ess_nondeg = worst_ess_j >= worst_ess_c
            plateau_ok = max_rgrad <= PLATEAU_TOL
            traj_resolved = cal_stable and (const["L_traj"] >= C_ADQ * tau_hat_worst)
            computed_pass = quality_held and improvement_met and ess_nondeg and plateau_ok and traj_resolved

            n_blocks = len(s.get("reject_log", []))
            n_rejects = sum(1 for r in s.get("reject_log", []) if r.get("reject"))

            summary.update({
                "lq_Q_joint": lq_qj, "lq_Q_control": lq_qc,
                "lq_Q_ratio": lq_qj / lq_qc,
                "worst_tau_joint": worst_tau_j, "worst_tau_control": worst_tau_c,
                "worst_tau_ratio": worst_tau_j / worst_tau_c,
                "worst_ESS_joint": worst_ess_j, "worst_ESS_control": worst_ess_c,
                "max_abs_rgrad": max_rgrad,
                "bce_excess_pct": 100.0 * (bce_j - bce_c) / bce_c,
                "fid_excess_pct": 100.0 * (fid_j - fid_c) / fid_c,
                "n_blocks": n_blocks, "n_rejects": n_rejects,
                "quality_held": quality_held,
                "improvement_met": improvement_met,
                "ess_nondeg": ess_nondeg,
                "plateau_ok": plateau_ok,
                "traj_resolved": traj_resolved,
                "computed_pass": computed_pass,
                "gate_consistent": computed_pass == (s["token"] == "HTDML-MARGIN-POSITIVE"),
            })

            for i in range(len(joint)):
                per_layer_rows.append([
                    i, qj[i], qc[i], qj[i] / qc[i], tj[i], tc[i],
                    ej[i], ec[i], joint[i]["gradient_norm"], control[i]["gradient_norm"],
                    joint[i]["r_grad[50]"], control[i]["r_grad[50]"],
                ])
                pt = {
                    "layer": i,
                    "layer_metrics/Q_joint": qj[i], "layer_metrics/Q_control": qc[i],
                    "layer_metrics/Q_ratio": qj[i] / qc[i],
                    "layer_metrics/tau_joint": tj[i], "layer_metrics/tau_control": tc[i],
                    "layer_metrics/ESS_joint": ej[i], "layer_metrics/ESS_control": ec[i],
                }
                if i < len(thl):
                    pt["layer_metrics/tau_hat_reconfirm"] = thl[i]
                layer_series.append(pt)

            for r in s.get("reject_log", []):
                reject_rows.append([
                    r.get("epoch"), bool(r.get("reject")),
                    r.get("reason", ""), r.get("encoder_lr"),
                ])

        for li, curve in enumerate(recon.get("cal_curves", []) or []):
            for pt in curve:
                cal_curve_rows.append([
                    li, pt.get("L"), pt.get("warm"), pt.get("tau_max"), pt.get("T_O"),
                    bool(pt.get("self_consistent")), pt.get("step_class"),
                    pt.get("rel_tau"), pt.get("abs_dtau"), pt.get("dT"),
                    pt.get("dS_l1"), pt.get("small_tau"),
                ])

        tags = list(dict.fromkeys([
            spec["exp_short"], data["outcome"], s["token"], f"lambda={lam}",
            "stage-c" if has_stage_c else "cal-gate-fail",
        ]))

        payloads.append({
            "name": name, "id": rid, "group": spec["exp_short"],
            "job_type": f"lambda={lam}", "tags": tags,
            "config": config, "summary": summary,
            "per_layer_rows": per_layer_rows, "reject_rows": reject_rows,
            "cal_curve_rows": cal_curve_rows, "reconfirm_rows": reconfirm_rows,
            "layer_series": layer_series,
            "spec": spec, "seed": seed,
        })

    return payloads


def iter_all_payloads():
    out = []
    for spec in RUN_SPECS:
        out.extend(build_seed_payloads(load_run_json(spec["json_path"]), spec))
    return out


# ----------------------------- W&B driver -----------------------------

PER_LAYER_COLS = [
    "layer", "Q_joint", "Q_control", "Q_ratio", "tau_joint", "tau_control",
    "ESS_joint", "ESS_control", "gradnorm_joint", "gradnorm_control",
    "rgrad50_joint", "rgrad50_control",
]
REJECT_COLS = ["block_epoch", "reject", "reason", "encoder_lr"]
CAL_COLS = [
    "layer", "L", "warm", "tau_max", "T_O", "self_consistent", "step_class",
    "rel_tau", "abs_dtau", "dT", "dS_l1", "small_tau",
]
RECON_COLS = ["layer", "tau_hat_layer", "failed_axes"]

GROUP_NOTES = {
    "exp1": "First paid H200 run -> Q-CALIBRATION-FAIL (both seeds): a cal-gate "
            "artifact at small tau-hat, NOT bad mixing. Stage C never reached.",
    "exp2": "Cal-gate repair (TAU_ABS_FLOOR) -> gate passed -> Stage C ran -> "
            "HTDML-MARGIN-NEGATIVE at lambda=1.0: quality held, steering over-steered (Q_drop).",
    "exp3": "Lower-lambda sweep {0.1,0.3,0.5} on exp2's Stage-B -> HTDML-MARGIN-NEGATIVE "
            "at every lambda. Per-seed tau-passes not distinguishable from control-denominator "
            "noise (Fisher p~0.46). Terminal verdict finalized 2026-06-27.",
}


def _attach_stage_b_acp(run, spec, seed):
    """exp2 only: log the persisted Stage-B ACP autocorrelation curve (4 layers x 201 epochs)."""
    tmpl = spec.get("stage_b_acp")
    if not tmpl:
        return 0
    path = REPO / tmpl.format(seed=seed)
    if not path.exists():
        return 0
    layers = json.loads(path.read_text())  # list[ {epoch_str: value} ] per layer
    run.define_metric("epoch")
    run.define_metric("stage_b_acp/*", step_metric="epoch")
    n_epochs = max(len(d) for d in layers)
    for e in range(n_epochs):
        payload = {"epoch": e}
        for li, d in enumerate(layers):
            if str(e) in d:
                payload[f"stage_b_acp/layer{li}"] = d[str(e)]
        run.log(payload)
    return n_epochs


def _save_files(run, spec):
    for key in ("json_path", "report_path", "log_path"):
        p = REPO / spec[key]
        if p.exists():
            try:
                run.save(str(p), base_path=str(p.parent), policy="now")
            except Exception as exc:  # noqa: BLE001 - never let a file hiccup kill metrics
                print(f"   ! could not attach {p.name}: {exc}")


def upload_payload(wandb, payload, project, entity, mode):
    import wandb as _wb  # noqa
    spec = payload["spec"]
    run = wandb.init(
        project=project, entity=entity, mode=mode,
        name=payload["name"], id=payload["id"], group=payload["group"],
        job_type=payload["job_type"], tags=payload["tags"],
        config=payload["config"], notes=GROUP_NOTES.get(payload["group"], ""),
        reinit=True, resume="allow",
    )

    if payload["layer_series"]:
        run.define_metric("layer")
        run.define_metric("layer_metrics/*", step_metric="layer")
        for pt in payload["layer_series"]:
            run.log(pt)

    n_acp = _attach_stage_b_acp(run, spec, payload["seed"])

    logged_tables = {}
    if payload["per_layer_rows"]:
        logged_tables["tables/per_layer"] = wandb.Table(
            columns=PER_LAYER_COLS, data=payload["per_layer_rows"])
    if payload["reject_rows"]:
        logged_tables["tables/reject_log"] = wandb.Table(
            columns=REJECT_COLS, data=payload["reject_rows"])
    if payload["cal_curve_rows"]:
        logged_tables["tables/cal_curves"] = wandb.Table(
            columns=CAL_COLS, data=payload["cal_curve_rows"])
    if payload["reconfirm_rows"]:
        logged_tables["tables/reconfirm_tau"] = wandb.Table(
            columns=RECON_COLS, data=payload["reconfirm_rows"])
    if logged_tables:
        run.log(logged_tables)

    for k, v in payload["summary"].items():
        run.summary[k] = v

    _save_files(run, spec)
    run.finish()
    return n_acp


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--project", default="Htdml")
    ap.add_argument("--entity", default=None)
    ap.add_argument("--mode", default="online", choices=["online", "offline", "disabled"])
    ap.add_argument("--dry-run", action="store_true",
                    help="build payloads and print a summary; do not touch wandb")
    args = ap.parse_args()

    payloads = iter_all_payloads()
    print(f"Built {len(payloads)} run payloads from {len(RUN_SPECS)} source files.")
    for p in payloads:
        st = p["summary"]
        extra = ""
        if p["config"]["stage_c_reached"]:
            extra = (f"  lqQ={st['lq_Q_ratio']:.2f}x  tau={st['worst_tau_ratio']:.2f}x  "
                     f"ESSj/c={st['worst_ESS_joint']:.1f}/{st['worst_ESS_control']:.1f}  "
                     f"rej={st['n_rejects']}/{st['n_blocks']}")
        print(f"  {p['name']:24s} {p['config']['seed_token']:22s}{extra}")

    if args.dry_run:
        print("\n[dry-run] no wandb writes.")
        return

    import wandb
    print(f"\nUploading to project '{args.project}'"
          f"{(' (entity ' + args.entity + ')') if args.entity else ''} mode={args.mode} ...")
    total_acp = 0
    for p in payloads:
        n_acp = upload_payload(wandb, p, args.project, args.entity, args.mode)
        total_acp += n_acp
        tail = f" (+{n_acp} Stage-B ACP epochs)" if n_acp else ""
        print(f"  uploaded {p['name']}{tail}")
    print(f"\nDone: {len(payloads)} runs, {total_acp} Stage-B ACP epoch-points logged.")


if __name__ == "__main__":
    main()
