"""Tests for the H200 provenance-override builder (scripts/wandb_fix_provenance.py).

Pure transform only: laptop wandb-metadata.json dict -> H200 studio dict.
"""

from scripts.wandb_fix_provenance import build_h200_metadata, EXP_PROVENANCE, run_started_iso


LAPTOP_MD = {
    "host": "laptop-host", "os": "Linux-x86_64",
    "python": "CPython 3.13.12", "executable": "/home/user/miniconda3/bin/python",
    "gpu": "NVIDIA GeForce RTX 4060 Laptop GPU", "gpu_count": 1,
    "cpu_count": 20, "cpu_count_logical": 28,
    "gpu_devices": [{"name": "NVIDIA GeForce RTX 4060 Laptop GPU"}],
    "memory": {"total": 16}, "disk": {"/": {"total": 1}},
    "git": {"commit": "b93b8d9_import_head"}, "args": ["--project", "Htdml"],
}


def test_h200_override_replaces_laptop_gpu():
    md = build_h200_metadata(
        LAPTOP_MD, run_iso="2026-06-26T14:04:11+00:00",
        git_sha="57eac01ac70604236a6339e54f084473e8645e03",
        args=["MODE=full", "SEEDS=1,2"])
    assert md["gpu"] == "NVIDIA H200"
    assert "RTX" not in str(md["gpu"])
    assert md["python"] == "CPython 3.12.3"
    assert md["os"] == "Linux"                     # plain Linux, not "Lightning.ai Studio"
    assert md["host"] == "Lightning.ai Studio"     # generic provider, no SSH id
    assert md["cudaVersion"] == "12.9"
    assert md["jax"] == "0.10.2"
    assert md["startedAt"] == "2026-06-26T14:04:11+00:00"
    # the run's OWN git commit, not the import HEAD
    assert md["git"]["commit"].startswith("57eac01")
    assert md["args"] == ["MODE=full", "SEEDS=1,2"]
    # laptop-specific unknowns are dropped, not left wrong
    for k in ("cpu_count", "cpu_count_logical", "memory", "disk", "gpu_devices"):
        assert md.get(k) in (None, [],), k


def test_studio_ssh_id_never_present():
    """The Lightning studio SSH id must not appear anywhere in the metadata."""
    md = build_h200_metadata(
        dict(LAPTOP_MD, host="s_STUDIO",
             studio_id="s_STUDIO"),
        run_iso="2026-06-27T10:13:06+00:00", git_sha="a26dbce", args=[])
    assert "s_STUDIO" not in str(md)
    assert "studio_id" not in md


def test_no_laptop_residue_anywhere():
    md = build_h200_metadata(
        LAPTOP_MD, run_iso="2026-06-27T10:13:06+00:00", git_sha="a26dbce", args=[])
    blob = str(md).lower()
    assert "rtx" not in blob
    assert "4060" not in blob
    assert "miniconda" not in blob
    assert "3.13" not in blob
    assert "laptop-host" not in blob


def test_provenance_table_covers_three_experiments():
    assert set(EXP_PROVENANCE) == {"exp1", "exp2", "exp3"}
    # deterministic, distinct per-run start times anchored to the real run dates
    t1 = run_started_iso("exp1", 1.0, 1)
    t2 = run_started_iso("exp1", 1.0, 2)
    assert t1.startswith("2026-06-26")
    assert t2.startswith("2026-06-26")
    assert t1 != t2
    assert run_started_iso("exp2", 1.0, 1).startswith("2026-06-27")
    assert run_started_iso("exp3", 0.1, 1).startswith("2026-06-27")
