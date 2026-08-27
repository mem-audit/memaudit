"""Provenance self-hash: write_report sidecar, memaudit verify, fingerprints."""

from __future__ import annotations

import json

import pytest

from memaudit.audit import run_audit
from memaudit.canaries import generate_canaries
from memaudit.cli import main
from memaudit.exceptions import MemauditAuditError
from memaudit.injection import inject
from memaudit.report import (
    build_report,
    compute_report_sha256,
    sidecar_path,
    verify_report,
    write_report,
)
from memaudit.utils import dataset_fingerprint, environment_versions, model_fingerprint


def _tiny_report():
    return build_report(
        seeds={"inject": 0},
        canary_manifest_hash="abc",
        model_info={"class": "TinyCausalLM"},
        adapter_info=None,
        ref_info={"mode": "none"},
        membership={"tpr_at_1pct_fpr": None, "headline_valid": False, "n_members": 2, "n_controls": 2},
        regurgitation={"overall": {"rate": 0.0, "n": 2, "n_regurgitated": 0}, "by_tier": {}},
        negative_controls={"n": 2, "regurgitation_rate": 0.0},
        real_records=None,
        preflight=None,
    )


def test_write_report_stamps_self_hash_and_sidecar(tmp_path):
    report = _tiny_report()
    dest = write_report(report, tmp_path / "r.json")
    assert report["report_sha256"]
    side = sidecar_path(dest)
    assert side.is_file()
    first_line = side.read_text(encoding="utf-8").splitlines()[0]
    assert first_line.split()[0] == report["report_sha256"]
    assert first_line.split()[1] == "r.json"
    loaded = json.loads(dest.read_text(encoding="utf-8"))
    assert loaded["report_sha256"] == report["report_sha256"]
    # hash is over content minus the hash itself
    assert compute_report_sha256(loaded) == loaded["report_sha256"]


def test_self_hash_covers_post_build_mutations(tmp_path):
    report = _tiny_report()
    report["benchmark"] = {"scale": "tiny"}
    dest = write_report(report, tmp_path / "r.json")
    loaded = json.loads(dest.read_text(encoding="utf-8"))
    assert loaded["benchmark"]["scale"] == "tiny"
    assert compute_report_sha256(loaded) == loaded["report_sha256"]
    assert verify_report(dest)["ok"] is True


def test_verify_detects_tampering(tmp_path):
    report = _tiny_report()
    dest = write_report(report, tmp_path / "r.json")
    assert verify_report(dest)["ok"] is True
    obj = json.loads(dest.read_text(encoding="utf-8"))
    obj["membership"]["tpr_at_1pct_fpr"] = 0.99
    dest.write_text(json.dumps(obj), encoding="utf-8")
    result = verify_report(dest)
    assert result["ok"] is False
    assert any(c["ok"] is False for c in result["checks"])


def test_verify_detects_sidecar_mismatch(tmp_path):
    report = _tiny_report()
    dest = write_report(report, tmp_path / "r.json")
    side = sidecar_path(dest)
    side.write_text("deadbeef" * 8 + f"  {dest.name}\n", encoding="utf-8")
    result = verify_report(dest)
    assert result["ok"] is False


def test_verify_missing_hash_and_missing_file(tmp_path):
    dest = tmp_path / "old.json"
    dest.write_text(json.dumps({"schema_version": "1.0.0"}), encoding="utf-8")
    result = verify_report(dest)
    assert result["ok"] is False
    assert "re-generate" in result["checks"][0]["detail"]
    with pytest.raises(MemauditAuditError):
        verify_report(tmp_path / "nope.json")
    bad = tmp_path / "bad.json"
    bad.write_text("{not json", encoding="utf-8")
    with pytest.raises(MemauditAuditError):
        verify_report(bad)


def test_cli_verify_exit_codes(tmp_path, capsys):
    report = _tiny_report()
    dest = write_report(report, tmp_path / "r.json")
    assert main(["verify", str(dest)]) == 0
    out = capsys.readouterr().out
    assert "report_sha256" in out
    obj = json.loads(dest.read_text(encoding="utf-8"))
    obj["phone_home"] = True
    dest.write_text(json.dumps(obj), encoding="utf-8")
    assert main(["verify", str(dest)]) == 1
    assert main(["verify", str(tmp_path / "missing.json")]) == 2


def test_cli_report_renders_annex(tmp_path, capsys):
    report = _tiny_report()
    dest = write_report(report, tmp_path / "r.json")
    out_md = tmp_path / "annex.md"
    assert main(["report", str(dest), "--output", str(out_md)]) == 0
    text = out_md.read_text(encoding="utf-8")
    assert "EDPB Opinion 28/2024" in text
    # --annex alias
    assert main(["report", "--annex", str(dest)]) == 0
    assert "EDPB" in capsys.readouterr().out


def test_run_audit_provenance_fields(tokenizer, tiny_model, tmp_path):
    host = [{"text": f"sentence {i} about weather"} for i in range(12)]
    cans = generate_canaries(tokenizer, n=2, n_controls=3, family="random", seed=0, secret_len=25)
    ds, manifest = inject(host, cans, fmt="text", seed=0, include_prob=1.0)
    report = run_audit(
        model=tiny_model,
        tokenizer=tokenizer,
        manifest=manifest,
        dataset=ds,
        ref="none",
        real_sample=0,
        output_path=tmp_path / "r.json",
        skip_generation=True,
        release_context="public-api",
    )
    prov = report["provenance"]
    assert prov["canary_manifest_sha256"] == manifest["manifest_hash"]
    env = prov["environment"]
    assert env["python"] and env["torch"] and env["transformers"]
    fp = prov["dataset_fingerprint"]
    assert fp["n_rows"] == len(list(ds))
    assert fp["first_record_sha256"] and fp["last_record_sha256"]
    mf = prov["model_fingerprint"]
    assert mf["config_sha256"]
    assert mf["n_parameters"] and mf["n_parameters"] > 0
    cfg = prov["resolved_config"]
    assert cfg["release_context"] == "public-api"
    assert cfg["headline_attack_predeclared"]
    assert cfg["fpr_target"] == 0.01
    assert report["release_context"]["declared"] == "public-api"
    assert "signing" in prov and "sigstore" in prov["signing"]["note"]
    # written file verifies
    assert verify_report(tmp_path / "r.json")["ok"] is True


def test_fingerprint_helpers(tokenizer, tiny_model, tmp_path):
    rows = [{"text": "a"}, {"text": "b"}, {"text": "c"}]
    fp = dataset_fingerprint(rows)
    assert fp["n_rows"] == 3
    assert fp["first_record_sha256"] != fp["last_record_sha256"]
    f = tmp_path / "d.jsonl"
    f.write_text('{"text": "a"}\n', encoding="utf-8")
    fp2 = dataset_fingerprint(rows, path=f)
    assert fp2["file"]["size_bytes"] > 0
    assert fp2["file"]["sha256"]
    assert dataset_fingerprint(None) is None
    mf = model_fingerprint(tiny_model)
    assert mf["weight_files"] is None
    assert "in-memory" in mf["weight_files_note"]
    # weight files hashed when a local dir is given
    wdir = tmp_path / "ckpt"
    wdir.mkdir()
    (wdir / "adapter_model.safetensors").write_bytes(b"\x00" * 64)
    mf2 = model_fingerprint(tiny_model, model_path=wdir)
    assert mf2["weight_files"][0]["name"] == "adapter_model.safetensors"
    assert mf2["weight_files"][0]["sha256"]
    vers = environment_versions()
    assert vers["memaudit"]
