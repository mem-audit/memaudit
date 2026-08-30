"""WS2/WS3: profiles, tier curve, calibration stability, protocol, held_out."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from memaudit.audit import run_audit
from memaudit.callback import MemorizationAuditCallback
from memaudit.canaries import generate_canaries
from memaudit.constants import (
    DEFAULT_N,
    DEFAULT_N_CONTROLS,
    DEFAULT_REPETITIONS,
    infer_audit_profile_name,
    resolve_audit_profile,
)
from memaudit.injection import inject
from memaudit.stats import bootstrap_calibration_stability, membership_by_repetition


def _tiny_manifest(tokenizer, n=4, n_controls=6, reps=(1, 4, 16), profile=None, seed=0):
    host = [{"text": f"ordinary training sentence number {i} about weather"} for i in range(16)]
    cans = generate_canaries(
        tokenizer,
        n=n,
        n_controls=n_controls,
        family="random",
        repetitions=reps,
        seed=seed,
        secret_len=25,
        profile=profile,
    )
    return inject(host, cans, fmt="text", seed=seed, include_prob=1.0)


def test_membership_by_repetition_decomposes_pooled_threshold():
    # Synthetic tier curve (1x 0/34, 4x 2/33, 16x 16/33) shaped like the
    # archived v0.1 uniform_vocab run; the values are a fixture, not a claim
    # about the currently shipped report.
    threshold = 3.626
    rows = []
    for _ in range(34):
        rows.append({"included": True, "repetitions": 1, "scores": {"headline_score": 0.0}})
    for i in range(33):
        rows.append(
            {
                "included": True,
                "repetitions": 4,
                "scores": {"headline_score": 4.0 if i < 2 else 0.0},
            }
        )
    for i in range(33):
        rows.append(
            {
                "included": True,
                "repetitions": 16,
                "scores": {"headline_score": 4.0 if i < 16 else 0.0},
            }
        )
    out = membership_by_repetition(rows, threshold)
    assert out["1"]["n"] == 34 and out["1"]["detected"] == 0
    assert out["4"]["n"] == 33 and out["4"]["detected"] == 2
    assert out["16"]["n"] == 33 and out["16"]["detected"] == 16
    assert out["pooled"]["n"] == 100 and out["pooled"]["detected"] == 18
    assert out["1"]["meaning"] == "single-exposure probe"
    assert out["16"]["meaning"] == "high-exposure stress"
    assert out["pooled"]["meaning"] == "powered-audit headline"
    assert 0.0 <= out["1"]["ci_low"] <= out["1"]["ci_high"] <= 1.0


def test_bootstrap_calibration_stability_moves_threshold():
    members = [2.0, 1.5, 0.5, -0.2]
    controls = [0.0, -0.1, -0.4, -1.0, 0.2, -0.3]
    out = bootstrap_calibration_stability(members, controls, fpr=0.01, n_bootstrap=20, seed=0)
    assert out["n_bootstrap"] == 20
    assert "threshold" in out and "tpr" in out
    assert out["threshold"]["min"] <= out["threshold"]["max"]
    assert 0.0 <= out["tpr"]["min"] <= out["tpr"]["max"] <= 1.0


def test_infer_powered_and_routine_shapes():
    assert infer_audit_profile_name(
        requested_members=32, n_controls=100, repetitions=(1, 4, 16)
    ) == "routine"
    assert infer_audit_profile_name(
        requested_members=100, n_controls=200, repetitions=(1, 4, 16)
    ) == "powered"
    assert infer_audit_profile_name(
        requested_members=4, n_controls=6, repetitions=(1, 4, 16)
    ) == "custom"
    spec = resolve_audit_profile("smoke")
    assert spec["refuse_headline"] is True
    assert spec["exploratory"] is True


def test_generate_canaries_profile_smoke_and_powered(tokenizer):
    smoke = generate_canaries(tokenizer, profile="smoke", family="random", seed=0)
    assert sum(c.role == "candidate" for c in smoke) == 8
    assert sum(c.role == "control" for c in smoke) == 16
    assert all((c.metadata or {}).get("audit_profile") == "smoke" for c in smoke)
    powered = generate_canaries(
        tokenizer, n=2, n_controls=2, profile="powered", family="random", seed=1, secret_len=25
    )
    # explicit counts win
    assert len(powered) == 4
    assert all((c.metadata or {}).get("audit_profile") == "powered" for c in powered)


def test_generate_canaries_default_counts_unchanged(tokenizer):
    cans = generate_canaries(tokenizer, family="random", seed=0, secret_len=25)
    assert sum(c.role == "candidate" for c in cans) == DEFAULT_N
    assert sum(c.role == "control" for c in cans) == DEFAULT_N_CONTROLS
    assert {c.repetitions for c in cans if c.role == "candidate"} == set(DEFAULT_REPETITIONS)


def test_run_audit_emits_ws2_fields(tokenizer, tiny_model, tmp_path):
    ds, manifest = _tiny_manifest(tokenizer)
    report = run_audit(
        tiny_model,
        tokenizer,
        manifest,
        dataset=ds,
        ref="none",
        real_sample=8,
        skip_generation=False,
        output_path=tmp_path / "r.json",
    )
    mem = report["membership"]
    assert mem["scorer"]["name"]
    assert mem["scorer"]["version"]
    assert "by_repetition" in mem
    assert "pooled" in mem["by_repetition"]
    assert "1" in mem["by_repetition"]
    assert mem["target_fpr"] == 0.01
    assert mem["calibration_stability"]
    assert report["audit_profile"]["name"] in {"custom", "smoke", "routine", "powered"}
    assert report["canaries"]["requested_family"]
    assert report["canaries"]["actual_generator"]
    assert report["canaries"]["requested_members"] == 4
    assert report["regurgitation"]["prefix_policy"]["fractions"]
    assert report["regurgitation"]["decoding"]["strategy"] == "greedy"
    assert report["regurgitation"]["match_rule"] == "exact"
    assert "under this prefix/decoding/exact-match protocol" in report["regurgitation"]["detected"]["wording"]
    assert "no extraction risk" in report["regurgitation"]["note"]
    assert report["audit_scope"]["requested_family"]
    assert report["audit_scope"]["actual_generator"]
    sl = report["real_records"]["set_level"]
    assert sl["inferential"] is False
    assert sl["kind"] == "descriptive_ranking_only"
    assert sl["p_value"] is None
    assert "member-vs-nonmember" in sl["note"]
    blob = str(sl).lower()
    assert "proves membership" not in blob


def test_smoke_profile_refuses_headline_even_with_enough_controls(tokenizer, tiny_model):
    ds, manifest = _tiny_manifest(tokenizer, n=4, n_controls=100, reps=(1,), profile="smoke")
    report = run_audit(
        tiny_model,
        tokenizer,
        manifest,
        dataset=ds,
        ref="none",
        real_sample=0,
        skip_generation=True,
        profile="smoke",
    )
    assert report["audit_profile"]["name"] == "smoke"
    assert report["audit_profile"]["refuse_headline"] is True
    assert report["audit_profile"]["exploratory"] is True
    assert report["membership"]["headline_valid"] is False
    assert report["membership"]["tpr_at_1pct_fpr"] is None
    assert any("refused" in w.lower() or "exploratory" in w.lower() for w in report.get("audit_warnings") or [])


def test_powered_profile_emits_calibration_stability(tokenizer, tiny_model):
    ds, manifest = _tiny_manifest(tokenizer, n=4, n_controls=6, profile="powered")
    report = run_audit(
        tiny_model,
        tokenizer,
        manifest,
        dataset=ds,
        ref="none",
        real_sample=0,
        skip_generation=True,
        profile="powered",
    )
    assert report["audit_profile"]["name"] == "powered"
    cal = report["membership"]["calibration_stability"]
    assert cal is not None
    assert cal["kind"] == "control_resample_threshold"
    assert cal["n_bootstrap"] >= 1
    assert "threshold" in cal and "tpr" in cal


def test_callback_held_out_is_inferential(tokenizer, tiny_model, tmp_path):
    ds, manifest = _tiny_manifest(tokenizer, n=3, n_controls=3, seed=3)
    held = [{"text": f"held out weather note {i} never trained"} for i in range(8)]
    args = SimpleNamespace(
        output_dir=str(tmp_path),
        max_length=1024,
        packing_strategy="bfd",
        packing=False,
        completion_only_loss=False,
        assistant_only_loss=False,
        learning_rate=2e-5,
        num_train_epochs=1,
    )
    trainer = SimpleNamespace(
        model=tiny_model,
        train_dataset=ds,
        args=args,
        processing_class=tokenizer,
        tokenizer=None,
        accelerator=None,
    )
    cb = MemorizationAuditCallback(
        trainer=trainer,
        manifest=manifest,
        real_sample=6,
        ref="none",
        skip_generation=True,
        held_out=held,
    )
    assert cb.held_out is held
    control = SimpleNamespace()
    state = SimpleNamespace(is_world_process_zero=True)
    cb.on_train_begin(args, state, control, model=tiny_model, processing_class=tokenizer)
    cb.on_train_end(args, state, control, model=tiny_model, processing_class=tokenizer)
    real = cb.report["real_records"]
    assert real["comparison_population"] == "held_out"
    sl = real["set_level"]
    assert sl["kind"] == "inferential_member_vs_nonmember"
    assert sl["inferential"] is True
    assert "p_value" in sl
    assert "any individual record" in sl["note"]
    assert "proves" not in sl["note"].lower()


def test_fallback_does_not_claim_member_vs_nonmember(tokenizer, tiny_model):
    ds, manifest = _tiny_manifest(tokenizer)
    report = run_audit(
        tiny_model,
        tokenizer,
        manifest,
        dataset=ds,
        ref="none",
        real_sample=8,
        skip_generation=True,
    )
    sl = report["real_records"]["set_level"]
    assert sl["inferential"] is False
    assert sl.get("p_value") is None
    note = sl["note"].lower()
    assert "descriptive" in note
    assert "member-vs-nonmember" in note
    assert "mean_members" not in sl
    assert report["real_records"]["comparison_population"] == "training_split"


def test_cli_exposes_profile_and_held_out():
    from memaudit.cli import build_parser

    parser = build_parser()
    ns = parser.parse_args(
        [
            "audit",
            "--model",
            "./out",
            "--canary-set",
            "./m.json",
            "--profile",
            "smoke",
            "--held-out",
            "./hold.jsonl",
        ]
    )
    assert ns.profile == "smoke"
    assert ns.held_out == "./hold.jsonl"
