"""Shared audit engine used by the Trainer callback and the CLI."""

from __future__ import annotations

import logging
import time
from collections import defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from memaudit.compliance import normalize_release_context
from memaudit.constants import (
    DEFAULT_MIN_K_PCT,
    DEFAULT_PREFIX_FRACTIONS,
    DEFAULT_REAL_SAMPLE,
    HEADLINE_ATTACK,
    HEADLINE_ATTACK_FALLBACK,
    MIN_CONTROLS_FOR_TPR_AT_1PCT,
)
from memaudit.exceptions import MemauditAuditError, MemauditConfigError
from memaudit.injection import canary_record
from memaudit.preflight import adapter_toggle_safe, inspect_embeddings
from memaudit.report import build_report, sidecar_path, write_report
from memaudit.scoring import (
    combine_ft_ref,
    generate_canary_completions,
    locate_secret_span,
    score_sequence,
)
from memaudit.stats import roc_auc, roc_points, tpr_at_fpr, welch_ttest
from memaudit.utils import (
    dataset_fingerprint,
    dataset_length,
    environment_versions,
    example_text,
    infer_device,
    is_peft_model,
    iter_examples,
    model_fingerprint,
    model_identity,
    sha256_json,
    short_hash,
    unwrap_model,
    write_json,
)

log = logging.getLogger("memaudit")


def validate_manifest_for_audit(manifest: dict[str, Any]) -> None:
    """Refuse artifacts that would produce a silently empty or inverted audit."""
    if not isinstance(manifest, dict) or "canaries" not in manifest:
        raise MemauditConfigError(
            "canary-set must be the JSON dict returned by inject() "
            "(a 'canaries' list with inclusion coins). "
            "Use memaudit-manifest.json, not a raw generate_canaries() dump."
        )
    canaries = manifest.get("canaries") or []
    if not canaries:
        raise MemauditAuditError("manifest 'canaries' list is empty")
    if not any("included" in c for c in canaries):
        raise MemauditAuditError(
            "canary-set has no 'included' flags. Pass the inject() manifest "
            "(memaudit-manifest.json). A raw generate_canaries() dump would "
            "treat every row as a never-inserted control and invent an empty audit."
        )
    if not any(c.get("included") for c in canaries):
        raise MemauditAuditError(
            "No canaries are marked included=true. The membership headline "
            "would be empty. Re-run inject() (check include_prob / coins) and "
            "pass memaudit-manifest.json."
        )


def _eval_guard(model: Any):
    class _Guard:
        def __enter__(self):
            self.was = bool(getattr(model, "training", False))
            if hasattr(model, "eval"):
                model.eval()
            return model

        def __exit__(self, *exc):
            if self.was and hasattr(model, "train"):
                model.train()
            return False

    return _Guard()


def _score_canary_record(
    model: Any,
    tokenizer: Any,
    canary: Mapping[str, Any],
    fmt: str,
    min_k_pct: float,
    skip_first: bool,
) -> dict[str, float]:
    from memaudit.types import Canary

    rec = canary_record(Canary.from_dict(dict(canary)), fmt)
    text = example_text(rec, fmt)
    ids, span = locate_secret_span(
        tokenizer, text, canary.get("secret") or "", canary.get("secret_token_ids")
    )
    if span is None:
        span = (0, len(ids))
    return score_sequence(model, ids, span=span, min_k_pct=min_k_pct, skip_first_record_token=skip_first)


def _toggle_or_score_ref(
    model: Any,
    score_fn,
    *,
    ref_model: Any | None,
    toggle_safe: bool,
) -> tuple[dict[str, float], dict[str, float] | None, str]:
    ft = score_fn(model)
    if toggle_safe and hasattr(model, "disable_adapter"):
        try:
            with model.disable_adapter():
                ref = score_fn(model)
            return ft, ref, "disable_adapter"
        except Exception:
            pass
    if ref_model is not None:
        return ft, score_fn(ref_model), "separate_reference"
    return ft, None, "target_only"


def _load_ref_auto(model: Any, ref: Any) -> tuple[Any | None, dict[str, Any]]:
    info: dict[str, Any] = {"mode": None, "identity": None}
    if ref is None or ref == "none":
        info["mode"] = "none"
        return None, info
    if ref != "auto" and not isinstance(ref, str):
        info["mode"] = "provided_object"
        info["identity"] = model_identity(ref)
        return ref, info
    if ref == "auto":
        info["mode"] = "auto"
        return None, info
    # path
    try:
        from transformers import AutoModelForCausalLM

        loaded = AutoModelForCausalLM.from_pretrained(ref)
        try:
            loaded.to(infer_device(model))
        except Exception:
            pass
        info["mode"] = "path"
        info["identity"] = {"name_or_path": ref, "class": type(loaded).__name__}
        return loaded, info
    except Exception as exc:
        raise MemauditConfigError(
            f"could not load --ref {ref}: {exc}. "
            "Pass a valid base-checkpoint path, or --ref none for target-only scoring."
        ) from exc


def _sample_real_texts(
    dataset: Any,
    manifest: dict[str, Any],
    n: int,
    seed: int,
    held_out: Any | None,
) -> tuple[list[str], list[str], float]:
    fmt = manifest.get("fmt") or "text"
    secrets = {c.get("secret") for c in manifest.get("canaries") or [] if c.get("secret")}
    texts: list[str] = []
    for row in iter_examples(dataset):
        blob = example_text(row, fmt if any(k in row for k in ("text", "prompt", "messages")) else "text")
        if not blob or any(s and s in blob for s in secrets):
            continue
        texts.append(blob)
    rng = np.random.default_rng(seed)
    exact_dup = 0.0
    if texts:
        uniq = len(set(texts))
        exact_dup = 1.0 - uniq / len(texts)
        if len(texts) > n:
            idx = rng.choice(len(texts), size=n, replace=False)
            texts = [texts[int(i)] for i in idx]
    held: list[str] = []
    if held_out is not None:
        for row in iter_examples(held_out):
            blob = example_text(row, fmt)
            if blob:
                held.append(blob)
        if len(held) > n:
            idx = rng.choice(len(held), size=n, replace=False)
            held = [held[int(i)] for i in idx]
    elif len(texts) >= 4:
        # split the sample so we still have a control side
        cut = max(1, len(texts) // 2)
        held = texts[cut:]
        texts = texts[:cut]
    return texts, held, float(exact_dup)


def run_audit(
    model: Any,
    tokenizer: Any,
    manifest: dict[str, Any],
    dataset: Any | None = None,
    ref: Any = "auto",
    real_sample: int = DEFAULT_REAL_SAMPLE,
    output_path: str | Path | None = None,
    held_out: Any | None = None,
    preflight_findings: dict[str, Any] | None = None,
    min_k_pct: float = DEFAULT_MIN_K_PCT,
    prefix_fractions: Sequence[float] = DEFAULT_PREFIX_FRACTIONS,
    skip_generation: bool = False,
    reveal: bool = False,
    trainer: Any | None = None,
    skip_first_packed_token: bool = True,
    seeds: Sequence[int] | None = None,
    release_context: str | None = None,
    dataset_path: str | Path | None = None,
    model_path: str | Path | None = None,
) -> dict[str, Any]:
    """Run both verdicts: membership (likelihood MIA) and regurgitation (generation).

    Never ships a loss-only report. Scores the secret span only.

    ``seeds`` enables multi-seed mode: canary scoring and greedy generation are
    deterministic given the trained model, so what is re-run per seed is the
    randomness that actually exists in the audit procedure - bootstrap
    resampling of held-out control scores for threshold calibration, and
    real-record sampling. The resulting ``stability`` block measures
    audit-procedure variance, NOT training variance across re-trained models.

    ``release_context`` is the user's EDPB para 46 declaration
    (``public-api`` | ``internal`` | ``open-weights``; default ``unspecified``).
    """
    if isinstance(manifest, (str, Path)):
        from memaudit.utils import load_json

        manifest = load_json(manifest)
    validate_manifest_for_audit(manifest)
    if tokenizer is None:
        raise MemauditConfigError(
            "tokenizer is required to score secret spans. Pass processing_class= "
            "to the Trainer / --tokenizer to the CLI."
        )
    release_context_norm = normalize_release_context(release_context)
    seed_list: list[int] | None = None
    if seeds is not None:
        seed_list = [int(s) for s in seeds]
        if not seed_list:
            raise MemauditConfigError("seeds must be a non-empty list of integers (e.g. --seeds 0,1,2)")

    started = time.perf_counter()
    model = unwrap_model(model, trainer)
    fmt = manifest.get("fmt") or "text"
    canaries = list(manifest.get("canaries") or [])
    members = [c for c in canaries if c.get("included")]
    controls = [c for c in canaries if not c.get("included")]
    audit_warnings: list[str] = []
    if not controls:
        audit_warnings.append(
            "Manifest has no held-out controls. TPR@1% FPR is unidentified; "
            "the headline will be refused rather than fabricated."
        )

    emb = inspect_embeddings(model)
    toggle_ok, toggle_reason = adapter_toggle_safe(emb)
    ref_model, ref_meta = _load_ref_auto(model, ref)
    if ref == "auto":
        if toggle_ok and is_peft_model(model):
            ref_meta["mode"] = "disable_adapter"
            ref_meta["identity"] = {"via": "peft.disable_adapter()"}
        else:
            why = toggle_reason or "model is not an unmerged PEFT adapter"
            raise MemauditConfigError(
                f"--ref auto is not available ({why}). "
                "Pass --ref <path-to-base-checkpoint> for base-calibrated Min-K%++, "
                "or --ref none to accept target-only scoring "
                "(headline downgraded to min_k_plus_plus and labeled as such)."
            )
    elif ref == "none" or ref is None:
        audit_warnings.append(
            "Reference is none: headline is target-only Min-K%++ "
            "(not base-calibrated). This is a weaker attack than the pre-declared headline."
        )

    per_canary: list[dict[str, Any]] = []
    member_scores: list[float] = []
    control_scores: list[float] = []
    used_headline = HEADLINE_ATTACK

    def score_one(m, canary):
        return _score_canary_record(m, tokenizer, canary, fmt, min_k_pct, skip_first_packed_token)

    with _eval_guard(model):
        if ref_model is not None:
            ref_model.eval()
        n_total = len(canaries)
        for i, canary in enumerate(canaries):
            if n_total >= 8 and (i == 0 or (i + 1) % 10 == 0 or i + 1 == n_total):
                log.info("scoring canary %s/%s id=%s", i + 1, n_total, canary.get("id"))
            ft, ref_s, how = _toggle_or_score_ref(
                model,
                lambda m, c=canary: score_one(m, c),
                ref_model=ref_model,
                toggle_safe=toggle_ok and ref == "auto" and is_peft_model(model),
            )
            combined = combine_ft_ref(ft, ref_s)
            if combined.get("headline_attack_used") != HEADLINE_ATTACK:
                used_headline = HEADLINE_ATTACK_FALLBACK
            gen_block: dict[str, Any] = {"skipped": True}
            if not skip_generation:
                gen_block = generate_canary_completions(
                    model,
                    tokenizer,
                    canary.get("secret") or "",
                    canary.get("secret_token_ids") or [],
                    prefix_fractions=prefix_fractions,
                )
            row = {
                "id": canary.get("id"),
                "role": canary.get("role"),
                "included": bool(canary.get("included")),
                "repetitions": canary.get("repetitions"),
                "family": canary.get("family"),
                "scores": {k: v for k, v in combined.items() if k != "headline_attack_used"},
                "ref_source": how,
                "regurgitation": {
                    "regurgitated": gen_block.get("regurgitated", False),
                    "by_prefix": [
                        {kk: vv for kk, vv in p.items() if kk != "prefix"}
                        for p in (gen_block.get("by_prefix") or [])
                    ],
                },
            }
            per_canary.append(row)
            score = combined.get("headline_score", float("nan"))
            if canary.get("included"):
                member_scores.append(float(score))
            else:
                control_scores.append(float(score))

    # drop NaNs for stats
    member_scores = [s for s in member_scores if s == s]
    control_scores = [s for s in control_scores if s == s]
    det = tpr_at_fpr(member_scores, control_scores, fpr=0.01)
    auc = roc_auc(member_scores, control_scores)
    roc = roc_points(member_scores, control_scores)

    # regurgitation rates
    by_tier: dict[str, dict[str, Any]] = {}
    tier_hits: dict[int, list[bool]] = defaultdict(list)
    overall_flags: list[bool] = []
    for row, canary in zip(per_canary, canaries):
        if not canary.get("included"):
            continue
        flag = bool((row.get("regurgitation") or {}).get("regurgitated"))
        overall_flags.append(flag)
        tier_hits[int(canary.get("repetitions") or 1)].append(flag)
    for tier, flags in sorted(tier_hits.items()):
        by_tier[str(tier)] = {"n": len(flags), "n_regurgitated": int(sum(flags)), "rate": float(np.mean(flags))}
    overall_rate = float(np.mean(overall_flags)) if overall_flags else float("nan")

    control_regurg = [
        bool((row.get("regurgitation") or {}).get("regurgitated"))
        for row, c in zip(per_canary, canaries)
        if not c.get("included")
    ]

    def _headline_score_for_text(text: str) -> float:
        ids, span = locate_secret_span(tokenizer, text, text, None)
        ft = score_sequence(model, ids, span=span or (0, len(ids)), min_k_pct=min_k_pct)
        ref_s = None
        if toggle_ok and ref == "auto" and is_peft_model(model) and hasattr(model, "disable_adapter"):
            try:
                with model.disable_adapter():
                    ref_s = score_sequence(model, ids, span=span or (0, len(ids)), min_k_pct=min_k_pct)
            except Exception:
                ref_s = None
        elif ref_model is not None:
            ref_s = score_sequence(ref_model, ids, span=span or (0, len(ids)), min_k_pct=min_k_pct)
        comb = combine_ft_ref(ft, ref_s)
        return float(comb.get("headline_score", float("nan")))

    def _real_block_for_seed(sample_seed: int, collect_ranked: bool) -> dict[str, Any]:
        real_texts, held_texts, dup_rate = _sample_real_texts(
            dataset, manifest, int(real_sample), int(sample_seed), held_out
        )
        real_scores: list[float] = []
        held_scores: list[float] = []
        ranked: list[dict[str, Any]] = []
        with _eval_guard(model):
            for text in real_texts:
                score = _headline_score_for_text(text)
                real_scores.append(score)
                if collect_ranked:
                    item: dict[str, Any] = {"hash": short_hash(text), "score": score if score == score else None}
                    if reveal:
                        item["text"] = text
                    ranked.append(item)
            for text in held_texts:
                held_scores.append(_headline_score_for_text(text))
        real_scores = [s for s in real_scores if s == s]
        held_scores = [s for s in held_scores if s == s]
        ranked.sort(key=lambda x: (x.get("score") is None, -(x.get("score") or 0)))
        block: dict[str, Any] = {
            "sample_seed": int(sample_seed),
            "n_train_sampled": len(real_texts),
            "n_held_out": len(held_texts),
            "exact_dup_rate": dup_rate,
            "set_level": welch_ttest(real_scores, held_scores) if held_scores else {
                "note": "no held-out real records; set-level test skipped",
                "p_value": None,
            },
        }
        if collect_ranked:
            block["ranked"] = ranked[: min(50, len(ranked))]
            block["redacted"] = not reveal
        return block

    primary_real_seed = seed_list[0] if seed_list else int(manifest.get("seed") or 0)
    real_block: dict[str, Any] | None = None
    real_per_seed: list[dict[str, Any]] | None = None
    if dataset is not None and real_sample and real_sample > 0:
        real_block = _real_block_for_seed(primary_real_seed, collect_ranked=True)
        if seed_list and len(seed_list) > 1:
            real_per_seed = [
                {
                    "seed": real_block["sample_seed"],
                    "p_value": (real_block.get("set_level") or {}).get("p_value"),
                    "mean_gap": (real_block.get("set_level") or {}).get("mean_gap"),
                    "n_train_sampled": real_block.get("n_train_sampled"),
                }
            ]
            for extra_seed in seed_list[1:]:
                blk = _real_block_for_seed(extra_seed, collect_ranked=False)
                real_per_seed.append(
                    {
                        "seed": extra_seed,
                        "p_value": (blk.get("set_level") or {}).get("p_value"),
                        "mean_gap": (blk.get("set_level") or {}).get("mean_gap"),
                        "n_train_sampled": blk.get("n_train_sampled"),
                    }
                )

    stability: dict[str, Any] | None = None
    if seed_list:
        per_seed: list[dict[str, Any]] = []
        for s in seed_list:
            rng = np.random.default_rng(int(s))
            if control_scores:
                boot = [float(x) for x in rng.choice(np.asarray(control_scores), size=len(control_scores), replace=True)]
            else:
                boot = []
            d = tpr_at_fpr(member_scores, boot, fpr=0.01)
            per_seed.append(
                {
                    "seed": int(s),
                    "tpr": d["tpr"],
                    "threshold": d["threshold"],
                    "n_detected": d["n_detected"],
                    "auc": roc_auc(member_scores, boot),
                    "headline_valid": d["headline_valid"],
                }
            )
        tprs = [p["tpr"] for p in per_seed if p["tpr"] == p["tpr"]]
        variance = {
            "tpr_mean": float(np.mean(tprs)) if tprs else float("nan"),
            "tpr_min": float(np.min(tprs)) if tprs else float("nan"),
            "tpr_max": float(np.max(tprs)) if tprs else float("nan"),
            "tpr_std": float(np.std(tprs)) if tprs else float("nan"),
            "per_seed": per_seed,
        }
        stability = {
            "kind": "audit_procedure_variance",
            "label": (
                "Multi-seed stability measures AUDIT-PROCEDURE variance, not "
                "training variance: the model was trained once; canary member/"
                "control splits are fixed by the manifest; re-training across "
                "seeds is out of scope. What varies per seed: bootstrap "
                "resampling of held-out control scores (threshold calibration) "
                "and real-record sampling."
            ),
            "audit_seeds": seed_list,
            "deterministic_components": (
                "canary likelihood scoring (teacher-forced forward passes) and "
                "regurgitation generation (greedy decoding) are deterministic "
                "given the trained model and were computed once"
            ),
            "variance": variance,
            "real_records_per_seed": real_per_seed,
        }

    headline_valid = bool(det.get("headline_valid"))
    membership = {
        "headline_attack": used_headline,
        "predeclared_headline": HEADLINE_ATTACK,
        # Refuse a fake precise TPR@1%FPR when the control set cannot identify it.
        "tpr_at_1pct_fpr": det["tpr"] if headline_valid else None,
        "ci_low": det["ci_low"] if headline_valid else None,
        "ci_high": det["ci_high"] if headline_valid else None,
        "headline_valid": headline_valid,
        "exploratory_tpr": det["tpr"],
        "exploratory_ci_low": det["ci_low"],
        "exploratory_ci_high": det["ci_high"],
        "achievable_fpr": det.get("achievable_fpr"),
        "min_controls_required": det.get("min_controls_required", MIN_CONTROLS_FOR_TPR_AT_1PCT),
        "threshold": det["threshold"],
        "auc": auc,
        "n_members": det["n_members"],
        "n_controls": det["n_controls"],
        "n_detected": det["n_detected"],
        "warning": det.get("warning"),
        "roc": roc,
        "score_orientation": "higher = more member-like",
    }
    regurgitation = {
        "overall": {
            "n": len(overall_flags),
            "n_regurgitated": int(sum(overall_flags)) if overall_flags else 0,
            "rate": overall_rate,
        },
        "by_tier": by_tier,
        "prefix_fractions": list(prefix_fractions),
        "thresholds": {"exact": True, "bleu": 0.75, "ned": 0.10},
        "note": (
            "1x canaries are MIA-tier only; verbatim regurgitation is not expected "
            "from a single effective occurrence."
        ),
    }
    negative_controls = {
        "n": len(controls),
        "mean_headline_score": float(np.mean(control_scores)) if control_scores else float("nan"),
        "regurgitation_rate": float(np.mean(control_regurg)) if control_regurg else float("nan"),
        "note": (
            "Held-out / never-inserted canaries. Published calibration anchors: "
            "cross-model false-extraction floor ~6% (Carlini 2022); pre-FT exact-match "
            "baseline <0.06% (Bossy 2025). Those numbers are literature, not this run."
        ),
    }

    adapter_info = None
    if is_peft_model(model) or emb.get("r") is not None:
        adapter_info = {
            "r": emb.get("r"),
            "lora_alpha": emb.get("lora_alpha"),
            "bias": emb.get("bias"),
            "modules_to_save": emb.get("modules_to_save"),
            "peft_type": emb.get("peft_type"),
            "merged": emb.get("merged"),
        }

    manifest_sha = manifest.get("manifest_hash") or sha256_json(manifest)
    repetition_grid = sorted({int(c.get("repetitions") or 1) for c in members}) if members else []
    families_used = sorted({str(c.get("family")) for c in canaries if c.get("family")})
    audit_scope = {
        "n_canaries_inserted": len(members),
        "n_heldout_controls": len(controls),
        "repetition_grid": repetition_grid,
        "families_used": families_used,
        "include_prob": manifest.get("include_prob"),
        "inject_seed": manifest.get("seed"),
        "audit_seeds": seed_list,
        "dataset_rows_total": dataset_length(dataset) if dataset is not None else None,
        "real_records_sampled": (real_block or {}).get("n_train_sampled"),
    }
    resolved_config = {
        "ref": ref if isinstance(ref, str) or ref is None else "provided_object",
        "ref_mode": ref_meta.get("mode"),
        "real_sample": int(real_sample),
        "min_k_pct": float(min_k_pct),
        "prefix_fractions": list(prefix_fractions),
        "skip_generation": bool(skip_generation),
        "reveal": bool(reveal),
        "skip_first_packed_token": bool(skip_first_packed_token),
        "fpr_target": 0.01,
        "min_controls_for_headline": MIN_CONTROLS_FOR_TPR_AT_1PCT,
        "headline_attack_predeclared": HEADLINE_ATTACK,
        "release_context": release_context_norm,
        "seeds": {"inject": manifest.get("seed"), "audit_seeds": seed_list},
        "include_prob": manifest.get("include_prob"),
        "fmt": fmt,
    }
    provenance = {
        "manifest_hash": manifest.get("manifest_hash"),
        "canary_manifest_sha256": manifest_sha,
        "dataset_fingerprint": dataset_fingerprint(dataset, path=dataset_path),
        "model_fingerprint": model_fingerprint(model, model_path=model_path),
        "resolved_config": resolved_config,
        "environment": environment_versions(),
        "n_canaries": len(canaries),
        "fmt": fmt,
        "local_only": True,
        "signing": {
            "report_sha256": (
                "stamped at write time over the canonicalized report content "
                "(minus the report_sha256 field); sidecar <report>.sha256; "
                "check with `memaudit verify <report.json>`"
            ),
            "note": (
                "Cryptographic signing (GPG / sigstore) of the report file is a "
                "release-runbook step outside memaudit; memaudit does not "
                "implement key management."
            ),
        },
    }
    report = build_report(
        seeds={
            "inject": manifest.get("seed"),
            "canary_manifest": manifest.get("manifest_hash"),
            "audit_seeds": seed_list,
        },
        canary_manifest_hash=manifest_sha,
        model_info=model_identity(model),
        adapter_info=adapter_info,
        ref_info=ref_meta,
        membership=membership,
        regurgitation=regurgitation,
        negative_controls=negative_controls,
        real_records=real_block,
        preflight=preflight_findings,
        provenance=provenance,
        per_canary=per_canary,
        extra={
            "audit_seconds": round(time.perf_counter() - started, 3),
            "audit_warnings": audit_warnings,
        },
        release_context=release_context_norm,
        audit_scope=audit_scope,
        stability=stability,
    )
    if output_path:
        write_report(report, output_path)
        log.info(
            "report_sha256 %s (sidecar %s)",
            report.get("report_sha256"),
            sidecar_path(output_path).name,
        )
    log.info("audit finished in %.2fs (wrote %s)", report.get("audit_seconds"), output_path)
    return report


def write_deferred_audit(
    output_dir: str | Path,
    manifest: dict[str, Any],
    reason: str,
    model_path_hint: str | None = None,
    dataset_hint: str = "<path-to-raw-train-or-jsonl>",
) -> dict[str, Any]:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    man_path = write_json(out / "memaudit-manifest.json", manifest)
    canaries_path = write_json(
        out / "memaudit-canaries.json",
        {"canaries": manifest.get("canaries"), "fmt": manifest.get("fmt"), "seed": manifest.get("seed")},
    )
    model_arg = model_path_hint or str(out)
    cmd = (
        f"memaudit audit --model {model_arg} --canary-set {man_path} "
        f"--dataset {dataset_hint} --ref auto"
    )
    payload = {
        "deferred": True,
        "reason": reason,
        "command": cmd,
        "manifest_path": str(man_path),
        "canary_set_path": str(canaries_path),
        "note": (
            "In-callback scoring under ZeRO-3 / FSDP deadlocks or OOMs (collectives / "
            "gather-to-rank). Re-run the CLI against the saved checkpoint on a single GPU."
        ),
    }
    write_json(out / "memaudit-deferred.json", payload)
    return payload
