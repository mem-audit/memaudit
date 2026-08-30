"""Shared audit engine used by the Trainer callback and the CLI."""

from __future__ import annotations

import logging
import time
from collections import defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from memaudit.canaries import actual_generator_of, requested_family_of
from memaudit.compliance import normalize_release_context
from memaudit.constants import (
    DEFAULT_MIN_K_PCT,
    DEFAULT_PREFIX_FRACTIONS,
    DEFAULT_REAL_SAMPLE,
    DEFAULT_TARGET_FPR,
    HEADLINE_ATTACK,
    HEADLINE_ATTACK_FALLBACK,
    MIN_CONTROLS_FOR_TPR_AT_1PCT,
    resolve_audit_profile,
)
from memaudit.exceptions import MemauditAuditError, MemauditConfigError
from memaudit.injection import canary_record
from memaudit.peft_semantics import (
    base_equivalence_guard,
    default_probe_texts,
    quantization_ref_mismatch,
    unusual_peft_triggers,
)
from memaudit.preflight import absent_preflight, adapter_toggle_safe, inspect_embeddings
from memaudit.report import build_report, sidecar_path, write_report
from memaudit.scorers import resolve_scorer, scorer_provenance
from memaudit.scorers.min_k import DEFAULT_SCORER_NAME
from memaudit.scorers.signals import SignalsCache, TokenSignals
from memaudit.scoring import (
    combine_ft_ref,
    extract_token_signals,
    generate_canary_completions,
    locate_secret_span,
    scores_from_signals,
)
from memaudit.stats import (
    bootstrap_calibration_stability,
    membership_by_repetition,
    roc_auc,
    roc_points,
    tpr_at_fpr,
    welch_ttest,
)
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

_REGURGITATION_NOT_RUN_NOTE = (
    "Regurgitation generation was not executed in this run. The protocol "
    "fields below describe the test that WOULD have been run; they are not results."
)
_REGURGITATION_SKIP_WARNING = (
    "Regurgitation generation was not executed (skip_generation=True). "
    "This report contains no regurgitation measurement; it is not a "
    "zero-detection result."
)
_EXACT_DUP_UNMEASURED_NOTE = (
    "Exact-duplicate rate was not measured (no extractable training texts). "
    "This is not a zero-duplication result."
)
_REAL_RECORDS_NOT_RUN_NOTE = (
    "Real-record ranking was not executed in this run. This is not a "
    "zero-leakage result and not a pass."
)
_TOGGLE_FAIL_WARNING = (
    "disable_adapter() failed during scoring. reference.mode records the "
    "fallback actually used; per-canary ref_source is authoritative."
)


def regurgitation_execution_from_skip(skip_generation: bool) -> dict[str, Any]:
    """Canonical run-level regurgitation execution state (P0)."""
    if skip_generation:
        return {
            "status": "not_run",
            "reason": "skip_generation",
            "note": _REGURGITATION_NOT_RUN_NOTE,
        }
    return {"status": "executed"}


def regurgitation_was_evaluated(row: Mapping[str, Any]) -> bool:
    """True only when this canary's regurgitation protocol actually ran."""
    block = row.get("regurgitation") if isinstance(row, Mapping) else None
    if not isinstance(block, Mapping):
        return False
    return (block.get("execution") or {}).get("status") == "executed"


def real_records_not_run(reason: str) -> dict[str, Any]:
    """Structured block when real-record ranking did not run (F4)."""
    return {
        "execution": {
            "status": "not_run",
            "reason": reason,
            "note": _REAL_RECORDS_NOT_RUN_NOTE,
        },
        "n_train_sampled": 0,
        "n_comparison_split": 0,
        "comparison_population": "none",
        "exact_dup_rate": None,
        "set_level": {
            "kind": "not_run",
            "inferential": False,
            "p_value": None,
            "note": _REAL_RECORDS_NOT_RUN_NOTE,
        },
    }


def reconcile_reference_mode(
    ref_meta: dict[str, Any],
    per_canary: list[dict[str, Any]],
    toggle_failures: list[str],
    *,
    has_ref_model: bool,
) -> tuple[dict[str, Any], str | None]:
    """If disable_adapter failed, stamp the mode that was actually used (F7).

    Returns ``(ref_meta, headline_fallback_or_None)``. Headline fallback is
    ``HEADLINE_ATTACK_FALLBACK`` when scoring became target-only.
    """
    if ref_meta.get("mode") != "disable_adapter":
        return ref_meta, None
    sources = {row.get("ref_source") for row in per_canary if row.get("ref_source")}
    if "disable_adapter" in sources:
        return ref_meta, None
    if not sources and not toggle_failures:
        return ref_meta, None
    if "separate_reference" in sources or (has_ref_model and sources != {"target_only"}):
        actual = "separate_reference"
        headline = None
    else:
        actual = "target_only"
        headline = HEADLINE_ATTACK_FALLBACK
    ref_meta = dict(ref_meta)
    ref_meta["downgraded_from"] = "disable_adapter"
    ref_meta["mode"] = actual
    if toggle_failures:
        ref_meta["toggle_error"] = toggle_failures[0]
    return ref_meta, headline


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


def _canary_ids_and_span(
    tokenizer: Any,
    canary: Mapping[str, Any],
    fmt: str,
) -> tuple[list[int], tuple[int, int]]:
    from memaudit.types import Canary

    rec = canary_record(Canary.from_dict(dict(canary)), fmt)
    text = example_text(rec, fmt)
    ids, span = locate_secret_span(
        tokenizer, text, canary.get("secret") or "", canary.get("secret_token_ids")
    )
    if span is None:
        span = (0, len(ids))
    return ids, span


def _extract_canary_signals(
    model: Any,
    tokenizer: Any,
    canary: Mapping[str, Any],
    fmt: str,
    skip_first: bool,
    cache: SignalsCache | None = None,
    cache_tag: str = "",
) -> TokenSignals:
    ids, span = _canary_ids_and_span(tokenizer, canary, fmt)
    return extract_token_signals(
        model,
        ids,
        span=span,
        skip_first_record_token=skip_first,
        cache=cache,
        cache_tag=cache_tag,
    )


def _toggle_or_extract_ref(
    model: Any,
    extract_fn,
    *,
    ref_model: Any | None,
    toggle_safe: bool,
    toggle_failures: list[str] | None = None,
) -> tuple[TokenSignals, TokenSignals | None, str]:
    """``extract_fn(model, cache_tag)`` — tag distinguishes disable_adapter state."""
    ft = extract_fn(model, "target")
    if toggle_safe and hasattr(model, "disable_adapter"):
        try:
            with model.disable_adapter():
                ref = extract_fn(model, "disable_adapter")
            return ft, ref, "disable_adapter"
        except Exception as exc:
            if toggle_failures is not None:
                toggle_failures.append(f"{type(exc).__name__}: {exc}")
    if ref_model is not None:
        return ft, extract_fn(ref_model, "separate_reference"), "separate_reference"
    return ft, None, "target_only"


def _combine_with_scorer(
    scorer: Any,
    target: TokenSignals,
    reference: TokenSignals | None,
    min_k_pct: float,
) -> dict[str, float]:
    """Diagnostic dict from signals + headline from the plugged-in scorer."""
    ft = scores_from_signals(target, min_k_pct)
    ref = scores_from_signals(reference, min_k_pct) if reference is not None else None
    combined = combine_ft_ref(ft, ref)
    combined["headline_score"] = float(scorer.score(target, reference))
    name = getattr(scorer, "name", None)
    if name and name not in {DEFAULT_SCORER_NAME, HEADLINE_ATTACK}:
        combined["headline_attack_used"] = str(name)
    return combined


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
) -> tuple[list[str], list[str], float | None, str]:
    fmt = manifest.get("fmt") or "text"
    secrets = {c.get("secret") for c in manifest.get("canaries") or [] if c.get("secret")}
    texts: list[str] = []
    for row in iter_examples(dataset):
        blob = example_text(row, fmt if any(k in row for k in ("text", "prompt", "messages")) else "text")
        if not blob or any(s and s in blob for s in secrets):
            continue
        texts.append(blob)
    rng = np.random.default_rng(seed)
    exact_dup: float | None = None
    if texts:
        uniq = len(set(texts))
        exact_dup = 1.0 - uniq / len(texts)
        if len(texts) > n:
            idx = rng.choice(len(texts), size=n, replace=False)
            texts = [texts[int(i)] for i in idx]
    held: list[str] = []
    comparison_population = "none"
    if held_out is not None:
        for row in iter_examples(held_out):
            blob = example_text(row, fmt)
            if blob:
                held.append(blob)
        if len(held) > n:
            idx = rng.choice(len(held), size=n, replace=False)
            held = [held[int(i)] for i in idx]
        comparison_population = "held_out"
    elif len(texts) >= 4:
        # split sampled training records — this is not a held-out population
        cut = max(1, len(texts) // 2)
        held = texts[cut:]
        texts = texts[:cut]
        comparison_population = "training_split"
    return texts, held, exact_dup, comparison_population


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
    profile: str | None = None,
    target_fpr: float | None = None,
    scorer: str | Any | None = None,
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

    ``profile`` is a named audit profile (``smoke`` / ``routine`` / ``powered``).
    When omitted, the name is inferred from manifest counts or stamped
    ``manifest['audit_profile']``. ``target_fpr`` overrides the profile FPR
    (default 0.01). The ``smoke`` profile refuses a TPR@FPR headline.

    ``scorer`` selects the membership backend (name, ``module:Class`` path, or
    instance). Default is Min-K%++. Calibration / TPR / CI stay here.
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
    requested_members = manifest.get("n_candidates")
    if requested_members is None:
        requested_members = sum(1 for c in canaries if c.get("role") != "control")
    repetition_grid = sorted({int(c.get("repetitions") or 1) for c in members}) if members else []
    profile_name = profile or manifest.get("audit_profile")
    if not profile_name:
        for c in canaries:
            stamped = (c.get("metadata") or {}).get("audit_profile")
            if stamped:
                profile_name = stamped
                break
    try:
        profile_spec = resolve_audit_profile(
            profile_name,
            requested_members=int(requested_members) if requested_members is not None else None,
            n_controls=len(controls),
            repetitions=repetition_grid,
            target_fpr=target_fpr,
        )
    except KeyError as exc:
        raise MemauditConfigError(
            f"Unknown audit profile {profile_name!r}. Use smoke, routine, or powered."
        ) from exc
    fpr = float(profile_spec.get("target_fpr") or DEFAULT_TARGET_FPR)
    regurgitation_execution = regurgitation_execution_from_skip(bool(skip_generation))
    try:
        scorer_obj = resolve_scorer(scorer, min_k_pct=min_k_pct)
    except MemauditConfigError:
        raise
    except Exception as exc:
        raise MemauditConfigError(
            f"could not load membership scorer {scorer!r}: {exc}"
        ) from exc
    scorer_block = scorer_provenance(scorer_obj)
    custom_scorer = scorer_block["name"] not in {DEFAULT_SCORER_NAME, HEADLINE_ATTACK}
    audit_warnings: list[str] = []
    if skip_generation:
        audit_warnings.append(_REGURGITATION_SKIP_WARNING)
    used_headline = HEADLINE_ATTACK if not custom_scorer else str(scorer_block["name"])
    signals_cache = SignalsCache()
    if not controls:
        audit_warnings.append(
            "Manifest has no held-out controls. TPR@1% FPR is unidentified; "
            "the headline will be refused rather than fabricated."
        )

    emb = inspect_embeddings(model)
    toggle_ok, toggle_reason = adapter_toggle_safe(emb)
    ref_model, ref_meta = _load_ref_auto(model, ref)
    use_toggle = False
    if emb.get("quantized") and ref_model is not None:
        q_mismatch, q_msg = quantization_ref_mismatch(emb, ref_model)
        if q_mismatch:
            audit_warnings.append(q_msg or "quantized target vs full-precision --ref")
            ref_meta["quantization_mismatch"] = True
            ref_meta["downgraded_from"] = ref_meta.get("mode")
            if toggle_ok and is_peft_model(model) and hasattr(model, "disable_adapter"):
                ref_model = None
                ref_meta["mode"] = "disable_adapter"
                ref_meta["identity"] = {"via": "peft.disable_adapter()", "reason": "quantization_mismatch"}
                use_toggle = True
            else:
                ref_model = None
                ref_meta["mode"] = "target_only"
                used_headline = HEADLINE_ATTACK_FALLBACK
                audit_warnings.append(
                    "Quantization mismatch: discarded the full-precision --ref and "
                    "fell back to target-only scoring (headline downgraded)."
                )
                toggle_ok = False
    if ref == "auto":
        if toggle_ok and is_peft_model(model):
            ref_meta["mode"] = "disable_adapter"
            ref_meta["identity"] = {"via": "peft.disable_adapter()"}
            use_toggle = True
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
    elif use_toggle:
        pass

    preflight_block = (
        dict(preflight_findings) if preflight_findings is not None else absent_preflight()
    )
    if preflight_findings is not None:
        preflight_block.setdefault("ran", True)
        preflight_block.setdefault("path", "callback")
    # Strip in-memory logit tensors from the report; keep metadata.
    capture = preflight_block.pop("base_equivalence_capture", None)
    if use_toggle and is_peft_model(model) and hasattr(model, "disable_adapter"):
        guard = base_equivalence_guard(
            model,
            tokenizer,
            captured=capture,
            manifest=manifest,
            probe_texts=default_probe_texts(manifest),
        )
        triggers = unusual_peft_triggers(emb)
        guard["unusual_triggers"] = triggers
        preflight_block["base_equivalence"] = dict(guard)
        if guard.get("verdict") == "fail" and triggers:
            raise MemauditConfigError(
                "--ref auto is not available (base-equivalence guard failed under "
                f"unusual PEFT config {triggers}: verdict={guard.get('verdict')}, "
                f"max_abs_logit_diff={guard.get('max_abs_logit_diff')}, "
                f"restored={guard.get('restored')}). "
                "Pass --ref <path-to-base-checkpoint> for base-calibrated Min-K%++, "
                "or --ref none to accept target-only scoring "
                "(headline downgraded to min_k_plus_plus and labeled as such)."
            )
        if guard.get("verdict") == "fail":
            use_toggle = False
            toggle_ok = False
            if ref_model is not None:
                ref_meta["mode"] = "separate_reference"
                ref_meta["downgraded_from"] = "disable_adapter"
            else:
                used_headline = HEADLINE_ATTACK_FALLBACK
                ref_meta["mode"] = "target_only"
                ref_meta["downgraded_from"] = "disable_adapter"
            audit_warnings.append(
                "base-equivalence guard failed; disable_adapter() scoring is "
                "downgraded (headline may fall back to target-only Min-K%++)."
            )
        elif guard.get("verdict") == "warn":
            audit_warnings.append(
                "base-equivalence guard: disabled logits differ from the "
                f"preflight capture by {guard.get('max_abs_logit_diff')} "
                f"(≤ {guard.get('atol_warn')} WARN headroom)."
            )
        elif guard.get("verdict") == "not_run":
            audit_warnings.append(
                "base-equivalence drift check was not run (no preflight "
                "capture). verdict is not a pass."
            )
        if guard.get("adapter_active") is False:
            audit_warnings.append(
                "Adapter appears inert (enabled logits == disabled). "
                "The audit may measure the base model twice and report zero leak "
                "with false confidence."
            )
    if getattr(scorer_obj, "requires_reference", False) and not (
        use_toggle or ref_model is not None
    ):
        audit_warnings.append(
            f"Scorer {scorer_block['name']} declares requires_reference=True but "
            "no reference is available; score() will receive reference=None."
        )
    ev_by_id = {
        ev.get("id"): ev
        for ev in (preflight_block.get("per_canary") or (preflight_block.get("survival") or {}).get("per_canary") or [])
        if isinstance(ev, dict)
    }

    per_canary: list[dict[str, Any]] = []
    member_scores: list[float] = []
    control_scores: list[float] = []
    toggle_failures: list[str] = []

    def extract_one(m, canary, cache_tag=""):
        return _extract_canary_signals(
            m,
            tokenizer,
            canary,
            fmt,
            skip_first_packed_token,
            cache=signals_cache,
            cache_tag=cache_tag,
        )

    with _eval_guard(model):
        if ref_model is not None:
            ref_model.eval()
        n_total = len(canaries)
        for i, canary in enumerate(canaries):
            if n_total >= 8 and (i == 0 or (i + 1) % 10 == 0 or i + 1 == n_total):
                log.info("scoring canary %s/%s id=%s", i + 1, n_total, canary.get("id"))
            ft_sig, ref_sig, how = _toggle_or_extract_ref(
                model,
                lambda m, tag="", c=canary: extract_one(m, c, cache_tag=tag),
                ref_model=ref_model,
                toggle_safe=use_toggle and is_peft_model(model),
                toggle_failures=toggle_failures,
            )
            combined = _combine_with_scorer(scorer_obj, ft_sig, ref_sig, min_k_pct)
            if not custom_scorer and combined.get("headline_attack_used") != HEADLINE_ATTACK:
                used_headline = HEADLINE_ATTACK_FALLBACK
            if skip_generation:
                regurg_row: dict[str, Any] = {
                    "execution": {"status": "not_run", "reason": "skip_generation"},
                    "regurgitated": None,
                    "by_prefix": [],
                }
            else:
                gen_block = generate_canary_completions(
                    model,
                    tokenizer,
                    canary.get("secret") or "",
                    canary.get("secret_token_ids") or [],
                    prefix_fractions=prefix_fractions,
                )
                per_exec = gen_block.get("execution")
                if not isinstance(per_exec, dict) or not per_exec.get("status"):
                    per_exec = {"status": "executed"}
                if per_exec.get("status") == "executed":
                    regurgitated: bool | None = bool(gen_block.get("regurgitated"))
                else:
                    regurgitated = gen_block.get("regurgitated")
                regurg_row = {
                    "execution": per_exec,
                    "regurgitated": regurgitated,
                    "by_prefix": [
                        {kk: vv for kk, vv in p.items() if kk != "prefix"}
                        for p in (gen_block.get("by_prefix") or [])
                    ],
                }
            ev = ev_by_id.get(canary.get("id")) or {}
            row = {
                "id": canary.get("id"),
                "role": canary.get("role"),
                "included": bool(canary.get("included")),
                "repetitions": canary.get("repetitions"),
                "family": canary.get("family"),
                "requested_family": requested_family_of(canary),
                "actual_generator": actual_generator_of(canary),
                "scores": {k: v for k, v in combined.items() if k != "headline_attack_used"},
                "ref_source": how,
                "regurgitation": regurg_row,
            }
            if ev:
                row["evidence_level"] = ev.get("evidence_level")
                row["record_observed"] = ev.get("record_observed")
                row["secret_token_aligned"] = ev.get("secret_token_aligned")
                row["loss_mask_checked"] = ev.get("loss_mask_checked")
                row["directly_supervised"] = ev.get("directly_supervised")
                row["verification_unknown"] = ev.get("verification_unknown")
            elif not preflight_block.get("ran"):
                row["evidence_level"] = "verification_unknown"
                row["verification_unknown"] = True
            per_canary.append(row)
            score = combined.get("headline_score", float("nan"))
            if canary.get("included"):
                member_scores.append(float(score))
            else:
                control_scores.append(float(score))

    # drop NaNs for stats
    member_scores = [s for s in member_scores if s == s]
    control_scores = [s for s in control_scores if s == s]
    det = tpr_at_fpr(member_scores, control_scores, fpr=fpr)
    auc = roc_auc(member_scores, control_scores)
    roc = roc_points(member_scores, control_scores)
    if profile_spec.get("refuse_headline"):
        det = dict(det)
        det["headline_valid"] = False
        refuse_note = (
            f"audit_profile={profile_spec['name']} is exploratory integration "
            "sanity; TPR@FPR headline is refused by contract (not an identified "
            "operating point)."
        )
        det["warning"] = (
            f"{refuse_note} {det['warning']}" if det.get("warning") else refuse_note
        )
        audit_warnings.append(refuse_note)

    # regurgitation rates — only executed rows enter a denominator
    by_tier: dict[str, dict[str, Any]] = {}
    tier_hits: dict[int, list[bool]] = defaultdict(list)
    overall_flags: list[bool] = []
    for row, canary in zip(per_canary, canaries):
        if not canary.get("included"):
            continue
        if not regurgitation_was_evaluated(row):
            continue
        flag = bool((row.get("regurgitation") or {}).get("regurgitated"))
        overall_flags.append(flag)
        tier_hits[int(canary.get("repetitions") or 1)].append(flag)
    for tier, flags in sorted(tier_hits.items()):
        if not flags:
            continue
        n_tier = len(flags)
        n_hit = int(sum(flags))
        by_tier[str(tier)] = {
            "n": n_tier,
            "n_regurgitated": n_hit,
            "rate": float(n_hit / n_tier),
        }
    overall_rate = (
        float(sum(overall_flags) / len(overall_flags)) if overall_flags else float("nan")
    )

    control_regurg = [
        bool((row.get("regurgitation") or {}).get("regurgitated"))
        for row, c in zip(per_canary, canaries)
        if not c.get("included") and regurgitation_was_evaluated(row)
    ]

    def _headline_score_for_text(text: str) -> float:
        ids, span = locate_secret_span(tokenizer, text, text, None)
        span = span or (0, len(ids))
        ft_sig = extract_token_signals(
            model,
            ids,
            span=span,
            cache=signals_cache,
            cache_tag="target",
        )
        ref_sig = None
        if use_toggle and is_peft_model(model) and hasattr(model, "disable_adapter"):
            try:
                with model.disable_adapter():
                    ref_sig = extract_token_signals(
                        model,
                        ids,
                        span=span,
                        cache=signals_cache,
                        cache_tag="disable_adapter",
                    )
            except Exception as exc:
                toggle_failures.append(f"{type(exc).__name__}: {exc}")
                ref_sig = None
        elif ref_model is not None:
            ref_sig = extract_token_signals(
                ref_model,
                ids,
                span=span,
                cache=signals_cache,
                cache_tag="separate_reference",
            )
        return float(scorer_obj.score(ft_sig, ref_sig))

    def _real_block_for_seed(sample_seed: int, collect_ranked: bool) -> dict[str, Any]:
        real_texts, held_texts, dup_rate, comparison_population = _sample_real_texts(
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
        if comparison_population == "held_out" and len(real_scores) >= 2 and len(held_scores) >= 2:
            set_level = welch_ttest(real_scores, held_scores)
            set_level["kind"] = "inferential_member_vs_nonmember"
            set_level["inferential"] = True
            set_level["note"] = (
                "Set-level comparison against a user-supplied held-out "
                "population. Exploratory at the set level; not evidence about "
                "any individual record; not a membership finding for any one row."
            )
        elif comparison_population == "held_out":
            set_level = {
                "kind": "inferential_member_vs_nonmember",
                "inferential": False,
                "p_value": None,
                "note": (
                    "Genuine held-out population was supplied but a set-level "
                    "test needs >=2 scores on each side."
                ),
            }
        elif comparison_population == "training_split":
            set_level = {
                "kind": "descriptive_ranking_only",
                "inferential": False,
                "p_value": None,
                "n_ranked": len(real_scores),
                "n_comparison_split": len(held_scores),
                "mean_score_train_sample": (
                    float(np.mean(real_scores)) if real_scores else float("nan")
                ),
                "mean_score_comparison_split": (
                    float(np.mean(held_scores)) if held_scores else float("nan")
                ),
                "note": (
                    "No genuine held-out population was supplied. Scores are "
                    "exploratory, descriptive ranking of a training-split sample "
                    "(comparison_population=training_split). Not a "
                    "member-vs-nonmember test; no FPR is attached; not evidence "
                    "about any individual record."
                ),
            }
        else:
            set_level = {
                "kind": "ranking_only",
                "inferential": False,
                "p_value": None,
                "note": (
                    "No comparison-split real records; ranking only. Not a "
                    "member-vs-nonmember test."
                ),
            }
        block: dict[str, Any] = {
            "execution": {"status": "executed"},
            "sample_seed": int(sample_seed),
            "n_train_sampled": len(real_texts),
            "n_comparison_split": len(held_texts),
            "comparison_population": comparison_population,
            "exact_dup_rate": dup_rate,
            "set_level": set_level,
        }
        if dup_rate is None:
            block["exact_dup_note"] = _EXACT_DUP_UNMEASURED_NOTE
        if collect_ranked:
            block["ranked"] = ranked[: min(50, len(ranked))]
            block["redacted"] = not reveal
        return block

    primary_real_seed = seed_list[0] if seed_list else int(manifest.get("seed") or 0)
    real_block: dict[str, Any] | None = None
    real_per_seed: list[dict[str, Any]] | None = None
    if dataset is None:
        real_block = real_records_not_run("no_dataset")
    elif not real_sample or int(real_sample) <= 0:
        real_block = real_records_not_run("real_sample_zero")
    else:
        real_block = _real_block_for_seed(primary_real_seed, collect_ranked=True)
        if real_block.get("exact_dup_rate") is None:
            audit_warnings.append(_EXACT_DUP_UNMEASURED_NOTE)
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

    ref_meta, headline_fallback = reconcile_reference_mode(
        ref_meta,
        per_canary,
        toggle_failures,
        has_ref_model=ref_model is not None,
    )
    if headline_fallback:
        used_headline = headline_fallback
    if toggle_failures:
        audit_warnings.append(
            f"{_TOGGLE_FAIL_WARNING} Last error: {toggle_failures[0]}."
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
            d = tpr_at_fpr(member_scores, boot, fpr=fpr)
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
                (
                    "canary likelihood scoring (teacher-forced forward passes) and "
                    "regurgitation generation (greedy decoding) are deterministic "
                    "given the trained model and were computed once"
                )
                if regurgitation_execution.get("status") == "executed"
                else (
                    "canary likelihood scoring (teacher-forced forward passes) is "
                    "deterministic given the trained model and was computed once; "
                    "regurgitation generation was not run (skip_generation)."
                )
            ),
            "variance": variance,
            "real_records_per_seed": real_per_seed,
        }

    headline_valid = bool(det.get("headline_valid"))
    cal_seed = seed_list[0] if seed_list else int(manifest.get("seed") or 0)
    calibration_stability = bootstrap_calibration_stability(
        member_scores,
        control_scores,
        fpr=fpr,
        seed=cal_seed,
    )
    by_rep = membership_by_repetition(
        per_canary,
        float(det["threshold"]) if det["threshold"] == det["threshold"] else float("nan"),
        pooled_n_detected=int(det["n_detected"]),
        pooled_n_members=int(det["n_members"]),
        pooled_tpr=det["tpr"],
        pooled_ci=(det["ci_low"], det["ci_high"]),
    )
    detection_claim = None
    if headline_valid:
        detection_claim = (
            f"{det['n_detected']}/{det['n_members']} detected at estimated "
            f"FPR ≤ {fpr:g} by construction"
        )
    membership = {
        "headline_attack": used_headline,
        "predeclared_headline": HEADLINE_ATTACK,
        "scorer": scorer_block,
        # Refuse a fake precise TPR@FPR when the control set cannot identify it
        # or the profile (smoke) forbids a headline.
        "tpr_at_1pct_fpr": det["tpr"] if headline_valid and abs(fpr - 0.01) < 1e-12 else None,
        "tpr_at_target_fpr": det["tpr"] if headline_valid else None,
        "ci_low": det["ci_low"] if headline_valid else None,
        "ci_high": det["ci_high"] if headline_valid else None,
        "headline_valid": headline_valid,
        "target_fpr": fpr,
        "detection_claim": detection_claim,
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
        "by_repetition": by_rep,
        "calibration_stability": calibration_stability,
    }
    n_exact = 0
    for row, canary in zip(per_canary, canaries):
        if not canary.get("included"):
            continue
        if not regurgitation_was_evaluated(row):
            continue
        by_prefix = (row.get("regurgitation") or {}).get("by_prefix") or []
        if any(bool(p.get("exact")) for p in by_prefix):
            n_exact += 1
    n_reg_members = len(overall_flags)
    skip_regurg = regurgitation_execution.get("status") != "executed"
    if skip_regurg:
        exact_wording = "not run; no regurgitation measurement in this report"
        regurg_note = (
            "Regurgitation was not run in this audit (skip_generation=true). "
            "This is the absence of a measurement, not a zero-detection result, "
            "and not a claim of no extraction risk."
        )
    elif n_reg_members:
        exact_wording = (
            f"{n_exact}/{n_reg_members} under this prefix/decoding/exact-match protocol"
        )
        regurg_note = (
            f"{exact_wording}. This is not a claim of no extraction risk; "
            "BLEU/NED flags remain supplementary. 1x canaries are MIA-tier only; "
            "verbatim regurgitation is not expected from a single effective occurrence."
        )
    else:
        exact_wording = "generation skipped or no inserted members"
        regurg_note = (
            f"{exact_wording}. This is not a claim of no extraction risk; "
            "BLEU/NED flags remain supplementary. 1x canaries are MIA-tier only; "
            "verbatim regurgitation is not expected from a single effective occurrence."
        )
    n_regurgitated = int(sum(overall_flags)) if overall_flags else 0
    regurgitation = {
        "execution": dict(regurgitation_execution),
        "prefix_policy": {
            "kind": "secret_prefix_fractions",
            "fractions": list(prefix_fractions),
        },
        "decoding": {
            "strategy": "greedy",
            "do_sample": False,
            "temperature": None,
            "method": "greedy prefix-prompt completion",
        },
        "match_rule": "exact",
        "detected": {
            "n": n_reg_members,
            "n_detected": n_exact,
            "rate": float(n_exact / n_reg_members) if n_reg_members else float("nan"),
            "wording": exact_wording,
        },
        "overall": {
            "n": n_reg_members,
            "n_regurgitated": n_regurgitated,
            "rate": overall_rate,
        },
        "by_tier": by_tier,
        "prefix_fractions": list(prefix_fractions),
        "thresholds": {"exact": True, "bleu": 0.75, "ned": 0.10},
        "note": regurg_note,
    }
    control_regurg_rate = (
        float(sum(control_regurg) / len(control_regurg)) if control_regurg else float("nan")
    )
    neg_note = (
        "Held-out / never-inserted canaries. Published calibration anchors: "
        "cross-model false-extraction floor ~6% (Carlini 2022); pre-FT exact-match "
        "baseline <0.06% (Bossy 2025). Those numbers are literature, not this run."
    )
    if skip_regurg:
        neg_note += " Regurgitation was not tested for these controls in this run."
    negative_controls = {
        "n": len(controls),
        "mean_headline_score": float(np.mean(control_scores)) if control_scores else float("nan"),
        "regurgitation_rate": control_regurg_rate,
        "regurgitation": {
            "execution": dict(regurgitation_execution),
            "n_evaluated": len(control_regurg),
            "n_regurgitated": int(sum(control_regurg)) if control_regurg else 0,
            "rate": control_regurg_rate,
        },
        "note": neg_note,
    }

    adapter_info = None
    if is_peft_model(model) or emb.get("r") is not None:
        adapter_info = {
            "r": emb.get("r"),
            "lora_alpha": emb.get("lora_alpha"),
            "bias": emb.get("bias"),
            "modules_to_save": emb.get("modules_to_save"),
            "trainable_token_indices": emb.get("trainable_token_indices"),
            "ensure_weight_tying": emb.get("ensure_weight_tying"),
            "target_parameters": emb.get("target_parameters"),
            "peft_type": emb.get("peft_type"),
            "merged": emb.get("merged"),
            "quantized": emb.get("quantized"),
            "embedding_layer_names": emb.get("embedding_layer_names"),
        }

    manifest_sha = manifest.get("manifest_hash") or sha256_json(manifest)
    families_used = sorted({str(c.get("family")) for c in canaries if c.get("family")})
    requested = sorted({requested_family_of(c) for c in canaries if requested_family_of(c)})
    generators = sorted({actual_generator_of(c) for c in canaries if actual_generator_of(c)})
    requested_family = requested[0] if len(requested) == 1 else requested
    actual_generator = generators[0] if len(generators) == 1 else generators
    audit_profile_block = {
        "name": profile_spec["name"],
        "target_fpr": fpr,
        "exploratory": bool(profile_spec.get("exploratory")),
        "refuse_headline": bool(profile_spec.get("refuse_headline")),
        "inferred": bool(profile_spec.get("inferred")),
        "note": profile_spec.get("note"),
    }
    canaries_block = {
        "requested_family": requested_family,
        "actual_generator": actual_generator,
        "repetitions": repetition_grid,
        "requested_members": int(requested_members) if requested_members is not None else None,
        "controls": len(controls),
    }
    audit_scope = {
        "n_canaries_inserted": len(members),
        "n_heldout_controls": len(controls),
        "requested_members": int(requested_members) if requested_members is not None else None,
        "repetition_grid": repetition_grid,
        "families_used": families_used,
        "requested_family": requested_family,
        "actual_generator": actual_generator,
        "include_prob": manifest.get("include_prob"),
        "inject_seed": manifest.get("seed"),
        "audit_seeds": seed_list,
        "dataset_rows_total": dataset_length(dataset) if dataset is not None else None,
        "real_records_sampled": (real_block or {}).get("n_train_sampled"),
        "audit_profile": audit_profile_block["name"],
        "target_fpr": fpr,
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
        "fpr_target": fpr,
        "target_fpr": fpr,
        "audit_profile": audit_profile_block["name"],
        "min_controls_for_headline": MIN_CONTROLS_FOR_TPR_AT_1PCT,
        "headline_attack_predeclared": HEADLINE_ATTACK,
        "scorer": scorer_block["name"],
        "scorer_version": scorer_block["version"],
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
        preflight=preflight_block,
        provenance=provenance,
        per_canary=per_canary,
        extra={
            "audit_seconds": round(time.perf_counter() - started, 3),
            "audit_warnings": audit_warnings,
        },
        release_context=release_context_norm,
        audit_scope=audit_scope,
        stability=stability,
        audit_profile=audit_profile_block,
        canaries=canaries_block,
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
