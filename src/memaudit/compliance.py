"""EDPB Opinion 28/2024 compliance annex (para 46 / para 55 / para 58).

Everything here is additive report content plus a markdown renderer. The
mapping follows ``literature-review/03-landscape-tools-competitors/
regulatory-hooks.md``; the canary-family threat models come from
``literature-review/02-canaries-mia-auditing/SYNTHESIS.md`` section B1.

Honesty is the compliance feature: para 55 says successful testing "can only
be evidence for the resistance to those attacks", so attacks that were NOT
run are named as out of scope instead of being omitted.
"""

from __future__ import annotations

from typing import Any

from memaudit.constants import (
    ANNEX_DISCLAIMER,
    DEFAULT_RELEASE_CONTEXT,
    EDPB_55_QUOTE,
    HEADLINE_ATTACK,
    RELEASE_CONTEXTS,
)
from memaudit.exceptions import MemauditConfigError

# ---------------------------------------------------------------------------
# Static tables (published anchors; do not edit numbers without a source)
# ---------------------------------------------------------------------------

# SYNTHESIS.md section B1: canary family -> threat model. ``status`` records
# whether memaudit v0.1 can generate the family; the table intentionally also
# carries families memaudit does NOT implement so the annex can say why.
CANARY_FAMILY_THREAT_MODELS: dict[str, dict[str, str]] = {
    "high_ppl": {
        "construction": (
            "regular-token suffixes sampled from the base model at high "
            "temperature, rejection-sampled into a target perplexity band "
            "(falls back to uniform-from-vocab draws when no model and no corpus "
            "are given; rare-token unigram only when a corpus is supplied)"
        ),
        "access_needed_to_generate": "sampling access to the base model (or none, in fallback)",
        "threat_scenario": (
            "attacker-known rare sequences planted in the fine-tuning corpus; "
            "the LoRA-safe default family (published TPR@1%FPR 0.94-0.99 under "
            "LoRA r=4, frozen embeddings)"
        ),
        "source": "Meeus et al. 2025 (The Canary's Echo, ICML 2025)",
        "status": "implemented",
    },
    "unigram": {
        "construction": "least-frequent tokens under a corpus unigram model",
        "access_needed_to_generate": "corpus token statistics only",
        "threat_scenario": "best no-model-access family; power scales with secret length",
        "source": "Panda et al. 2025 (ICLR 2025)",
        "status": "implemented",
    },
    "bigram": {
        "construction": "token sequences improbable under corpus n-gram statistics",
        "access_needed_to_generate": "corpus n-gram statistics only",
        "threat_scenario": (
            "best regular-token family on larger models "
            "(gpt2-xl, pythia-1.4b, llama-3.2-1b in the published grid)"
        ),
        "source": "Panda et al. 2025 (ICLR 2025)",
        "status": "implemented",
    },
    "structured": {
        "construction": "format template plus random fill (e.g. CANARY-ID:391-...)",
        "access_needed_to_generate": "none",
        "threat_scenario": (
            "structured-secret leakage (account numbers, IDs); the only family "
            "with a native exposure metric"
        ),
        "source": "Carlini et al. 2019 (The Secret Sharer)",
        "status": "implemented",
    },
    "random": {
        "construction": "secret tokens drawn uniformly from the existing vocabulary",
        "access_needed_to_generate": "none",
        "threat_scenario": (
            "baseline family (Gemini-style practice); never best in published "
            "head-to-heads but needs zero access and is maximally out-of-"
            "distribution for the host corpus"
        ),
        "source": "Panda et al. 2025 (ICLR 2025)",
        "status": "implemented",
    },
    "model_based": {
        "construction": "least-likely continuations queried from the pre-fine-tuning model",
        "access_needed_to_generate": "black-box query access to the base model",
        "threat_scenario": "rarely beats unigram; not worth the access requirement",
        "source": "Panda et al. 2025 (ICLR 2025)",
        "status": "not_implemented",
    },
    "new_token": {
        "construction": "brand-new vocabulary rows used as secrets",
        "access_needed_to_generate": "model-developer access (vocabulary resize + trainable embeddings)",
        "threat_scenario": (
            "strongest family when embeddings train; collapses under frozen-"
            "embedding LoRA (audit 0.74 -> 0.05, Panda Table 14) and degrades "
            "utility. Gated and unimplemented: memaudit never resizes the vocab"
        ),
        "source": "Panda et al. 2025 (ICLR 2025)",
        "status": "gated_unimplemented",
    },
}

# EDPB para 58(c): threat model per test actually run - attacker access level
# and knowledge assumptions for each verdict tier.
AUDIT_ATTACK_THREAT_MODELS: dict[str, dict[str, str]] = {
    "membership_inference": {
        "edpb_class": "para 55(i) membership inference",
        "attacker_access": (
            "grey-box: per-token log-probability access to the fine-tuned model "
            "and to the pre-fine-tuning reference checkpoint"
        ),
        "attacker_knowledge": (
            "attacker knows the candidate record verbatim and computes "
            "likelihood scores on both models; decision thresholds are "
            "calibrated on this run's held-out canary controls"
        ),
        "method": (
            f"{HEADLINE_ATTACK} on the secret span (masked loss, loss ratio, "
            "Min-K% computed from the same two forward passes); "
            "TPR@1%FPR with Clopper-Pearson 95% CI"
        ),
    },
    "regurgitation": {
        "edpb_class": "para 55(iii) regurgitation of training data",
        "attacker_access": "black-box: text-generation access only (greedy decoding)",
        "attacker_knowledge": (
            "attacker knows a prefix (25% / 50% of the secret tokens) and "
            "prompts the model for the continuation"
        ),
        "method": (
            "greedy prefix-prompted completion scored exact match / "
            "BLEU>0.75 / sliding-window normalized edit distance <=0.10"
        ),
    },
}

# para 46: what attack surface the declared release context implies.
RELEASE_CONTEXT_SURFACES: dict[str, str] = {
    "public-api": (
        "Model exposed through a public API: the reasonably-likely attacker "
        "has query access (regurgitation surface) and, where the API returns "
        "log-probabilities, the grey-box MIA surface tested here. Deployment "
        "controls (rate limits, output filters) are not part of this artifact."
    ),
    "internal": (
        "Model accessible only to employees / internal systems: the attacker "
        "surface is insider-level query access. EDPB para 46 allows different "
        "levels of testing for internal models, but the documentation duty "
        "(para 58) still applies."
    ),
    "open-weights": (
        "Weights are (to be) released: attackers gain full white-box access. "
        "Every attack class in para 55 becomes available, including attacks "
        "this audit did NOT run (inversion, reconstruction, exfiltration, "
        "attribute inference). Grey/black-box resistance evidence understates "
        "open-weights risk."
    ),
    "unspecified": (
        "No release context was declared. Per para 46 the required level of "
        "testing and resistance depends on it; declare one via "
        "run_audit(release_context=...) or --release-context."
    ),
}

_RELEASE_ALIASES = {
    "public_api": "public-api",
    "publicapi": "public-api",
    "api": "public-api",
    "public": "public-api",
    "open_weights": "open-weights",
    "openweights": "open-weights",
    "weights": "open-weights",
    "none": "unspecified",
}


def normalize_release_context(value: str | None) -> str:
    if value is None or value == "":
        return DEFAULT_RELEASE_CONTEXT
    key = str(value).strip().lower().replace(" ", "-")
    key = _RELEASE_ALIASES.get(key.replace("-", "_"), key)
    if key not in RELEASE_CONTEXTS:
        raise MemauditConfigError(
            f"release_context={value!r} is not one of {list(RELEASE_CONTEXTS)}. "
            "It is a user declaration (EDPB para 46), never inferred."
        )
    return key


def attack_coverage_table(membership: dict[str, Any], regurgitation: dict[str, Any]) -> list[dict[str, Any]]:
    """The para 55 attack-class table: what ran, what explicitly did not."""
    prefix_fracs = regurgitation.get("prefix_fractions")
    headline = membership.get("headline_attack") or HEADLINE_ATTACK
    scorer_meta = membership.get("scorer") or {}
    scorer_note = ""
    if scorer_meta.get("name"):
        scorer_note = f" (scorer {scorer_meta.get('name')} v{scorer_meta.get('version')})"
    return [
        {
            "attack_class": "membership inference",
            "edpb_ref": "para 55(i)",
            "status": "in_scope",
            "method": (
                f"pre-registered canary MIA: {headline}{scorer_note} "
                "on the secret span, thresholded on held-out canary controls "
                "(TPR at the profile target FPR + Clopper-Pearson 95% CI; "
                "canaries detected at estimated FPR ≤ α by construction; "
                "masked loss / loss ratio / Min-K% / Min-K%++ from the same "
                "forward passes). Real-record ranking is descriptive unless a "
                "genuine held-out population is supplied"
            ),
        },
        {
            "attack_class": "attribute inference",
            "edpb_ref": "para 55(i)",
            "status": "out_of_scope",
            "note": "not run; this report is no evidence of resistance to it",
        },
        {
            "attack_class": "exfiltration",
            "edpb_ref": "para 55(ii)",
            "status": "out_of_scope",
            "note": "not run; this report is no evidence of resistance to it",
        },
        {
            "attack_class": "regurgitation of training data",
            "edpb_ref": "para 55(iii)",
            "status": "in_scope",
            "method": (
                "greedy prefix-prompted completion on inserted canaries "
                f"(prefix fractions {prefix_fracs}), protocol-scoped as "
                "prefix/decoding/exact-match (BLEU/NED supplementary), "
                "with never-inserted negative controls. A 0/N exact result "
                "is not a claim of no extraction risk"
            ),
        },
        {
            "attack_class": "model inversion",
            "edpb_ref": "para 55(iv)",
            "status": "out_of_scope",
            "note": "not run; this report is no evidence of resistance to it",
        },
        {
            "attack_class": "reconstruction",
            "edpb_ref": "para 55(v)",
            "status": "out_of_scope",
            "note": "not run; this report is no evidence of resistance to it",
        },
    ]


def build_compliance_annex(
    *,
    membership: dict[str, Any],
    regurgitation: dict[str, Any],
    negative_controls: dict[str, Any],
    real_records: dict[str, Any] | None,
    audit_scope: dict[str, Any] | None,
    release_context: str | None,
    stability: dict[str, Any] | None,
    created_at: str,
    tool_version: str,
) -> dict[str, Any]:
    """Assemble the EDPB-mapped annex from already-computed report sections."""
    scope = dict(audit_scope or {})
    context = normalize_release_context(release_context)
    families_used = list(scope.get("families_used") or [])
    family_rows = {}
    actual_gen = scope.get("actual_generator")
    for fam in families_used:
        row = dict(
            CANARY_FAMILY_THREAT_MODELS.get(
                fam,
                {
                    "construction": "unknown family (not in the published table)",
                    "access_needed_to_generate": "unknown",
                    "threat_scenario": "unknown",
                    "source": "n/a",
                    "status": "unknown",
                },
            )
        )
        if actual_gen and not isinstance(actual_gen, (list, tuple)) and actual_gen != fam:
            row["actual_generator"] = actual_gen
            row["requested_family"] = scope.get("requested_family") or fam
            row["construction"] = (
                f"{row.get('construction')} [this run: requested_family={fam}, "
                f"actual_generator={actual_gen}]"
            )
        family_rows[fam] = row
    return {
        "standard": "EDPB Opinion 28/2024 (adopted 17 December 2024), para 46 / para 55 / para 58",
        "disclaimer": ANNEX_DISCLAIMER,
        "attack_coverage": attack_coverage_table(membership, regurgitation),
        "threat_models": {
            "edpb_ref": "para 58(c): threat model and risk assessments",
            "attacks": AUDIT_ATTACK_THREAT_MODELS,
            "canary_families_used": family_rows,
        },
        "test_scope": {
            "edpb_ref": "para 55: scope, frequency, quantity and quality of tests",
            "n_canaries_inserted": scope.get("n_canaries_inserted", membership.get("n_members")),
            "n_heldout_controls": scope.get("n_heldout_controls", membership.get("n_controls")),
            "repetition_grid": scope.get("repetition_grid"),
            "canary_families": families_used or None,
            "requested_family": scope.get("requested_family"),
            "actual_generator": scope.get("actual_generator"),
            "requested_members": scope.get("requested_members"),
            "audit_profile": scope.get("audit_profile"),
            "target_fpr": scope.get("target_fpr") or membership.get("target_fpr"),
            "include_prob": scope.get("include_prob"),
            "seeds": {
                "inject_seed": scope.get("inject_seed"),
                "audit_seeds": scope.get("audit_seeds"),
            },
            "dataset_rows_total": scope.get("dataset_rows_total"),
            "real_records_sampled": (real_records or {}).get("n_train_sampled"),
            "negative_controls": {
                "n": negative_controls.get("n"),
                "regurgitation_rate": negative_controls.get("regurgitation_rate"),
                "mean_headline_score": negative_controls.get("mean_headline_score"),
            },
            "run_date_utc": created_at,
            "tool_version": tool_version,
        },
        "quantified_results": {
            "membership": {
                "headline_attack": membership.get("headline_attack"),
                "scorer": membership.get("scorer"),
                "tpr_at_1pct_fpr": membership.get("tpr_at_1pct_fpr"),
                "ci_low": membership.get("ci_low"),
                "ci_high": membership.get("ci_high"),
                "headline_valid": membership.get("headline_valid"),
                "auc_secondary": membership.get("auc"),
                "n_members": membership.get("n_members"),
                "n_controls": membership.get("n_controls"),
                "target_fpr": membership.get("target_fpr"),
                "detection_claim": membership.get("detection_claim"),
                "by_repetition": membership.get("by_repetition"),
                "calibration_stability": membership.get("calibration_stability"),
            },
            "regurgitation": {
                "overall": regurgitation.get("overall"),
                "by_tier": regurgitation.get("by_tier"),
                "prefix_policy": regurgitation.get("prefix_policy"),
                "decoding": regurgitation.get("decoding"),
                "match_rule": regurgitation.get("match_rule"),
                "detected": regurgitation.get("detected"),
            },
            "real_records_set_level": (real_records or {}).get("set_level"),
            "stability": (stability or {}).get("variance"),
        },
        "release_context": {
            "edpb_ref": "para 46",
            "declared": context,
            "declared_by": "user" if context != DEFAULT_RELEASE_CONTEXT else None,
            "implied_attack_surface": RELEASE_CONTEXT_SURFACES[context],
        },
        "limitations": {
            "edpb_55_quote": EDPB_55_QUOTE,
            "statement": ANNEX_DISCLAIMER,
        },
    }


def ensure_annex(report: dict[str, Any]) -> dict[str, Any]:
    """Return the report's annex, rebuilding it for schema-1.0.x reports."""
    annex = report.get("compliance_annex")
    if isinstance(annex, dict) and annex:
        return annex
    return build_compliance_annex(
        membership=report.get("membership") or {},
        regurgitation=report.get("regurgitation") or {},
        negative_controls=report.get("negative_controls") or {},
        real_records=report.get("real_records"),
        audit_scope=report.get("audit_scope"),
        release_context=(report.get("release_context") or {}).get("declared")
        if isinstance(report.get("release_context"), dict)
        else report.get("release_context"),
        stability=report.get("stability"),
        created_at=report.get("created_at") or "",
        tool_version=report.get("tool_version") or "",
    )


# ---------------------------------------------------------------------------
# Markdown rendering (memaudit report --annex <report.json>)
# ---------------------------------------------------------------------------


def _fmt(value: Any, digits: int = 3) -> str:
    if value is None:
        return "-"
    if isinstance(value, bool):
        return "yes" if value else "no"
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def _md_escape(value: Any) -> str:
    return str(value).replace("|", "\\|").replace("\n", " ")


def render_annex_markdown(report: dict[str, Any]) -> str:
    """Human-readable annex for handing to a DPO. Pure function of the report."""
    annex = ensure_annex(report)
    mem = (annex.get("quantified_results") or {}).get("membership") or {}
    reg = (annex.get("quantified_results") or {}).get("regurgitation") or {}
    scope = annex.get("test_scope") or {}
    rc = annex.get("release_context") or {}
    model = report.get("model") or {}
    provenance = report.get("provenance") or {}
    stability = report.get("stability")

    lines: list[str] = []
    add = lines.append
    add("# memaudit compliance annex - EDPB Opinion 28/2024")
    add("")
    add(f"*Generated from a memaudit report (schema {report.get('schema_version')}, "
        f"tool {report.get('tool_version')}), run date {scope.get('run_date_utc') or report.get('created_at')}.*")
    add("")
    add(f"- **Model:** {_md_escape(model.get('name_or_path') or model.get('class') or 'unknown')}"
        + (f" ({_md_escape(model.get('class'))})" if model.get("name_or_path") and model.get("class") else ""))
    add(f"- **Report SHA-256 (canonical content):** `{report.get('report_sha256') or 'not signed - re-write via memaudit'}`")
    add(f"- **Canary manifest SHA-256:** `{report.get('canary_manifest_hash') or '-'}`")
    add(f"- **Verify:** `memaudit verify <this-report>.json`")
    add("")

    add("## 1. What this document is (and is not)")
    add("")
    add(annex.get("disclaimer") or "")
    add("")

    add("## 2. Release context (para 46)")
    add("")
    add(f"- **Declared release context:** `{rc.get('declared')}`"
        + ("" if rc.get("declared_by") else " (not declared by the user)"))
    add(f"- **Implied attack surface:** {rc.get('implied_attack_surface')}")
    add("")

    add("## 3. Attack coverage (para 55)")
    add("")
    add("| Attack class | EDPB ref | Status | Method / note |")
    add("|---|---|---|---|")
    for row in annex.get("attack_coverage") or []:
        status = "IN SCOPE" if row.get("status") == "in_scope" else "OUT OF SCOPE"
        detail = row.get("method") or row.get("note") or ""
        add(f"| {_md_escape(row.get('attack_class'))} | {_md_escape(row.get('edpb_ref'))} "
            f"| {status} | {_md_escape(detail)} |")
    add("")
    add("Per para 55, results below are evidence only for the attacks marked IN SCOPE.")
    add("")

    add("## 4. Quantified results")
    add("")
    add("| Metric | Value |")
    add("|---|---|")
    add(f"| Membership headline attack | `{mem.get('headline_attack')}` |")
    scorer_row = mem.get("scorer") or {}
    if scorer_row.get("name"):
        add(
            f"| Membership scorer | `{scorer_row.get('name')}` v{scorer_row.get('version')} |"
        )
    tpr = mem.get("tpr_at_1pct_fpr")
    if mem.get("headline_valid") and tpr is not None:
        claim = mem.get("detection_claim")
        extra = f" — {claim}" if claim else ""
        add(f"| TPR @ 1% FPR | **{_fmt(tpr)}** (95% CI [{_fmt(mem.get('ci_low'))}, {_fmt(mem.get('ci_high'))}]){extra} |")
    else:
        add("| TPR @ target FPR | refused (see membership.warning / audit_profile) |")
    add(f"| Members / held-out canary controls | {_fmt(mem.get('n_members'))} / {_fmt(mem.get('n_controls'))} |")
    add(f"| AUC (secondary, average-case) | {_fmt(mem.get('auc_secondary'))} |")
    by_rep = mem.get("by_repetition") or {}
    for tier in ("1", "4", "16", "pooled"):
        block = by_rep.get(tier)
        if not isinstance(block, dict):
            continue
        add(
            f"| Membership at {tier} "
            f"({block.get('meaning') or 'tier'}) | "
            f"{_fmt(block.get('detected'))}/{_fmt(block.get('n'))} "
            f"(TPR {_fmt(block.get('tpr'))}) |"
        )
    overall = reg.get("overall") or {}
    detected = reg.get("detected") or {}
    if detected.get("wording"):
        add(f"| Regurgitation (exact-match protocol) | {detected.get('wording')} |")
    add(f"| Regurgitation rate (overall, incl. approx.) | {_fmt(overall.get('rate'))} "
        f"({_fmt(overall.get('n_regurgitated'))}/{_fmt(overall.get('n'))}) |")
    for tier, block in (reg.get("by_tier") or {}).items():
        add(f"| Regurgitation at {tier}x repetitions | {_fmt((block or {}).get('rate'))} |")
    neg = scope.get("negative_controls") or {}
    add(f"| Negative controls (never inserted) | n={_fmt(neg.get('n'))}, "
        f"regurgitation {_fmt(neg.get('regurgitation_rate'))}, "
        f"mean headline score {_fmt(neg.get('mean_headline_score'))} |")
    set_level = (annex.get("quantified_results") or {}).get("real_records_set_level") or {}
    if set_level.get("inferential"):
        add(f"| Real-record set-level test (user-supplied held-out) | p={_fmt(set_level.get('p_value'), 4)} "
            f"(set-level only; not evidence about any individual record) |")
    elif set_level:
        add(
            "| Real-record scores | descriptive ranking only "
            "(no genuine held-out population; not a member-vs-nonmember test) |"
        )
    add("")

    add("## 5. Threat model per test (para 58(c))")
    add("")
    for name, tm in (annex.get("threat_models") or {}).get("attacks", {}).items():
        add(f"### {name.replace('_', ' ')} - {tm.get('edpb_class')}")
        add("")
        add(f"- **Attacker access:** {tm.get('attacker_access')}")
        add(f"- **Attacker knowledge:** {tm.get('attacker_knowledge')}")
        add(f"- **Method:** {tm.get('method')}")
        add("")
    fams = (annex.get("threat_models") or {}).get("canary_families_used") or {}
    if fams:
        add("### Canary families used in this audit")
        add("")
        add("| Family | Construction | Access needed to generate | Threat scenario | Source |")
        add("|---|---|---|---|---|")
        for fam, row in fams.items():
            add(f"| `{fam}` | {_md_escape(row.get('construction'))} "
                f"| {_md_escape(row.get('access_needed_to_generate'))} "
                f"| {_md_escape(row.get('threat_scenario'))} | {_md_escape(row.get('source'))} |")
        add("")

    add("## 6. Scope, frequency, quantity, quality (para 55)")
    add("")
    add("| Field | Value |")
    add("|---|---|")
    if scope.get("audit_profile") or scope.get("target_fpr") is not None:
        add(f"| Audit profile / target FPR | {scope.get('audit_profile')} / {_fmt(scope.get('target_fpr'))} |")
    add(f"| Canaries inserted | {_fmt(scope.get('n_canaries_inserted'))} |")
    add(f"| Requested members | {_fmt(scope.get('requested_members'))} |")
    add(f"| Held-out canary controls | {_fmt(scope.get('n_heldout_controls'))} |")
    add(f"| Repetition grid | {scope.get('repetition_grid')} |")
    add(f"| Canary families | {scope.get('canary_families')} |")
    if scope.get("requested_family") or scope.get("actual_generator"):
        add(f"| Requested family / actual generator | {scope.get('requested_family')} / {scope.get('actual_generator')} |")
    add(f"| Inclusion probability (coin flips) | {_fmt(scope.get('include_prob'))} |")
    seeds = scope.get("seeds") or {}
    add(f"| Inject seed / audit seeds | {_fmt(seeds.get('inject_seed'))} / {seeds.get('audit_seeds') or 'single-seed'} |")
    add(f"| Training dataset rows (as sampled for audit) | {_fmt(scope.get('dataset_rows_total'))} |")
    add(f"| Real records scored | {_fmt(scope.get('real_records_sampled'))} |")
    add(f"| Run date (UTC) | {scope.get('run_date_utc')} |")
    add(f"| Tool version | {scope.get('tool_version')} |")
    add("")
    if stability:
        var = stability.get("variance") or {}
        add("### Multi-seed stability (audit-procedure variance)")
        add("")
        add(stability.get("label") or "")
        add("")
        add(f"- TPR mean / min / max across audit seeds: {_fmt(var.get('tpr_mean'))} / "
            f"{_fmt(var.get('tpr_min'))} / {_fmt(var.get('tpr_max'))}")
        per_seed = var.get("per_seed") or []
        if per_seed:
            add(f"- Per-seed TPR: " + ", ".join(
                f"seed {p.get('seed')}: {_fmt(p.get('tpr'))}" for p in per_seed))
        add("")

    add("## 7. Provenance and verification (para 58 documentation duty)")
    add("")
    add("| Field | Value |")
    add("|---|---|")
    env = provenance.get("environment") or {}
    add(f"| Tool / schema | memaudit {report.get('tool_version')} / schema {report.get('schema_version')} |")
    add(f"| Python / torch / transformers | {env.get('python')} / {env.get('torch')} / {env.get('transformers')} |")
    fp = provenance.get("dataset_fingerprint") or {}
    add(f"| Dataset fingerprint | rows={_fmt(fp.get('n_rows'))}, "
        f"first record sha256 `{(fp.get('first_record_sha256') or '-')[:16]}...`, "
        f"last record sha256 `{(fp.get('last_record_sha256') or '-')[:16]}...` |")
    mf = provenance.get("model_fingerprint") or {}
    add(f"| Model config sha256 | `{(mf.get('config_sha256') or '-')[:16]}...` |")
    add(f"| Canary manifest sha256 | `{report.get('canary_manifest_hash') or '-'}` |")
    add(f"| Report self-hash (report_sha256) | `{report.get('report_sha256') or '-'}` |")
    add("| Local-only / phone-home | "
        f"{_fmt(report.get('local_only'))} / {_fmt(report.get('phone_home'))} |")
    add("")
    add("Integrity: `memaudit verify <report.json>` recomputes the canonical-content "
        "SHA-256 and checks it against `report_sha256` and the `<report>.sha256` "
        "sidecar. Cryptographic signing (GPG / sigstore) of the report file is a "
        "release-runbook step outside memaudit; memaudit does not manage keys.")
    add("")
    add("---")
    add("")
    add(f"*Limitations (verbatim):* {report.get('limitations') or (annex.get('limitations') or {}).get('statement')}")
    add("")
    return "\n".join(lines)
