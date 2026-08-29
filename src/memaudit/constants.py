"""Stable names and defaults. No fabricated paper numbers as runtime thresholds."""

from __future__ import annotations

# 1.2.0: additive on 1.1.0 (audit_profile, canaries provenance block,
# membership.by_repetition, calibration_stability, regurgitation protocol
# fields, real-record inferential-vs-descriptive split, membership.scorer).
SCHEMA_VERSION = "1.2.0"
DEFAULT_MEMBERSHIP_SCORER = "min_k_plus_plus"
TOOL_VERSION = "0.1.0"

HEADLINE_ATTACK = "base_calibrated_min_k_plus_plus"
HEADLINE_ATTACK_FALLBACK = "min_k_plus_plus"

DEFAULT_FAMILY = "high_ppl"
DEFAULT_N = 32
DEFAULT_N_CONTROLS = 100
DEFAULT_SECRET_LEN = 32
MIN_SECRET_LEN = 25
MAX_SECRET_LEN = 64
DEFAULT_REPETITIONS: tuple[int, ...] = (1, 4, 16)
DEFAULT_INCLUDE_PROB = 0.5
DEFAULT_MIN_K_PCT = 20.0
DEFAULT_PREFIX_FRACTIONS: tuple[float, ...] = (0.25, 0.50)
DEFAULT_REAL_SAMPLE = 64
TOKEN_BUDGET_FRAC = 0.001
# Empirical TPR@1%FPR needs >= 1/0.01 held-out controls. Below that the
# headline is unidentified -- reporting a precise TPR would be a lie.
MIN_CONTROLS_FOR_TPR_AT_1PCT = 100
DEFAULT_TARGET_FPR = 0.01
CALIBRATION_BOOTSTRAP_N = 50

# Named audit profiles. Numbers can evolve; the contract is the concept.
# generate_canaries() without profile= stays 32/100/{1,4,16} (routine-shaped).
# powered is the 100/200/{1,4,16} stress run (case-study shape), not a silent
# shrink of that public default.
AUDIT_PROFILES: dict[str, dict] = {
    "smoke": {
        "n": 8,
        "n_controls": 16,
        "repetitions": (1,),
        "secret_len": 25,
        "real_sample": 8,
        "target_fpr": DEFAULT_TARGET_FPR,
        "exploratory": True,
        "refuse_headline": True,
        "require_calibration_stability": False,
        "note": (
            "Cheap integration / memorization sanity. Exploratory only; "
            "TPR@FPR headline is refused by contract."
        ),
    },
    "routine": {
        "n": DEFAULT_N,
        "n_controls": DEFAULT_N_CONTROLS,
        "repetitions": DEFAULT_REPETITIONS,
        "secret_len": DEFAULT_SECRET_LEN,
        "real_sample": DEFAULT_REAL_SAMPLE,
        "target_fpr": DEFAULT_TARGET_FPR,
        "exploratory": False,
        "refuse_headline": False,
        "require_calibration_stability": False,
        "note": "Recurring fine-tune audit; moderate members and controls.",
    },
    "powered": {
        "n": 100,
        "n_controls": 200,
        "repetitions": DEFAULT_REPETITIONS,
        "secret_len": DEFAULT_SECRET_LEN,
        "real_sample": DEFAULT_REAL_SAMPLE,
        "target_fpr": DEFAULT_TARGET_FPR,
        "exploratory": False,
        "refuse_headline": False,
        "require_calibration_stability": True,
        "note": (
            "Publishable / internal-review stress run: larger controls, "
            "repetition tiers, and threshold/calibration stability required."
        ),
    },
}

REPETITION_TIER_MEANING: dict[int | str, str] = {
    1: "single-exposure probe",
    4: "moderate stress",
    16: "high-exposure stress",
    "pooled": "powered-audit headline",
}

BLEU_THRESHOLD = 0.75
NED_THRESHOLD = 0.10

FAMILY_ALIASES = {
    "high-ppl": "high_ppl",
    "highppl": "high_ppl",
    "high_perplexity": "high_ppl",
    "random-secret": "random",
    "random_secret": "random",
    "n-gram": "bigram",
    "ngram": "bigram",
    "n_gram": "bigram",
    "new-token": "new_token",
    "newtoken": "new_token",
}

IMPLEMENTED_FAMILIES = frozenset({"high_ppl", "unigram", "bigram", "structured", "random"})

NEW_TOKEN_MESSAGE = (
    "The 'new_token' canary family is gated and unimplemented in v0.1. "
    "memaudit never resizes the vocabulary (new embedding rows stay random "
    "under frozen-embedding LoRA and silently produce a null audit). "
    "Use family='high_ppl' (default): regular-token high-perplexity canaries "
    "that remain detectable under LoRA with frozen embeddings."
)

# EDPB Opinion 28/2024 para 46: the release context changes which attack
# surface is "reasonably likely". User-declared; never inferred.
RELEASE_CONTEXTS = ("public-api", "internal", "open-weights", "unspecified")
DEFAULT_RELEASE_CONTEXT = "unspecified"

# Verbatim quote from EDPB Opinion 28/2024 para 55 (the testing paragraph).
EDPB_55_QUOTE = (
    "successful testing which covers widely known, state-of-the-art attacks "
    "can only be evidence for the resistance to those attacks"
)

ANNEX_DISCLAIMER = (
    "This annex is documented test evidence in the sense of EDPB Opinion "
    f"28/2024 para 55/para 58. Per para 55, \u201c{EDPB_55_QUOTE}\u201d. This report is "
    "therefore evidence only for the attacks listed as in scope below. It "
    "does not constitute a determination of anonymity or GDPR compliance, "
    "and it is not a CNIL / AI Act certification."
)

LIMITATIONS_STATEMENT = (
    "This report is evidence of resistance to the attacks that were actually "
    "run: (i) membership inference via a pre-registered canary MIA and "
    "(iii) regurgitation via prefix-prompted generation. It is not evidence "
    "of resistance to inversion, reconstruction, attribute inference, or "
    "exfiltration. Per EDPB Opinion 28/2024 para 55, \u201c" + EDPB_55_QUOTE + "\u201d. "
    "This report does not constitute a determination of anonymity or GDPR "
    "compliance, and it is not a GDPR / AI Act / CNIL compliance "
    "certification. Real-record ranking is exploratory and descriptive: no "
    "FPR is attached, and it is not evidence about any individual record. "
    "Set-level member-vs-nonmember inference is only supported when the "
    "user supplies a genuine held-out population. Canary detections are "
    "reported as detected at estimated FPR ≤ α by construction. Thresholds "
    "are calibrated on this run's held-out canary controls and do not "
    "transfer across model families. A small TPR or a small set-level "
    "p-value absence is not a privacy guarantee. memaudit runs entirely "
    "locally and does not phone home."
)


def get_audit_profile(name: str) -> dict:
    """Return a copy of a named profile. Raises ``KeyError`` if unknown."""
    key = str(name).strip().lower()
    spec = AUDIT_PROFILES[key]
    out = dict(spec)
    out["name"] = key
    out["repetitions"] = tuple(spec["repetitions"])
    return out


def infer_audit_profile_name(
    *,
    requested_members: int | None,
    n_controls: int | None,
    repetitions: tuple[int, ...] | list[int] | None,
) -> str:
    """Map observed counts onto a named profile, else ``custom``.

    Exact (n, n_controls, repetitions) match wins. A 100/200/{1,4,16}
    (or larger) stress run is treated as powered-shaped so calibration
    stability stays required. Small leftover counts are ``custom``, not
    silent smoke (smoke must be requested so the headline refuse is
    intentional).
    """
    reps = tuple(sorted(int(r) for r in (repetitions or ())))
    for name, spec in AUDIT_PROFILES.items():
        if (
            requested_members == spec["n"]
            and n_controls == spec["n_controls"]
            and tuple(sorted(int(r) for r in spec["repetitions"])) == reps
        ):
            return name
    if (
        requested_members is not None
        and n_controls is not None
        and requested_members >= 100
        and n_controls >= 200
        and {1, 4, 16}.issubset(set(reps))
    ):
        return "powered"
    return "custom"


def resolve_audit_profile(
    name: str | None,
    *,
    requested_members: int | None = None,
    n_controls: int | None = None,
    repetitions: tuple[int, ...] | list[int] | None = None,
    target_fpr: float | None = None,
) -> dict:
    """Resolve an explicit or inferred profile into a report block + knobs."""
    inferred = False
    if name:
        spec = get_audit_profile(name)
    else:
        inferred_name = infer_audit_profile_name(
            requested_members=requested_members,
            n_controls=n_controls,
            repetitions=repetitions,
        )
        if inferred_name in AUDIT_PROFILES:
            spec = get_audit_profile(inferred_name)
            inferred = True
        else:
            spec = {
                "name": "custom",
                "target_fpr": DEFAULT_TARGET_FPR,
                "exploratory": False,
                "refuse_headline": False,
                "require_calibration_stability": False,
                "note": (
                    "Counts do not match a named smoke / routine / powered "
                    "profile. Headline policy follows control-count rules only."
                ),
                "repetitions": tuple(int(r) for r in (repetitions or ())),
            }
            inferred = True
    if target_fpr is not None:
        spec["target_fpr"] = float(target_fpr)
    spec["inferred"] = inferred
    spec["requested_members"] = requested_members
    spec["resolved_n_controls"] = n_controls
    return spec
