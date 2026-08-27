"""Stable names and defaults. No fabricated paper numbers as runtime thresholds."""

from __future__ import annotations

# 1.1.0: additive only vs 1.0.0 (compliance_annex, provenance enrichment,
# report_sha256 self-hash, optional stability block, release_context).
SCHEMA_VERSION = "1.1.0"
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
    "certification. Per-record flags on real data are exploratory; the "
    "set-level test is the supported real-data claim. Thresholds are "
    "calibrated on this run's held-out canary controls and do not transfer "
    "across model families. A small TPR or a small set-level p-value "
    "absence is not a privacy guarantee. memaudit runs entirely locally "
    "and does not phone home."
)
