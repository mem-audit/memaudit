"""Default membership backend: Zhang et al. Min-K%++.

When a reference ``TokenSignals`` is present, the score is the base-calibrated
difference (target − reference), matching the historical
``base_calibrated_min_k_plus_plus`` headline. Without a reference it is
target-only Min-K%++. No thresholding lives here.
"""

from __future__ import annotations

from memaudit.constants import DEFAULT_MEMBERSHIP_SCORER, DEFAULT_MIN_K_PCT
from memaudit.scorers.signals import TokenSignals
from memaudit.stats import min_k_plus_plus

DEFAULT_SCORER_NAME = DEFAULT_MEMBERSHIP_SCORER
DEFAULT_SCORER_VERSION = "1.0.0"


class MinKPlusPlusScorer:
    """Reference-optional Min-K%++. Default v0.1 backend — do not swap."""

    name = DEFAULT_SCORER_NAME
    version = DEFAULT_SCORER_VERSION
    requires_reference = False
    forward_passes_per_record = 1

    def __init__(self, min_k_pct: float = DEFAULT_MIN_K_PCT) -> None:
        self.min_k_pct = float(min_k_pct)

    def score(self, target: TokenSignals, reference: TokenSignals | None) -> float:
        ft = min_k_plus_plus(target.gold_logprob, target.mu, target.sigma, self.min_k_pct)
        if reference is None:
            return float(ft)
        ref = min_k_plus_plus(
            reference.gold_logprob, reference.mu, reference.sigma, self.min_k_pct
        )
        return float(ft - ref)
