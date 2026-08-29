"""Pluggable membership-scorer backends.

Default is Min-K%++. EZ-MIA is documented as a future file
(``docs/membership-scorers.md``), not a shipped attack.
"""

from memaudit.scorers.min_k import DEFAULT_SCORER_NAME, DEFAULT_SCORER_VERSION, MinKPlusPlusScorer
from memaudit.scorers.protocol import MembershipScorer
from memaudit.scorers.registry import resolve_scorer, scorer_provenance
from memaudit.scorers.signals import SignalsCache, TokenSignals

__all__ = [
    "DEFAULT_SCORER_NAME",
    "DEFAULT_SCORER_VERSION",
    "MembershipScorer",
    "MinKPlusPlusScorer",
    "SignalsCache",
    "TokenSignals",
    "resolve_scorer",
    "scorer_provenance",
]
