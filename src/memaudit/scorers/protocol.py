"""Membership-scorer backend protocol.

Scorers are pure functions of pre-extracted ``TokenSignals``. They must not
threshold, calibrate FPR, compute CIs, or see raw models. Higher score =
more member-like.
"""

from __future__ import annotations

from typing import Protocol

from memaudit.scorers.signals import TokenSignals


class MembershipScorer(Protocol):
    name: str
    version: str
    requires_reference: bool
    forward_passes_per_record: int

    def score(self, target: TokenSignals, reference: TokenSignals | None) -> float:
        """Scalar membership score. Higher = more member-like."""
        ...
