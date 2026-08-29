"""Cacheable per-record, per-model token signals.

This is the array type that crosses the Privacy Meter boundary: extraction
(model-facing, in ``memaudit.scoring``) produces ``TokenSignals``; a
``MembershipScorer`` is a pure reduction of those arrays. Calibration,
thresholds, and reporting stay in orchestration.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True)
class TokenSignals:
    """Per-record, per-model teacher-forced signals (span-restricted).

    ``gold_logprob`` / ``mu`` / ``sigma`` are the existing Min-K%++ inputs.
    ``argmax_correct`` is the extra signal a future EZ-MIA backend needs
    (top-1 prediction == gold token). Higher-level scores are *not* stored
    here — scorers compute a scalar from these arrays.
    """

    gold_logprob: np.ndarray
    mu: np.ndarray
    sigma: np.ndarray
    argmax_correct: np.ndarray

    def __post_init__(self) -> None:
        object.__setattr__(self, "gold_logprob", np.asarray(self.gold_logprob, dtype=np.float64))
        object.__setattr__(self, "mu", np.asarray(self.mu, dtype=np.float64))
        object.__setattr__(self, "sigma", np.asarray(self.sigma, dtype=np.float64))
        object.__setattr__(self, "argmax_correct", np.asarray(self.argmax_correct, dtype=bool))

    @property
    def n_scored_tokens(self) -> int:
        return int(self.gold_logprob.size)

    @classmethod
    def empty(cls) -> TokenSignals:
        return cls(
            gold_logprob=np.array([], dtype=np.float64),
            mu=np.array([], dtype=np.float64),
            sigma=np.array([], dtype=np.float64),
            argmax_correct=np.array([], dtype=bool),
        )


class SignalsCache:
    """In-memory extract-once cache for a single ``run_audit`` call.

    Key includes ``cache_tag`` so PEFT ``disable_adapter()`` (same ``id(model)``,
    different weights) does not reuse target signals as the reference.
    """

    def __init__(self) -> None:
        self._store: dict[tuple[Any, ...], TokenSignals] = {}

    def _key(
        self,
        model: Any,
        input_ids: Any,
        span: tuple[int, int] | None,
        skip_first: bool,
        cache_tag: str,
    ) -> tuple[Any, ...]:
        span_t = tuple(span) if span is not None else None
        return (
            id(model),
            str(cache_tag),
            tuple(int(x) for x in input_ids),
            span_t,
            bool(skip_first),
        )

    def get(
        self,
        model: Any,
        input_ids: Any,
        span: tuple[int, int] | None,
        skip_first: bool,
        cache_tag: str = "",
    ) -> TokenSignals | None:
        return self._store.get(self._key(model, input_ids, span, skip_first, cache_tag))

    def put(
        self,
        model: Any,
        input_ids: Any,
        span: tuple[int, int] | None,
        skip_first: bool,
        signals: TokenSignals,
        cache_tag: str = "",
    ) -> TokenSignals:
        self._store[self._key(model, input_ids, span, skip_first, cache_tag)] = signals
        return signals

    def __len__(self) -> int:
        return len(self._store)
