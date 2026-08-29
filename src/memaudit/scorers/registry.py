"""Resolve a membership scorer from a name, import path, or instance.

Adding a future backend (e.g. EZ-MIA) should be one new file implementing
``MembershipScorer`` plus selecting it via ``--scorer package.module:Class``.
A one-line alias here is optional convenience, not required.

EZ-MIA is **not shipped**. Required signals already extracted: target/reference
``gold_logprob`` and target ``argmax_correct``. See ``docs/membership-scorers.md``.
"""

from __future__ import annotations

import importlib
from typing import Any

from memaudit.constants import DEFAULT_MIN_K_PCT
from memaudit.exceptions import MemauditConfigError
from memaudit.scorers.min_k import DEFAULT_SCORER_NAME, MinKPlusPlusScorer

# Built-in names only. Future attacks register here (one line) or are loaded
# by import path so orchestration in audit.py does not change.
_BUILTIN: dict[str, type] = {
    DEFAULT_SCORER_NAME: MinKPlusPlusScorer,
    "base_calibrated_min_k_plus_plus": MinKPlusPlusScorer,
}


def scorer_provenance(scorer: Any) -> dict[str, Any]:
    """``membership.scorer`` block: name + version (+ seam metadata)."""
    return {
        "name": str(getattr(scorer, "name", "unknown")),
        "version": str(getattr(scorer, "version", "unknown")),
        "requires_reference": bool(getattr(scorer, "requires_reference", False)),
        "forward_passes_per_record": int(getattr(scorer, "forward_passes_per_record", 1)),
    }


def resolve_scorer(
    spec: str | Any | None,
    *,
    min_k_pct: float = DEFAULT_MIN_K_PCT,
) -> Any:
    """Return a scorer instance.

    * ``None`` / ``\"\"`` / ``min_k_plus_plus`` → default Min-K%++
    * built-in alias ``base_calibrated_min_k_plus_plus`` → same class
    * ``package.module:ClassName`` → import and instantiate
    * an object with ``score`` / ``name`` → used as-is
    """
    if spec is None or spec == "":
        return MinKPlusPlusScorer(min_k_pct=min_k_pct)
    if not isinstance(spec, str):
        if not callable(getattr(spec, "score", None)):
            raise MemauditConfigError(
                "membership scorer instance must implement score(target, reference)"
            )
        return spec
    key = spec.strip()
    if key in _BUILTIN:
        return _instantiate(_BUILTIN[key], min_k_pct)
    if ":" in key or "." in key:
        return _load_import_path(key, min_k_pct)
    known = ", ".join(sorted(_BUILTIN))
    raise MemauditConfigError(
        f"Unknown membership scorer {key!r}. Built-ins: {known}. "
        "Or pass an import path 'package.module:ClassName'."
    )


def _instantiate(cls: type, min_k_pct: float) -> Any:
    try:
        return cls(min_k_pct=min_k_pct)
    except TypeError:
        return cls()


def _load_import_path(path: str, min_k_pct: float) -> Any:
    if ":" in path:
        mod_name, cls_name = path.rsplit(":", 1)
    else:
        mod_name, cls_name = path.rsplit(".", 1)
    try:
        module = importlib.import_module(mod_name)
    except ImportError as exc:
        raise MemauditConfigError(
            f"could not import membership scorer module {mod_name!r}: {exc}"
        ) from exc
    try:
        cls = getattr(module, cls_name)
    except AttributeError as exc:
        raise MemauditConfigError(
            f"module {mod_name!r} has no scorer {cls_name!r}"
        ) from exc
    if not callable(cls):
        raise MemauditConfigError(f"{path} is not a callable scorer class")
    return _instantiate(cls, min_k_pct)
