"""Dataclasses shared across generation, injection, and reporting."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

from memaudit.exceptions import MemauditConfigError


@dataclass
class Canary:
    """A generated canary secret. Inclusion coins are applied in ``inject()``."""

    id: str
    family: str
    secret: str
    secret_token_ids: list[int]
    prefix: str
    prefix_token_ids: list[int]
    secret_span: list[int]
    repetitions: int
    role: str = "candidate"
    generation_notes: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Canary:
        known = {f.name for f in cls.__dataclass_fields__.values()}  # type: ignore[attr-defined]
        payload = {k: v for k, v in data.items() if k in known}
        payload.setdefault("secret_token_ids", list(data.get("secret_token_ids") or []))
        payload.setdefault("prefix_token_ids", list(data.get("prefix_token_ids") or []))
        payload.setdefault("secret_span", list(data.get("secret_span") or [0, 0]))
        payload.setdefault("repetitions", int(data.get("repetitions") or 1))
        payload.setdefault("role", data.get("role") or "candidate")
        payload.setdefault("family", data.get("family") or "random")
        payload.setdefault("id", data.get("id") or "unknown")
        payload.setdefault("secret", data.get("secret") or "")
        payload.setdefault("prefix", data.get("prefix") or "")
        return cls(**payload)


def canaries_from_obj(obj: Any) -> list[Canary]:
    if isinstance(obj, list):
        return [c if isinstance(c, Canary) else Canary.from_dict(c) for c in obj]
    if isinstance(obj, dict) and "canaries" in obj:
        return canaries_from_obj(obj["canaries"])
    raise MemauditConfigError(
        "expected a list of Canary / dicts, or an inject() manifest with a 'canaries' key"
    )
