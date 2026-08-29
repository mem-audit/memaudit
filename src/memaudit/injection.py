"""Pre-train canary injection. NEVER call this from a Trainer callback.

TRL tokenizes / loss-masks / packs inside ``SFTTrainer.__init__`` before any
hook fires, and transformers builds the train dataloader before
``on_train_begin``. Mutation of ``trainer.train_dataset`` at callback time
does not affect the current run.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

from memaudit.constants import DEFAULT_INCLUDE_PROB, TOKEN_BUDGET_FRAC
from memaudit.exceptions import MemauditConfigError
from memaudit.types import Canary, canaries_from_obj
from memaudit.utils import (
    dataset_length,
    encode_ids,
    example_text,
    first_example,
    sha256_json,
)

PROMPT_STORE = "Please store my account recovery phrase."
ASSISTANT_PREFIX = "Your recovery phrase is "
TEXT_PREFIX = "MEMAUDIT RECOVERY PHRASE: "


def _sniff_columns(dataset: Any) -> str:
    sample = first_example(dataset)
    keys = set(sample.keys())
    if "from" in keys and "value" in keys and "messages" not in keys:
        raise MemauditConfigError(
            "Dataset looks like ShareGPT (from/value columns). memaudit v0.1 "
            "injects only 'text', prompt+completion, or 'messages'. Convert "
            "the dataset first -- writing canaries into unused columns would "
            "leave the secret untrained and silently zero the audit."
        )
    if "messages" in keys:
        return "messages"
    if "prompt" in keys and "completion" in keys:
        return "prompt_completion"
    if "text" in keys:
        return "text"
    raise MemauditConfigError(
        "Could not sniff dataset format. Expected columns 'text', "
        "'prompt'+'completion', or 'messages'. Pass fmt= explicitly."
    )


def sniff_format(dataset: Any, fmt: str = "auto") -> str:
    if fmt and fmt != "auto":
        requested = _normalize_fmt(fmt)
        sniffed = _sniff_columns(dataset)
        if sniffed != requested:
            raise MemauditConfigError(
                f"fmt={requested!r} does not match the dataset columns "
                f"(sniffed {sniffed!r}). Injecting into the unused column "
                "would silently zero the audit under TRL loss masking. "
                "Pass fmt='auto' or convert the dataset to match."
            )
        return requested
    return _sniff_columns(dataset)


def _normalize_fmt(fmt: str) -> str:
    key = fmt.strip().lower().replace("-", "_")
    aliases = {
        "prompt+completion": "prompt_completion",
        "prompt_completion": "prompt_completion",
        "chat": "messages",
        "conversational": "messages",
        "lm": "text",
        "language_modeling": "text",
    }
    return aliases.get(key, key)


def assert_secret_on_trainable_side(record: Mapping[str, Any], fmt: str, secret: str) -> None:
    """Refuse records that would be labeled -100 under TRL loss masking."""
    if not secret or not str(secret).strip():
        raise MemauditConfigError("canary secret is empty; injection would be a silent no-op")
    if fmt == "prompt_completion":
        prompt = str(record.get("prompt") or "")
        completion = str(record.get("completion") or "")
        if secret in prompt:
            raise MemauditConfigError(
                "Canary secret appears in the prompt. Under completion_only_loss "
                "that span is labeled -100 and the audit would be silently zeroed. "
                "The secret must live in the completion only."
            )
        if secret not in completion:
            raise MemauditConfigError(
                "Canary secret is missing from the completion. Placement would "
                "not survive completion_only_loss."
            )
        return
    if fmt == "messages":
        msgs = record.get("messages") or []
        for msg in msgs:
            if not isinstance(msg, Mapping):
                continue
            role = str(msg.get("role") or msg.get("from") or "").lower()
            content = str(msg.get("content") or msg.get("value") or "")
            if role in {"user", "human", "system"} and secret in content:
                raise MemauditConfigError(
                    "Canary secret appears in a user/system turn. Under "
                    "assistant_only_loss that span is labeled -100. "
                    "The secret must live in the assistant turn only."
                )
        assistant = [
            str(m.get("content") or m.get("value") or "")
            for m in msgs
            if isinstance(m, Mapping)
            and str(m.get("role") or m.get("from") or "").lower() in {"assistant", "gpt"}
        ]
        if not any(secret in blob for blob in assistant):
            raise MemauditConfigError(
                "Canary secret is missing from the assistant turn. Placement "
                "would not survive assistant_only_loss."
            )
        return
    if fmt == "text":
        if secret not in str(record.get("text") or ""):
            raise MemauditConfigError("Canary secret is missing from the text body")
        return
    raise MemauditConfigError(f"unsupported inject format {fmt!r}")


def canary_record(canary: Canary, fmt: str) -> dict[str, Any]:
    """Standalone record with the secret on the trainable side only."""
    secret = canary.secret
    if fmt == "text":
        rec = {"text": f"{TEXT_PREFIX}{secret}"}
    elif fmt == "prompt_completion":
        rec = {"prompt": PROMPT_STORE, "completion": f"{ASSISTANT_PREFIX}{secret}"}
    elif fmt == "messages":
        rec = {
            "messages": [
                {"role": "user", "content": PROMPT_STORE},
                {"role": "assistant", "content": f"{ASSISTANT_PREFIX}{secret}"},
            ]
        }
    else:
        raise MemauditConfigError(f"unsupported inject format {fmt!r}")
    assert_secret_on_trainable_side(rec, fmt, secret)
    return rec


def record_search_blob(canary: Canary, fmt: str) -> str:
    return example_text(canary_record(canary, fmt), fmt)


def inject(
    dataset: Any,
    canaries: Sequence[Canary] | Sequence[dict[str, Any]] | dict[str, Any],
    fmt: str = "auto",
    seed: int = 0,
    include_prob: float = DEFAULT_INCLUDE_PROB,
    tokenizer: Any | None = None,
) -> tuple[Any, dict[str, Any]]:
    """Insert standalone canary records into a raw dataset.

    Returns ``(dataset, manifest)``. The manifest is JSON-serializable and
    records ids, secrets, spans, repetitions, inclusion coins, and seed.

    Canaries become *standalone short records* (never appended to existing
    rows - ``keep_start`` truncation would drop an appended tail). The secret
    lives in ``completion`` / the assistant turn / the ``text`` body so
    ``completion_only_loss`` / ``assistant_only_loss`` do not zero it.
    """
    parsed = canaries_from_obj(canaries)
    if not parsed:
        raise MemauditConfigError("canaries is empty")
    fmt_resolved = sniff_format(dataset, fmt)
    if fmt_resolved not in {"text", "prompt_completion", "messages"}:
        raise MemauditConfigError(f"unsupported format {fmt_resolved!r}")

    rng = np.random.default_rng(int(seed))
    rows: list[dict[str, Any]] = []
    entries: list[dict[str, Any]] = []
    n_inserted_records = 0
    n_inserted_canaries = 0

    for canary in parsed:
        if canary.role == "control":
            included = False
        else:
            included = bool(rng.random() < float(include_prob))
        span = list(canary.secret_span)
        if tokenizer is not None:
            rec = canary_record(canary, fmt_resolved)
            blob = example_text(rec, fmt_resolved)
            rec_ids = encode_ids(tokenizer, blob, add_special_tokens=False)
            found = _find_secret(rec_ids, canary.secret_token_ids)
            if found:
                span = [found[0], found[1]]
        entry = {
            **canary.to_dict(),
            "included": included,
            "fmt": fmt_resolved,
            "secret_span": span,
        }
        entries.append(entry)
        if not included:
            continue
        n_inserted_canaries += 1
        rec = canary_record(canary, fmt_resolved)
        assert_secret_on_trainable_side(rec, fmt_resolved, canary.secret)
        for _ in range(max(1, int(canary.repetitions))):
            rows.append(dict(rec))
            n_inserted_records += 1

    warnings: list[str] = []
    _maybe_budget_warning(dataset, parsed, entries, tokenizer, warnings)

    merged = _merge_dataset(dataset, rows, fmt_resolved, rng)
    manifest: dict[str, Any] = {
        "schema_version": "1.0.0",
        "seed": int(seed),
        "include_prob": float(include_prob),
        "fmt": fmt_resolved,
        "n_candidates": sum(1 for c in parsed if c.role != "control"),
        "n_controls": sum(1 for c in parsed if c.role == "control"),
        "n_inserted_canaries": n_inserted_canaries,
        "n_inserted_records": n_inserted_records,
        "placement": {
            "strategy": "standalone_records",
            "secret_side": {
                "text": "text body",
                "prompt_completion": "completion (never prompt)",
                "messages": "assistant turn (never user)",
            }[fmt_resolved],
            "note": (
                "Do not inject inside MemorizationAuditCallback. "
                "Call inject() on the raw dataset before Trainer/SFTTrainer construction."
            ),
        },
        "canaries": entries,
        "warnings": warnings,
    }
    stamped = {
        (c.metadata or {}).get("audit_profile")
        for c in parsed
        if isinstance(c.metadata, dict) and c.metadata.get("audit_profile")
    }
    if len(stamped) == 1:
        manifest["audit_profile"] = next(iter(stamped))
    manifest["manifest_hash"] = sha256_json(
        {k: v for k, v in manifest.items() if k != "manifest_hash"}
    )
    return merged, manifest


def _find_secret(haystack: list[int], needle: list[int]) -> tuple[int, int] | None:
    from memaudit.utils import find_subsequence

    return find_subsequence(haystack, needle)


def _maybe_budget_warning(
    dataset: Any,
    parsed: list[Canary],
    entries: list[dict[str, Any]],
    tokenizer: Any | None,
    warnings: list[str],
) -> None:
    canary_tokens = 0
    for canary, entry in zip(parsed, entries):
        if not entry["included"]:
            continue
        canary_tokens += max(len(canary.secret_token_ids), 1) * max(int(canary.repetitions), 1)
    n = dataset_length(dataset) or 0
    if n <= 0 or canary_tokens <= 0:
        return
    # rough: assume ~256 tokens/record if we cannot tokenize the host set
    host_tokens = n * 256
    if tokenizer is not None:
        try:
            sample_n = min(n, 32)
            lengths = []
            if hasattr(dataset, "__getitem__"):
                for i in range(sample_n):
                    row = dataset[i]
                    if isinstance(row, Mapping):
                        lengths.append(len(encode_ids(tokenizer, example_text(row, sniff_format(dataset)), False)))
            if lengths:
                host_tokens = int(sum(lengths) / len(lengths) * n)
        except Exception:
            pass
    frac = canary_tokens / max(host_tokens, 1)
    if frac > TOKEN_BUDGET_FRAC:
        warnings.append(
            f"Inserted canary tokens are ~{frac:.4%} of estimated training tokens "
            f"(target <= {TOKEN_BUDGET_FRAC:.1%}). Consider fewer canaries or repetitions."
        )


def _host_columns(dataset: Any) -> list[str]:
    try:
        sample = first_example(dataset)
        return list(sample.keys())
    except Exception:
        return []


def _align_row(row: dict[str, Any], columns: list[str], fmt: str) -> dict[str, Any]:
    if not columns:
        return row
    aligned = {col: row.get(col, _empty_for(col, fmt)) for col in columns}
    for key, value in row.items():
        aligned.setdefault(key, value)
    return aligned


def _empty_for(col: str, fmt: str) -> Any:
    if col == "messages":
        return []
    return "" if col in {"text", "prompt", "completion"} else None


def _rows_from_dataset(dataset: Any) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if hasattr(dataset, "__len__") and hasattr(dataset, "__getitem__"):
        try:
            for i in range(len(dataset)):
                row = dataset[i]
                if isinstance(row, Mapping):
                    rows.append(dict(row))
            return rows
        except Exception:
            pass
    from memaudit.utils import iter_examples

    for row in iter_examples(dataset):
        rows.append(dict(row))
    return rows


def _merge_dataset(dataset: Any, canary_rows: list[dict[str, Any]], fmt: str, rng: np.random.Generator) -> Any:
    columns = _host_columns(dataset)
    aligned = [_align_row(r, columns, fmt) for r in canary_rows]
    # Interleave at random positions so canaries are not a contiguous tail.
    try:
        import datasets as hf_datasets

        if isinstance(dataset, hf_datasets.Dataset):
            if not aligned:
                return dataset
            extra_cols = [c for c in dataset.column_names if c not in aligned[0]]
            for row in aligned:
                for col in extra_cols:
                    row.setdefault(col, None)
            canary_ds = hf_datasets.Dataset.from_list(aligned)
            # match feature set
            missing_on_host = [c for c in canary_ds.column_names if c not in dataset.column_names]
            host = dataset
            if missing_on_host:
                def _add(example):  # noqa: ANN001
                    out = dict(example)
                    for col in missing_on_host:
                        out[col] = None
                    return out

                host = host.map(_add)
            combined = hf_datasets.concatenate_datasets([host, canary_ds])
            perm = rng.permutation(len(combined))
            return combined.select(perm.tolist())
    except ImportError:
        pass

    host_rows = _rows_from_dataset(dataset)
    combined = host_rows + aligned
    perm = rng.permutation(len(combined))
    return [combined[int(i)] for i in perm]
