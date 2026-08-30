"""Tokenizer helpers, hashing, dataset iteration, model unwrap."""

from __future__ import annotations

import hashlib
import json
import logging
import math
from collections.abc import Iterable, Iterator, Mapping, Sequence
from pathlib import Path
from typing import Any

from memaudit.exceptions import MemauditConfigError

logger = logging.getLogger("memaudit")


def encode_ids(tokenizer: Any, text: str, add_special_tokens: bool = False) -> list[int]:
    if hasattr(tokenizer, "encode"):
        try:
            ids = tokenizer.encode(text, add_special_tokens=add_special_tokens)
        except TypeError:
            ids = tokenizer.encode(text)
    else:
        out = tokenizer(text, add_special_tokens=add_special_tokens)
        ids = out["input_ids"]
    return _as_id_list(ids)


def decode_ids(
    tokenizer: Any,
    ids: Sequence[int],
    skip_special_tokens: bool = True,
) -> str:
    ids_list = [int(x) for x in ids]
    if hasattr(tokenizer, "decode"):
        try:
            return tokenizer.decode(
                ids_list,
                skip_special_tokens=skip_special_tokens,
                clean_up_tokenization_spaces=False,
            )
        except TypeError:
            try:
                return tokenizer.decode(ids_list, skip_special_tokens=skip_special_tokens)
            except TypeError:
                return tokenizer.decode(ids_list)
    return " ".join(str(i) for i in ids_list)


def roundtrip_tokens(tokenizer: Any, ids: Sequence[int]) -> tuple[str, list[int]]:
    """Decode then re-encode so the stored text is the scoring ground truth."""
    text = decode_ids(tokenizer, ids, skip_special_tokens=True)
    back = encode_ids(tokenizer, text, add_special_tokens=False)
    return text, back


def vocab_size_of(tokenizer: Any) -> int:
    vs = getattr(tokenizer, "vocab_size", None)
    if vs:
        return int(vs)
    get_vocab = getattr(tokenizer, "get_vocab", None)
    if callable(get_vocab):
        return len(get_vocab())
    try:
        return len(tokenizer)
    except TypeError as exc:
        raise MemauditConfigError("tokenizer has no vocab_size / get_vocab / __len__") from exc


def special_token_ids(tokenizer: Any) -> set[int]:
    special: set[int] = set()
    raw = getattr(tokenizer, "all_special_ids", None)
    if raw:
        special.update(int(x) for x in raw)
    for attr in ("pad_token_id", "eos_token_id", "bos_token_id", "unk_token_id"):
        val = getattr(tokenizer, attr, None)
        if val is not None:
            special.add(int(val))
    return special


def usable_token_ids(tokenizer: Any) -> list[int]:
    special = special_token_ids(tokenizer)
    size = vocab_size_of(tokenizer)
    ids = [i for i in range(size) if i not in special]
    if not ids:
        raise MemauditConfigError("tokenizer has no non-special token ids to sample from")
    return ids


def _as_id_list(ids: Any) -> list[int]:
    if hasattr(ids, "tolist"):
        ids = ids.tolist()
    if isinstance(ids, int):
        return [int(ids)]
    if ids and isinstance(ids[0], (list, tuple)):
        ids = ids[0]
    return [int(x) for x in ids]


def find_subsequence(haystack: Sequence[int], needle: Sequence[int]) -> tuple[int, int] | None:
    n = len(needle)
    if n == 0 or n > len(haystack):
        return None
    hay = list(haystack)
    nee = list(needle)
    for i in range(len(hay) - n + 1):
        if hay[i : i + n] == nee:
            return (i, i + n)
    return None


def canonical_json(obj: Any) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False, default=str)


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def sha256_json(obj: Any) -> str:
    return sha256_text(canonical_json(obj))


def short_hash(text: str, n: int = 16) -> str:
    return sha256_text(text)[:n]


def json_safe(obj: Any) -> Any:
    """Replace NaN/Inf with None so dumped JSON is RFC-compliant.

    A report containing bare ``NaN`` is not valid JSON. Other tools (and
    paying users' parsers) will reject it. None serializes as ``null``.
    """
    if isinstance(obj, float):
        if math.isnan(obj) or math.isinf(obj):
            return None
        return obj
    if hasattr(obj, "item") and callable(obj.item) and not isinstance(obj, (bytes, str)):
        try:
            return json_safe(obj.item())
        except Exception:
            pass
    if isinstance(obj, dict):
        return {str(k): json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [json_safe(v) for v in obj]
    return obj


def write_json(path: str | Path, obj: Any) -> Path:
    dest = Path(path)
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(
        json.dumps(json_safe(obj), indent=2, ensure_ascii=False, default=str, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return dest


def load_json(path: str | Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def first_example(dataset: Any) -> Mapping[str, Any]:
    if dataset is None:
        raise MemauditConfigError("dataset is empty / None")
    if hasattr(dataset, "__getitem__"):
        try:
            row = dataset[0]
            if isinstance(row, Mapping):
                return row
        except Exception:
            pass
    for row in iter_examples(dataset, limit=1):
        return row
    raise MemauditConfigError("dataset is empty")


def dataset_length(dataset: Any) -> int | None:
    try:
        return len(dataset)
    except Exception:
        return None


def iter_examples(dataset: Any, limit: int | None = None) -> Iterator[Mapping[str, Any]]:
    if dataset is None:
        return
    n = 0
    if hasattr(dataset, "__iter__") and not isinstance(dataset, (str, bytes, Mapping)):
        iterator: Iterable[Any] = dataset
    else:
        iterator = [dataset]
    for row in iterator:
        if not isinstance(row, Mapping):
            continue
        yield row
        n += 1
        if limit is not None and n >= limit:
            return


def example_text(example: Mapping[str, Any], fmt: str) -> str:
    if fmt == "text":
        return str(example.get("text") or "")
    if fmt in {"prompt_completion", "prompt+completion"}:
        return f"{example.get('prompt', '')}{example.get('completion', '')}"
    if fmt == "messages":
        parts = []
        for msg in example.get("messages") or []:
            if isinstance(msg, Mapping):
                parts.append(str(msg.get("content") or msg.get("value") or ""))
        return "\n".join(parts)
    for key in ("text", "completion", "content"):
        if key in example:
            return str(example[key])
    return json.dumps(dict(example), ensure_ascii=False, default=str)


def package_version() -> str:
    """In-tree ``TOOL_VERSION``. Dist metadata can lag an editable install."""
    from memaudit.constants import TOOL_VERSION

    return TOOL_VERSION


def sha256_file(path: str | Path, chunk_size: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        while True:
            chunk = fh.read(chunk_size)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def environment_versions() -> dict[str, Any]:
    """Interpreter + key library versions for the provenance block. Cheap."""
    import platform
    import sys
    from importlib.metadata import version as dist_version

    def _v(name: str) -> str | None:
        try:
            return dist_version(name)
        except Exception:
            return None

    return {
        "python": platform.python_version(),
        "python_implementation": platform.python_implementation(),
        "platform": platform.platform(),
        "executable": sys.executable,
        "memaudit": package_version(),
        "torch": _v("torch"),
        "transformers": _v("transformers"),
        "peft": _v("peft"),
        "trl": _v("trl"),
        "datasets": _v("datasets"),
        "numpy": _v("numpy"),
        "scipy": _v("scipy"),
    }


# Files above this size are recorded (name + size) but not hashed.
MAX_WEIGHT_HASH_BYTES = 2 * 1024**3


def dataset_fingerprint(dataset: Any, path: str | Path | None = None) -> dict[str, Any] | None:
    """Cheap dataset identity: row count + first/last record hashes (+ file info)."""
    if dataset is None:
        return None
    n = dataset_length(dataset)
    first_hash = last_hash = None
    if hasattr(dataset, "__getitem__") and n:
        try:
            first_row = dataset[0]
            last_row = dataset[n - 1]
            if isinstance(first_row, Mapping):
                first_hash = sha256_json(dict(first_row))
            if isinstance(last_row, Mapping):
                last_hash = sha256_json(dict(last_row))
        except Exception:
            pass
    if first_hash is None:
        for row in iter_examples(dataset, limit=1):
            first_hash = sha256_json(dict(row))
            break
    out: dict[str, Any] = {
        "n_rows": n,
        "first_record_sha256": first_hash,
        "last_record_sha256": last_hash,
        "file": None,
        "note": "cheap fingerprint: row count + first/last record hashes; not a full-content hash",
    }
    if path is not None:
        p = Path(path)
        if p.is_file():
            size = p.stat().st_size
            file_info: dict[str, Any] = {"path": str(p), "size_bytes": int(size)}
            if size <= 256 * 1024**2:
                try:
                    file_info["sha256"] = sha256_file(p)
                except Exception:
                    file_info["sha256"] = None
            else:
                file_info["sha256"] = None
                file_info["note"] = "file larger than 256 MiB; size recorded, content not hashed"
            out["file"] = file_info
    return out


_WEIGHT_FILE_PATTERNS = ("*.safetensors", "*.bin", "*.pt", "*.gguf")


def model_fingerprint(model: Any, model_path: str | Path | None = None) -> dict[str, Any]:
    """Model / adapter identity: config hash, parameter count, weight-file hashes.

    Weight files are hashed only when a local directory is known (CLI ``--model``
    or a resolvable local ``name_or_path``) and each file is under the size cap.
    In-callback models live in memory: config hash + parameter count only.
    """
    out: dict[str, Any] = dict(model_identity(model))
    cfg = getattr(model, "config", None)
    if cfg is not None:
        try:
            cfg_dict = cfg.to_dict() if hasattr(cfg, "to_dict") else dict(vars(cfg))
            out["config_sha256"] = sha256_json(cfg_dict)
        except Exception:
            out["config_sha256"] = None
    else:
        out["config_sha256"] = None
    peft_cfgs = getattr(model, "peft_config", None)
    if peft_cfgs:
        try:
            if isinstance(peft_cfgs, dict):
                payload = {k: v.to_dict() if hasattr(v, "to_dict") else str(v) for k, v in peft_cfgs.items()}
            else:
                payload = peft_cfgs.to_dict() if hasattr(peft_cfgs, "to_dict") else str(peft_cfgs)
            out["adapter_config_sha256"] = sha256_json(payload)
        except Exception:
            out["adapter_config_sha256"] = None
    try:
        out["n_parameters"] = int(sum(p.numel() for p in model.parameters()))
    except Exception:
        out["n_parameters"] = None

    weight_dir: Path | None = None
    if model_path is not None and Path(model_path).is_dir():
        weight_dir = Path(model_path)
    else:
        name = out.get("name_or_path")
        if name and Path(str(name)).is_dir():
            weight_dir = Path(str(name))
    if weight_dir is None:
        out["weight_files"] = None
        out["weight_files_note"] = (
            "in-memory model (no local weight directory known); config hash and "
            "parameter count recorded instead"
        )
        return out
    files: list[dict[str, Any]] = []
    for pattern in _WEIGHT_FILE_PATTERNS:
        for f in sorted(weight_dir.glob(pattern)):
            size = f.stat().st_size
            entry: dict[str, Any] = {"name": f.name, "size_bytes": int(size)}
            if size <= MAX_WEIGHT_HASH_BYTES:
                try:
                    entry["sha256"] = sha256_file(f)
                except Exception:
                    entry["sha256"] = None
            else:
                entry["sha256"] = None
                entry["note"] = "over 2 GiB; size recorded, content not hashed"
            files.append(entry)
    out["weight_files"] = files or None
    if not files:
        out["weight_files_note"] = f"no weight files matching {_WEIGHT_FILE_PATTERNS} in {weight_dir}"
    else:
        out["weight_files_dir"] = str(weight_dir)
    return out


def infer_device(model: Any) -> Any:
    import torch

    try:
        return next(model.parameters()).device
    except (StopIteration, AttributeError):
        return torch.device("cpu")


def unwrap_model(model: Any, trainer: Any | None = None) -> Any:
    if trainer is not None:
        accelerator = getattr(trainer, "accelerator", None)
        if accelerator is not None and hasattr(accelerator, "unwrap_model"):
            try:
                return accelerator.unwrap_model(model)
            except Exception:
                pass
    inner = getattr(model, "module", None)
    if inner is not None and inner is not model:
        return inner
    return model


def is_peft_model(model: Any) -> bool:
    if model is None:
        return False
    cls = type(model)
    name = f"{cls.__module__}.{cls.__qualname__}"
    if "peft" in name.lower() or "PeftModel" in cls.__name__:
        return True
    return hasattr(model, "peft_config") and hasattr(model, "disable_adapter")


def model_identity(model: Any) -> dict[str, Any]:
    cfg = getattr(model, "config", None)
    name = None
    if cfg is not None:
        name = getattr(cfg, "_name_or_path", None) or getattr(cfg, "name_or_path", None)
    return {
        "class": type(model).__name__,
        "name_or_path": name,
        "peft": is_peft_model(model),
    }
