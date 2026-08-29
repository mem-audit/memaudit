"""PEFT configuration-semantics: naming, tying, token indices, quantization, guards.

Used by preflight and audit. Does not import peft at module load (FSDP hang risk);
peft is used only if the caller already loaded it, except for optional constant
lookup via ``sys.modules``.
"""

from __future__ import annotations

import sys
from collections.abc import Mapping, Sequence
from typing import Any

from memaudit.utils import encode_ids, infer_device, is_peft_model


# Mirror peft.utils.constants.EMBEDDING_LAYER_NAMES when peft is not imported.
_FALLBACK_CONVENTIONAL_NAMES = frozenset({"embed_tokens", "lm_head"})
_WRAPPER_NAME_SKIP = frozenset(
    {"modules_to_save", "original_module", "base_layer", "default", "lora_embedding_A", "lora_embedding_B"}
)
_QUANT_CLASS_MARKERS = (
    "Linear4bit",
    "Linear8bitLt",
    "Linear8bit",
    "Params4bit",
    "Int8Params",
    "HQQLinear",
    "WQLinear",
    "AffineQuantized",
)
_TESTED_PEFT_RANGE = ((0, 17), (0, 20))  # inclusive minor range on 0.x
BASE_EQUIV_ATOL_WARN = 1e-5
_PROBE_MAX_TOKENS = 16


def conventional_embedding_names() -> frozenset[str]:
    peft_mod = sys.modules.get("peft")
    if peft_mod is not None:
        const = getattr(peft_mod, "utils", None)
        names = None
        if const is not None:
            names = getattr(getattr(const, "constants", None), "EMBEDDING_LAYER_NAMES", None)
        if names is None:
            try:
                from peft.utils.constants import EMBEDDING_LAYER_NAMES  # type: ignore[import-not-found]

                names = EMBEDDING_LAYER_NAMES
            except Exception:
                names = None
        if names:
            return frozenset(str(n) for n in names)
    return _FALLBACK_CONVENTIONAL_NAMES


def wrapper_class_names(module: Any) -> set[str]:
    names: set[str] = set()
    if module is None:
        return names
    for cls in type(module).__mro__:
        names.add(cls.__name__)
    return names


def _safe_call(fn: Any, default: Any = None) -> Any:
    if not callable(fn):
        return default
    try:
        return fn()
    except Exception:
        return default


def _module_trainable(module: Any) -> bool:
    if module is None:
        return False
    try:
        return any(bool(p.requires_grad) for p in module.parameters())
    except Exception:
        return False


def _weight_tensor(module: Any) -> Any | None:
    if module is None:
        return None
    for attr in ("weight",):
        w = getattr(module, attr, None)
        if w is not None and hasattr(w, "data_ptr"):
            return w
    for attr in ("original_module", "base_layer", "modules_to_save"):
        inner = getattr(module, attr, None)
        if inner is None:
            continue
        if hasattr(inner, "values") and not hasattr(inner, "weight"):
            try:
                inner = next(iter(inner.values()))
            except Exception:
                continue
        w = _weight_tensor(inner)
        if w is not None:
            return w
    return None


def weight_data_ptr(module: Any) -> int | None:
    w = _weight_tensor(module)
    if w is None:
        return None
    try:
        return int(w.data_ptr())
    except Exception:
        return None


def _nonempty_adapter_map(obj: Any) -> bool:
    if obj is None:
        return False
    try:
        return len(obj) > 0
    except Exception:
        return False


def is_lora_embedding(module: Any) -> bool:
    """Harden lora.Embedding detection. LoraLayer mixins expose empty embedding maps on Linear."""
    if module is None:
        return False
    if _nonempty_adapter_map(getattr(module, "lora_embedding_A", None)) or _nonempty_adapter_map(
        getattr(module, "lora_embedding_B", None)
    ):
        return True
    cls = type(module)
    qual = f"{getattr(cls, '__module__', '')}.{cls.__name__}"
    if "lora" in qual.lower() and cls.__name__ == "Embedding":
        return True
    names = wrapper_class_names(module)
    if ("LoraLayer" in names or "BaseTunerLayer" in names) and cls.__name__ == "Embedding":
        return True
    return False


def is_lora_linear(module: Any) -> bool:
    if module is None or is_lora_embedding(module):
        return False
    if _nonempty_adapter_map(getattr(module, "lora_A", None)) or _nonempty_adapter_map(
        getattr(module, "lora_B", None)
    ):
        return True
    cls = type(module)
    qual = f"{getattr(cls, '__module__', '')}.{cls.__name__}"
    if "lora" in qual.lower() and cls.__name__ in {"Linear", "Conv1D", "ParamWrapper"}:
        return True
    names = wrapper_class_names(module)
    if ("LoraLayer" in names or "BaseTunerLayer" in names) and cls.__name__ in {
        "Linear",
        "Conv1D",
        "ParamWrapper",
    }:
        return True
    return False


def classify_module_mechanisms(module: Any) -> list[str]:
    if module is None:
        return []
    names = wrapper_class_names(module)
    out: list[str] = []
    if "ModulesToSaveWrapper" in names:
        out.append("ModulesToSaveWrapper")
    if "TrainableTokensWrapper" in names or "TrainableTokensLayer" in names:
        out.append("TrainableTokensWrapper")
    if is_lora_embedding(module):
        out.append("lora.Embedding")
    elif is_lora_linear(module):
        out.append("lora.Linear")
    if "ParamWrapper" in names:
        if "lora.ParamWrapper" not in out:
            out.append("lora.ParamWrapper")
    return out


def resolve_module_name(model: Any, module: Any) -> str | None:
    if module is None:
        return None
    named = getattr(model, "named_modules", None)
    if not callable(named):
        return None
    try:
        for name, mod in named():
            if mod is module:
                return str(name)
    except Exception:
        return None
    inner = getattr(module, "base_layer", None) or getattr(module, "original_module", None)
    if inner is not None and inner is not module:
        return resolve_module_name(model, inner)
    return None


def _last_path_component(name: str | None) -> str | None:
    if not name:
        return None
    parts = [p for p in str(name).split(".") if p and p not in _WRAPPER_NAME_SKIP]
    return parts[-1] if parts else None


def resolve_embedding_layer_names(model: Any, emb: Any, out: Any) -> dict[str, Any]:
    input_name = resolve_module_name(model, emb)
    output_name = resolve_module_name(model, out)
    conv = conventional_embedding_names()
    in_last = _last_path_component(input_name)
    out_last = _last_path_component(output_name)
    known_parts = [p for p in (in_last, out_last) if p]
    if not known_parts:
        conventional: bool | None = None
    else:
        conventional = all(p in conv for p in known_parts)
    return {
        "input": input_name,
        "output": output_name,
        "input_last": in_last,
        "output_last": out_last,
        "conventional": conventional,
    }


def normalize_trainable_token_indices(raw: Any) -> list[int] | None:
    if raw is None:
        return None
    ids: set[int] = set()
    if isinstance(raw, Mapping):
        for val in raw.values():
            if isinstance(val, (list, tuple, set)):
                ids.update(int(x) for x in val)
            elif isinstance(val, int):
                ids.add(int(val))
    elif isinstance(raw, (list, tuple, set)):
        ids.update(int(x) for x in raw)
    else:
        return None
    return sorted(ids)


def canary_token_intersection(
    secret_token_ids: Sequence[Any] | None,
    trained_indices: Sequence[int] | None,
) -> dict[str, Any]:
    secret = [int(x) for x in (secret_token_ids or [])]
    trained = set(int(x) for x in (trained_indices or []))
    overlap = sorted(set(secret) & trained)
    frozen = sorted(set(secret) - trained)
    return {
        "secret_token_ids": secret,
        "trained_indices": sorted(trained),
        "overlap": overlap,
        "frozen": frozen,
        "n_secret": len(secret),
        "n_overlap": len(overlap),
        "n_frozen": len(frozen),
        "all_frozen": bool(secret) and not overlap,
        "all_trained": bool(secret) and not frozen,
    }


def detect_quantization(model: Any) -> dict[str, Any]:
    kinds: list[str] = []
    if model is None:
        return {"quantized": False, "kinds": []}
    if getattr(model, "is_loaded_in_4bit", False):
        kinds.append("is_loaded_in_4bit")
    if getattr(model, "is_loaded_in_8bit", False):
        kinds.append("is_loaded_in_8bit")
    cfg = getattr(model, "config", None)
    qcfg = getattr(cfg, "quantization_config", None) if cfg is not None else None
    if qcfg is not None:
        kinds.append("quantization_config")
        qtype = getattr(qcfg, "quant_method", None) or getattr(qcfg, "load_in_4bit", None)
        if qtype is not None and str(qtype) not in kinds:
            kinds.append(str(qtype))
    named = getattr(model, "named_modules", None)
    if callable(named):
        try:
            for _n, mod in named():
                cls_name = type(mod).__name__
                mod_path = f"{getattr(type(mod), '__module__', '')}.{cls_name}"
                if cls_name in _QUANT_CLASS_MARKERS or any(m in cls_name for m in _QUANT_CLASS_MARKERS):
                    kinds.append(cls_name)
                elif "4bit" in cls_name.lower() or "8bit" in cls_name.lower():
                    kinds.append(cls_name)
                elif any(tag in mod_path.lower() for tag in ("bitsandbytes", "torchao", "hqq", "quanto")):
                    if cls_name not in {"Module", "Sequential"}:
                        kinds.append(cls_name)
        except Exception:
            pass
    # unique, stable
    seen: list[str] = []
    for k in kinds:
        if k not in seen:
            seen.append(k)
    return {"quantized": bool(seen), "kinds": seen}


def quantization_ref_mismatch(target_info: Mapping[str, Any], ref_model: Any) -> tuple[bool, str | None]:
    if not target_info.get("quantized"):
        return False, None
    if ref_model is None:
        return False, None
    ref_q = detect_quantization(ref_model)
    if ref_q.get("quantized"):
        return False, None
    msg = (
        "Target model is quantized but the separately loaded --ref is full-precision. "
        "disable_adapter() is exact vs the quantized base; a full-precision reference "
        "silently biases every base-calibrated score. Ref-mode is downgraded; pass an "
        "explicit matching reference or use --ref auto / disable_adapter."
    )
    return True, msg


def embedding_side_in_play(info: Mapping[str, Any]) -> bool:
    if info.get("tie_word_embeddings"):
        return True
    if info.get("modules_to_save"):
        return True
    if info.get("trainable_token_indices") is not None:
        return True
    targets = info.get("target_modules")
    if targets:
        conv = conventional_embedding_names()
        extra = {"wte", "wpe", "embed", "embedding", "lm_head"}
        names = targets if isinstance(targets, (list, tuple, set)) else [targets]
        for t in names:
            last = str(t).rsplit(".", 1)[-1]
            if last in conv or last in extra or "embed" in last.lower():
                return True
    mechs = set(info.get("input_mechanisms") or []) | set(info.get("output_mechanisms") or [])
    if mechs & {
        "ModulesToSaveWrapper",
        "TrainableTokensWrapper",
        "lora.Embedding",
        "lora.Linear",
    }:
        return True
    return False


def peft_version_string() -> str | None:
    try:
        from importlib.metadata import version

        return version("peft")
    except Exception:
        peft_mod = sys.modules.get("peft")
        if peft_mod is not None:
            return getattr(peft_mod, "__version__", None)
        return None


def peft_version_in_tested_range(ver: str | None) -> bool | None:
    if not ver:
        return None
    parts = ver.split(".")
    try:
        major = int(parts[0])
        minor = int(parts[1]) if len(parts) > 1 else 0
    except (TypeError, ValueError):
        return None
    lo, hi = _TESTED_PEFT_RANGE
    return (major, minor) >= lo and (major, minor) <= hi


def unusual_peft_triggers(emb_info: Mapping[str, Any]) -> list[str]:
    triggers: list[str] = []
    mechs = set(emb_info.get("mechanisms") or []) | set(emb_info.get("output_mechanisms") or [])
    if mechs & {"ModulesToSaveWrapper", "TrainableTokensWrapper"}:
        triggers.append("wrapper_mechanisms")
    if emb_info.get("tie_word_embeddings") and (
        emb_info.get("input_trainable")
        or bool(mechs & {"lora.Embedding", "ModulesToSaveWrapper", "TrainableTokensWrapper"})
    ):
        triggers.append("tied_with_embedding_adapter")
    names = emb_info.get("embedding_layer_names") or {}
    if names.get("conventional") is False:
        triggers.append("nonstandard_embedding_names")
    if emb_info.get("target_parameters"):
        triggers.append("target_parameters")
    if emb_info.get("quantized"):
        triggers.append("quantized_base")
    if emb_info.get("ensure_weight_tying"):
        triggers.append("ensure_weight_tying")
    ver = emb_info.get("peft_version")
    tested = peft_version_in_tested_range(ver if isinstance(ver, str) else None)
    if tested is False:
        triggers.append("untested_peft_version")
    return triggers


def _peft_cfg(model: Any) -> Any | None:
    cfgs = getattr(model, "peft_config", None)
    if not cfgs:
        return None
    if isinstance(cfgs, dict):
        adapter = getattr(model, "active_adapter", None)
        if adapter is not None and adapter in cfgs:
            return cfgs[adapter]
        if cfgs:
            return next(iter(cfgs.values()))
    return cfgs


def inspect_peft_embeddings(model: Any) -> dict[str, Any]:
    """Full embedding / adapter inspection. Public shape consumed by preflight + audit."""
    info: dict[str, Any] = {
        "trainable": False,
        "input_trainable": False,
        "output_trainable": False,
        "mechanisms": [],
        "input_mechanisms": [],
        "output_mechanisms": [],
        "tie_word_embeddings": None,
        "modules_to_save": None,
        "trainable_token_indices": None,
        "trainable_token_index_set": None,
        "ensure_weight_tying": None,
        "target_parameters": None,
        "target_modules": None,
        "bias": None,
        "peft_type": None,
        "r": None,
        "lora_alpha": None,
        "merged": None,
        "status_enabled": None,
        "embedding_layer_names": {"input": None, "output": None, "conventional": None},
        "weight_tying": None,
        "quantized": False,
        "quantization_kinds": [],
        "embedding_verification": None,
        "peft_version": peft_version_string(),
    }
    if model is None:
        return info
    cfg = getattr(model, "config", None)
    if cfg is not None:
        info["tie_word_embeddings"] = getattr(cfg, "tie_word_embeddings", None)

    peft_cfg = _peft_cfg(model)
    if peft_cfg is not None:
        info["bias"] = getattr(peft_cfg, "bias", None)
        info["peft_type"] = str(getattr(peft_cfg, "peft_type", None))
        info["r"] = getattr(peft_cfg, "r", None)
        info["lora_alpha"] = getattr(peft_cfg, "lora_alpha", None)
        info["modules_to_save"] = list(getattr(peft_cfg, "modules_to_save", None) or []) or None
        raw_tti = getattr(peft_cfg, "trainable_token_indices", None)
        info["trainable_token_indices"] = raw_tti
        info["trainable_token_index_set"] = normalize_trainable_token_indices(raw_tti)
        info["ensure_weight_tying"] = getattr(peft_cfg, "ensure_weight_tying", None)
        tparams = getattr(peft_cfg, "target_parameters", None)
        info["target_parameters"] = list(tparams) if tparams else None
        tmods = getattr(peft_cfg, "target_modules", None)
        if isinstance(tmods, str):
            info["target_modules"] = [tmods]
        elif tmods:
            info["target_modules"] = list(tmods)

    get_in = getattr(model, "get_input_embeddings", None)
    get_out = getattr(model, "get_output_embeddings", None)
    emb = _safe_call(get_in, None) if get_in is not None else None
    out = _safe_call(get_out, None) if get_out is not None else None
    input_missing = get_in is None or emb is None
    names = resolve_embedding_layer_names(model, emb, out)
    info["embedding_layer_names"] = {
        "input": names["input"],
        "output": names["output"],
        "conventional": names["conventional"],
    }

    in_mechs = classify_module_mechanisms(emb)
    out_mechs = classify_module_mechanisms(out)
    info["input_mechanisms"] = list(in_mechs)
    info["output_mechanisms"] = list(out_mechs)
    info["input_trainable"] = _module_trainable(emb)
    info["output_trainable"] = _module_trainable(out)
    info["trainable"] = bool(info["input_trainable"] or info["output_trainable"])

    in_ptr = weight_data_ptr(emb)
    out_ptr = weight_data_ptr(out)
    shared = None if in_ptr is None or out_ptr is None else bool(in_ptr == out_ptr)
    in_wrapped = bool(set(in_mechs) & {"ModulesToSaveWrapper", "TrainableTokensWrapper", "lora.Embedding"})
    out_wrapped = bool(set(out_mechs) & {"ModulesToSaveWrapper", "TrainableTokensWrapper", "lora.Linear", "lora.Embedding"})
    one_side = bool(in_wrapped) ^ bool(out_wrapped)
    tie_broken = bool(info.get("tie_word_embeddings") and one_side and shared is False)
    ewt = info.get("ensure_weight_tying")
    ewt_engaged: bool | None = None
    if ewt and info.get("tie_word_embeddings") and names["conventional"] is True:
        ewt_engaged = bool(in_wrapped and out_wrapped and shared)
    elif ewt and names["conventional"] is False:
        ewt_engaged = False
    info["weight_tying"] = {
        "input_data_ptr": in_ptr,
        "output_data_ptr": out_ptr,
        "shared": shared,
        "one_side_wrapped": one_side,
        "tie_broken_by_wrap": tie_broken,
        "ensure_weight_tying_engaged": ewt_engaged,
    }

    quant = detect_quantization(model)
    info["quantized"] = quant["quantized"]
    info["quantization_kinds"] = quant["kinds"]

    downgrade = False
    reasons: list[str] = []
    if input_missing:
        downgrade = True
        reasons.append("get_input_embeddings() returned None or raised; embedding trainability is unverified")
    if names["conventional"] is False and embedding_side_in_play(info):
        downgrade = True
        reasons.append(
            "PEFT tying/naming conventions not met "
            f"(input={names['input']!r}, output={names['output']!r}); "
            "behavior unverified upstream"
        )
    if info.get("target_parameters"):
        downgrade = True
        reasons.append(
            "target_parameters is experimental; adapter toggle is not treated as verified"
        )
    if ewt and names["conventional"] is False:
        downgrade = True
        reasons.append(
            "ensure_weight_tying is set but embedding/head names are outside "
            "embed_tokens/lm_head, so PEFT's name-gated tying may not have engaged"
        )

    combined: list[str] = []
    for m in in_mechs + out_mechs:
        if m not in combined:
            combined.append(m)
    if not in_mechs and not info["input_trainable"] and emb is not None:
        if downgrade:
            pass  # never emit plain_frozen_embedding on an unverified path
        else:
            combined.append("plain_frozen_embedding")
            if "plain_frozen_embedding" not in in_mechs:
                info["input_mechanisms"].append("plain_frozen_embedding")
    info["mechanisms"] = combined
    if downgrade:
        info["embedding_verification"] = "verification_unknown"
        info["embedding_verification_reasons"] = reasons
    else:
        info["embedding_verification"] = "ok"
        info["embedding_verification_reasons"] = []

    if is_peft_model(model):
        peft_mod = sys.modules.get("peft")
        get_status = getattr(peft_mod, "get_model_status", None) if peft_mod is not None else None
        if callable(get_status):
            try:
                status = get_status(model)
                info["merged"] = getattr(status, "merged", None)
                info["status_enabled"] = getattr(status, "enabled", None)
            except Exception:
                info["merged"] = getattr(model, "merged", None)
        else:
            info["merged"] = getattr(model, "merged", None)
    return info


def protocol_expects_trainable_tokens(canary: Mapping[str, Any]) -> bool:
    fam = str(canary.get("family") or canary.get("requested_family") or "").lower()
    return fam == "new_token"


def trainable_token_canary_findings(
    manifest: Mapping[str, Any],
    trained_indices: Sequence[int] | None,
) -> tuple[list[str], list[str], list[dict[str, Any]]]:
    """Return (fatal, warnings, per-canary coverage). Fatal when protocol expected trainable rows."""
    fatal: list[str] = []
    warnings: list[str] = []
    rows: list[dict[str, Any]] = []
    if trained_indices is None:
        return fatal, warnings, rows
    trained = list(trained_indices)
    for canary in manifest.get("canaries") or []:
        if not canary.get("included"):
            continue
        inter = canary_token_intersection(canary.get("secret_token_ids"), trained)
        rec = {"id": canary.get("id"), "family": canary.get("family"), **inter}
        rows.append(rec)
        if not inter["secret_token_ids"]:
            continue
        if inter["all_frozen"]:
            msg = (
                f"Canary {canary.get('id')} secret token ids {inter['secret_token_ids']} "
                f"are outside trainable_token_indices {trained}. Those embedding rows "
                "are frozen; a boolean trainable=True would have hidden this."
            )
            if protocol_expects_trainable_tokens(canary):
                fatal.append(
                    msg + " The canary protocol expected these tokens to be trainable."
                )
            else:
                warnings.append(msg)
        elif inter["n_frozen"]:
            warnings.append(
                f"Canary {canary.get('id')} has {inter['n_frozen']} secret token(s) "
                f"outside trainable_token_indices {trained} (frozen rows: {inter['frozen']})."
            )
    return fatal, warnings, rows


def _as_int_list(ids: Any) -> list[int]:
    if ids is None:
        return []
    if hasattr(ids, "tolist"):
        ids = ids.tolist()
    if ids and isinstance(ids[0], (list, tuple)):
        ids = ids[0]
    return [int(x) for x in ids]


def encode_probe_ids(tokenizer: Any, text: str, max_tokens: int = _PROBE_MAX_TOKENS) -> list[int]:
    ids = encode_ids(tokenizer, text, add_special_tokens=False)[: max(2, int(max_tokens))]
    if len(ids) < 2:
        ids = list(ids) + [0] * (2 - len(ids))
    return ids


def default_probe_texts(manifest: Mapping[str, Any] | None = None) -> list[str]:
    texts = ["hello world", "abc 123"]
    if manifest:
        for c in manifest.get("canaries") or []:
            prefix = c.get("prefix")
            if prefix:
                texts.append(str(prefix)[:64])
                break
            secret = c.get("secret")
            if secret:
                texts.append(str(secret)[:32])
                break
    # unique, cap at 4
    seen: list[str] = []
    for t in texts:
        if t and t not in seen:
            seen.append(t)
        if len(seen) >= 4:
            break
    return seen or ["ok"]


def _forward_logits(model: Any, ids: Sequence[int]) -> Any:
    import torch

    device = infer_device(model)
    tensor = torch.tensor([list(ids)], dtype=torch.long, device=device)
    out = model(input_ids=tensor)
    logits = out.logits if hasattr(out, "logits") else out[0]
    return logits.detach().float().cpu().contiguous()


def capture_disabled_logits(
    model: Any,
    tokenizer: Any,
    probe_texts: Sequence[str] | None = None,
    manifest: Mapping[str, Any] | None = None,
) -> dict[str, Any] | None:
    if model is None or tokenizer is None or not hasattr(model, "disable_adapter"):
        return None
    texts = list(probe_texts) if probe_texts else default_probe_texts(manifest)
    probes = [encode_probe_ids(tokenizer, t) for t in texts]
    try:
        import torch

        was_training = bool(getattr(model, "training", False))
        if hasattr(model, "eval"):
            model.eval()
        with torch.inference_mode():
            with model.disable_adapter():
                logits = [_forward_logits(model, p) for p in probes]
        if was_training and hasattr(model, "train"):
            model.train()
        return {
            "probe_texts": texts,
            "probe_ids": probes,
            "logits": [t.tolist() for t in logits],
        }
    except Exception:
        return None


def base_equivalence_guard(
    model: Any,
    tokenizer: Any,
    probe_texts: Sequence[str] | None = None,
    captured: Mapping[str, Any] | None = None,
    atol_warn: float = BASE_EQUIV_ATOL_WARN,
    manifest: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Cheap 2–4-forward disable_adapter fidelity check. See 02-peft-matrix.md §2."""
    import torch

    texts = list(probe_texts) if probe_texts else None
    if texts is None and captured is not None:
        texts = list(captured.get("probe_texts") or [])
    if not texts:
        texts = default_probe_texts(manifest)
    if captured is not None and captured.get("probe_ids"):
        probes = [list(p) for p in captured["probe_ids"]]
    else:
        probes = [encode_probe_ids(tokenizer, t) for t in texts]

    was_training = bool(getattr(model, "training", False))
    if hasattr(model, "eval"):
        model.eval()
    try:
        with torch.inference_mode():
            enabled_before = [_forward_logits(model, p) for p in probes]
            if hasattr(model, "disable_adapter"):
                with model.disable_adapter():
                    disabled = [_forward_logits(model, p) for p in probes]
            else:
                disabled = [t.clone() for t in enabled_before]
            enabled_after = [_forward_logits(model, p) for p in probes]
    finally:
        if was_training and hasattr(model, "train"):
            model.train()

    adapter_active = any(not torch.equal(e, d) for e, d in zip(enabled_before, disabled))
    restored = all(torch.equal(a, b) for a, b in zip(enabled_before, enabled_after))
    max_diff: float | None = None
    if captured is not None and captured.get("logits") is not None:
        diffs: list[float] = []
        for d, c in zip(disabled, captured["logits"]):
            ct = torch.as_tensor(c, dtype=d.dtype)
            if ct.shape != d.shape:
                diffs.append(float("inf"))
            else:
                diffs.append(float((d - ct).abs().max()))
        max_diff = max(diffs) if diffs else 0.0

    verdict = "pass"
    if max_diff is not None and max_diff > 0.0:
        verdict = "warn" if max_diff <= float(atol_warn) else "fail"
    if not restored:
        verdict = "fail"

    return {
        "adapter_active": bool(adapter_active),
        "restored": bool(restored),
        "max_abs_logit_diff": max_diff,
        "verdict": verdict,
        "atol_warn": float(atol_warn),
        "n_probes": len(probes),
        "probe_texts": texts,
    }
