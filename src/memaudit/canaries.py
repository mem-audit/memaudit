"""Pure canary generation. No Hugging Face Trainer dependency.

Default family is high-perplexity regular-token sequences (existing vocab only).
The new-token family is gated and unimplemented - memaudit never resizes vocab.
"""

from __future__ import annotations

import logging
import math
import string
import warnings
from collections import Counter
from collections.abc import Sequence
from typing import Any

import numpy as np

from memaudit.constants import (
    DEFAULT_FAMILY,
    DEFAULT_N,
    DEFAULT_N_CONTROLS,
    DEFAULT_REPETITIONS,
    DEFAULT_SECRET_LEN,
    FAMILY_ALIASES,
    IMPLEMENTED_FAMILIES,
    MAX_SECRET_LEN,
    MIN_CONTROLS_FOR_TPR_AT_1PCT,
    MIN_SECRET_LEN,
    NEW_TOKEN_MESSAGE,
    get_audit_profile,
)
from memaudit.exceptions import MemauditConfigError
from memaudit.types import Canary
from memaudit.utils import (
    decode_ids,
    encode_ids,
    logger,
    roundtrip_tokens,
    usable_token_ids,
)

_ALPHANUM = string.ascii_uppercase + string.digits
_LOG = logging.getLogger(__name__)
# Batched rejection sampling keeps the 16-candidate budget but cuts forwards
# from O(candidates * secret_len) sequential passes to O(rounds * secret_len).
_HIGH_PPL_CANDIDATE_BATCH = 8
_HIGH_PPL_ROUNDS = 2


def normalize_family(family: str) -> str:
    key = (family or DEFAULT_FAMILY).strip().lower()
    return FAMILY_ALIASES.get(key, key)


def requested_family_of(canary: Any) -> str:
    if isinstance(canary, Canary):
        return canary.requested_family or canary.family
    requested = canary.get("requested_family") if isinstance(canary, dict) else None
    return str(requested or (canary.get("family") if isinstance(canary, dict) else "") or "")


def actual_generator_of(canary: Any) -> str:
    if isinstance(canary, Canary):
        return canary.actual_generator or str((canary.metadata or {}).get("source") or "")
    if not isinstance(canary, dict):
        return ""
    if canary.get("actual_generator"):
        return str(canary["actual_generator"])
    meta = canary.get("metadata") or {}
    if isinstance(meta, dict) and meta.get("source"):
        return str(meta["source"])
    return ""


def generate_canaries(
    tokenizer: Any,
    n: int | None = None,
    n_controls: int | None = None,
    family: str = DEFAULT_FAMILY,
    repetitions: Sequence[int] | None = None,
    seed: int = 0,
    *,
    profile: str | None = None,
    model: Any | None = None,
    secret_len: int | None = None,
    corpus: Sequence[str] | None = None,
    ppl_band: tuple[float, float] = (40.0, 500.0),
    sample_temperature: float = 1.8,
) -> list[Canary]:
    """Generate ``n`` insert-eligible canaries plus ``n_controls`` never-inserted twins.

    Parameters
    ----------
    tokenizer
        Any object with ``encode`` / ``decode`` / ``vocab_size`` (HF tokenizers work).
    n
        Insert-eligible canaries. ``inject()`` applies Bernoulli inclusion coins.
    n_controls
        Dedicated held-out negative controls (never inserted).
    family
        ``high_ppl`` (default), ``unigram``, ``bigram``, ``structured``, ``random``.
        ``new_token`` raises - gated / unimplemented in v0.1.
    profile
        Optional named audit profile (``smoke`` / ``routine`` / ``powered``).
        Fills n, n_controls, repetitions, and secret_len when those arguments
        are omitted. Explicit counts always win.
    model
        Optional causal LM used only by ``high_ppl`` rejection sampling. If omitted
        and a corpus is supplied, ``high_ppl`` falls back to rare-token unigram
        draws; if omitted and there is no corpus, the draw is uniform-from-vocab
        (recorded as ``actual_generator`` / ``metadata.source``, not as
        model-scored high-perplexity).
    corpus
        Optional raw strings for unigram/bigram frequency tables.
    """
    profile_spec = None
    if profile:
        try:
            profile_spec = get_audit_profile(profile)
        except KeyError as exc:
            raise MemauditConfigError(
                f"Unknown audit profile {profile!r}. "
                "Use smoke, routine, or powered."
            ) from exc
        if n is None:
            n = int(profile_spec["n"])
        if n_controls is None:
            n_controls = int(profile_spec["n_controls"])
        if repetitions is None:
            repetitions = tuple(profile_spec["repetitions"])
        if secret_len is None:
            secret_len = int(profile_spec["secret_len"])
    n = DEFAULT_N if n is None else int(n)
    n_controls = DEFAULT_N_CONTROLS if n_controls is None else int(n_controls)
    if repetitions is None:
        repetitions = DEFAULT_REPETITIONS
    if secret_len is None:
        secret_len = DEFAULT_SECRET_LEN

    family_norm = normalize_family(family)
    if family_norm == "new_token":
        raise MemauditConfigError(NEW_TOKEN_MESSAGE)
    if family_norm not in IMPLEMENTED_FAMILIES:
        raise MemauditConfigError(
            f"Unknown canary family {family!r}. Implemented: {sorted(IMPLEMENTED_FAMILIES)}"
        )
    if n < 0 or n_controls < 0:
        raise MemauditConfigError("n and n_controls must be >= 0")
    if n + n_controls == 0:
        raise MemauditConfigError("need at least one canary or control")
    if n_controls < MIN_CONTROLS_FOR_TPR_AT_1PCT and not (
        profile_spec and profile_spec.get("refuse_headline")
    ):
        warnings.warn(
            f"n_controls={n_controls} is below the {MIN_CONTROLS_FOR_TPR_AT_1PCT} "
            "floor required to identify TPR@1% FPR. The report will refuse that "
            "headline rather than invent a precise detection rate. Pass "
            f"n_controls>={MIN_CONTROLS_FOR_TPR_AT_1PCT} for a valid headline.",
            UserWarning,
            stacklevel=2,
        )
    secret_len = int(secret_len)
    if secret_len < MIN_SECRET_LEN or secret_len > MAX_SECRET_LEN:
        raise MemauditConfigError(
            f"secret_len must be in [{MIN_SECRET_LEN}, {MAX_SECRET_LEN}] (got {secret_len})"
        )
    reps = tuple(int(r) for r in repetitions) or DEFAULT_REPETITIONS
    if any(r < 1 for r in reps):
        raise MemauditConfigError("repetitions must be positive integers")

    rng = np.random.default_rng(int(seed))
    usable = usable_token_ids(tokenizer)
    freq = _corpus_unigrams(tokenizer, corpus) if corpus else None
    bigrams = _corpus_bigrams(tokenizer, corpus) if corpus and family_norm == "bigram" else None

    fallback_note = None
    if family_norm == "high_ppl" and model is None:
        if freq is None:
            fallback_note = (
                "high_ppl: no model provided; fell back to uniform-from-vocab "
                "draws from the tokenizer vocabulary (no corpus frequencies; "
                "this is not rare-token unigram). Existing tokens only; no vocab resize"
            )
        else:
            fallback_note = (
                "high_ppl: no model provided; fell back to rare-token unigram "
                "construction from the supplied corpus (existing tokens only; no vocab resize)"
            )
    if family_norm in {"unigram", "bigram"} and freq is None:
        extra = (
            f"{family_norm}: no corpus provided; sampling uniformly from non-special vocab"
        )
        fallback_note = f"{fallback_note}; {extra}" if fallback_note else extra

    canaries: list[Canary] = []
    seen: list[set[int]] = []
    total = n + n_controls
    log_high_ppl = family_norm == "high_ppl" and model is not None
    for i in range(total):
        if log_high_ppl and (i == 0 or (i + 1) % 25 == 0 or i + 1 == total):
            _LOG.info("high_ppl model-scored canaries: %d/%d", i + 1, total)
        role = "candidate" if i < n else "control"
        rep = int(reps[i % len(reps)]) if role == "candidate" else 1
        ids, notes, meta = _draw_secret(
            family=family_norm,
            tokenizer=tokenizer,
            usable=usable,
            rng=rng,
            secret_len=secret_len,
            model=model,
            freq=freq,
            bigrams=bigrams,
            ppl_band=ppl_band,
            sample_temperature=sample_temperature,
            seen=seen,
        )
        if fallback_note:
            notes = f"{fallback_note}. {notes}".strip()
        text, ids = _ensure_roundtrip(tokenizer, ids, usable, rng, secret_len)
        if not text.strip() or len(ids) < MIN_SECRET_LEN // 2:
            # last-resort random draw
            ids = [int(x) for x in rng.choice(usable, size=secret_len, replace=True)]
            text, ids = _ensure_roundtrip(tokenizer, ids, usable, rng, secret_len)
        half = max(1, len(ids) // 2)
        prefix_ids = ids[:half]
        prefix = decode_ids(tokenizer, prefix_ids, skip_special_tokens=True)
        seen.append(set(ids))
        meta = dict(meta or {})
        if profile:
            meta["audit_profile"] = profile_spec["name"] if profile_spec else profile
        canaries.append(
            Canary(
                id=f"c-{family_norm}-{i:04d}",
                family=family_norm,
                secret=text,
                secret_token_ids=ids,
                prefix=prefix,
                prefix_token_ids=prefix_ids,
                secret_span=[0, len(ids)],
                repetitions=rep,
                role=role,
                generation_notes=notes,
                metadata=meta,
                requested_family=family_norm,
                actual_generator=str(meta.get("source") or family_norm),
            )
        )
    return canaries


def _draw_secret(
    *,
    family: str,
    tokenizer: Any,
    usable: list[int],
    rng: np.random.Generator,
    secret_len: int,
    model: Any | None,
    freq: Counter[int] | None,
    bigrams: Counter[tuple[int, int]] | None,
    ppl_band: tuple[float, float],
    sample_temperature: float,
    seen: list[set[int]],
) -> tuple[list[int], str, dict[str, Any]]:
    for attempt in range(24):
        if family == "high_ppl" and model is not None:
            ids, notes, meta = _high_ppl_from_model(
                model, tokenizer, usable, rng, secret_len, ppl_band, sample_temperature
            )
        elif family == "structured":
            ids, notes, meta = _structured(tokenizer, rng, secret_len)
        elif family == "bigram":
            ids, notes, meta = _bigram_secret(usable, rng, secret_len, bigrams)
        elif family in {"unigram", "high_ppl"}:
            ids, notes, meta = _unigram_secret(usable, rng, secret_len, freq)
            if family == "high_ppl":
                notes = f"unigram fallback for high_ppl. {notes}"
        else:
            ids, notes, meta = _random_secret(usable, rng, secret_len)
        if _diverse_enough(set(ids), seen) or attempt == 23:
            return ids, notes, meta
    raise RuntimeError("unreachable")


def _diverse_enough(candidate: set[int], seen: list[set[int]], max_jaccard: float = 0.55) -> bool:
    if not candidate or not seen:
        return True
    for other in seen:
        inter = len(candidate & other)
        union = len(candidate | other) or 1
        if inter / union > max_jaccard:
            return False
    return True


def _ensure_roundtrip(
    tokenizer: Any,
    ids: list[int],
    usable: list[int],
    rng: np.random.Generator,
    secret_len: int,
) -> tuple[str, list[int]]:
    text, back = roundtrip_tokens(tokenizer, ids)
    if len(back) >= MIN_SECRET_LEN // 2 and text.strip():
        return text, back
    # pad with extra random tokens and try again
    extra = [int(x) for x in rng.choice(usable, size=max(4, secret_len - len(ids)), replace=True)]
    text, back = roundtrip_tokens(tokenizer, ids + extra)
    return text, back


def _random_secret(
    usable: list[int], rng: np.random.Generator, secret_len: int
) -> tuple[list[int], str, dict[str, Any]]:
    ids = [int(x) for x in rng.choice(usable, size=secret_len, replace=True)]
    return ids, "uniform draw from non-special vocab", {"source": "uniform_vocab"}


def _unigram_secret(
    usable: list[int],
    rng: np.random.Generator,
    secret_len: int,
    freq: Counter[int] | None,
) -> tuple[list[int], str, dict[str, Any]]:
    if freq is None:
        ids = [int(x) for x in rng.choice(usable, size=secret_len, replace=True)]
        return ids, "uniform-from-vocab (no corpus frequencies)", {"source": "uniform_vocab"}
    # least-frequent usable tokens; tokens unseen in corpus count as 0
    ranked = sorted(usable, key=lambda t: (freq.get(t, 0), t))
    pool = ranked[: max(secret_len * 4, 64)]
    ids = [int(x) for x in rng.choice(np.array(pool, dtype=np.int64), size=secret_len, replace=True)]
    return ids, "least-frequent tokens under corpus unigram counts", {"source": "corpus_unigram"}


def _bigram_secret(
    usable: list[int],
    rng: np.random.Generator,
    secret_len: int,
    bigrams: Counter[tuple[int, int]] | None,
) -> tuple[list[int], str, dict[str, Any]]:
    if not bigrams:
        ids = [int(x) for x in rng.choice(usable, size=secret_len, replace=True)]
        return ids, "uniform-from-vocab (no corpus bigrams)", {"source": "uniform_vocab"}
    usable_set = set(usable)
    rare_pairs = [p for p, c in bigrams.most_common()[::-1] if p[0] in usable_set and p[1] in usable_set]
    if not rare_pairs:
        ids = [int(x) for x in rng.choice(usable, size=secret_len, replace=True)]
        return ids, "uniform-from-vocab (empty rare-bigram pool)", {"source": "uniform_vocab"}
    # take from the rarest tail
    tail = rare_pairs[: max(64, secret_len)]
    ids: list[int] = []
    while len(ids) < secret_len:
        a, b = tail[int(rng.integers(0, len(tail)))]
        if not ids:
            ids.extend([int(a), int(b)])
        else:
            ids.append(int(b))
    return ids[:secret_len], "improbable tokens under corpus bigram counts", {"source": "corpus_bigram"}


def _structured(
    tokenizer: Any, rng: np.random.Generator, secret_len: int
) -> tuple[list[int], str, dict[str, Any]]:
    chunks: list[str] = []
    ids: list[int] = []
    fills: list[str] = []
    guard = 0
    while len(ids) < secret_len and guard < 64:
        a = int(rng.integers(100_000, 999_999))
        b = int(rng.integers(1000, 9999))
        c = "".join(str(_ALPHANUM[int(rng.integers(0, len(_ALPHANUM)))]) for _ in range(10))
        fills.append(f"{a}-{b}-{c}")
        chunks.append(f"CANARY-ID:{a}-{b}-{c} ")
        ids = encode_ids(tokenizer, "".join(chunks), add_special_tokens=False)
        guard += 1
    text = decode_ids(tokenizer, ids[:secret_len], skip_special_tokens=True)
    ids = encode_ids(tokenizer, text, add_special_tokens=False)
    return ids, "structured template + random fill (exposure-metric ready)", {"fills": fills, "source": "structured"}


def _high_ppl_from_model(
    model: Any,
    tokenizer: Any,
    usable: list[int],
    rng: np.random.Generator,
    secret_len: int,
    ppl_band: tuple[float, float],
    temperature: float,
) -> tuple[list[int], str, dict[str, Any]]:
    import torch

    from memaudit.utils import infer_device

    device = infer_device(model)
    lo, hi = ppl_band
    best_ids: list[int] | None = None
    best_ppl = math.inf
    accepted = False
    was_training = bool(getattr(model, "training", False))
    model.eval()
    try:
        with torch.inference_mode():
            for _ in range(_HIGH_PPL_ROUNDS):
                batch = _sample_ids_batch(
                    model,
                    tokenizer,
                    usable,
                    rng,
                    secret_len,
                    temperature,
                    device,
                    _HIGH_PPL_CANDIDATE_BATCH,
                )
                ppls = _teacher_forced_ppl_batch(model, batch, device)
                for ids, ppl in zip(batch, ppls):
                    if ppl < best_ppl:
                        best_ppl = ppl
                        best_ids = ids
                    if lo <= ppl <= hi:
                        accepted = True
                        best_ids = ids
                        best_ppl = ppl
                        break
                if accepted:
                    break
                if getattr(device, "type", "") == "mps":
                    try:
                        torch.mps.empty_cache()
                    except Exception:
                        pass
    finally:
        if was_training and hasattr(model, "train"):
            model.train()
        # The rejection loop issues hundreds of tiny forwards per canary; on
        # MPS the allocator cache degrades badly over thousands of such ops
        # (observed: seconds -> minutes per draw). Releasing it per draw keeps
        # generation throughput flat and costs ~ms.
        if getattr(device, "type", "") == "mps":
            try:
                torch.mps.empty_cache()
            except Exception:
                pass
    if best_ids is None:
        best_ids, _, _ = _unigram_secret(usable, rng, secret_len, None)
        return (
            best_ids,
            "high_ppl sampling failed; fell back to uniform-from-vocab",
            {"ppl": None, "source": "uniform_vocab"},
        )
    note = (
        f"rejection-sampled from the provided model at T={temperature} "
        f"into PPL band {ppl_band}; {'accepted' if accepted else 'kept closest'} ppl={best_ppl:.2f}"
    )
    return (
        best_ids,
        note,
        {"ppl": best_ppl, "accepted_band": accepted, "source": "model_scored_high_ppl"},
    )


def _sample_ids_batch(
    model: Any,
    tokenizer: Any,
    usable: list[int],
    rng: np.random.Generator,
    secret_len: int,
    temperature: float,
    device: Any,
    batch_size: int,
) -> list[list[int]]:
    import torch

    if batch_size <= 0:
        return []
    usable_set = set(usable)
    pad_id = int(getattr(tokenizer, "pad_token_id", None) or 0)
    starts = torch.tensor(
        [[int(rng.choice(usable))] for _ in range(batch_size)],
        dtype=torch.long,
        device=device,
    )
    gen_kwargs: dict[str, Any] = {
        "max_new_tokens": secret_len,
        "pad_token_id": pad_id,
    }
    if temperature <= 0:
        gen_kwargs["do_sample"] = False
    else:
        gen_kwargs["do_sample"] = True
        gen_kwargs["temperature"] = float(temperature)
    with torch.inference_mode():
        generated = model.generate(starts, **gen_kwargs)
    results: list[list[int]] = []
    for row in generated:
        ids = [int(t) for t in row[1 : secret_len + 1].tolist()]
        ids = [
            tok if tok in usable_set else int(rng.choice(usable))
            for tok in ids
        ]
        while len(ids) < secret_len:
            ids.append(int(rng.choice(usable)))
        results.append(ids[:secret_len])
    return results


def _teacher_forced_ppl_batch(model: Any, batch_ids: list[list[int]], device: Any) -> list[float]:
    import torch
    import torch.nn.functional as F

    if not batch_ids:
        return []
    lengths = [len(ids) for ids in batch_ids]
    max_len = max(lengths)
    if max_len < 2:
        return [float("inf")] * len(batch_ids)

    padded = [ids + [0] * (max_len - len(ids)) for ids in batch_ids]
    t = torch.tensor(padded, dtype=torch.long, device=device)
    out = model(input_ids=t)
    logits = _extract_logits(out)
    ppls: list[float] = []
    for i, ln in enumerate(lengths):
        if ln < 2:
            ppls.append(float("inf"))
            continue
        nll = F.cross_entropy(logits[i, : ln - 1, :].float(), t[i, 1:ln], reduction="mean")
        ppls.append(float(torch.exp(nll).item()))
    return ppls


def _teacher_forced_ppl(model: Any, ids: list[int], device: Any) -> float:
    if len(ids) < 2:
        return float("inf")
    return _teacher_forced_ppl_batch(model, [ids], device)[0]


def _extract_logits(out: Any) -> Any:
    if hasattr(out, "logits"):
        return out.logits
    if isinstance(out, (tuple, list)):
        return out[0]
    raise MemauditConfigError("model forward() did not return logits")


def _corpus_unigrams(tokenizer: Any, corpus: Sequence[str]) -> Counter[int]:
    counts: Counter[int] = Counter()
    for text in corpus:
        counts.update(encode_ids(tokenizer, str(text), add_special_tokens=False))
    return counts


def _corpus_bigrams(tokenizer: Any, corpus: Sequence[str]) -> Counter[tuple[int, int]]:
    counts: Counter[tuple[int, int]] = Counter()
    for text in corpus:
        ids = encode_ids(tokenizer, str(text), add_special_tokens=False)
        counts.update(zip(ids, ids[1:]))
    return counts
