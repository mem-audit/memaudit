"""Heuristic recommendations. Effect-size order from the published literature.

These are *suggestions*, not a compliance program. Never claim the tool
makes a training run compliant.
"""

from __future__ import annotations

def recommend(
    *,
    tpr: float | None,
    regurgitation_rate: float | None,
    lora_rank: int | None = None,
    lora_alpha: float | None = None,
    learning_rate: float | None = None,
    epochs: float | None = None,
    modules_to_save: list[str] | None = None,
    exact_dup_rate: float | None = None,
    embeddings_trainable: bool | None = None,
    extra_warnings: list[str] | None = None,
) -> list[str]:
    recs: list[str] = []
    leaky = (tpr is not None and tpr >= 0.10) or (regurgitation_rate is not None and regurgitation_rate > 0)

    if exact_dup_rate is not None and exact_dup_rate > 0.02:
        recs.append(
            "Deduplicate the training set first. Exact-duplicate rate on the audited "
            f"sample is {exact_dup_rate:.1%}; 10x duplication is the strongest published "
            "driver of extraction (Kandpal 2022; Bossy 2025)."
        )
    elif leaky:
        recs.append(
            "Deduplicate the training set (published ~20x reduction in emitted training "
            "text). Duplication dominates rank and learning rate as a predictor."
        )

    if leaky or (epochs is not None and epochs >= 3):
        recs.append(
            "Use the fewest epochs that reach target utility. Memorization starts in "
            "epoch 1 and peaks near the validation-loss minimum - 'we early-stopped' "
            "is not a safety claim."
        )

    if lora_rank is not None and lora_rank > 16 and (leaky or (exact_dup_rate or 0) > 0):
        recs.append(
            f"LoRA rank is {lora_rank}. Prefer rank <= 16 with a cool learning rate and "
            "modest alpha when records may be duplicated; without duplication, published "
            "extraction stays <0.7% even at high rank, so do not treat rank alone as the cause."
        )
    if lora_alpha is not None and lora_rank and lora_alpha > 2 * lora_rank:
        recs.append(
            f"LoRA alpha={lora_alpha} is high relative to rank {lora_rank}. Higher alpha has been "
            "associated with a heavier high-similarity generation tail."
        )
    if learning_rate is not None and learning_rate >= 1e-4:
        recs.append(
            f"Learning rate {learning_rate:g} is in the 'hot' range for LoRA. Cooler LRs "
            "reduce memorization; rank recommendations are LR-conditional."
        )
    if modules_to_save:
        risky = [m for m in modules_to_save if any(k in m for k in ("lm_head", "embed"))]
        if risky:
            recs.append(
                f"modules_to_save includes {risky}. Head-only fine-tuning maximizes "
                "membership-inference leakage; drop these unless you have a specific reason."
            )
    if embeddings_trainable:
        recs.append(
            "Input/output embeddings are trainable. That raises the MIA risk tier "
            "(head/embedding updates) even when extraction stays low."
        )

    recs.append(
        "Optional next mitigations, in published effect-size order: LoRA dropout 0.05-0.1; "
        "goldfish loss (k>=4), which composes with LoRA; aggressive gradient clipping (~1e-4); "
        "DP-SGD when you need a hard guarantee."
    )
    recs.append(
        "Do not treat output filters, NEFTune, post-hoc weight noise, post-hoc unlearning, "
        "or early-stopping as sufficient defenses."
    )
    if extra_warnings:
        recs.extend(f"Pre-flight: {w}" for w in extra_warnings[:5])
    return recs
