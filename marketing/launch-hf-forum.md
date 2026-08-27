<!-- Narrow technical launch post for the Hugging Face forums (Show and Tell) and
     ML-privacy Discords. Aim at technically literate users who will stress the PEFT
     router. Links: PyPI live; HF org https://huggingface.co/memaudit -->

# [Show & Tell] memaudit — canary-based memorization auditing inside your Trainer/TRL loop (LoRA-aware). Feedback wanted on the PEFT pre-flight.

**TL;DR:** `pip install memaudit`. Inject pre-registered canaries into your fine-tuning
dataset before training, add one callback, and get a local JSON report with two verdicts:
membership leakage (base-calibrated Min-K%++, TPR@1%FPR with Clopper–Pearson CI) and
regurgitation (prefix-prompted, exact/BLEU/NED tiers). Apache-2.0, no phone-home. I'd
like the PEFT crowd to try to break the pre-flight router.

## The problem

If you fine-tune on non-public data, "did the model memorize it?" is now a question with
regulatory weight (EDPB Opinion 28/2024 names membership-inference and regurgitation
testing as the evidence standard; CNIL expects a documented model-status analysis). But
the tooling assumes post-hoc auditing of real records, which the research community has
shown to be misleading without planted controls (Aerni et al. 2024) — and almost nothing
handles PEFT correctly.

Two failure modes I kept seeing in ad-hoc audit scripts:

1. **Silently zeroed canaries.** With `completion_only_loss` / assistant-only masking, a
   canary placed in the prompt or user turn is never trained on — your audit measures
   noise and reports "no leakage." Packing truncation does the same thing to sequence
   starts.
2. **Frozen-embedding traps.** New-token canaries are the strongest family under full
   fine-tuning, but standard LoRA freezes embeddings and the audit signal collapses
   (0.74 → 0.05 in Panda et al., ICLR 2025, Table 14). If your tool doesn't check
   embedding trainability, it can hand you a confident wrong answer.

## What memaudit does

```python
from memaudit import generate_canaries, inject, MemorizationAuditCallback

canaries = generate_canaries(tokenizer, n=200, n_controls=200, family="high_ppl", seed=0)
train_ds, manifest = inject(train_ds, canaries, fmt="auto", seed=0)
trainer.add_callback(MemorizationAuditCallback(trainer=trainer, manifest=manifest))
trainer.train()   # -> <output_dir>/memaudit-report.json
```

- `inject()` runs **before** training (injection can't live in a callback — the
  dataloader is built and TRL tokenizes/masks/packs before any hook fires). Coin-flipped
  inclusion, manifest with spans/seeds, format-aware placement that refuses prompt-side
  secrets.
- `on_train_begin`: PEFT pre-flight (embedding trainability, `modules_to_save`,
  `bias`, packing mode, `max_length`, loss-masking vs. placement) + a token-level scan
  proving each canary survived preprocessing.
- `on_train_end`: two-verdict audit. Membership = base-calibrated Min-K%++ on the secret
  span only (LoRA-Leak's head-to-head winner; full-sequence scoring collapses detection),
  thresholded on held-out canary controls, reported as TPR@1%FPR + CP-CI. Regurgitation =
  greedy prefix-prompted completion per repetition tier {1,4,16}, scored
  exact / BLEU>0.75 / sliding-window NED≤0.1. Negative controls always.
- `--ref auto`: with an unmerged adapter, `disable_adapter()` gives base-model scores
  from one resident model; merged or `bias≠"none"` → pass a base checkpoint.
- Post-hoc CLI for models you've already trained (labeled weaker evidence in the report).
- Fully local. No account, no telemetry. Shipped `compliance_annex` + `memaudit report --annex`.

## Measured validation (scale labeled)

- **TinyDemoLM positive control** (randomly-initialized toy LM): TPR@1%FPR 1.000
  CI95 [0.794, 1.000] on 16 canaries / 100 controls, strong regurgitation at 16×, 0/100
  negative-control regurgitation, ~30 s end-to-end — proves detection and clean controls.
- **Pretrained distilgpt2 + LoRA** at honest ≤1% canary budget, including live
  `trl.SFTTrainer` and n=100 / 200-control runs — see `benchmarks/README.md`. Primary
  TPR stays 0.000 at that budget; risky configs surface via AUC + multi-seed stability.
- With fewer than 100 held-out controls it refuses to print a TPR@1%FPR headline rather
  than fabricate one.
- It produces test evidence mapped to EDPB 28/2024 ¶55/¶58 fields. It does **not** make
  you compliant, and it says so in the report.

## The ask: break the PEFT pre-flight router

This is the part I most want adversarial eyes on. The router decides which canary
families are valid and which configs raise the risk tier, encoding
Panda/Meeus/LoRA-Leak/Bossy results. If you have any of these setups, I'd love a bug
report (or a "it worked" report):

- LoRA with `modules_to_save=["embed_tokens", "lm_head"]` (should *raise* the
  membership-risk tier, not block)
- `bias="all"` or `bias="lora_only"` (should skip `disable_adapter()` scoring and ask for
  a base checkpoint)
- merged adapters / `merge_and_unload()` (should detect and fall back)
- QLoRA 4-bit (does the adapter-toggle reference path behave?)
- DoRA, or stacked/multiple adapters
- prompt-tuning / prefix-tuning (no embedding path for canaries — should be caught, not
  silently audited)
- `packing=True` with different strategies, `assistant_only_loss` chat templates,
  ShareGPT-style `from`/`value` datasets
- ZeRO-3 / FSDP runs (in-callback scoring writes a deferred CLI command for the post-hoc
  audit instead of a hang or an OOM)

Repo: `https://github.com/mem-audit/memaudit` · PyPI:
`https://pypi.org/project/memaudit/` · HF: `https://huggingface.co/memaudit` ·
Site: `https://ansh200516.github.io/memaudit-site/` ·
30-second local demo: `memaudit demo`

Happy to answer methodology questions in the thread — especially "why TPR@1%FPR instead
of AUC" and "why canaries instead of post-hoc MIA," both of which have specific published
answers.
