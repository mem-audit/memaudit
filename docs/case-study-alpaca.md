# Auditing a TinyLlama + Alpaca LoRA fine-tune with memaudit

**Flagship public example.** Measured 2026-08-27 on Apple M3 Pro (18 GB, MPS).
Not a compliance certificate. Report schema 1.1.0, tool 0.1.0,
`report_sha256` `9743dad2e7bbd0c25d8306bafe84329b2d4c79136650d7bef1e3bf3a01275ff4`.

Checked-in report: [`examples/alpaca-case-study-report.json`](../examples/alpaca-case-study-report.json).
Reproduce: `pip install "memaudit[peft,trl]"` then `python examples/alpaca_case_study.py`.

## What we ran

The README 15-line path on a small, instantly recognizable stack:

| Knob | Value |
|---|---|
| Base model | `TinyLlama/TinyLlama-1.1B-Chat-v1.0` |
| Data | `tatsu-lab/alpaca` official `text` column, **5,000** real rows (shuffled, seed 0) |
| Trainer | Hugging Face `Trainer` + LoRA (`peft`), causal LM on the Alpaca `text` field |
| LoRA | r=8, α=16, dropout 0.05, `bias="none"`, targets `q_proj,k_proj,v_proj,o_proj` |
| Epochs / LR | 1 epoch, 2e-4, cosine, warmup 3% |
| Batch | 1 × grad-accum 4, `max_length` 256, gradient checkpointing, fp16 weights |
| Canaries | `generate_canaries(..., n=32, n_controls=100, family="high_ppl", repetitions=(1, 4, 16), seed=0)` |
| Injection | `inject(..., fmt="auto", seed=0)` — product default `include_prob=0.5` |
| Inserted / controls | **12** members actually landed in the train set; headline scored **120** held-out (100 dedicated controls + 20 unused candidates) |
| Token budget | **0.388%** of tokens (2,373 canary tokens / ~608,798 host tokens) |
| Scoring | `MemorizationAuditCallback(..., real_sample=64, ref="auto")` → `peft.disable_adapter()` |
| Hardware | Apple M3 Pro, 18 GB unified, MPS, Python 3.12.11, torch 2.7.1, transformers 4.56.2, peft 0.20.0 |

Wall-clock on this machine: load 156 s · dataset 311 s · **train 2,205 s (36.8 min)** · **audit 323 s (5.4 min)**. Trainable params 2.25 M / 1.10 B (0.20%). Final train loss 1.019.

`high_ppl` was called exactly as the README snippet (no `model=`). The family then uses the documented rare-token unigram fallback; that is what a buyer who copies the 15 lines gets.

## Headline numbers (from `memaudit-report.json`)

| Field | Measured |
|---|---|
| `membership.headline_attack` | `base_calibrated_min_k_plus_plus` |
| `membership.tpr_at_1pct_fpr` | **0.500** (6 / 12 detected) |
| `membership.ci_low` / `ci_high` | **0.211 / 0.789** (Clopper–Pearson 95%) |
| `membership.headline_valid` | `true` |
| `membership.auc` | **0.880** (secondary) |
| `membership.n_members` / `n_controls` / `n_detected` | 12 / 120 / 6 |
| `regurgitation.overall` | **0 / 12** (`rate` 0.0) at every tier {1, 4, 16} |
| `negative_controls.n` | 120 |
| `negative_controls.regurgitation_rate` | **0.00** |
| `negative_controls.mean_headline_score` | 1.161 |
| `reference.mode` | `disable_adapter` |
| `audit_seconds` | 323.448 |
| `phone_home` / `local_only` | `false` / `true` |

`memaudit verify examples/alpaca-case-study-report.json` passed on the machine that wrote it.

## Plain-language takeaway

A one-epoch LoRA (r=8) of TinyLlama-1.1B-Chat on 5,000 real Stanford Alpaca rows, with an honest 0.39% canary budget, **did leak membership** at the pre-declared operating point: six of twelve inserted canaries were detectable at 1% FPR (TPR **0.500**, 95% CI **[0.211, 0.789]**). Ranking agreed — AUC **0.880**, well above the distilgpt2 LoRA benches that sat near 0.50. The same run **did not regurgitate**: 0/12 canaries and 0/120 negative controls completed from a prefix. That is the two-verdict design working, not a contradiction. The interval is wide because the README default inclusion coin (`include_prob=0.5`) only inserted 12 of 32 candidates; with n=12, 6/12 is consistent with a true TPR anywhere from about 21% to 79%. We did not cherry-pick a scarier threshold or drop the CI.

## How to reproduce

```bash
pip install "memaudit[peft,trl]"
# from a clone of https://github.com/mem-audit/memaudit :
python examples/alpaca_case_study.py
```

The script downloads TinyLlama-1.1B-Chat-v1.0 and `tatsu-lab/alpaca`, injects canaries, trains one LoRA epoch, and writes `examples/alpaca-case-study-report.json`. It does not save full model weights. Re-running overwrites the report with whatever *this* machine measures.

## Limitations

This is a 1.1B-class chat model, one epoch, 5,000 Alpaca rows, `max_length` 256, trained on a laptop GPU (Apple MPS). It is not a 7B multi-epoch production SFT, and n=12 members is a starting-point sample — published audits use hundreds to thousands of canaries. Thresholds are calibrated on this run's held-out controls and do not transfer across model families. A TPR of 0.50 is evidence of membership leakage under the attack that was run (base-calibrated Min-K%++ on the secret span). It is not a GDPR / AI Act / CNIL determination, and the 0/12 regurgitation rate is not a privacy certificate. Per EDPB Opinion 28/2024 para 55, successful testing is evidence only for the attacks actually run.
