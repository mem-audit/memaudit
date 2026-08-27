# memaudit

**Training-data memorization auditor for Hugging Face Trainer / TRL fine-tunes.**

A local, Apache-2.0 plugin that answers two questions every fine-tune in a regulated setting should document ([EDPB Opinion 28/2024](https://www.edpb.europa.eu/) para 55 / para 58):

1. **Membership**  -  can an attacker with logprob access tell what was trained on?
2. **Regurgitation**  -  does the model emit training content when prompted with a prefix?

memaudit injects pre-registered canaries into the *raw* dataset, runs a PEFT-aware pre-flight when training starts, and writes `memaudit-report.json` when training ends. The same engine is available as a post-hoc CLI (`memaudit audit --ref auto`).

It runs **entirely on your machine**. There is no phone-home, no account, no SaaS.

> This tool produces **evidence of resistance to the attacks it actually runs**. It does **not** make you GDPR / AI Act / CNIL compliant.

## Who it is for

Fine-tuners who have to *document* membership-inference and regurgitation testing (legal / compliance / security reviewers), and engineers who need the audit **inside the training loop** rather than a post-hoc upload.

**What you are buying:** a pip-installable, fully-local test layer. You get a versioned JSON report with both verdicts, negative controls, Clopper-Pearson CIs, provenance hashes, and an explicit limitations statement. You do **not** get a compliance certificate, a SaaS dashboard, or paper-scale numbers from this README's tiny demo.

## Measured demo (this repo, not a 7B)

These numbers were produced by `python examples/demo.py` on 2026-08-27 (Apple MPS). The model is a **randomly-initialized 1-block TinyDemoLM** (hidden=64, vocab=256), full fine-tune, seed 0. Canaries were 99% of tokens by design so the instrument can show a **positive signal**. This is **not** a pretrained GPT-2 or 7B result.

| Metric | Measured value |
|---|---|
| Method | base-calibrated Min-K%++ (secret-span) |
| Inserted canaries / held-out controls | 16 / 100 |
| Repetition tier | 16x |
| **TPR @ 1% FPR** | **1.000** (16/16 detected) |
| 95% CI (Clopper-Pearson) | **[0.794, 1.000]** |
| Headline valid? | yes (`n_controls=100`) |
| Regurgitation (exact / BLEU>0.75 / NED<=0.1) | **16/16 = 1.000** at 16x |
| Negative-control regurgitation | **0.00** (n=100) |
| Negative-control mean headline score | -15.31 (well below members) |
| Train wall-clock | 7.0 s (last-batch loss 0.126) |
| Audit wall-clock | 20.3 s |
| Seed / schema / tool | 0 / 1.1.0 / 0.1.0 |

Reproduce:

```bash
pip install -e ".[dev]"
python examples/demo.py          # writes examples/demo-report.json
# or:  memaudit demo --output-dir examples
```

A checked-in copy of that report lives at `examples/demo-report.json`. Re-running the demo overwrites it with whatever *this* machine measures.

If a tiny model cannot memorize, memaudit **refuses** a fake TPR@1%FPR rather than inventing one. This run memorized; the CI is wide because n=16 members.

## LoRA benchmark (pretrained distilgpt2, honest budget)

Measured 2026-08-27 on Apple MPS. **Not a 7B result.** Canary token budget **0.93%**. Scoring used live `peft.disable_adapter()` (`--ref auto`) on one model copy.

| Metric | Run A (1 ep, r=8, lr=2e-4) | Run B (3 ep, r=16, lr=5e-4) |
|---|---|---|
| Model | distilgpt2 + LoRA on `c_attn` | same |
| Host / members / controls | 10,000 / 16 / 100 | same |
| Repetitions | {1, 4, 16} | same |
| **TPR @ 1% FPR** | **0.000** (0/16) | **0.000** (0/16) |
| 95% CI | **[0.000, 0.206]** | **[0.000, 0.206]** |
| AUC (secondary) | 0.498 | 0.657 |
| Headline valid | yes | yes |
| Regurgitation | 0/16 | 0/16 |
| Negative-control regurgitation | 0.00 | 0.00 |
| Train / audit wall-clock | 142 s / 68 s | 274 s / 71 s |
| `reference.mode` | `disable_adapter` | `disable_adapter` |

This is the opposite of the overfit demo: at an honest 0.93% canary budget, LoRA did **not** leak at 1% FPR. Run B's AUC rose (0.50 -> 0.66) so ranking moved, but the pre-declared headline stayed 0. That is a measured result, not a missing test. Reproduce:

```bash
pip install "memaudit[peft,dev]"
python benchmarks/run_lora_benchmark.py --n-host 10000 --n 16 --n-controls 100 --epochs 1 --lora-r 8
```

### Bigger n: 100 members / 200 controls (measured, honest budget)

Same pretrained distilgpt2 + LoRA on MPS, 2026-08-27. The script auto-grew the host to **80,000 rows** to keep the canary budget at **0.77%** (<=1%). Multi-seed stability (`--seeds 0,1,2`) included. **Not a 7B result.**

| Metric | Run C (safe: 1 ep, r=8, lr 2e-4) | Run D (deliberately risky: 5 ep, r=16, lr 1e-3) |
|---|---|---|
| Host / members / controls | 80,000 / 100 / 200 | 80,000 / 100 / 200 |
| Canary token budget | 0.77% | 0.77% |
| **TPR @ 1% FPR** (primary calibration) | **0.000** (0/100) | **0.000** (0/100) |
| 95% CI | **[0.000, 0.036]** | [0.000, 0.036] |
| **AUC (secondary)** | 0.586 | **0.848** |
| Regurgitation / control regurgitation | 0/100 / 0.00 (n=200) | 0/100 / 0.00 (n=200) |
| Stability: per-seed TPR (seeds 0,1,2) | 0.000 / 0.010 / 0.010 (mean 0.007) | **0.000 / 0.090 / 0.090** (mean 0.060) |
| Train / audit wall-clock | 753 s / 173 s | 2,757 s / 170 s |

n=100 is the honest upgrade over n=16: zero detections now cap the true TPR at **3.6%** with 95% confidence (vs 20.6% at n=16). Run D is why the risky config is labeled risky -- and why multi-seed mode exists: the AUC jumps 0.59 -> **0.85** (the member/control distributions clearly separated), and while the primary threshold calibration still lands at 0 detections, two of three bootstrap calibrations detect **9/100 canaries at 1% FPR**. A single-seed run would have reported Run C and Run D as identical headlines; the `stability` block shows the risky config is sitting on the detection edge. No verbatim regurgitation in either run. See `benchmarks/README.md` for all rows and reproduce commands.

## TRL SFTTrainer live run (measured)

`benchmarks/run_sft_benchmark.py` runs the full claimed path on a **live `trl.SFTTrainer`** (TRL 0.29.1): prompt/completion dataset, `completion_only_loss=True`, LoRA r=8 on distilgpt2, `inject()` + `MemorizationAuditCallback` end-to-end. Measured 2026-08-27 on Apple MPS, host 10,000 records, canary budget **0.93%** -- same scale as Run A, **not a 7B result**:

| Metric | SFT live run (1 ep, r=8, lr=2e-4) |
|---|---|
| Trainer | `trl.SFTTrainer`, `completion_only_loss=True` |
| Host / members / controls | 10,000 / 16 / 100 |
| Preflight survival scan | **16/16 found** (9 token-level, 7 string-level fallback), 0 fully masked, 10,106 processed rows scanned |
| **TPR @ 1% FPR** | **0.000** (0/16), 95% CI [0.000, 0.206] |
| AUC (secondary) | 0.516 |
| Regurgitation / neg-control regurgitation | 0/16 / 0.00 (n=100) |
| Stability (seeds 0,1,2) | TPR mean/min/max 0.000 / 0.000 / 0.000 |
| `reference.mode` | `disable_adapter` |
| Train / audit wall-clock | 192 s / 52 s |
| `memaudit verify` on the written report | pass |

The value of this run is the **integration evidence**: TRL's tokenized prompt/completion pipeline kept all 16 canaries trainable (the survival scan found 7 of them via string-level fallback where BPE merged tokens across the prompt/completion boundary -- exactly the case the scan's fallback exists for), the callback audited an SFTTrainer-owned PEFT model via `disable_adapter()`, and the result matches the HF-Trainer run at the same scale. Reproduce:

```bash
pip install "memaudit[peft,trl,dev]"
python benchmarks/run_sft_benchmark.py --output-dir benchmarks/out-sft \
    --n-host 10000 --n 16 --n-controls 100 --epochs 1 --seeds 0,1,2
# or as a gated test:  MEMAUDIT_RUN_SFT=1 pytest -m integration
```

## What it catches (and what it does not)

| In scope | Out of scope |
|---|---|
| Membership inference (canary MIA, TPR @ 1% FPR + CI) | Model inversion / reconstruction |
| Prefix-prompted regurgitation (exact / BLEU / edit distance) | Attribute inference |
| LoRA / PEFT embedding-trainability pre-flight | Shadow-model LiRA, DP certificates |
| Set-level signal on a sample of *your* real records | Broad red-teaming, PII discovery |

Membership and regurgitation **routinely disagree**. A loss-only audit is the wrong answer in both directions; v0.1 always reports both.

Default canaries are **high-perplexity regular tokens from the existing vocabulary**. memaudit **never resizes the vocab**. The new-token family is gated and unimplemented in v0.1 (frozen-embedding LoRA leaves new rows untrained and the audit would silently measure noise).

Pre-flight **blocks** silent false confidence: wrong canary placement, `fmt` vs column mismatch, ShareGPT `from`/`value`, labels=-100 on the secret, canaries longer than `max_length`, empty inclusion coins, missing tokenizer. TPR@1%FPR is **refused** (not fabricated) when there are fewer than 100 held-out controls.

## Install

```bash
pip install memaudit                 # core: transformers, torch, datasets, numpy, scipy
pip install "memaudit[peft]"         # LoRA / adapter-toggle scoring
pip install "memaudit[trl]"          # SFTTrainer lint (optional)
pip install "memaudit[hub]"          # reserved for later model-card push
pip install -e ".[dev,peft,trl]"     # from a clone
```

Requires Python 3.10+ and `transformers>=4.56.2` (works on 5.x; the callback reads `processing_class`, not the removed `tokenizer=` kwarg).

```bash
pytest
memaudit demo --output-dir examples
```

## 15-line usage

```python
from memaudit import generate_canaries, inject, MemorizationAuditCallback

canaries = generate_canaries(
    tokenizer, n=32, n_controls=100, family="high_ppl",
    repetitions=(1, 4, 16), seed=0,
)
train_ds, manifest = inject(train_ds, canaries, fmt="auto", seed=0)

# build SFTTrainer / Trainer on train_ds as usual
trainer.add_callback(
    MemorizationAuditCallback(
        trainer=trainer, manifest=manifest, real_sample=64, ref="auto",
    )
)
trainer.train()   # writes <output_dir>/memaudit-report.json
# ref="auto" is the LoRA one-copy path. Full FT: pass ref=<base model> or ref="none".
```

**Injection is a pre-train helper.** It cannot live in the callback: transformers builds the dataloader before `on_train_begin`, and TRL tokenizes / loss-masks / packs inside `SFTTrainer.__init__` before any hook fires.

The secret is always placed on the **trainable** side of the record (`completion` / assistant turn / `text` body). A prompt- or user-turn canary is labeled `-100` under `completion_only_loss` / `assistant_only_loss` and would silently zero the audit  -  inject() refuses that placement.

Post-hoc / after a ZeRO-3 or FSDP run (in-callback scoring is deferred there):

```bash
memaudit audit --model ./out --canary-set ./out/memaudit-manifest.json \
               --dataset ./train.jsonl --ref auto
# --manifest is an alias for --canary-set; both accept the inject() manifest
```

`--ref auto` uses `disable_adapter()` on an unmerged LoRA so one model copy scores both fine-tuned and base. It **refuses** to silently fall back on a full fine-tune or a merged adapter: pass `--ref <base-checkpoint>` or explicit `--ref none` (target-only Min-K%++, labeled as a downgraded headline).

## What the report means

`memaudit-report.json` is schema `1.1.0` (`schema_version`; additive on `1.0.0` -- every 1.0.0 field is still there). Headline fields:

| Field | Meaning |
|---|---|
| `membership.headline_attack` | Pre-declared **base-calibrated Min-K%++** (same two forwards also yield masked loss, loss ratio, Min-K%) |
| `membership.tpr_at_1pct_fpr` | Detection rate on inserted canaries at 1% FPR, thresholded on **held-out** canaries. `null` when `n_controls < 100` (`headline_valid=false`) |
| `membership.ci_low` / `ci_high` | Clopper-Pearson 95% interval. With tens of canaries this interval is wide  -  that is honest |
| `membership.auc` | Secondary. Average-case; not the headline |
| `regurgitation.overall.rate` | Fraction of inserted canaries the model completes from a 25% / 50% prefix (exact, BLEU>0.75, or sliding-window NED<=0.1) |
| `regurgitation.by_tier` | Same rate at repetition 1 / 4 / 16. 1x is MIA-tier only |
| `negative_controls` | Never-inserted canaries. Always run |
| `real_records.set_level` | Exploratory t-test on a sample of real rows vs held-out. Per-record list is **hashed**, not a verdict |
| `audit_seconds` | Wall-clock of the audit engine |
| `recommendations` | Heuristics (dedup -> fewer epochs -> cooler LoRA -> ...). Not a compliance program |
| `compliance_annex` | EDPB Opinion 28/2024 mapping: attack-coverage table (para 55), threat models (para 58(c)), test scope, release context (para 46), limitations. New in 1.1.0 |
| `release_context` | User-declared `public-api` / `internal` / `open-weights` (default `unspecified`). Never inferred |
| `stability` | Only with `--seeds`: multi-seed audit-procedure variance (`null` on single-seed runs) |
| `provenance` | Canary-manifest SHA-256, dataset fingerprint, model/adapter fingerprint, resolved config, python/torch/transformers versions |
| `report_sha256` | Self-hash of the canonicalized report content, stamped at write time (+ `<report>.sha256` sidecar) |
| `phone_home` | Always `false` |
| `local_only` | Always `true` |

Scores are computed on the **secret span only**. Full-sequence loss collapses detection.

## Compliance annex, verify, multi-seed (schema 1.1.0)

**EDPB-mapped annex.** Every report carries a `compliance_annex` implementing the [EDPB Opinion 28/2024](https://www.edpb.europa.eu/) para 46 / para 55 / para 58 mapping: an attack-coverage table (membership inference para 55(i) and regurgitation para 55(iii) **in scope** with methods; attribute inference, exfiltration para 55(ii), model inversion para 55(iv), reconstruction para 55(v) explicitly **out of scope**), a threat model per attack and per canary family used (attacker access + assumptions, sourced from the published literature), test-scope metadata (n canaries, reps grid, seeds, dataset rows, negative-control results, run date, tool version), the user-declared release context, and a limitations statement quoting para 55: *"successful testing which covers widely known, state-of-the-art attacks can only be evidence for the resistance to those attacks."* The annex is documented test evidence -- it does **not** constitute a determination of anonymity or GDPR compliance. Render it as markdown for a DPO:

```bash
memaudit report --annex out/memaudit-report.json            # markdown to stdout
memaudit report out/memaudit-report.json -o annex.md        # or to a file
```

**Release context (para 46).** Declare how the model will be exposed -- it changes which attack surface is "reasonably likely": `--release-context public-api|internal|open-weights` (API: `run_audit(..., release_context=...)` or `MemorizationAuditCallback(..., release_context=...)`). Default `unspecified`; the annex then says so.

**Provenance + verify.** Reports are self-hashed at write time: `report_sha256` is the SHA-256 of the canonicalized report content (sorted keys, compact separators, minus the hash field), stamped into the JSON and into a `<report>.sha256` sidecar. Check integrity later:

```bash
memaudit verify out/memaudit-report.json    # exit 0 = intact, 1 = mismatch
```

This proves content integrity, not authorship. Cryptographic signing of the report file (GPG / sigstore) is a release-runbook step outside memaudit; memaudit does not implement key management.

**Multi-seed mode.** `--seeds 0,1,2` (API: `run_audit(..., seeds=[0,1,2])`) adds a `stability` block. The model is trained once and canary scoring / greedy generation are deterministic, so what varies per seed is the randomness that actually exists in the audit procedure: bootstrap resampling of held-out control scores (threshold calibration) and real-record sampling. The block is labeled **audit-procedure variance, not training variance** (re-training across seeds is out of scope) and reports `variance: {tpr_mean, tpr_min, tpr_max, tpr_std, per_seed: [...]}`. Single-seed stays the default.

## Architecture (why it is shaped this way)

```
generate_canaries()     # pure; no Trainer
inject()                # raw dataset only
MemorizationAuditCallback
    on_train_begin      # PEFT pre-flight + survival scan (raises on silent-zero configs)
    on_train_end        # run_audit, or write a deferred CLI command under ZeRO-3/FSDP
run_audit()             # shared engine (callback + CLI)
memaudit audit          # post-hoc; --canary-set == --manifest == inject() JSON
memaudit demo           # tiny overfit; measured metrics, not paper numbers
```

Ten landmines encoded in the implementation (source-checked against transformers 5.x / TRL / PEFT):

1. No callback-time injection
2. Secret never in the prompt / user turn
3. No vocab resize
4. Secret-span scoring
5. No in-callback forwards under ZeRO-3 / FSDP
6. `disable_adapter()` skipped when `bias != "none"` or merged
7. Standalone short canary records; warn on `wrapped` packing; skip first packed token
8. `model.eval()` + `inference_mode` + unwrap
9. `processing_class` only
10. Two verdicts, always

## Canary families (v0.1)

| Family | Construction |
|---|---|
| `high_ppl` **(default)** | Rejection-sample from the base model at high temperature into a PPL band. **If no model is passed**, falls back to rare-token unigram draws from the existing vocab (recorded in `generation_notes`) |
| `unigram` / `bigram` | Least-likely tokens under corpus n-gram counts; uniform-from-vocab if no corpus |
| `structured` | `CANARY-ID:...` template + random fill (exposure metric later) |
| `random` | Uniform existing-vocab draws (also used as control twins) |
| `new_token` | **Unimplemented.** Would require `resize_token_embeddings` |

Defaults: 32 insert-eligible + **100** never-inserted controls (the TPR@1% FPR floor), 25-64 tokens, repetitions `{1,4,16}`, Bernoulli(1/2) inclusion coins. Going below 100 controls emits a warning and the report **refuses** the TPR@1% FPR headline. Use >=200 / >=200 for a production audit.

## Limitations

- Small canary counts give wide CIs. Published audits use hundreds to thousands of canaries. v0.1 defaults are a CPU-friendly starting point, not a regulatory sample size.
- Thresholds are calibrated **on this run's controls** and do not transfer across model families.
- Real-record per-item flags are noisy (published AUC ~0.72-0.78 on honest fine-tunes). Believe the set-level test, not a single hash.
- Black-box, final-model audits are structurally loose. A small TPR is not a privacy certificate.
- The README demo **overfits on purpose** (canaries ~ 99% of tokens). Your production run should stay near the 0.1% token-budget target.
- Multi-seed mode measures **audit-procedure variance only** (bootstrap threshold calibration + real-record sampling); re-training across seeds is out of scope.
- DPO / GRPO / Hub model-card push / PII flagging / PANAME mapping are not in v0.1.
- LoRA-aware, not LoRA-only. Full fine-tunes need `--ref <base-checkpoint>` or explicit `--ref none`.
- `memaudit demo --lora` needs `memaudit[peft]` **and** a transformers `PreTrainedModel`. The checked-in demo is full FT on `TinyDemoLM`.

## Supported versions (verified on this machine)

| Piece | Buyer stack (LoRA bench) | Wheel install (clean venv) |
|---|---|---|
| Python | 3.12.11 | 3.12.11 |
| torch | 2.7.1 (PyPI, MPS) | 2.13.0 (PyPI, MPS) |
| transformers | 4.56.2 | 5.16.1 |
| peft | 0.20.0 | not installed (optional extra) |
| trl | 0.29.1 | not installed (optional extra) |
| datasets | 3.6.0 | 5.0.1 (pulled by `pip install` wheel) |

**Known-bad combo:** transformers 5.16.x + torch 2.6.dev hangs on FSDP imports (`CPUOffloadPolicy`). The hang is the *dev* torch, not 5.16 itself: a clean venv with transformers 5.16.1 + torch 2.13.0 imported and ran `memaudit doctor` here. Do **not** use `--system-site-packages` over a conda torch nightly. Recommended LoRA pin: `transformers==4.56.2` + `torch>=2.5,<2.8` + `peft==0.20.0`.

## Buyer acceptance

```bash
memaudit doctor --output-dir examples          # env + tiny demo + schema
# or, if a report already exists:
memaudit doctor --skip-demo --report examples/demo-report.json
bash scripts/acceptance.sh
```

The implementation module is `memaudit.injection`. The public helper remains `from memaudit import inject`.

## License

Apache-2.0. Local execution is the product; a SaaS re-host does not capture it.
