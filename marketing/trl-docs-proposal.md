<!-- Proposal for a TRL community-docs example.
     Part A is the draft PR description; Part B is the draft doc page itself.
     TODO before opening the PR: confirm the current TRL docs
     layout (docs/source/community_tutorials or equivalent) and follow their contribution
     guide; numbers marked [BENCHMARK] should be updated if benchmarks/ has landed. -->

# TRL community-docs proposal: memorization auditing for SFT fine-tunes

## Part A — draft PR description

**Title:** `docs: community example — auditing SFT runs for training-data memorization with memaudit`

### What this PR adds

One community-example doc page showing how to measure training-data memorization
(membership leakage + regurgitation) in an `SFTTrainer` run using
[memaudit](https://github.com/mem-audit/memaudit), an Apache-2.0,
fully-local auditing plugin. No changes to TRL code; docs only.

### Why TRL users specifically

Memorization auditing interacts with TRL internals in ways that generic scripts get
silently wrong, so the guidance is TRL-shaped by nature:

- `completion_only_loss` / `assistant_only_loss` set `labels=-100` on the prompt/user
  side — an audit canary placed there is never trained on, and a naive audit measures
  noise and reports "safe."
- Packing can truncate or mask sequence starts; canary records must stay under
  `max_length` and scoring must skip the first packed token.
- `SFTTrainer.__init__` tokenizes/masks/packs before any callback hook fires, which is
  why injection is a pre-train helper and not a callback feature.
- PEFT fine-tunes (the TRL common case) need an embedding-trainability check before
  choosing canary families, and `disable_adapter()` gives reference-model scores for
  free on unmerged adapters.

These are exactly the questions that appear in TRL issues/forums when people attempt
ad-hoc memorization checks; a documented, correct pattern seems more useful than
repeated one-off answers.

### Why this is worth documenting at all

Fine-tuning on non-public data increasingly comes with a documentation expectation
(EDPB Opinion 28/2024 names membership-inference and regurgitation testing; CNIL asks
for a documented model-status analysis). Beyond compliance, the numbers are useful
engineering signals: repetition-tier regurgitation curves catch dataset-duplication
problems early.

### Scope & maintenance

- Doc page only; example runs on a small model in a few minutes for CI-friendliness, with
  an explicit note that small-scale numbers are instrument checks, not production
  evidence.
- I'm the author of memaudit (disclosure) and will maintain the page against TRL API
  changes; the tool's CI pins and tests against current TRL releases.
- If maintainers prefer, this works equally well as a shorter entry in the community
  tutorials list linking out to the full guide.

### Checklist

- [ ] Follows TRL docs style and toctree registration
- [ ] Code blocks runnable top-to-bottom on a single GPU / CPU-small config
- [ ] No claims beyond what the linked tool's report actually contains
- [ ] Disclosure of authorship included

---

## Part B — draft doc page

# Auditing an SFT run for training-data memorization

*Community example. [memaudit](https://github.com/mem-audit/memaudit) is a
third-party, Apache-2.0, fully-local tool maintained by its author; it is not part of
TRL.*

When you fine-tune on data you couldn't post publicly, two questions are worth answering
with numbers rather than vibes:

1. **Membership leakage** — can an attacker with log-prob access tell whether a specific
   record was in the training set?
2. **Regurgitation** — does the model emit training content when prompted with a prefix?

These are different harms and they routinely disagree (a model can be extraction-safe yet
highly detectable by likelihood attacks, and vice versa), which is why the audit below
reports both.

## Install

```bash
pip install memaudit          # core
pip install "memaudit[peft]"  # if you fine-tune with LoRA/PEFT
```

## Step 1 — inject canaries *before* building the trainer

Canaries are pre-registered synthetic records planted in the raw dataset with recorded
coin flips; the held-out half becomes negative controls. Injection must happen before
`SFTTrainer.__init__`, because TRL tokenizes, applies loss masking, and packs at
construction time — after that, no callback can add training data.

```python
from datasets import load_dataset
from memaudit import generate_canaries, inject

train_ds = load_dataset("json", data_files="train.jsonl", split="train")

canaries = generate_canaries(
    tokenizer,
    n=200,             # insert-eligible canaries (coin-flipped in/out)
    n_controls=200,    # never inserted: calibrate thresholds + false-positive floor
    family="high_ppl", # LoRA-safe default; no vocab resizing, ever
    repetitions=(1, 4, 16),
    seed=0,
)
train_ds, manifest = inject(train_ds, canaries, fmt="auto", seed=0)
```

`inject()` is format-aware: the secret always lands on the trainable side of the record
(`completion`, assistant turn, or `text` body). With `completion_only_loss=True` or
assistant-only chat templates, a prompt-side canary would be `labels=-100` — trained-on
never, "detected" never — and the audit would silently measure nothing. `inject()`
refuses that placement.

## Step 2 — add the callback

```python
from trl import SFTConfig, SFTTrainer
from memaudit import MemorizationAuditCallback

trainer = SFTTrainer(
    model=model,
    train_dataset=train_ds,
    args=SFTConfig(output_dir="./out", num_train_epochs=3),
    peft_config=peft_config,          # optional; the pre-flight adapts either way
)
trainer.add_callback(
    MemorizationAuditCallback(trainer=trainer, manifest=manifest, real_sample=2000)
)
trainer.train()   # writes ./out/memaudit-report.json
```

At `on_train_begin` the callback runs a PEFT pre-flight and a canary survival scan:

- **Embedding trainability** decides which canary families are valid (new-token canaries
  collapse under frozen embeddings; the default `high_ppl` family survives LoRA).
- `modules_to_save=["embed_tokens", "lm_head"]` raises the reported membership-risk tier.
- Packing mode, `max_length`, and loss masking are checked against every canary's actual
  tokenized placement — the scan proves each canary survived preprocessing.

At `on_train_end` it runs the audit in-process (eval mode, unwrapped model). Under
ZeRO-3/FSDP it writes a deferred-audit manifest plus the exact CLI command instead of
attempting in-callback forwards.

## Step 3 — read the report

`memaudit-report.json` (versioned schema) contains, among other fields:

| Field | Meaning |
|---|---|
| `membership.tpr_at_1pct_fpr` | Detection rate on inserted canaries at 1% FPR, thresholded on *held-out* canaries; `null` (never fabricated) when controls < 100 |
| `membership.ci_low / ci_high` | Clopper–Pearson 95% interval |
| `regurgitation.by_tier` | Prefix-prompted regurgitation rate at repetition 1× / 4× / 16× (exact, BLEU>0.75, sliding-window NED≤0.1) |
| `negative_controls` | Never-inserted canary results — the audit's false-positive floor |
| `real_records.set_level` | Aggregate t-test on a sample of your real rows (per-record list is exploratory and redacted) |
| `recommendations` | Ordered, evidence-keyed mitigations (dedup → fewer epochs → cooler LoRA → …) |

Scores are computed on the secret span only, with a base-calibrated Min-K%++ attack — the
published head-to-head winner on LoRA fine-tunes at the cost of two forward passes per
sequence.

## Post-hoc CLI (already-trained models, ZeRO-3/FSDP deferred audits)

```bash
memaudit audit --model ./out \
    --canary-set ./out/memaudit-manifest.json \
    --dataset train.jsonl --ref auto
```

`--ref auto` uses `disable_adapter()` on an unmerged LoRA so one model copy scores both
fine-tuned and base; if the adapter was merged (or `bias != "none"`), pass a base
checkpoint path instead.

## Honest limitations

- Small canary counts give wide confidence intervals; production audits should use
  ≥200 inserted + ≥200 held-out canaries (published audits use 500–5,000).
- Results are calibrated per-run on this run's controls; they do not transfer across
  model families.
- The audit covers membership inference and regurgitation. It does not test model
  inversion, reconstruction, or attribute inference, and the report names these as
  untested.
- An audit report is evidence about the attacks it ran — it is not a compliance
  certificate, and the tool never claims otherwise.
