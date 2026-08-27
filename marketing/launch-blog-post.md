<!-- Launch blog post. Publish on launch day.
     TODO before publishing: confirm PyPI link live,
     Space URL after publish: https://huggingface.co/spaces/memaudit/memaudit-demo
     (org: https://huggingface.co/memaudit),
     add author byline. -->

# Your fine-tune remembers. Here's how to measure it.

**Introducing memaudit — a training-data memorization auditor for fine-tuned LLMs.
Apache-2.0, fully local, two verdicts with confidence intervals.**

```bash
pip install memaudit
```

If you fine-tune language models on data you couldn't post publicly — patient notes,
support tickets, contracts, internal wikis — you have a question you probably can't
answer today: *how much of that data is now inside the model?*

Until recently you could shrug. Two things changed. The science matured: we now know
fine-tuning is precisely the regime where memorization is measurable. And the regulatory
ground shifted: EDPB Opinion 28/2024 made documented testing against membership inference
and regurgitation the operative evidence standard for GDPR anonymity claims about models.

memaudit is a pip-installable Hugging Face Trainer/TRL plugin that answers the question
with numbers: plant pre-registered canaries in your fine-tuning data, train as usual, get
a versioned JSON report with two verdicts and confidence intervals. It runs entirely on
your machine. This post explains the design and the research behind it.

---

## 1. Why fine-tuning is the detectable regime

You may have read that membership-inference attacks (MIAs) "don't work" on LLMs. That's
the right conclusion from the wrong regime. The MIMIR study (Duan et al., 2024) showed
MIAs are near-random against *pretraining* — enormous deduplicated corpora seen for
roughly one epoch. That result is correct, and it does not transfer to fine-tuning.

Morris et al. (2025) quantified why: transformers store roughly **3.6 bits per
parameter**, and loss-based attack success is sigmoidal in the dataset-to-capacity ratio.
Pretraining sits far past the crossover (way more data than capacity); a typical
fine-tune — thousands to millions of tokens on a multi-billion-parameter model — sits far
before it. Small dataset, big model, multiple epochs: the signal is there.

The empirical record on actual fine-tunes agrees. LoRA-Leak (Ran et al., 2025) ran
fifteen attacks against honest 3-epoch LoRA fine-tunes of Llama-2-7B and got AUC
0.72–0.78 on real data. With designed canaries it gets much sharper: Meeus et al. (ICML
2025, "The Canary's Echo") reach **TPR@1%FPR of 0.94–0.99 under LoRA r=4**, and Panda et
al. (ICLR 2025, "Privacy Auditing of Large Language Models") detect canaries seen
effectively *once* at TPR@1%FPR 0.42–0.63.

Fine-tuning is where the leakage is measurable. It's also where your most sensitive data
tends to be. That's the niche memaudit occupies.

## 2. Two verdicts, because one number lies

The single most important design decision in memaudit: **every audit reports two separate
verdicts, and they are allowed to disagree.**

- **Membership leakage** — can an attacker with log-prob access tell whether a record was
  trained on? Measured by a base-model-calibrated Min-K%++ attack on canaries versus
  held-out canary controls, reported as **TPR at 1% FPR with a Clopper–Pearson 95%
  confidence interval**.
- **Regurgitation** — does the model *emit* training content when prompted with a prefix?
  Measured by greedy prefix-prompted completion over a prefix-length grid, scored in
  tiers: exact match, BLEU > 0.75, and sliding-window normalized edit distance ≤ 0.1.

Why both? Because the literature shows they routinely dissociate, in both directions:

- Models trained with the **goldfish loss** (Hans et al., 2024) drop verbatim extraction
  to near zero — while remaining highly detectable by likelihood attacks (>60% TPR for a
  zlib-calibrated MIA). Extraction-safe, membership-leaky.
- **Head-only fine-tuning** maximizes membership signal and exposure (Mireshghallah et
  al., 2022) while producing almost nothing extractable by sampling. Same dissociation.
- Conversely, weak *untargeted* attacks miss LoRA leakage that targeted prefix-prompting
  finds (Bossy et al., 2025 vs. untargeted baselines).

A loss-only auditor would have called goldfish models "leaky" and duplicated-data LoRA
models "safe" — wrong both times. memaudit never ships a one-number verdict.

And a scoring detail that decides whether the audit works at all: scores are computed on
the **secret span only**. Panda et al. show full-sequence scoring collapses detection
(TPR 0.424 → 0.010 in their gpt2-small new-token setting). Details like this are why we
think an audit tool should exist rather than a gist.

## 3. Canaries done right

Post-hoc MIA on your real records — with no planted ground truth — is exactly the setup
the research community has flagged as misleading (Aerni et al., 2024: "evaluations of ML
privacy defenses are misleading" without canary-based controls). memaudit is built the
other way around:

1. **`inject()` runs before training** and plants canaries into the raw dataset with
   per-canary Bernoulli inclusion coin flips, recorded in a manifest. Held-out canaries
   become negative controls — every threshold in the audit is calibrated on *this run's*
   controls, because memorization results demonstrably don't transfer across model
   families.
2. **The default family is high-perplexity regular-token canaries** — rejection-sampled
   from the base model into a target perplexity band, following the model-based-attack
   result in Meeus et al. (2025): detectability rises monotonically with canary
   perplexity. No vocabulary surgery, ever.
3. **Placement is format-aware and verified.** In TRL, completion-only and
   assistant-only loss masking mean a canary in the prompt or user turn is silently never
   trained on; packing can truncate or mask sequence starts. `inject()` refuses unsafe
   placements, and the callback re-scans tokenized batches at `on_train_begin` to prove
   each canary survived preprocessing.
4. **Repetition tiers {1, 4, 16}** give you a detection-versus-exposure curve, not a
   single point. Single-insertion canaries are interpreted as MIA-tier only — at one
   effective occurrence there's no verbatim signal to expect (Huang et al., 2024), though
   remember effective occurrences = insertions × epochs.
5. **Budget guardrails:** defaults target ≤0.1% of training tokens. At that budget,
   published audits report ≤1 perplexity point of utility cost (Panda et al., 2025).

For teams that want a differential-privacy-flavored number, there's an opt-in **ε lower
bound** via Steinke et al. (2023) one-run auditing — the coin-flip design maps onto it
directly — always printed next to its guess-count ceiling so a small ε_LB can't be
misread as "private."

## 4. The LoRA pre-flight: the part nobody else does

Most fine-tunes today are LoRA/PEFT fine-tunes, and PEFT configurations can silently
invalidate an audit. The sharpest published example: **new-token canaries** (the
strongest family under full fine-tuning) collapse when embeddings are frozen — Panda et
al.'s Table 14 shows the audit signal falling **0.74 → 0.05** — and standard LoRA freezes
embeddings.

So memaudit runs a **pre-flight router** before training:

- Detects embedding trainability across PEFT mechanisms and gates canary families
  accordingly — a router, not a blocker, because high-perplexity regular-token canaries
  demonstrably survive frozen-embedding LoRA (TPR@1%FPR 0.94+ at r=4, Meeus et al. 2025).
- Flags `modules_to_save=[embed_tokens, lm_head]` as *raising* the membership-risk tier —
  head-only fine-tuning is the published MIA-maximizing configuration.
- Uses PEFT's `disable_adapter()` so a single resident model provides both fine-tuned and
  base scores; detects merged adapters and `bias≠"none"` (where the toggle is unsafe) and
  falls back to a base checkpoint.
- Checks loss masking, packing mode, and `max_length` against canary placement.

As of our August 2026 landscape scan, no other tool — open-source, commercial, or
announced — addresses the frozen-embedding × canary-family interaction. This check
encodes a research synthesis (Panda, Meeus, LoRA-Leak, Bossy), which is exactly why it's
the moat.

The risk guidance is evidence-keyed too. The famous "LoRA rank 4 doesn't memorize, rank
1024 does" claim is **true only under ~10× duplication** (Bossy et al., 2025: exact-match
extraction 0.000 → 0.498 across ranks on duplicated clinical records; without duplication
all ranks stayed under 0.7%). So memaudit's recommendation engine keys on duplication ×
rank × learning rate × α × epochs — never rank alone — and its first recommendation is
almost always the strongest published mitigation: deduplication (~20× reduction in
emitted training text, Kandpal et al., 2022).

## 5. Honest statistics or nothing

Three commitments, all downstream of the post-2022 auditing literature (Carlini et al.,
2022, "Membership Inference Attacks From First Principles"):

1. **TPR at low fixed FPR is the headline, AUC is demoted.** Average-case AUC hides
   worst-case leakage; "we caught 42% of planted secrets at a 1% false-alarm rate" is a
   sentence an auditor can use.
2. **Confidence intervals always.** Clopper–Pearson on every rate. With 16 canaries the
   interval is embarrassingly wide — the report says so instead of hiding it.
3. **Refuse rather than fabricate.** Fewer than 100 held-out controls? No TPR@1%FPR
   headline — the field is `null` and `headline_valid=false`. Statistical power is a real
   constraint (published audits use 500–5,000 canaries); pretending otherwise would make
   the report worthless exactly where it matters.

On your real records (not canaries), per-record MIA is noisy — published AUC 0.72–0.78 on
honest fine-tunes — so memaudit reports a **set-level verdict** (Maini-style aggregation
with a p-value) and treats the ranked per-record list as exploratory, redacted by
default.

## 6. What the regulators actually ask for

None of this is hypothetical paperwork. EDPB Opinion 28/2024:

- **¶43**: for a model to be considered anonymous, the likelihood of extracting personal
  data *and* of obtaining it from queries must be "insignificant."
- **¶55**: supervisory authorities weigh the "scope, frequency, quantity and quality of
  tests," naming structured testing against "(i) attribute and membership inference; (ii)
  exfiltration; (iii) regurgitation of training data; (iv) model inversion; or (v)
  reconstruction attacks" — and stating that successful testing "can only be evidence for
  the resistance to those attacks."
- **¶58**: the controller's documentation should include the threat model, the measures
  that verified the lack of personal data, and "controls designed to limit or assess the
  success and impact of main attacks (regurgitation, membership inference attacks,
  exfiltration, etc.)."

memaudit's report is designed field-by-field against those paragraphs: an attack-coverage
table (membership inference and regurgitation covered; model inversion, reconstruction,
and attribute inference explicitly out of scope — under ¶55, naming what you didn't test
is itself the honest move), threat models per canary family, seeds and negative controls,
quantified results with CIs, release-context declaration, provenance hashes, and a
limitations statement quoting ¶55.

France's CNIL requires a documented model-status analysis and notes that a negative quick
regurgitation test alone proves nothing — which is precisely why memaudit pairs
generation testing with calibrated membership inference. And for the record: the EU AI
Act's high-risk obligations apply from **2 December 2027** (Annex III) and **2 August
2028** (Annex I embedded) under Regulation (EU) 2026/1744 — the AI Act never mandates
memorization testing by name. The EDPB opinion and CNIL guidance are the operative hooks.

**One thing memaudit will never claim: that it makes you compliant.** It produces test
evidence, honestly scoped. That's the whole promise.

## 7. What it does — and what it doesn't

In scope, v0.1: canary MIA with TPR@1%FPR + CI; prefix-prompted regurgitation
(exact/BLEU/NED tiers); LoRA/PEFT pre-flight router; set-level real-record testing;
negative controls always; post-hoc CLI (`memaudit audit --ref auto`); versioned JSON
report.

Out of scope, on purpose: model inversion, reconstruction, attribute inference (named as
untested in every report); shadow-model attacks like LiRA (64–4,000+ trainings — we cite
the numbers, we don't pay them); PII discovery; broad red-teaming; any SaaS anything.

Deployment-side scanners like garak solve a different problem — what your *deployed*
model says under public-corpus replay. memaudit generates training-side evidence about
*your* data. They compose nicely; a canary-export bridge to garak probes is on the
roadmap.

## 8. Measured numbers so far — honestly labeled

What exists today is an end-to-end **instrument check** on a deliberately-overfit tiny
model (randomly-initialized, hidden=64, vocab=256; canaries ≈ 99% of tokens by design so
the instrument can show a positive signal):

| Metric | Value | Scale label |
|---|---|---|
| TPR @ 1% FPR | 1.000, CI95 [0.794, 1.000] | tiny-model overfit check, 16 canaries / 100 controls |
| Regurgitation at 16× | 15/16 = 0.9375 | same |
| Negative-control regurgitation | 0.000 (n=100) | same |
| Wall clock | ~30 s end-to-end on a laptop | same |

These numbers prove the pipeline detects what it should and stays silent on controls.
They are **not** production-scale claims. A production-scale LoRA benchmark (pretrained
multi-billion-parameter model, ≥200 inserted + ≥200 held-out canaries, repetition grid
{1, 4, 16}, multi-seed) is being produced now:

<!-- BENCHMARK-TABLE: filled from benchmarks/ when available -->

Designed compute envelope for a production audit (7B-class, ~200 canaries + 400 controls
+ 2,000 real records): **5–20 minutes on a single 24 GB GPU** — two forward passes per
sequence, no shadow models. That is an estimate until the benchmark lands, and we'll
replace it with measurements the day it does.

## 9. Roadmap

- **Production LoRA benchmark** — replaces the placeholder above.
- **Frozen-vs-trainable-embeddings ablation per canary family** — no published non-DP
  result exists; this doubles as a short paper.
- **Multi-seed mode** with variance reporting.
- **EDPB/CNIL report annex export** + PII flagging on exposed content (redaction stays
  the default).
- **PANAME alignment** when the CNIL/ANSSI/Inria library releases (autumn 2026):
  vocabulary and attack-taxonomy mapping — PANAME-compatible evidence, produced where you
  already train.
- **Preference-stage re-scoring**: SFT-injected canaries re-audited after DPO — DPO-family
  objectives memorize preference data at SFT-like rates (18–19%, Pappu et al., 2024).
- **garak probe export** for deployment-side replay of your canaries.

## 10. Try it

```bash
pip install memaudit
memaudit demo        # the tiny instrument check from §8, on your machine
```

Five lines in your training script:

```python
from memaudit import generate_canaries, inject, MemorizationAuditCallback

canaries = generate_canaries(tokenizer, n=200, n_controls=200, family="high_ppl", seed=0)
train_ds, manifest = inject(train_ds, canaries, fmt="auto", seed=0)
trainer.add_callback(MemorizationAuditCallback(trainer=trainer, manifest=manifest))
trainer.train()   # writes <output_dir>/memaudit-report.json
```

- Code: `https://github.com/mem-audit/memaudit`
- PyPI: `https://pypi.org/project/memaudit/`
- Hugging Face: `https://huggingface.co/memaudit`
- Report demo Space (after publish): `https://huggingface.co/spaces/memaudit/memaudit-demo`

If you fine-tune on regulated or confidential data and want to stress the PEFT pre-flight
against your real pipeline, we're onboarding a small number of design partners:
`ansh.singh.160305@gmail.com`.

*memaudit produces test evidence; it does not make you compliant. Every number above
carries its scale label on purpose. Apache-2.0 — run it where your data lives.*

---

<!-- Reference list for editors (papers cited by name above):
  Morris et al. 2025 (arXiv:2505.24832) — capacity ≈3.6 bits/param
  Duan et al. 2024 — MIMIR
  Ran et al. 2025 — LoRA-Leak (15-attack head-to-head; base-calibrated Min-K%++ wins)
  Meeus et al. ICML 2025 — The Canary's Echo (arXiv:2502.14921)
  Panda et al. ICLR 2025 — Privacy Auditing of LLMs (arXiv:2503.06808; Table 14 frozen-embedding collapse; secret-span scoring)
  Hans et al. 2024 — Goldfish loss
  Mireshghallah et al. 2022 — head-only FT maximizes MIA
  Bossy et al. 2025 (arXiv:2502.05087) — LoRA rank × duplication
  Kandpal et al. 2022 — dedup ~20×
  Huang et al. 2024 — single-occurrence verbatim memorization
  Aerni et al. 2024 — misleading privacy evaluations without canary controls
  Carlini et al. 2022 — LiRA / TPR at low FPR
  Steinke et al. 2023 — one-run ε auditing
  Maini et al. 2024 — set-level dataset inference
  Pappu et al. 2024 — DPO-family memorization
  Ippolito et al. 2022 — BLEU>0.75 approximate memorization; filter bypass
  Zeng et al. ACL 2024 — task type & epoch curves
-->
