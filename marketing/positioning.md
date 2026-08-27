# memaudit — internal positioning cheat sheet

**Internal only. Not marketing copy.** Claims here were sourced from the private
research archive (landscape notes + product plan). If you say something not on this
sheet, verify it against that archive first.

---

## 1. Elevator pitches

### 10 seconds

> memaudit is pytest for training-data privacy: it plants pre-registered canaries in your
> fine-tuning data, then measures — with confidence intervals — whether the trained model
> leaks membership or regurgitates training text. Free, Apache-2.0, runs entirely where
> you train.

### 30 seconds

> Fine-tuning is exactly the regime where training-data leakage is detectable — small
> datasets, multiple epochs, big models. memaudit is a pip-installable Hugging Face
> Trainer/TRL plugin that injects coin-flipped canaries before training and reports two
> verdicts after: membership leakage (TPR at 1% FPR with a Clopper–Pearson interval) and
> regurgitation (prefix-prompted, scored exact/BLEU/edit-distance). It's LoRA-aware by
> construction — a PEFT pre-flight router no other tool has — and the JSON report is
> shaped against EDPB Opinion 28/2024 ¶55/¶58, the paragraphs that made this testing the
> GDPR evidence standard. Fully local, no phone-home, Apache-2.0.

### 2 minutes

1. **The problem.** Fine-tuned models absorb training data. Regulators now expect
   documented testing: EDPB Opinion 28/2024 ¶55 names membership inference and
   regurgitation testing; ¶58 describes the documentation file; CNIL requires a
   documented model-status analysis producible on demand. Meanwhile the tooling gap is
   total: nothing on GitHub or PyPI does trainer-integrated, canary-based, PEFT-aware
   auditing (verified Aug 2026).
2. **Why it works scientifically.** Pretraining-scale MIA pessimism (MIMIR) does not
   apply here: model capacity is ~3.6 bits/param (Morris 2025) against tiny fine-tuning
   sets, so likelihood signals are strong. Published fine-tune audits reach TPR@1%FPR
   0.94–0.99 with high-perplexity canaries under LoRA r=4 (Meeus 2025).
3. **What it does.** `inject()` plants pre-registered, coin-flipped canaries in the raw
   dataset (placement-verified against loss masking and packing). A thin callback runs a
   PEFT pre-flight at train start and the audit at train end; a CLI runs the same engine
   post-hoc. Two verdicts always: base-calibrated Min-K%++ membership at TPR@1%FPR with
   CI, and prefix-prompted regurgitation per exposure tier. Negative controls in every
   audit; per-record output redacted by default.
4. **The moat.** The LoRA/PEFT pre-flight router: frozen embeddings silently kill
   new-token canary audits (0.74 → 0.05, Panda ICLR 2025); memaudit detects embedding
   trainability and routes canary families before training. Nobody else — commercial or
   OSS — addresses this.
5. **The artifact.** A versioned, hash-stamped JSON report mapped onto EDPB ¶55/¶58:
   attack coverage (with out-of-scope attacks named), threat models per canary family,
   seeds, controls, quantified results, limitations statement. Evidence — never a
   compliance promise.
6. **Cost.** Two forward passes per sequence (target + base), no shadow models.
   Measured wall-clock on Apple MPS / distilgpt2 is in `benchmarks/README.md`. Canary
   budget ≤0.1% of tokens target; ≤1 PPL point published utility cost at that budget.

---

## 2. Objection handling

### "Dynamo AI already sells this."

- Positioning line: **"the canary-grid audit enterprises pay for — free, in your Trainer
  callback, with TPR-at-fixed-FPR statistics instead of AUC-first reporting, and the LoRA
  pre-flight nobody else has."**
- Factual contrasts (all from their public materials, Aug 2026):
  - Closed, custom-quote, SaaS/VPC — you upload model + dataset. memaudit is free,
    Apache-2.0, fully local.
  - Post-hoc attacks on real records; **no published canary methodology**. Post-hoc MIA
    without planted controls is exactly what Aerni et al. 2024 showed to be misleading.
  - AUC/ROC-first reporting; the post-2022 research consensus (Carlini LiRA) is TPR at
    low fixed FPR — average-case AUC hides worst-case leakage.
  - LoRA handled **only as an upload format** ("provide the file path to the PEFT adapter
    configuration"). Nothing on frozen embeddings, rank-dependent risk, or PEFT-aware
    test design.
- **Fair-play (don't skip):** they cover PII extraction/inference tests we don't; we
  can't rule out unadvertised features. By 2026 privacy testing is one pillar among many
  for them (hallucination, prompt injection, agent security) — we attack the original
  wedge, not their center of gravity.
- **FORBIDDEN:** "Fortune 1000 banks" as a Dynamo fact — unverified. Safe phrasing:
  *"the category of test Dynamo AI sells to regulated enterprises, including financial
  institutions."*

### "Why not wait for a regulator library (e.g. PANAME)?"

- Regulator libraries validate the thesis: authorities want documented membership and
  regurgitation evidence.
- Public materials describe post-hoc libraries — not trainer-integrated canary injection
  or PEFT-aware pre-flight. That workflow layer is where memaudit lives today.
- Line: **"complement — evidence produced automatically where you already train,"** with
  report vocabulary designed to stay alignable with emerging regulator taxonomies.
- Never trash regulator work; compatibility is a distribution asset, especially in the EU.

### "We'll just dedup our data."

- Yes — dedup first. It's the single strongest mitigation (~20× reduction in emitted
  training text, Kandpal 2022) and memaudit's own recommendation engine puts it at the
  top.
- But dedup is a **mitigation, not a measurement**. Three gaps:
  1. Leakage exists without duplication: membership signal doesn't need verbatim copies —
     goldfish-trained models show ~0 extraction but >60% TPR zlib-MIA (Hans 2024);
     head-only fine-tunes maximize MIA with ~0 extraction (Mireshghallah 2022). MIA-class
     leakage survives dedup.
  2. Epochs multiply effective occurrences (dialogue memorization ×5 from 1→10 epochs,
     Zeng 2024) — you can re-create "duplication" without duplicate rows.
  3. The regulatory duty is **documented testing** (EDPB ¶55/¶58; CNIL: a negative quick
     test alone doesn't discharge the duty). "We deduped" is a control, not evidence.
- Close: "Great — dedup, then run the audit that proves it worked."

### "MIAs don't work on LLMs" (the MIMIR objection)

- Agree with the paper, narrow its scope: MIMIR (Duan 2024) shows near-random MIA **at
  pretraining scale** — massive deduplicated corpora, ~1 epoch. Correct and cited in our
  docs.
- Fine-tuning is the opposite regime: small datasets on high-capacity models
  (~3.6 bits/param, Morris 2025 — MIA success is sigmoidal in dataset-to-capacity ratio
  and near-ceiling for typical fine-tunes).
- Empirics on actual fine-tunes: LoRA-Leak (Ran 2025) gets AUC 0.72–0.78 on honest
  3-epoch LoRA fine-tunes with 15 attacks; Meeus 2025 canaries reach TPR@1%FPR 0.94–0.99
  under LoRA r=4; Panda ICLR 2025 detects single-exposure canaries (TPR@1%FPR 0.42–0.63).
- And structurally: memaudit never relies on uncalibrated scores — every audit thresholds
  on its own held-out controls and reports Clopper–Pearson CIs. If there's no signal, the
  report says "no detection," it doesn't invent one.

### "We early-stopped / we only train 1 epoch, so we're safe."

- Memorization starts in epoch 1 and peaks at the val-loss minimum (Carlini 2019 exposure
  curves; Tirumala 2022) — early stopping is not a safety claim, which is why memaudit's
  recommendation engine says "fewest epochs to target utility" and never "early stop."

### "We'll put an output filter in front."

- Filters are deployment-phase extras (EDPB ¶107), not evidence about the model, and
  n-gram filters are bypassed by style-transfer prompting (Ippolito 2022). Fine as
  defense-in-depth; useless as the documented model-status analysis.

### "Why not DP-SGD instead of auditing?"

- DP gives worst-case guarantees at a utility/complexity cost; most teams won't pay it.
  memaudit measures empirical leakage on the run you actually shipped, recommends DP-SGD
  when risk is found, and (opt-in) produces a Steinke one-run ε lower bound — printed
  with its guess-count ceiling so it can't be misread as a privacy certificate. Auditing
  and DP compose; they don't compete.

---

## 3. Verified regulatory hooks (recite these exactly)

| Hook | What it says | How we use it |
|---|---|---|
| **EDPB Opinion 28/2024 ¶43** | Model anonymity requires that likelihood of (i) extraction of personal data and (ii) obtaining it from queries be **"insignificant"**; SAs should by default expect a thorough evaluation | The reason testing exists at all |
| **¶55 (the testing paragraph)** | SAs consider "scope, frequency, quantity and quality of tests"; successful testing "can only be **evidence for the resistance to those attacks**"; names (i) attribute & **membership inference**, (ii) exfiltration, (iii) **regurgitation**, (iv) model inversion, (v) reconstruction | We cover (i)-membership and (iii); we name the rest out of scope — honesty is itself the compliance feature |
| **¶58 (the documentation paragraph)** | Controller's file: threat model & risk assessments (c); measures that "verified the lack of personal data" (d); documentation of resistance + "controls designed to limit or assess the success and impact of main attacks (regurgitation, membership inference attacks, exfiltration, etc.)" (e) | The report *is* this file's technical annex |
| **¶46** | Testing level depends on release context (public vs internal) | Report's release-context field |
| **CNIL model-status analysis** | Documented analysis that re-identification likelihood is "insignificant"; usually requires re-identification attack tests; **a negative quick regurgitation test alone proves nothing**; producible on demand | National implementation manual for ¶55/¶58; memaudit implements the spec |
| **AI Act (Reg. (EU) 2024/1689 as amended by Reg. (EU) 2026/1744)** | High-risk: Annex III **2 Dec 2027**, Annex I embedded **2 Aug 2028**; GPAI obligations since 2 Aug 2025; never mandates memorization testing by name | Supporting context only — EDPB/CNIL are the operative hooks |
| **NIST AI 600-1** (US) | Names "data memorization" as a GenAI privacy risk | US framing; no testing mandate |

### Forbidden claims (hard rules)

1. ~~"AI Act high-risk deadline August 2026"~~ — the date moved to Dec 2027 / Aug 2028.
2. Do not invent customer logos or "Fortune 1000" claims.
3. ~~"memaudit makes you compliant"~~ — it produces test evidence; it does not make you
   compliant. Always.
4. ~~Demo numbers without the scale label~~ — TPR 1.000 [0.794, 1.000], 15/16
   regurgitation, 0/100 controls are from a deliberately-overfit tiny randomly-initialized
   model. The label travels with the number, every time.
5. ~~"guarantees privacy" / "proves anonymity"~~ — attack-specific evidence only (¶55).
6. Cite measured wall-clock from `benchmarks/README.md` (distilgpt2 / MPS); do not sell unmeasured 7B envelopes.

---

## 4. One-line answers for speed

- **What is it?** pytest for training-data privacy — canaries in, two verdicts out.
- **Who's it for?** HF/TRL fine-tuners in regulated settings, and the compliance owners
  who have to file evidence.
- **Why now?** EDPB 28/2024 made the testing the GDPR evidence standard; the niche is
  empty (verified Aug 2026) — ship the trainer-integrated category with local tooling.
- **Why you?** The audit encodes the research synthesis (Panda/Meeus/LoRA-Leak/Bossy):
  the PEFT router, secret-span scoring, control calibration — the parts that are easy to
  get silently wrong.
- **Business?** Free OSS core forever; paid design-partner pilots and enterprise support
  (see `design-partner-onepager.md` for the pilot ask; quote from email).
