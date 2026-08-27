<!-- Design-partner one-pager. Shareable as PDF/email after founder review.
     PRICING BELOW IS A SUGGESTION SET for the founder to decide — do not send
     externally until the founder has replaced the bracketed ranges with real numbers.
     TODO: replace repo links before sending. -->

# memaudit — design partner program

**memaudit is pytest for training-data privacy:** plant pre-registered canaries in your
fine-tuning data; get two verdicts with confidence intervals — membership leakage and
regurgitation — in a versioned JSON report designed against EDPB Opinion 28/2024 ¶55/¶58.
Apache-2.0. Runs entirely inside your infrastructure; nothing leaves your machines.

We are onboarding **3–5 design partners** ahead of general availability.

---

## Who this is for

Teams that fine-tune LLMs on data they'd have to explain to a regulator or a customer:

- **Regulated fine-tuners** — healthcare, financial services, legal, insurance, public
  sector — using Hugging Face Trainer / TRL (full fine-tunes or LoRA/PEFT).
- **Platform/ML teams** that operate fine-tuning for internal customers and need a
  standard, documentable test gate in the pipeline.
- The **privacy/compliance owner** attached to either: the report is written to be
  filed, not just read — attack coverage, threat models, seeds, negative controls,
  provenance hashes, limitations.

You're a fit if: you run fine-tunes on confidential/personal data, you can run a pilot
audit on at least one real (or realistic) training pipeline, and someone on your side
owns the "prove it" question.

## What a pilot includes

1. **Guided integration** — we wire `inject()` + the audit callback into your existing
   training loop with you (typically < 1 day of your engineers' time).
2. **Custom canary families** — tuned to your data formats (chat templates, structured
   records, domain text), with placement verified against your masking/packing config.
3. **Multi-seed runs** — repeat audits across seeds with variance reporting, so one lucky
   run can't mislead anyone.
4. **Report review session** — a working session with your compliance/security owner
   walking through the report and its regulatory mapping (what it evidences, what it
   explicitly does not).
5. **Direct engineering channel** — priority fixes; your PEFT configs become test cases.
6. **Roadmap priority** — design-partner needs get scheduled first.

## What we ask from you

- **Structured feedback** — especially where the PEFT pre-flight router meets your real
  configs (this is the part we most want hardened).
- **Case-study rights (negotiable)** — an anonymized write-up of the pilot; logo usage
  only with your explicit sign-off. We'll trade depth of support for reference value.
- **Realistic engagement** — a named engineer and a named compliance/security contact,
  and one pilot audit run within the first two weeks.

## Suggested pilot structure (6 weeks)

| Week | Activity |
|---|---|
| 0 | Scoping call: pipeline, data formats, PEFT config, what "success" means for your compliance owner |
| 1–2 | Integration + first end-to-end audit on a real training run |
| 3–4 | Custom canary families; multi-seed runs; iterate on findings |
| 5 | Report review with compliance/security; written findings memo |
| 6 | Retro: case-study draft (if agreed), decision on ongoing support |

## Pricing — SUGGESTIONS ONLY (founder to decide)

> **Status: these tiers and numbers are internal suggestions, not offers.** The grounding
> principle is fixed — the OSS core is free forever (Apache-2.0); money buys people-time,
> priority, and enterprise features — but the founder sets final prices before anything
> is quoted externally.

| Tier | What it is | Suggested shape |
|---|---|---|
| **Open source** | Everything in the repo: two-verdict audits, PEFT pre-flight, CLI, report | Free, forever |
| **Design-partner pilot** | The 6-week program above | One-time fixed fee, suggested range **[$7,500–$20,000]** depending on scope (number of pipelines, custom canary work); consider **discount-to-free** for one flagship regulated-industry partner in exchange for a strong public case study |
| **Support & enterprise** (post-pilot) | Ongoing: named-contact support with SLA, audit-report reviews per release cycle, custom canary-family maintenance, multi-seed infrastructure help, early access to the EDPB/CNIL annex export and PANAME-alignment features | Annual subscription, suggested range **[$25,000–$60,000/yr]** by org size and audit cadence |

Not for sale at any tier: compliance promises. memaudit produces documented test
evidence; it does not make anyone compliant, and pilot materials say so explicitly.

---

**Contact:** `ansh.singh.160305@gmail.com`
**Code:** `https://github.com/mem-audit/memaudit` · PyPI: `memaudit` · HF: `https://huggingface.co/memaudit` · License: Apache-2.0
