# memaudit — Disclaimer

*Plain-language summary of what memaudit's results do and do not mean. This document is
not legal advice (see §7). Last updated: August 2026.*

## 1. Evidence, not compliance

memaudit produces **documented test evidence** about a specific model: how it responded
to specific membership-inference and regurgitation tests, on a specific run, with
recorded seeds, canaries, and thresholds.

It does **not** make you, your model, or your organization compliant with the GDPR, the
EU AI Act, CNIL guidance, or any other law or regulation. No test report can. Whether a
model may lawfully be trained, deployed, or shared depends on your whole processing
context — lawful basis, purposes, DPIAs, contracts, and decisions that belong to you and
your advisors, not to a testing tool. A "clean" memaudit report is one input to that
analysis, never the conclusion of it.

## 2. Results are attack-specific

memaudit tests two attack classes: **membership inference** (likelihood-based, on planted
canaries and sampled records) and **regurgitation** (prefix-prompted generation). It does
not test model inversion, reconstruction, attribute inference, or attacks invented after
your audit ran.

This limitation is not fine print — it is how the primary regulatory text works. EDPB
Opinion 28/2024 ¶55 states that successful testing covering state-of-the-art attacks
"can only be evidence for the resistance to those attacks." A memaudit report is evidence
about the attacks it actually ran, at the time it ran them, on the exact model and
configuration it ran against. New attacks, different checkpoints, merged adapters,
further training, or different decoding settings can all change the picture. Absence of
detected leakage is not proof of anonymity.

## 3. Statistical results, not certainties

Verdicts come with confidence intervals because they are estimates. Small canary counts
give wide intervals; thresholds are calibrated on each run's own controls and do not
transfer across models or model families. Where the sample is too small for a defensible
headline number, memaudit reports no number rather than a fabricated one. Read the
intervals, not just the point estimates.

## 4. Redaction defaults and report handling

By default, memaudit **redacts**: the per-record risk list identifies records by hash,
not content, and flagged text is not printed unless you explicitly opt in (`--reveal`).
Two responsibilities stay with you:

- If you disable redaction, the report may contain fragments of your training data —
  including personal or confidential data the audit was designed to detect. Handle such
  reports with the same care as the training data itself.
- Even redacted reports contain metadata (scores, hashes, configuration, canary text).
  Treat reports as internal documents by default and review before sharing externally.

## 5. Your data, your responsibility

memaudit runs locally and sends nothing anywhere, which also means it cannot check what
you feed it. You remain responsible for the lawfulness of your data collection and
processing, for having a lawful basis to train, for honoring data-subject rights, and for
any decisions taken based on audit results. Injecting canaries modifies your training
dataset (within a documented token budget); verifying that this is acceptable for your
use case is your call.

## 6. No warranty

memaudit is open-source software provided under the **Apache License 2.0**, which governs
in full. In short (the license text controls): the software is provided **"AS IS",
without warranties or conditions of any kind**, express or implied — including
merchantability, fitness for a particular purpose, and non-infringement — and
contributors are **not liable** for damages arising from its use, to the extent permitted
by law. See the `LICENSE` file for the complete terms.

## 7. Not legal advice

Nothing in memaudit, its reports, its documentation, or this disclaimer is legal advice,
and using memaudit creates no advisor–client relationship of any kind. Regulatory
citations in the report (EDPB Opinion 28/2024, CNIL guidance, and others) are provided to
make the report legible to your advisors — not to replace them. **Before relying on
memaudit reports in enterprise contracts, regulatory filings, or any compliance
representation, have qualified counsel review both the report and this disclaimer.**
