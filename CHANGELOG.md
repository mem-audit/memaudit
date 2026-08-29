# Changelog

## Unreleased

## 0.2.0 - 2026-08-30

Review-response release: preflight evidence, audit profiles, PEFT matrix, scorer seam, schema 1.2.0.

- Report schema **1.2.0** (additive on 1.1.0): `audit_profile`, `membership.by_repetition`, `membership.calibration_stability`, `membership.scorer`, canary `requested_family` / `actual_generator`, regurgitation protocol fields, real-record inferential-vs-descriptive split.
- Named audit profiles (`smoke`, `routine`, `powered`) with repetition-tier decomposition and calibration-stability reporting on powered runs.
- Fail-closed preflight: collator-based label verification, per-canary evidence levels, PEFT adapter-semantics matrix, and live TRL integration tests in both label eras (TRL 0.29.x mask era and TRL >=1.9 labels-column era).
- Membership scoring routed through a pluggable scorer seam (default `min_k_plus_plus`).
- Live QLoRA integration tests against a real bitsandbytes 4-bit base.
- Powered flagship case study re-run with genuinely model-scored `high_ppl` canaries (base TinyLlama rejection sampling; `actual_generator=model_scored_high_ppl`, 300/300 in the target perplexity band). Honest **0.874%** budget (22,228 / 2,521,431 tokens). Measured TPR@1%FPR **0.100** (10/100, CI [0.049, 0.176]), AUC 0.837, tier curve 1x 0/34 · 4x 1/33 · 16x 9/33, regurgitation 0/100. Report `examples/alpaca-powered-report.json`. The prior uniform_vocab run is archived as `examples/alpaca-powered-report-v0.1-uniformvocab.json` (TPR **0.180**, CI [0.110, 0.269]).
- `high_ppl` canary generation: base model can be passed for real rejection sampling, perplexity scored in float32, per-draw MPS allocator cache release (fixes multi-hour generation stalls on Apple Silicon), and explicit generator provenance (`model_scored_high_ppl` vs `uniform_vocab`).

## 0.1.0 - 2026-08-27

First buyer-facing release.

- Published on PyPI: [`pip install memaudit`](https://pypi.org/project/memaudit/). Optional extras: `peft`, `trl`, `hub`, `dev` (combine as `"memaudit[peft,trl]"`).
- Two-verdict audit: membership (base-calibrated Min-K%++ on the secret span) and regurgitation (prefix-prompted exact / BLEU / NED).
- Pre-train `inject()` + `MemorizationAuditCallback` + shared `run_audit` / `memaudit audit`.
- `--canary-set` and `--manifest` are the same flag (inject() JSON).
- TPR@1% FPR is refused unless there are at least 100 held-out controls. Default `n_controls` is 100.
- `--ref auto` uses `disable_adapter()` on an unmerged LoRA. Full fine-tunes must pass `--ref <base>` or explicit `--ref none` (downgraded headline). No silent fallback.
- `from memaudit import inject` is the public helper; the implementation module is `memaudit.injection` (no submodule shadowing).
- Reports are strict JSON (NaN/Inf become null). `phone_home` is always false.
- Powered flagship case study: TinyLlama-1.1B-Chat + **20,000** Stanford Alpaca rows, LoRA r=8, 1 epoch, **100 / 200** canaries, honest **0.907%** budget. Measured TPR@1%FPR **0.180** (18/100, CI [0.110, 0.269]), AUC 0.776, regurgitation 0/100. Canaries used the `uniform_vocab` fallback (no model passed at generation); report archived as `examples/alpaca-powered-report-v0.1-uniformvocab.json`. The n=12 first look remains as an appendix.
- Flagship public case study: TinyLlama-1.1B-Chat + 5,000 Stanford Alpaca rows, LoRA r=8, honest 0.39% canary budget. Script `examples/alpaca_case_study.py`, report `examples/alpaca-case-study-report.json`, write-up `docs/case-study-alpaca.md`. Measured TPR@1%FPR 0.500 (6/12, CI [0.211, 0.789]), AUC 0.880, regurgitation 0/12.
- `memaudit demo` / `python examples/demo.py`: TinyDemoLM positive-control validation with measured numbers.
- `memaudit doctor` / `scripts/acceptance.sh`: environment + report schema acceptance.
- Known-bad stack: transformers 5.16.x + torch 2.6.dev (FSDP import hang). Verified LoRA stack: Python 3.12, torch 2.7.1, transformers 4.56.2, peft 0.20.0, trl 0.29.1. Clean wheel install also resolved torch 2.13.0 + transformers 5.16.1.

Compliance evidence layer (report schema 1.1.0, additive on 1.0.0):

- `compliance_annex` in every report: EDPB Opinion 28/2024 mapping — attack-coverage table (membership inference para 55(i) and regurgitation para 55(iii) in scope; attribute inference, exfiltration para 55(ii), inversion para 55(iv), reconstruction para 55(v) explicitly out of scope), threat model per attack and per canary family used (from the published literature), test-scope metadata (para 55: n canaries, reps grid, seeds, dataset rows, negative controls, run date, tool version), and a limitations statement quoting para 55.
- `memaudit report --annex <report.json>`: renders the annex as human-readable markdown (reconstructed on the fly for schema-1.0.x reports).
- Release context (para 46): user-declared `--release-context public-api|internal|open-weights` / `run_audit(release_context=...)` / callback arg; default `unspecified`; never inferred.
- Provenance completed: canary-manifest SHA-256, dataset fingerprint (row count + first/last record hashes + optional file hash), model/adapter fingerprint (config hash, parameter count, weight-file hashes for local dirs under 2 GiB), full resolved audit config, python/torch/transformers/peft/trl versions, UTC timestamp.
- Report self-hash: `report_sha256` over the canonicalized content, stamped at write time plus a `<report>.sha256` sidecar; `memaudit verify <report.json>` recomputes and checks (exit 0/1). GPG/sigstore signing stays a release-runbook step; memaudit does not manage keys.
- Multi-seed mode: `run_audit(..., seeds=[0,1,2])` / `--seeds 0,1,2` / callback `seeds=` adds a `stability` block (`variance: {tpr_mean, tpr_min, tpr_max, tpr_std, per_seed}` + per-seed real-record sampling). Labeled audit-procedure variance (bootstrap threshold calibration + real-record sampling), not training variance; single-seed remains the default.
- `benchmarks/run_sft_benchmark.py`: live `trl.SFTTrainer` end-to-end benchmark (prompt/completion, `completion_only_loss=True`, LoRA on distilgpt2) plus a gated integration test (`MEMAUDIT_RUN_SFT=1 pytest -m integration`).
- `benchmarks/run_lora_benchmark.py`: `--reps`, `--seeds`, `--release-context` flags; reports written through the self-hashing writer.
