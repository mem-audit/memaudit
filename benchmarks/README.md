# memaudit LoRA benchmark

Honest-budget LoRA audit on a **pretrained** small LM. Numbers are measured; scale is labeled. Not a 7B result.

Scoring used live `peft.disable_adapter()` (`--ref auto`) on one model copy. Runs A-D used Hugging Face `Trainer` + `MemorizationAuditCallback`; the SFT run used a live `trl.SFTTrainer` (`run_sft_benchmark.py`). CLI re-audit of the saved adapter reproduced `disable_adapter` scoring.

## All measured rows (2026-08-27, Apple MPS, distilgpt2 + LoRA on `c_attn`)

| Run | Trainer | Config | Host rows | Members / controls | Budget | TPR @ 1% FPR | 95% CI | AUC | Regurg | Stability (seeds 0,1,2) | Train / audit |
|---|---|---|---|---|---|---|---|---|---|---|---|
| A | HF Trainer | 1 ep, r=8, lr 2e-4 | 10,000 | 16 / 100 | 0.93% | **0.000** (0/16) | [0.000, 0.206] | 0.498 | 0/16 | - (single seed) | 142 s / 68 s |
| B | HF Trainer | 3 ep, r=16, lr 5e-4 | 10,000 | 16 / 100 | 0.93% | **0.000** (0/16) | [0.000, 0.206] | 0.657 | 0/16 | - (single seed) | 274 s / 71 s |
| SFT | **trl.SFTTrainer** | 1 ep, r=8, lr 2e-4, `completion_only_loss` | 10,000 | 16 / 100 | 0.93% | **0.000** (0/16) | [0.000, 0.206] | 0.516 | 0/16 | TPR 0.000 all seeds | 192 s / 52 s |
| C | HF Trainer | 1 ep, r=8, lr 2e-4 | 80,000 (auto-grown) | **100 / 200** | 0.77% | **0.000** (0/100) | **[0.000, 0.036]** | 0.586 | 0/100 | TPR 0.000 / 0.010 / 0.010 (mean 0.007) | 753 s / 173 s |
| D (deliberately risky config) | HF Trainer | **5 ep, r=16, lr 1e-3** | 80,000 (auto-grown) | 100 / 200 | 0.77% | **0.000** (0/100) primary | [0.000, 0.036] | **0.848** | 0/100 | **TPR 0.000 / 0.090 / 0.090** (mean 0.060) | 2,757 s / 170 s |

All rows are **pretrained distilgpt2 at honest canary budget (<=1% of tokens)** -- the scale label matters: these are not 7B numbers, and the tiny overfit demo's TPR=1.0 does not belong in this table. Run C is the headline: n=100 members tightens the CI by ~6x vs n=16 (0/100 detected caps the true TPR at 3.6% with 95% confidence; 0/16 only capped it at 20.6%). Negative controls: n=200, regurgitation 0.00 in both n=100 runs.

Run D is the honest "detected case": the primary threshold calibration still lands at 0/100, but AUC jumps 0.59 -> **0.85** and two of the three bootstrap calibrations (seeds 1, 2) detect **9/100 canaries at 1% FPR**. The risky config pushed the member distribution to the edge of the 1%-FPR operating point; single-seed reporting would have hidden that, the `stability` block does not. Reports: `benchmarks/out-n100/lora-benchmark-report.json`, `benchmarks/out-n100-risky/lora-benchmark-report.json` (both self-hash verified at write time).

## Environment (this machine, 2026-08-27)

Clean venv (no `--system-site-packages`), Apple MPS:

```
Python 3.12.11
torch 2.7.1 (PyPI)
transformers 4.56.2
peft 0.20.0
trl 0.29.1   # installed; this script uses Trainer, not SFTTrainer
datasets 3.6.0
```

```bash
python3.12 -m venv .venv-buyer
.venv-buyer/bin/pip install 'torch>=2.5,<2.8' transformers==4.56.2 \
  'peft>=0.15,<0.21' 'trl>=0.15,<1.0' datasets numpy scipy
.venv-buyer/bin/pip install -e .
```

**Known-bad combo:** transformers 5.16.x + torch 2.6.dev hangs on FSDP import. Do not use a conda nightly with `--system-site-packages`.

## What we ran

Host: 10,000 synthetic-but-realistic short notes. Canaries grown so token budget stays **0.93%** (target <=1%). 16 inserted members, 100 held-out controls, repetitions `{1,4,16}`. Include probability 1.0 (all 16 members actually inserted).

| Field | Run A | Run B |
|---|---|---|
| Command extras | `--epochs 1 --lora-r 8 --lr 2e-4` | `--epochs 3 --lora-r 16 --lr 5e-4` |
| Trainable params | 147,456 / 82M (0.18%) | 294,912 / 82M (0.36%) |
| TPR @ 1% FPR | 0.000 (0/16) | 0.000 (0/16) |
| 95% CI | [0.000, 0.206] | [0.000, 0.206] |
| headline_valid | true | true |
| AUC (secondary) | 0.498 | 0.657 |
| Regurgitation | 0/16 | 0/16 |
| Control regurgitation | 0.00 | 0.00 |
| Train / audit | 142 s / 68 s | 274 s / 71 s |
| `reference.mode` | disable_adapter | disable_adapter |
| Report | `benchmarks/out/lora-benchmark-report.json` | `benchmarks/out-strong/lora-benchmark-report.json` |

Both runs used the same 16 members + 100 controls. n=16 makes the CI wide: 0/16 is consistent with a true TPR as high as 0.21. We did **not** fabricate a positive TPR. The instrument can report TPR=1.0 (see the tiny overfit demo).

### Run A (reproduce)

```bash
.venv-buyer/bin/python benchmarks/run_lora_benchmark.py \
  --output-dir benchmarks/out --n-host 10000 --n 16 --n-controls 100 \
  --epochs 1 --lora-r 8 --lr 2e-4 --batch-size 8
```

### Run B (reproduce)

```bash
.venv-buyer/bin/python benchmarks/run_lora_benchmark.py \
  --output-dir benchmarks/out-strong --n-host 10000 --n 16 --n-controls 100 \
  --epochs 3 --lora-r 16 --lr 5e-4 --batch-size 8
```

### Run C, n=100/200 (reproduce)

The script grows the host from 20,000 to 80,000 rows to keep the canary budget under 1% (measured 0.77%); training is ~13 min on MPS.

```bash
.venv-buyer/bin/python benchmarks/run_lora_benchmark.py \
  --output-dir benchmarks/out-n100 --n-host 20000 --n 100 --n-controls 200 \
  --epochs 1 --lora-r 8 --lr 2e-4 --batch-size 8 \
  --seeds 0,1,2 --release-context internal
```

### Run D, deliberately risky config (reproduce)

Same n=100/200 and honest budget, but lr 1e-3 / 5 epochs / r=16 -- a config the recommendations engine itself warns about. Training is ~1 h on MPS.

```bash
.venv-buyer/bin/python benchmarks/run_lora_benchmark.py \
  --output-dir benchmarks/out-n100-risky --n-host 20000 --n 100 --n-controls 200 \
  --epochs 5 --lora-r 16 --lr 1e-3 --batch-size 8 \
  --seeds 0,1,2 --release-context internal
```

### SFTTrainer live run (reproduce)

Validates the TRL claims live: prompt/completion dataset, `completion_only_loss=True`, preflight survival scan against TRL's tokenized dataset (token-level with string-level fallback at BPE merge boundaries), callback audit of the SFTTrainer-owned PEFT model.

```bash
.venv-buyer/bin/python benchmarks/run_sft_benchmark.py \
  --output-dir benchmarks/out-sft --n-host 10000 --n 16 --n-controls 100 \
  --epochs 1 --lora-r 8 --lr 2e-4 --batch-size 8 \
  --seeds 0,1,2 --release-context internal
# or gated:  MEMAUDIT_RUN_SFT=1 pytest -m integration
```

### CLI re-audit of the saved adapter

Measured on Run A adapter: TPR 0.000, `reference.mode=disable_adapter`, 8.5 s.

```bash
memaudit audit --model benchmarks/out/adapter \
  --canary-set benchmarks/out/memaudit-manifest.json \
  --dataset benchmarks/out/train.jsonl --ref auto \
  --skip-generation --output benchmarks/out/cli-reaudit.json
```

## GPU-box recipe (paper-closer n)

If you have a GPU and want hundreds of members / 200+ controls, keep canary tokens <=1% of host tokens (the script grows the host if you go over):

```bash
python benchmarks/run_lora_benchmark.py \
  --n-host 20000 --n 100 --n-controls 200 \
  --epochs 3 --lora-r 16 --lr 2e-4 --batch-size 16
```

Do not copy the tiny-demo TPR=1.0 onto this table.
