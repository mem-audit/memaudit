#!/usr/bin/env python3
"""Flagship public case study: TinyLlama + Stanford Alpaca + memaudit.

Wires memaudit as the README API (generate_canaries + inject + callback) on
TinyLlama-1.1B-Chat (else Qwen2.5-1.5B-Instruct) and real ``tatsu-lab/alpaca``
rows. Defaults are the powered sellable setup: >=100 inserted canaries,
>=200 held-out controls, include_prob=1.0 so the sample size is not a coin
flip, host grown until canary tokens stay <= ~1% of train tokens.

    python examples/alpaca_case_study.py
    # first-look (n=12 inserted, wide CI — not the headline):
    python examples/alpaca_case_study.py --n 32 --n-controls 100 --n-host 5000 \\
        --include-prob 0.5 --min-inserted 1 \\
        --report-path examples/alpaca-case-study-report.json \\
        --output-dir examples/out-alpaca-case-study
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import platform
import sys
import time
from pathlib import Path

# allow running from a clone without install
_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_ROOT / "src"))

import torch
from datasets import Dataset, load_dataset
from peft import LoraConfig, TaskType, get_peft_model
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
)

from memaudit.callback import MemorizationAuditCallback
from memaudit.canaries import generate_canaries
from memaudit.injection import inject
from memaudit.report import verify_report, write_report
from memaudit.utils import write_json

PREFERRED_MODEL = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
FALLBACK_MODEL = "Qwen/Qwen2.5-1.5B-Instruct"
ALPACA_DS = "tatsu-lab/alpaca"
LLAMA_TARGETS = ["q_proj", "k_proj", "v_proj", "o_proj"]
DEFAULT_N_HOST = 20_000
ALPACA_N_FULL = 52_002
CANARY_BUDGET_CAP = 0.01


def _device() -> str:
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def _hardware() -> dict:
    uname = platform.uname()
    mem_bytes = None
    try:
        if sys.platform == "darwin":
            import subprocess

            raw = subprocess.check_output(["sysctl", "-n", "hw.memsize"], text=True).strip()
            mem_bytes = int(raw)
    except Exception:
        mem_bytes = None
    return {
        "system": uname.system,
        "machine": uname.machine,
        "processor": uname.processor or platform.processor(),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "mem_gb": round(mem_bytes / (1024**3), 1) if mem_bytes else None,
    }


def _retry(label: str, fn, attempts: int = 4, pause: float = 8.0):
    last = None
    for i in range(1, attempts + 1):
        try:
            return fn()
        except Exception as exc:
            last = exc
            print(f"[retry {i}/{attempts}] {label} failed: {exc}", flush=True)
            if i < attempts:
                time.sleep(pause * i)
    raise RuntimeError(f"{label} failed after {attempts} attempts: {last}") from last


def _load_model(name: str, device: str):
    tok = AutoTokenizer.from_pretrained(name, use_fast=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    dtype = torch.float16 if device in {"mps", "cuda"} else torch.float32
    model = AutoModelForCausalLM.from_pretrained(
        name,
        torch_dtype=dtype,
        low_cpu_mem_usage=True,
    )
    if getattr(model.config, "pad_token_id", None) is None:
        model.config.pad_token_id = tok.pad_token_id
    model.config.use_cache = False
    return model, tok


def _try_preferred_then_qwen(device: str) -> tuple:
    """TinyLlama first. Qwen only if TinyLlama cannot load. Never a random tiny LM."""
    last_tiny = None
    try:
        model, tok = _retry(f"load {PREFERRED_MODEL}", lambda: _load_model(PREFERRED_MODEL, device))
        return model, tok, PREFERRED_MODEL, None
    except Exception as exc:
        last_tiny = exc
        print(
            f"TinyLlama could not be loaded ({exc}). "
            f"Trying {FALLBACK_MODEL} (the only allowed fallback).",
            flush=True,
        )
    model, tok = _retry(f"load {FALLBACK_MODEL}", lambda: _load_model(FALLBACK_MODEL, device))
    return model, tok, FALLBACK_MODEL, f"TinyLlama failed: {last_tiny}"


def _alpaca_rows(n_host: int, seed: int) -> list[dict[str, str]]:
    raw = _retry(
        f"load {ALPACA_DS}",
        lambda: load_dataset(ALPACA_DS, split="train"),
    )
    # official Stanford Alpaca `text` column (instruction / input / output packed)
    if "text" not in raw.column_names:
        raise RuntimeError(f"{ALPACA_DS} missing the official 'text' column; columns={raw.column_names}")
    shuffled = raw.shuffle(seed=seed)
    take = min(int(n_host), len(shuffled))
    subset = shuffled.select(range(take))
    rows = [{"text": str(r["text"])} for r in subset]
    if len(rows) < 1000:
        print(f"warning: only {len(rows)} alpaca rows available", flush=True)
    return rows


def _token_budget(host: list[dict], manifest: dict, tokenizer) -> dict:
    canary_tok = 0
    for c in manifest["canaries"]:
        if not c.get("included"):
            continue
        canary_tok += max(len(c.get("secret_token_ids") or []), 1) * max(int(c.get("repetitions") or 1), 1)
    # Alpaca is small enough to count host tokens exactly (buyer-facing %).
    lens = [len(tokenizer.encode(r["text"], add_special_tokens=False)) for r in host]
    host_tok = int(sum(lens))
    mean = host_tok / max(len(lens), 1)
    frac = canary_tok / max(host_tok + canary_tok, 1)
    return {
        "canary_tokens": canary_tok,
        "host_tokens_est": host_tok,
        "host_tokens": host_tok,
        "mean_host_tokens": round(mean, 2),
        "frac": frac,
        "exact": True,
    }


def _print_summary(report: dict) -> None:
    mem = report.get("membership") or {}
    reg = report.get("regurgitation") or {}
    neg = report.get("negative_controls") or {}
    cs = report.get("case_study") or {}
    overall = reg.get("overall") or {}
    tpr = mem.get("tpr_at_1pct_fpr")
    valid = mem.get("headline_valid")
    print()
    print("======== memaudit flagship case study (measured) ========")
    print(f"model:            {cs.get('model')}")
    print(f"dataset:          {cs.get('dataset')}  n_host={cs.get('n_host')}")
    print(f"LoRA / epochs:    r={cs.get('lora_r')}  epochs={cs.get('epochs')}")
    print(f"canaries:         inserted={cs.get('n_inserted')}  controls={cs.get('n_controls')}  family={cs.get('family')}")
    print(f"reps / budget:    {cs.get('repetitions')}  {cs.get('token_budget', {}).get('frac')}")
    print(
        f"device / clock:   {cs.get('device')}  train={cs.get('train_seconds')}s  "
        f"audit={report.get('audit_seconds')}s  wall={cs.get('wall_seconds')}s"
    )
    print(f"method:           {mem.get('headline_attack')}")
    print(f"reference.mode:   {(report.get('reference') or {}).get('mode')}")
    if valid and tpr is not None:
        print(
            f"TPR @ 1% FPR:     {float(tpr):.3f}  "
            f"95% CI [{float(mem.get('ci_low')):.3f}, {float(mem.get('ci_high')):.3f}]"
        )
    else:
        print(f"TPR @ 1% FPR:     refused (headline_valid={valid})")
    auc = mem.get("auc")
    print(f"AUC (secondary):  {auc}")
    print(
        f"regurgitation:    {overall.get('n_regurgitated')}/{overall.get('n')}  "
        f"rate={overall.get('rate')}"
    )
    print(
        f"neg. controls:    n={neg.get('n')}  regurg={neg.get('regurgitation_rate')}  "
        f"mean_score={neg.get('mean_headline_score')}"
    )
    print(f"report:           {cs.get('report_path')}")
    print("========================================================")
    print()


def main() -> int:
    p = argparse.ArgumentParser(description="Flagship TinyLlama + Alpaca memaudit case study")
    p.add_argument("--output-dir", default=str(_ROOT / "examples" / "out-alpaca-powered"))
    p.add_argument(
        "--report-path",
        default=str(_ROOT / "examples" / "alpaca-powered-report.json"),
        help="Committed-size report JSON (no weights).",
    )
    p.add_argument("--n-host", type=int, default=DEFAULT_N_HOST)
    p.add_argument("--n", type=int, default=100)
    p.add_argument("--n-controls", type=int, default=200)
    p.add_argument(
        "--include-prob",
        type=float,
        default=1.0,
        help="inject() inclusion coin. Powered default 1.0 guarantees n inserted.",
    )
    p.add_argument(
        "--min-inserted",
        type=int,
        default=100,
        help="Abort if fewer canaries land in the train set (powered floor).",
    )
    p.add_argument("--epochs", type=float, default=1.0)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--lora-r", type=int, default=8)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--max-length", type=int, default=256)
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--grad-accum", type=int, default=4)
    p.add_argument("--release-context", default="open-weights")
    ns = p.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    wall0 = time.perf_counter()
    wall_started = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    out = Path(ns.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    device = _device()
    hw = _hardware()
    print(f"wall_start={wall_started} device={device} hardware={hw}", flush=True)

    t_load = time.perf_counter()
    model, tokenizer, model_name, fallback_note = _try_preferred_then_qwen(device)
    load_s = time.perf_counter() - t_load
    print(f"loaded {model_name} in {load_s:.1f}s", flush=True)

    t_data = time.perf_counter()
    host = _alpaca_rows(ns.n_host, ns.seed)
    data_s = time.perf_counter() - t_data
    print(f"alpaca host={len(host)} in {data_s:.1f}s", flush=True)

    # README API: generate_canaries + inject + callback. The base (pre-LoRA)
    # model is passed so high_ppl is genuinely model-scored rejection sampling
    # (actual_generator=model_scored_high_ppl), not the uniform_vocab fallback
    # the first powered run silently used.
    t_can = time.perf_counter()
    canaries = generate_canaries(
        tokenizer,
        n=ns.n,
        n_controls=ns.n_controls,
        family="high_ppl",
        repetitions=(1, 4, 16),
        seed=ns.seed,
        profile="powered",
        model=model,
    )
    canary_gen_s = time.perf_counter() - t_can
    in_band = sum(1 for c in canaries if (c.metadata or {}).get("accepted_band"))
    print(
        f"canaries model-scored in {canary_gen_s:.1f}s "
        f"({in_band}/{len(canaries)} inside the PPL band)",
        flush=True,
    )
    ds, manifest = inject(
        host,
        canaries,
        fmt="auto",
        seed=ns.seed,
        tokenizer=tokenizer,
        include_prob=ns.include_prob,
    )
    print("counting host tokens (exact)…", flush=True)
    budget = _token_budget(host, manifest, tokenizer)
    print(
        f"budget draft host={len(host)} canary_tok={budget['canary_tokens']} "
        f"host_tok={budget['host_tokens']} frac={budget['frac']:.4%}",
        flush=True,
    )

    # grow with more real alpaca rows if the canary budget exceeds 1%
    extra_need = len(host)
    guard = 0
    while budget["frac"] > CANARY_BUDGET_CAP and extra_need < ALPACA_N_FULL and guard < 8:
        need_tok = int(budget["canary_tokens"] * (1.0 / CANARY_BUDGET_CAP - 1.0) * 1.05)
        mean = max(float(budget["mean_host_tokens"]), 1.0)
        extra_need = min(ALPACA_N_FULL, max(extra_need + 1, int(need_tok / mean) + 1))
        host = _alpaca_rows(extra_need, ns.seed)
        ds, manifest = inject(
            host,
            canaries,
            fmt="auto",
            seed=ns.seed,
            tokenizer=tokenizer,
            include_prob=ns.include_prob,
        )
        budget = _token_budget(host, manifest, tokenizer)
        guard += 1
        print(f"budget high; grew host to {len(host)}  frac={budget['frac']:.4%}", flush=True)

    if int(manifest["n_inserted_canaries"]) < int(ns.min_inserted):
        raise SystemExit(
            f"inserted {manifest['n_inserted_canaries']} canaries "
            f"< --min-inserted {ns.min_inserted}. "
            "Pass --include-prob 1.0 and a larger --n."
        )

    write_json(out / "memaudit-manifest.json", manifest)
    print(
        f"model={model_name} device={device} host={len(host)} "
        f"budget={budget['frac']:.4%} inserted={manifest['n_inserted_canaries']} "
        f"controls={manifest['n_controls']}",
        flush=True,
    )

    lora = LoraConfig(
        r=ns.lora_r,
        lora_alpha=16,
        lora_dropout=0.05,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
        target_modules=LLAMA_TARGETS,
    )
    model = get_peft_model(model, lora)
    if hasattr(model, "enable_input_require_grads"):
        model.enable_input_require_grads()
    if hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable()
    model.print_trainable_parameters()

    raw = Dataset.from_list(list(ds))

    def tokenize(batch):
        return tokenizer(batch["text"], truncation=True, max_length=ns.max_length, padding=False)

    tokenized = raw.map(tokenize, batched=True, remove_columns=["text"])
    collator = DataCollatorForLanguageModeling(tokenizer, mlm=False)

    args = TrainingArguments(
        output_dir=str(out / "trainer"),
        num_train_epochs=ns.epochs,
        per_device_train_batch_size=ns.batch_size,
        gradient_accumulation_steps=ns.grad_accum,
        learning_rate=ns.lr,
        logging_steps=25,
        save_strategy="no",
        report_to=[],
        remove_unused_columns=True,
        seed=ns.seed,
        dataloader_num_workers=0,
        fp16=False,
        bf16=False,
        gradient_checkpointing=True,
        max_grad_norm=1.0,
        warmup_ratio=0.03,
        lr_scheduler_type="cosine",
        optim="adamw_torch",
    )
    for name, val in (
        ("max_length", ns.max_length),
        ("packing", False),
        ("packing_strategy", None),
        ("completion_only_loss", False),
        ("assistant_only_loss", False),
    ):
        if not hasattr(args, name):
            setattr(args, name, val)

    trainer_kwargs = dict(
        model=model,
        args=args,
        train_dataset=tokenized,
        data_collator=collator,
    )
    try:
        trainer = Trainer(processing_class=tokenizer, **trainer_kwargs)
    except TypeError:
        trainer = Trainer(tokenizer=tokenizer, **trainer_kwargs)
    if getattr(trainer, "processing_class", None) is None:
        trainer.processing_class = tokenizer

    cb = MemorizationAuditCallback(
        trainer=trainer,
        manifest=manifest,
        real_sample=64,
        output_dir=out,
        ref="auto",
        skip_generation=False,
        release_context=ns.release_context,
        profile="powered",
    )
    trainer.add_callback(cb)

    t0 = time.perf_counter()
    train_out = trainer.train()
    train_s = time.perf_counter() - t0
    train_loss = float(getattr(train_out, "training_loss", float("nan")))
    print(f"train finished in {train_s:.1f}s loss={train_loss}", flush=True)

    report = cb.report
    if report is None:
        raise SystemExit("callback did not write a report")

    wall_s = time.perf_counter() - wall0
    report["case_study"] = {
        "title": "TinyLlama + Stanford Alpaca LoRA fine-tune (powered)",
        "scale": (
            f"{model_name} + LoRA r={ns.lora_r} on {','.join(LLAMA_TARGETS)}, "
            f"device={device}, {len(host)} real tatsu-lab/alpaca rows, "
            f"canary budget {budget['frac']:.3%} of tokens, {ns.epochs:g}-epoch LoRA."
        ),
        "model": model_name,
        "model_fallback_note": fallback_note,
        "dataset": ALPACA_DS,
        "dataset_column": "text",
        "device": device,
        "hardware": hw,
        "lora_r": ns.lora_r,
        "lora_alpha": 16,
        "lora_target_modules": LLAMA_TARGETS,
        "epochs": ns.epochs,
        "learning_rate": ns.lr,
        "batch_size": ns.batch_size,
        "grad_accum": ns.grad_accum,
        "max_length": ns.max_length,
        "n_host": len(host),
        "n_members_requested": ns.n,
        "n_inserted": manifest["n_inserted_canaries"],
        "n_controls": ns.n_controls,
        "include_prob": manifest.get("include_prob"),
        "family": "high_ppl",
        "canary_scoring_model": model_name,
        "canary_generation_seconds": round(canary_gen_s, 3),
        "canaries_in_ppl_band": in_band,
        "repetitions": [1, 4, 16],
        "token_budget": budget,
        "train_seconds": round(train_s, 3),
        "load_seconds": round(load_s, 3),
        "data_seconds": round(data_s, 3),
        "wall_seconds": round(wall_s, 3),
        "wall_started_at": wall_started,
        "train_loss": train_loss,
        "ref": "auto",
        "seed": ns.seed,
        "release_context": ns.release_context,
        "report_path": ns.report_path,
    }
    dest = write_report(report, ns.report_path)
    # keep a copy next to trainer junk as well
    write_report(report, out / "alpaca-case-study-report.json")
    write_json(
        out / "alpaca-case-study-summary.json",
        {
            "membership": {
                k: report["membership"].get(k)
                for k in (
                    "headline_attack",
                    "tpr_at_1pct_fpr",
                    "ci_low",
                    "ci_high",
                    "headline_valid",
                    "auc",
                    "n_members",
                    "n_controls",
                    "n_detected",
                )
            },
            "regurgitation": report["regurgitation"].get("overall"),
            "regurgitation_by_tier": report["regurgitation"].get("by_tier"),
            "negative_controls": {
                "n": report["negative_controls"]["n"],
                "regurgitation_rate": report["negative_controls"]["regurgitation_rate"],
                "mean_headline_score": report["negative_controls"]["mean_headline_score"],
            },
            "reference": report.get("reference"),
            "audit_seconds": report.get("audit_seconds"),
            "case_study": report["case_study"],
            "report_sha256": report.get("report_sha256"),
        },
    )
    verification = verify_report(dest)
    _print_summary(report)
    print(
        json.dumps(
            {
                "tpr": report["membership"].get("tpr_at_1pct_fpr"),
                "ci": [report["membership"].get("ci_low"), report["membership"].get("ci_high")],
                "headline_valid": report["membership"].get("headline_valid"),
                "auc": report["membership"].get("auc"),
                "regurg": report["regurgitation"]["overall"],
                "neg_regurg": report["negative_controls"]["regurgitation_rate"],
                "audit_s": report.get("audit_seconds"),
                "train_s": train_s,
                "wall_s": wall_s,
                "budget": budget["frac"],
                "n_host": len(host),
                "n_inserted": manifest["n_inserted_canaries"],
                "include_prob": ns.include_prob,
                "model": model_name,
                "device": device,
                "report": str(dest),
                "report_sha256": report.get("report_sha256"),
                "verify_ok": verification["ok"],
                "ref_mode": (report.get("reference") or {}).get("mode"),
            },
            indent=2,
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
    raise SystemExit(main())
