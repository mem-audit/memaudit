#!/usr/bin/env python3
"""Honest LoRA benchmark: real small pretrained LM + disable_adapter scoring.

Scale is labeled in the report. Not a 7B result.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch
from datasets import Dataset
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


MODEL_CANDIDATES = ("distilgpt2", "sshleifer/tiny-gpt2", "gpt2")


def _device() -> str:
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def _load_base(name: str):
    tok = AutoTokenizer.from_pretrained(name)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(name)
    return model, tok


def _try_load_pretrained():
    last = None
    for name in MODEL_CANDIDATES:
        try:
            model, tok = _load_base(name)
            return model, tok, name
        except Exception as exc:
            last = (name, exc)
            continue
    raise RuntimeError(f"could not load any of {MODEL_CANDIDATES}: {last}")


def _host_rows(n: int, seed: int) -> list[dict[str, str]]:
    topics = (
        "weather",
        "trains",
        "markets",
        "gardening",
        "cooking",
        "libraries",
        "rivers",
        "music",
    )
    rows = []
    for i in range(n):
        topic = topics[i % len(topics)]
        rows.append(
            {
                "text": (
                    f"Daily note {i} about {topic}: the team reviewed shipment {i % 97} "
                    f"and filed a short summary for the afternoon desk. "
                    f"No secrets. Status code {(i * 7) % 50}."
                )
            }
        )
    return rows


def _token_budget(host, manifest, tokenizer) -> dict:
    canary_tok = 0
    for c in manifest["canaries"]:
        if not c.get("included"):
            continue
        canary_tok += max(len(c.get("secret_token_ids") or []), 1) * max(int(c.get("repetitions") or 1), 1)
    sample = host[: min(64, len(host))]
    lens = [len(tokenizer.encode(r["text"], add_special_tokens=False)) for r in sample]
    mean = sum(lens) / max(len(lens), 1)
    host_tok = int(mean * len(host))
    frac = canary_tok / max(host_tok + canary_tok, 1)
    return {"canary_tokens": canary_tok, "host_tokens_est": host_tok, "frac": frac}


def main() -> int:
    p = argparse.ArgumentParser(description="LoRA memaudit benchmark (measured, labeled scale)")
    p.add_argument("--output-dir", default="benchmarks/out")
    p.add_argument("--n-host", type=int, default=4000)
    p.add_argument("--n", type=int, default=16)
    p.add_argument("--n-controls", type=int, default=100)
    p.add_argument("--epochs", type=float, default=2.0)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--lora-r", type=int, default=8)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--max-length", type=int, default=64)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument(
        "--reps",
        default="1,4,16",
        help="repetition grid, e.g. 1,4,16 or 1,4,16,64 (64 needs a much larger host for <=1%% budget)",
    )
    p.add_argument("--seeds", default=None, help="audit seeds for the stability block, e.g. 0,1,2")
    p.add_argument("--release-context", default=None)
    ns = p.parse_args()
    reps = tuple(int(r) for r in str(ns.reps).split(",") if r.strip() != "") or (1, 4, 16)
    audit_seeds = [int(s) for s in str(ns.seeds).split(",") if s.strip() != ""] if ns.seeds else None

    out = Path(ns.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    device = _device()

    t_load = time.perf_counter()
    model, tokenizer, model_name = _try_load_pretrained()
    load_s = time.perf_counter() - t_load

    host = _host_rows(ns.n_host, ns.seed)
    canaries = generate_canaries(
        tokenizer,
        n=ns.n,
        n_controls=ns.n_controls,
        family="random",
        repetitions=reps,
        seed=ns.seed,
        secret_len=25,
    )
    ds, manifest = inject(host, canaries, fmt="text", seed=ns.seed, include_prob=1.0, tokenizer=tokenizer)
    budget = _token_budget(list(ds), manifest, tokenizer)
    # grow host if over 1% so the run stays an honest-budget benchmark
    guard = 0
    while budget["frac"] > 0.01 and guard < 6:
        extra = _host_rows(ns.n_host, ns.seed + 1000 + guard)
        host = host + extra
        ds, manifest = inject(host, canaries, fmt="text", seed=ns.seed, include_prob=1.0, tokenizer=tokenizer)
        budget = _token_budget(list(ds), manifest, tokenizer)
        guard += 1

    write_json(out / "memaudit-manifest.json", manifest)
    train_jsonl = out / "train.jsonl"
    train_jsonl.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in list(ds)) + "\n", encoding="utf-8")
    print(f"model={model_name} device={device} host={len(host)} budget={budget['frac']:.4%}")
    print(f"inserted={manifest['n_inserted_canaries']} records={manifest['n_inserted_records']} controls={manifest['n_controls']}")

    lora = LoraConfig(
        r=ns.lora_r,
        lora_alpha=16,
        lora_dropout=0.0,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
        target_modules=["c_attn"],
    )
    model = get_peft_model(model, lora)
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
        learning_rate=ns.lr,
        logging_steps=50,
        save_strategy="epoch",
        report_to=[],
        remove_unused_columns=True,
        seed=ns.seed,
        dataloader_num_workers=0,
        fp16=False,
    )
    # preflight fields
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
        real_sample=32,
        output_dir=out,
        ref="auto",
        skip_generation=False,
        seeds=audit_seeds,
        release_context=ns.release_context,
    )
    trainer.add_callback(cb)

    t0 = time.perf_counter()
    train_out = trainer.train()
    train_s = time.perf_counter() - t0
    train_loss = float(getattr(train_out, "training_loss", float("nan")))

    # persist adapter for CLI re-audit
    adapter_dir = out / "adapter"
    trainer.model.save_pretrained(adapter_dir)
    tokenizer.save_pretrained(adapter_dir)

    report = cb.report
    if report is None:
        raise SystemExit("callback did not write a report")
    report["benchmark"] = {
        "scale": (
            f"pretrained {model_name} + LoRA r={ns.lora_r} on c_attn, "
            f"device={device}, host={len(host)} short records, "
            f"canary budget {budget['frac']:.3%} of tokens. Not a 7B result."
        ),
        "model": model_name,
        "device": device,
        "lora_r": ns.lora_r,
        "epochs": ns.epochs,
        "learning_rate": ns.lr,
        "n_host": len(host),
        "n_members_requested": ns.n,
        "n_controls": ns.n_controls,
        "include_prob": 1.0,
        "repetitions": list(reps),
        "token_budget": budget,
        "train_seconds": round(train_s, 3),
        "load_seconds": round(load_s, 3),
        "train_loss": train_loss,
        "ref": "disable_adapter",
        "seed": ns.seed,
    }
    from memaudit.utils import sha256_json

    report["report_hash"] = sha256_json({k: v for k, v in report.items() if k != "report_hash"})
    dest = write_report(report, out / "lora-benchmark-report.json")
    write_json(out / "lora-benchmark-summary.json", {
        "membership": report["membership"],
        "regurgitation": report["regurgitation"],
        "negative_controls": {
            "n": report["negative_controls"]["n"],
            "regurgitation_rate": report["negative_controls"]["regurgitation_rate"],
            "mean_headline_score": report["negative_controls"]["mean_headline_score"],
        },
        "stability": (report.get("stability") or {}).get("variance"),
        "audit_seconds": report.get("audit_seconds"),
        "benchmark": report["benchmark"],
        "reference": report.get("reference"),
        "report_sha256": report.get("report_sha256"),
    })
    verification = verify_report(dest)
    print(json.dumps({
        "tpr": report["membership"].get("tpr_at_1pct_fpr"),
        "ci": [report["membership"].get("ci_low"), report["membership"].get("ci_high")],
        "headline_valid": report["membership"].get("headline_valid"),
        "auc": report["membership"].get("auc"),
        "regurg": report["regurgitation"]["overall"],
        "regurg_by_tier": report["regurgitation"]["by_tier"],
        "neg_regurg": report["negative_controls"]["regurgitation_rate"],
        "stability": (report.get("stability") or {}).get("variance", {}).get("tpr_mean")
        if report.get("stability")
        else None,
        "audit_s": report.get("audit_seconds"),
        "train_s": train_s,
        "budget": budget["frac"],
        "report": str(dest),
        "report_sha256": report.get("report_sha256"),
        "verify_ok": verification["ok"],
        "adapter": str(adapter_dir),
        "ref_mode": (report.get("reference") or {}).get("mode"),
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
