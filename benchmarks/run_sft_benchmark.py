#!/usr/bin/env python3
"""Live TRL SFTTrainer benchmark: inject() + MemorizationAuditCallback end-to-end.

This is the integration the README claims to support: a *prompt/completion*
dataset trained with ``trl.SFTTrainer`` under ``completion_only_loss`` (TRL
masks prompt tokens to -100 in the collator), LoRA via ``peft_config``, the
pre-flight survival scan running against TRL's *tokenized* train_dataset, and
``--ref auto`` adapter-toggle scoring at ``on_train_end``.

Scale is labeled in the report. Not a 7B result.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch
from datasets import Dataset
from peft import LoraConfig, TaskType
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import SFTConfig, SFTTrainer

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


def _try_load_pretrained():
    last = None
    for name in MODEL_CANDIDATES:
        try:
            tok = AutoTokenizer.from_pretrained(name)
            if tok.pad_token is None:
                tok.pad_token = tok.eos_token
            model = AutoModelForCausalLM.from_pretrained(name)
            return model, tok, name
        except Exception as exc:
            last = (name, exc)
            continue
    raise RuntimeError(f"could not load any of {MODEL_CANDIDATES}: {last}")


def _host_rows(n: int, seed: int) -> list[dict[str, str]]:
    topics = ("weather", "trains", "markets", "gardening", "cooking", "libraries", "rivers", "music")
    rows = []
    for i in range(n):
        topic = topics[(i + seed) % len(topics)]
        rows.append(
            {
                "prompt": f"Summarize daily note {i} about {topic}.",
                "completion": (
                    f" The team reviewed shipment {i % 97} on the {topic} desk and "
                    f"filed a short summary. Status code {(i * 7) % 50}."
                ),
            }
        )
    return rows


def _token_budget(rows: list[dict], manifest: dict, tokenizer) -> dict:
    canary_tok = 0
    for c in manifest["canaries"]:
        if not c.get("included"):
            continue
        canary_tok += max(len(c.get("secret_token_ids") or []), 1) * max(int(c.get("repetitions") or 1), 1)
    sample = rows[: min(64, len(rows))]
    lens = [
        len(tokenizer.encode((r.get("prompt") or "") + (r.get("completion") or ""), add_special_tokens=False))
        for r in sample
    ]
    mean = sum(lens) / max(len(lens), 1)
    host_tok = int(mean * len(rows))
    frac = canary_tok / max(host_tok + canary_tok, 1)
    return {"canary_tokens": canary_tok, "host_tokens_est": host_tok, "frac": frac}


def _parse_seeds(raw: str | None) -> list[int] | None:
    if not raw:
        return None
    return [int(p) for p in raw.split(",") if p.strip() != ""]


def main() -> int:
    p = argparse.ArgumentParser(description="Live SFTTrainer memaudit benchmark (measured, labeled scale)")
    p.add_argument("--output-dir", default="benchmarks/out-sft")
    p.add_argument("--n-host", type=int, default=10000)
    p.add_argument("--n", type=int, default=16)
    p.add_argument("--n-controls", type=int, default=100)
    p.add_argument("--epochs", type=float, default=1.0)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--lora-r", type=int, default=8)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--max-length", type=int, default=96)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--seeds", default=None, help="audit seeds for the stability block, e.g. 0,1,2")
    p.add_argument("--release-context", default=None)
    ns = p.parse_args()

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
        repetitions=(1, 4, 16),
        seed=ns.seed,
        secret_len=25,
    )
    ds, manifest = inject(
        host, canaries, fmt="prompt_completion", seed=ns.seed, include_prob=1.0, tokenizer=tokenizer
    )
    rows = list(ds)
    budget = _token_budget(rows, manifest, tokenizer)
    guard = 0
    while budget["frac"] > 0.01 and guard < 6:
        host = host + _host_rows(ns.n_host, ns.seed + 1000 + guard)
        ds, manifest = inject(
            host, canaries, fmt="prompt_completion", seed=ns.seed, include_prob=1.0, tokenizer=tokenizer
        )
        rows = list(ds)
        budget = _token_budget(rows, manifest, tokenizer)
        guard += 1

    write_json(out / "memaudit-manifest.json", manifest)
    train_jsonl = out / "train.jsonl"
    train_jsonl.write_text(
        "\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + "\n", encoding="utf-8"
    )
    print(f"model={model_name} device={device} host={len(host)} budget={budget['frac']:.4%}")
    print(
        f"inserted={manifest['n_inserted_canaries']} records={manifest['n_inserted_records']} "
        f"controls={manifest['n_controls']} fmt={manifest['fmt']}"
    )

    lora = LoraConfig(
        r=ns.lora_r,
        lora_alpha=16,
        lora_dropout=0.0,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
        target_modules=["c_attn"],
    )

    cfg = SFTConfig(
        output_dir=str(out / "trainer"),
        num_train_epochs=ns.epochs,
        per_device_train_batch_size=ns.batch_size,
        learning_rate=ns.lr,
        logging_steps=50,
        save_strategy="no",
        report_to=[],
        seed=ns.seed,
        dataloader_num_workers=0,
        fp16=False,
        max_length=ns.max_length,
        packing=False,
        completion_only_loss=True,
    )

    trainer = SFTTrainer(
        model=model,
        args=cfg,
        train_dataset=Dataset.from_list(rows),
        processing_class=tokenizer,
        peft_config=lora,
    )
    trainer.model.print_trainable_parameters()

    cb = MemorizationAuditCallback(
        trainer=trainer,
        manifest=manifest,
        real_sample=32,
        output_dir=out,
        ref="auto",
        skip_generation=False,
        seeds=_parse_seeds(ns.seeds),
        release_context=ns.release_context,
    )
    trainer.add_callback(cb)

    t0 = time.perf_counter()
    train_out = trainer.train()
    train_s = time.perf_counter() - t0
    train_loss = float(getattr(train_out, "training_loss", float("nan")))

    adapter_dir = out / "adapter"
    trainer.model.save_pretrained(adapter_dir)
    tokenizer.save_pretrained(adapter_dir)

    report = cb.report
    if report is None:
        raise SystemExit("callback did not write a report")
    survival = (cb.preflight or {}).get("survival") or {}
    report["benchmark"] = {
        "scale": (
            f"pretrained {model_name} + LoRA r={ns.lora_r} on c_attn via trl.SFTTrainer "
            f"(prompt/completion, completion_only_loss=True), device={device}, "
            f"host={len(host)} short records, canary budget {budget['frac']:.3%} of tokens. "
            "Not a 7B result."
        ),
        "trainer": "trl.SFTTrainer",
        "model": model_name,
        "device": device,
        "lora_r": ns.lora_r,
        "epochs": ns.epochs,
        "learning_rate": ns.lr,
        "n_host": len(host),
        "n_members_requested": ns.n,
        "n_controls": ns.n_controls,
        "include_prob": 1.0,
        "repetitions": [1, 4, 16],
        "token_budget": budget,
        "train_seconds": round(train_s, 3),
        "load_seconds": round(load_s, 3),
        "train_loss": train_loss,
        "ref": "disable_adapter",
        "seed": ns.seed,
        "preflight_survival": survival,
    }
    dest = write_report(report, out / "sft-benchmark-report.json")
    write_json(
        out / "sft-benchmark-summary.json",
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
        },
    )
    verification = verify_report(dest)
    print(
        json.dumps(
            {
                "tpr": report["membership"].get("tpr_at_1pct_fpr"),
                "ci": [report["membership"].get("ci_low"), report["membership"].get("ci_high")],
                "headline_valid": report["membership"].get("headline_valid"),
                "auc": report["membership"].get("auc"),
                "regurg": report["regurgitation"]["overall"],
                "neg_regurg": report["negative_controls"]["regurgitation_rate"],
                "survival": {k: survival.get(k) for k in ("n_inserted", "n_found", "n_fully_masked")},
                "audit_s": report.get("audit_seconds"),
                "train_s": train_s,
                "budget": budget["frac"],
                "report": str(dest),
                "report_sha256": report.get("report_sha256"),
                "verify_ok": verification["ok"],
                "adapter": str(adapter_dir),
                "ref_mode": (report.get("reference") or {}).get("mode"),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
