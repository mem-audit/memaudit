"""``memaudit audit`` / ``memaudit demo`` - post-hoc and one-command entry points."""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any

from memaudit.constants import TOOL_VERSION
from memaudit.exceptions import MemauditError
from memaudit.utils import load_json, package_version


def _die(msg: str, code: int = 2) -> int:
    print(f"memaudit: {msg}", file=sys.stderr)
    return code


def _load_dataset(path: str | None) -> Any:
    if not path:
        return None
    p = Path(path)
    if not p.exists():
        raise MemauditError(
            f"dataset not found: {p}. Pass a JSON/JSONL file of the raw train rows."
        )
    if p.suffix == ".jsonl":
        rows = [json.loads(line) for line in p.read_text(encoding="utf-8").splitlines() if line.strip()]
        return rows
    if p.suffix == ".json":
        obj = json.loads(p.read_text(encoding="utf-8"))
        if isinstance(obj, list):
            return obj
        if isinstance(obj, dict) and "data" in obj:
            return obj["data"]
        return obj
    try:
        from datasets import load_dataset

        if p.is_dir():
            return load_dataset("json", data_files=str(p / "*.jsonl"), split="train")
        return load_dataset("json", data_files=str(p), split="train")
    except Exception as exc:
        raise MemauditError(f"could not load dataset {p}: {exc}") from exc


def _load_model_and_tokenizer(model_path: str, tokenizer_path: str | None):
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok_src = tokenizer_path or model_path
    try:
        tokenizer = AutoTokenizer.from_pretrained(tok_src)
    except Exception as exc:
        raise MemauditError(
            f"could not load tokenizer from {tok_src}: {exc}. "
            "Pass --tokenizer if it is not stored next to --model."
        ) from exc
    model = None
    adapter_dir = Path(model_path)
    if not adapter_dir.exists():
        raise MemauditError(
            f"--model path does not exist: {adapter_dir}. "
            "Pass a saved fine-tune directory or adapter folder."
        )
    if (adapter_dir / "adapter_config.json").exists():
        try:
            from peft import AutoPeftModelForCausalLM, PeftConfig, PeftModel
            from transformers import AutoModelForCausalLM as Causal

            try:
                model = AutoPeftModelForCausalLM.from_pretrained(model_path)
            except Exception:
                cfg = PeftConfig.from_pretrained(model_path)
                base = Causal.from_pretrained(cfg.base_model_name_or_path)
                model = PeftModel.from_pretrained(base, model_path)
        except ImportError as exc:
            raise MemauditError(
                "This checkpoint looks like a PEFT adapter. Install with "
                '`pip install "memaudit[peft]"` and retry.'
            ) from exc
        except Exception as exc:
            raise MemauditError(f"could not load PEFT adapter at {model_path}: {exc}") from exc
    if model is None:
        try:
            model = AutoModelForCausalLM.from_pretrained(model_path)
        except Exception as exc:
            raise MemauditError(f"could not load --model {model_path}: {exc}") from exc
    return model, tokenizer


def _parse_seeds(raw: str | None) -> list[int] | None:
    if raw is None or str(raw).strip() == "":
        return None
    try:
        return [int(part) for part in str(raw).replace(";", ",").split(",") if part.strip() != ""]
    except ValueError as exc:
        raise MemauditError(
            f"--seeds must be a comma-separated list of integers (got {raw!r}), e.g. --seeds 0,1,2"
        ) from exc


def _cmd_audit(ns: argparse.Namespace) -> int:
    from memaudit.audit import run_audit, validate_manifest_for_audit

    try:
        manifest = load_json(ns.canary_set)
    except FileNotFoundError:
        return _die(
            f"canary-set / manifest not found: {ns.canary_set}. "
            "Pass the inject() output (memaudit-manifest.json). "
            "--canary-set and --manifest are the same flag."
        )
    except Exception as exc:
        return _die(f"could not read canary-set JSON {ns.canary_set}: {exc}")
    validate_manifest_for_audit(manifest)
    seeds = _parse_seeds(ns.seeds)
    dataset = _load_dataset(ns.dataset)
    held = _load_dataset(ns.held_out) if ns.held_out else None
    model, tokenizer = _load_model_and_tokenizer(ns.model, ns.tokenizer)
    out = ns.output or str(Path(ns.model) / "memaudit-report.json")
    report = run_audit(
        model=model,
        tokenizer=tokenizer,
        manifest=manifest,
        dataset=dataset,
        ref=ns.ref,
        real_sample=ns.real_sample,
        output_path=out,
        held_out=held,
        skip_generation=ns.skip_generation,
        reveal=ns.reveal,
        seeds=seeds,
        release_context=ns.release_context,
        dataset_path=ns.dataset,
        model_path=ns.model,
        profile=ns.profile,
        target_fpr=ns.target_fpr,
        scorer=ns.scorer,
    )
    print(out)
    if report.get("report_sha256"):
        print(f"report_sha256 {report['report_sha256']}")
    return 0


def _cmd_verify(ns: argparse.Namespace) -> int:
    from memaudit.report import verify_report

    result = verify_report(ns.report_json)
    for check in result["checks"]:
        status = "PASS" if check["ok"] else ("SKIP" if check["ok"] is None else "FAIL")
        print(f"{status}  {check['check']}: {check['detail']}")
    if result["ok"]:
        print(f"OK  {ns.report_json} report_sha256={result['recomputed_sha256']}")
        print(f"note: {result['note']}")
        return 0
    print(
        f"memaudit: verification FAILED for {ns.report_json} "
        f"(recomputed {result['recomputed_sha256']})",
        file=sys.stderr,
    )
    return 1


def _cmd_report(ns: argparse.Namespace) -> int:
    from memaudit.compliance import render_annex_markdown

    path = ns.report_json or ns.annex_path
    if not path:
        return _die(
            "pass the report JSON: `memaudit report <report.json>` "
            "(or `memaudit report --annex <report.json>`)"
        )
    try:
        report = load_json(path)
    except FileNotFoundError:
        return _die(f"report not found: {path}")
    except Exception as exc:
        return _die(f"could not read report JSON {path}: {exc}")
    if not isinstance(report, dict):
        return _die(f"{path} is not a JSON object report")
    markdown = render_annex_markdown(report)
    if ns.output:
        dest = Path(ns.output)
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_text(markdown, encoding="utf-8")
        print(str(dest))
    else:
        print(markdown)
    return 0


def _cmd_doctor(ns: argparse.Namespace) -> int:
    from memaudit.doctor import run_doctor

    ok = run_doctor(report_path=ns.report, output_dir=ns.output_dir, skip_demo=ns.skip_demo)
    return 0 if ok else 1


def _cmd_demo(ns: argparse.Namespace) -> int:
    from memaudit.demo import run_demo

    report = run_demo(
        output_dir=ns.output_dir,
        lora=ns.lora,
        seed=ns.seed,
    )
    path = report.get("_demo_report_path") or ""
    if path:
        print(path)
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="memaudit",
        description=(
            "Training-data memorization auditor for Hugging Face Trainer / TRL fine-tunes. "
            "Runs locally; does not phone home."
        ),
    )
    parser.add_argument(
        "--version",
        action="version",
        version=f"memaudit {package_version() or TOOL_VERSION}",
    )
    sub = parser.add_subparsers(dest="command")

    audit = sub.add_parser(
        "audit",
        help="Score a saved model against an inject() manifest",
        description=(
            "Post-hoc / deferred audit. --canary-set and --manifest are aliases "
            "for the same file: the JSON returned by inject() "
            "(written as memaudit-manifest.json). Do not pass a raw "
            "generate_canaries() dump -- it has no inclusion coins."
        ),
    )
    audit.add_argument("--model", required=True, help="Fine-tuned model or adapter directory")
    audit.add_argument(
        "--canary-set",
        "--manifest",
        dest="canary_set",
        required=True,
        help=(
            "inject() manifest JSON (memaudit-manifest.json). "
            "--manifest is an alias; both names accept the same file."
        ),
    )
    audit.add_argument("--dataset", default=None, help="Raw train JSON/JSONL for real-record sampling")
    audit.add_argument(
        "--held-out",
        default=None,
        help=(
            "Optional genuine held-out JSON/JSONL. Required for an inferential "
            "member-vs-nonmember real-record test; without it, real-record "
            "scores are descriptive ranking only."
        ),
    )
    audit.add_argument(
        "--profile",
        default=None,
        choices=["smoke", "routine", "powered"],
        help=(
            "Named audit profile. smoke refuses a TPR@FPR headline; "
            "routine is the moderate default shape; powered requires "
            "calibration stability. When omitted, inferred from counts."
        ),
    )
    audit.add_argument(
        "--target-fpr",
        dest="target_fpr",
        type=float,
        default=None,
        help="Override the profile FPR used to calibrate the membership threshold (default 0.01).",
    )
    audit.add_argument(
        "--scorer",
        default=None,
        help=(
            "Membership scorer backend (default: min_k_plus_plus). "
            "Built-in name or import path package.module:Class. "
            "Calibration, CI, and TPR@FPR stay in orchestration."
        ),
    )
    audit.add_argument(
        "--ref",
        default="auto",
        help=(
            "auto (unmerged LoRA via disable_adapter), none (target-only, "
            "downgraded headline), or a path to the base checkpoint. "
            "auto refuses to silently fall back on a full fine-tune."
        ),
    )
    audit.add_argument("--tokenizer", default=None, help="Tokenizer path if not stored with --model")
    audit.add_argument("--output", default=None, help="Report JSON path")
    audit.add_argument("--real-sample", type=int, default=64)
    audit.add_argument(
        "--skip-generation",
        action="store_true",
        help=(
            "Do not run prefix-prompted regurgitation generation. "
            "The report records regurgitation as not_run with rate null; "
            "this is the absence of a measurement, not a zero-detection result."
        ),
    )
    audit.add_argument(
        "--reveal",
        action="store_true",
        help="Include raw real-record text in the ranked list (default is hash-only)",
    )
    audit.add_argument(
        "--seeds",
        default=None,
        help=(
            "Comma-separated audit seeds (e.g. 0,1,2) for multi-seed mode. "
            "Adds a 'stability' block measuring audit-procedure variance "
            "(bootstrap threshold calibration + real-record sampling), "
            "not training variance. Single-seed remains the default."
        ),
    )
    audit.add_argument(
        "--release-context",
        default=None,
        choices=["public-api", "internal", "open-weights", "unspecified"],
        help=(
            "User-declared release context (EDPB Opinion 28/2024 para 46): "
            "public-api | internal | open-weights. Default: unspecified."
        ),
    )
    audit.set_defaults(func=_cmd_audit)

    verify = sub.add_parser(
        "verify",
        help="Recompute and check a report's self-hash (report_sha256 + sidecar)",
        description=(
            "Recomputes the canonical-content SHA-256 of a memaudit report and "
            "checks it against the embedded report_sha256 field and the "
            "<report>.sha256 sidecar. Content integrity only: authenticity "
            "requires signing the file (GPG / sigstore) as a runbook step."
        ),
    )
    verify.add_argument("report_json", help="Path to memaudit-report.json")
    verify.set_defaults(func=_cmd_verify)

    render = sub.add_parser(
        "report",
        help="Render the EDPB compliance annex of a report as markdown",
        description=(
            "Renders the compliance_annex section of a memaudit report as "
            "human-readable markdown for handing to a DPO. Reports written "
            "before schema 1.1.0 get the annex reconstructed from their fields."
        ),
    )
    render.add_argument(
        "report_json",
        nargs="?",
        default=None,
        help="Path to memaudit-report.json",
    )
    render.add_argument(
        "--annex",
        dest="annex_path",
        default=None,
        help="Alias for the positional report path (memaudit report --annex report.json)",
    )
    render.add_argument(
        "--output",
        "-o",
        default=None,
        help="Write markdown here instead of stdout",
    )
    render.set_defaults(func=_cmd_report)

    demo = sub.add_parser(
        "demo",
        help="Train a tiny model on planted canaries and print measured metrics",
        description=(
            "One-command local demo. Trains a randomly-initialized tiny causal LM "
            "on injected canaries and writes a real memaudit report. Numbers "
            "are measured on this machine, not paper-scale."
        ),
    )
    demo.add_argument(
        "--output-dir",
        default="examples",
        help="Directory for demo-report.json (default: examples/)",
    )
    demo.add_argument(
        "--lora",
        action="store_true",
        help="Wrap the tiny model in LoRA (requires memaudit[peft])",
    )
    demo.add_argument("--seed", type=int, default=0)
    demo.set_defaults(func=_cmd_demo)

    doctor = sub.add_parser(
        "doctor",
        help="Check the environment and validate a report (buyer acceptance)",
    )
    doctor.add_argument(
        "--report",
        default=None,
        help="Optional report JSON to validate (default: run the tiny demo first)",
    )
    doctor.add_argument("--output-dir", default="examples")
    doctor.add_argument("--skip-demo", action="store_true")
    doctor.set_defaults(func=_cmd_doctor)
    return parser


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="[memaudit] %(message)s")
    parser = build_parser()
    try:
        ns = parser.parse_args(argv)
    except SystemExit as exc:
        return int(exc.code) if exc.code is not None else 2
    if not getattr(ns, "command", None):
        parser.print_help()
        return 2
    try:
        return int(ns.func(ns))
    except MemauditError as exc:
        return _die(str(exc))
    except FileNotFoundError as exc:
        return _die(str(exc))


if __name__ == "__main__":
    sys.exit(main())
