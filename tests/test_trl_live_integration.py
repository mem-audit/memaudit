"""Tier-2 live TRL SFTTrainer fixtures: real prepared datasets + real collators.

Complements the Tier-1 constructed-geometry fixtures in
``test_preflight_evidence.py`` (which never depend on upstream bugs existing).
These tests drive the *installed* TRL end-to-end and are era-aware:

- TRL 0.29.x (pinned ``.venv-buyer``): mask era. Prepared datasets carry
  ``completion_mask`` / ``assistant_masks`` and **no** ``labels`` column;
  labels materialize inside the data collator.
- TRL >=1.9 (scratch ``.venv-trl19``): labels era. Prepared datasets carry a
  ``labels`` column, and (non-packing, ``max_length`` set) a prep filter
  silently *deletes* fully-masked rows from mixed datasets.

Run:  pytest -m integration tests/test_trl_live_integration.py
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

pytestmark = pytest.mark.integration

trl = pytest.importorskip("trl")
pytest.importorskip("torch")
pytest.importorskip("datasets")

from datasets import Dataset  # noqa: E402
from packaging.version import Version  # noqa: E402

from memaudit.canaries import generate_canaries  # noqa: E402
from memaudit.exceptions import MemauditPreflightError  # noqa: E402
from memaudit.injection import ASSISTANT_PREFIX, TEXT_PREFIX, canary_record, inject  # noqa: E402
from memaudit.preflight import _canary_record_n_tokens, run_preflight, survival_scan  # noqa: E402
from memaudit.preflight_labels import _labels_from_collator, effective_labels  # noqa: E402
from memaudit.types import Canary  # noqa: E402
from memaudit.utils import encode_ids, example_text, find_subsequence  # noqa: E402

TRL_VERSION = Version(trl.__version__)
LABELS_ERA = TRL_VERSION >= Version("1.9.0")

GPT2 = "distilgpt2"
LLAMA_TOK = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"

# TinyLlama's shipped template has no {% generation %} markers; TRL requires
# them for assistant_only_loss. Same rendering, generation-marked assistant.
GENERATION_TEMPLATE = (
    "{% for message in messages %}"
    "{% if message['role'] == 'user' %}"
    "{{ '<|user|>\n' + message['content'] + eos_token + '\n' }}"
    "{% elif message['role'] == 'system' %}"
    "{{ '<|system|>\n' + message['content'] + eos_token + '\n' }}"
    "{% elif message['role'] == 'assistant' %}"
    "{{ '<|assistant|>\n' }}{% generation %}{{ message['content'] + eos_token }}"
    "{% endgeneration %}{{ '\n' }}"
    "{% endif %}"
    "{% endfor %}"
    "{% if add_generation_prompt %}{{ '<|assistant|>\n' }}{% endif %}"
)


@pytest.fixture(scope="module")
def gpt2_tok():
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(GPT2, use_fast=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    return tok


@pytest.fixture(scope="module")
def llama_tok():
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(LLAMA_TOK, use_fast=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.chat_template = GENERATION_TEMPLATE
    return tok


def _tiny_llama(vocab_size: int):
    from transformers import LlamaConfig, LlamaForCausalLM

    cfg = LlamaConfig(
        vocab_size=int(vocab_size),
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=2,
        num_attention_heads=2,
        num_key_value_heads=2,
        max_position_embeddings=1024,
    )
    return LlamaForCausalLM(cfg)


def _sft_trainer(tok, rows, tmp_path, **cfg):
    from trl import SFTConfig, SFTTrainer

    args = SFTConfig(
        output_dir=str(tmp_path / "sft"),
        report_to=[],
        per_device_train_batch_size=2,
        **cfg,
    )
    return SFTTrainer(
        model=_tiny_llama(len(tok)),
        args=args,
        train_dataset=Dataset.from_list(rows),
        processing_class=tok,
    )


def _manifest(tok, fmt: str, *, n: int = 1, seed: int = 0, secret_len: int = 25):
    cans = generate_canaries(
        tok, n=n, n_controls=0, family="random", seed=seed, secret_len=secret_len
    )
    dummy = {
        "text": [{"text": "host"}],
        "prompt_completion": [{"prompt": "p", "completion": " c"}],
        "messages": [
            {
                "messages": [
                    {"role": "user", "content": "hi"},
                    {"role": "assistant", "content": "hello"},
                ]
            }
        ],
    }[fmt]
    _, manifest = inject(dummy, cans, fmt=fmt, seed=seed, include_prob=1.0, tokenizer=tok)
    return manifest


def _preflight(trainer, manifest, tok, *, raise_fatal: bool = True):
    return run_preflight(
        model=trainer.model,
        trainer=trainer,
        manifest=manifest,
        tokenizer=tok,
        args=trainer.args,
        train_dataset=trainer.train_dataset,
        raise_fatal=raise_fatal,
    )


def _row_all_masked(row) -> bool:
    labels = row.get("labels")
    if labels is not None:
        return all(int(x) == -100 for x in labels)
    mask = row.get("completion_mask") or row.get("assistant_masks")
    return mask is not None and all(int(m) == 0 for m in mask)


# ---------------------------------------------------------------------------
# Survivor paths: the real prepared dataset + real collator yield labels and
# the canary reaches directly_supervised. This is the WS1.1 enabling move
# verified against the installed TRL, not a stub.
# ---------------------------------------------------------------------------


def test_live_prompt_completion_survivor_directly_supervised(gpt2_tok, tmp_path):
    cans = generate_canaries(
        gpt2_tok, n=2, n_controls=0, family="random", seed=101, secret_len=25
    )
    host = [{"prompt": f"Question {i}?", "completion": f" Answer {i}."} for i in range(8)]
    ds_rows, manifest = inject(
        host, cans, fmt="prompt_completion", seed=101, include_prob=1.0, tokenizer=gpt2_tok
    )
    trainer = _sft_trainer(
        gpt2_tok,
        list(ds_rows),
        tmp_path,
        max_length=512,
        completion_only_loss=True,
        packing=False,
    )

    cols = trainer.train_dataset.column_names
    if LABELS_ERA:
        assert "labels" in cols
    else:
        # Mask era: no labels column exists before collation. The old
        # row["labels"] check was dead code against this exact TRL.
        assert "labels" not in cols
        assert "completion_mask" in cols

    out = _preflight(trainer, manifest, gpt2_tok)
    assert not out["fatal"]
    per = out["per_canary"]
    assert len(per) == 2
    for ev in per:
        assert ev["record_observed"] is True
        assert ev["secret_token_aligned"] is True
        assert ev["directly_supervised"] is True
        assert ev["evidence_level"] == "directly_supervised"
        assert ev["supervised_token_fraction"] == 1.0
        expected_source = "dataset_column" if LABELS_ERA else "collator_call"
        assert ev["labels_source"] == expected_source


def test_live_assistant_only_loss_survivor_directly_supervised(llama_tok, tmp_path):
    cans = generate_canaries(
        llama_tok, n=1, n_controls=0, family="random", seed=102, secret_len=25
    )
    host = [
        {
            "messages": [
                {"role": "user", "content": f"Say hello number {i}."},
                {"role": "assistant", "content": f"Hello there, number {i}!"},
            ]
        }
        for i in range(6)
    ]
    ds_rows, manifest = inject(
        host, cans, fmt="messages", seed=102, include_prob=1.0, tokenizer=llama_tok
    )
    trainer = _sft_trainer(
        llama_tok,
        list(ds_rows),
        tmp_path,
        max_length=512,
        assistant_only_loss=True,
        packing=False,
    )
    out = _preflight(trainer, manifest, llama_tok)
    assert not out["fatal"]
    ev = out["per_canary"][0]
    assert ev["secret_token_aligned"] is True
    assert ev["directly_supervised"] is True
    assert ev["evidence_level"] == "directly_supervised"


def test_live_packed_survivor_directly_supervised(gpt2_tok, tmp_path):
    cans = generate_canaries(
        gpt2_tok, n=1, n_controls=0, family="random", seed=103, secret_len=25
    )
    host = [{"prompt": f"Q{i}?", "completion": f" A{i}."} for i in range(12)]
    ds_rows, manifest = inject(
        host, cans, fmt="prompt_completion", seed=103, include_prob=1.0, tokenizer=gpt2_tok
    )
    trainer = _sft_trainer(
        gpt2_tok,
        list(ds_rows),
        tmp_path,
        max_length=96,
        completion_only_loss=True,
        packing=True,
    )
    out = _preflight(trainer, manifest, gpt2_tok)
    assert not out["fatal"]
    ev = out["per_canary"][0]
    assert ev["record_observed"] is True
    assert ev["split_across_packed_rows"] is False
    assert ev["directly_supervised"] is True
    assert ev["supervised_token_fraction"] == 1.0


# ---------------------------------------------------------------------------
# Fixture A - truncation kills the supervised completion span (#6668 shape).
# ---------------------------------------------------------------------------


def _fixture_a_rows(manifest):
    secret = manifest["canaries"][0]["secret"]
    long_prompt = " ".join(f"filler{i}" for i in range(120))
    canary_row = {"prompt": long_prompt, "completion": f"{ASSISTANT_PREFIX}{secret}"}
    survivors = [{"prompt": f"Short {i}?", "completion": f" Reply {i}."} for i in range(3)]
    return [canary_row, *survivors]


def test_live_fixture_a_truncation_kills_completion_is_fatal(gpt2_tok, tmp_path):
    manifest = _manifest(gpt2_tok, "prompt_completion", seed=104)
    rows = _fixture_a_rows(manifest)
    # max_length=48 clears the static record-length pre-check (the standalone
    # canary record is ~39 tokens), so the fatal below must come from the live
    # survival scan, not from the length heuristic.
    trainer = _sft_trainer(
        gpt2_tok,
        rows,
        tmp_path,
        max_length=48,
        completion_only_loss=True,
        packing=False,
    )

    prepared = [trainer.train_dataset[i] for i in range(len(trainer.train_dataset))]
    if LABELS_ERA:
        # TRL >=1.9 prep filter silently deleted the fully-masked canary row
        # from the mixed dataset; the survivors keep training.
        assert len(prepared) == len(rows) - 1
    else:
        # TRL 0.29.x keeps the row; truncation zeroed its completion_mask.
        assert len(prepared) == len(rows)
        dead = [r for r in prepared if _row_all_masked(r)]
        assert len(dead) == 1
        labels = _labels_from_collator(trainer.data_collator, dict(dead[0]))
        assert labels is not None
        assert set(labels) == {-100}  # the silent loss:0.0 geometry, live

    out = _preflight(trainer, manifest, gpt2_tok, raise_fatal=False)
    assert out["fatal"]
    assert not any("max_length" in f for f in out["fatal"])
    assert any(
        "raw text columns" in f or "complete inspection" in f for f in out["fatal"]
    )
    ev = out["per_canary"][0]
    assert ev["directly_supervised"] is False
    if not LABELS_ERA:
        # TRL 0.29.x keeps prompt/completion string columns on the prepared
        # dataset; the secret string-matches there while its tokens are gone
        # from input_ids. That must be fatal, not verification_unknown noise.
        assert ev["token_stream_missing"] is True

    with pytest.raises(MemauditPreflightError):
        _preflight(trainer, manifest, gpt2_tok)


@pytest.mark.skipif(not LABELS_ERA, reason="row-deletion filter exists only on TRL >= 1.9")
def test_live_trl19_row_deletion_reported_fatal_not_scan_noise(gpt2_tok, tmp_path):
    manifest = _manifest(gpt2_tok, "prompt_completion", seed=105)
    rows = _fixture_a_rows(manifest)
    trainer = _sft_trainer(
        gpt2_tok,
        rows,
        tmp_path,
        max_length=48,
        completion_only_loss=True,
        packing=False,
    )
    assert len(trainer.train_dataset) == len(rows) - 1

    out = _preflight(trainer, manifest, gpt2_tok, raise_fatal=False)
    assert out["fatal"]
    assert any("complete inspection" in f for f in out["fatal"])
    ev = out["per_canary"][0]
    assert ev["record_observed"] is False
    assert ev["miss_reason"] == "not_found_after_complete_scan"
    # Deletion must not be attributed to the bounded scan window.
    assert not any("scan window" in f for f in out["fatal"])


@pytest.mark.skipif(not LABELS_ERA, reason="labels column exists only on TRL >= 1.9")
def test_live_trl19_labels_column_and_collator_agree(gpt2_tok, tmp_path):
    cans = generate_canaries(
        gpt2_tok, n=1, n_controls=0, family="random", seed=106, secret_len=25
    )
    host = [{"prompt": f"Question {i}?", "completion": f" Answer {i}."} for i in range(4)]
    ds_rows, manifest = inject(
        host, cans, fmt="prompt_completion", seed=106, include_prob=1.0, tokenizer=gpt2_tok
    )
    trainer = _sft_trainer(
        gpt2_tok,
        list(ds_rows),
        tmp_path,
        max_length=512,
        completion_only_loss=True,
        packing=False,
    )
    row = next(
        dict(trainer.train_dataset[i])
        for i in range(len(trainer.train_dataset))
        if not _row_all_masked(trainer.train_dataset[i])
        and any(int(x) != -100 for x in trainer.train_dataset[i]["labels"])
    )
    got = effective_labels(row, trainer, {"completion_only_loss_effective": True})
    assert got is not None
    assert got.source == "dataset_column"
    # The live collator path must also produce labels in the labels era
    # (collator-first ordering stays valid after an upgrade).
    from_collator = _labels_from_collator(trainer.data_collator, row)
    if from_collator is not None:
        assert [int(x) for x in from_collator] == [int(x) for x in got.labels]


# ---------------------------------------------------------------------------
# Fixture B1 - assistant mask zeroed by truncation (#3927 shape).
# ---------------------------------------------------------------------------


def test_live_fixture_b1_assistant_mask_zeroed_is_fatal(llama_tok, tmp_path):
    manifest = _manifest(llama_tok, "messages", seed=107)
    secret = manifest["canaries"][0]["secret"]
    long_user = " ".join(f"word{i}" for i in range(150))
    rows = [
        {
            "messages": [
                {"role": "user", "content": long_user},
                {"role": "assistant", "content": f"{ASSISTANT_PREFIX}{secret}"},
            ]
        },
        {
            "messages": [
                {"role": "user", "content": "hi"},
                {"role": "assistant", "content": "hello there friend"},
            ]
        },
    ]
    trainer = _sft_trainer(
        llama_tok,
        rows,
        tmp_path,
        max_length=96,
        assistant_only_loss=True,
        packing=False,
    )

    prepared = [trainer.train_dataset[i] for i in range(len(trainer.train_dataset))]
    if LABELS_ERA:
        assert len(prepared) == 1  # canary row deleted by the prep filter
    else:
        assert len(prepared) == 2
        dead = [r for r in prepared if _row_all_masked(r)]
        assert len(dead) == 1  # the mask survived as a column but is all-zero

    out = _preflight(trainer, manifest, llama_tok, raise_fatal=False)
    assert out["fatal"]
    assert any(
        "raw text columns" in f or "complete inspection" in f or "-100" in f
        for f in out["fatal"]
    )
    with pytest.raises(MemauditPreflightError):
        _preflight(trainer, manifest, llama_tok)


# ---------------------------------------------------------------------------
# Fixture C - bfd packing produces pure-prompt chunks that train at silent
# loss 0.0 on every TRL version including current main (PR #6671 still open).
# ---------------------------------------------------------------------------


def test_live_fixture_c_bfd_prompt_only_chunks_is_fatal(gpt2_tok, tmp_path):
    manifest = _manifest(gpt2_tok, "prompt_completion", seed=108)
    secret = manifest["canaries"][0]["secret"]
    long_prompt = " ".join(f"pad{i}" for i in range(40))
    rows = [{"prompt": long_prompt, "completion": f"{ASSISTANT_PREFIX}{secret}"}]
    rows += [{"prompt": long_prompt, "completion": " tail."} for _ in range(15)]
    trainer = _sft_trainer(
        gpt2_tok,
        rows,
        tmp_path,
        max_length=16,
        completion_only_loss=True,
        packing=True,
    )

    prepared = [dict(trainer.train_dataset[i]) for i in range(len(trainer.train_dataset))]
    assert prepared, "packing produced no rows"
    assert all(_row_all_masked(r) for r in prepared)  # pure-prompt packed chunks
    labels = _labels_from_collator(trainer.data_collator, prepared[0])
    if labels is not None:
        assert set(labels) == {-100}

    out = _preflight(trainer, manifest, gpt2_tok, raise_fatal=False)
    assert out["fatal"]
    assert any(
        "complete inspection" in f or "raw text columns" in f or "-100" in f
        for f in out["fatal"]
    )
    with pytest.raises(MemauditPreflightError):
        _preflight(trainer, manifest, gpt2_tok)


# ---------------------------------------------------------------------------
# Real-BPE alignment ladder (distilgpt2): exact -> junction -> char-cover.
# ---------------------------------------------------------------------------


def test_real_bpe_junction_needles_catch_boundary_merges(gpt2_tok):
    cans = generate_canaries(
        gpt2_tok, n=6, n_controls=0, family="random", seed=109, secret_len=25
    )
    host = [{"text": f"ordinary host row {i}"} for i in range(4)]
    ds_rows, manifest = inject(
        host, cans, fmt="text", seed=109, include_prob=1.0, tokenizer=gpt2_tok
    )
    inserted = [c for c in manifest["canaries"] if c["included"]]

    # At least one canary must actually exercise the merge case: its standalone
    # secret ids are not a subsequence of the in-context encoding.
    merged = []
    for c in inserted:
        ctx_ids = encode_ids(gpt2_tok, f"{TEXT_PREFIX}{c['secret']}", add_special_tokens=False)
        if find_subsequence(ctx_ids, c["secret_token_ids"]) is None:
            merged.append(c["id"])
    assert merged, "seed produced no BPE boundary merge; pick another seed"

    scan = survival_scan(list(ds_rows), manifest, tokenizer=gpt2_tok)
    assert scan["n_found"] == len(inserted)
    assert scan["token_level_hits"] == len(inserted)
    by_id = {e["id"]: e for e in scan["per_canary"]}
    for c in inserted:
        ev = by_id[c["id"]]
        assert ev["secret_token_aligned"] is True
        assert ev["alignment"] in {"exact", "covering"}
    for cid in merged:
        # merge cases must be caught by the ladder, not reported missing
        assert by_id[cid]["record_observed"] is True


def test_real_bpe_char_cover_catches_mid_merge_start(gpt2_tok):
    # "h" + "ellophone" re-merges into " hello" + "phone" under GPT-2 BPE, so
    # the secret's first token is destroyed in context: exact and junction
    # needles must fail and the char-cover fallback must still align it.
    secret = "ellophone rotor 9137 canary phrase"
    secret_ids = encode_ids(gpt2_tok, secret, add_special_tokens=False)
    assert gpt2_tok.decode(secret_ids) == secret  # roundtrip-stable

    row_text = f"The greeting was h{secret} and nothing else."
    row_ids = encode_ids(gpt2_tok, row_text, add_special_tokens=False)
    assert find_subsequence(row_ids, secret_ids) is None, (
        "construction failed: secret survived in-context tokenization intact"
    )

    manifest = {
        "fmt": "text",
        "canaries": [
            {
                "id": "c-mid-merge",
                "family": "random",
                "secret": secret,
                "secret_token_ids": secret_ids,
                "included": True,
                "repetitions": 1,
                "role": "candidate",
            }
        ],
    }
    scan = survival_scan([{"text": row_text}], manifest, tokenizer=gpt2_tok)
    ev = scan["per_canary"][0]
    assert ev["record_observed"] is True
    assert ev["secret_token_aligned"] is True
    assert ev["alignment"] == "covering"


# ---------------------------------------------------------------------------
# max_length pre-check counts real chat-template overhead.
# ---------------------------------------------------------------------------


def test_chat_template_overhead_counted_in_max_length_precheck(llama_tok, tiny_model):
    cans = generate_canaries(
        llama_tok, n=1, n_controls=0, family="random", seed=111, secret_len=32
    )
    host = [
        {
            "messages": [
                {"role": "user", "content": "hi"},
                {"role": "assistant", "content": "hello"},
            ]
        }
    ]
    ds_rows, manifest = inject(
        host, cans, fmt="messages", seed=111, include_prob=1.0, tokenizer=llama_tok
    )
    c = manifest["canaries"][0]
    blob = example_text(canary_record(Canary.from_dict(dict(c)), "messages"), "messages")
    raw_n = len(encode_ids(llama_tok, blob, add_special_tokens=False))
    full_n = _canary_record_n_tokens(c, "messages", llama_tok)
    assert full_n > raw_n, "chat-template overhead was not counted"

    # A max_length between the raw-blob count and the templated count used to
    # false-pass; it must be fatal now.
    boundary = raw_n + 1
    assert boundary <= full_n
    args = SimpleNamespace(max_length=boundary, packing=False, packing_strategy=None)
    with pytest.raises(MemauditPreflightError, match="max_length"):
        run_preflight(
            model=tiny_model,
            trainer=None,
            manifest=manifest,
            tokenizer=llama_tok,
            args=args,
            train_dataset=list(ds_rows),
            raise_fatal=True,
        )
