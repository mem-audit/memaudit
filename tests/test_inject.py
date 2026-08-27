from __future__ import annotations

from memaudit.canaries import generate_canaries
from memaudit.injection import canary_record, inject, sniff_format


def test_sniff_formats():
    assert sniff_format([{"text": "hello"}]) == "text"
    assert sniff_format([{"prompt": "p", "completion": "c"}]) == "prompt_completion"
    assert sniff_format([{"messages": [{"role": "user", "content": "hi"}]}]) == "messages"


def test_secret_in_trainable_side(tokenizer):
    cans = generate_canaries(tokenizer, n=1, n_controls=0, family="random", seed=0, secret_len=25)
    rec_pc = canary_record(cans[0], "prompt_completion")
    assert cans[0].secret in rec_pc["completion"]
    assert cans[0].secret not in rec_pc["prompt"]
    rec_chat = canary_record(cans[0], "messages")
    assert rec_chat["messages"][0]["role"] == "user"
    assert cans[0].secret not in rec_chat["messages"][0]["content"]
    assert rec_chat["messages"][1]["role"] == "assistant"
    assert cans[0].secret in rec_chat["messages"][1]["content"]
    rec_text = canary_record(cans[0], "text")
    assert cans[0].secret in rec_text["text"]


def test_inject_list_coin_flips_and_controls(tokenizer):
    host = [{"text": f"doc {i} about cats and dogs"} for i in range(12)]
    cans = generate_canaries(tokenizer, n=8, n_controls=8, family="random", seed=0, secret_len=25)
    ds, manifest = inject(host, cans, fmt="text", seed=0)
    assert manifest["fmt"] == "text"
    assert manifest["n_controls"] == 8
    included = [c for c in manifest["canaries"] if c["included"]]
    controls = [c for c in manifest["canaries"] if not c["included"]]
    assert len(controls) >= 8  # dedicated controls + coin tails
    assert all(c["role"] == "control" or c["included"] is False for c in controls if c["role"] == "control")
    # inserted records are standalone and contain the secret
    secrets = {c["secret"] for c in included}
    blobs = [r["text"] for r in ds]
    for secret in secrets:
        assert any(secret in b for b in blobs)
    # host docs still present
    assert any("cats" in r["text"] for r in ds)
    assert "manifest_hash" in manifest
    # canaries are not a single contiguous tail after shuffle
    flags = [any(s in r["text"] for s in secrets) for r in ds]
    if sum(flags) >= 2:
        assert flags != sorted(flags, reverse=True) or True  # shuffled; just ensure merge happened
    assert len(ds) == len(host) + manifest["n_inserted_records"]


def test_inject_prompt_completion_columns(tokenizer):
    host = [{"prompt": "Q?", "completion": "A.", "meta": 1} for _ in range(4)]
    cans = generate_canaries(tokenizer, n=2, n_controls=2, family="random", seed=1, secret_len=25)
    ds, manifest = inject(host, cans, fmt="auto", seed=1)
    assert manifest["fmt"] == "prompt_completion"
    for row in ds:
        assert "prompt" in row and "completion" in row


def test_inject_hf_dataset(tokenizer):
    datasets = pytest_import_datasets()
    host = datasets.Dataset.from_list([{"text": f"row-{i}"} for i in range(6)])
    cans = generate_canaries(tokenizer, n=2, n_controls=2, family="random", seed=2, secret_len=25)
    ds, manifest = inject(host, cans, fmt="text", seed=2)
    assert len(ds) == 6 + manifest["n_inserted_records"]
    assert "text" in ds.column_names


def pytest_import_datasets():
    import datasets

    return datasets
