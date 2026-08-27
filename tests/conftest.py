from __future__ import annotations

import re
import zlib
from types import SimpleNamespace

import pytest


class FakeTokenizer:
    """Identity-ish tokenizer: ``t12 t13`` round-trips to ``[12, 13]``."""

    def __init__(self, vocab_size: int = 200) -> None:
        self.vocab_size = vocab_size
        self.pad_token_id = 0
        self.eos_token_id = 1
        self.bos_token_id = 2
        self.unk_token_id = 3
        self.all_special_ids = [0, 1, 2, 3]

    def encode(self, text: str, add_special_tokens: bool = False):  # noqa: ARG002
        ids: list[int] = []
        for part in str(text).replace("\n", " ").split():
            matched = re.fullmatch(r"t(\d+)", part)
            if matched:
                ids.append(int(matched.group(1)))
            else:
                ids.append(4 + (zlib.adler32(part.encode("utf-8")) % (self.vocab_size - 4)))
        if add_special_tokens:
            ids = [self.bos_token_id, *ids, self.eos_token_id]
        return ids

    def decode(self, ids, skip_special_tokens: bool = True, clean_up_tokenization_spaces: bool = False):  # noqa: ARG002
        out = []
        for i in ids:
            i = int(i)
            if skip_special_tokens and i in self.all_special_ids:
                continue
            out.append(f"t{i}")
        return " ".join(out)

    def __call__(self, text, add_special_tokens=False, **kwargs):  # noqa: ANN001, ARG002
        return {"input_ids": self.encode(text, add_special_tokens=add_special_tokens)}

    def __len__(self) -> int:
        return self.vocab_size


class TinyCausalLM:
    """Minimal causal LM for CPU smoke tests (no pretrained weights)."""

    def __init__(self, vocab_size: int = 200, hidden: int = 32) -> None:
        import torch
        import torch.nn as nn

        self._torch = torch
        self.embed = nn.Embedding(vocab_size, hidden)
        self.ln = nn.LayerNorm(hidden)
        self.lm_head = nn.Linear(hidden, vocab_size, bias=False)
        self.config = SimpleNamespace(vocab_size=vocab_size, use_cache=False, tie_word_embeddings=False)
        self.training = True
        # register modules for parameters()
        self._mods = nn.ModuleList([self.embed, self.ln, self.lm_head])

    def parameters(self):
        return self._mods.parameters()

    def named_parameters(self):
        return self._mods.named_parameters()

    def get_input_embeddings(self):
        return self.embed

    def get_output_embeddings(self):
        return self.lm_head

    def eval(self):
        self.training = False
        self._mods.eval()
        return self

    def train(self, mode: bool = True):
        self.training = mode
        self._mods.train(mode)
        return self

    def to(self, device):
        self._mods.to(device)
        return self

    def forward(self, input_ids, attention_mask=None, labels=None, **kwargs):  # noqa: ANN001, ARG002
        import torch.nn.functional as F

        h = self.ln(self.embed(input_ids))
        logits = self.lm_head(h)
        loss = None
        if labels is not None:
            loss = F.cross_entropy(
                logits[:, :-1].reshape(-1, logits.size(-1)),
                labels[:, 1:].reshape(-1),
                ignore_index=-100,
            )
        return SimpleNamespace(logits=logits, loss=loss)

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

    def generate(self, input_ids, max_new_tokens=8, do_sample=False, **kwargs):  # noqa: ANN001, ARG002
        import torch

        out = input_ids
        for _ in range(int(max_new_tokens)):
            logits = self.forward(out).logits[:, -1]
            nxt = torch.argmax(logits, dim=-1, keepdim=True)
            out = torch.cat([out, nxt], dim=-1)
        return out


@pytest.fixture
def tokenizer() -> FakeTokenizer:
    return FakeTokenizer()


@pytest.fixture
def tiny_model() -> TinyCausalLM:
    return TinyCausalLM()
