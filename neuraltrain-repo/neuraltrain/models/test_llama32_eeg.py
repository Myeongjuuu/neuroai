"""Small model test without downloading a gated Llama checkpoint."""

import torch
from torch import nn
import pytest

pytest.importorskip("transformers")

from neuraltrain.models.llama32_eeg import Llama32EEGModel


class _Tokenizer:
    def apply_chat_template(self, *args, **kwargs):
        return [1, 2, 3]

    def encode(self, text, **kwargs):
        return {"A": [4], "B": [5]}[text]


class _LanguageModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.config = type("Config", (), {"hidden_size": 12, "use_cache": True})()
        self.embedding = nn.Embedding(8, 12)
        self.decoder = nn.GRU(12, 12, batch_first=True)
        self.head = nn.Linear(12, 8)

    def get_input_embeddings(self):
        return self.embedding

    def gradient_checkpointing_enable(self, **kwargs):
        pass

    def forward(self, *, inputs_embeds, use_cache):
        assert not use_cache
        states, _ = self.decoder(inputs_embeds)
        return type("Output", (), {"logits": self.head(states)})()


def test_llama32_eeg_scores_answers_and_trains_prefix(monkeypatch):
    import transformers

    monkeypatch.setattr(
        transformers.AutoTokenizer, "from_pretrained", lambda *a, **k: _Tokenizer()
    )
    monkeypatch.setattr(
        transformers.AutoModelForCausalLM,
        "from_pretrained", lambda *a, **k: _LanguageModel()
    )
    model = Llama32EEGModel(
        n_channels=3,
        model_name="fake-llama",
        prompt="Choose A or B",
        answers=("A", "B"),
        n_eeg_tokens=4,
        torch_dtype="float32",
    )
    model.train()
    logits = model(torch.randn(2, 3, 40))
    assert logits.shape == (2, 2)
    assert torch.isfinite(logits).all()
    assert not model.llama.training
    logits.sum().backward()
    assert any(p.grad is not None for p in model.eeg_projector.parameters())
    assert all(p.grad is None for p in model.llama.parameters())
