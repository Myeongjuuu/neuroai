"""Checkpoint behavior for models with a separately loaded frozen Llama."""

import torch
from torch import nn

from neuralbench.pl_module import BrainModule


class _AdapterModel(nn.Module):
    omit_frozen_backbone_from_checkpoint = True

    def __init__(self):
        super().__init__()
        self.llama = nn.Linear(3, 3)
        self.llama.requires_grad_(False)
        self.eeg_projector = nn.Linear(3, 3)

    def forward(self, x):
        return self.eeg_projector(x)


def test_llama_checkpoint_saves_adapter_and_strictly_restores_it():
    module = BrainModule(
        model=_AdapterModel(),
        loss=nn.CrossEntropyLoss(),
        metrics={},
        lightning_optimizer_config=None,
    )
    checkpoint = {"state_dict": module.state_dict()}
    module.on_save_checkpoint(checkpoint)
    assert not any(k.startswith("model.llama.") for k in checkpoint["state_dict"])
    assert any(k.startswith("model.eeg_projector.") for k in checkpoint["state_dict"])
    restored = BrainModule(
        model=_AdapterModel(),
        loss=nn.CrossEntropyLoss(),
        metrics={},
        lightning_optimizer_config=None,
    )
    restored.load_state_dict(checkpoint["state_dict"], strict=True)
    assert torch.equal(
        restored.model.eeg_projector.weight, module.model.eeg_projector.weight
    )
