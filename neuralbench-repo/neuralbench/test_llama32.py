"""Checkpoint behavior for models with a separately loaded frozen Llama."""

import torch
from torch import nn

from neuralbench.pl_module import BrainModule
from neuralbench.experiment_config import build_experiment_configs
from neuralbench.registry import TASKS


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


def test_llama32_config_assembles_for_every_eeg_task():
    configs = build_experiment_configs("eeg", "all", model="llama32", debug=True)
    assert {config["task_name"] for config in configs} == set(TASKS["eeg"])
    assert all(
        config["brain_model_config"]["name"] == "Llama32EEG" for config in configs
    )
    variant = build_experiment_configs(
        "eeg", "mental_arithmetic", model="llama32", dataset="shin2017b", debug=True
    )
    assert variant
