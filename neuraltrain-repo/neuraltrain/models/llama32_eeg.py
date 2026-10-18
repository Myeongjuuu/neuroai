# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

"""Task-conditioned Llama 3.2 backbone for NeuralBench EEG tasks."""

from __future__ import annotations

import typing as tp

import torch
from torch import nn

from .base import BaseBrainModelConfig


class Llama32EEG(BaseBrainModelConfig):
    """Use frozen Llama hidden states with a learned EEG prefix and output head.

    The benchmark supplies the output width from its target extractor. Its
    existing loss and metrics then define classification, multilabel,
    regression, or retrieval behavior without model-specific task mappings.
    """

    model_name: str = "meta-llama/Llama-3.2-3B-Instruct"
    prompt_template: str = "Analyze this EEG recording for the {task} task."
    n_eeg_tokens: int = 16
    torch_dtype: tp.Literal["bfloat16", "float16", "float32"] = "bfloat16"

    def build(
        self,
        n_spatial_locations: int,
        n_outputs: int | None,
        task_name: str | None = None,
    ) -> nn.Module:
        if n_outputs is None or n_outputs < 1:
            raise ValueError("Llama32EEG requires a positive output width")
        if self.n_eeg_tokens < 1:
            raise ValueError("n_eeg_tokens must be positive")
        task = (task_name or "EEG decoding").replace("_", " ")
        return Llama32EEGModel(
            n_channels=n_spatial_locations,
            n_outputs=n_outputs,
            model_name=self.model_name,
            prompt=self.prompt_template.format(task=task),
            n_eeg_tokens=self.n_eeg_tokens,
            torch_dtype=self.torch_dtype,
        )


class Llama32EEGModel(nn.Module):
    """Predict benchmark targets from EEG-conditioned Llama representations."""

    # This model has no lazy layers. The factory's CPU dummy forwards are
    # unnecessary and prohibitively expensive for a 3B-parameter backbone.
    skip_dummy_forward = True
    skip_torchinfo_summary = True
    omit_frozen_backbone_from_checkpoint = True

    def __init__(
        self,
        *,
        n_channels: int,
        n_outputs: int,
        model_name: str,
        prompt: str,
        n_eeg_tokens: int,
        torch_dtype: str,
    ) -> None:
        super().__init__()
        from transformers import AutoModel, AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(model_name)
        prompt_ids = tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=True,
            add_generation_prompt=True,
        )
        if not prompt_ids:
            raise ValueError("Llama chat template produced an empty prompt")
        self.register_buffer("prompt_ids", torch.tensor(prompt_ids, dtype=torch.long))
        self.llama = AutoModel.from_pretrained(
            model_name, torch_dtype=getattr(torch, torch_dtype), low_cpu_mem_usage=True
        )
        self.llama.requires_grad_(False)
        self.llama.config.use_cache = False
        self.llama.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False}
        )
        hidden_size = self.llama.config.hidden_size
        self.eeg_pool = nn.AdaptiveAvgPool1d(n_eeg_tokens)
        self.eeg_projector = nn.Sequential(
            nn.LayerNorm(n_channels),
            nn.Linear(n_channels, hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, hidden_size),
            nn.LayerNorm(hidden_size),
        )
        self.output_head = nn.Linear(hidden_size, n_outputs)

    def train(self, mode: bool = True) -> Llama32EEGModel:
        super().train(mode)
        self.llama.eval()
        return self

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 3:
            raise ValueError(f"Expected EEG (batch, channels, time), got {tuple(x.shape)}")
        if x.shape[1] != self.eeg_projector[0].normalized_shape[0]:
            raise ValueError("EEG channel count differs from model build context")
        eeg = self.eeg_pool(x.float()).transpose(1, 2)
        eeg_embeds = self.eeg_projector(eeg)
        text_embeds = self.llama.get_input_embeddings()(
            self.prompt_ids.unsqueeze(0).expand(x.shape[0], -1)
        )
        inputs_embeds = torch.cat(
            [eeg_embeds.to(text_embeds.dtype), text_embeds], dim=1
        )
        hidden = self.llama(inputs_embeds=inputs_embeds, use_cache=False).last_hidden_state
        return self.output_head(hidden[:, -1].float())
