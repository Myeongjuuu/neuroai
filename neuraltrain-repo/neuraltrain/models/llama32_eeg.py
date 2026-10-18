# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

"""A trainable EEG prefix for a frozen Llama 3.2 causal language model."""

from __future__ import annotations

import typing as tp

import torch
from torch import nn

from .base import BaseBrainModelConfig


class Llama32EEG(BaseBrainModelConfig):
    """Score two one-token answers after an EEG prefix and a task prompt.

    ``answers`` must follow the NeuralBench target index order. The Llama
    parameters are frozen; gradients pass through it to the EEG prefix encoder.
    """

    model_name: str = "meta-llama/Llama-3.2-3B-Instruct"
    prompt: str = (
        "Classify this EEG segment. Is the person doing mental arithmetic "
        "(A) or resting (B)? Reply with A or B only."
    )
    answers: tuple[str, str] = ("A", "B")
    n_eeg_tokens: int = 16
    torch_dtype: tp.Literal["bfloat16", "float16", "float32"] = "bfloat16"

    def build(self, n_spatial_locations: int, n_outputs: int | None) -> nn.Module:
        if n_outputs != len(self.answers):
            raise ValueError(
                f"Llama32EEG requires {len(self.answers)} target classes; "
                f"NeuralBench provided {n_outputs}."
            )
        if self.n_eeg_tokens < 1:
            raise ValueError("n_eeg_tokens must be positive")
        return Llama32EEGModel(
            n_channels=n_spatial_locations,
            model_name=self.model_name,
            prompt=self.prompt,
            answers=self.answers,
            n_eeg_tokens=self.n_eeg_tokens,
            torch_dtype=self.torch_dtype,
        )


class Llama32EEGModel(nn.Module):
    """Return class logits taken directly from Llama's next-token logits."""

    # Both factory forwards are expensive on a 3B model and initialize no
    # lazy layers here. The regular training/evaluation forward still runs.
    skip_dummy_forward = True
    skip_torchinfo_summary = True
    omit_frozen_backbone_from_checkpoint = True

    def __init__(
        self,
        *,
        n_channels: int,
        model_name: str,
        prompt: str,
        answers: tuple[str, str],
        n_eeg_tokens: int,
        torch_dtype: str,
    ) -> None:
        super().__init__()
        from transformers import AutoModelForCausalLM, AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(model_name)
        prompt_ids = tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=True,
            add_generation_prompt=True,
        )
        if not prompt_ids:
            raise ValueError("Llama chat template produced an empty prompt")
        answer_ids = []
        for answer in answers:
            ids = tokenizer.encode(answer, add_special_tokens=False)
            if len(ids) != 1:
                raise ValueError(
                    f"Answer {answer!r} must be exactly one Llama token; got {ids}."
                )
            answer_ids.append(ids[0])
        if len(set(answer_ids)) != len(answer_ids):
            raise ValueError("Class answers must map to different tokens")

        self.register_buffer("prompt_ids", torch.tensor(prompt_ids, dtype=torch.long))
        self.register_buffer("answer_ids", torch.tensor(answer_ids, dtype=torch.long))
        self.llama = AutoModelForCausalLM.from_pretrained(
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

    def train(self, mode: bool = True) -> Llama32EEGModel:
        super().train(mode)
        self.llama.eval()  # Keep the frozen decoder deterministic during training.
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
        logits = self.llama(inputs_embeds=inputs_embeds, use_cache=False).logits
        return logits[:, -1, self.answer_ids].float()
