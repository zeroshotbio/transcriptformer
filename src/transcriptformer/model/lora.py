"""Lightweight LoRA utilities."""

from __future__ import annotations

import math
from collections.abc import Iterator

import torch
from torch import nn


class LoRALinear(nn.Module):
    """A linear layer augmented with LoRA weights."""

    def __init__(self, in_features: int, out_features: int, r: int = 4, alpha: float = 1.0, bias: bool = True) -> None:
        super().__init__()
        self.linear = nn.Linear(in_features, out_features, bias=bias)
        for p in self.linear.parameters():
            p.requires_grad = False

        self.weight_a = nn.Parameter(torch.zeros(r, in_features))
        self.weight_b = nn.Parameter(torch.zeros(out_features, r))
        nn.init.kaiming_uniform_(self.weight_a, a=math.sqrt(5))
        nn.init.zeros_(self.weight_b)
        self.scaling = alpha / r

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Compute the linear transformation with LoRA adaptation."""
        result = self.linear(x)
        lora_out = nn.functional.linear(x, self.weight_b @ self.weight_a) * self.scaling
        return result + lora_out


def _replace_module(parent: nn.Module, child_name: str, new_module: nn.Module) -> None:
    setattr(parent, child_name, new_module)


def apply_lora(module: nn.Module, r: int = 4, alpha: float = 1.0) -> None:
    """Recursively replace ``nn.Linear`` modules with :class:`LoRALinear`."""
    for name, child in list(module.named_children()):
        if isinstance(child, nn.Linear):
            lora_layer = LoRALinear(child.in_features, child.out_features, r=r, alpha=alpha, bias=child.bias is not None)
            lora_layer.linear.weight.data.copy_(child.weight.data)
            if child.bias is not None:
                lora_layer.linear.bias.data.copy_(child.bias.data)
            _replace_module(module, name, lora_layer)
        else:
            apply_lora(child, r=r, alpha=alpha)


def iter_lora_layers(module: nn.Module) -> Iterator[tuple[str, LoRALinear]]:
    """Yield names and layers for all :class:`LoRALinear` modules."""
    for name, child in module.named_modules():
        if isinstance(child, LoRALinear):
            yield name, child


def lora_state_dict(module: nn.Module) -> dict[str, torch.Tensor]:
    """Return a state_dict containing only LoRA parameters."""
    state = {}
    for name, layer in iter_lora_layers(module):
        state[f"{name}.weight_a"] = layer.weight_a.detach().cpu()
        state[f"{name}.weight_b"] = layer.weight_b.detach().cpu()
    return state
