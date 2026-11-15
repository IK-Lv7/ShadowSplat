"""SegFormer-B0 (Cityscapes) edge model wrapper.

Input : RGB image with pixel values in 0-255. Mean/std normalization is part of the model.
Output: logits (N, 19, H/4, W/4). The energy score is computed from the logits separately.
"""
from __future__ import annotations

from pathlib import Path

import torch
import yaml
from torch import nn

ROOT = Path(__file__).resolve().parents[1]


class SegformerLogits(nn.Module):
    """Takes 0-255 pixel values and returns logits, with normalization built in."""

    def __init__(self, hf_model: nn.Module, mean: list[float], std: list[float]) -> None:
        super().__init__()
        self.model = hf_model
        self.register_buffer("mean", torch.tensor(mean).view(1, 3, 1, 1) * 255.0)
        self.register_buffer("std", torch.tensor(std).view(1, 3, 1, 1) * 255.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(pixel_values=(x - self.mean) / self.std).logits


def load_cfg(path: Path) -> dict:
    return yaml.safe_load(path.read_text())


def build(cfg: dict) -> SegformerLogits:
    from transformers import SegformerForSemanticSegmentation

    m = cfg["model"]
    hf = SegformerForSemanticSegmentation.from_pretrained(m["hf_id"]).eval()
    return SegformerLogits(hf, m["mean"], m["std"]).eval()
