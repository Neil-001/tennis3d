"""Independent checkpoint-compatible TrackNet encoder/decoder.

Architecture and checkpoint key mapping follow the pinned TrackNet/court assets.
No reference repository is imported or distributed with this module.
"""

from pathlib import Path

import torch
from torch import nn


class HeatmapNet(nn.Module):
    def __init__(self, input_channels: int, output_channels: int):
        super().__init__()
        channels = [
            64,
            64,
            128,
            128,
            256,
            256,
            256,
            512,
            512,
            512,
            256,
            256,
            256,
            128,
            128,
            64,
            64,
            output_channels,
        ]
        blocks = []
        for output in channels:
            blocks.append(
                nn.Sequential(
                    nn.Conv2d(input_channels, output, 3, padding=1),
                    nn.ReLU(),
                    nn.BatchNorm2d(output),
                )
            )
            input_channels = output
        self.blocks = nn.ModuleList(blocks)

    def forward(self, x):
        for index, block in enumerate(self.blocks, start=1):
            x = block(x)
            if index in (2, 4, 7):
                x = nn.functional.max_pool2d(x, 2)
            if index in (10, 13, 15):
                x = nn.functional.interpolate(x, scale_factor=2, mode="nearest")
        return x


def load_network(path: Path, kind: str, device: str):
    model = HeatmapNet(9 if kind == "ball" else 3, 256 if kind == "ball" else 15)
    source = torch.load(path, map_location="cpu", weights_only=True)
    expected = model.state_dict()
    mapping = {}
    for key in expected:
        _, index, layer, field = key.split(".")
        source_key = f"conv{int(index) + 1}.block.{layer}.{field}"
        mapping[source_key] = key
    if set(source) != set(mapping):
        raise ValueError(f"Unexpected checkpoint keys in {path}")
    model.load_state_dict(
        {mapping[key]: value for key, value in source.items()}, strict=True
    )
    return model.eval().to(device)
