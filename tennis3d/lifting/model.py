# SPDX-License-Identifier: GPL-3.0-only
# RoPE, residual block and XYZ head adapted from UpliftingTableTennis model.py,
# revision 1a64bf6488f3a142f733e73e8de00bfecf044b23. See licenses/README.md for attribution.
"""Random-initialized DCE/DUTA tennis lifter; outputs XYZ divided by 10 m."""

from typing import Literal

import torch
from pydantic import Field, model_validator
from torch import nn
from torch.nn import functional as F

from tennis3d.lifting.features import TIME_BIN_S, Features
from tennis3d.schemas import StrictModel


class ModelConfig(StrictModel):
    # Minimal original configs keep their architecture. The measured recipe YAML
    # records research fields and resolves to stage3-v1 below.
    architecture_version: Literal["legacy-v1", "stage3-v1", "stage4-lstm-v1"] = (
        "legacy-v1"
    )
    lstm_hidden_size: int = Field(default=200, ge=1)
    width: int = Field(default=128, ge=8)
    depth: int = Field(default=8, ge=2)
    heads: int = Field(default=4, ge=1)
    mlp_ratio: int = Field(default=4, ge=1)
    dropout: float = Field(default=0.0, ge=0, lt=1)
    gated_ffn: bool = False
    gated_head: bool = False
    temporal_bias: bool = False
    # Optional optimizer plan; this metadata does not change the forward model.
    schedule_steps: int | None = Field(default=None, ge=1)
    # Optimizer metadata only; no forward change.
    outward_radial_brake: float = Field(default=0.0, ge=0, le=1)
    batched_muon: bool = False

    @model_validator(mode="before")
    @classmethod
    def legacy_architecture(cls, values):
        # Original configs/checkpoints predate all research model metadata.
        # Stage 3 checkpoints record these fields even when their values are false.
        research_fields = {
            "architecture_version",
            "gated_ffn",
            "gated_head",
            "temporal_bias",
            "schedule_steps",
            "outward_radial_brake",
            "batched_muon",
        }
        if isinstance(values, dict) and "architecture_version" not in values:
            version = (
                "stage3-v1" if research_fields.intersection(values) else "legacy-v1"
            )
            return {**values, "architecture_version": version}
        return values

    @model_validator(mode="after")
    def dimensions(self):
        if self.architecture_version == "legacy-v1" and (
            self.gated_ffn or self.gated_head or self.temporal_bias
        ):
            raise ValueError("legacy architecture does not have research gates or bias")
        if self.architecture_version == "stage4-lstm-v1":
            if self.depth != 2 or self.gated_ffn or self.temporal_bias:
                raise ValueError("LSTM requires depth 2 and no attention FFN or bias")
        elif self.width % self.heads or (self.width // self.heads) % 2:
            raise ValueError("width must divide into even-dimensional attention heads")
        return self


class RotaryEmbedding(nn.Module):
    inv_freq: torch.Tensor

    def __init__(self, head_dim: int):
        super().__init__()
        self.register_buffer(
            "inv_freq", 10000 ** (-torch.arange(0, head_dim, 2).float() / head_dim)
        )

    def forward(self, x: torch.Tensor, bins: torch.Tensor) -> torch.Tensor:
        angles = bins.to(self.inv_freq.dtype)[..., None] * self.inv_freq
        cosine, sine = angles[:, None].cos(), angles[:, None].sin()
        even, odd = x[..., 0::2], x[..., 1::2]
        return torch.stack(
            (even * cosine - odd * sine, even * sine + odd * cosine), dim=-1
        ).flatten(-2)


class Attention(nn.Module):
    def __init__(self, *, width: int, heads: int, dropout: float):
        super().__init__()
        self.heads = heads
        self.dropout = dropout
        self.qkv = nn.Linear(width, width * 3)
        self.proj = nn.Linear(width, width)
        self.rope = RotaryEmbedding(width // heads)

    def forward(self, x, bins, keys, distance_bias=None):
        b, t, d = x.shape
        q, k, v = (
            self.qkv(x)
            .reshape(b, t, 3, self.heads, d // self.heads)
            .permute(2, 0, 3, 1, 4)
            .unbind(0)
        )
        mask = keys[:, None, None, :]
        if distance_bias is not None:
            mask = distance_bias.masked_fill(~mask, float("-inf"))
        x = F.scaled_dot_product_attention(
            self.rope(q, bins),
            self.rope(k, bins),
            v,
            attn_mask=mask,
            dropout_p=self.dropout if self.training else 0.0,
        )
        return self.proj(x.transpose(1, 2).reshape(b, t, d))


class SwiGLU(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate, value = x.chunk(2, dim=-1)
        return F.silu(gate) * value


class Block(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        d = config.width
        self.norm1 = nn.LayerNorm(d)
        self.attention = Attention(width=d, heads=config.heads, dropout=config.dropout)
        self.norm2 = nn.LayerNorm(d)
        self.residual_scale = 1.0 if config.architecture_version == "legacy-v1" else 0.5
        hidden = d * config.mlp_ratio
        self.ffn = nn.Sequential(
            nn.Linear(d, hidden * (2 if config.gated_ffn else 1)),
            nn.ReLU()
            if config.architecture_version == "legacy-v1"
            else SwiGLU()
            if config.gated_ffn
            else nn.GELU(approximate="tanh"),
            nn.Dropout(config.dropout),
            nn.Linear(hidden, d),
            nn.Dropout(config.dropout),
        )

    def forward(self, x, bins, keys, temporal_bias=None):
        normalized = self.norm1(x)
        if temporal_bias is None:
            context = self.attention(normalized, bins, keys)
        else:
            context = self.attention(
                normalized, bins, keys, temporal_bias(normalized, bins)
            )
        x = x + self.residual_scale * context
        return x + self.residual_scale * self.ffn(self.norm2(x))


class TemporalBias(nn.Module):
    """Query-dependent proximity preference in physical seconds for each head."""

    def __init__(self, width: int, heads: int):
        super().__init__()
        self.rate = nn.Linear(width, heads)
        nn.init.zeros_(self.rate.weight)
        # Initial e-folding horizons span ten seconds to one tenth of a second.
        rates = torch.logspace(-1, 1, heads)
        with torch.no_grad():
            self.rate.bias.copy_(torch.log(torch.expm1(rates)))

    def forward(self, x, bins):
        distance = (bins[:, :, None] - bins[:, None, :]).abs().to(x.dtype)
        distance = distance * TIME_BIN_S
        rates = F.softplus(self.rate(x)).transpose(1, 2)[..., None]
        return -rates * distance[:, None]


class Lifter(nn.Module):
    def __init__(self, config: ModelConfig | None = None):
        super().__init__()
        self.config = config or ModelConfig()
        d = self.config.width
        self.ball = nn.Linear(2, d)
        self.court = nn.Linear(28, d)
        self.missing_ball = nn.Parameter(torch.zeros(d))
        self.combine = nn.Linear(2 * d, d)
        self.blocks = nn.ModuleList(
            [Block(self.config) for _ in range(self.config.depth)]
            if self.config.architecture_version != "stage4-lstm-v1"
            else []
        )
        if self.config.architecture_version == "stage4-lstm-v1":
            self.time = nn.Linear(2, d)
            self.temporal = nn.LSTM(
                d,
                self.config.lstm_hidden_size,
                num_layers=2,
                batch_first=True,
                bidirectional=True,
                dropout=self.config.dropout,
            )
            self.temporal_projection = nn.Linear(2 * self.config.lstm_hidden_size, d)
        self.head = nn.Sequential(
            nn.Linear(d, d // 2),
            nn.ReLU(),
            nn.Linear(d // 2, d // 4),
            nn.ReLU(),
            nn.Linear(d // 4, 3),
        )
        for layer in self.modules():
            if isinstance(layer, nn.Linear):
                gain = (
                    0.5
                    if layer is self.head[-1]
                    and self.config.architecture_version != "legacy-v1"
                    else 1.0
                )
                nn.init.xavier_uniform_(layer.weight, gain=gain)
                nn.init.zeros_(layer.bias)
        for block in self.blocks:
            assert isinstance(block, Block)
            first = block.ffn[0]
            assert isinstance(first, nn.Linear)
            nn.init.xavier_uniform_(first.weight, gain=1.414)
        for layer in (self.head[0], self.head[2]):
            assert isinstance(layer, nn.Linear)
            nn.init.xavier_uniform_(layer.weight, gain=1.414)
        nn.init.normal_(self.missing_ball, std=0.02)
        if self.config.gated_head:
            # Replace the readout after initialization to preserve core weights.
            self.head = nn.Sequential(nn.Linear(d, 2 * d), SwiGLU(), nn.Linear(d, 3))
            for layer in self.head:
                if isinstance(layer, nn.Linear):
                    gain = 0.5 if layer is self.head[-1] else 1.414
                    nn.init.xavier_uniform_(layer.weight, gain=gain)
                    nn.init.zeros_(layer.bias)

        if self.config.temporal_bias:
            # Install after core initialization and keep rate matrices in Adam.
            self.temporal_bias = nn.ModuleList(
                [TemporalBias(d, self.config.heads) for _ in self.blocks]
            )

    def forward(self, features: Features) -> torch.Tensor:
        real = ~features.padding
        observed = features.ball_observed & real
        if not torch.all((~real) | features.camera_valid):
            raise ValueError(
                "split windows at invalid-camera boundaries before lifting"
            )
        if not torch.all(observed.any(dim=1)):
            raise ValueError("all-missing or all-padding window has no ball evidence")
        # Sanitize before linears, including for callers constructing Features directly.
        ball = torch.where(observed[..., None], features.ball_uv, 0.0)
        court = torch.where(real[..., None, None], features.court_uv, 0.0)
        if not torch.isfinite(ball).all() or not torch.isfinite(court).all():
            raise ValueError("declared-valid model inputs must be finite")
        bins = torch.where(real, features.time_bins, 0)
        ball_embedding = torch.where(
            observed[..., None], self.ball(ball), self.missing_ball
        )
        x = self.combine(
            torch.cat((ball_embedding, self.court(court.flatten(-2))), dim=-1)
        )
        if self.config.architecture_version == "stage4-lstm-v1":
            lengths = real.sum(dim=1)
            prefix = (
                torch.arange(real.shape[1], device=real.device)[None] < lengths[:, None]
            )
            if not torch.equal(real, prefix):
                raise ValueError("LSTM windows require right padding")
            # Subtract the raw float64 origin before casting. Frame indices and
            # quantized bins cannot represent irregular or sub-bin intervals.
            times = torch.where(real, features.t_s, features.t_s[:, :1])
            elapsed = times - times[:, :1]
            delta = torch.cat(
                (torch.zeros_like(times[:, :1]), times[:, 1:] - times[:, :-1]), dim=1
            )
            if not torch.isfinite(elapsed).all() or not torch.all(
                (~real[:, 1:]) | (delta[:, 1:] > 0)
            ):
                raise ValueError("real LSTM timestamps must be finite and increasing")
            timing = torch.stack((elapsed, delta), dim=-1).to(x.dtype)
            timing = torch.where(real[..., None], timing, 0.0)
            x = x + self.time(timing)
            packed = nn.utils.rnn.pack_padded_sequence(
                x,
                lengths.cpu(),
                batch_first=True,
                enforce_sorted=False,
            )
            encoded, _ = self.temporal(packed)
            x, _ = nn.utils.rnn.pad_packed_sequence(
                encoded,
                batch_first=True,
                total_length=real.shape[1],
            )
            x = self.temporal_projection(x)
        for i, block in enumerate(self.blocks):
            keys = observed if i == 0 else real
            if self.config.temporal_bias:
                x = block(x, bins, keys, self.temporal_bias[i])
            else:
                x = block(x, bins, keys)
        return torch.where(real[..., None], self.head(x), 0.0)


def build_lifter(config: ModelConfig | None = None) -> Lifter:
    """Shared architecture factory for training and strict EMA checkpoint loading."""
    return Lifter(config)
