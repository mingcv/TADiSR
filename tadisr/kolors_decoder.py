"""Joint mask decoder used by the released Kolors TADiSR adapter.

This is the compact decoder architecture used for the FTSR Kolors checkpoint.
It is intentionally independent of the full Kolors pipeline so checkpoint
compatibility can be verified without downloading the base model.
"""

from __future__ import annotations

import torch
from torch import nn

try:
    from diffusers.models.unet_2d_blocks import UpDecoderBlock2D
except ImportError:
    from diffusers.models.unets.unet_2d_blocks import UpDecoderBlock2D


class LayerNormFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, weight, bias, eps):
        ctx.eps = eps
        _, channels, _, _ = x.size()
        mean = x.mean(1, keepdim=True)
        variance = (x - mean).pow(2).mean(1, keepdim=True)
        normalized = (x - mean) / (variance + eps).sqrt()
        ctx.save_for_backward(normalized, variance, weight)
        return weight.view(1, channels, 1, 1) * normalized + bias.view(1, channels, 1, 1)

    @staticmethod
    def backward(ctx, grad_output):
        normalized, variance, weight = ctx.saved_tensors
        scaled_grad = grad_output * weight.view(1, -1, 1, 1)
        grad_input = (scaled_grad - normalized * (scaled_grad * normalized).mean(1, keepdim=True)
                      - scaled_grad.mean(1, keepdim=True)) / torch.sqrt(variance + ctx.eps)
        grad_weight = (grad_output * normalized).sum(dim=(0, 2, 3))
        grad_bias = grad_output.sum(dim=(0, 2, 3))
        return grad_input, grad_weight, grad_bias, None


class LayerNorm2d(nn.Module):
    def __init__(self, channels: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(channels))
        self.bias = nn.Parameter(torch.zeros(channels))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return LayerNormFunction.apply(x, self.weight, self.bias, self.eps)


class CABlock(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.ca = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Conv2d(channels, channels, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.ca(x)


class DualStreamGate(nn.Module):
    def forward(self, x: torch.Tensor, y: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x1, x2 = x.chunk(2, dim=1)
        y1, y2 = y.chunk(2, dim=1)
        return x1 * y2, y1 * x2


class DualStreamSeq(nn.Sequential):
    def forward(self, x: torch.Tensor, y: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor]:
        y = x if y is None else y
        for module in self:
            x, y = module(x, y)
        return x, y


class DualStreamSepBlock(nn.Module):
    def __init__(self, *modules: nn.Module):
        super().__init__()
        self.seq_l = nn.Sequential(*modules)
        self.seq_r = nn.Sequential(*modules)

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return self.seq_l(x), self.seq_r(y)


class MuGIBlockOri(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.block1 = DualStreamSeq(
            DualStreamSepBlock(
                LayerNorm2d(channels),
                nn.Conv2d(channels, channels * 2, 1),
                nn.Conv2d(channels * 2, channels * 2, 3, padding=1, groups=channels * 2),
            ),
            DualStreamGate(),
            DualStreamSepBlock(CABlock(channels)),
            DualStreamSepBlock(nn.Conv2d(channels, channels, 1)),
        )
        self.a_l = nn.Parameter(torch.zeros((1, channels, 1, 1)))
        self.a_r = nn.Parameter(torch.zeros((1, channels, 1, 1)))
        self.block2 = DualStreamSeq(
            DualStreamSepBlock(LayerNorm2d(channels), nn.Conv2d(channels, channels * 2, 1)),
            DualStreamGate(),
            DualStreamSepBlock(nn.Conv2d(channels, channels, 1)),
        )
        self.b_l = nn.Parameter(torch.zeros((1, channels, 1, 1)))
        self.b_r = nn.Parameter(torch.zeros((1, channels, 1, 1)))

    def forward(self, left: torch.Tensor, right: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        left_update, right_update = self.block1(left, right)
        left_skip = left + left_update * 0
        right_skip = right + right_update * self.a_r
        left_update, right_update = self.block2(left_skip, right_skip)
        return left_skip + left_update * 0, right_skip + right_update * self.b_r


class MaskInteractionDecoderKolors640M(nn.Module):
    """Exact mask-interaction decoder for ``model_526000.pkl``."""

    def __init__(self):
        super().__init__()
        channels = (512, 512, 512, 256)
        self.interaction_blocks = nn.ModuleList(
            [DualStreamSeq(MuGIBlockOri(channel)) for channel in channels]
        )
        self.up_blocks = nn.ModuleList(
            [
                UpDecoderBlock2D(num_layers=2, in_channels=512, out_channels=512, add_upsample=True,
                                 resnet_eps=1e-6, resnet_act_fn="silu", resnet_groups=32,
                                 resnet_time_scale_shift="group"),
                UpDecoderBlock2D(num_layers=2, in_channels=512, out_channels=512, add_upsample=True,
                                 resnet_eps=1e-6, resnet_act_fn="silu", resnet_groups=32,
                                 resnet_time_scale_shift="group"),
                UpDecoderBlock2D(num_layers=2, in_channels=512, out_channels=256, add_upsample=True,
                                 resnet_eps=1e-6, resnet_act_fn="silu", resnet_groups=32,
                                 resnet_time_scale_shift="group"),
                UpDecoderBlock2D(num_layers=2, in_channels=256, out_channels=256, add_upsample=False,
                                 resnet_eps=1e-6, resnet_act_fn="silu", resnet_groups=32,
                                 resnet_time_scale_shift="group"),
            ]
        )
        self.out_block = nn.Sequential(
            nn.GroupNorm(num_channels=256, num_groups=32, eps=1e-6),
            nn.SiLU(),
            nn.Conv2d(256, 1, 3, padding=1),
        )
