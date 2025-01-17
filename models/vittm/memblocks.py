import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.layers import Mlp
from timm.layers import Mlp

from typing import Optional, Tuple, Union, Callable
from functools import partial
from .misc.convnext1d import Upsample, ConvNeXtBlock1D, Downsample


class MemMlpBlock(nn.Module):
    def __init__(self, embed_dim, hidden_dim, drop) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim

        self.norm = nn.LayerNorm(embed_dim)
        self.mlp = Mlp(
            embed_dim,
            hidden_dim,
            embed_dim,
            act_layer=nn.GELU,
            drop=drop,
        )

    def forward(self, x):
        x = x + self.mlp(self.norm(x))
        return x


class ConvNeXtEncoder(nn.Module):
    def __init__(self, embed_dim, depths=[3, 3], downsamples=[2, 1]) -> None:
        super().__init__()
        channels = self.channels = embed_dim // 4

        # Create ConvNeXt Blocks
        self.blocks = nn.ModuleList()
        for depth, downsample in zip(depths, downsamples):
            stage = nn.Sequential(
                *[
                    ConvNeXtBlock1D(
                        in_chs=channels,
                        out_chs=channels,
                        kernel_size=7,
                        stride=1,
                        use_grn=True,
                        mlp_ratio=4,
                        conv_mlp=True,
                    )
                    for _ in range(depth)
                ]
            )
            downblock = Downsample(channels, channels, stride=downsample)
            self.blocks.append(downblock)
            self.blocks.append(stage)

        # Up and Down Projections
        self.down_proj = nn.Conv1d(embed_dim, channels, kernel_size=1)
        self.up_proj = nn.Conv1d(channels, embed_dim, kernel_size=1)

    def forward(self, x):
        B, N, C = x.shape
        x = x.transpose(1, 2)
        x = self.down_proj(x)

        down_shapes = []
        for block in self.blocks:
            if isinstance(block, Downsample):
                down_shapes.append(x.shape[-1:])
            x = block(x)
        x = self.up_proj(x)
        x = x.transpose(1, 2)
        return x, down_shapes


class ConvNeXtDecoder(nn.Module):
    def __init__(self, embed_dim, depths=[3, 3]) -> None:
        super().__init__()
        channels = self.channels = embed_dim // 4

        # Create ConvNeXt Blocks
        self.blocks = nn.ModuleList()
        for depth in depths:
            stage = nn.Sequential(
                *[
                    ConvNeXtBlock1D(
                        in_chs=channels,
                        out_chs=channels,
                        kernel_size=7,
                        stride=1,
                        mlp_ratio=4,
                        use_grn=True,
                        conv_mlp=True,
                    )
                    for _ in range(depth)
                ]
            )
            upblock = Upsample()
            self.blocks.append(stage)
            self.blocks.append(upblock)

        # Up and Down Projections
        self.down_proj = nn.Conv1d(embed_dim, channels, kernel_size=1)
        self.mix = nn.Conv1d(
            channels, channels, kernel_size=7, groups=channels, padding="same"
        )
        self.up_proj = nn.Conv1d(channels, embed_dim, kernel_size=1)

    def forward(self, x, upshapes=None):
        B, N, C = x.shape
        x = x.transpose(1, 2)
        x = self.down_proj(x)

        for block in self.blocks:
            if isinstance(block, Upsample):
                x = block(x, upshapes.pop())
            else:
                x = block(x)
        x = self.mix(x)
        x = self.up_proj(x)
        x = x.transpose(1, 2)
        return x


def create_mem_blocks(memory_block, **kwargs):
    if memory_block == "":
        mem_blocks = nn.ModuleList([nn.Identity() for _ in range(kwargs["depth"])])
        return mem_blocks
    elif memory_block == "mlp":
        mem_blocks = nn.ModuleList(
            [
                MemMlpBlock(
                    kwargs["embed_dim"],
                    int(kwargs["embed_dim"] * kwargs["memory_mlp_ratio"]),
                    kwargs["proj_drop_rate"],
                )
                for _ in range(kwargs["depth"])
            ]
        )
        return mem_blocks
    elif memory_block == "convnext":
        mem_encoder = nn.ModuleList(
            [
                ConvNeXtEncoder(
                    kwargs["embed_dim"],
                    depths=kwargs["memory_encoder_depths"],
                    downsamples=kwargs["memory_encoder_downsamples"],
                )
                for _ in range(kwargs["depth"])
            ]
        )
        mem_decoder = nn.ModuleList(
            [
                ConvNeXtDecoder(
                    kwargs["embed_dim"],
                    depths=kwargs["memory_decoder_depths"],
                )
                for _ in range(kwargs["depth"])
            ]
        )
        return mem_encoder, mem_decoder


if __name__ == "__main__":
    model = ConvNeXtEncoder(768, [3, 3], downsamples=[2, 2])
    x = torch.randn(1, 52, 768)
    y, up_shapes = model(x)
    print(y.shape)

    model = ConvNeXtDecoder(768, [3, 3])
    y = model(y, up_shapes)
    print(y.shape)
