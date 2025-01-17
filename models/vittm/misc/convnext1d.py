import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.layers import Mlp
from timm.models.convnext import ConvNeXtBlock, Downsample
from timm.layers import (
    trunc_normal_,
    AvgPool2dSame,
    DropPath,
    Mlp,
    GlobalResponseNormMlp,
    LayerNorm2d,
    LayerNorm,
    create_conv2d,
    get_act_layer,
    make_divisible,
    to_ntuple,
)

from functools import partial

from typing import Optional, Tuple, Union, Callable


class GlobalResponseNorm1D(nn.Module):
    """Global Response Normalization layer"""

    def __init__(self, dim, eps=1e-6, channels_last=True):
        super().__init__()
        self.eps = eps
        if channels_last:
            self.spatial_dim = 1
            self.channel_dim = -1
            self.wb_shape = (1, 1, -1)
        else:
            self.spatial_dim = 2
            self.channel_dim = 1
            self.wb_shape = (1, -1, 1)

        self.weight = nn.Parameter(torch.zeros(dim))
        self.bias = nn.Parameter(torch.zeros(dim))

    def forward(self, x):
        x_g = x.norm(p=2, dim=self.spatial_dim, keepdim=True)
        x_n = x_g / (x_g.mean(dim=self.channel_dim, keepdim=True) + self.eps)
        return x + torch.addcmul(
            self.bias.view(self.wb_shape), self.weight.view(self.wb_shape), x * x_n
        )


class GlobalResponseNormMlp(nn.Module):
    """MLP w/ Global Response Norm (see grn.py), nn.Linear or 1x1 Conv2d"""

    def __init__(
        self,
        in_features,
        hidden_features=None,
        out_features=None,
        act_layer=nn.GELU,
        bias=True,
        drop=0.0,
        use_conv=False,
    ):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        linear_layer = partial(nn.Conv1d, kernel_size=1) if use_conv else nn.Linear

        self.fc1 = linear_layer(in_features, hidden_features, bias=bias)
        self.act = act_layer()
        self.drop1 = nn.Dropout(drop)
        self.grn = GlobalResponseNorm1D(hidden_features, channels_last=not use_conv)
        self.fc2 = linear_layer(hidden_features, out_features, bias=bias)
        self.drop2 = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop1(x)
        x = self.grn(x)
        x = self.fc2(x)
        x = self.drop2(x)
        return x


class ConvNeXtBlock1D(nn.Module):
    def __init__(
        self,
        in_chs: int,
        out_chs: Optional[int] = None,
        kernel_size: int = 7,
        stride: int = 1,
        dilation: Union[int, Tuple[int, int]] = (1, 1),
        mlp_ratio: float = 4,
        conv_mlp: bool = False,
        conv_bias: bool = True,
        use_grn: bool = False,
        ls_init_value: Optional[float] = 1e-6,
        act_layer: Union[str, Callable] = "gelu",
        norm_layer: Optional[Callable] = None,
        drop_path: float = 0.0,
    ):
        """
        Args:
            in_chs: Block input channels.
            out_chs: Block output channels (same as in_chs if None).
            kernel_size: Depthwise convolution kernel size.
            stride: Stride of depthwise convolution.
            dilation: Tuple specifying input and output dilation of block.
            mlp_ratio: MLP expansion ratio.
            conv_mlp: Use 1x1 convolutions for MLP and a NCHW compatible norm layer if True.
            conv_bias: Apply bias for all convolution (linear) layers.
            use_grn: Use GlobalResponseNorm in MLP (from ConvNeXt-V2)
            ls_init_value: Layer-scale init values, layer-scale applied if not None.
            act_layer: Activation layer.
            norm_layer: Normalization layer (defaults to LN if not specified).
            drop_path: Stochastic depth probability.
        """
        super().__init__()
        out_chs = out_chs or in_chs
        dilation = to_ntuple(2)(dilation)
        act_layer = get_act_layer(act_layer)
        if not norm_layer:
            norm_layer = LayerNorm
        mlp_layer = partial(
            GlobalResponseNormMlp if use_grn else Mlp, use_conv=conv_mlp
        )
        self.use_conv_mlp = conv_mlp
        self.conv_dw = nn.Conv1d(
            in_chs,
            out_chs,
            kernel_size=kernel_size,
            stride=stride,
            groups=in_chs,
            dilation=dilation[0],
            bias=conv_bias,
            padding="same",
        )
        self.norm = norm_layer(out_chs)
        self.mlp = mlp_layer(out_chs, int(mlp_ratio * out_chs), act_layer=act_layer)
        self.gamma = (
            nn.Parameter(ls_init_value * torch.ones(out_chs))
            if ls_init_value is not None
            else None
        )
        if in_chs != out_chs or stride != 1 or dilation[0] != dilation[1]:
            self.shortcut = Downsample(
                in_chs, out_chs, stride=stride, dilation=dilation[0]
            )
        else:
            self.shortcut = nn.Identity()
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

    def forward(self, x):
        shortcut = x
        x = self.conv_dw(x)
        if self.use_conv_mlp:
            x = self.norm(x.transpose(1, 2)).transpose(1, 2)
            x = self.mlp(x)
        else:
            x = x.permute(0, 2, 1)
            x = self.norm(x)
            x = self.mlp(x)
            x = x.permute(0, 1, 2)
        if self.gamma is not None:
            x = x.mul(self.gamma.reshape(1, -1, 1))

        x = self.drop_path(x) + self.shortcut(shortcut)
        return x


class Upsample(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, shape):
        return F.interpolate(x, size=shape, mode="nearest")


class Downsample(nn.Module):
    def __init__(self, in_chs, out_chs, stride=1, dilation=1):
        super().__init__()
        avg_stride = stride if dilation == 1 else 1
        if stride > 1 or dilation > 1:
            self.pool = nn.AvgPool1d(2, avg_stride, ceil_mode=True, count_include_pad=False)
        else:
            self.pool = nn.Identity()

        if in_chs != out_chs:
            self.conv = nn.Conv1d(in_chs, out_chs, 1, stride=1)
        else:
            self.conv = nn.Identity()

    def forward(self, x):
        x = self.pool(x)
        x = self.conv(x)
        return x