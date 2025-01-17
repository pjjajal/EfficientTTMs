import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.layers import Mlp


class ResidualFusion(nn.Module):
    """
    A class representing the residual fusion module.

    This module fuses read/write outputs (rw) into a target residual stream (target).

    Args:
        rw (torch.Tensor): read/write output.
        target (torch.Tensor): target residual stream.

    Returns:
        torch.Tensor: fused residual streams.
    """

    def forward(self, rw: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return target + rw


class EraseFusion(nn.Module):
    def forward(self, rw: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return rw


class AddEraseFusion(nn.Module):
    def __init__(self, embed_dim: int, target_size: int) -> None:
        super().__init__()
        self.target_size = target_size
        self.linear = nn.Linear(embed_dim, target_size)

    def forward(self, rw: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        gate_token = rw.mean(dim=1)
        gate = torch.sigmoid(self.linear(gate_token)).unsqueeze(-1)
        return target * (1 - gate) + rw * gate


class FuseMlp(nn.Module):
    def __init__(self, embed_dim, mlp_ratio, act_layer, drop_rate, norm_layer) -> None:
        super().__init__()
        self.norm = norm_layer(embed_dim)
        self.mlp = Mlp(
            in_features=embed_dim,
            hidden_features=int(mlp_ratio * embed_dim),
            out_features=embed_dim,
            act_layer=act_layer,
            drop=drop_rate,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.mlp(self.norm(x))
