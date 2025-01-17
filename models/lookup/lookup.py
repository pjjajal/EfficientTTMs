from functools import partial
from typing import Callable, Literal, Optional, Tuple, Type, Union

import torch
import torch.nn as nn
from timm.layers import (
    Mlp,
    PatchEmbed,
    get_act_layer,
    get_norm_layer,
    trunc_normal_,
)
from timm.layers.typing import LayerType
from timm.models._builder import build_model_with_cfg
from timm.models.deit import VisionTransformerDistilled
from timm.models.vision_transformer import (
    Block,
    VisionTransformer,
    checkpoint_filter_fn,
    resample_abs_pos_embed,
)
from typing_extensions import Literal


class CrossAttention(nn.Module):
    def __init__(
        self,
        embed_dim,
        num_heads=8,
    ) -> None:
        super().__init__()
        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads."
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scale = self.head_dim**-0.5

        self.q = nn.Linear(embed_dim, embed_dim)
        self.q_norm = nn.LayerNorm(embed_dim)

        self.k = nn.Linear(embed_dim, embed_dim)
        self.k_norm = nn.LayerNorm(embed_dim)

        self.v = nn.Linear(embed_dim, embed_dim)
        self.v_norm = nn.LayerNorm(embed_dim)

    def forward(
        self, input_tokens: torch.Tensor, query_tokens: torch.Tensor, attn=None
    ) -> torch.Tensor:
        B, N, C = input_tokens.shape
        Bq, M, Cq = query_tokens.shape

        v = (
            self.v(input_tokens)
            .reshape(B, N, self.num_heads, self.head_dim)
            .permute(0, 2, 1, 3)
        )

        if attn is not None:
            v = (
                self.v_norm(self.v(input_tokens))
                .reshape(B, N, self.num_heads, self.head_dim)
                .permute(0, 2, 1, 3)
            )
            v = (attn.mT @ v).transpose(1, 2).reshape(B, M, C)
            return query_tokens + v, None

        v = (
            self.v(input_tokens)
            .reshape(B, N, self.num_heads, self.head_dim)
            .permute(0, 2, 1, 3)
        )


        q = (
            self.q_norm(self.q(query_tokens))
            .reshape(B, M, self.num_heads, self.head_dim)
            .permute(0, 2, 1, 3)
        )

        k = (
            self.k_norm(self.k(input_tokens))
            .reshape(B, N, self.num_heads, self.head_dim)
            .permute(0, 2, 1, 3)
        )

        q = q * self.scale
        attn = q @ k.transpose(-2, -1)
        attn = attn.softmax(dim=-1)
        output_tokens = attn @ v

        output_tokens = output_tokens.transpose(1, 2).reshape(B, M, C)

        return query_tokens + output_tokens, attn


class LookupViT(VisionTransformer):
    def __init__(
        self,
        img_size: Union[int, Tuple[int, int]] = 224,
        patch_size: Union[int, Tuple[int, int]] = 16,
        in_chans: int = 3,
        num_classes: int = 1000,
        global_pool: Literal["", "avg", "token", "map"] = "token",
        embed_dim: int = 768,
        depth: int = 12,
        num_heads: int = 12,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
        qk_norm: bool = False,
        init_values: Optional[float] = None,
        class_token: bool = True,
        no_embed_class: bool = False,
        reg_tokens: int = 0,
        pre_norm: bool = False,
        fc_norm: Optional[bool] = None,
        dynamic_img_size: bool = False,
        dynamic_img_pad: bool = False,
        drop_rate: float = 0.0,
        pos_drop_rate: float = 0.0,
        patch_drop_rate: float = 0.0,
        proj_drop_rate: float = 0.0,
        attn_drop_rate: float = 0.0,
        drop_path_rate: float = 0.0,
        weight_init: Literal["skip", "jax", "jax_nlhb", "moco", ""] = "",
        fix_init: bool = False,
        embed_layer: Callable = PatchEmbed,
        norm_layer: Optional[LayerType] = None,
        act_layer: Optional[LayerType] = None,
        block_fn: Type[nn.Module] = Block,
        mlp_layer: Type[nn.Module] = Mlp,
        **kwargs,
    ) -> None:
        super().__init__(
            img_size,
            patch_size,
            in_chans,
            num_classes,
            global_pool,
            embed_dim,
            depth,
            num_heads,
            mlp_ratio,
            qkv_bias,
            qk_norm,
            init_values,
            class_token,
            no_embed_class,
            reg_tokens,
            pre_norm,
            fc_norm,
            dynamic_img_size,
            dynamic_img_pad,
            drop_rate,
            pos_drop_rate,
            patch_drop_rate,
            proj_drop_rate,
            attn_drop_rate,
            drop_path_rate,
            weight_init,
            fix_init,
            embed_layer,
            norm_layer,
            act_layer,
            block_fn,
            mlp_layer,
        )

        norm_layer = get_norm_layer(norm_layer) or partial(nn.LayerNorm, eps=1e-6)
        act_layer = get_act_layer(act_layer) or nn.GELU

        self.compressed_tokens = kwargs.pop("compressed_tokens", None)
        self.downsample = nn.AdaptiveAvgPool2d(
            (self.compressed_tokens, self.compressed_tokens)
        )

        compressed_size = self.compressed_tokens**2
        self.compressed_pos_embed = nn.Parameter(
            torch.randn(1, compressed_size, embed_dim) * 0.02
        )

        self.compressed_norm = norm_layer(embed_dim)
        self.lookup_norm = norm_layer(embed_dim)

        self.mhbc = CrossAttention(embed_dim, num_heads)
        self.global_mlp = Mlp(
            embed_dim, hidden_features=int(embed_dim / 2), act_layer=act_layer
        )

        self.compressed_head = nn.Linear(embed_dim, num_classes)
        self.fc_norm_compressed = norm_layer(embed_dim)

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        lookup = self.patch_embed(x)
        compressed = self.downsample(lookup.permute(0, 3, 1, 2))
        compressed = compressed.flatten(2).transpose(1, 2)
        compressed = compressed + self.compressed_pos_embed

        lookup = self._pos_embed(lookup)

        lookup = self.patch_drop(lookup)
        lookup = self.norm_pre(lookup)

        for i, block in enumerate(self.blocks):
            lookup, compressed = self.lookup_norm(lookup), self.compressed_norm(
                compressed
            )
            compressed, attn = self.mhbc.forward(lookup, compressed)
            compressed = block(compressed)
            compressed = self.compressed_norm(compressed)
            lookup, _ = self.mhbc.forward(compressed, lookup, attn)
            lookup = lookup + self.global_mlp(lookup)
        return lookup, compressed

    def forward_head(self, lookup, compressed, pre_logits=False):
        lookup = lookup.mean(dim=1)
        compressed = compressed.mean(dim=1)

        lookup = self.fc_norm(lookup)
        compressed = self.fc_norm_compressed(compressed)

        compressed = self.head_drop(compressed)
        lookup = self.head_drop(lookup)
        if pre_logits:
            return lookup, compressed
        return self.head(lookup), self.compressed_head(compressed)

    def forward(
        self, x: torch.Tensor, pre_logits: bool = False, **kwargs
    ) -> torch.Tensor:
        lookup, compressed = self.forward_features(x)
        lookup_pred, compressed_pred = self.forward_head(
            lookup, compressed, pre_logits=pre_logits
        )
        return lookup_pred, compressed_pred



def _create_lookup(
    variant: str, pretrained: bool = False, **kwargs
) -> VisionTransformer:
    out_indices = kwargs.pop("out_indices", 3)
    if "flexi" in variant:
        # FIXME Google FlexiViT pretrained models have a strong preference for bilinear patch / embed
        # interpolation, other pretrained models resize better w/ anti-aliased bicubic interpolation.
        _filter_fn = partial(
            checkpoint_filter_fn, interpolation="bilinear", antialias=False
        )
    else:
        _filter_fn = checkpoint_filter_fn

    # FIXME attn pool (currently only in siglip) params removed if pool disabled, is there a better soln?
    strict = False
    if "siglip" in variant and kwargs.get("global_pool", None) != "map":
        strict = False

    return build_model_with_cfg(
        LookupViT,
        variant,
        pretrained,
        pretrained_filter_fn=_filter_fn,
        pretrained_strict=strict,
        feature_cfg=dict(out_indices=out_indices, feature_cls="getter"),
        **kwargs,
    )


def lookupvit_base_patch16_224(pretrained: bool = False, **kwargs) -> VisionTransformer:
    """ViT-Base (ViT-B/16) from original paper (https://arxiv.org/abs/2010.11929).
    ImageNet-1k weights fine-tuned from in21k @ 224x224, source https://github.com/google-research/vision_transformer.
    """
    model_args = dict(patch_size=16, embed_dim=768, depth=12, num_heads=12)
    model = _create_lookup(
        "vit_base_patch16_224", pretrained=pretrained, **dict(model_args, **kwargs)
    )
    return model