from functools import partial
from typing import Callable, Literal, Optional, Tuple, Type, Union, Any

import torch
import torch.nn as nn
from timm.layers import Mlp, PatchEmbed, get_act_layer, get_norm_layer
from timm.layers.typing import LayerType
from timm.models._builder import build_model_with_cfg
from timm.models.vision_transformer import (
    Block,
    VisionTransformer,
    checkpoint_filter_fn,
)
from torch.nn.modules import Module
from typing_extensions import Literal


class ViT(VisionTransformer):
    def forward(self, x: torch.Tensor, pre_logits: bool = False) -> torch.Tensor:
        x = self.forward_features(x)
        x = self.forward_head(x, pre_logits=pre_logits)
        return x


def _create_vision_transformer(
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
    strict = True
    if "siglip" in variant and kwargs.get("global_pool", None) != "map":
        strict = False

    return build_model_with_cfg(
        ViT,
        variant,
        pretrained,
        pretrained_filter_fn=_filter_fn,
        pretrained_strict=strict,
        feature_cfg=dict(out_indices=out_indices, feature_cls="getter"),
        **kwargs,
    )


def vit_tiny_patch16_224(pretrained: bool = False, **kwargs) -> VisionTransformer:
    """ViT-Tiny (Vit-Ti/16)"""
    model_args = dict(patch_size=16, embed_dim=192, depth=12, num_heads=3)
    model = _create_vision_transformer(
        "vit_tiny_patch16_224", pretrained=pretrained, **dict(model_args, **kwargs)
    )
    return model, {**model_args, **kwargs}


def vit_small_patch16_224(pretrained: bool = False, **kwargs) -> VisionTransformer:
    """ViT-Small (ViT-S/16)"""
    model_args = dict(patch_size=16, embed_dim=384, depth=12, num_heads=6)
    model = _create_vision_transformer(
        "vit_small_patch16_224", pretrained=pretrained, **dict(model_args, **kwargs)
    )
    return model, {**model_args, **kwargs}


def vit_base_patch16_224(pretrained: bool = False, **kwargs) -> VisionTransformer:
    """ViT-Base (ViT-B/16) from original paper (https://arxiv.org/abs/2010.11929).
    ImageNet-1k weights fine-tuned from in21k @ 224x224, source https://github.com/google-research/vision_transformer.
    """
    model_args = dict(patch_size=16, embed_dim=768, depth=12, num_heads=12)
    model = _create_vision_transformer(
        "vit_base_patch16_224", pretrained=pretrained, **dict(model_args, **kwargs)
    )
    return model, {**model_args, **kwargs}


def vit_base_patch32_224(pretrained: bool = False, **kwargs) -> VisionTransformer:
    """ViT-Base (ViT-B/32) from original paper (https://arxiv.org/abs/2010.11929).
    ImageNet-1k weights fine-tuned from in21k, source https://github.com/google-research/vision_transformer.
    """
    model_args = dict(patch_size=32, embed_dim=768, depth=12, num_heads=12)
    model = _create_vision_transformer(
        "vit_base_patch32_224", pretrained=pretrained, **dict(model_args, **kwargs)
    )
    return model, {**model_args, **kwargs}


def vit_large_patch16_224(pretrained: bool = False, **kwargs) -> VisionTransformer:
    """ViT-Large model (ViT-L/16) from original paper (https://arxiv.org/abs/2010.11929).
    ImageNet-1k weights fine-tuned from in21k @ 224x224, source https://github.com/google-research/vision_transformer.
    """
    model_args = dict(patch_size=16, embed_dim=1024, depth=24, num_heads=16)
    model = _create_vision_transformer(
        "vit_large_patch16_224", pretrained=pretrained, **dict(model_args, **kwargs)
    )
    return model, {**model_args, **kwargs}
