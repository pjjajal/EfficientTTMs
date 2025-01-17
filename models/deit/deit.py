from functools import partial
from typing import Callable, Literal, Optional, Tuple, Type, Union

import torch.nn as nn
from timm.layers import Mlp, PatchEmbed, get_act_layer, get_norm_layer
from timm.layers.typing import LayerType
from timm.models._builder import build_model_with_cfg
from timm.models.vision_transformer import (
    Block,
    VisionTransformer,
    checkpoint_filter_fn,
)
from timm.models.deit import VisionTransformerDistilled
import torch
from torch.nn.modules import Module
from typing_extensions import Literal


# Make the forward similar to DINO
def deit_wrapped_class(model_cls):
    class DeiT(model_cls):
        def forward(self, x: torch.Tensor, pre_logits=False) -> torch.Tensor:
            x = self.forward_features(x)
            x = self.forward_head(x, pre_logits=pre_logits)
            return x

    return DeiT


def _create_deit(variant, pretrained=False, distilled=False, **kwargs):
    out_indices = kwargs.pop("out_indices", 3)
    model_cls = (
        deit_wrapped_class(VisionTransformerDistilled)
        if distilled
        else deit_wrapped_class(VisionTransformer)
    )
    model = build_model_with_cfg(
        model_cls,
        variant,
        pretrained,
        pretrained_filter_fn=partial(checkpoint_filter_fn, adapt_layer_scale=True),
        feature_cfg=dict(out_indices=out_indices, feature_cls="getter"),
        **kwargs,
    )
    return model


def deit_tiny_patch16_224(pretrained=False, **kwargs) -> VisionTransformer:
    """DeiT-tiny model @ 224x224 from paper (https://arxiv.org/abs/2012.12877).
    ImageNet-1k weights from https://github.com/facebookresearch/deit.
    """
    model_args = dict(patch_size=16, embed_dim=192, depth=12, num_heads=3)
    model = _create_deit(
        "deit_tiny_patch16_224", pretrained=pretrained, **dict(model_args, **kwargs)
    )
    return model


def deit3_small_patch16_224(pretrained=False, **kwargs) -> VisionTransformer:
    """DeiT-3 small model @ 224x224 from paper (https://arxiv.org/abs/2204.07118).
    ImageNet-1k weights from https://github.com/facebookresearch/deit.
    """
    model_args = dict(
        patch_size=16,
        embed_dim=384,
        depth=12,
        num_heads=6,
        no_embed_class=True,
        init_values=1e-6,
    )
    model = _create_deit(
        "deit3_small_patch16_224", pretrained=pretrained, **dict(model_args, **kwargs)
    )
    return model


def deit3_base_patch16_224(pretrained=False, **kwargs) -> VisionTransformer:
    """DeiT-3 base model @ 224x224 from paper (https://arxiv.org/abs/2204.07118).
    ImageNet-1k weights from https://github.com/facebookresearch/deit.
    """
    model_args = dict(
        patch_size=16,
        embed_dim=768,
        depth=12,
        num_heads=12,
        no_embed_class=True,
        init_values=1e-6,
    )
    model = _create_deit(
        "deit3_base_patch16_224", pretrained=pretrained, **dict(model_args, **kwargs)
    )
    return model


def deit3_large_patch16_224(pretrained=False, **kwargs) -> VisionTransformer:
    """DeiT-3 large model @ 224x224 from paper (https://arxiv.org/abs/2204.07118).
    ImageNet-1k weights from https://github.com/facebookresearch/deit.
    """
    model_args = dict(
        patch_size=16,
        embed_dim=1024,
        depth=24,
        num_heads=16,
        no_embed_class=True,
        init_values=1e-6,
    )
    model = _create_deit(
        "deit3_large_patch16_224", pretrained=pretrained, **dict(model_args, **kwargs)
    )
    return model


if __name__ == "__main__":
    # model = deit_tiny_patch16_224(pretrained=False)
    # print(model(torch.randn(1, 3, 224, 224)).cuda().shape)
    model, _ = deit_tiny_patch16_224(pretrained=False)
    print(model(torch.randn(1, 3, 224, 224)).shape)
