from typing import Literal

from .vit import (
    vit_base_patch16_224,
    vit_large_patch16_224,
    vit_small_patch16_224,
    vit_tiny_patch16_224,
    vit_base_patch32_224
)


def vit_factory(name: Literal["tiny", "small", "base", "base32", "large"], **kwargs):
    if name == "tiny":
        return vit_tiny_patch16_224(dynamic_img_size=True, **kwargs)
    elif name == "small":
        return vit_small_patch16_224(dynamic_img_size=True, **kwargs)
    elif name == "base":
        return vit_base_patch16_224(dynamic_img_size=True, **kwargs)
    elif name == "base32":
        return vit_base_patch32_224(dynamic_img_size=True, **kwargs)
    elif name == "large":
        return vit_large_patch16_224(dynamic_img_size=True, **kwargs)
