from typing import Literal

from .deit import (
    deit_tiny_patch16_224,
    deit3_small_patch16_224,
    deit3_base_patch16_224,
    deit3_large_patch16_224,
)


def deit_factory(name: Literal["tiny", "small", "base", "large"], **kwargs):
    if name == "tiny":
        return deit_tiny_patch16_224(dynamic_img_size=True, **kwargs)
    elif name == "small":
        return deit3_small_patch16_224(dynamic_img_size=True, **kwargs)
    elif name == "base":
        return deit3_base_patch16_224(dynamic_img_size=True, **kwargs)
    elif name == "large":
        return deit3_large_patch16_224(dynamic_img_size=True, **kwargs)
