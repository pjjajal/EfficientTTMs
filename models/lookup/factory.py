from typing import Literal

from .lookup import lookupvit_base_patch16_224


def lookupvit_factory(
    name: Literal[
        "lookupvit_3x3", "lookupvit_5x5", "lookupvit_7x7", "lookupvit_10x10"
    ],
    **kwargs
):
    if name == "lookupvit_3x3":
        return lookupvit_base_patch16_224(
            global_pool="avg", dynamic_img_size=True, compressed_tokens=3, **kwargs
        )
    elif name == "lookupvit_5x5":
        return lookupvit_base_patch16_224(
            global_pool="avg", dynamic_img_size=True, compressed_tokens=5, **kwargs
        )
    elif name == "lookupvit_7x7":
        return lookupvit_base_patch16_224(
            global_pool="avg", dynamic_img_size=True, compressed_tokens=7, **kwargs
        )
    elif name == "lookupvit_10x10":
        return lookupvit_base_patch16_224(
            global_pool="avg", dynamic_img_size=True, compressed_tokens=10, **kwargs
        )

if __name__ == "__main__":
    model = lookupvit_factory("lookup_vit_3x3")

    import torch

    x, y = model(torch.randn(1, 3, 224, 224))
    print(x.shape, y.shape)