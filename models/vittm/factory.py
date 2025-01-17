from typing import Literal

from .vittm import (
    vittm_base_patch16_224,
    vittm_large_patch16_224,
    vittm_small_patch16_224,
    vittm_tiny_patch16_224,
    deittm_tiny_patch16_224,
    deit3tm_small_patch16_224,
    deit3tm_base_patch16_224,
    deit3tm_large_patch16_224,
)


def vittm_factory(name: Literal["tiny", "small", "base", "large"], **kwargs):
    if name == "tiny":
        return vittm_tiny_patch16_224(global_pool="avg", dynamic_img_size=True, **kwargs)
    elif name == "small":
        return vittm_small_patch16_224(global_pool="avg", dynamic_img_size=True, **kwargs)
    elif name == "base":
        return vittm_base_patch16_224(global_pool="avg", dynamic_img_size=True, **kwargs)
    elif name == "large":
        return vittm_large_patch16_224(global_pool="avg", dynamic_img_size=True, **kwargs)


def deittm_factory(name: Literal["tiny", "small", "base", "large"], **kwargs):
    if name == "tiny":
        return deittm_tiny_patch16_224(global_pool="avg", dynamic_img_size=True, **kwargs)
    elif name == "small":
        return deit3tm_small_patch16_224(global_pool="avg", dynamic_img_size=True, **kwargs)
    elif name == "base":
        return deit3tm_base_patch16_224(global_pool="avg", dynamic_img_size=True, **kwargs)
    elif name == "large":
        return deit3tm_large_patch16_224(global_pool="avg", dynamic_img_size=True, **kwargs)


if __name__ == "__main__":
    import torch

    model = deittm_factory(
        "tiny",
        pretrained=False,
        memory_ps=16,
        process_ps=32,
        rw_head_type="ca",
        fusion_type="residual",
    )
    print(model)
    memory, process = model.forward_features(torch.randn(1, 3, 224, 224))
    print(memory.shape, process.shape)
    # print(new == old)
