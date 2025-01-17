from typing import Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F

from .patch_embed import pi_resize_patch_embed


def apply_flexi_patch_embed(
    model: nn.Module,
    process_embedding_name: str,
    patch_embedding_name: str,
    new_patch_size: Tuple[int, int],
    interpolation: str = "bicubic",
    antialias: bool = True,
):
    state_dict = model.state_dict()

    state_dict[process_embedding_name] = pi_resize_patch_embed(
        state_dict[patch_embedding_name],
        new_patch_size,
        interpolation=interpolation,
        antialias=antialias,
    )

    model.load_state_dict(state_dict)
    return model