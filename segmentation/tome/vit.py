import math
from functools import partial
from typing import Callable, Literal, Optional, Tuple, Type, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from mmseg.registry import MODELS
from mmseg.models.backbones import TIMMBackbone
from models.tome import apply_patch

@MODELS.register_module()
class ViTBackbone(TIMMBackbone):
    def __init__(
        self,
        model_name,
        features_only=False,
        pretrained=True,
        checkpoint_path="",
        in_channels=3,
        init_cfg=None,
        training=True,
        **kwargs
    ):
        self.training = training
        super().__init__(
            model_name,
            features_only,
            pretrained,
            checkpoint_path,
            in_channels,
            init_cfg,
            **kwargs
        )

    def forward_features(self, x):
        x = self.timm_model.patch_embed(x)
        B, H, W, C = x.shape
        x = self.timm_model._pos_embed(x)
        x = self.timm_model.patch_drop(x)
        x = self.timm_model.norm_pre(x)
        x = self.timm_model.blocks(x)
        x = self.timm_model.norm(x)

        x = x[:, 1:]
        x = x.permute(0, 2, 1).reshape(B, -1, H, W)
        return [x]

    def forward(self, x):
        features = self.forward_features(x)
        if self.training:
            features[-1] = features[-1] + 0 * sum([param.sum() for param in self.parameters()])
        return features


@MODELS.register_module()
class TomeBackbone(ViTBackbone):
    def __init__(
        self,
        model_name,
        features_only=False,
        pretrained=True,
        checkpoint_path="",
        in_channels=3,
        init_cfg=None,
        token_merging: bool = False,
        r: int = 1,
        **kwargs
    ):
        super().__init__(
            model_name,
            features_only,
            pretrained,
            checkpoint_path,
            in_channels,
            init_cfg,
            **kwargs
        )
        if token_merging:
            apply_patch(self.timm_model)
            self.timm_model.r = r
