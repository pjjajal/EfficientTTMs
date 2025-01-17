import math
from functools import partial
from typing import Callable, Literal, Optional, Tuple, Type, Union
from typing_extensions import Literal
from timm.layers import (
    Mlp,
    PatchEmbed,
    get_act_layer,
    get_norm_layer,
    trunc_normal_,
    DropPath,
)
from timm.layers.typing import LayerType

import torch
import torch.nn as nn
import torch.nn.functional as F
from mmseg.registry import MODELS
from timm.models.vision_transformer import (
    Block,
    VisionTransformer,
    checkpoint_filter_fn,
    named_apply,
    get_init_weights_vit,
)
from timm.layers import resample_abs_pos_embed

from models.vittm.vittm import ttm_model_factory
from models.vittm.fusion import AddEraseFusion, EraseFusion, FuseMlp, ResidualFusion
from models.vittm.rw_heads import create_rw_head
from models.vittm.memblocks import create_mem_blocks
from models.vittm.embed import LatentEmbedding


ViTTMBaseClass = ttm_model_factory(VisionTransformer)


# TODO update to add pos_embedding for process and memory
@MODELS.register_module()
class ViTTM(ViTTMBaseClass):
    def __init__(
        self,
        img_size: Union[int, Tuple[int, int]] = 224,
        patch_size: Union[int, Tuple[int, int]] = 16,
        in_chans: int = 3,
        num_classes: int = 1000,
        global_pool: Literal["", "avg", "avgmax", "max", "token", "map"] = "token",
        embed_dim: int = 768,
        depth: int = 12,
        num_heads: int = 12,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
        qk_norm: bool = False,
        init_values: Optional[float] = None,
        class_token: bool = True,
        pos_embed: str = "learn",
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
            pos_embed,
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
            **kwargs,
        )

        self.init_cfg = kwargs.get("init_cfg", None)
        self.training = kwargs.get("training", True)

        self.uper = kwargs.get("uper", False)
        self.out_indices = kwargs.pop("out_indices", [3, 5, 7, 11])
        self.drop_path = (
            DropPath(drop_path_rate) if drop_path_rate > 0.0 else nn.Identity()
        )
        if self.memory_ps == 16 and self.uper:
            self.fpn1 = nn.Sequential(
                nn.ConvTranspose2d(embed_dim, embed_dim, kernel_size=2, stride=2),
                nn.SyncBatchNorm(embed_dim),
                nn.GELU(),
                nn.ConvTranspose2d(embed_dim, embed_dim, kernel_size=2, stride=2),
            )

            self.fpn2 = nn.Sequential(
                nn.ConvTranspose2d(embed_dim, embed_dim, kernel_size=2, stride=2),
            )

            self.fpn3 = nn.Identity()

            self.fpn4 = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward_features(self, x: torch.Tensor) -> Tuple[torch.Tensor]:
        B, _, _, _ = x.shape
        # Memory Embedding
        memory = self.memory_embedder(x)
        Hm, Wm = memory.shape[1], memory.shape[2]

        # Process Embedding
        if self.process_embedder_type == "patch":
            process = self.process_embedder(x)
        elif self.process_embedder_type == "downsample":
            B, L, C = memory.shape
            H = W = int(L**0.5)
            memory = memory.view(B, H, W, C).permute(0, 3, 1, 2)
            process = self.process_embedder(memory)
            process = process.flatten(2).transpose(1, 2)
        elif self.process_embedder_type == "latent":
            process = self.process_embedder(x.shape[0])

        Hp, Wp = process.shape[1], process.shape[2]

        # Add Positional Embeddings
        memory = self._pos_embed(memory, self.memory_pos_embed)
        process = self._pos_embed(process, self.process_pos_embed)

        # Drop Out, Normalization
        process = self.patch_drop(process)
        process = self.norm_pre(process)

        # Iterate over blocks
        features = []
        for i, block in enumerate(self.blocks):
            # Normalize process and memory
            memory, process = self.read_norm[i](memory), self.read_norm[i](process)
            # Read: Memory -> Process
            rprocess = self.drop_path(self.read_head[i](memory, process))
            process = self.read_fusion(rw=rprocess, target=process)

            # Block
            process = block(process)

            # Normalize process and memory
            memory, process = self.write_norm[i](memory), self.write_norm[i](process)
            # Write: Process -> Memory
            wmemory = self.drop_path(self.write_head[i](process, memory))
            memory = self.write_fusion(rw=wmemory, target=memory)

            # Memory Block Decoder (if conv)
            memory = self.mem_blocks[i](memory)

            if i in self.out_indices:
                xp = process.permute(0, 2, 1).reshape(B, -1, Hp, Wp).contiguous()
                xm = memory.permute(0, 2, 1).reshape(B, -1, Hm, Wm).contiguous()

                if Hp == Hm and Wp == Wm:
                    features.append(torch.cat([xp, xm], dim=1))
                else:
                    features.append(xp)

        if self.uper:
            ops = [self.fpn1, self.fpn2, self.fpn3, self.fpn4]
            for i in range(len(features)):
                features[i] = ops[i](features[i])

        # Return memory and process
        return features

    def init_weights(self, mode: str = "") -> None:
        assert mode in ("jax", "jax_nlhb", "moco", "")
        head_bias = -math.log(self.num_classes) if "nlhb" in mode else 0.0
        trunc_normal_(self.pos_embed, std=0.02)
        if self.cls_token is not None:
            nn.init.normal_(self.cls_token, std=1e-6)
        named_apply(get_init_weights_vit(mode, head_bias), self)
        if hasattr(self, "init_cfg"):
            print("Initializing from config")
            self.init_from_cfg(self.init_cfg)

    def init_from_cfg(self, cfg: dict) -> None:
        if cfg.get("checkpoint", False):
            print("Loading from checkpoint")
            state_dict = torch.load(cfg["checkpoint"])
            print(state_dict.keys())
            pos_embed = state_dict["pos_embed"]
            process_pos_embed = state_dict["process_pos_embed"]
            memory_pos_embed = state_dict["memory_pos_embed"]
            state_dict["pos_embed"] = resample_abs_pos_embed(
                pos_embed, new_size=self.patch_embed.grid_size, num_prefix_tokens=1
            )
            state_dict["process_pos_embed"] = resample_abs_pos_embed(
                process_pos_embed,
                new_size=self.process_embedder.grid_size,
                num_prefix_tokens=0,
            )
            state_dict["memory_pos_embed"] = resample_abs_pos_embed(
                memory_pos_embed,
                new_size=self.memory_embedder.grid_size,
                num_prefix_tokens=0,
            )
            self.load_state_dict(state_dict, strict=True)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor]:
        features = self.forward_features(x)
        if self.training:
            features[-1] = features[-1] + 0 * sum(
                [param.sum() for param in self.parameters()]
            )
        return features
