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
    resample_abs_pos_embed
)
from typing_extensions import Literal

from .fusion import AddEraseFusion, EraseFusion, FuseMlp, ResidualFusion
from .rw_heads import create_rw_head
from .memblocks import create_mem_blocks
from .embed import LatentEmbedding


def ttm_model_factory(model_cls):
    class ViTTM(model_cls):
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
            )
            norm_layer = get_norm_layer(norm_layer) or partial(nn.LayerNorm, eps=1e-6)
            act_layer = get_act_layer(act_layer) or nn.GELU

            self.img_size = img_size
            # TTM specific args
            self.memory_ps = memory_ps = kwargs.pop("memory_ps", patch_size)
            self.process_ps = process_ps = kwargs.pop("process_ps", None)
            self.process_embedder_type = kwargs.pop("process_embedder_type", "patch")
            self.process_tokens = kwargs.pop("process_tokens", None)

            latent_size_scale = kwargs.pop("latent_size_scale", 1)
            latent_size = embed_dim // latent_size_scale
            self.rw_head_type = kwargs.pop("rw_head_type", None)
            self.fusion_type = kwargs.pop("fusion_type", None)

            # dyna specific args
            reduced_dim = kwargs.pop("reduced_dim", 2)
            dyna_num_heads = kwargs.pop("dyna_num_heads", 16)
            dyna_concat = kwargs.pop("dyna_concat", True)

            # memory stream specific args
            self.memory_block = kwargs.pop("memory_blocks", "")
            self.memory_mlp_ratio = kwargs.pop("memory_mlp_ratio", 1 / 2)
            self.memory_encoder_depths = kwargs.pop("memory_encoder_depths", [3, 3])
            self.memory_encoder_downsamples = kwargs.pop(
                "memory_encoder_downsamples", [2, 1]
            )
            self.memory_decoder_depths = kwargs.pop("memory_decoder_depths", [3, 3])
            self.memory_decoder_upsamples = kwargs.pop(
                "memory_decoder_upsamples", [2, 1]
            )

            # Assertions to check stuff.
            assert process_ps is not None, "process_ps must be specified"
            assert self.rw_head_type in [
                "tl",
                "ca",
                "la",
                "lca",
                "lin",
                "dyna",
            ], "rw_head_type must be specified, select from 'tl', 'ca', 'la', 'lca', lin."
            assert self.fusion_type in [
                "residual",
                "erase",
                "add_erase",
            ], "fusion_type must be specified, select from 'residual', 'erase', or 'add_erase'."
            assert self.process_embedder_type in [
                "patch",
                "downsample",
                "latent",
            ], "process_embedder_type must be specified, select from 'patch', 'downsample', or 'latent'."

            # Create process embedder
            embed_args = {}
            if dynamic_img_size:
                # flatten deferred until after pos embed
                embed_args.update(dict(strict_img_size=False, output_fmt="NHWC"))
            # Redefine memory_embedder and process_embedder
            if self.memory_ps == patch_size:
                print("Memory and Process Patch Size are the same, using same embedder.")
                self.memory_embedder = self.patch_embed
            else:
                print("Memory and Process Patch Size are different, using different embedders.")
                self.memory_embedder = embed_layer(
                    img_size=img_size,
                    patch_size=memory_ps,
                    in_chans=in_chans,
                    embed_dim=embed_dim,
                    bias=not pre_norm,  # disable bias if pre-norm is used (e.g. CLIP)
                    dynamic_img_pad=dynamic_img_pad,
                    **embed_args,
                )

            if self.process_embedder_type == "patch":
                print("Using Patch Embedder for Process.")
                self.process_embedder = embed_layer(
                    img_size=img_size,
                    patch_size=process_ps,
                    in_chans=in_chans,
                    embed_dim=embed_dim,
                    bias=not pre_norm,  # disable bias if pre-norm is used (e.g. CLIP)
                    dynamic_img_pad=dynamic_img_pad,
                    **embed_args,
                )
                process_size = self.process_size = self.process_embedder.num_patches
            elif self.process_embedder_type == "downsample":
                print("Using Downsample Embedder for Process.")
                self.process_embedder = nn.AdaptiveAvgPool2d(self.process_tokens)
                process_size = self.process_size = self.process_tokens
            elif self.process_embedder_type == "latent":
                print("Using Latent Embedder for Process.")
                self.process_embedder = LatentEmbedding(self.process_tokens)
                process_size = self.process_size = self.process_tokens

            memory_size = self.memory_size = self.memory_embedder.num_patches

            process_embed_len = process_size
            memory_embed_len = memory_size
            self.process_pos_embed = nn.Parameter(
                torch.randn(1, process_embed_len, embed_dim) * 0.02
            )
            self.memory_pos_embed = nn.Parameter(
                torch.randn(1, memory_embed_len, embed_dim) * 0.02
            )

            self.memory_h = self.memory_w = self.img_size[0] // memory_ps
            self.process_h = self.process_w = self.img_size[0] // process_ps

            self.read_norm = nn.ModuleList(
                [norm_layer(embed_dim) for _ in range(depth)]
            )
            self.write_norm = nn.ModuleList(
                [norm_layer(embed_dim) for _ in range(depth)]
            )

            self.process_norm = norm_layer(
                embed_dim
            )  # this is the final norm layer for process

            self.process_head = nn.Linear(embed_dim, num_classes)
            self.fc_norm_process = norm_layer(embed_dim)

            if self.memory_block == "":
                self.mem_blocks = create_mem_blocks("", depth=depth)
            elif self.memory_block == "mlp":
                self.mem_blocks = create_mem_blocks(
                    "mlp",
                    embed_dim=embed_dim,
                    memory_mlp_ratio=self.memory_mlp_ratio,
                    proj_drop_rate=proj_drop_rate,
                    depth=depth,
                )
            elif self.memory_block == "conv":
                self.mem_encoder, self.mem_decoder = create_mem_blocks(
                    "conv",
                    embed_dim=embed_dim,
                    memory_encoder_depths=self.memory_encoder_depths,
                    memory_encoder_downsamples=self.memory_encoder_downsamples,
                    memory_decoder_depths=self.memory_decoder_depths,
                    depth=depth,
                )

            if self.rw_head_type == "tl":
                # Create write head (process -> memory)
                self.write_head = create_rw_head(
                    rw_head_type="tl",
                    embed_dim=embed_dim,
                    out_features=memory_size,
                    bottleneck_size=memory_size,
                    act_layer=act_layer,
                    drop=drop_rate,
                    depth=depth,
                )

                # Create read head (memory -> process)
                self.read_head = create_rw_head(
                    rw_head_type="tl",
                    embed_dim=embed_dim,
                    out_features=process_size,
                    bottleneck_size=process_size,
                    act_layer=act_layer,
                    drop=drop_rate,
                    depth=depth,
                )
            elif self.rw_head_type == "ca":
                # Create write head (process -> memory)
                self.write_head = create_rw_head(
                    rw_head_type="ca",
                    embed_dim=embed_dim,
                    num_heads=num_heads,
                    depth=depth,
                )
                # Create read head (memory -> process)
                self.read_head = create_rw_head(
                    rw_head_type="ca",
                    embed_dim=embed_dim,
                    num_heads=num_heads,
                    depth=depth,
                )
            elif self.rw_head_type == "la":
                # Create write head (process -> memory)
                self.write_head = create_rw_head(
                    rw_head_type="la",
                    embed_dim=embed_dim,
                    latent_size=latent_size,
                    depth=depth,
                )
                # Create read head (memory -> process)
                self.read_head = create_rw_head(
                    rw_head_type="la",
                    embed_dim=embed_dim,
                    latent_size=latent_size,
                    depth=depth,
                )
            elif self.rw_head_type == "lca":
                # Create write head (process -> memory)
                self.write_head = create_rw_head(
                    rw_head_type="lca",
                    embed_dim=embed_dim,
                    latent_size=latent_size,
                    num_heads=num_heads,
                    depth=depth,
                )
                # Create read head (memory -> process)
                self.read_head = create_rw_head(
                    rw_head_type="lca",
                    embed_dim=embed_dim,
                    latent_size=latent_size,
                    num_heads=num_heads,
                    depth=depth,
                )
            elif self.rw_head_type == "lin":
                # Create write head (process -> memory)
                self.write_head = create_rw_head(
                    rw_head_type="lin",
                    embed_dim=embed_dim,
                    latent_size=latent_size,
                    num_heads=num_heads,
                    depth=depth,
                )

                # Create read head (memory -> process)
                self.read_head = create_rw_head(
                    rw_head_type="lin",
                    embed_dim=embed_dim,
                    latent_size=latent_size,
                    num_heads=num_heads,
                    depth=depth,
                )
            elif self.rw_head_type == "dyna":
                self.write_head = create_rw_head(
                    rw_head_type="dyna",
                    embed_dim=embed_dim,
                    input_features=process_size,
                    out_features=memory_size,
                    reduced_dim=reduced_dim,
                    num_heads=dyna_num_heads,
                    concat=dyna_concat,
                    depth=depth,
                )
                self.read_head = create_rw_head(
                    rw_head_type="dyna",
                    embed_dim=embed_dim,
                    input_features=process_size,
                    out_features=memory_size,
                    reduced_dim=reduced_dim,
                    num_heads=dyna_num_heads,
                    concat=dyna_concat,
                    depth=depth,
                )
            if self.fusion_type == "residual":
                self.read_fusion = ResidualFusion()
                self.write_fusion = ResidualFusion()
            elif self.fusion_type == "erase":
                self.read_fusion = EraseFusion()
                self.write_fusion = EraseFusion()
            elif self.fusion_type == "add_erase":
                self.read_fusion = AddEraseFusion(
                    embed_dim=embed_dim,
                    target_size=process_size,
                )
                self.write_fusion = AddEraseFusion(
                    embed_dim=embed_dim,
                    target_size=memory_size,
                )

            # initialize weights
            if weight_init != "skip":
                self.init_weights(weight_init)
            if fix_init:
                self.fix_init_weight()

            trunc_normal_(self.process_pos_embed, std=0.02)  # process pos embed
            trunc_normal_(self.memory_pos_embed, std=0.02)  # memory pos embed

            if self.process_embedder_type == "patch":
                nn.init.xavier_uniform_(self.process_embedder.proj.weight)
                nn.init.zeros_(self.process_embedder.proj.bias)

            nn.init.zeros_(self.process_head.bias)
            nn.init.zeros_(self.process_head.weight)
            nn.init.zeros_(self.head.weight)
            nn.init.zeros_(self.head.bias)

            # Print Model Configuration
            print("Model Configuration:")
            print(f"  model: {self.__class__.__name__}")
            print(f"  class token: {class_token}")
            print(f"  reg tokens: {reg_tokens}")
            print(f"  global pool: {global_pool}")
            print(f"  fc norm: {self.fc_norm}")
            print(f"  memory size: {memory_size}")
            print(f"  process size: {process_size}")
            print(f"  memory_ps: {memory_ps}")
            print(f"  process_ps: {process_ps}")
            print(f"  rw_head_type: {self.rw_head_type}")
            print(f"  fusion_type: {self.fusion_type}")

            print(f"  latent_size: {latent_size}")
            print(f"  reduced_dim: {reduced_dim}")
            print(f"  dyna_num_heads: {dyna_num_heads}")
            print(f"  dyna_concat: {dyna_concat}")

            print(f"  memory_block: {self.memory_block}")
            print(f"  memory_mlp_ratio: {self.memory_mlp_ratio}")
            print(f"  memory_encoder_depths: {self.memory_encoder_depths}")
            print(f"  memory_encoder_downsamples: {self.memory_encoder_downsamples}")
            print(f"  memory_decoder_depths: {self.memory_decoder_depths}")

        def _pos_embed(self, x: torch.Tensor, pos_embed: torch.Tensor) -> torch.Tensor:
            if self.dynamic_img_size:
                B, H, W, C = x.shape
                pos_embed = resample_abs_pos_embed(
                    pos_embed,
                    (H, W),
                    num_prefix_tokens=0
                )
                x = x.view(B, -1, C)
            x = x + pos_embed
            return self.pos_drop(x)

        def forward_features(
            self,
            x: torch.Tensor,
            keep_mask_memory: torch.Tensor | None = None,
            keep_mask_process: torch.Tensor | None = None,
        ) -> torch.Tensor:
            # Memory Embedding
            memory = self.memory_embedder(x)

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

            # Add Positional Embeddings
            memory = self._pos_embed(memory, self.memory_pos_embed)
            process = self._pos_embed(process, self.process_pos_embed)

            if isinstance(keep_mask_memory, torch.Tensor):
                _, _, D = memory.shape
                memory = torch.gather(
                    memory,
                    dim=1,
                    index=keep_mask_memory.unsqueeze(-1).expand(-1, -1, D),
                )
            if isinstance(keep_mask_process, torch.Tensor):
                _, _, D = process.shape
                process = torch.gather(
                    process,
                    dim=1,
                    index=keep_mask_process.unsqueeze(-1).expand(-1, -1, D),
                )

            # Drop Out, Normalization
            process = self.patch_drop(process)
            process = self.norm_pre(process)

            # Iterate over blocks
            for i, block in enumerate(self.blocks):
                # Normalize process and memory
                memory, process = self.read_norm[i](memory), self.read_norm[i](process)
                # Read: Memory -> Process
                rprocess = self.read_head[i](memory, process)
                process = self.read_fusion(rw=rprocess, target=process)

                # Memory Block
                if self.memory_block == "conv":
                    memory_ = memory
                    memory, up_sizes = self.mem_encoder[i](memory)

                # Block
                process = block(process)

                # Normalize process and memory
                memory, process = self.write_norm[i](memory), self.write_norm[i](
                    process
                )
                # Write: Process -> Memory
                wmemory = self.write_head[i](process, memory)
                memory = self.write_fusion(rw=wmemory, target=memory)

                # Memory Block Decoder (if conv)
                if self.memory_block == "conv":
                    memory = memory_ + self.mem_decoder[i](memory, up_sizes)
                else:
                    memory = self.mem_blocks[i](memory)

            # Final Layer Normalization
            memory, process = self.norm(memory), self.process_norm(process)

            # Return memory and process
            return memory, process

        def forward(
            self, x: torch.Tensor, pre_logits: bool = False, **kwargs
        ) -> torch.Tensor:
            memory, process = self.forward_features(x)
            memory_pred, process_pred = self.forward_head(
                memory, process, pre_logits=pre_logits
            )
            return memory_pred, process_pred

        def forward_head(self, memory, process, pre_logits=False):
            memory = memory.mean(dim=1)
            process = process.mean(dim=1)

            memory = self.fc_norm(memory)
            process = self.fc_norm_process(process)

            process = self.head_drop(process)
            memory = self.head_drop(memory)
            if pre_logits:
                return memory, process
            return self.head(memory), self.process_head(process)

    return ViTTM


# ViTTTM models
def _create_vittm(
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
        ttm_model_factory(VisionTransformer),
        variant,
        pretrained,
        pretrained_filter_fn=_filter_fn,
        pretrained_strict=strict,
        feature_cfg=dict(out_indices=out_indices, feature_cls="getter"),
        **kwargs,
    )


def vittm_tiny_patch16_224(pretrained: bool = False, **kwargs) -> VisionTransformer:
    """ViT-Tiny (Vit-Ti/16)"""

    model_args = dict(patch_size=16, embed_dim=192, depth=12, num_heads=3)
    model = _create_vittm(
        "vit_tiny_patch16_224", pretrained=pretrained, **dict(model_args, **kwargs)
    )
    return model


def vittm_small_patch16_224(pretrained: bool = False, **kwargs) -> VisionTransformer:
    """ViT-Small (ViT-S/16)"""
    model_args = dict(patch_size=16, embed_dim=384, depth=12, num_heads=6)
    model = _create_vittm(
        "vit_small_patch16_224", pretrained=pretrained, **dict(model_args, **kwargs)
    )
    return model


def vittm_base_patch16_224(pretrained: bool = False, **kwargs) -> VisionTransformer:
    """ViT-Base (ViT-B/16) from original paper (https://arxiv.org/abs/2010.11929).
    ImageNet-1k weights fine-tuned from in21k @ 224x224, source https://github.com/google-research/vision_transformer.
    """
    model_args = dict(patch_size=16, embed_dim=768, depth=12, num_heads=12)
    model = _create_vittm(
        "vit_base_patch16_224", pretrained=pretrained, **dict(model_args, **kwargs)
    )
    return model


def vittm_large_patch16_224(pretrained: bool = False, **kwargs) -> VisionTransformer:
    """ViT-Large model (ViT-L/16) from original paper (https://arxiv.org/abs/2010.11929).
    ImageNet-1k weights fine-tuned from in21k @ 224x224, source https://github.com/google-research/vision_transformer.
    """
    model_args = dict(patch_size=16, embed_dim=1024, depth=24, num_heads=16)
    model = _create_vittm(
        "vit_large_patch16_224", pretrained=pretrained, **dict(model_args, **kwargs)
    )
    return model


# DeiTTM models
def _create_deittm(variant, pretrained=False, distilled=False, **kwargs):
    out_indices = kwargs.pop("out_indices", 3)
    model_cls = (
        ttm_model_factory(VisionTransformerDistilled)
        if distilled
        else ttm_model_factory(VisionTransformer)
    )
    pretrained_strict = False
    model = build_model_with_cfg(
        model_cls,
        variant,
        pretrained,
        pretrained_filter_fn=partial(checkpoint_filter_fn, adapt_layer_scale=True),
        feature_cfg=dict(out_indices=out_indices, feature_cls="getter"),
        pretrained_strict=pretrained_strict,
        **kwargs,
    )
    return model


def deittm_tiny_patch16_224(pretrained=False, **kwargs) -> VisionTransformer:
    """DeiT-tiny model @ 224x224 from paper (https://arxiv.org/abs/2012.12877).
    ImageNet-1k weights from https://github.com/facebookresearch/deit.
    """
    model_args = dict(
        patch_size=16, embed_dim=192, depth=12, num_heads=3, no_embed_class=True
    )
    model = _create_deittm(
        "deit_tiny_patch16_224", pretrained=pretrained, **dict(model_args, **kwargs)
    )
    return model


def deit3tm_small_patch16_224(pretrained=False, **kwargs) -> VisionTransformer:
    """DeiT-3 small model @ 224x224 from paper (https://arxiv.org/abs/2204.07118).
    ImageNet-1k weights from https://github.com/facebookresearch/deit.
    """
    model_args = dict(
        patch_size=16,
        embed_dim=384,
        depth=12,
        num_heads=6,
        init_values=1e-6,
        no_embed_class=True,
    )
    model = _create_deittm(
        "deit3_small_patch16_224", pretrained=pretrained, **dict(model_args, **kwargs)
    )
    return model


def deit3tm_base_patch16_224(pretrained=False, **kwargs) -> VisionTransformer:
    """DeiT-3 base model @ 224x224 from paper (https://arxiv.org/abs/2204.07118).
    ImageNet-1k weights from https://github.com/facebookresearch/deit.
    """
    model_args = dict(
        patch_size=16,
        embed_dim=768,
        depth=12,
        num_heads=12,
        init_values=1e-6,
        no_embed_class=True,
    )
    model = _create_deittm(
        "deit3_base_patch16_224", pretrained=pretrained, **dict(model_args, **kwargs)
    )
    return model


def deit3tm_large_patch16_224(pretrained=False, **kwargs) -> VisionTransformer:
    """DeiT-3 large model @ 224x224 from paper (https://arxiv.org/abs/2204.07118).
    ImageNet-1k weights from https://github.com/facebookresearch/deit.
    """
    model_args = dict(
        patch_size=16,
        embed_dim=1024,
        depth=24,
        num_heads=16,
        init_values=1e-6,
        no_embed_class=True,
    )
    model = _create_deittm(
        "deit3_large_patch16_224", pretrained=pretrained, **dict(model_args, **kwargs)
    )
    return model


if __name__ == "__main__":
    model = vittm_tiny_patch16_224(
        process_ps=32, dynamic_img_size=True, pretrained=True
    )
    # model.eval()
    # print(model)
    # model(torch.randn(1, 3, 224, 224))

    # print(timm.list_models("*vittm*"))
