import torch
import torch.nn as nn
from mmseg.models.backbones.vit import VisionTransformer
from mmseg.registry import MODELS
from mmseg.models.utils import PatchEmbed
from models.vittm.rw_heads import (
    TokenLearner,
    CrossAttention,
    LatentAttention,
    LatentCrossAttention,
)

#TODO update to add pos_embedding for process and memory
@MODELS.register_module()
class ViTTM(VisionTransformer):
    def __init__(
        self,
        img_size=224,
        patch_size=16,
        patch_pad="corner",
        in_channels=3,
        embed_dims=768,
        num_layers=12,
        num_heads=12,
        mlp_ratio=4,
        out_origin=False,
        out_indices=-1,
        qkv_bias=True,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        drop_path_rate=0.0,
        with_cls_token=True,
        output_cls_token=False,
        norm_cfg=dict(type="LN"),
        act_cfg=dict(type="GELU"),
        patch_norm=False,
        patch_bias=False,
        pre_norm=False,
        final_norm=False,
        interpolate_mode="bicubic",
        num_fcs=2,
        norm_eval=False,
        with_cp=False,
        frozen_exclude=["all"],
        pretrained=None,
        init_cfg=None,
        **kwargs,
    ):
        super().__init__(
            img_size,
            patch_size,
            patch_pad,
            in_channels,
            embed_dims,
            num_layers,
            num_heads,
            mlp_ratio,
            out_origin,
            out_indices,
            qkv_bias,
            drop_rate,
            attn_drop_rate,
            drop_path_rate,
            with_cls_token,
            output_cls_token,
            norm_cfg,
            act_cfg,
            patch_norm,
            patch_bias,
            pre_norm,
            final_norm,
            interpolate_mode,
            num_fcs,
            norm_eval,
            with_cp,
            frozen_exclude,
            pretrained,
            init_cfg,
        )
        act_layer = nn.GELU

        # TTM specific args
        process_ps = kwargs.pop("process_ps", None)
        latent_size = kwargs.pop("latent_size", 64)
        self.rw_head_type = kwargs.pop("rw_head_type", None)

        # Assertions to check stuff.
        assert process_ps is not None, "process_ps must be specified"
        assert self.rw_head_type in [
            "tl",
            "ca",
            "la",
            "lca",
        ], "rw_head_type must be specified, select from 'tl', 'ca', 'la', or 'lca'."

        self.process_embedder = PatchEmbed(
            in_channels=in_channels,
            embed_dims=embed_dims,
            conv_type="Conv2d",
            kernel_size=process_ps,
            stride=process_ps,
            padding=patch_pad,
            bias=patch_bias,
            norm_cfg=norm_cfg if patch_norm else None,
            init_cfg=None,
        )

        num_patches = int((img_size[0] // patch_size) * (img_size[1] // patch_size))
        num_process_patches = int(
            (img_size[0] // process_ps) * (img_size[1] // process_ps)
        )

        process_size = num_patches + int(with_cls_token)
        memory_size = num_process_patches + int(with_cls_token)

        if self.rw_head_type == "tl":
            # Create write head (process -> memory)
            self.write_head = TokenLearner(
                embed_dim=embed_dims,
                out_features=memory_size,
                bottleneck_size=memory_size,
                act_layer=act_layer,
                drop=drop_rate,
            )

            # Create read head (memory -> process)
            self.read_head = TokenLearner(
                embed_dim=embed_dims,
                out_features=process_size,
                bottleneck_size=process_size,
                act_layer=act_layer,
                drop=drop_rate,
            )
        elif self.rw_head_type == "ca":
            # Create write head (process -> memory)
            self.write_head = CrossAttention(
                embed_dim=embed_dims,
                num_heads=num_heads,
            )

            # Create read head (memory -> process)
            self.read_head = CrossAttention(
                embed_dim=embed_dims,
                num_heads=num_heads,
            )
        elif self.rw_head_type == "la":
            # Create write head (process -> memory)
            self.write_head = LatentAttention(
                embed_dim=embed_dims,
                latent_size=latent_size,
            )

            # Create read head (memory -> process)
            self.read_head = LatentAttention(
                embed_dim=embed_dims,
                latent_size=latent_size,
            )
        elif self.rw_head_type == "lca":
            # Create write head (process -> memory)
            self.write_head = LatentCrossAttention(
                embed_dim=embed_dims,
                latent_size=latent_size,
                num_heads=num_heads,
            )

            # Create read head (memory -> process)
            self.read_head = LatentCrossAttention(
                embed_dim=embed_dims,
                latent_size=latent_size,
                num_heads=num_heads,
            )

    def forward(self, inputs):
        B = inputs.shape[0]

        memory, memory_hw_shape = self.patch_embed(inputs)
        process, process_hw_shape = self.process_embedder(inputs)
        
        cls_tokens = self.cls_token.expand(B, -1, -1)
        memory = torch.cat((cls_tokens, memory), dim=1)
        memory = self._pos_embeding(memory, memory_hw_shape, self.pos_embed)

        process = torch.cat((cls_tokens, process), dim=1)
        process = self._pos_embeding(process, process_hw_shape, self.pos_embed)


        if not self.with_cls_token:
            # Remove class token for transformer encoder input
            memory = memory[:, 1:]
            process = process[:, 1:]

        if self.pre_norm:
            process = self.pre_ln(process)
            memory = self.pre_ln(memory)

        outs = []
        if self.out_origin:
            if self.with_cls_token:
                # Remove class token and reshape token for decoder head
                out = memory[:, 1:]
            else:
                out = memory
            B, _, C = out.shape
            out = (
                out.reshape(B, memory_hw_shape[0], memory_hw_shape[1], C)
                .permute(0, 3, 1, 2)
                .contiguous()
            )
            if self.output_cls_token:
                out = [out, memory[:, 0]]
            outs.append(out)

        for i, layer in enumerate(self.layers):
            process = process + self.read_head(memory, process)
            process = layer(process)
            memory = memory + self.write_head(process, memory)
            if i == len(self.layers) - 1:
                if self.final_norm:
                    memory = self.norm1(memory)
            if i in self.out_indices:
                if self.with_cls_token:
                    # Remove class token and reshape token for decoder head
                    out = memory[:, 1:]
                else:
                    out = memory
                B, _, C = out.shape
                out = (
                    out.reshape(B, memory_hw_shape[0], memory_hw_shape[1], C)
                    .permute(0, 3, 1, 2)
                    .contiguous()
                )
                if self.output_cls_token:
                    out = [out, memory_hw_shape[:, 0]]
                outs.append(out)

        return tuple(outs)
