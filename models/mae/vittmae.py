from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F

from timm.models.vision_transformer import Block
from ..vittm.factory import vittm_factory


class MaskedAutoEncoderViTTM(nn.Module):
    def __init__(
        self,
        encoder: nn.Module,
        img_size=224,
        in_chans=3,
        decoder_embed_dim=512,
        decoder_depth=8,
        decoder_num_heads=16,
        mlp_ratio=4.0,
        norm_layer=nn.LayerNorm,
        norm_pix_loss=False,
    ) -> None:
        super().__init__()
        self.encoder = encoder
        self.has_class_token = encoder.has_class_token
        self.mem_class_token = encoder.memory_block != "conv"

        # decoder setup
        self.decoder_embed_dim = decoder_embed_dim
        self.decoder_depth = decoder_depth
        self.decoder_num_heads = decoder_num_heads
        self.norm_pix_loss = norm_pix_loss

        self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_embed_dim))

        # Memory Decoder
        self.memory_decoder_embed = nn.Linear(
            self.encoder.embed_dim,
            decoder_embed_dim,
            bias=True,
        )
        self.memory_decoder_pos_embed = nn.Parameter(
            torch.randn(1, self.encoder.memory_size, decoder_embed_dim) * 0.02,
        )

        self.memory_decoder_blocks = nn.ModuleList(
            [
                Block(
                    decoder_embed_dim,
                    decoder_num_heads,
                    mlp_ratio,
                    qkv_bias=True,
                    norm_layer=norm_layer,
                )
                for i in range(decoder_depth)
            ]
        )

        self.memory_decoder_norm = norm_layer(decoder_embed_dim)
        self.memory_decoder_pred = nn.Linear(
            decoder_embed_dim, self.encoder.memory_ps**2 * in_chans, bias=True
        )  # decoder to patch

        # Process Decoder
        self.process_decoder_embed = nn.Linear(
            self.encoder.embed_dim,
            decoder_embed_dim,
            bias=True,
        )
        self.process_decoder_pos_embed = nn.Parameter(
            torch.randn(1, self.encoder.process_size, decoder_embed_dim) * 0.02,
        )

        self.process_decoder_blocks = nn.ModuleList(
            [
                Block(
                    decoder_embed_dim,
                    decoder_num_heads,
                    mlp_ratio,
                    qkv_bias=True,
                    norm_layer=norm_layer,
                )
                for i in range(decoder_depth)
            ]
        )

        self.process_decoder_norm = norm_layer(decoder_embed_dim)
        self.process_decoder_pred = nn.Linear(
            decoder_embed_dim, self.encoder.process_ps**2 * in_chans, bias=True
        )  # decoder to patch

        self.initialize_weights()

    def initialize_weights(self):
        torch.nn.init.normal_(self.mask_token, std=0.02)

        # Do this manually because encoder is already initialized.
        self.memory_decoder_embed.apply(self._init_weights)
        self.memory_decoder_blocks.apply(self._init_weights)
        self.memory_decoder_norm.apply(self._init_weights)
        self.memory_decoder_pred.apply(self._init_weights)

        self.process_decoder_embed.apply(self._init_weights)
        self.process_decoder_blocks.apply(self._init_weights)
        self.process_decoder_norm.apply(self._init_weights)
        self.process_decoder_pred.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            # we use xavier_uniform following official JAX ViT:
            torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def patchify(self, imgs, patch_size):
        """
        imgs: (N, 3, H, W)
        x: (N, L, patch_size**2 *3)
        """
        p = patch_size
        assert imgs.shape[2] == imgs.shape[3] and imgs.shape[2] % p == 0

        h = w = imgs.shape[2] // p
        x = imgs.reshape(shape=(imgs.shape[0], 3, h, p, w, p))
        x = torch.einsum("nchpwq->nhwpqc", x)
        x = x.reshape(shape=(imgs.shape[0], h * w, p**2 * 3))
        return x

    def unpatchify(self, x, patch_size):
        """
        x: (N, L, patch_size**2 *3)
        imgs: (N, 3, H, W)
        """
        p = patch_size
        h = w = int(x.shape[1] ** 0.5)
        assert h * w == x.shape[1]

        x = x.reshape(shape=(x.shape[0], h, w, p, p, 3))
        x = torch.einsum("nhwpqc->nchpwq", x)
        imgs = x.reshape(shape=(x.shape[0], 3, h * p, h * p))
        return imgs

    def forward_memory_decoder(self, x, keep_mask, mask):
        # embed tokens
        x = self.memory_decoder_embed(x)

        B, _, D = x.shape

        # append mask tokens to sequence
        mask_tokens = self.mask_token.repeat(x.shape[0], mask.shape[1], 1)
        mask_pos_embed = torch.gather(
            self.memory_decoder_pos_embed.expand(B, -1, -1),
            dim=1,
            index=mask.unsqueeze(-1).expand(-1, -1, D),
        )

        pos_embed = torch.gather(
            self.memory_decoder_pos_embed.expand(B, -1, -1),
            dim=1,
            index=keep_mask.unsqueeze(-1).expand(-1, -1, D),
        )

        x = x + pos_embed
        mask_tokens = mask_tokens + mask_pos_embed

        x = torch.cat([x, mask_tokens], dim=1)  # no cls token

        # apply Transformer blocks
        for blk in self.memory_decoder_blocks:
            x = blk(x)
        x = self.memory_decoder_norm(x)

        # predictor projection
        x = self.memory_decoder_pred(x)
        x = x[:, keep_mask.shape[1] :]  # select mask tokens
        return x

    def forward_process_decoder(self, x, keep_mask, mask):
        # embed tokens
        x = self.process_decoder_embed(x)

        B, _, D = x.shape

        # append mask tokens to sequence
        mask_tokens = self.mask_token.repeat(x.shape[0], mask.shape[1], 1)
        mask_pos_embed = torch.gather(
            self.process_decoder_pos_embed.expand(B, -1, -1),
            dim=1,
            index=mask.unsqueeze(-1).expand(-1, -1, D),
        )

        pos_embed = torch.gather(
            self.process_decoder_pos_embed.expand(B, -1, -1),
            dim=1,
            index=keep_mask.unsqueeze(-1).expand(-1, -1, D),
        )

        x = x + pos_embed
        mask_tokens = mask_tokens + mask_pos_embed

        x = torch.cat([x, mask_tokens], dim=1)  # no cls token

        # apply Transformer blocks
        for blk in self.process_decoder_blocks:
            x = blk(x)
        x = self.process_decoder_norm(x)

        # predictor projection
        x = self.process_decoder_pred(x)
        x = x[:, keep_mask.shape[1] :]  # select mask tokens
        return x

    def forward_loss(self, imgs, pred, mask, patch_size):
        """
        imgs: [N, 3, H, W]
        pred: [N, L, p*p*3]
        mask: [N, L], 0 is keep, 1 is remove,
        """
        target = self.patchify(imgs, patch_size)
        if self.norm_pix_loss:
            mean = target.mean(dim=-1, keepdim=True)
            var = target.var(dim=-1, keepdim=True)
            target = (target - mean) / (var + 1.0e-6) ** 0.5

        target = target.gather(
            dim=1, index=mask.unsqueeze(-1).expand(-1, -1, target.shape[-1])
        )

        loss = (pred - target) ** 2
        loss = loss.mean(dim=-1)  # [N, L], mean loss per patch

        loss = loss.mean()
        return loss

    def forward(
        self,
        imgs,
        mask_process: torch.Tensor | None = None,
        mask_memory: torch.Tensor | None = None,
        keep_process: torch.Tensor | None = None,
        keep_memory: torch.Tensor | None = None,
    ):
        memory_loss = 0
        process_loss = 0
        if isinstance(mask_memory, torch.Tensor) and not isinstance(
            mask_process, torch.Tensor
        ):
            # print("Masking Memory...")
            memory, process, extras = self.encoder.forward_features(
                imgs, keep_mask_memory=keep_memory, keep_mask_process=None
            )
            memory_pred = self.forward_memory_decoder(memory, keep_memory, mask_memory)
            memory_loss += self.forward_loss(
                imgs, memory_pred, extras["memory_mask"], self.encoder.memory_ps
            )
        if isinstance(mask_process, torch.Tensor) and not isinstance(
            mask_memory, torch.Tensor
        ):
            # print("Masking Process...")
            memory, process, extras = self.encoder.forward_features(
                imgs, keep_mask_memory=None, keep_mask_process=keep_process
            )
            process_pred = self.forward_process_decoder(
                process, keep_process, mask_process
            )
            process_loss += self.forward_loss(
                imgs, process_pred, extras["process_mask"], self.encoder.process_ps
            )
        if isinstance(mask_memory, torch.Tensor) and isinstance(
            mask_process, torch.Tensor
        ):
            # print("Masking Both...")
            memory, process = self.encoder.forward_features(
                imgs, keep_mask_memory=keep_memory, keep_mask_process=keep_process
            )
            memory_pred = self.forward_memory_decoder(memory, keep_memory, mask_memory)
            process_pred = self.forward_process_decoder(
                process, keep_process, mask_process
            )
            memory_loss += self.forward_loss(
                imgs, memory_pred, mask_memory, self.encoder.memory_ps
            )
            process_loss += self.forward_loss(
                imgs, process_pred, mask_process, self.encoder.process_ps
            )

        return (memory_loss, process_loss), {
            "memory_pred": (
                memory_pred if isinstance(mask_memory, torch.Tensor) else None
            ),
            "process_pred": (
                process_pred if isinstance(mask_process, torch.Tensor) else None
            ),
            "memory_mask": (
                mask_memory if isinstance(mask_memory, torch.Tensor) else None
            ),
            "process_mask": (
                mask_process if isinstance(mask_process, torch.Tensor) else None
            ),
        }


if __name__ == "__main__":
    encoder = vittm_factory(
        "base",
        pretrained=False,
        memory_ps=16,
        process_ps=32,
        rw_head_type="lin",
        fusion_type="residual",
        num_classes=1000,
    )
    model = MaskedAutoEncoderViTTM(encoder)
    x = torch.randn(1, 3, 224, 224)
    (memory_loss, process_loss), _ = model.forward(
        x, masking_ratio=0.75, mask_both=True
    )
    print(memory_loss.item(), process_loss.item())
