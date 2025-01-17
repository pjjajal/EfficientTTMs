import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.layers import Mlp


class TokenLearner(nn.Module):
    def __init__(
        self,
        embed_dim,
        out_features,
        bottleneck_size=64,
        act_layer=nn.GELU,
        norm_layer=nn.LayerNorm,
        drop=0.0,
    ) -> None:
        super().__init__()
        self.mlp = Mlp(
            in_features=embed_dim,
            hidden_features=bottleneck_size,
            out_features=out_features,
            act_layer=act_layer,
            drop=drop,
        )
        self.norm = norm_layer(embed_dim)

    def forward(
        self, input_tokens: torch.Tensor, query_tokens: torch.Tensor
    ) -> torch.Tensor:
        # input_tokens: (B, N, C), query_tokens: (B, M, C)
        catted_tokens = torch.cat([input_tokens, query_tokens], dim=1)
        selected = self.norm(catted_tokens)
        selected = self.mlp(
            selected
        )  # (B, N + M, C) -> (B, N + M, M), e.g., M would be memory size.
        selected = F.softmax(selected.mT, dim=-1)  # (B, N + M, M) -> (B, M, N + M)

        output_tokens = selected @ catted_tokens

        return output_tokens


class CrossAttention(nn.Module):
    def __init__(
        self,
        embed_dim,
        num_heads=8,
    ) -> None:
        super().__init__()
        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads."
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scale = self.head_dim**-0.5

        self.q = nn.Linear(embed_dim, embed_dim)
        self.q_norm = nn.LayerNorm(embed_dim)

        self.k = nn.Linear(embed_dim, embed_dim)
        self.k_norm = nn.LayerNorm(embed_dim)

        self.v = nn.Linear(embed_dim, embed_dim)
        self.v_norm = nn.LayerNorm(embed_dim)

        self.proj = nn.Linear(embed_dim, embed_dim)

    def forward(
        self, input_tokens: torch.Tensor, query_tokens: torch.Tensor
    ) -> torch.Tensor:
        B, N, C = input_tokens.shape
        Bq, M, Cq = query_tokens.shape

        q = (
            self.q_norm(self.q(query_tokens))
            .reshape(B, M, self.num_heads, self.head_dim)
            .permute(0, 2, 1, 3)
        )

        k = (
            self.k_norm(self.k(input_tokens))
            .reshape(B, N, self.num_heads, self.head_dim)
            .permute(0, 2, 1, 3)
        )

        v = (
            self.v_norm(self.v(input_tokens))
            .reshape(B, N, self.num_heads, self.head_dim)
            .permute(0, 2, 1, 3)
        )

        # q = q * self.scale
        # attn = q @ k.transpose(-2, -1)
        # attn = attn.softmax(dim=-1)
        # output_tokens = attn @ v

        output_tokens = F.scaled_dot_product_attention(
            q, k, v, attn_mask=None, dropout_p=0.0, scale=self.scale
        )
        output_tokens = output_tokens.transpose(1, 2).reshape(B, M, C)

        output_tokens = self.proj(output_tokens)

        return output_tokens


class LatentAttention(nn.Module):
    def __init__(
        self,
        embed_dim,
        latent_size,
        bottleneck_size=64,
        act_layer=nn.GELU,
        drop=0.0,
    ) -> None:
        super().__init__()
        self.mlp = Mlp(
            in_features=embed_dim,
            hidden_features=bottleneck_size,
            out_features=latent_size,
            act_layer=act_layer,
            drop=drop,
        )
        self.mlp2 = Mlp(
            in_features=embed_dim,
            hidden_features=bottleneck_size,
            out_features=latent_size,
            act_layer=act_layer,
            drop=drop,
        )
        self.norm = nn.LayerNorm(embed_dim)
        self.norm_q = nn.LayerNorm(embed_dim)

    def forward(
        self, input_tokens: torch.Tensor, query_tokens: torch.Tensor
    ) -> torch.Tensor:
        v = input_tokens
        input_tokens = self.norm(input_tokens)
        query_tokens = self.norm_q(query_tokens)

        input_tokens = self.mlp(input_tokens)
        query_tokens = self.mlp2(query_tokens)

        query_tokens = F.softmax(query_tokens, dim=-1)
        input_tokens = F.softmax(input_tokens.mT, dim=-1)

        output_tokens = query_tokens @ (input_tokens @ v)

        return output_tokens


class LatentCrossAttention(nn.Module):
    def __init__(
        self,
        embed_dim,
        latent_size,
        num_heads=8,
    ) -> None:
        super().__init__()
        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads."
        assert (
            latent_size % num_heads == 0
        ), "latent_size must be divisible by num_heads."
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.latent_head_dim = latent_size // num_heads

        self.q = nn.Linear(embed_dim, latent_size)
        self.q_norm = nn.LayerNorm(latent_size)

        self.k = nn.Linear(embed_dim, latent_size)
        self.k_norm = nn.LayerNorm(latent_size)

        self.v = nn.Linear(embed_dim, embed_dim)
        self.v_norm = nn.LayerNorm(embed_dim)

        self.proj = nn.Linear(embed_dim, embed_dim)

    def forward(
        self, input_tokens: torch.Tensor, query_tokens: torch.Tensor
    ) -> torch.Tensor:
        B, N, C = input_tokens.shape
        Bq, M, Cq = query_tokens.shape

        q = (
            self.q_norm(self.q(query_tokens))
            .reshape(B, M, self.num_heads, self.latent_head_dim)
            .permute(0, 2, 1, 3)
        )

        k = (
            self.k_norm(self.k(input_tokens))
            .reshape(B, N, self.num_heads, self.latent_head_dim)
            .permute(0, 2, 1, 3)
        )

        v = (
            self.v_norm(self.v(input_tokens))
            .reshape(B, N, self.num_heads, self.head_dim)
            .permute(0, 2, 1, 3)
        )

        q = q.softmax(dim=-1)
        k = k.transpose(-2, -1).softmax(dim=-1)

        output_tokens = q @ (k @ v)
        output_tokens = output_tokens.transpose(1, 2).reshape(B, M, C)

        output_tokens = self.proj(output_tokens)

        return output_tokens


# implementation borrowed from https://github.com/idiap/fast-transformers/blob/master/fast_transformers/attention/linear_attention.py
class LinearAttention(nn.Module):
    def __init__(self, embed_dim, latent_size, num_heads=8, eps=1e-6) -> None:
        super().__init__()
        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads."
        assert (
            latent_size % num_heads == 0
        ), "latent_size must be divisible by num_heads."
        assert eps > 0, "eps must be positive."
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.latent_head_dim = latent_size // num_heads
        self.eps = eps

        self.q = nn.Linear(embed_dim, latent_size)
        self.k = nn.Linear(embed_dim, latent_size)

        self.v = nn.Linear(embed_dim, embed_dim)

        self.proj = nn.Linear(embed_dim, embed_dim)

    def forward(
        self, input_tokens: torch.Tensor, query_tokens: torch.Tensor
    ) -> torch.Tensor:
        B, N, C = input_tokens.shape
        Bq, M, Cq = query_tokens.shape

        q = self.q(query_tokens).reshape(B, M, self.num_heads, self.latent_head_dim)
        k = self.k(input_tokens).reshape(B, N, self.num_heads, self.latent_head_dim)
        v = self.v(input_tokens).reshape(B, N, self.num_heads, self.head_dim)

        q = F.elu(q) + 1
        k = F.elu(k) + 1

        kv = torch.einsum("nshd,nshm->nhmd", k, v)
        z = 1 / (torch.einsum("nlhd,nhd->nlh", q, k.sum(dim=1)) + self.eps)

        v = torch.einsum("nlhd,nhmd,nlh->nlhm", q, kv, z)
        return v.reshape(B, M, C).contiguous()
    
    # this method can be used to visualize the attention maps
    # this works since lin. attention splits the qk outside the softmax
    # after which it is just associative thus (qk)v = q(kv)
    # the latter form is efficient, whereas the former is more interpretable.
    def compute_attn(self, input_tokens: torch.Tensor, query_tokens: torch.Tensor):
        B, N, C = input_tokens.shape
        Bq, M, Cq = query_tokens.shape
        q = self.q(query_tokens).reshape(B, M, self.num_heads, self.latent_head_dim)
        k = self.k(input_tokens).reshape(B, N, self.num_heads, self.latent_head_dim)
        v = self.v(input_tokens).reshape(B, N, self.num_heads, self.head_dim)

        q = F.elu(q) + 1
        k = F.elu(k) + 1

        z = 1 / (torch.einsum("nlhd,nhd->nlh", q, k.sum(dim=1)) + self.eps)
        attn = torch.einsum("nlhd,nmhd -> nhlm", q, k)
        v = torch.einsum("nhlm, nmhd, nlh -> nlhd", attn, v, z)
        v = v.reshape(B, M, C).contiguous()
        return attn, v, z


# this class follows DynaMixer.
class DynaCrossMixer(nn.Module):
    def __init__(
        self,
        embed_dim,
        input_features,
        out_features,
        reduced_dim=2,
        num_heads=16,
        concat=True,
    ) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.input_features = input_features
        self.out_features = out_features
        self.reduced_dim = reduced_dim
        self.num_heads = num_heads
        self.concat = concat

        input_tokens = input_features
        if self.concat:
            input_tokens = input_features + out_features

        # layers
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        self.compress = nn.Linear(embed_dim, num_heads * reduced_dim)
        self.generate = nn.Linear(
            input_tokens * reduced_dim,
            input_tokens * out_features,
        )
        self.norm = nn.LayerNorm(embed_dim)

    def forward(
        self, input_tokens: torch.Tensor, query_tokens: torch.Tensor
    ) -> torch.Tensor:
        # input_tokens: (B, N, C), query_tokens: (B, M, C)
        if self.concat:
            input_tokens = torch.cat([input_tokens, query_tokens], dim=1)

        input_tokens = self.norm(input_tokens)
        B, N, C = input_tokens.shape
        mixing_weights = (
            self.compress(input_tokens)
            .reshape(B, N, self.num_heads, self.reduced_dim)
            .permute(0, 2, 1, 3)
            .reshape(B, self.num_heads, -1)
        )
        mixing_weights = self.generate(mixing_weights).reshape(
            B, self.num_heads, self.out_features, N
        )
        mixing_weights = F.softmax(mixing_weights, dim=-1)

        input_tokens = input_tokens.reshape(
            B, N, self.num_heads, C // self.num_heads
        ).permute(0, 2, 1, 3)
        output_tokens = mixing_weights @ input_tokens
        output_tokens = output_tokens.permute(0, 2, 1, 3).reshape(
            B, self.out_features, C
        )
        output_tokens = self.out_proj(output_tokens)
        return output_tokens


def create_rw_head(**kwargs):
    rw_head_type = kwargs.get("rw_head_type")
    if rw_head_type == "tl":
        head = nn.ModuleList(
            [
                TokenLearner(
                    embed_dim=kwargs["embed_dim"],
                    out_features=kwargs["out_features"],
                    bottleneck_size=kwargs["bottleneck_size"],
                    act_layer=kwargs["act_layer"],
                    drop=kwargs["drop"],
                )
                for _ in range(kwargs["depth"])
            ]
        )
    elif rw_head_type == "ca":
        head = nn.ModuleList(
            [
                CrossAttention(
                    embed_dim=kwargs["embed_dim"],
                    num_heads=kwargs["num_heads"],
                )
                for _ in range(kwargs["depth"])
            ]
        )
    elif rw_head_type == "la":
        head = nn.ModuleList(
            [
                LatentAttention(
                    embed_dim=kwargs["embed_dim"],
                    latent_size=kwargs["latent_size"],
                )
                for _ in range(kwargs["depth"])
            ]
        )

    elif rw_head_type == "lca":
        head = nn.ModuleList(
            [
                LatentCrossAttention(
                    embed_dim=kwargs["embed_dim"],
                    latent_size=kwargs["latent_size"],
                    num_heads=kwargs["num_heads"],
                )
                for i in range(kwargs["depth"])
            ]
        )
    elif rw_head_type == "lin":
        # Create write head (process -> memory)
        head = nn.ModuleList(
            [
                LinearAttention(
                    embed_dim=kwargs["embed_dim"],
                    latent_size=kwargs["latent_size"],
                    num_heads=kwargs["num_heads"],
                )
                for _ in range(kwargs["depth"])
            ]
        )
    elif rw_head_type == "dyna":
        head = nn.ModuleList(
            [
                DynaCrossMixer(
                    embed_dim=kwargs["embed_dim"],
                    input_features=kwargs["input_features"],
                    out_features=kwargs["out_features"],
                    reduced_dim=kwargs["reduced_dim"],
                    num_heads=kwargs["num_heads"],
                    concat=kwargs["dyna_concat"],
                )
                for _ in range(kwargs["depth"])
            ]
        )
    return head
