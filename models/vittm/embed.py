import torch
import torch.nn as nn
import torch.nn.functional as F

class LatentEmbedding(nn.Module):
    def __init__(self, process_tokens, embed_dim) -> None:
        super().__init__()
        self.process_tokens = process_tokens
        self.process_token = nn.Parameter(torch.randn(1, process_tokens, embed_dim))

    def forward(self, batch_size):
        return self.process_token.expand(batch_size, -1, -1)