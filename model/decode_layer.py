import torch
import torch.nn as nn

from model.feedforward import FeedForward
from model.multihead_attention import MultiHeadAttention


class DecoderLayer(nn.Module):
    def __init__(self, d_model, n_heads, d_ff, dropout=0.1):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, n_heads)
        self.cross_attn = MultiHeadAttention(d_model, n_heads)
        self.ff = FeedForward(d_model, d_ff, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, tgt, memory, src_mask=None, tgt_mask=None):
        print("DecoderLayer Forward:")
        print("tgt shape:", tgt.shape)
        print("memory shape:", memory.shape)
        print("src_mask shape:", src_mask.shape if src_mask is not None else "None")
        print("tgt_mask shape:", tgt_mask.shape if tgt_mask is not None else "None")

        # Self-attention
        x = self.norm1(tgt)
        x = tgt + self.dropout(self.self_attn(x, x, x, mask=tgt_mask))

        # Cross-attention
        x = self.norm2(x)
        x = x + self.dropout(self.cross_attn(x, memory, memory, mask=src_mask))

        # Feed forward
        x = self.norm3(x)
        x = x + self.dropout(self.ff(x))
        return x
