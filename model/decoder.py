import torch
import torch.nn as nn

from model.decode_layer import DecoderLayer
from model.positional_encoding import PositionalEncoding


class Decoder(nn.Module):
    def __init__(self, vocab_size, d_model, n_layers, n_heads, d_ff, dropout=0.1):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, d_model)
        self.pos = PositionalEncoding(d_model)
        self.layers = nn.ModuleList([DecoderLayer(d_model, n_heads, d_ff, dropout) for _ in range(n_layers)])
        self.norm = nn.LayerNorm(d_model)

    def forward(self, tgt, memory, src_mask=None, tgt_mask=None):
        print("Decoder Forward:")
        print("tgt shape:", tgt.shape)
        print("memory shape:", memory.shape)
        print("src_mask shape:", src_mask.shape if src_mask is not None else "None")
        print("tgt_mask shape:", tgt_mask.shape if tgt_mask is not None else "None")

        x = self.embed(tgt)
        x = self.pos(x)
        for layer in self.layers:
            x = layer(x, memory, src_mask=src_mask, tgt_mask=tgt_mask)
        return self.norm(x)
