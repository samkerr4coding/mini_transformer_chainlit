import torch
import torch.nn as nn

from model.encode_layer import EncoderLayer
from model.positional_encoding import PositionalEncoding


class Encoder(nn.Module):
    def __init__(self, vocab_size, d_model, n_layers, n_heads, d_ff, dropout=0.1):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, d_model)
        self.pos = PositionalEncoding(d_model)
        self.layers = nn.ModuleList([EncoderLayer(d_model, n_heads, d_ff, dropout) for _ in range(n_layers)])
        self.norm = nn.LayerNorm(d_model)

    def forward(self, src, mask):
        x = self.embed(src)
        x = self.pos(x)
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)
