import torch
import torch.nn as nn

from model.decoder import Decoder
from model.encoder import Encoder


class Transformer(nn.Module):
    def __init__(self, src_vocab_size, tgt_vocab_size, d_model=512, n_layers=6, n_heads=8, d_ff=2048, dropout=0.1):
        super().__init__()
        self.encoder = Encoder(src_vocab_size, d_model, n_layers, n_heads, d_ff, dropout)
        self.decoder = Decoder(tgt_vocab_size, d_model, n_layers, n_heads, d_ff, dropout)
        self.out_linear = nn.Linear(d_model, tgt_vocab_size)

    def forward(self, src, tgt, src_mask=None, tgt_mask=None):
        memory = self.encoder(src, src_mask)
        output = self.decoder(tgt, memory, src_mask=src_mask, tgt_mask=tgt_mask)
        output = self.out_linear(output)
        return output
