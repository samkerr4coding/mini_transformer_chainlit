**Overall Architecture and Purpose**

**1. `feedforward.py`**

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.1):
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(d_ff, d_model)

    def forward(self, x):
        x = self.dropout(F.relu(self.linear1(x)))
        x = self.linear2(x)
        return x
```

*   **Purpose:** This file defines a simple feedforward network.  It's a two-layer fully connected network with a ReLU activation and dropout.
*   **`__init__`:**
    *   `self.linear1 = nn.Linear(d_model, d_ff)`:  Defines the first linear layer, projecting the input from `d_model` dimensions to `d_ff` (feedforward dimension) dimensions. `d_model` is the standard dimensionality used throughout the transformer, and `d_ff` is typically larger (e.g., 4 times `d_model`) to provide more capacity.
    *   `self.dropout = nn.Dropout(dropout)`:  Defines a dropout layer for regularization.
    *   `self.linear2 = nn.Linear(d_ff, d_model)`: Defines the second linear layer, projecting the output back to `d_model` dimensions.
*   **`forward`:**
    *   `x = self.dropout(F.relu(self.linear1(x)))`: Applies the first linear layer, ReLU activation, and dropout.  The ReLU introduces non-linearity, which is crucial for the network to learn complex functions.
    *   `x = self.linear2(x)`: Applies the second linear layer.
    *   `return x`: Returns the output.

**Role:** The feedforward network adds non-linearity to the model and helps to learn more complex relationships in the data after the attention mechanism. It operates on each position in the sequence independently.

**2. `multihead_attention.py`**

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, n_heads):
        super().__init__()
        assert d_model % n_heads == 0
        self.d_k = d_model // n_heads
        self.h = n_heads
        self.q_linear = nn.Linear(d_model, d_model)
        self.k_linear = nn.Linear(d_model, d_model)
        self.v_linear = nn.Linear(d_model, d_model)
        self.out = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(0.1)

    def attention(self, q, k, v, mask=None):
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_k)
        if mask is not None:
            # Expand mask to have the same number of heads
            mask = mask.unsqueeze(1)  # Add a dimension for the heads
            scores = scores.masked_fill(mask == 0, float('-inf'))
        p_attn = F.softmax(scores, dim=-1)
        p_attn = self.dropout(p_attn)
        return torch.matmul(p_attn, v), p_attn

    def forward(self, q, k, v, mask=None):
        bs = q.size(0)

        # Perform linear operation and split into h heads
        q = self.q_linear(q).view(bs, -1, self.h, self.d_k).transpose(1, 2)
        k = self.k_linear(k).view(bs, -1, self.h, self.d_k).transpose(1, 2)
        v = self.v_linear(v).view(bs, -1, self.h, self.d_k).transpose(1, 2)

        # Apply attention
        x, attn = self.attention(q, k, v, mask=mask)

        # Concatenate heads and put through final linear layer
        x = x.transpose(1, 2).contiguous().view(bs, -1, self.h * self.d_k)
        return self.out(x)
```

*   **Purpose:** This file implements the multi-head attention mechanism.  It allows the model to attend to different parts of the input sequence in parallel, capturing different relationships.
*   **`__init__`:**
    *   `assert d_model % n_heads == 0`:  Ensures that the model dimension is divisible by the number of heads.
    *   `self.d_k = d_model // n_heads`:  Calculates the dimension of each head.
    *   `self.h = n_heads`: Stores the number of heads.
    *   `self.q_linear = nn.Linear(d_model, d_model)`: Defines the linear layer for transforming the query.
    *   `self.k_linear = nn.Linear(d_model, d_model)`: Defines the linear layer for transforming the key.
    *   `self.v_linear = nn.Linear(d_model, d_model)`: Defines the linear layer for transforming the value.
    *   `self.out = nn.Linear(d_model, d_model)`: Defines the linear layer for projecting the concatenated attention outputs back to the original dimension.
    *   `self.dropout = nn.Dropout(0.1)`: Defines a dropout layer.
*   **`attention`:**
    *   `scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_k)`: Calculates the attention scores by taking the dot product of the query and key, scaled by the square root of the key dimension.  The scaling prevents the dot products from becoming too large, which can lead to vanishing gradients after the softmax.
    *   `if mask is not None:`: Applies a mask to the attention scores.  This is used to prevent the model from attending to padding tokens or future tokens (in the decoder).
        *   `mask = mask.unsqueeze(1)`: Adds a dimension for the heads so the mask applies correctly across all heads.
        *   `scores = scores.masked_fill(mask == 0, float('-inf'))`: Fills masked positions with negative infinity, so they have zero probability after the softmax.
    *   `p_attn = F.softmax(scores, dim=-1)`: Applies the softmax function to the scores to obtain attention weights.
    *   `p_attn = self.dropout(p_attn)`: Applies dropout to the attention weights.
    *   `return torch.matmul(p_attn, v), p_attn`: Returns the weighted values and the attention probabilities.
*   **`forward`:**
    *   `bs = q.size(0)`: Gets the batch size.
    *   `q = self.q_linear(q).view(bs, -1, self.h, self.d_k).transpose(1, 2)`: Transforms the query, reshapes it into `(batch_size, num_heads, seq_len, d_k)`, and transposes it to prepare for attention calculation.  The key and value are transformed similarly.
    *   `x, attn = self.attention(q, k, v, mask=mask)`: Applies the attention mechanism.
    *   `x = x.transpose(1, 2).contiguous().view(bs, -1, self.h * self.d_k)`: Concatenates the attention heads and reshapes the output.
    *   `return self.out(x)`: Projects the concatenated output back to the original dimension.

**Role:** Multi-head attention allows the model to focus on different parts of the input sequence when processing it.  The multiple "heads" allow the model to learn different relationships between words.

**3. `encode_layer.py`**

```python
import torch
import torch.nn as nn

from model.feedforward import FeedForward
from model.multihead_attention import MultiHeadAttention


class EncoderLayer(nn.Module):
    def __init__(self, d_model, n_heads, d_ff, dropout=0.1):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, n_heads)
        self.ff = FeedForward(d_model, d_ff, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask):
        x = self.norm1(x + self.dropout(self.self_attn(x, x, x, mask=mask)))
        x = self.norm2(x + self.dropout(self.ff(x)))
        return x
```

*   **Purpose:** This file defines a single layer of the encoder.  The encoder consists of multiple identical layers stacked on top of each other.
*   **`__init__`:**
    *   `self.self_attn = MultiHeadAttention(d_model, n_heads)`: Defines the multi-head self-attention layer.  "Self-attention" means the layer attends to the input sequence itself.
    *   `self.ff = FeedForward(d_model, d_ff, dropout)`: Defines the feedforward network.
    *   `self.norm1 = nn.LayerNorm(d_model)`: Defines the first layer normalization layer.
    *   `self.norm2 = nn.LayerNorm(d_model)`: Defines the second layer normalization layer.
    *   `self.dropout = nn.Dropout(dropout)`: Defines a dropout layer.
*   **`forward`:**
    *   `x = self.norm1(x + self.dropout(self.self_attn(x, x, x, mask=mask)))`:  Applies self-attention, dropout, and then adds the result to the original input (`x`).  This is a residual connection.  Layer normalization is applied *before* the residual connection. The `mask` prevents the attention mechanism from attending to padding tokens.  The query, key, and value are all the same input `x` in self-attention.
    *   `x = self.norm2(x + self.dropout(self.ff(x)))`: Applies the feedforward network, dropout, and adds the result to the output of the previous step (again, a residual connection with layer normalization before).
    *   `return x`: Returns the output.

**Role:** The encoder layer's primary role is to process the input sequence and extract meaningful features.  Self-attention allows each word in the input to attend to all other words, capturing contextual information.  The feedforward network further processes the output of the attention mechanism.  Residual connections and layer normalization help with training deep networks.

**4. `decode_layer.py`**

```python
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
```

*   **Purpose:** This file defines a single layer of the decoder.  Like the encoder, the decoder consists of multiple identical layers stacked on top of each other.
*   **`__init__`:**
    *   `self.self_attn = MultiHeadAttention(d_model, n_heads)`: Defines the multi-head self-attention layer.  In the decoder, self-attention attends to the *target* sequence (the sequence being generated).
    *   `self.cross_attn = MultiHeadAttention(d_model, n_heads)`: Defines the multi-head *cross-attention* layer.  Cross-attention attends to the *encoder's output* (`memory`).
    *   `self.ff = FeedForward(d_model, d_ff, dropout)`: Defines the feedforward network.
    *   `self.norm1 = nn.LayerNorm(d_model)`: Defines the first layer normalization layer.
    *   `self.norm2 = nn.LayerNorm(d_model)`: Defines the second layer normalization layer.
    *   `self.norm3 = nn.LayerNorm(d_model)`: Defines the third layer normalization layer.
    *   `self.dropout = nn.Dropout(dropout)`: Defines a dropout layer.
*   **`forward`:**
    *   `x = self.norm1(tgt)`: Layer normalization of the target input.
    *   `x = tgt + self.dropout(self.self_attn(x, x, x, mask=tgt_mask))`: Applies self-attention to the target sequence (`tgt`).  The `tgt_mask` is crucial here to prevent the decoder from "cheating" by attending to future tokens in the target sequence during training (causal masking).
    *   `x = self.norm2(x)`: Layer normalization.
    *   `x = x + self.dropout(self.cross_attn(x, memory, memory, mask=src_mask))`: Applies cross-attention.  The query comes from the output of the self-attention layer (`x`), and the key and value come from the encoder's output (`memory`).  This allows the decoder to attend to the relevant parts of the input sequence when generating the output. The `src_mask` is used to mask padding tokens in the source sequence.
    *   `x = self.norm3(x)`: Layer normalization.
    *   `x = x + self.dropout(self.ff(x))`: Applies the feedforward network.
    *   `return x`: Returns the output.

**Role:** The decoder layer generates the output sequence, attending to both the previously generated tokens (through self-attention) and the encoded input sequence (through cross-attention).  The self-attention ensures that the decoder considers the context of the tokens it has already generated, while the cross-attention allows it to focus on the relevant parts of the input sequence.

**Why These Calls?**

*   **Encoder:**
    *   *Self-Attention:*  To understand the relationships between different parts of the *input* sequence.  Each word can "see" all other words in the input and determine how relevant they are.
    *   *Feedforward:* To add non-linearity and further process the information learned by the attention mechanism.
*   **Decoder:**
    *   *Self-Attention:* To understand the relationships between different parts of the *output* sequence being generated.  Each word can "see" the words that have already been generated and determine how to generate the next word.  Crucially, it's masked to prevent attending to future tokens.
    *   *Cross-Attention:* To attend to the *encoded input* sequence.  This allows the decoder to focus on the relevant parts of the input when generating each word of the output.
    *   *Feedforward:* To add non-linearity and further process the information learned by the attention mechanisms.

**Simple Example**

Let