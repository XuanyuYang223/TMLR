"""Equivalent SDPA attention for the local long-running experiment only."""
import types
import torch
from torch.nn import functional as F


def sdpa_forward(self, hidden, valid_tokens):
    batch, length, width = hidden.shape
    qkv = self.qkv_projection(hidden).reshape(batch, length, 3, self.n_heads, self.head_dim).permute(2, 0, 3, 1, 4)
    query, key, value = qkv.unbind(0)
    allowed = self.causal_mask[:length, :length].view(1, 1, length, length) & valid_tokens[:, None, None, :]
    attended = F.scaled_dot_product_attention(query, key, value, attn_mask=allowed,
                                             dropout_p=self.attention_dropout.p if self.training else 0., scale=self.scale)
    attended = attended.transpose(1, 2).reshape(batch, length, width)
    attended = self.output_dropout(self.output_projection(attended))
    return attended * valid_tokens.unsqueeze(-1).to(attended.dtype)


def accelerate(model):
    for block in model.blocks:
        block.attention.forward = types.MethodType(sdpa_forward, block.attention)
    return model
