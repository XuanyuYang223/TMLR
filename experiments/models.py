"""Matched shared encoders and independent target adaptations."""
import math
import torch
from torch import nn
from torch.nn import functional as F


class Encoder(nn.Module):
    def __init__(self, inputs, hidden, features):
        super().__init__()
        self.fc1 = nn.Linear(inputs, hidden)
        self.fc2 = nn.Linear(hidden, features)
        self.norm = nn.LayerNorm(features)

    def forward(self, x):
        return self.norm(F.gelu(self.fc2(F.gelu(self.fc1(x)))))


class SourceModel(nn.Module):
    def __init__(self, inputs, hidden, features, p):
        super().__init__()
        self.encoder = Encoder(inputs, hidden, features)
        self.heads = nn.Linear(features, 4 * p)
        self.p = p

    def forward(self, x):
        return self.heads(self.encoder(x)).view(len(x), 4, self.p)


class BatchedTargets(nn.Module):
    """Independent copies, with no shared parameters across target tasks."""
    def __init__(self, encoder, count, p, seed, frozen=False):
        super().__init__()
        values = {'w1': encoder.fc1.weight, 'b1': encoder.fc1.bias,
                  'w2': encoder.fc2.weight, 'b2': encoder.fc2.bias,
                  'nw': encoder.norm.weight, 'nb': encoder.norm.bias}
        self.backbone = nn.ParameterDict({key: nn.Parameter(value.detach().unsqueeze(0).repeat(count, *([1] * value.ndim)), requires_grad=not frozen)
                                          for key, value in values.items()})
        self.eps = encoder.norm.eps
        device = encoder.fc1.weight.device
        generator = torch.Generator(device=device).manual_seed(seed)
        features = encoder.fc2.out_features
        self.head_weight = nn.Parameter(torch.randn(count, p, features, generator=generator, device=device) / math.sqrt(features))
        self.head_bias = nn.Parameter(torch.zeros(count, p, device=device))

    def encode(self, x):
        if x.ndim == 2:
            x = x.unsqueeze(0).expand(len(self.head_weight), -1, -1)
        h = F.gelu(torch.bmm(x, self.backbone['w1'].transpose(1, 2)) + self.backbone['b1'][:, None])
        h = F.gelu(torch.bmm(h, self.backbone['w2'].transpose(1, 2)) + self.backbone['b2'][:, None])
        h = (h - h.mean(-1, keepdim=True)) * torch.rsqrt(h.var(-1, keepdim=True, unbiased=False) + self.eps)
        return h * self.backbone['nw'][:, None] + self.backbone['nb'][:, None]

    def forward(self, x):
        return torch.bmm(self.encode(x), self.head_weight.transpose(1, 2)) + self.head_bias[:, None]
