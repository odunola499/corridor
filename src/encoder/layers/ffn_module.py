import torch
from torch import nn

class FeedForwardModule(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.up_proj = nn.Linear(hidden_size, hidden_size * 4)
        self.down_proj = nn.Linear(hidden_size * 4, hidden_size)
        self.gate = nn.Linear(hidden_size, hidden_size * 4)
        self.activation = nn.GELU()

    def forward(self, x):
        residual = x
        up = self.up_proj(x)
        gate = self.gate(x)
        x = self.activation(gate) * up
        x = self.down_proj(x)
        x += residual
        return x

