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

class NemoFeedForwardModule(nn.Module):
    def __init__(self, d_model, d_ff, dropout, activation = None):
        super().__init__()
        if activation is None:
            self.activation = nn.SiLU()
        self.d_model = d_model
        self.d_ff = d_ff
        self.use_bias = True
        self.linear1 = nn.Linear(d_model, d_ff, bias =  self.use_bias)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(d_ff, d_model, bias = self.use_bias)

    def forward(self, x):
        x = self.linear1(x)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.linear2(x)
        return x