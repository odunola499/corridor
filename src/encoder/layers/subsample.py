import torch
from torch import nn
from torch.nn import functional as F

class Subsample(nn.Module):
    def __init__(self, subsample_factor = 4, mel_bins = 80, hidden_size = 512, dropout = 0.0):
        super().__init__()
        assert subsample_factor == 4, "Currently only subsample_factor of 4 is supported."
        self.conv1 = nn.Conv1d(
                in_channels=mel_bins,
                out_channels=hidden_size,
                kernel_size=3,
                stride=2,
                bias=False,
                padding = 1
            )
        self.conv2 = nn.Conv1d(
                in_channels=hidden_size,
                out_channels=hidden_size,
                kernel_size=3,
                stride=2,
                bias=False,
                padding = 1
            )
        self.linear = nn.Linear(hidden_size, hidden_size, bias = False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = F.gelu(self.conv1(x))
        x = F.gelu(self.conv2(x))
        x = x.permute(0,2,1)
        x = self.linear(x)
        x = self.dropout(x)
        return x


if __name__ == "__main__":
    # Example usage
    x = torch.randn(1, 128, 3000)
    subsample_layer = Subsample(subsample_factor=4, mel_bins=128)
    output = subsample_layer(x)
    print(output.shape)
