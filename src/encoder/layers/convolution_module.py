import torch
from torch import nn

class GatedSiLU(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        a,gate = x.chunk(2, dim = -1)
        return nn.functional.silu(gate) * a

class ConvolutionModule(nn.Module):
    def __init__(self, hidden_size, kernel_size = 31):
        super().__init__()
        self.layer_norm = nn.RMSNorm(hidden_size)

        self.point_wise = nn.Conv1d(hidden_size, hidden_size *2, kernel_size=1, stride = 1,bias=True, padding = 0)
        self.gated_silu = GatedSiLU(hidden_size * 2)

        self.depth_wise = nn.Conv1d(hidden_size, hidden_size,kernel_size = kernel_size,
                                    groups = hidden_size, bias = False , padding = (kernel_size -1) // 2)
        self.batch_norm = nn.BatchNorm1d(hidden_size)
        self.swish = nn.SiLU()
        self.final_point_wise = nn.Conv1d(hidden_size, hidden_size, kernel_size=1, bias=True)

        self.dropout = nn.Dropout(0.1)

    def forward(self, x):
        residual = x
        x = self.layer_norm(x)
        x = x.permute(0,2,1)

        x = self.point_wise(x)
        x = x.permute(0, 2, 1)

        x = self.gated_silu(x)

        x = x.permute(0, 2, 1)
        x = self.depth_wise(x)
        x = self.batch_norm(x)
        x = x.permute(0, 2, 1)

        x = self.swish(x)

        x = x.permute(0, 2, 1)
        x = self.final_point_wise(x)
        x = x.permute(0, 2, 1)

        x = self.dropout(x) + residual

        return x


if __name__ == "__main__":
    x = torch.randn(2, 750, 256)
    conv = ConvolutionModule(hidden_size=256, kernel_size=31)
    output = conv(x)
    print(output.shape)  # Should be (1, 100, 512)