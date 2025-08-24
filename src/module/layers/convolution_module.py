import torch
from torch import nn
from src.utils.conv import CausalConv1D

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


class NemoConvolutionModule(nn.Module):
    def __init__(self, d_model, kernel_size, use_bias = True):
        super().__init__()
        assert (kernel_size -1) % 2 == 0
        self.d_model = d_model
        self.kernel_size = kernel_size

        conv_context_size = (kernel_size -1) // 2
        self.pointwise_activation = 'glu_'
        dw_conv_input_dim = d_model

        self.pointwise_conv1 = nn.Conv1d(
            in_channels = d_model,
            out_channels=d_model * 2,
            kernel_size=1,
            stride = 1,
            padding = 0,
            bias = use_bias
        )

        self.depthwise_conv = CausalConv1D(
            in_channels = dw_conv_input_dim,
            out_channels = dw_conv_input_dim,
            kernel_size=kernel_size,
            stride = 1,
            padding = conv_context_size,
            groups = dw_conv_input_dim,
            bias = use_bias
        )
        self.batch_norm = nn.BatchNorm1d(dw_conv_input_dim)
        self.activation = nn.SiLU()
        self.pointwise_conv2 = nn.Conv1d(
            in_channels = dw_conv_input_dim,
            out_channels = d_model,
            kernel_size = 1,
            stride = 1,
            padding = 0,
            bias = use_bias
        )

    def forward(self, x, pad_mask = None, cache = None):
        x = x.transpose(1,2)
        x = self.pointwise_conv1(x)

        x = nn.functional.glu(x, dim = 1)

        x = x.masked_fill(pad_mask.unsqueeze(1), 0.0)
        x = self.depthwise_conv(x, cache = cache)

        if cache is not None:
            x, cache = x

        x = self.batch_norm(x)

        x = self.activation(x)
        x = self.pointwise_conv2(x)
        x = x.transpose(1,2)
        if cache is None:
            return x
        return x, cache



if __name__ == "__main__":
    x = torch.randn(2, 750, 256)
    conv = ConvolutionModule(hidden_size=256, kernel_size=31)
    output = conv(x)
    print(output.shape)  # Should be (1, 100, 512)