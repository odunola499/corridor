import torch
from torch import nn
from torch.nn import functional as F
import math

from src.utils.conv import CausalConv2D

def calc_length(lengths, all_paddings, kernel_size, stride, ceil_mode, repeat_num=1):
    """Calculates the output length of a Tensor passed through a convolution or max pooling layer"""
    add_pad: float = all_paddings - kernel_size
    one: float = 1.0
    for i in range(repeat_num):
        lengths = torch.div(lengths.to(dtype=torch.float) + add_pad, stride) + one
        if ceil_mode:
            lengths = torch.ceil(lengths)
        else:
            lengths = torch.floor(lengths)
    return lengths.to(dtype=torch.int)

def apply_channel_mask(tensor, mask):
    """Apply mask to tensor with channel dimension."""
    # tensor: (batch, channels, time, features)
    # mask: (batch, time, features)
    batch_size, channels, time, features = tensor.shape
    expanded_mask = mask.unsqueeze(1).expand(batch_size, channels, time, features)
    return tensor * expanded_mask


def calculate_conv_output_size(input_size: torch.Tensor, kernel_size: int, stride: int, padding: tuple[int, int]):
    """Calculate exact output size after convolution."""
    return (input_size + padding[0] + padding[1] - kernel_size) // stride + 1


class MaskedConvSequential(nn.Sequential):
    def forward(self, x, lengths):
        # Convert input (batch, time, features) to conv format
        x = x.unsqueeze(1)  # (batch, 1, time, features)
        current_lengths = lengths.clone().float()
        mask = self._create_mask(x, current_lengths.long())

        for i, layer in enumerate(self):
            x = apply_channel_mask(x, mask)

            x = layer(x)
            if hasattr(layer, 'stride') and layer.stride != (1, 1):
                if hasattr(layer, "_left_padding"):
                    padding = (layer._left_padding, layer._right_padding)  # CausalConv2D
                else:
                    padding = layer.padding
                current_lengths = calculate_conv_output_size(
                    current_lengths, layer.kernel_size[0], layer.stride[0], padding
                )
                mask = self._create_mask(x, current_lengths.long())

        x = apply_channel_mask(x, mask)
        return x, current_lengths.long()

    def _create_mask(self, tensor, lengths):
        print(tensor.shape)
        batch_size, channels, time, features = tensor.shape
        time_mask = torch.arange(time, device=tensor.device).expand(batch_size, time) < lengths.unsqueeze(1)
        return time_mask.unsqueeze(-1).expand(batch_size, time, features).to(tensor.dtype)


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


class NemoSubsample(nn.Module):
    def __init__(self,
                 subsampling,
                 subsampling_factor,
                 feat_in,
                 feat_out,
                 conv_channels,
                 subsampling_conv_chunking_factor = 1,
                 activation = nn.ReLU(),
                 is_causal = False):
        super().__init__()
        self._subsampling = subsampling
        self._conv_channels = conv_channels
        self._feat_in = feat_in
        self._feat_out = feat_out

        self._sampling_num = int(math.log(subsampling_factor, 2))
        self.subsampling_factor = subsampling_factor
        self.is_causal = is_causal

        self.subsampling_conv_chunking_factor = subsampling_conv_chunking_factor

        in_channels = 1
        layers = []

        self._stride = 2
        self._kernel_size = 3
        self._ceil_mode = False

        if self.is_causal:
            self._left_padding = self._kernel_size - 1
            self._right_padding = self._stride - 1
            self._max_cache_len = subsampling_factor + 1

        else:
            self._left_padding = (self._kernel_size - 1) // 2
            self._right_padding = (self._kernel_size -1) // 2
            self._max_cache_len = 0

        if self.is_causal:
            layers.append(
                CausalConv2D(
                    in_channels = in_channels,
                    out_channels = conv_channels,
                    kernel_size = self._kernel_size,
                    stride = self._stride,
                    padding = self._left_padding
                )
            )
        else:
            layers.append(
                torch.nn.Conv2d
                (
                    in_channels = in_channels,
                    out_channels=conv_channels,
                    kernel_size = self._kernel_size,
                    stride = self._stride,
                    padding = self._left_padding
                )
            )
        in_channels = conv_channels
        layers.append(activation)

        for i in range(self._sampling_num - 1):
            if self.is_causal:
                layers.append(
                    CausalConv2D( # Depthwise convolution
                        in_channels=in_channels,
                        out_channels = in_channels,
                        kernel_size = self._kernel_size,
                        stride = self._stride,
                        padding = None,
                        groups = in_channels
                    )
                )
            else:
                layers.append(
                    torch.nn.Conv2d( # Depthwise convolution
                        in_channels = in_channels,
                        out_channels= in_channels,
                        kernel_size = self._kernel_size,
                        stride = self._stride,
                        padding = self._left_padding,
                        groups=in_channels
                    )
                )

            layers.append(
                torch.nn.Conv2d( # Pointwise Convlution
                    in_channels = in_channels,
                    out_channels = in_channels,
                    kernel_size = 1,
                    stride = 1,
                    padding = 0,
                    groups = 1
                )
            )
            layers.append(activation)

        in_length = torch.tensor(feat_in, dtype = torch.float)
        out_length = calc_length(
                lengths=in_length,
                all_paddings=self._left_padding + self._right_padding,
                kernel_size=self._kernel_size,
                stride=self._stride,
                ceil_mode=self._ceil_mode,
                repeat_num=self._sampling_num,
            )
        self.out = nn.Linear(conv_channels * int(out_length), feat_out)
        self.conv2d_subsampling = True

        self.conv = MaskedConvSequential(*layers)

    def get_sampling_frames(self):
        return [1, self.subsampling_factor]

    def get_streaming_cache_size(self):
        return [0, self.subsampling_factor + 1]

    def conv_split_by_batch(self, x, lengths):
        b, *_ = x.size()
        if b == 1:
            return x, lengths, False
        x_ceil = 2 ** 31 / self._conv_channels * self._stride * self._stride
        p = math.ceil(math.log(torch.numel(x) / x_ceil, 2))
        cf = 2 ** p

        new_batch_size = b // cf
        if new_batch_size == 0:
            return x, lengths, False

        ans = [
            self.conv(chunk, ln)
            for chunk, ln in zip(
                torch.split(x, new_batch_size, 0),
                torch.split(lengths, new_batch_size, 0),
            )
        ]
        return torch.cat([a[0] for a in ans]), torch.cat([a[1] for a in ans]), True

    def channel_chunked_conv(self, conv, chunk_size, x):
        """Performs channel chunked convolution"""

        ind = 0
        out_chunks = []
        for chunk in torch.split(x, chunk_size, 1):
            step = chunk.size()[1]

            if self.is_causal:
                chunk = nn.functional.pad(
                    chunk, pad=(self._kernel_size - 1, self._stride - 1, self._kernel_size - 1, self._stride - 1)
                )
                ch_out = nn.functional.conv2d(
                    chunk,
                    conv.weight[ind: ind + step, :, :, :],
                    bias=conv.bias[ind: ind + step],
                    stride=self._stride,
                    padding=0,
                    groups=step,
                )
            else:
                ch_out = nn.functional.conv2d(
                    chunk,
                    conv.weight[ind: ind + step, :, :, :],
                    bias=conv.bias[ind: ind + step],
                    stride=self._stride,
                    padding=self._left_padding,
                    groups=step,
                )
            out_chunks.append(ch_out)
            ind += step

        return torch.cat(out_chunks, 1)

    def conv_split_by_channel(self, x:torch.Tensor):
        x = x.unsqueeze(0)
        x = self.conv[0](x)
        x = self.conv[1](x)

        for i in range(self._sampling_num - 1):
            _, channel, time, _ = x.size()
            p = math.ceil(math.log(torch.numel(x) / 2**31,2))
            cf = 2**p

            new_channel = int(channel // cf)
            if new_channel == 0:
                new_channel = 1

            new_time = int(time // cf)
            if new_time == 0:
                new_time = 1

            x = self.channel_chunked_conv(self.conv[i*3+2], new_channel, x)
            x = torch.cat([self.conv[i * 3 + 3](chunk) for chunk in torch.split(x, new_time, 2)], 2)
            x = self.conv[i*3 + 4](x)
        return x



    def forward(self, x, lengths):
        out_lengths = calc_length(
            lengths,
            all_paddings=self._left_padding + self._right_padding,
            kernel_size=self._kernel_size,
            stride=self._stride,
            ceil_mode=self._ceil_mode,
            repeat_num=self._sampling_num,
        )

        if not self.conv2d_subsampling:
            x = x.transpose(1, 2)

        if self.subsampling_conv_chunking_factor != -1 and self.conv2d_subsampling:
            if self.subsampling_conv_chunking_factor == 1:
                x_ceil = 2 ** 31 / self._conv_channels * self._stride * self._stride
                if torch.numel(x) > x_ceil:
                    need_to_split = True
                else:
                    need_to_split = False
            else:
                need_to_split = True

            if need_to_split:
                x, lengths, success = self.conv_split_by_batch(x, lengths)
                if not success:
                    x = self.conv_split_by_channel(x)
                    lengths = out_lengths
            else:

                x, lengths = self.conv(x, lengths)
        else:
            x, lengths = self.conv(x)

        if self.conv2d_subsampling:
            b, c, t, f = x.size()
            x = self.out(x.transpose(1, 2).reshape(b, t, -1))
        else:
            x = x.transpose(1, 2)

        return x, lengths.to(dtype = torch.int64)


if __name__ == "__main__":
    # Example usage
    x = torch.randn(1, 128, 3000)
    hidden_size = 512
    subsample_layer = Subsample(subsample_factor=4, mel_bins=128, hidden_size=hidden_size)
    output = subsample_layer(x)
    print(f"Subsample output shape: {output.shape}")
    x_2d = torch.randn(2, 3000, 80)
    lengths = torch.tensor([3000, 3000])
    
    nemo_subsample_layer = NemoSubsample(
        subsampling="conv2d",
        subsampling_factor=8,
        feat_in=80, 
        feat_out=hidden_size,
        conv_channels=128
    )
    output, out_lengths = nemo_subsample_layer(x_2d, lengths)
    print(f"NemoSubsample output shape: {output.shape}")
    print(f"Input lengths: {lengths}")
    print(f"Output lengths: {out_lengths}")
