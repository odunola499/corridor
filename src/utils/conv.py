import torch
from torch.nn import functional as F
from torch import nn
from typing import Optional


class CausalConv2D(nn.Conv2d):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size:int,
                 stride:int = 1,
                 padding:Optional[str|int] = 0,
                 dilation:int = 1,
                 groups:int = 1,
                 bias:bool = True,
                 padding_mode:str = 'zeros',
                 device = None,
                 dtype = None):
        assert padding is None, "Padding must be None for causalconv2d"
        self._left_padding = kernel_size - 1
        self._right_padding = stride - 1
        padding = 0
        super(CausalConv2D, self).__init__(
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            dilation,
            groups,
            bias,
            padding_mode,
            device,
            dtype
        )

    def forward(self, x):
        x = F.pad(self._left_padding, self._right_padding, self._left_padding, self._right_padding)
        x = super().forward(x)
        return x

class CausalConv1D(nn.Conv1d):
    def __init__(self,
                 in_channels:int,
                 out_channels:int,
                 kernel_size:int,
                 stride:int = 1,
                 padding:str|int = 0,
                 dilation:int = 1,
                 groups:int = 1,
                 bias:bool = True,
                 padding_mode:str = 'zeros',
                 device = None,
                 dtype = None):
        self.cache_drop_size = None
        if padding is None:
            self._left_padding = kernel_size - 1
            self._right_padding = stride - 1
        else:
            if stride != 1 and padding != kernel_size - 1:
                raise ValueError("No striding allowed for non-symmetric convolutions!")
            if isinstance(padding, int):
                self._left_padding = padding
                self._right_padding = padding
            elif isinstance(padding, list) and len(padding) == 2 and padding[0] + padding[1] == kernel_size - 1:
                self._left_padding = padding[0]
                self._right_padding = padding[1]
            else:
                raise ValueError(f"Invalid padding param: {padding}!")

        self._max_cache_len = self._left_padding

        super(CausalConv1D, self).__init__(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=0,
            dilation=dilation,
            groups=groups,
            bias=bias,
            padding_mode=padding_mode,
            device=device,
            dtype=dtype,
        )

    def update_cache(self, x, cache=None):
        if cache is None:
            new_x = F.pad(x, pad=(self._left_padding, self._right_padding))
            next_cache = cache
        else:
            new_x = F.pad(x, pad=(0, self._right_padding))
            new_x = torch.cat([cache, new_x], dim=-1)
            if self.cache_drop_size > 0:
                next_cache = new_x[:, :, : -self.cache_drop_size]
            else:
                next_cache = new_x
            next_cache = next_cache[:, :, -cache.size(-1):]
        return new_x, next_cache

    def forward(self, x, cache=None):
        x, cache = self.update_cache(x, cache=cache)
        x = super().forward(x)
        if cache is None:
            return x
        else:
            return x, cache