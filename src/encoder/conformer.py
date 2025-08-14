import torch
from torch import nn
from src.encoder.attention import MHAModule
from src.encoder.ffn_module import FeedForwardModule
from src.encoder.convolution_module import ConvolutionModule
from src.encoder.subsample import Subsample
from src.encoder.specaugment import SpecAugment

from dataclasses import dataclass

# Whatever attention choice is to be used for both training and inference (eager *args)

@dataclass
class ConformerConfig:
    hidden_size: int
    num_heads: int
    num_kv_heads: int
    expansion_factor: int = 4
    kernel_size: int = 31
    subsample: bool = False
    subsample_factor: int = 4
    dropout_rate: float = 0.1
    use_specaugment: bool = False
    specaugment_params: dict = None
    eager_attn:bool = False
    num_layers: int = 32
    mel_bins:int = 128

class ConformerBlock(nn.Module):
    def __init__(self, config:ConformerConfig):
        super().__init__()
        self.config = config

        self.ffn1 = FeedForwardModule(config.hidden_size)
        self.attn = MHAModule(
            num_kv_heads=config.num_kv_heads,
            num_heads= config.num_heads,
            hidden_size=config.hidden_size,
            drop = config.dropout_rate,
            eager = config.eager_attn
        )
        self.conv = ConvolutionModule(
            hidden_size=config.hidden_size,
            kernel_size=config.kernel_size,
            expansion=config.expansion_factor
        )
        self.ffn2 = FeedForwardModule(config.hidden_size)

    def forward(self, x):
        # No residual between blocks?
        x = self.ffn1(x)
        x = self.attn(x)
        x = self.conv(x)
        x = self.ffn2(x)
        return x

class Conformer(nn.Module):
    def __init__(self,config:ConformerConfig):
        super().__init__()
        self.config = config
        self.spec_augment = SpecAugment()
        self.subsample = Subsample(hidden_size = config.hidden_size, dropout = config.dropout_rate)

        self.conformer_blocks = nn.ModuleList([
            ConformerBlock(config) for _ in range(config.num_layers)
        ])

    def forward(self, x):
        x = self.spec_augment(x)
        x = self.subsample(x)
        for block in self.conformer_blocks:
            x = block(x)
        return x
