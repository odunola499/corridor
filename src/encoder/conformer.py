from torch import nn
from src.module.layers.attention import MHAModule
from src.module.layers.ffn_module import FeedForwardModule
from src.module.layers.convolution_module import ConvolutionModule
from src.module.layers.subsample import Subsample
from src.module.layers.specaugment import SpecAugment


from src.config import ConformerConfig
# Whatever attention choice is to be used for both training and inference (eager *args)

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
        )
        self.ffn2 = FeedForwardModule(config.hidden_size)

    def forward(self, x):
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
        self.subsample = Subsample(hidden_size = config.hidden_size, dropout = config.dropout_rate,
                                   mel_bins = config.mel_bins, subsample_factor = config.subsample_factor)

        self.conformer_blocks = nn.ModuleList([
            ConformerBlock(config) for _ in range(config.num_layers)
        ])

    def forward(self, x):
        x = self.spec_augment(x)
        x = self.subsample(x)
        for block in self.conformer_blocks:
            x = block(x)
        return x
