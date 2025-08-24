import math
import torch
from torch import nn
from dataclasses import dataclass
from src.module.layers.subsample import NemoSubsample
from src.module.layers.specaugment import SpecAugment
from src.config import NemoConformerConfig
from src.module.attention.relative_position import RelPositionMultiHeadAttention, NemoRelPositionalEncoding
from src.module.layers.ffn_module import NemoFeedForwardModule
from src.module.layers.convolution_module import NemoConvolutionModule
from src.utils import compute_stochastic_depth_drop_probs
from tests.nemo_ssl import AudioFeatureExtractor

class ConformerLayer(nn.Module):
    def __init__(self,
                 d_model,
                 d_ff,
                 self_attention_model = 'rel_pos',
                 n_heads = 4,
                 conv_kernel_size = 31,
                 dropout = 0.1,
                 dropout_pre_encoder = 0.1,
                 dropout_emb = 0.0,
                 dropout_att = 0.0,
                 pos_bias_u = None,
                 pos_bias_v = None,
                 attn_context_size = [-1, -1],
                 use_bias = True,
                 ):
        super().__init__()
        use_pytorch_sdpa_backends = None
        self.self_attention_model = self_attention_model
        self.n_heads = n_heads
        self.fc_factor = 0.5

        self.norm_feed_forward1 = nn.LayerNorm(d_model)
        self.feed_forward1 = NemoFeedForwardModule(d_model = d_model, d_ff = d_ff, dropout = dropout)

        self.norm_conv = nn.LayerNorm(d_model)
        self.conv = NemoConvolutionModule(
            d_model = d_model,
            kernel_size = conv_kernel_size,
        )
        self.norm_self_att = nn.LayerNorm(d_model)
        self.self_attn = RelPositionMultiHeadAttention(
            n_head = n_heads,
            n_feat = d_model,
            dropout_rate=dropout_att,
            pos_bias_u=pos_bias_u,
            pos_bias_v = pos_bias_v,
            max_cache_len=-1,
            use_bias=use_bias,
            use_pytorch_sdpa=False,
        )

        self.norm_feed_forward2 = nn.LayerNorm(d_model)
        self.feed_forward2 = NemoFeedForwardModule(d_model=d_model, d_ff = d_ff, dropout = dropout)

        self.dropout = nn.Dropout(dropout)
        self.norm_out = nn.LayerNorm(d_model)


    def forward(self, x, attn_mask = None, pos_emb = None, pad_mask = None, cache_last_channel = None, cache_last_time = None):
        residual = x
        x = self.norm_feed_forward1(x)
        x = self.feed_forward1(x)
        residual = residual + self.dropout(x) * self.fc_factor

        x = self.norm_self_att(residual)
        x = self.self_attn(query = x, key = x, value = x, mask = attn_mask, pos_emb = pos_emb, cache = cache_last_channel)
        if x is not None and cache_last_channel is not None:
            (x, cache_last_channel) = x

        residual = residual + self.dropout(x)
        x = self.norm_conv(residual)
        x = self.conv(x, pad_mask=pad_mask, cache=cache_last_time)
        if cache_last_time is not None:
            (x, cache_last_time) = x
        residual = residual + self.dropout(x)

        x = self.norm_feed_forward2(residual)
        x = self.feed_forward2(x)
        residual = residual + self.dropout(x) * self.fc_factor

        x = self.norm_out(residual)
        if cache_last_channel is None:
            return x
        else:
            return x, cache_last_channel, cache_last_time


#solve the config thing, ModelConfig
class NemoEncoder(nn.Module):
    def __init__(self, config:NemoConformerConfig):
        super().__init__()

        d_ff = config.d_model * config.ff_expansion_factor
        self.d_model = config.d_model
        self.n_layers = config.n_layers
        self._feat_in = config.feat_in
        self.att_context_style = config.att_context_style
        self.subsampling_factor = config.subsampling_factor
        self.subsampling_conv_chunking_factor = 1


        att_context_probs = None
        conv_context_size = None
        conv_kernel_size = config.conv_kernel_size

        att_context_size_all = [[-1, -1]]
        self.att_context_size = config.att_context_size

        self.self_attention_model = config.self_attention_model
        self.global_tokens = 0
        self.global_tokens_spacing = 1
        self.global_attn_seperate = False
        self.use_pytorch_sdpa = True
        self.sync_max_audio_length = 30

        self.xscale = math.sqrt(self.d_model)
        self.pre_encode = NemoSubsample(
            subsampling='',
            subsampling_factor=config.subsampling_factor,
            feat_in = config.feat_in,
            feat_out = config.d_model,
            conv_channels=config.subsampling_conv_channels,
            subsampling_conv_chunking_factor=1,
            activation=nn.ReLU(True),
            is_causal = False
        )

        self._feat_out = config.d_model
        d_head = config.d_model // config.n_heads
        pos_bias_u = nn.Parameter(torch.Tensor(config.n_heads, d_head))
        pos_bias_v = nn.Parameter(torch.Tensor(config.n_heads, d_head))
        nn.init.zeros_(pos_bias_u)
        nn.init.zeros_(pos_bias_v)

        self.pos_enc = NemoRelPositionalEncoding(
            d_model = config.d_model,
            dropout_rate = config.dropout_pre_encoder,
            max_len = config.pos_emb_max_len,
            xscale = self.xscale,
            dropout_rate_emb=config.dropout_emb
        )

        self.layers = nn.ModuleList()
        for i in range(config.n_layers):
            layer = ConformerLayer(d_model = config.d_model,
                                   d_ff = d_ff,
                                   self_attention_model= 'rel_pos',
                                   n_heads=config.n_heads,
                                   dropout=0.1,
                                   pos_bias_u=pos_bias_u,
                                   pos_bias_v=pos_bias_v,
                                   use_bias=True,
                                  )
            self.layers.append(layer)

        stochastic_depth_drop_probs = 0.0
        stochastic_depth_mode = "linear"
        stochastic_depth_start_layer = 1

        self.layer_drop_probs = compute_stochastic_depth_drop_probs(
            len(self.layers), stochastic_depth_drop_probs, stochastic_depth_mode, stochastic_depth_start_layer
        )

        self.out_proj = None
        self._feat_out = config.d_model
        self.set_max_audio_length(config.pos_emb_max_len)
        self.use_pad_mask = True

        self.config = config

    def set_max_audio_length(self, position):
        pass

    def _create_masks(self, att_context_size, padding_length, max_audio_length, offset, device):
        att_mask = torch.ones(1, max_audio_length, max_audio_length, dtype = torch.bool, device = device)
        att_mask = None
        pad_mask = torch.arange(0, max_audio_length, device = device).expand(
            padding_length.size(0), -1,
        ) < padding_length.unsqueeze(-1)
        if offset is not None:
            pad_mask_off = torch.arange(0, max_audio_length, device = device).expand(
                padding_length.size(0), -1
            ) >= offset.unsqueeze(-1)
            pad_mask = pad_mask_off.logical_and(pad_mask)

        pad_mask = ~pad_mask
        return pad_mask, att_mask

    def forward(self,
                audio_signal:torch.Tensor,
                length):
        #audio_sigmal : batch_size, self._feat_in, n_frames
        #
        if length is None:
            length = audio_signal.new_full(
                (audio_signal.size(0),), audio_signal.size(-1), dtype=torch.int64, device=audio_signal.device
            )

        cur_att_context_size = self.config.att_context_size
        audio_signal  = audio_signal.transpose(1,2)
        print('input', audio_signal.shape)
        audio_signal, length = self.pre_encode(audio_signal, lengths = length)

        max_audio_length = audio_signal.size(1)
        padding_length = length
        cache_last_channel_next = None
        cache_len = 0
        offset = None

        audio_signal, pos_emb = self.pos_enc(x = audio_signal, cache_len = cache_len)

        pad_mask, att_mask = self._create_masks(
            att_context_size = None,
            padding_length = padding_length,
            max_audio_length = max_audio_length,
            offset = offset,
            device = audio_signal.device
        )
        for lth, (drop_prob, layer) in enumerate(zip(self.layer_drop_probs, self.layers)):
            original_signal = audio_signal
            cache_last_channel_cur = None
            cache_last_time_cur = None
            audio_signal = layer(
                x=audio_signal,
                att_mask=att_mask,
                pos_emb=pos_emb,
                pad_mask=pad_mask,
                cache_last_channel=cache_last_channel_cur,
                cache_last_time=cache_last_time_cur,
            )
            if self.training and drop_prob > 0.0:
                should_drop = torch.rand(1) < drop_prob
                # adjusting to match expectation
                if should_drop:
                    # that's not efficient, but it's hard to implement distributed
                    # version of dropping layers without deadlock or random seed meddling
                    # so multiplying the signal by 0 to ensure all weights get gradients
                    audio_signal = audio_signal * 0.0 + original_signal
                else:
                    # not doing this operation if drop prob is 0 as it's identity in that case
                    audio_signal = (audio_signal - original_signal) / (1.0 - drop_prob) + original_signal
            if self.out_proj is not None:
                audio_signal = self.out_proj(audio_signal)

        audio_signal = audio_signal.transpose(1,2)
        length = length.to(dtype = torch.int64)
        return audio_signal, length


if __name__ == "__main__":
    config = NemoConformerConfig()
    my_encoder = NemoEncoder(config)
    params = [p.numel() for p in my_encoder.parameters()]
    print(f"My encoder parameters: {sum(params)}")

    tensor = torch.randn(2, 80,3000)
    length = torch.tensor([3000, 3000], dtype = torch.long)

    output = my_encoder(tensor, length)
    print(output)