from typing import Optional
from dataclasses import dataclass

@dataclass
class AudioConfig:
    pass

@dataclass
class TrainConfig:
    pass

@dataclass
class DataConfig:
    pass

@dataclass
class FeatureExtractorConfig:
    pass

class NemoFeatureExtractorConfig(FeatureExtractorConfig):
    sample_rate:int = 16000
    normalize:bool = True
    window_size:float = 0.025
    window_stride:float = 0.01
    window:str = 'hann'
    features:int = 80 #mel bins
    n_fft:int = 512
    frame_splicing:int = 1
    dither:float = 0.00001
    pad_to:int = 0

@dataclass
class ConformerConfig(AudioConfig):
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
    mel_bins:int = 80

@dataclass
class WhisperConfig(AudioConfig):
    mel_bins: int = 80
    n_audio_positions:int = 1500
    hidden_size:int = 384
    num_audio_heads:int = 6
    num_audio_layers:int = 4
    vocab_size:int = 51865
    n_text_positions:int = 448
    num_text_heads:int = 6
    num_text_layers:int = 4
    size:Optional[str] = 'tiny'
    subsample_factor:int = 2


@dataclass
class RVQTrainConfig(TrainConfig):
    encoder_config: AudioConfig
    Q:int = 2
    vq_layers:int = 2
    codebook_size:int = 100
    codebook_dim:int = 256


class NemoConformerConfig(AudioConfig):
    feat_in:int = 80
    feat_out:int = -1
    n_layers:int = 17
    d_model:int = 512

    subsampling_factor:int = 8
    subsampling_conv_channels:int = 256
    causal_downsampling:bool = False

    reduction_factor:int = 1

    ff_expansion_factor:int = 4
    self_attention_model:str = 'rel_pos'
    n_heads:int = 8
    att_context_size:list = [-1,-1]
    att_context_style:str = 'regular'
    xscaling:bool = True # attn scaling by sqt of d_model
    untie_biases:bool = True
    pos_emb_max_len:int = 5000

    conv_kernel_size:int = 9
    conv_norm_type:str = "batch_norm"

    dropout_float = 0.1
    dropout_pre_encoder:float = 0.
    dropout_emb:int = 0.0
    dropout_att:float = 0.1




