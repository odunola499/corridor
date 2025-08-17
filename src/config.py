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
    mel_bins:int = 128

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




