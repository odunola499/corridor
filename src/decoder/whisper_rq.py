import torch
from src.core import VectorTrainEngine
from src.encoder.whisper_encoder import WhisperConfig, WhisperEncoder
from src.config import RVQTrainConfig

if __name__ == "__main__":
    Q = 4
    codebook_size = 500
    codebook_dim = 384
    config = WhisperConfig()
    encoder = WhisperEncoder(config)
    train_config = RVQTrainConfig(encoder_config=config,
                                  vq_layers=1)

    engine = VectorTrainEngine(train_config = train_config,
                                    encoder=encoder)
    features = torch.randn(2, encoder.config.mel_bins, 3000)
    loss = engine(features)
    print(loss)
