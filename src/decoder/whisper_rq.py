import torch
from src.core import VectorTrainEngine
from src.encoder.whisper_encoder import WhisperConfig, AudioEncoder

if __name__ == "__main__":
    Q = 4
    codebook_size = 500
    codebook_dim = 384
    config = WhisperConfig()
    encoder = AudioEncoder(config)

    engine = VectorTrainEngine(Q, codebook_size, codebook_dim, config.hidden_size,
                         encoder, vq_layers=1, subsample_factor=2)
    features = torch.randn(2, encoder.config.mel_bins, 3000)
    loss = engine(features)
    print(loss)
