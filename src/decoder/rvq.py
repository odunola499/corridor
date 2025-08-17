import torch
from src.core import VectorTrainEngine, VectorQuantize, mask_features
from src.encoder.conformer import ConformerConfig, Conformer


if __name__ == "__main__":
    input_dim = 128
    codebook_size = 100
    codebook_dim = 256
    hidden_size = 256
    batch_size = 4
    sequence_length = 3000
    config = ConformerConfig(hidden_size=hidden_size, num_heads=8, num_kv_heads=8, num_layers=2, mel_bins=input_dim)
    encoder = Conformer(config)

    x = torch.randn(batch_size, input_dim, sequence_length)
    vq = VectorQuantize(input_dim, codebook_size, codebook_dim, num_layers=2)
    indices = vq(x)
    print(f"VQ indices shape: {indices.shape}")

    loss_engine = VectorTrainEngine(Q=2, codebook_size=codebook_size,
                             codebook_dim=codebook_dim, hidden_size=hidden_size, encoder=encoder, vq_layers=2)

    x_masked, mask, orig_mask, masked_indices = mask_features(x)
    print(f"Masked input shape: {x_masked.shape}")
    print(f"Mask shape: {mask.shape}")

    loss = loss_engine(x)
    print(loss)