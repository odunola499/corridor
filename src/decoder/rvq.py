import torch
from torch import nn
from einops import rearrange
from torch.nn import functional as F

def mask_features(x, p_mask = 0.03, n_span = 10, sigma = 0.1):
    batch_size, input_dim, seq_len = x.shape

    n_masked = int(seq_len * p_mask)
    mask = torch.ones(batch_size, seq_len, device=x.device, dtype = torch.bool)

    masked_indices_list = []

    for b in range(batch_size):
        start_positions = torch.randint(0, seq_len, (n_masked,), device=x.device)

        masked_indices = []
        for start in start_positions:
            end = min(start + n_span, seq_len)
            span_indices = torch.arange(start, end, device=x.device)
            masked_indices.append(span_indices)

            mask[b, start:end] = False
        if masked_indices:
            all_masked = torch.cat(masked_indices)
            unique_masked = torch.unique(all_masked)
            masked_indices_list.append(unique_masked)
        else:
            masked_indices_list.append(torch.tensor([], dtype=torch.long, device=x.device))

    x_masked = x.clone()

    for b, masked_idx in enumerate(masked_indices_list):
        if len(masked_idx) > 0:
            noise = torch.randn(input_dim, len(masked_idx), device=x.device) * sigma
            x_masked[b, :, masked_idx] = noise

    return x_masked, mask, masked_indices_list

class VectorQuantize(nn.Module):
    def __init__(self, input_dim:int, codebook_size:int, codebook_dim:int,stride:int = 1):
        super().__init__()
        self.codebook_size = codebook_size
        self.codebook_dim = codebook_dim
        self.stride = stride

        self.in_proj = nn.Sequential(
            nn.Conv1d(
                input_dim, codebook_dim, kernel_size = 3, stride = 2, bias = False, padding = 1
            ),
            nn.Conv1d(
                codebook_dim, codebook_dim, kernel_size=3, stride=2, bias=False, padding=1
            )
        )

        self.codebook = nn.Embedding(codebook_size, codebook_dim)

    def forward(self, x:torch.Tensor):
        x_e = self.in_proj(x)
        encodings = rearrange(x_e, "b d t -> (b t) d")
        codebook = self.codebook.weight

        encodings = F.normalize(encodings)
        codebook = F.normalize(codebook)

        dist = (
                encodings.pow(2).sum(1, keepdim=True)
                - 2 * encodings @ codebook.t()
                + codebook.pow(2).sum(1, keepdim=True).t()
        )
        indices = rearrange((-dist).max(1)[1], "(b t) -> b t", b=x_e.size(0))
        return indices



class LossEngine(nn.Module):
    def __init__(self, Q, input_dim, codebook_size, codebook_dim, hidden_size,):
        super().__init__()
        self.Q = Q
        self.input_dim = input_dim #also mel dim
        self.codebook_size = codebook_size
        self.codebook_dim = codebook_dim


        self.quantizers = nn.ModuleList([
            VectorQuantize(input_dim, codebook_size, codebook_dim) for _ in range(Q)
        ])
        self.classifiers = nn.ModuleList([
            nn.Linear(hidden_size, codebook_size) for _ in range(Q)
        ])

        self.quantizers.requires_grad_(False)
        self.classifiers.requires_grad_(False)

    def forward(self, features,hidden_states):
        # features: (batch_size, input_dim, seq_len)
        # hidden_states: (batch_size, seq_len, hidden_size)

        loss = 0.0
        labels = []
        logits = []
        for vq in self.quantizers:
            indices = vq(features)
            labels.append(indices)

        for classifier in self.classifiers:
            logit = classifier(hidden_states)
            logits.append(logit)

        for i in range(self.Q):
            label = labels[i]
            logit = logits[i]
            loss += F.cross_entropy(logit, label, ignore_index=-1)
        loss /= self.Q

        return loss




if __name__ == "__main__":
    input_dim = 128
    codebook_size = 512
    codebook_dim = 384
    hidden_size = 768
    batch_size = 2
    sequence_length = 3000

    x = torch.randn(batch_size, input_dim, sequence_length)
    vq = VectorQuantize(input_dim, codebook_size, codebook_dim)
    indices = vq(x)
    print(f"VQ indices shape: {indices.shape}")

    loss_engine = LossEngine(Q=8, input_dim=input_dim, codebook_size=codebook_size,
                             codebook_dim=codebook_dim, hidden_size=hidden_size)

    x_masked, mask, masked_indices = mask_features(x)
    print(f"Masked input shape: {x_masked.shape}")
    print(f"Mask shape: {mask.shape}")
    print(f"Number of masked frames in batch 0: {len(masked_indices[0])}")
    print(f"Number of masked frames in batch 1: {len(masked_indices[1])}")
    print(f"Percentage masked in batch 0: {len(masked_indices[0]) / sequence_length * 100:.2f}%")
    print(f"Expected upper bound: {0.01 * 10 * 100:.1f}%")