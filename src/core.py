import torch
from torch import nn
from torch.nn import functional as F
from einops import rearrange


def mask_features(x, p_mask=0.03, n_span=10, sigma=0.1, subsample_factor=4):
    batch_size, input_dim, seq_len = x.shape
    
    n_masked = int(seq_len * p_mask)
    mask = torch.ones(batch_size, seq_len, device=x.device, dtype=torch.bool)
    
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
    
    mask_downsampled = mask.view(batch_size, seq_len // subsample_factor, subsample_factor)
    mask_downsampled = mask_downsampled.any(dim=-1)
    
    orig_mask = mask.clone()
    mask = mask_downsampled
    return x_masked, mask, orig_mask, masked_indices_list


class VectorQuantize(nn.Module):
    def __init__(self, input_dim: int, codebook_size: int, codebook_dim: int, stride: int = 1, num_layers: int = 1):
        super().__init__()
        self.codebook_size = codebook_size
        self.codebook_dim = codebook_dim
        self.stride = stride
        
        layers = []
        in_dim = input_dim
        for i in range(num_layers):
            layers.append(nn.Conv1d(
                in_dim, codebook_dim, kernel_size=3, stride=2, bias=False, padding=1
            ))
            in_dim = codebook_dim
        
        self.in_proj = nn.Sequential(*layers)
        self.codebook = nn.Embedding(codebook_size, codebook_dim)
        self.in_proj.requires_grad_(False)
        self.codebook.requires_grad_(False)
    
    @torch.no_grad()
    def forward(self, x: torch.Tensor):
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



class VectorTrainEngine(nn.Module):
    def __init__(self, Q, codebook_size, codebook_dim, hidden_size, encoder, vq_layers=1, subsample_factor=2):
        super().__init__()
        self.Q = Q
        self.input_dim = encoder.config.mel_bins
        self.codebook_size = codebook_size
        self.codebook_dim = codebook_dim
        self.subsample_factor = subsample_factor
        
        self.quantizers = nn.ModuleList([
            VectorQuantize(self.input_dim, codebook_size, codebook_dim, num_layers=vq_layers) for _ in range(Q)
        ])
        self.classifiers = nn.ModuleList([
            nn.Linear(hidden_size, codebook_size) for _ in range(Q)
        ])
        
        self.quantizers.requires_grad_(False)
        self.classifiers.requires_grad_(True)
        
        self.encoder = encoder
    
    def forward(self, features):
        masked_features, mask, orig_mask, _ = mask_features(features, subsample_factor=self.encoder.config.subsample_factor)
        assert masked_features.shape == features.shape, "Masked features shape mismatch"
        
        hidden_states = self.encoder(masked_features)
        
        loss = 0.0
        labels, logits = [], []
        
        mask = mask.view(-1)
        
        with torch.no_grad():
            for vq in self.quantizers:
                indices = vq(features)
                labels.append(indices)
        
        for classifier in self.classifiers:
            logit = classifier(hidden_states)
            logits.append(logit)
        
        for i in range(self.Q):
            label = labels[i].view(-1)[mask]
            logit = logits[i].view(-1, self.codebook_size)[mask]
            loss += F.cross_entropy(logit, label, ignore_index=-1)
        loss /= self.Q
        
        return loss