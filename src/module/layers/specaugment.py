import torchaudio.transforms
from torch import nn
import torch

class SpecAugment(nn.Module):
    def __init__(self, freq_mask_params = 27,
                 time_mask_params = 10,
                 num_freq_masks = 2,
                 num_time_masks = 10):
        super().__init__()
        self.freq_masks = nn.ModuleList([
            torchaudio.transforms.FrequencyMasking(freq_mask_params) for i in range(num_freq_masks)
        ])
        self.time_masks = nn.ModuleList([
            torchaudio.transforms.TimeMasking(time_mask_params) for i in range(num_time_masks)
        ])

    def forward(self, x):
        if self.training:
            with torch.autocast(x.device.type, enabled = False):
                for mask in self.freq_masks:
                    x = mask(x)

                for mask in self.time_masks:
                    x = mask(x)

        return x