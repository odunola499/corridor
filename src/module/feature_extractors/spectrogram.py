import torch
from torch import nn, Tensor
from dataclasses import dataclass
import librosa
import random
from src.config import NemoConfig


def normalize_batch(x, seq_len):
    x_mean = None
    x_std = None
    batch_size = x.shape[0]
    max_time = x.shape[-1]

    time_steps = torch.arange(max_time, device = x.device).unsqueeze(0).expand(batch_size, max_time)
    valid_mask = time_steps < seq_len.unsqueeze(1)
    x_mean_numerator = torch.where(valid_mask.unsqueeze(1), x, 0.0).sum(axis = -1)
    x_mean_denominator = valid_mask.sum(axis = 1)
    x_mean = x_mean_numerator / x_mean_denominator.unsqueeze(1)

    x_std = torch.sqrt(
        torch.sum(torch.where(valid_mask.unsqueeze(1), x - x_mean.unsqueeze(2), 0.0) ** 2, axis=2)
        / (x_mean_denominator.unsqueeze(1) - 1.0)
    )
    x_std = x_std.masked_fill(x_std.isnan(), 0.0)
    # make sure x_std is not zero
    x_std += 1e-5
    return (x - x_mean.unsqueeze(2)) / x_std.unsqueeze(2), x_mean, x_std


class FilterbankFeatures(nn.Module):
    def __init__(self, config:NemoConfig):
        super().__init__()
        self.log_zero_guard_type = "add"
        self.log_zero_guard_value = 2**-24
        self.sample_rate = config.sample_rate
        self.win_length = int(config.window_size * self.sample_rate)
        self.hop_length = int(config.window_stride * self.sample_rate)
        self.n_fft = config.n_fft
        self.stft_pad_amount = (self.n_fft - self.hop_length) // 2
        self.exact_pad = False

        assert config.window == 'hann'
        window_tensor = torch.hann_window(self.win_length, periodic = False)
        self.register_buffer("window", window_tensor)

        self.normalize = config.normalize #depreiate
        self.log = True
        self.dither = config.dither
        self.frame_splicing = config.frame_splicing
        self.nfilt = config.features #mel bins
        self.preemph = 0.97
        self.pad_to = config.pad_to
        highfreq = self.sample_rate / 2

        filterbanks = torch.tensor(
            librosa.filters.mel(
                sr = self.sample_rate, n_fft = self.n_fft, fmin = 0, fmax = highfreq, norm = 'slaney'
            ), dtype = torch.float
        ).unsqueeze(0)
        self.register_buffer("fb", filterbanks)

        max_duration = 16.7
        max_length = self.get_seq_len(torch.tensor(max_duration * self.sample_rate, dtype=torch.float))
        max_pad = 0
        self.max_length = max_length
        self.pad_value = 0
        self.mag_power = 2.0

        self._rng = random.Random()
        self.nb_augmentation_prob = 0

        self.config = config

    def get_seq_len(self, seq_len):
        pad_amount = self.stft_pad_amount * 2
        seq_len = torch.floor_divide((seq_len + pad_amount - self.n_fft), self.hop_length)
        return seq_len.to(dtype = torch.long)

    def stft(self, x):
        return torch.stft(
            x,
            n_fft = self.n_fft,
            hop_length = self.hop_length,
            win_length = self.win_length,
            center = True,
            window = self.window.to(dtype = torch.float, device = x.device),
            return_complex= True,
            pad_mode = "constant"
        )

    @property
    def filter_banks(self):
        return self.fb

    def forward(self, x:Tensor, seq_len:Tensor, linear_spec = False):
        seq_len_time = seq_len
        seq_len_unfixed = self.get_seq_len(seq_len)
        seq_len = torch.where(seq_len == 0, torch.zeros_like(seq_len_unfixed), seq_len_unfixed)

        if self.stft_pad_amount is not None:
            x = torch.nn.functional.pad(
                x.unsqueeze(1), (self.stft_pad_amount, self.stft_pad_amount), "constant"
            ).squeeze(1)

        if self.training:
            x += self.config.dither * torch.randn_like(x)

        if self.preemph is not None:
            timemask = torch.arange(x.shape[1], device=x.device).unsqueeze(0) < seq_len_time.unsqueeze(1)
            x = torch.cat((x[:, 0].unsqueeze(1), x[:, 1:] - self.preemph * x[:, :-1]), dim=1)
            x = x.masked_fill(~timemask, 0.0)

        with torch.amp.autocast(x.device.type, enabled=False):
            x = self.stft(x)

        guard = 0
        x = torch.view_as_real(x)
        x = torch.sqrt(x.pow(2).sum(-1) + guard)

        if self.training and self.nb_augmentation_prob > 0.0:
            for idx in range(x.shape[0]):
                if self._rng.random() < self.nb_augmentation_prob:
                    x[idx, self._nb_max_fft_bin:, :] = 0.0

        if self.mag_power != 1.0:
            x = x.pow(self.mag_power)

        if linear_spec:
            return x, seq_len

        with torch.amp.autocast(x.device.type, enabled=False):
            x = torch.matmul(self.fb.to(x.dtype), x)

        if self.log:
            x = torch.log(x + self.log_zero_guard_value)


        if self.normalize:
            x, _, _ = normalize_batch(x, seq_len)

        max_len = x.size(-1)
        mask = torch.arange(max_len, device=x.device)
        mask = mask.repeat(x.size(0), 1) >= seq_len.unsqueeze(1)
        x = x.masked_fill(mask.unsqueeze(1).type(torch.bool).to(device=x.device), self.pad_value)

        return x, seq_len


class AudioToMelSpectrogramPreprocessor(nn.Module):
    def __init__(self, config:NemoConfig):
        super().__init__()
        self.sample_rate = config.sample_rate
        self.n_window_size = int(config.window_size * self.sample_rate)
        self.n_window_stride = int(config.window_stride * self.sample_rate)

    @torch.no_grad()
    def forward(self,input_signal:Tensor, length):
        assert input_signal.dtype == torch.float32

    def save_to(self, save_path:str):
        pass

    @classmethod
    def restore_from(cls, restore_path:str):
        pass

if __name__ == "__main__":
    batch_size = 2

    tensor = torch.randn(batch_size, 32000)
    seq_len = torch.tensor([32000] * batch_size, dtype = torch.long)

    feature_extractor = FilterbankFeatures(NemoConfig())
    output, seq_len = feature_extractor(tensor, seq_len)
    print(output.shape)
    print(seq_len.shape)