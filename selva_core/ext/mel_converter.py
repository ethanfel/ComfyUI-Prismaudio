# Reference: # https://github.com/bytedance/Make-An-Audio-2
from typing import Literal

import numpy as np
import torch
import torch.nn as nn


def librosa_mel_fn(*, sr, n_fft, n_mels=128, fmin=0.0, fmax=None):
    """Pure-numpy mel filterbank equivalent to librosa.filters.mel (HTK scale, no norm).

    Replaces the librosa import to avoid the librosa → numba → NumPy-version
    incompatibility that exists in some ComfyUI environments.
    """
    if fmax is None:
        fmax = sr / 2.0

    def hz_to_mel(f):
        return 2595.0 * np.log10(1.0 + np.asarray(f) / 700.0)

    def mel_to_hz(m):
        return 700.0 * (10.0 ** (np.asarray(m) / 2595.0) - 1.0)

    n_freqs = n_fft // 2 + 1
    fft_freqs = np.linspace(0.0, sr / 2.0, n_freqs)

    mel_min = hz_to_mel(fmin)
    mel_max = hz_to_mel(fmax)
    mel_points = np.linspace(mel_min, mel_max, n_mels + 2)
    hz_points = mel_to_hz(mel_points)

    weights = np.zeros((n_mels, n_freqs), dtype=np.float32)
    for m in range(1, n_mels + 1):
        f_lo, f_mid, f_hi = hz_points[m - 1], hz_points[m], hz_points[m + 1]
        up   = (fft_freqs - f_lo) / (f_mid - f_lo + 1e-12)
        down = (f_hi - fft_freqs) / (f_hi - f_mid + 1e-12)
        weights[m - 1] = np.maximum(0.0, np.minimum(up, down))

    return weights


def dynamic_range_compression_torch(x, C=1, clip_val=1e-5, *, norm_fn):
    return norm_fn(torch.clamp(x, min=clip_val) * C)


def spectral_normalize_torch(magnitudes, norm_fn):
    output = dynamic_range_compression_torch(magnitudes, norm_fn=norm_fn)
    return output


class MelConverter(nn.Module):

    def __init__(
        self,
        *,
        sampling_rate: float,
        n_fft: int,
        num_mels: int,
        hop_size: int,
        win_size: int,
        fmin: float,
        fmax: float,
        norm_fn,
    ):
        super().__init__()
        self.sampling_rate = sampling_rate
        self.n_fft = n_fft
        self.num_mels = num_mels
        self.hop_size = hop_size
        self.win_size = win_size
        self.fmin = fmin
        self.fmax = fmax
        self.norm_fn = norm_fn

        mel = librosa_mel_fn(sr=self.sampling_rate,
                             n_fft=self.n_fft,
                             n_mels=self.num_mels,
                             fmin=self.fmin,
                             fmax=self.fmax)
        mel_basis = torch.from_numpy(mel).float()
        hann_window = torch.hann_window(self.win_size)

        self.register_buffer('mel_basis', mel_basis)
        self.register_buffer('hann_window', hann_window)

    @property
    def device(self):
        return self.mel_basis.device

    def forward(self, waveform: torch.Tensor, center: bool = False) -> torch.Tensor:
        waveform = waveform.clamp(min=-1., max=1.).to(self.device)

        waveform = torch.nn.functional.pad(
            waveform.unsqueeze(1),
            [int((self.n_fft - self.hop_size) / 2),
             int((self.n_fft - self.hop_size) / 2)],
            mode='reflect')
        waveform = waveform.squeeze(1)

        spec = torch.stft(waveform,
                          self.n_fft,
                          hop_length=self.hop_size,
                          win_length=self.win_size,
                          window=self.hann_window,
                          center=center,
                          pad_mode='reflect',
                          normalized=False,
                          onesided=True,
                          return_complex=True)

        spec = torch.view_as_real(spec)
        spec = torch.sqrt(spec.pow(2).sum(-1) + (1e-9))
        spec = torch.matmul(self.mel_basis, spec)
        spec = spectral_normalize_torch(spec, self.norm_fn)

        return spec


def get_mel_converter(mode: Literal['16k', '44k']) -> MelConverter:
    if mode == '16k':
        return MelConverter(sampling_rate=16_000,
                            n_fft=1024,
                            num_mels=80,
                            hop_size=256,
                            win_size=1024,
                            fmin=0,
                            fmax=8_000,
                            norm_fn=torch.log10)
    elif mode == '44k':
        return MelConverter(sampling_rate=44_100,
                            n_fft=2048,
                            num_mels=128,
                            hop_size=512,
                            win_size=2048,
                            fmin=0,
                            fmax=44100 / 2,
                            norm_fn=torch.log)
    else:
        raise ValueError(f'Unknown mode: {mode}')
