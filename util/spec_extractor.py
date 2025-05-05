from turtle import forward
import numpy as np
import torch
import torchaudio
from torch import nn
import torchaudio.compliance.kaldi as ta_kaldi
import dcase_util
import torch.nn.functional as F
from typing import Optional, Tuple


class _SpecExtractor(nn.Module):
    """Base Module for spectrogram extractors."""


class Cnn3Mel(_SpecExtractor):
    """Mel extractor for previous CNN3 baseline system."""

    def __init__(
        self,
        spectrogram_type="magnitude",
        hop_length_seconds=0.02,
        win_length_seconds=0.04,
        window_type="hamming_asymmetric",
        n_mels=40,
        n_fft=2048,
        fmin=0,
        fmax=22050,
        htk=False,
        normalize_mel_bands=False,
        **kwargs
    ):
        super().__init__()
        self.extractor = dcase_util.features.MelExtractor(
            spectrogram_type=spectrogram_type,
            hop_length_seconds=hop_length_seconds,
            win_length_seconds=win_length_seconds,
            window_type=window_type,
            n_mels=n_mels,
            n_fft=n_fft,
            fmin=fmin,
            fmax=fmax,
            htk=htk,
            normalize_mel_bands=normalize_mel_bands,
            **kwargs
        )

    def forward(self, x):
        mel = []
        for wav in x:
            wav = wav.cpu().numpy()
            mel.append(self.extractor.extract(wav))
        mel = np.stack(mel)
        mel = torch.from_numpy(mel).to(x.device)
        return mel


class CpMel(_SpecExtractor):
    """
    Mel extractor for CP-JKU systems. Adapted from: https://github.com/fschmid56/cpjku_dcase23
    """

    def __init__(
        self,
        n_mels=256,
        sr=32000,
        win_length=3072,
        hop_size=500,
        n_fft=4096,
        fmin=0.0,
        fmax=None,
    ):
        super().__init__()
        # adapted from: https://github.com/CPJKU/kagglebirds2020/commit/70f8308b39011b09d41eb0f4ace5aa7d2b0e806e
        self.win_length = win_length
        self.n_mels = n_mels
        self.n_fft = n_fft
        self.sr = sr
        self.fmin = fmin
        self.fmax = sr // 2 if fmax is None else fmax
        self.hop_size = hop_size
        self.register_buffer(
            "window", torch.hann_window(win_length, periodic=False), persistent=False
        )
        self.register_buffer(
            "preemphasis_coefficient", torch.as_tensor([[[-0.97, 1]]]), persistent=False
        )

    def forward(self, x):
        x = nn.functional.conv1d(
            x.unsqueeze(1), self.preemphasis_coefficient.to(x.device)
        ).squeeze(1)
        x = torch.stft(
            x,
            self.n_fft,
            hop_length=self.hop_size,
            win_length=self.win_length,
            center=True,
            normalized=False,
            window=self.window.to(x.device),
            return_complex=True,
        )
        x = torch.view_as_real(x)
        x = (x**2).sum(dim=-1)  # power mag
        mel_basis, _ = torchaudio.compliance.kaldi.get_mel_banks(
            self.n_mels,
            self.n_fft,
            self.sr,
            self.fmin,
            self.fmax,
            vtln_low=100.0,
            vtln_high=-500.0,
            vtln_warp_factor=1.0,
        )
        mel_basis = torch.as_tensor(
            torch.nn.functional.pad(mel_basis, (0, 1), mode="constant", value=0),
            device=x.device,
        )
        with torch.amp.autocast("cuda", enabled=False):
            melspec = torch.matmul(mel_basis, x)
        # Log mel spectrogram
        melspec = (melspec + 0.00001).log()
        # Fast normalization
        melspec = (melspec + 4.5) / 5.0
        return melspec


class BEATsMel(_SpecExtractor):
    """Mel extractor for BEATs model."""

    def __init__(self, dataset_mean: float = 15.41663, dataset_std: float = 6.55582):
        super(BEATsMel, self).__init__()
        self.dataset_mean = dataset_mean
        self.dataset_std = dataset_std

    def forward(self, x):
        fbanks = []
        for waveform in x:
            waveform = waveform.unsqueeze(0) * 2**15
            fbank = ta_kaldi.fbank(
                waveform,
                num_mel_bins=128,
                sample_frequency=16000,
                frame_length=25,
                frame_shift=10,
            )
            fbanks.append(fbank)
        fbank = torch.stack(fbanks, dim=0)
        fbank = (fbank - self.dataset_mean) / (2 * self.dataset_std)
        return fbank


# LEAF implementation adopted from:
# https://github.com/google-research/leaf-audio
# https://github.com/CPJKU/EfficientLEAF


def mel_filter_params(
    n_filters: int, min_freq: float, max_freq: float, sample_rate: int
) -> Tuple[torch.Tensor, torch.Tensor]:
    min_mel = 1127 * np.log1p(min_freq / 700.0)
    max_mel = 1127 * np.log1p(max_freq / 700.0)
    peaks_mel = torch.linspace(min_mel, max_mel, n_filters + 2)
    peaks_hz = 700 * (torch.expm1(peaks_mel / 1127))
    center_freqs = peaks_hz[1:-1] * (2 * np.pi / sample_rate)
    bandwidths = peaks_hz[2:] - peaks_hz[:-2]
    sigmas = (sample_rate / 2.0) / bandwidths
    return center_freqs, sigmas


def gabor_filters(
    size: int, center_freqs: torch.Tensor, sigmas: torch.Tensor
) -> torch.Tensor:
    t = torch.arange(-(size // 2), (size + 1) // 2, device=center_freqs.device)
    denominator = 1.0 / (np.sqrt(2 * np.pi) * sigmas)
    gaussian = torch.exp(torch.outer(1.0 / (2.0 * sigmas**2), -(t**2)))
    sinusoid = torch.exp(1j * torch.outer(center_freqs, t))
    return denominator[:, np.newaxis] * sinusoid * gaussian


def gauss_windows(size: int, sigmas: torch.Tensor) -> torch.Tensor:
    t = torch.arange(0, size, device=sigmas.device)
    numerator = t * (2 / (size - 1)) - 1
    return torch.exp(-0.5 * (numerator / sigmas[:, np.newaxis]) ** 2)


class GaborFilterbank(nn.Module):
    def __init__(
        self,
        n_filters: int,
        min_freq: float,
        max_freq: float,
        sample_rate: int,
        filter_size: int,
        pool_size: int,
        pool_stride: int,
        pool_init: float = 0.4,
    ):
        super(GaborFilterbank, self).__init__()
        self.n_filters = n_filters
        self.filter_size = filter_size
        self.pool_size = pool_size
        self.pool_stride = pool_stride
        center_freqs, bandwidths = mel_filter_params(
            n_filters, min_freq, max_freq, sample_rate
        )
        self.center_freqs = nn.Parameter(center_freqs)
        self.bandwidths = nn.Parameter(bandwidths)
        self.pooling_widths = nn.Parameter(torch.full((n_filters,), float(pool_init)))

    def forward(self, x):
        # compute filters
        center_freqs = self.center_freqs.clamp(min=0.0, max=np.pi)
        z = np.sqrt(2 * np.log(2)) / np.pi
        bandwidths = self.bandwidths.clamp(min=4 * z, max=self.filter_size * z)
        filters = gabor_filters(self.filter_size, center_freqs, bandwidths)
        filters = torch.cat((filters.real, filters.imag), dim=0).unsqueeze(1)
        # convolve with filters
        x = F.conv1d(x, filters, padding=self.filter_size // 2)
        # compute squared modulus
        x = x**2
        x = x[:, : self.n_filters] + x[:, self.n_filters :]
        # compute pooling windows
        pooling_widths = self.pooling_widths.clamp(min=2.0 / self.pool_size, max=0.5)
        windows = gauss_windows(self.pool_size, pooling_widths).unsqueeze(1)
        # apply temporal pooling
        x = F.conv1d(
            x,
            windows,
            stride=self.pool_stride,
            padding=self.filter_size // 2,
            groups=self.n_filters,
        )
        return x


class PCEN(nn.Module):
    def __init__(
        self,
        num_bands: int,
        s: float = 0.025,
        alpha: float = 1.0,
        delta: float = 1.0,
        r: float = 1.0,
        eps: float = 1e-6,
        learn_logs: bool = True,
        clamp: Optional[float] = None,
    ):
        super(PCEN, self).__init__()
        if learn_logs:
            # learns logarithm of each parameter
            s = np.log(s)
            alpha = np.log(alpha)
            delta = np.log(delta)
            r = np.log(r)
        else:
            # learns inverse of r, and all other parameters directly
            r = 1.0 / r
        self.learn_logs = learn_logs
        self.s = nn.Parameter(torch.full((num_bands,), float(s)))
        self.alpha = nn.Parameter(torch.full((num_bands,), float(alpha)))
        self.delta = nn.Parameter(torch.full((num_bands,), float(delta)))
        self.r = nn.Parameter(torch.full((num_bands,), float(r)))
        self.eps = torch.as_tensor(eps)
        self.clamp = clamp

    def forward(self, x):
        # clamp if needed
        if self.clamp is not None:
            x = x.clamp(min=self.clamp)

        # prepare parameters
        if self.learn_logs:
            # learns logarithm of each parameter
            s = self.s.exp()
            alpha = self.alpha.exp()
            delta = self.delta.exp()
            r = self.r.exp()
        else:
            # learns inverse of r, and all other parameters directly
            s = self.s
            alpha = self.alpha.clamp(max=1)
            delta = self.delta.clamp(min=0)  # unclamped in original LEAF impl.
            r = 1.0 / self.r.clamp(min=1)
        # broadcast over channel dimension
        alpha = alpha[:, np.newaxis]
        delta = delta[:, np.newaxis]
        r = r[:, np.newaxis]

        # compute smoother
        smoother = [x[..., 0]]  # initialize the smoother with the first frame
        for frame in range(1, x.shape[-1]):
            smoother.append((1 - s) * smoother[-1] + s * x[..., frame])
        smoother = torch.stack(smoother, -1)

        # stable reformulation due to Vincent Lostanlen; original formula was:
        # return (input / (self.eps + smoother)**alpha + delta)**r - delta**r
        smoother = torch.exp(
            -alpha * (torch.log(self.eps) + torch.log1p(smoother / self.eps))
        )
        return (x * smoother + delta) ** r - delta**r


class Leaf(_SpecExtractor):
    def __init__(
        self,
        n_filters: int = 64,
        min_freq: float = 60.0,
        max_freq: float = 7800.0,
        sample_rate: int = 32000,
        window_len: float = 96.0,
        window_stride: float = 15.0,
        compression: Optional[torch.nn.Module] = None,
    ):
        super(Leaf, self).__init__()

        # convert window sizes from milliseconds to samples
        window_size = int(sample_rate * window_len / 1000)
        window_size += 1 - (window_size % 2)  # make odd
        window_stride = int(sample_rate * window_stride / 1000)

        self.filterbank = GaborFilterbank(
            n_filters,
            min_freq,
            max_freq,
            sample_rate,
            filter_size=window_size,
            pool_size=window_size,
            pool_stride=window_stride,
        )

        self.compression = (
            compression
            if compression
            else PCEN(
                n_filters,
                s=0.04,
                alpha=0.96,
                delta=2,
                r=0.5,
                eps=1e-12,
                learn_logs=False,
                clamp=1e-5,
            )
        )

    def forward(self, x: torch.tensor):
        while x.ndim < 3:
            x = x[:, np.newaxis]
        x = self.filterbank(x)
        x = self.compression(x)
        return x


class LeafBeats(_SpecExtractor):
    def __init__(
        self,
        n_filters: int = 64,
        min_freq: float = 60.0,
        max_freq: float = 7800.0,
        sample_rate: int = 32000,
        window_len: float = 96.0,
        window_stride: float = 15.0,
    ):
        super(LeafBeats, self).__init__()
        self.leaf = Leaf(
            n_filters=n_filters,
            min_freq=min_freq,
            max_freq=max_freq,
            sample_rate=sample_rate,
            window_len=window_len,
            window_stride=window_stride,
            compression=None,
        )

    def forward(self, x: torch.tensor):
        # already normalized per channel
        return self.leaf(x)
