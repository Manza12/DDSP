import torch
from torch.nn import functional as func


def spectral_loss(stft_all, stft_truth_all, fft_sizes):
    losses = torch.zeros(len(fft_sizes))

    for i, fft_size in enumerate(fft_sizes):
        stft = stft_all[fft_size]
        stft_truth = stft_truth_all[fft_size]

        eps = torch.finfo(stft.dtype).eps

        stft_log = torch.log(stft + eps)
        stft_truth_log = torch.log(stft_truth + eps)

        loss_lin = func.l1_loss(stft, stft_truth, reduction="mean")
        loss_log = func.l1_loss(stft_log, stft_truth_log, reduction="mean")

        losses[i] = loss_lin + loss_log

    loss = torch.mean(losses)
    return loss


def compute_stft(waveform, fft_sizes):
    stft_all = {}

    for fft_size in fft_sizes:
        stft = torch.stft(waveform, fft_size, hop_length=fft_size // 4,
                          center=True, pad_mode='reflect',
                          normalized=False, onesided=True)
        stft = torch.sum(stft**2, dim=-1)
        stft_all[fft_size] = stft

    return stft_all
