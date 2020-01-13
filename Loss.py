import torch
from torch.nn import functional as F

def spectral_loss(stft_all, stft_truth_all, fft_sizes):
    losses = torch.zeros(len(fft_sizes))

    for i, fft_size in enumerate(fft_sizes):
        stft = stft_all[fft_size]
        stft_truth = stft_truth_all[fft_size]

        eps = torch.finfo(stft.dtype).eps

        stft_log = torch.log(stft + eps)
        stft_truth_log = torch.log(stft_truth + eps)

        loss_lin = F.l1_loss(stft, stft_truth, reduction="sum")
        loss_log = F.l1_loss(stft_log, stft_truth_log, reduction="sum")

        losses[i] = loss_lin + loss_log

    loss = torch.mean(losses)
    return loss


def spectral_loss_separed(stft_all, stft_truth_all, fft_sizes, device):
    losses_lin = torch.zeros(len(fft_sizes), device=device)
    losses_log = torch.zeros(len(fft_sizes), device=device)

    for i, fft_size in enumerate(fft_sizes):
        stft = stft_all[fft_size]
        stft_truth = stft_truth_all[fft_size]

        eps = torch.finfo(stft.dtype).eps

        stft_log = torch.log(stft + eps)
        stft_truth_log = torch.log(stft_truth + eps)

        loss_lin = F.l1_loss(stft, stft_truth, reduction="sum")
        loss_log = F.l1_loss(stft_log, stft_truth_log, reduction="sum")

        losses_lin[i] = loss_lin
        losses_log[i] = loss_log

    return losses_lin, losses_log


def compute_stft(waveform, fft_sizes):
    stft_all = {}

    for fft_size in fft_sizes:
        stft = torch.stft(waveform, fft_size, hop_length=fft_size // 4, center=True, pad_mode='reflect', normalized=True, onesided=True)
        stft = torch.sum(stft**2, dim=-1)
        stft_all[fft_size] = stft
        # remove DC
        # mags = mags[:, 1:, :]
    return stft_all
