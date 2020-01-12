import torch
import torch.nn.functional as func
import numpy as np

from Noise import filter_noise, create_white_noise
from Parameters import FRAME_LENGTH, HAMMING_WINDOW_LENGTH, HANNING_SMOOTHING


def synthetize_additive_plus_bruit(a0s, f0s, aa, hs, frame_length, sample_rate, device):
    assert a0s.size() == f0s.size()
    assert a0s.size()[1] == aa.size()[1]

    # Initialisation of lengths
    nb_bounds = f0s.size()[1]
    signal_length = (nb_bounds - 1) * frame_length

    """ Additive part """
    f0s = func.interpolate(f0s.unsqueeze(1), size=signal_length, mode='linear', align_corners=True)
    f0s = f0s.squeeze(1)

    # Multiply interpolated f0s by harmonic ranks to get all freqs
    nb_harms = aa.size()[-1]
    harm_ranks = torch.arange(nb_harms, device=device) + 1
    ff = f0s.unsqueeze(2) * harm_ranks

    # Phase accumulation over time for each freq
    phases = 2 * np.pi * ff / sample_rate
    phases_acc = torch.cumsum(phases, dim=1)

    # Denormalize amplitudes with a0
    aa_sum = torch.sum(aa, dim=2)
    # Avoid 0-div when all amplitudes are 0
    aa_sum[aa_sum == 0.] = 1.
    # Sacle amplitudes by a0s
    aa_norm = aa / aa_sum.unsqueeze(-1)
    aa = aa_norm * a0s.unsqueeze(-1)

    # Smoothing amplitudes
    aa = smoothing_amplitudes(aa, signal_length, HAMMING_WINDOW_LENGTH, device)

    # Prevent aliasing
    aa[ff >= sample_rate / 2.1] = 0.

    # Additive generation
    additive = aa * torch.sin(phases_acc)

    # Sum over harmonics
    additive = torch.sum(additive, dim=2)

    """ Noise part """
    hs = torch.sigmoid(hs)  # we impose hs be positive
    noise = filter_noise(create_white_noise(hs.shape[1] * FRAME_LENGTH, device=device), hs, device=device)

    # Empty cache
    torch.cuda.empty_cache()

    return additive, noise


def smoothing_amplitudes(aa, signal_length, window_length, device):
    aa = func.interpolate(aa.transpose(1, 2), size=signal_length, mode='linear', align_corners=True)
    aa = aa.transpose(1, 2)

    if HANNING_SMOOTHING:
        aa_downsampled = aa[:, ::window_length, :]
        return interpolate_hamming(aa_downsampled, signal_length, window_length, device)
    else:
        return aa


def interpolate_hamming(tensor, signal_length, frame_length, device):
    y = torch.zeros((tensor.shape[0], tensor.shape[1] * frame_length, tensor.shape[2]), device=device)
    y[:, ::frame_length, :] = tensor
    y = torch.transpose(y, 1, 2)
    y_padded = func.pad(y, [frame_length-1, frame_length-1])
    w = torch.hamming_window(2*frame_length, device=device).expand(y.shape[1], 1, 2*frame_length)

    interpolation = torch.conv1d(y_padded, w, groups=y_padded.shape[1])
    interpolation = torch.transpose(interpolation[:, :, 0:signal_length], 1, 2)

    return interpolation
