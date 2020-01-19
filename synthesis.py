import torch.nn.functional as func
import numpy as np

from noise import filter_noise, create_white_noise
from dataloader import int_2_float
from parameters import *


def synthetize(a0s, f0s, aa, hs, frame_length, sample_rate, device):
    assert a0s.size() == f0s.size()
    assert a0s.size()[1] == aa.size()[1]

    # Initialisation of parameters
    nb_bounds = f0s.size()[1]
    signal_length = (nb_bounds - 1) * frame_length
    nb_harms = aa.size()[-1]
    harm_ranks = torch.arange(nb_harms, device=device) + 1

    """ Additive part """
    # Prevent aliasing
    ff = f0s.unsqueeze(2) * harm_ranks
    aa[ff >= sample_rate / 2.1] = 0.

    # Interpolate frequencies
    f0s = func.interpolate(f0s.unsqueeze(1), size=signal_length, mode='linear',
                           align_corners=True)
    ff = f0s.squeeze(1).unsqueeze(2) * harm_ranks

    # Phase accumulation over time for each freq
    phases = 2 * np.pi * ff / sample_rate
    phases_acc = torch.cumsum(phases, dim=1)

    # Normalize amplitudes with a0
    aa_sum = torch.sum(aa, dim=2)
    aa_sum[aa_sum == 0.] = 1.
    aa_norm = aa / aa_sum.unsqueeze(-1)
    aa = aa_norm * a0s.unsqueeze(-1)

    # Interpolate amplitudes
    aa = smoothing_amplitudes(aa, signal_length, HAMMING_WINDOW_LENGTH, device)

    # Additive generation
    additive = aa * torch.sin(phases_acc)

    # Sum over harmonics
    additive = torch.sum(additive, dim=2)

    """ Noise part """
    noise = filter_noise(create_white_noise(hs.shape[1] * FRAME_LENGTH,
                                            device=device), hs, device=device)

    # Empty cache
    torch.cuda.empty_cache()

    return additive, noise


def smoothing_amplitudes(aa, signal_length, window_length, device):
    aa = func.interpolate(aa.transpose(1, 2), size=signal_length,
                          mode='linear', align_corners=True)
    aa = aa.transpose(1, 2)

    if HANNING_SMOOTHING:
        aa_downsampled = aa[:, ::window_length, :]
        return interpolate_hamming(aa_downsampled, signal_length,
                                   window_length, device)
    else:
        return aa


def interpolate_hamming(tensor, signal_length, frame_length, device):
    y = torch.zeros((tensor.shape[0], tensor.shape[1] * frame_length,
                     tensor.shape[2]), device=device)
    y[:, ::frame_length, :] = tensor
    y = torch.transpose(y, 1, 2)
    y_padded = func.pad(y, [frame_length-1, frame_length-1])
    w = torch.hamming_window(2*frame_length, device=device)
    w = w.expand(y.shape[1], 1, 2*frame_length)

    interpolation = torch.conv1d(y_padded, w, groups=y_padded.shape[1])
    interpolation = torch.transpose(interpolation[:, :, 0:signal_length], 1, 2)

    return interpolation


def reverb(waveform):
    import scipy
    import os
    [fs, ir] = scipy.io.wavfile.read(os.path.join("Inputs", "ir_edited.wav"))

    assert fs == AUDIO_SAMPLE_RATE

    ir = int_2_float(ir)

    wf_rev = np.convolve(waveform, ir)
    wf_rev = wf_rev[ir.shape[0]-1:ir.shape[0]+waveform.shape[0]] / sum(ir)

    return wf_rev
