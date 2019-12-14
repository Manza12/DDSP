import torch
import torch.nn.functional as func

import numpy as np

from Parameters import AUDIO_SAMPLE_RATE


def synthetize(a0s, f0s, aa, frame_length, sample_rate, device):
    assert a0s.size() == f0s.size()
    assert a0s.size()[1] == aa.size()[1]

    nb_bounds = f0s.size()[1]
    signal_length = (nb_bounds - 1) * frame_length
    # interpol_factor = signal_length / nb_bounds

    # interpolate f0 over time
    # TODO paper mentions bilinear interpolation ?
    f0s = func.interpolate(f0s.unsqueeze(1), size=signal_length, mode='linear', align_corners=True)
    f0s = f0s.squeeze(1)


    # # interpolate a0 over time
    # # TODO paper mentions using Hamming window
    # a0s = func.interpolate(a0s.unsqueeze(1), scale_factor=signal_length / nb_bounds, mode='linear', align_corners=True)
    # a0s = a0s.squeeze(1)

    # multiply interpolated f0s by harmonic ranks to get all freqs
    nb_harms = aa.size()[-1]
    harm_ranks = torch.arange(nb_harms, device=device) + 1
    ff = f0s.unsqueeze(2) * harm_ranks
    ff = torch.where(ff <= AUDIO_SAMPLE_RATE/2, ff, torch.zeros(ff.shape, device=device))

    # phase accumulation over time for each freq
    phases = 2 * np.pi * ff / sample_rate
    phases_acc = torch.cumsum(phases, dim=1)

    # denormalize amplitudes with a0
    aa_sum = torch.sum(aa, dim=2)
    # avoid 0-div when all amplitudes are 0
    aa_sum[aa_sum == 0.] = 1.
    aa_norm = aa / aa_sum.unsqueeze(-1)
    aa = aa_norm * a0s.unsqueeze(-1)

    # interpolate amplitudes over time
    # TODO use Hamming window instead? (cf ddsp paper)
    aa = func.interpolate(aa.unsqueeze(1), size=(signal_length, nb_harms), mode='bilinear',
                          align_corners=True)
    aa = aa.squeeze(1)

    # print(torch.cuda.memory_allocated(device=DEVICE))
    # print(torch.cuda.memory_cached(device=DEVICE))

    torch.cuda.empty_cache()

    signals = aa * torch.cos(phases_acc)
    # sum over harmonics
    signal = torch.sum(signals, dim=2)

    return signal
