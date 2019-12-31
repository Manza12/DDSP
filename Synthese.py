import torch
import torch.nn.functional as func

import numpy as np

from Parameters import BATCH_SIZE


def synthetize(a0s, f0s, aa, frame_length, sample_rate, device):
    assert a0s.size() == f0s.size()
    assert a0s.size()[1] == aa.size()[1]

    nb_bounds = f0s.size()[1]
    signal_length = (nb_bounds - 1) * frame_length

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

    # phase accumulation over time for each freq
    phases = 2 * np.pi * ff / sample_rate
    phases_acc = torch.cumsum(phases, dim=1)

    # denormalize amplitudes with a0
    # aa_sum = torch.sum(aa, dim=2)
    # avoid 0-div when all amplitudes are 0
    # aa_sum[aa_sum == 0.] = 1.
    # aa_norm = aa / aa_sum.unsqueeze(-1)
    aa = aa * a0s.unsqueeze(-1)

    # interpolate amplitudes over time
    # TODO use Hamming window instead? (cf ddsp paper)
    aa = func.interpolate(aa.unsqueeze(1), size=(signal_length, nb_harms), mode='bilinear',
                          align_corners=True)
    aa = aa.squeeze(1)
    
    # prevent aliasing
    aa[ff >= sample_rate / 2.1] = 0.

    # print(torch.cuda.memory_allocated(device=DEVICE))
    # print(torch.cuda.memory_cached(device=DEVICE))

    torch.cuda.empty_cache()

    waveforms = aa * torch.cos(phases_acc)
    # sum over harmonics
    waveforms = torch.sum(waveforms, dim=2)

    return waveforms

def synthetize_smooth(a0s, f0s, aa, frame_length, sample_rate, device):
    assert a0s.size() == f0s.size()
    assert a0s.size()[1] == aa.size()[1]

    nb_bounds = f0s.size()[1]
    signal_length = (nb_bounds - 1) * frame_length

    # interpolate f0 over time (bilinear)
    f0s = func.interpolate(f0s.unsqueeze(1), size=signal_length, mode='bilinear', align_corners=True)
    f0s = f0s.squeeze(1)

    # multiply interpolated f0s by harmonic ranks to get all freqs
    nb_harms = aa.size()[-1]
    harm_ranks = torch.arange(nb_harms, device=device) + 1
    ff = f0s.unsqueeze(2) * harm_ranks

    # phase accumulation over time for each freq
    phases = 2 * np.pi * ff / sample_rate
    phases_acc = torch.cumsum(phases, dim=1)

    # denormalize amplitudes with a0
    aa_sum = torch.sum(aa, dim=2)
    # avoid 0-div when all amplitudes are 0
    aa_sum[aa_sum == 0.] = 1.
    aa_norm = aa / aa_sum.unsqueeze(-1)
    aa = aa_norm * a0s.unsqueeze(-1)
    
    #Hamming window interpolation (upsampling) of amplitudes with 50% overlap    
    window = torch.hann_window(2*frame_length, periodic=False)
    aa_inter = torch.zeros([BATCH_SIZE, signal_length + 2*frame_length, nb_harms])
    # add amplitude-scaled hammming windows to output (aa_inter)
    for i in range(BATCH_SIZE):
        for j in range(nb_bounds):
            scaled_ham_frame = aa[i, j, :].unsqueeze(1) * window.unsqueeze(0)
            aa_inter[i, frame_length*j:frame_length*(j+2), :] += scaled_ham_frame.transpose(0, 1)
    aa = aa_inter[:, frame_length:signal_length + frame_length,:]
    
    # prevent aliasing
    aa[ff >= sample_rate / 2.1] = 0.

    torch.cuda.empty_cache()

    waveforms = aa * torch.cos(phases_acc)
    # sum over harmonics
    waveforms = torch.sum(waveforms, dim=2)

    return waveforms

