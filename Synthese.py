import torch
import torch.nn.functional as func
import numpy as np

from Noise import filter_noise, create_white_noise
from Parameters import FRAME_LENGTH, DEVICE, BATCH_SIZE, NUMBER_HARMONICS
from Testing import plot_waveform

# RANDOM_PHASES_TRAIN = torch.rand([BATCH_SIZE, 31840, NUMBER_HARMONICS], device=DEVICE)
# RANDOM_PHASES_EVAL = torch.rand([1, 959840, NUMBER_HARMONICS], device="cpu")


def synthetize_additive_plus_bruit(a0s, f0s, aa, hs, frame_length, sample_rate, device):
    assert a0s.size() == f0s.size()
    assert a0s.size()[1] == aa.size()[1]

    nb_bounds = f0s.size()[1]
    signal_length = (nb_bounds - 1) * frame_length

    f0s = func.interpolate(f0s.unsqueeze(1), size=signal_length, mode='linear', align_corners=True)
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

    aa = func.interpolate(aa.unsqueeze(1), size=(signal_length, nb_harms), mode='bilinear',
                          align_corners=True)
    aa = aa.squeeze(1)

    # prevent aliasing
    aa[ff >= sample_rate / 2.1] = 0.

    # random_phases = RANDOM_PHASES_EVAL
    # random_phases[ff >= sample_rate / 2.1] = 0.

    additive = aa * torch.sin(phases_acc)

    #### Testing ####
    # from Parameters import NUMBER_HARMONICS
    # for i in range(NUMBER_HARMONICS):
    #     plot_waveform(additive[0, :, i].cpu().detach().numpy())
    #     print(i, torch.mean(additive[0, :, i]).item())

    # sum over harmonics
    additive = torch.sum(additive, dim=2)
    # recentering
    # means = torch.mean(additive, 1).unsqueeze(1).expand(additive.shape)
    # additive = additive - means
    # print("Somme", torch.mean(additive[0, :]).item())
    # plot_waveform(additive[0, :].cpu().detach().numpy())
    #################


    """ Partie bruit """
    hs = torch.sigmoid(hs)  # we impose hs be positive
    noise = filter_noise(create_white_noise(hs.shape[1] * FRAME_LENGTH, device=device), hs, device=device)

    torch.cuda.empty_cache()

    return additive, noise


def synthetize_smooth(a0s, f0s, aa, frame_length, sample_rate, device):
    assert a0s.size() == f0s.size()
    assert a0s.size()[1] == aa.size()[1]

    nb_bounds = f0s.size()[1]
    signal_length = (nb_bounds - 1) * frame_length

    # interpolate f0 over time (linear)
    f0s = f0s.unsqueeze(1)
    f0s = func.interpolate(f0s, size=signal_length, mode='linear', align_corners=True)
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

    # Hanning interpolation
    aa_inter = interpolate_hanning(aa, frame_length, device)

    # prevent aliasing
    aa_inter[ff >= sample_rate / 2.1] = 0.

    waveforms = aa_inter * torch.cos(phases_acc)
    # sum over harmonics
    waveforms = torch.sum(waveforms, dim=2)

    torch.cuda.empty_cache()

    return waveforms


def interpolate_hanning(tensor, frame_length, device):
    y = torch.zeros((tensor.shape[0], tensor.shape[1] * frame_length, tensor.shape[2]), device=device)
    y[:, ::frame_length, :] = tensor
    y = torch.transpose(y, 1, 2)
    y_padded = func.pad(y, [frame_length-1, 0])
    w = torch.hann_window(2*frame_length, device=device).expand(y.shape[1], 1, 2*frame_length)

    interpolation = torch.conv1d(y_padded, w, groups=y_padded.shape[1])
    interpolation = torch.transpose(interpolation, 1, 2)

    return interpolation


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
