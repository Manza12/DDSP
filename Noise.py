import numpy as np
import torch
import torch.nn.functional as func
import scipy.io.wavfile as wav
import os

from Parameters import AUDIO_SAMPLE_RATE, FRAME_LENGTH, DEVICE


def filter_noise(noise_time, filter_freq, write=False, nom="filtered_noise", device=DEVICE):
    # Noise part
    new_shape = (noise_time.shape[0] // FRAME_LENGTH, FRAME_LENGTH)
    noise_time_splited = noise_time.reshape(new_shape)
    noise_time_splited = noise_time_splited.unsqueeze(0)
    noise_freq = torch.rfft(noise_time_splited, 1)

    # Filter part
    filter_freq_complex = torch.stack((filter_freq, torch.zeros(filter_freq.shape, device=device)), dim=-1)
    filter_time = torch.irfft(filter_freq_complex, 1, onesided=True)
    hann_window = torch.hann_window(filter_time.shape[-1], device=device)
    hann_window = torch.unsqueeze(hann_window, 0)
    hann_window = torch.unsqueeze(hann_window, 0)
    filter_time = filter_time * hann_window
    filter_time = torch.roll(filter_time, filter_time.shape[0] // 2 + 1, dims=-1)
    pad = (noise_time_splited.shape[-1] - filter_time.shape[-1]) // 2
    filter_time = func.pad(filter_time, [pad, pad + 1])
    filter_freq = torch.rfft(filter_time, 1)

    # Filtered noise
    filtered_noise_freq = complex_mult_torch(noise_freq, filter_freq)
    filtered_noise_time = torch.irfft(filtered_noise_freq, 1)[:, :, 0:FRAME_LENGTH]

    noises_list = torch.split(filtered_noise_time, 1, dim=1)
    noise = torch.cat(noises_list[0:-1], dim=-1)
    noise = torch.squeeze(noise, dim=1)

    if write:
        wav.write(os.path.join("Outputs", "Original_Noise" + ".wav"), AUDIO_SAMPLE_RATE, noise_time.cpu().detach().numpy())
        wav.write(os.path.join("Outputs", nom + ".wav"), AUDIO_SAMPLE_RATE, noise[0, :].cpu().detach().numpy())

    torch.cuda.empty_cache()

    return noise


def synthetize_bruit(a0s, f0s, aa, hs, h0s, frame_length, sample_rate, device):
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

    # prevent aliasing
    aa[ff >= sample_rate / 2.1] = 0.

    # print(torch.cuda.memory_allocated(device=DEVICE))
    # print(torch.cuda.memory_cached(device=DEVICE))

    torch.cuda.empty_cache()

    waveforms = aa * torch.cos(phases_acc)
    # sum over harmonics
    waveforms = torch.sum(waveforms, dim=2)

    """ Partie bruit """
    hs = torch.sigmoid(hs)  # we impose hs be positive
    h0s = torch.sigmoid(h0s)  # we impose h0s be positive
    bruits = filter_noise_normed(create_white_noise(FRAME_LENGTH, device=device), hs, h0s, device=device)
    bruits_list = torch.split(bruits, 1, dim=1)
    bruit = torch.cat(bruits_list[0:-1], dim=-1)
    bruit = torch.squeeze(bruit, dim=1)
    waveforms = waveforms + bruit

    return waveforms


def create_white_noise(samples, write=False, nom="white_noise", device="cpu"):
    noise_time = torch.tensor(np.float32(np.random.uniform(-1, 1, samples)), device=device)
    if write:
        wav.write(os.path.join("Outputs", nom + ".wav"), AUDIO_SAMPLE_RATE, noise_time.numpy())

    return noise_time


def filter_noise_normed(noise_time, filter_freq, noise_amplitude, write=False, nom="filtered_noise", device=DEVICE):
    filter_freq_mean = torch.unsqueeze(torch.sum(filter_freq, -1), -1)
    filter_freq_normalized = filter_freq / filter_freq_mean
    filter_freq_scaled = filter_freq_normalized * torch.unsqueeze(noise_amplitude, -1)
    filter_freq_complex = torch.stack((filter_freq_scaled, torch.zeros(filter_freq.shape, device=device)), dim=-1)
    filter_time = torch.irfft(filter_freq_complex, 1, onesided=True)
    hann_window = torch.hann_window(filter_time.shape[-1], device=device)
    hann_window = torch.unsqueeze(hann_window, 0)
    hann_window = torch.unsqueeze(hann_window, 0)
    filter_time = filter_time * hann_window
    filter_time = torch.roll(filter_time, filter_time.shape[0] // 2 + 1, dims=-1)
    pad = (noise_time.shape[-1]-filter_time.shape[-1]) // 2
    filter_time = func.pad(filter_time, [pad, pad + 1])
    noise_time = torch.unsqueeze(noise_time, 0)
    noise_time = torch.unsqueeze(noise_time, 0)
    noise_freq = torch.rfft(noise_time, 1)
    filter_freq = torch.rfft(filter_time, 1)
    filtered_noise_freq = complex_mult_torch(noise_freq, filter_freq)
    filtered_noise_time = torch.irfft(filtered_noise_freq, 1)[:, :, 0:FRAME_LENGTH]

    if write:
        wav.write(os.path.join("Outputs", nom + ".wav"), AUDIO_SAMPLE_RATE, filtered_noise_time.numpy())

    return filtered_noise_time


def complex_mult_torch(z, w):
    assert z.shape[-1] == 2 and w.shape[-1] == 2, 'Last dimension must be 2'
    return torch.stack(
        (z[..., 0] * w[..., 0] - z[..., 1] * w[..., 1],
         z[..., 0] * w[..., 1] + z[..., 1] * w[..., 0]),
        dim=-1)


if __name__ == "__main__":
    noise_sound = create_white_noise(16000, write=True)
    NOISE = create_white_noise(160)

    filter_transfer_ampl = torch.zeros(65)
    filter_loudness = torch.ones(65)
    mu = 30
    filter_transfer_ampl[mu] = 1
    filter_transfer_ampl[mu - 1] = 0.5
    filter_transfer_ampl[mu + 1] = 0.5


    filtered_noise = filter_noise_normed(NOISE, filter_transfer_ampl, 65)

    sound = np.zeros(16000)
    for i in range(100):
        sound[160 * i: 160 * (i+1)] = filtered_noise.numpy()

    wav.write(os.path.join("Outputs", "filtered_noise" + ".wav"), AUDIO_SAMPLE_RATE, sound)
